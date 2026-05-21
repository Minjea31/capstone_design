# crash_test heading / depth 파이프라인 요약
# ------------------------------------------------------------
# - ROI polygon은 `crash.json`에서 읽습니다.
# - reference line group은 `crash_reference.json`에서 읽습니다.
# - YOLO 추론은 `crash.pt`를 사용합니다.
# - 입력은 USB 카메라 `CAMERA_INDEX = 2`입니다.
#
# 차선 그룹 매핑:
# - crash.json의 polygon index 0 -> LANE_DOWN
# - crash.json의 polygon index 1 -> LANE_UP
# - crash_reference.json의 group index 0,1 -> LANE_DOWN
# - crash_reference.json의 group index 2,3 -> LANE_UP
#
# 처리 흐름:
# 1. YOLO tracking으로 차량 bbox와 track_id를 얻습니다.
# 2. ROI polygon으로 bbox가 어느 차선인지 판정합니다.
# 3. 같은 차선의 reference line으로 heading 방향을 보정합니다.
# 4. Depth Anything V2로 bbox 내부 상대 깊이 변화를 추적합니다.
# 5. heading, line reference, depth trend를 합쳐 위험/사고를 표시합니다.

import json
import math
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

# ultralytics가 홈 디렉터리에 설정 파일을 만들지 못하는 환경 대응
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent

# 학습에 사용한 워크스페이스 내 Ultralytics 구현을 우선 사용합니다.
VENDORED_ULTRALYTICS_ROOT = PROJECT_ROOT / "yolov12" / "yolov12"
if VENDORED_ULTRALYTICS_ROOT.exists():
    sys.path.insert(0, str(VENDORED_ULTRALYTICS_ROOT))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

WEIGHTS_PATH = MODULE_DIR / "crash.pt"
ROI_JSON_PATH = MODULE_DIR / "crash.json"
HEADING_GROUP_JSON_PATH = MODULE_DIR / "crash_reference.json"
CAMERA_INDEX = 2
DEPTH_MODEL_ID = os.environ.get("DEPTH_MODEL_ID", "depth-anything/Depth-Anything-V2-Small-hf")
VEHICLE_CLASSES = [0]  # heading 계산에 사용할 검출 클래스 id
CONFIDENCE_THRESHOLD = 0.7  # 낮은 신뢰도 bbox를 제외해 tracking/heading 흔들림을 줄임
ENABLE_HEADING = True  # heading 계산 기능 on/off
ENABLE_DEPTH = True
HEADING_WINDOW = 4  # 최근 몇 프레임 이동을 평균내 heading 방향을 계산할지
DEPTH_WINDOW = 4
MIN_MOTION_MAGNITUDE = 0.1  # 이동량이 이 값보다 작으면 heading 계산 생략
TRACK_HISTORY_LENGTH = 20  # 차량별 위치 이력을 몇 프레임까지 유지할지
DEPTH_HISTORY_LENGTH = 20
LOCAL_DIRECTION_POINT_RADIUS = 2  # nearest point 좌우로 사용할 reference point 개수
REFERENCE_WEIGHT_EPSILON = 1.0  # 거리 가중 평균에서 0으로 나누는 문제 방지
LANE_OVERLAP_MIN_RATIO = 0.01  # bbox 대비 lane polygon 겹침 비율이 이 값보다 작으면 unknown 처리
DEBUG_HEADING_LOGS = True
DEBUG_HEADING_OVERLAY = True
ARROW_LENGTH_PIXELS = 40
ACCIDENT_STATIONARY_DISTANCE_THRESHOLD = 3.0
DEPTH_TREND_RELATIVE_THRESHOLD = 0.03
DEPTH_GREATER_MEANS_CLOSER = True
DEPTH_STATUS_MESSAGE = ""

LANE_DOWN = 0
LANE_UP = 1
LANE_UNKNOWN = -1

LANE_LABELS_KO = {
    LANE_DOWN: "하행",
    LANE_UP: "상행",
    LANE_UNKNOWN: "unknown",
}

LANE_REFERENCE_POLYLINE_INDEXES = {
    LANE_DOWN: (0, 1),
    LANE_UP: (2, 3),
}

LANE_EXPECTED_Y_SIGN = {
    LANE_DOWN: 1.0,
    LANE_UP: -1.0,
}

def log_heading_status(message):
    del message


def load_lane_polygons():
    roi_path = ROI_JSON_PATH
    if not roi_path.exists():
        log_heading_status(f"lane polygon 파일이 없습니다: {roi_path}")
        return {}

    try:
        with roi_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception as error:
        log_heading_status(f"lane polygon 파일 로드 실패: {error}")
        return {}

    polygons = {}
    for polygon_index, item in enumerate(payload):
        if not isinstance(item, list) or len(item) != 2:
            continue

        points_data, _ = item
        if not isinstance(points_data, list):
            continue

        points = []
        for point in points_data:
            if not isinstance(point, list) or len(point) != 2:
                continue
            points.append((int(point[0]), int(point[1])))

        if len(points) >= 3:
            lane_group = LANE_UNKNOWN
            if polygon_index == 0:
                lane_group = LANE_DOWN
            elif polygon_index == 1:
                lane_group = LANE_UP

            polygons[int(polygon_index)] = {
                "polygon_index": int(polygon_index),
                "lane_group": lane_group,
                "points": points,
            }

    if polygons:
        log_heading_status(f"lane polygon {len(polygons)}개를 로드했습니다: {roi_path}")
    return polygons


def load_reference_polylines():
    json_path = HEADING_GROUP_JSON_PATH
    if not json_path.exists():
        log_heading_status(f"reference polyline json 파일이 없습니다: {json_path}")
        return {}

    try:
        with json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception as error:
        log_heading_status(f"reference polyline json 로드 실패: {error}")
        return {}

    groups = payload.get("groups", []) if isinstance(payload, dict) else []
    if not isinstance(groups, list):
        return {}

    polylines = {}
    for polyline_index, group in enumerate(groups):
        if not isinstance(group, dict):
            continue

        points = group.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            continue

        normalized_points = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                continue
            normalized_points.append((float(point[0]), float(point[1])))

        if len(normalized_points) < 2:
            continue

        lane_group = LANE_UNKNOWN
        if polyline_index in LANE_REFERENCE_POLYLINE_INDEXES[LANE_DOWN]:
            lane_group = LANE_DOWN
        elif polyline_index in LANE_REFERENCE_POLYLINE_INDEXES[LANE_UP]:
            lane_group = LANE_UP

        polylines[int(polyline_index)] = {
            "polyline_index": int(polyline_index),
            "lane_group": lane_group,
            "points": normalized_points,
        }

    if polylines:
        log_heading_status(f"reference polyline {len(polylines)}개를 로드했습니다: {json_path}")
    return polylines


def get_bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def load_model():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"모델 가중치 파일을 찾지 못했습니다: {WEIGHTS_PATH}")

    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = patched_torch_load
        model = YOLO(str(WEIGHTS_PATH))
    finally:
        torch.load = original_torch_load

    return model


def load_depth_model():
    global DEPTH_STATUS_MESSAGE
    if not ENABLE_DEPTH:
        DEPTH_STATUS_MESSAGE = "DEPTH OFF"
        return None

    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError:
        DEPTH_STATUS_MESSAGE = "DEPTH UNAVAILABLE: transformers too old"
        return None

    try:
        processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID, local_files_only=True)
        model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID, local_files_only=True)
    except Exception:
        try:
            processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
            model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
        except Exception:
            DEPTH_STATUS_MESSAGE = "DEPTH UNAVAILABLE: download failed"
            return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    DEPTH_STATUS_MESSAGE = f"DEPTH ON: {Path(DEPTH_MODEL_ID).name}"
    return {
        "processor": processor,
        "model": model,
        "device": device,
    }


def estimate_depth_map(frame, depth_runner):
    if depth_runner is None:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb_frame.shape[:2]
    processor = depth_runner["processor"]
    model = depth_runner["model"]
    device = depth_runner["device"]

    inputs = processor(images=rgb_frame, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    depth = processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(height, width)],
    )[0]["predicted_depth"]
    return depth.detach().cpu().numpy().astype(np.float32)


def sample_bbox_depth(depth_map, xyxy):
    if depth_map is None:
        return None

    height, width = depth_map.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    # 박스 가장자리 배경 영향이 커서 중앙 영역만 샘플링합니다.
    crop_width = x2 - x1
    crop_height = y2 - y1
    inner_x1 = x1 + int(crop_width * 0.2)
    inner_x2 = x2 - int(crop_width * 0.2)
    inner_y1 = y1 + int(crop_height * 0.2)
    inner_y2 = y2 - int(crop_height * 0.2)

    if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
        inner_x1, inner_x2, inner_y1, inner_y2 = x1, x2, y1, y2

    roi_depth = depth_map[inner_y1:inner_y2, inner_x1:inner_x2]
    if roi_depth.size == 0:
        return None

    return float(np.median(roi_depth))


def compute_depth_info(depth_history):
    if not depth_history:
        return None

    current_depth = depth_history[-1]
    if len(depth_history) < 2:
        return {
            "current_depth": current_depth,
            "relative_change": 0.0,
            "trend": "unknown",
        }

    usable_values = list(depth_history)[-max(DEPTH_WINDOW + 1, 2):]
    start_depth = usable_values[0]
    end_depth = usable_values[-1]
    raw_change = end_depth - start_depth
    signed_change = raw_change if DEPTH_GREATER_MEANS_CLOSER else -raw_change
    relative_change = signed_change / max(abs(start_depth), 1e-6)

    if relative_change >= DEPTH_TREND_RELATIVE_THRESHOLD:
        trend = "approaching"
    elif relative_change <= -DEPTH_TREND_RELATIVE_THRESHOLD:
        trend = "receding"
    else:
        trend = "stable"

    return {
        "current_depth": current_depth,
        "relative_change": relative_change,
        "trend": trend,
    }


def normalize_vector(vector):
    vx, vy = vector
    magnitude = math.hypot(vx, vy)
    if magnitude == 0:
        return None
    return (vx / magnitude, vy / magnitude)


def format_vector(vector):
    if vector is None:
        return "None"
    return f"({vector[0]:.3f}, {vector[1]:.3f})"


def dot_product(vector_a, vector_b):
    return vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]


def build_reference_state(polylines):
    polylines_by_lane = defaultdict(list)
    for polyline_index in sorted(polylines):
        polyline = polylines[polyline_index]
        if polyline["lane_group"] == LANE_UNKNOWN:
            continue
        polylines_by_lane[polyline["lane_group"]].append(polyline)

    return {
        "all_polylines": [polylines[index] for index in sorted(polylines)],
        "polylines_by_lane": dict(polylines_by_lane),
    }


def get_bbox_polygon(xyxy):
    x1, y1, x2, y2 = xyxy
    return [
        (float(x1), float(y1)),
        (float(x2), float(y1)),
        (float(x2), float(y2)),
        (float(x1), float(y2)),
    ]


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    contour = np.array(points, dtype=np.float32)
    return float(abs(cv2.contourArea(contour)))


def compute_polygon_overlap_area(bbox_polygon, lane_polygon):
    all_x = [point[0] for point in bbox_polygon] + [point[0] for point in lane_polygon]
    all_y = [point[1] for point in bbox_polygon] + [point[1] for point in lane_polygon]
    min_x = int(math.floor(min(all_x)))
    min_y = int(math.floor(min(all_y)))
    max_x = int(math.ceil(max(all_x)))
    max_y = int(math.ceil(max(all_y)))

    width = max_x - min_x + 3
    height = max_y - min_y + 3
    if width <= 0 or height <= 0:
        return 0.0

    bbox_points = np.array(
        [[int(round(x - min_x + 1)), int(round(y - min_y + 1))] for x, y in bbox_polygon],
        dtype=np.int32,
    )
    lane_points = np.array(
        [[int(round(x - min_x + 1)), int(round(y - min_y + 1))] for x, y in lane_polygon],
        dtype=np.int32,
    )

    bbox_mask = np.zeros((height, width), dtype=np.uint8)
    lane_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(bbox_mask, [bbox_points], 1)
    cv2.fillPoly(lane_mask, [lane_points], 1)
    overlap_mask = cv2.bitwise_and(bbox_mask, lane_mask)
    return float(np.count_nonzero(overlap_mask))


def get_lane_group(xyxy, polygons):
    bbox_polygon = get_bbox_polygon(xyxy)
    bbox_area = polygon_area(bbox_polygon)
    overlap_by_lane = {
        LANE_DOWN: 0.0,
        LANE_UP: 0.0,
    }
    polygon_debug = []

    for polygon_index in sorted(polygons):
        polygon_info = polygons[polygon_index]
        lane_group = polygon_info["lane_group"]
        if lane_group not in overlap_by_lane:
            continue

        overlap_area = compute_polygon_overlap_area(bbox_polygon, polygon_info["points"])
        overlap_by_lane[lane_group] += overlap_area
        polygon_debug.append(
            {
                "polygon_index": polygon_index,
                "lane_group": lane_group,
                "overlap_area": overlap_area,
            }
        )

    down_overlap = overlap_by_lane[LANE_DOWN]
    up_overlap = overlap_by_lane[LANE_UP]
    down_ratio = down_overlap / bbox_area if bbox_area > 0 else 0.0
    up_ratio = up_overlap / bbox_area if bbox_area > 0 else 0.0

    lane_group = LANE_UNKNOWN
    selected_polygon = None
    if max(down_ratio, up_ratio) >= LANE_OVERLAP_MIN_RATIO:
        if down_overlap > up_overlap:
            lane_group = LANE_DOWN
        elif up_overlap > down_overlap:
            lane_group = LANE_UP

    for polygon_index in sorted(polygons):
        polygon_info = polygons[polygon_index]
        if polygon_info["lane_group"] == lane_group:
            selected_polygon = polygon_info
            break

    return (
        lane_group,
        selected_polygon,
        {
            "bbox_area": bbox_area,
            "overlap_down": down_overlap,
            "overlap_up": up_overlap,
            "overlap_ratio_down": down_ratio,
            "overlap_ratio_up": up_ratio,
            "polygon_debug": polygon_debug,
        },
    )


def normalize_reference_direction_for_lane(local_dir, lane_type):
    expected_y_sign = LANE_EXPECTED_Y_SIGN.get(lane_type)
    if expected_y_sign is None:
        return local_dir

    if expected_y_sign > 0 and local_dir[1] < 0:
        return (-local_dir[0], -local_dir[1])
    if expected_y_sign < 0 and local_dir[1] > 0:
        return (-local_dir[0], -local_dir[1])
    return local_dir


def get_local_reference_direction(cx, cy, polyline, lane_type):
    points = polyline["points"]
    if len(points) < 2:
        return None

    nearest_index = None
    nearest_distance_sq = None
    for point_index, point in enumerate(points):
        distance_sq = (cx - point[0]) ** 2 + (cy - point[1]) ** 2
        if nearest_distance_sq is None or distance_sq < nearest_distance_sq:
            nearest_distance_sq = distance_sq
            nearest_index = point_index

    if nearest_index is None:
        return None

    start_idx = max(0, nearest_index - LOCAL_DIRECTION_POINT_RADIUS)
    end_idx = min(len(points) - 1, nearest_index + LOCAL_DIRECTION_POINT_RADIUS)
    if start_idx == end_idx:
        return None

    start_point = points[start_idx]
    end_point = points[end_idx]
    local_dir = normalize_vector((end_point[0] - start_point[0], end_point[1] - start_point[1]))
    if local_dir is None:
        return None

    local_dir = normalize_reference_direction_for_lane(local_dir, lane_type)
    nearest_distance = math.sqrt(nearest_distance_sq)
    return {
        "polyline_index": polyline["polyline_index"],
        "nearest_point_index": nearest_index,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start_point": start_point,
        "end_point": end_point,
        "points": points,
        "nearest_distance": nearest_distance,
        "local_dir": local_dir,
    }


def combine_reference_directions(local_direction_infos):
    if not local_direction_infos:
        return None, []

    weighted_dx = 0.0
    weighted_dy = 0.0
    weighted_infos = []
    for info in local_direction_infos:
        weight = 1.0 / max(info["nearest_distance"], REFERENCE_WEIGHT_EPSILON)
        weighted_dx += info["local_dir"][0] * weight
        weighted_dy += info["local_dir"][1] * weight
        weighted_infos.append({**info, "weight": weight})

    combined_dir = normalize_vector((weighted_dx, weighted_dy))
    if combined_dir is None:
        return None, weighted_infos

    return combined_dir, weighted_infos


def compute_track_motion(track_history, window_size):
    if len(track_history) < 2:
        return None

    usable_points = list(track_history)[-max(window_size + 1, 2) :]
    motion_vectors = []
    for previous_point, current_point in zip(usable_points[:-1], usable_points[1:]):
        dx = current_point[0] - previous_point[0]
        dy = current_point[1] - previous_point[1]
        motion_vectors.append((dx, dy))

    if not motion_vectors:
        return None

    avg_dx = sum(vector[0] for vector in motion_vectors) / len(motion_vectors)
    avg_dy = sum(vector[1] for vector in motion_vectors) / len(motion_vectors)
    return (avg_dx, avg_dy)


def compute_raw_heading(motion_vec):
    motion_magnitude = math.hypot(motion_vec[0], motion_vec[1])
    if motion_magnitude < MIN_MOTION_MAGNITUDE:
        return None

    return {
        "raw_heading": (motion_vec[0] / motion_magnitude, motion_vec[1] / motion_magnitude),
        "motion_magnitude": motion_magnitude,
    }


def correct_heading(raw_heading, reference_direction):
    if raw_heading is None:
        return None

    if reference_direction is None:
        return {
            "corrected_heading": raw_heading,
            "dot_product": None,
            "flipped": False,
        }

    direction_dot = dot_product(raw_heading, reference_direction)
    if direction_dot < 0:
        return {
            "corrected_heading": (-raw_heading[0], -raw_heading[1]),
            "dot_product": direction_dot,
            "flipped": True,
        }

    return {
        "corrected_heading": raw_heading,
        "dot_product": direction_dot,
        "flipped": False,
    }


def build_debug_heading_info(track_id, detection_id, bbox_center, lane_group, lane_overlap_info, reference_direction, candidate_infos, raw_heading_info, corrected_heading_info):
    primary_info = None
    if candidate_infos:
        primary_info = min(candidate_infos, key=lambda item: item["nearest_distance"])

    return {
        "track_id": track_id,
        "detection_id": detection_id,
        "bbox_center": bbox_center,
        "lane_group": lane_group,
        "lane_group_name": LANE_LABELS_KO.get(lane_group, "unknown"),
        "lane_overlap_info": lane_overlap_info,
        "reference_direction": reference_direction,
        "candidate_infos": candidate_infos,
        "primary_info": primary_info,
        "raw_heading": raw_heading_info["raw_heading"] if raw_heading_info else None,
        "motion_magnitude": raw_heading_info["motion_magnitude"] if raw_heading_info else None,
        "corrected_heading": corrected_heading_info["corrected_heading"] if corrected_heading_info else None,
        "dot_product": corrected_heading_info["dot_product"] if corrected_heading_info else None,
        "flipped": corrected_heading_info["flipped"] if corrected_heading_info else False,
        "depth_info": None,
        "risk_detected": False,
    }


def log_vehicle_heading_debug(debug_info):
    del debug_info


def draw_reference_lines(frame, heading_state):
    reference_state = heading_state["reference_state"]
    for polyline in reference_state["all_polylines"]:
        lane_group = polyline["lane_group"]
        color = (0, 165, 255) if lane_group == LANE_DOWN else (0, 255, 0)
        points = np.array([(int(round(x)), int(round(y))) for x, y in polyline["points"]], dtype=np.int32)
        cv2.polylines(frame, [points], False, color, 2)


def draw_status_overlay(frame):
    if not DEPTH_STATUS_MESSAGE:
        return
    cv2.putText(
        frame,
        DEPTH_STATUS_MESSAGE,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_debug_overlay(frame, xyxy, debug_info):
    if not DEBUG_HEADING_OVERLAY or debug_info is None:
        return

    center_x, center_y = debug_info["bbox_center"]
    start_point = (int(round(center_x)), int(round(center_y)))

    if debug_info["corrected_heading"] is not None:
        corrected_end = (
            int(round(start_point[0] + debug_info["corrected_heading"][0] * ARROW_LENGTH_PIXELS)),
            int(round(start_point[1] + debug_info["corrected_heading"][1] * ARROW_LENGTH_PIXELS)),
        )
        cv2.arrowedLine(frame, start_point, corrected_end, (0, 0, 255), 2, tipLength=0.25)

    if debug_info.get("accident_detected"):
        x1, y1, x2, _ = [int(round(value)) for value in xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, int(round(xyxy[3]))), (0, 0, 255), 2)
        cv2.putText(
            frame,
            "ACCIDENT",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    elif debug_info.get("risk_detected"):
        x1, y1, _, _ = [int(round(value)) for value in xyxy]
        cv2.putText(
            frame,
            "RISK",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    depth_info = debug_info.get("depth_info")
    if depth_info is not None:
        x1, _, _, y2 = [int(round(value)) for value in xyxy]
        trend = depth_info["trend"]
        trend_label = {
            "approaching": "CLOSER",
            "receding": "FARTHER",
            "stable": "STABLE",
            "unknown": "DEPTH ?",
        }.get(trend, "DEPTH ?")
        cv2.putText(
            frame,
            f"{trend_label} {depth_info['current_depth']:.2f}",
            (x1, min(y2 + 20, frame.shape[0] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "사고 발생",
            (x1, max(y1 - 40, 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


def orientation(a, b, c):
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) < 1e-6:
        return 0
    return 1 if value > 0 else 2


def on_segment(a, b, c):
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def arrow_intersects_polyline(arrow_start, arrow_end, polyline_points):
    if len(polyline_points) < 2:
        return False

    for start_point, end_point in zip(polyline_points[:-1], polyline_points[1:]):
        if segments_intersect(arrow_start, arrow_end, start_point, end_point):
            return True
    return False


def is_center_stationary(track_history):
    if not track_history or len(track_history) < 2:
        return False

    usable_points = list(track_history)[-max(HEADING_WINDOW + 1, 2):]
    start_point = usable_points[0]
    end_point = usable_points[-1]
    distance = math.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1])
    return distance <= ACCIDENT_STATIONARY_DISTANCE_THRESHOLD


def detect_accident(debug_info, track_history, heading_state):
    corrected_heading = debug_info.get("corrected_heading")
    bbox_center = debug_info.get("bbox_center")
    lane_group = debug_info.get("lane_group")
    if corrected_heading is None or bbox_center is None:
        return False
    if lane_group == LANE_UNKNOWN or not is_center_stationary(track_history):
        return False

    arrow_start = bbox_center
    arrow_end = (
        bbox_center[0] + corrected_heading[0] * ARROW_LENGTH_PIXELS,
        bbox_center[1] + corrected_heading[1] * ARROW_LENGTH_PIXELS,
    )

    lane_polylines = heading_state["reference_state"]["polylines_by_lane"].get(lane_group, [])
    for polyline in lane_polylines:
        if arrow_intersects_polyline(arrow_start, arrow_end, polyline["points"]):
            return True
    return False


def draw_tracking_results(frame, result, heading_state=None, track_histories=None, depth_histories=None, last_heading_by_track=None):
    annotated = frame.copy()
    boxes = result.boxes

    if heading_state and heading_state.get("enabled"):
        draw_reference_lines(annotated, heading_state)
    draw_status_overlay(annotated)

    if boxes is None or len(boxes) == 0:
        return annotated

    xyxy_list = boxes.xyxy.cpu().tolist()
    conf_list = boxes.conf.cpu().tolist()
    track_ids = None
    if getattr(boxes, "id", None) is not None:
        track_ids = boxes.id.int().cpu().tolist()

    for index, (xyxy, conf) in enumerate(zip(xyxy_list, conf_list)):
        del conf
        x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
        bbox_center = get_bbox_center(xyxy)
        track_id = int(track_ids[index]) if track_ids is not None else None
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

        if not heading_state or not heading_state.get("enabled"):
            continue
        if track_id is None or track_histories is None:
            continue

        try:
            track_history = track_histories.get(track_id)
            depth_history = depth_histories.get(track_id) if depth_histories is not None else None
            motion_vec = compute_track_motion(track_history, HEADING_WINDOW) if track_history else None
            raw_heading_info = compute_raw_heading(motion_vec) if motion_vec is not None else None

            cx, cy = bbox_center
            lane_group, polygon_info, lane_overlap_info = get_lane_group(xyxy, heading_state["lane_polygons"])
            del polygon_info

            reference_direction = None
            candidate_infos = []
            if lane_group != LANE_UNKNOWN:
                lane_polylines = heading_state["reference_state"]["polylines_by_lane"].get(lane_group, [])
                local_direction_infos = []
                for polyline in lane_polylines:
                    local_info = get_local_reference_direction(cx, cy, polyline, lane_group)
                    if local_info is not None:
                        local_direction_infos.append(local_info)
                reference_direction, candidate_infos = combine_reference_directions(local_direction_infos)

            corrected_heading_info = None
            if raw_heading_info is not None:
                corrected_heading_info = correct_heading(raw_heading_info["raw_heading"], reference_direction)

            debug_info = build_debug_heading_info(
                track_id=track_id,
                detection_id=index,
                bbox_center=bbox_center,
                lane_group=lane_group,
                lane_overlap_info=lane_overlap_info,
                reference_direction=reference_direction,
                candidate_infos=candidate_infos,
                raw_heading_info=raw_heading_info,
                corrected_heading_info=corrected_heading_info,
            )

            if corrected_heading_info is not None and last_heading_by_track is not None:
                last_heading_by_track[track_id] = debug_info
            elif last_heading_by_track is not None:
                cached_info = last_heading_by_track.get(track_id)
                if cached_info is not None:
                    debug_info = {
                        **cached_info,
                        "bbox_center": bbox_center,
                        "detection_id": index,
                        "lane_overlap_info": lane_overlap_info,
                    }

            if debug_info is None or debug_info["corrected_heading"] is None:
                continue

            debug_info["depth_info"] = compute_depth_info(depth_history)
            if debug_info["depth_info"] is not None:
                debug_info["risk_detected"] = (
                    debug_info["flipped"] and debug_info["depth_info"]["trend"] == "approaching"
                )
            debug_info["accident_detected"] = detect_accident(debug_info, track_history, heading_state)
            log_vehicle_heading_debug(debug_info)
            draw_debug_overlay(annotated, xyxy, debug_info)
        except Exception as error:
            log_heading_status(f"track_id={track_id} heading 계산 실패, 해당 객체는 건너뜁니다: {error}")

    return annotated


def build_heading_state():
    if not ENABLE_HEADING:
        log_heading_status("ENABLE_HEADING=False 이므로 heading 기능을 비활성화합니다.")
        return {"enabled": False, "reason": "disabled_by_config"}

    lane_polygons = load_lane_polygons()
    if not lane_polygons:
        return {"enabled": False, "reason": "missing_lane_polygons"}

    reference_polylines = load_reference_polylines()
    if not reference_polylines:
        return {"enabled": False, "reason": "missing_reference_polylines"}

    try:
        reference_state = build_reference_state(reference_polylines)
        log_heading_status("heading 활성화 완료 [lane_polygons + reference_polylines]")
        return {
            "enabled": True,
            "lane_polygons": lane_polygons,
            "reference_state": reference_state,
        }
    except Exception as error:
        log_heading_status(f"reference geometry 오류로 heading 기능을 비활성화합니다: {error}")
        return {"enabled": False, "reason": "invalid_reference_geometry"}


def update_track_histories(track_histories, result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0 or getattr(boxes, "id", None) is None:
        return

    xyxy_list = boxes.xyxy.cpu().tolist()
    track_ids = boxes.id.int().cpu().tolist()

    for track_id, xyxy in zip(track_ids, xyxy_list):
        history = track_histories[int(track_id)]
        history.append(get_bbox_center(xyxy))


def update_depth_histories(depth_histories, result, depth_map):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0 or getattr(boxes, "id", None) is None:
        return

    xyxy_list = boxes.xyxy.cpu().tolist()
    track_ids = boxes.id.int().cpu().tolist()

    for track_id, xyxy in zip(track_ids, xyxy_list):
        sampled_depth = sample_bbox_depth(depth_map, xyxy)
        if sampled_depth is None:
            continue
        history = depth_histories[int(track_id)]
        history.append(sampled_depth)


def open_camera():
    capture = cv2.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"카메라를 열 수 없습니다: index={CAMERA_INDEX}")
    return capture


def main():
    model = load_model()
    depth_runner = load_depth_model()
    heading_state = build_heading_state()
    track_histories = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LENGTH))
    depth_histories = defaultdict(lambda: deque(maxlen=DEPTH_HISTORY_LENGTH))
    last_heading_by_track = {}
    capture = open_camera()

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        depth_map = estimate_depth_map(frame, depth_runner)

        results = model.track(
            frame,
            persist=True,
            classes=VEHICLE_CLASSES,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        if results:
            try:
                update_track_histories(track_histories, results[0])
                update_depth_histories(depth_histories, results[0], depth_map)
            except Exception as error:
                log_heading_status(f"track history 업데이트 실패, heading 계산을 이번 프레임에서 건너뜁니다: {error}")

        annotated_frame = draw_tracking_results(
            frame,
            results[0],
            heading_state=heading_state,
            track_histories=track_histories,
            depth_histories=depth_histories,
            last_heading_by_track=last_heading_by_track,
        )

        cv2.imshow("Crash Heading Test", annotated_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
