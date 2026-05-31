import json
import math
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

# ultralytics가 홈 디렉터리에 설정 파일을 만들지 못하는 환경 대응
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
MODULE_DIR = Path(__file__).resolve().parent

# 학습에 사용한 워크스페이스 내 Ultralytics 구현을 우선 사용합니다.
VENDORED_ULTRALYTICS_ROOT = MODULE_DIR / "yolov12" / "yolov12"
if VENDORED_ULTRALYTICS_ROOT.exists():
    sys.path.insert(0, str(VENDORED_ULTRALYTICS_ROOT))

import cv2
import numpy as np
import pandas as pd
import requests
import torch
from ultralytics import YOLO

API_KEY = "f742615880e442b4a0e549771bbef99d"
API_URL = "https://openapi.its.go.kr:9443/cctvInfo"
WEIGHTS_PATH = MODULE_DIR / "pruned.pt"
YAML_PATH = MODULE_DIR / "pruned.yaml"
DIRECT_CCTV_URL = ""
DIRECT_CCTV_NAME = ""
HEADING_GROUP_JSON_PATH = MODULE_DIR / "heading_2.json"

# 청주 인근 고속도로 CCTV를 조회합니다.
# 보고 싶은 화면으로 바꾸려면 아래 CCTV_INDEX 값을 수정하세요.
TARGET_NAME = "청주"
CCTV_INDEX = 2
VEHICLE_CLASSES = [0]  # heading 계산에 사용할 검출 클래스 id
CONFIDENCE_THRESHOLD = 0.3  # 낮은 신뢰도 bbox를 제외해 tracking/heading 흔들림을 줄임
LIST_LIMIT = 20  # CCTV 목록 출력 개수
FRAME_WAIT_TIMEOUT_SECONDS = 10  # 프레임 수신 대기 최대 시간
FRAME_WAIT_INTERVAL_SECONDS = 0.2  # 프레임 재시도 간격
ENABLE_HEADING = True  # heading 계산 기능 on/off
HEADING_WINDOW = 4  # 최근 몇 프레임 이동을 평균내 heading 방향을 계산할지
MIN_MOTION_MAGNITUDE = 0.1  # 이동량이 이 값보다 작으면 heading 계산 생략
TRACK_HISTORY_LENGTH = 20  # 차량별 위치 이력을 몇 프레임까지 유지할지
LOCAL_DIRECTION_POINT_RADIUS = 2  # nearest point 좌우로 사용할 reference point 개수
REFERENCE_WEIGHT_EPSILON = 1.0  # 거리 가중 평균에서 0으로 나누는 문제 방지
LANE_OVERLAP_MIN_RATIO = 0.01  # bbox 대비 lane polygon 겹침 비율이 이 값보다 작으면 unknown 처리
DEBUG_HEADING_LOGS = True
DEBUG_HEADING_OVERLAY = True

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

REQUEST_PARAMS = {
    "apiKey": API_KEY,
    "type": "all",
    "cctvType": "1",
    "minX": 127.25,
    "maxX": 127.60,
    "minY": 36.50,
    "maxY": 36.80,
    "getType": "json",
}


def log_heading_status(message):
    print(f"[heading] {message}")


def resolve_roi_path(cctv_index):
    return MODULE_DIR / "main" / "roi" / f"roi_data_{int(cctv_index)}.json"


def load_lane_polygons(cctv_index):
    roi_path = resolve_roi_path(cctv_index)
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


def resolve_heading_group_json_path(cctv_index):
    current_path = Path(HEADING_GROUP_JSON_PATH)
    if current_path.exists():
        return current_path
    return MODULE_DIR / f"heading_{int(cctv_index)}.json"


def load_reference_polylines(cctv_index):
    json_path = resolve_heading_group_json_path(cctv_index)
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


def fetch_cctv_dataframe():
    query = "&".join(f"{key}={value}" for key, value in REQUEST_PARAMS.items())
    url = f"{API_URL}?{query}"
    response = urlopen(url)
    payload = json.loads(response.read().decode("utf-8"))

    data = payload.get("response", {}).get("data", [])
    if not data:
        raise RuntimeError("CCTV 목록을 불러오지 못했습니다.")

    return pd.json_normalize(data)


def select_cctv(cctv_df):
    print("조회된 CCTV 목록:")
    for index, row in cctv_df.head(LIST_LIMIT).iterrows():
        print(f"[{index}] {row['cctvname']}")

    if CCTV_INDEX is not None:
        if CCTV_INDEX < 0 or CCTV_INDEX >= len(cctv_df):
            raise RuntimeError(f"CCTV_INDEX 범위가 잘못되었습니다: {CCTV_INDEX}")

        row = cctv_df.iloc[CCTV_INDEX]
        return row["cctvname"], row["cctvurl"]

    matched = cctv_df[cctv_df["cctvname"].astype(str).str.contains(TARGET_NAME, na=False)]
    if matched.empty:
        names = cctv_df["cctvname"].astype(str).dropna().tolist()
        sample_names = ", ".join(names[:10]) if names else "없음"
        raise RuntimeError(
            f"이름으로 CCTV를 찾지 못했습니다: {TARGET_NAME}\n"
            f"현재 조회된 CCTV 예시: {sample_names}"
        )

    row = matched.iloc[0]
    return row["cctvname"], row["cctvurl"]


def resolve_stream_url(cctv_url):
    response = requests.get(cctv_url, allow_redirects=True, timeout=10)
    response.raise_for_status()

    playlist_url = response.url
    playlist_text = response.text

    for line in playlist_text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return urljoin(playlist_url, line)

    return cctv_url


def open_capture(url):
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if capture.isOpened():
        return capture

    capture.release()
    capture = cv2.VideoCapture(url)
    if capture.isOpened():
        return capture

    capture.release()
    return None


def connect_capture(stream_url, cctv_url):
    capture = open_capture(stream_url)
    if capture is not None:
        return capture

    print("실제 스트림 URL 연결 실패, 원본 cctvurl로 다시 시도합니다.")
    return open_capture(cctv_url)


def read_frame_with_timeout(capture, timeout_seconds):
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        ok, frame = capture.read()
        if ok:
            return True, frame

        time.sleep(FRAME_WAIT_INTERVAL_SECONDS)

    return False, None


def load_model():
    if not YAML_PATH.exists():
        raise FileNotFoundError(f"모델 설정 파일을 찾지 못했습니다: {YAML_PATH}")
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"모델 가중치 파일을 찾지 못했습니다: {WEIGHTS_PATH}")

    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = patched_torch_load
        model = YOLO(str(YAML_PATH), task="detect").load(str(WEIGHTS_PATH))
    finally:
        torch.load = original_torch_load

    print("YOLO model yaml:", YAML_PATH)
    print("YOLO model weights:", WEIGHTS_PATH)
    print("tracking classes:", VEHICLE_CLASSES)
    return model


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
    }


def log_vehicle_heading_debug(debug_info):
    if not DEBUG_HEADING_LOGS or debug_info is None:
        return

    candidate_texts = []
    for info in debug_info["candidate_infos"]:
        candidate_texts.append(
            "polyline={polyline} k={k} range={start}~{end} dist={dist:.1f} weight={weight:.3f} local={local}".format(
                polyline=info["polyline_index"],
                k=info["nearest_point_index"],
                start=info["start_idx"],
                end=info["end_idx"],
                dist=info["nearest_distance"],
                weight=info.get("weight", 0.0),
                local=format_vector(info["local_dir"]),
            )
        )

    dot_value = debug_info["dot_product"]
    dot_text = "None" if dot_value is None else f"{dot_value:.3f}"
    overlap_info = debug_info["lane_overlap_info"] or {}
    log_heading_status(
        "track_id={track_id} det={det} center=({cx:.1f}, {cy:.1f}) lane={lane} overlap_down={overlap_down:.1f} overlap_up={overlap_up:.1f} ratio_down={ratio_down:.3f} ratio_up={ratio_up:.3f} raw={raw} ref={ref} dot={dot} corrected={corrected} flipped={flipped} candidates=[{candidates}]".format(
            track_id=debug_info["track_id"],
            det=debug_info["detection_id"],
            cx=debug_info["bbox_center"][0],
            cy=debug_info["bbox_center"][1],
            lane=debug_info["lane_group_name"],
            overlap_down=overlap_info.get("overlap_down", 0.0),
            overlap_up=overlap_info.get("overlap_up", 0.0),
            ratio_down=overlap_info.get("overlap_ratio_down", 0.0),
            ratio_up=overlap_info.get("overlap_ratio_up", 0.0),
            raw=format_vector(debug_info["raw_heading"]),
            ref=format_vector(debug_info["reference_direction"]),
            dot=dot_text,
            corrected=format_vector(debug_info["corrected_heading"]),
            flipped=debug_info["flipped"],
            candidates="; ".join(candidate_texts) if candidate_texts else "none",
        )
    )


def draw_reference_lines(frame, heading_state):
    reference_state = heading_state["reference_state"]
    for polyline in reference_state["all_polylines"]:
        lane_group = polyline["lane_group"]
        color = (0, 165, 255) if lane_group == LANE_DOWN else (0, 255, 0)
        points = np.array([(int(round(x)), int(round(y))) for x, y in polyline["points"]], dtype=np.int32)
        cv2.polylines(frame, [points], False, color, 2)
        cv2.putText(
            frame,
            f"poly_{polyline['polyline_index']}",
            tuple(points[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_debug_overlay(frame, xyxy, debug_info):
    if not DEBUG_HEADING_OVERLAY or debug_info is None:
        return

    center_x, center_y = debug_info["bbox_center"]
    start_point = (int(round(center_x)), int(round(center_y)))
    lane_group = debug_info["lane_group"]
    lane_color = (0, 165, 255) if lane_group == LANE_DOWN else (0, 255, 0) if lane_group == LANE_UP else (180, 180, 180)

    if debug_info["corrected_heading"] is not None:
        corrected_end = (
            int(round(start_point[0] + debug_info["corrected_heading"][0] * 40)),
            int(round(start_point[1] + debug_info["corrected_heading"][1] * 40)),
        )
        cv2.arrowedLine(frame, start_point, corrected_end, (0, 0, 255), 2, tipLength=0.25)

    if debug_info["reference_direction"] is not None:
        reference_end = (
            int(round(start_point[0] + debug_info["reference_direction"][0] * 32)),
            int(round(start_point[1] + debug_info["reference_direction"][1] * 32)),
        )
        cv2.arrowedLine(frame, start_point, reference_end, (255, 255, 0), 2, tipLength=0.22)

    for info in debug_info["candidate_infos"]:
        point_color = (255, 200, 0) if info is debug_info["primary_info"] else (160, 160, 0)

        for point_index in range(info["start_idx"], info["end_idx"] + 1):
            px, py = info["points"][point_index]
            cv2.circle(frame, (int(round(px)), int(round(py))), 3, point_color, -1)

        cv2.line(
            frame,
            (int(round(info["start_point"][0])), int(round(info["start_point"][1]))),
            (int(round(info["end_point"][0])), int(round(info["end_point"][1]))),
            point_color,
            2,
        )

    cv2.circle(frame, start_point, 4, (255, 255, 255), -1)
    label = f"{debug_info['lane_group_name']} {'flip' if debug_info['flipped'] else 'keep'}"
    cv2.putText(
        frame,
        label,
        (int(round(xyxy[0])), max(15, int(round(xyxy[1])) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        lane_color,
        2,
        cv2.LINE_AA,
    )


def draw_tracking_results(frame, result, heading_state=None, track_histories=None, last_heading_by_track=None):
    annotated = frame.copy()
    boxes = result.boxes

    if heading_state and heading_state.get("enabled"):
        draw_reference_lines(annotated, heading_state)

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
        color = (0, 255, 0)
        track_id = int(track_ids[index]) if track_ids is not None else None

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

        if not heading_state or not heading_state.get("enabled"):
            continue
        if track_id is None or track_histories is None:
            continue

        try:
            track_history = track_histories.get(track_id)
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

            log_vehicle_heading_debug(debug_info)
            draw_debug_overlay(annotated, xyxy, debug_info)
        except Exception as error:
            log_heading_status(f"track_id={track_id} heading 계산 실패, 해당 객체는 건너뜁니다: {error}")

    return annotated


def build_heading_state(cctv_index):
    if not ENABLE_HEADING:
        log_heading_status("ENABLE_HEADING=False 이므로 heading 기능을 비활성화합니다.")
        return {"enabled": False, "reason": "disabled_by_config"}

    lane_polygons = load_lane_polygons(cctv_index)
    if not lane_polygons:
        return {"enabled": False, "reason": "missing_lane_polygons"}

    reference_polylines = load_reference_polylines(cctv_index)
    if not reference_polylines:
        return {"enabled": False, "reason": "missing_reference_polylines"}

    try:
        reference_state = build_reference_state(reference_polylines)
        log_heading_status(f"index={cctv_index} heading 활성화 완료 [lane_polygons + reference_polylines]")
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


def main():
    model = load_model()
    if DIRECT_CCTV_URL:
        cctv_name = DIRECT_CCTV_NAME or "직접 지정 CCTV"
        cctv_url = DIRECT_CCTV_URL
        selected_cctv_index = CCTV_INDEX
    else:
        cctv_df = fetch_cctv_dataframe()
        cctv_name, cctv_url = select_cctv(cctv_df)
        selected_cctv_index = CCTV_INDEX

    print("선택 CCTV:", cctv_name)
    print("cctvurl:", cctv_url)

    heading_state = build_heading_state(selected_cctv_index)
    track_histories = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LENGTH))
    last_heading_by_track = {}

    stream_url = resolve_stream_url(cctv_url)
    print("stream url:", stream_url)

    capture = connect_capture(stream_url, cctv_url)

    if capture is None:
        raise RuntimeError("OpenCV에서 CCTV 스트림을 열지 못했습니다.")

    while True:
        if capture is None or not capture.isOpened():
            print("캡처가 닫혀 있어 재연결을 시도합니다.")
            time.sleep(1)
            stream_url = resolve_stream_url(cctv_url)
            print("재연결 stream url:", stream_url)
            capture = connect_capture(stream_url, cctv_url)
            continue

        ok, frame = read_frame_with_timeout(capture, FRAME_WAIT_TIMEOUT_SECONDS)
        if not ok:
            print(f"[프레임 수신 불가] {FRAME_WAIT_TIMEOUT_SECONDS}초 동안 기다렸지만 프레임이 오지 않아 재연결합니다")
            capture.release()
            cv2.destroyWindow("ITS CCTV Live")
            time.sleep(1)

            stream_url = resolve_stream_url(cctv_url)
            print("재연결 stream url:", stream_url)
            capture = connect_capture(stream_url, cctv_url)

            if capture is None:
                print("재연결 실패, 1초 후 다시 시도합니다.")
                time.sleep(1)
            continue

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
            except Exception as error:
                log_heading_status(f"track history 업데이트 실패, heading 계산을 이번 프레임에서 건너뜁니다: {error}")

        annotated_frame = draw_tracking_results(
            frame,
            results[0],
            heading_state=heading_state,
            track_histories=track_histories,
            last_heading_by_track=last_heading_by_track,
        )

        cv2.imshow("ITS CCTV Live", annotated_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
