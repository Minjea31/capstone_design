import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent

VENDORED_ULTRALYTICS_ROOT = PROJECT_ROOT / "yolov12" / "yolov12"
if VENDORED_ULTRALYTICS_ROOT.exists():
    sys.path.insert(0, str(VENDORED_ULTRALYTICS_ROOT))

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from utils.accident import AccidentLogDetector
from utils.video_source import find_saved_videos, get_video_delay_ms, open_video_capture, resolve_video_path

WEIGHTS_PATH = MODULE_DIR / "crash.pt"
REFERENCE_DIR = MODULE_DIR / "referenceline"
ROI_JSON_PATH = REFERENCE_DIR / "crash.json"
HEADING_GROUP_JSON_PATH = REFERENCE_DIR / "crash_reference.json"
VEHICLE_CLASSES = [0]
CONFIDENCE_THRESHOLD = 0.7
ENABLE_HEADING = True
HEADING_WINDOW = 4
MIN_MOTION_MAGNITUDE = 0.1
TRACK_HISTORY_LENGTH = 20
LOCAL_DIRECTION_POINT_RADIUS = 2
REFERENCE_WEIGHT_EPSILON = 1.0
LANE_OVERLAP_MIN_RATIO = 0.01
DEBUG_HEADING_LOGS = os.environ.get("DEBUG_HEADING_LOGS", "0").lower() in {"1", "true", "yes"}
DEBUG_HEADING_OVERLAY = True
DEBUG_LOG_EVERY_FRAMES = 15
ARROW_LENGTH_PIXELS = 80
ACCIDENT_STATIONARY_DISTANCE_THRESHOLD = 3.0
HEADING_CENTER_STABILITY_THRESHOLD = 3.0
ACCIDENT_PERSISTENCE_SECONDS = 10.0
ACCIDENT_SCORE_INC = 1.0
ACCIDENT_OVERLAP_SCORE_INC = 1.5
ACCIDENT_SCORE_DEC = 0.6
ACCIDENT_ON_THRESHOLD = 3.0
ACCIDENT_OFF_THRESHOLD = 1.0
ACCIDENT_MIN_STATIONARY_FRAMES = 3
MAX_ACCIDENT_SCORE = 5.0
CONGESTION_MIN_STATIONARY_FRAMES = 6
CONGESTION_MIN_VEHICLES_PER_LANE = 2
CONGESTION_SCORE_INC = 0.8
CONGESTION_SCORE_DEC = 0.5
CONGESTION_ON_THRESHOLD = 3.0
CONGESTION_OFF_THRESHOLD = 1.0
MAX_CONGESTION_SCORE = 5.0
SAME_FLOW_DOT_THRESHOLD = 0.65
ACCIDENT_CONFLICT_DOT_THRESHOLD = 0.35
BBOX_OVERLAP_IOU_THRESHOLD = 0.01
BBOX_OVERLAP_MIN_RATIO_THRESHOLD = 0.02
SAME_FLOW_OVERLAP_IOU_THRESHOLD = 0.02
SAME_FLOW_OVERLAP_MIN_RATIO_THRESHOLD = 0.04

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
    if DEBUG_HEADING_LOGS:
        print(f"[crash_detect] {message}", flush=True)


def lane_group_from_label(label):
    try:
        label = int(label)
    except (TypeError, ValueError):
        return LANE_UNKNOWN

    if label == LANE_DOWN:
        return LANE_DOWN
    if label == LANE_UP:
        return LANE_UP
    return LANE_UNKNOWN


def lane_group_from_reference_group(group, polyline_index):
    label_counts = defaultdict(int)
    roi_labels = group.get("roi_labels", [])
    if isinstance(roi_labels, list):
        for label in roi_labels:
            lane_group = lane_group_from_label(label)
            if lane_group != LANE_UNKNOWN:
                label_counts[lane_group] += 1

    if label_counts[LANE_DOWN] > label_counts[LANE_UP]:
        return LANE_DOWN
    if label_counts[LANE_UP] > label_counts[LANE_DOWN]:
        return LANE_UP

    if polyline_index in LANE_REFERENCE_POLYLINE_INDEXES[LANE_DOWN]:
        return LANE_DOWN
    if polyline_index in LANE_REFERENCE_POLYLINE_INDEXES[LANE_UP]:
        return LANE_UP
    return LANE_UNKNOWN


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

        points_data, label = item
        if not isinstance(points_data, list):
            continue

        points = []
        for point in points_data:
            if not isinstance(point, list) or len(point) != 2:
                continue
            points.append((int(point[0]), int(point[1])))

        if len(points) >= 3:
            lane_group = lane_group_from_label(label)

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

        lane_group = lane_group_from_reference_group(group, polyline_index)

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


def is_center_stable(current_center, previous_center, threshold=HEADING_CENTER_STABILITY_THRESHOLD):
    if current_center is None or previous_center is None:
        return False
    return math.hypot(current_center[0] - previous_center[0], current_center[1] - previous_center[1]) <= threshold


def load_model():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"모델 가중치 파일을 찾지 못했습니다: {WEIGHTS_PATH}")

    log_heading_status(f"YOLO 모델 로딩 시작: {WEIGHTS_PATH}")
    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = patched_torch_load
        model = YOLO(str(WEIGHTS_PATH))
    finally:
        torch.load = original_torch_load

    log_heading_status("YOLO 모델 로딩 완료")
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


def is_lane_straddling(lane_overlap_info, min_overlap_ratio=0.1):
    if not lane_overlap_info:
        return False
    return (
        lane_overlap_info.get("overlap_ratio_down", 0.0) >= min_overlap_ratio
        and lane_overlap_info.get("overlap_ratio_up", 0.0) >= min_overlap_ratio
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

    usable_points = list(track_history)[-max(window_size + 1, 2):]
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
    if not DEBUG_HEADING_LOGS or not debug_info.get("should_log"):
        return

    candidate_info = debug_info.get("accident_candidate_info") or {}
    overlap = candidate_info.get("overlap") or {}
    dot = candidate_info.get("dot")
    dot_text = "none" if dot is None else f"{dot:.2f}"
    motion = debug_info.get("motion_magnitude")
    motion_text = "none" if motion is None else f"{motion:.2f}"

    log_heading_status(
        "frame={frame} track={track} lane={lane} motion={motion} stationary={stationary} "
        "acc_score={acc_score:.1f} cong_score={cong_score:.1f} accident={accident} congestion={congestion} "
        "candidate={candidate} reason={reason} other={other} iou={iou:.2f} min_overlap={min_overlap:.2f} dot={dot}".format(
            frame=debug_info.get("frame_index", "-"),
            track=debug_info.get("track_id", "-"),
            lane=debug_info.get("lane_group_name", "unknown"),
            motion=motion_text,
            stationary=debug_info.get("stationary_frames", 0),
            acc_score=debug_info.get("accident_score", 0.0),
            cong_score=debug_info.get("congestion_score", 0.0),
            accident=debug_info.get("accident_detected", False),
            congestion=debug_info.get("congestion_detected", False),
            candidate=candidate_info.get("candidate", False),
            reason=candidate_info.get("reason", "-"),
            other=candidate_info.get("other_track_id", "-"),
            iou=overlap.get("iou", 0.0),
            min_overlap=overlap.get("min_ratio", 0.0),
            dot=dot_text,
        )
    )


def draw_reference_lines(frame, heading_state):
    reference_state = heading_state["reference_state"]
    for polyline in reference_state["all_polylines"]:
        lane_group = polyline["lane_group"]
        color = (0, 165, 255) if lane_group == LANE_DOWN else (0, 255, 0)
        points = np.array([(int(round(x)), int(round(y))) for x, y in polyline["points"]], dtype=np.int32)
        cv2.polylines(frame, [points], False, color, 2)


def draw_debug_overlay(frame, xyxy, debug_info):
    if not DEBUG_HEADING_OVERLAY or debug_info is None:
        return

    center_x, center_y = debug_info["bbox_center"]
    start_point = (int(round(center_x)), int(round(center_y)))

    if debug_info.get("corrected_heading") is not None:
        corrected_end = (
            int(round(start_point[0] + debug_info["corrected_heading"][0] * ARROW_LENGTH_PIXELS)),
            int(round(start_point[1] + debug_info["corrected_heading"][1] * ARROW_LENGTH_PIXELS)),
        )
        cv2.arrowedLine(frame, start_point, corrected_end, (0, 0, 255), 2, tipLength=0.1)

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
    elif debug_info.get("congestion_detected"):
        x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            "CONGESTION",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
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


def point_to_segment_distance(point, seg_start, seg_end):
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    if t < 0:
        return math.hypot(px - x1, py - y1)
    if t > 1:
        return math.hypot(px - x2, py - y2)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def arrow_intersects_arrow(start_a, end_a, start_b, end_b):
    if segments_intersect(start_a, end_a, start_b, end_b):
        return True

    distance_threshold = 8.0
    if point_to_segment_distance(start_a, start_b, end_b) <= distance_threshold:
        return True
    if point_to_segment_distance(end_a, start_b, end_b) <= distance_threshold:
        return True
    if point_to_segment_distance(start_b, start_a, end_a) <= distance_threshold:
        return True
    if point_to_segment_distance(end_b, start_a, end_a) <= distance_threshold:
        return True

    return False


def get_arrow_direction(arrow_segment):
    if arrow_segment is None or len(arrow_segment) != 2:
        return None

    start_point, end_point = arrow_segment
    if start_point is None or end_point is None:
        return None

    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    magnitude = math.hypot(dx, dy)
    if magnitude <= 1e-6:
        return None
    return dx / magnitude, dy / magnitude


def get_arrow_dot(arrow_a, arrow_b):
    direction_a = get_arrow_direction(arrow_a)
    direction_b = get_arrow_direction(arrow_b)
    if direction_a is None or direction_b is None:
        return None
    return direction_a[0] * direction_b[0] + direction_a[1] * direction_b[1]


def arrows_are_same_flow(arrow_a, arrow_b):
    dot = get_arrow_dot(arrow_a, arrow_b)
    return dot is not None and dot >= SAME_FLOW_DOT_THRESHOLD


def arrows_are_collision_flow(arrow_a, arrow_b):
    dot = get_arrow_dot(arrow_a, arrow_b)
    return dot is not None and dot <= ACCIDENT_CONFLICT_DOT_THRESHOLD


def is_center_stationary(track_history):
    if not track_history or len(track_history) < 2:
        return False

    usable_points = list(track_history)[-max(HEADING_WINDOW + 1, 2):]
    start_point = usable_points[0]
    end_point = usable_points[-1]
    distance = math.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1])
    return distance <= ACCIDENT_STATIONARY_DISTANCE_THRESHOLD


def is_entry_stationary(entry, min_frames):
    cached_info = entry.get("cached_info") or {}
    return cached_info.get("stationary_frames", 0) >= min_frames


def is_entry_lane_aligned(entry):
    debug_info = entry.get("debug_info") or {}
    lane_group = debug_info.get("lane_group")
    lane_overlap_info = entry.get("lane_overlap_info")
    if lane_group == LANE_UNKNOWN or lane_group is None:
        return False
    if is_lane_straddling(lane_overlap_info):
        return False
    return entry.get("arrow_segment") is not None


def get_congestion_candidate_track_ids(frame_entries):
    lane_entries = defaultdict(list)
    for entry in frame_entries:
        debug_info = entry.get("debug_info") or {}
        lane_group = debug_info.get("lane_group")
        if not is_entry_stationary(entry, CONGESTION_MIN_STATIONARY_FRAMES):
            continue
        if not is_entry_lane_aligned(entry):
            continue
        lane_entries[lane_group].append(entry)

    congestion_track_ids = set()
    for entries in lane_entries.values():
        if len(entries) < CONGESTION_MIN_VEHICLES_PER_LANE:
            continue

        for entry in entries:
            has_same_flow_neighbor = any(
                other is not entry and arrows_are_same_flow(entry.get("arrow_segment"), other.get("arrow_segment"))
                for other in entries
            )
            if has_same_flow_neighbor:
                congestion_track_ids.add(entry["track_id"])

    return congestion_track_ids


def get_bbox_overlap_metrics(xyxy_a, xyxy_b):
    ax1, ay1, ax2, ay2 = xyxy_a
    bx1, by1, bx2, by2 = xyxy_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    min_area = min(area_a, area_b)

    return {
        "intersection": inter_area,
        "iou": inter_area / union_area if union_area > 0 else 0.0,
        "min_ratio": inter_area / min_area if min_area > 0 else 0.0,
    }


def bboxes_overlap(overlap):
    return (
        overlap.get("iou", 0.0) >= BBOX_OVERLAP_IOU_THRESHOLD
        or overlap.get("min_ratio", 0.0) >= BBOX_OVERLAP_MIN_RATIO_THRESHOLD
    )


def same_flow_overlap_is_severe(overlap):
    return (
        overlap.get("iou", 0.0) >= SAME_FLOW_OVERLAP_IOU_THRESHOLD
        or overlap.get("min_ratio", 0.0) >= SAME_FLOW_OVERLAP_MIN_RATIO_THRESHOLD
    )


def detect_accident(entry, other_entries):
    """Return accident candidate details for this vehicle."""
    debug_info = entry.get("debug_info") or {}
    arrow_segment = entry.get("arrow_segment")
    if not other_entries:
        return {"candidate": False, "reason": "no_other_vehicle"}

    lane_group = debug_info.get("lane_group")
    lane_overlap_info = entry.get("lane_overlap_info")
    lane_conflict = is_lane_straddling(lane_overlap_info) or lane_group in (None, LANE_UNKNOWN)
    best_non_candidate = {"candidate": False, "reason": "no_conflict", "overlap": {}}

    for other in other_entries:
        other_debug_info = other.get("debug_info") or {}
        other_lane_group = other_debug_info.get("lane_group")
        other_lane_conflict = is_lane_straddling(other.get("lane_overlap_info")) or other_lane_group in (None, LANE_UNKNOWN)

        other_arrow = other.get("arrow_segment")
        same_lane = lane_group not in (None, LANE_UNKNOWN) and lane_group == other_debug_info.get("lane_group")
        same_flow = arrows_are_same_flow(arrow_segment, other_arrow)
        collision_flow = arrows_are_collision_flow(arrow_segment, other_arrow)
        overlap = get_bbox_overlap_metrics(entry["xyxy"], other["xyxy"])
        overlap_candidate = bboxes_overlap(overlap)
        dot = get_arrow_dot(arrow_segment, other_arrow)
        non_candidate_reason = "below_overlap_threshold" if overlap.get("intersection", 0.0) > 0 else "no_conflict"
        if overlap.get("min_ratio", 0.0) > (best_non_candidate.get("overlap") or {}).get("min_ratio", 0.0):
            best_non_candidate = {
                "candidate": False,
                "reason": non_candidate_reason,
                "other_track_id": other.get("track_id"),
                "overlap": overlap,
                "dot": dot,
                "same_lane": same_lane,
                "same_flow": same_flow,
                "collision_flow": collision_flow,
                "lane_conflict": lane_conflict or other_lane_conflict,
                "overlap_candidate": False,
            }

        if overlap_candidate:
            if same_lane and same_flow and not lane_conflict and not other_lane_conflict:
                if not same_flow_overlap_is_severe(overlap):
                    best_non_candidate = {
                        "candidate": False,
                        "reason": "same_flow_overlap_below_threshold",
                        "other_track_id": other.get("track_id"),
                        "overlap": overlap,
                        "dot": dot,
                        "same_lane": same_lane,
                        "same_flow": same_flow,
                        "collision_flow": collision_flow,
                        "lane_conflict": False,
                        "overlap_candidate": False,
                    }
                    continue
                reason = "same_flow_bbox_overlap"
            else:
                reason = "bbox_overlap"

            return {
                "candidate": True,
                "reason": reason,
                "other_track_id": other.get("track_id"),
                "overlap": overlap,
                "dot": dot,
                "same_lane": same_lane,
                "same_flow": same_flow,
                "collision_flow": collision_flow,
                "lane_conflict": lane_conflict or other_lane_conflict,
                "overlap_candidate": True,
            }

        arrow_candidate = False
        if arrow_segment is not None and other_arrow is not None:
            arrow_start, arrow_end = arrow_segment
            other_start, other_end = other_arrow
            if arrow_start is not None and arrow_end is not None and other_start is not None and other_end is not None:
                arrow_candidate = arrow_intersects_arrow(arrow_start, arrow_end, other_start, other_end)

        if not arrow_candidate:
            continue

        if same_lane and same_flow and not lane_conflict and not other_lane_conflict:
            continue
        if collision_flow or lane_conflict or other_lane_conflict or not same_lane:
            return {
                "candidate": True,
                "reason": "heading_conflict",
                "other_track_id": other.get("track_id"),
                "overlap": overlap,
                "dot": dot,
                "same_lane": same_lane,
                "same_flow": same_flow,
                "collision_flow": collision_flow,
                "lane_conflict": lane_conflict or other_lane_conflict,
                "overlap_candidate": False,
            }

    return best_non_candidate


def build_accident_detections(frame_entries):
    detections = []
    for entry in frame_entries:
        debug_info = entry.get("debug_info") or {}
        lane_group = debug_info.get("lane_group")
        direction_label = None if lane_group in (None, LANE_UNKNOWN) else int(lane_group)
        bbox = [int(round(value)) for value in entry["xyxy"]]
        detections.append(
            {
                "bbox": bbox,
                "center": debug_info.get("bbox_center") or get_bbox_center(entry["xyxy"]),
                "track_id": entry.get("track_id"),
                "direction_label": direction_label,
            }
        )
    return detections


def map_accident_events_to_entries(frame_entries, accident_events):
    event_by_entry = {}
    if not accident_events:
        return event_by_entry

    for event in accident_events:
        event_indexes = {
            int(index)
            for index in event.get("detection_indexes", [])
            if index is not None
        }
        event_track_ids = {
            int(track_id)
            for track_id in event.get("track_ids", [])
            if track_id is not None
        }
        event_bboxes = {
            tuple(int(value) for value in bbox)
            for bbox in (event.get("bbox_a"), event.get("bbox_b"))
            if bbox
        }

        for entry_index, entry in enumerate(frame_entries):
            track_id = entry.get("track_id")
            bbox = tuple(int(round(value)) for value in entry.get("xyxy", []))
            matched = entry_index in event_indexes
            matched = matched or (track_id is not None and int(track_id) in event_track_ids)
            matched = matched or (bbox and bbox in event_bboxes)
            if matched:
                event_by_entry[entry_index] = event

    return event_by_entry


def build_accident_candidate_info(event, entry):
    if event is None:
        return {"candidate": False, "reason": "not_active", "overlap": {}}

    other_track_id = None
    track_id = entry.get("track_id")
    for candidate_track_id in event.get("track_ids", []):
        if candidate_track_id is not None and candidate_track_id != track_id:
            other_track_id = candidate_track_id
            break

    return {
        "candidate": True,
        "reason": event.get("reason", "accident"),
        "other_track_id": other_track_id,
        "overlap": {
            "iou": float(event.get("iou", 0.0)),
            "min_ratio": float(event.get("min_overlap", 0.0)),
        },
        "dot": None,
        "same_lane": False,
        "same_flow": False,
        "collision_flow": False,
        "lane_conflict": False,
        "overlap_candidate": True,
        "held": bool(event.get("held", False)),
        "moving_overlap_seconds": event.get("moving_overlap_seconds", 0.0),
    }


def draw_tracking_results(
    frame,
    result,
    heading_state=None,
    track_histories=None,
    last_heading_by_track=None,
    frame_index=None,
    accident_detector=None,
    source_name="-",
    frame_time_seconds=None,
):
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

    frame_entries = []
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

            motion_mag = raw_heading_info["motion_magnitude"] if raw_heading_info is not None else None
            cached_info = last_heading_by_track.get(track_id) if last_heading_by_track is not None else None
            center_stable = False
            if cached_info is not None and "debug_info" in cached_info:
                center_stable = is_center_stable(bbox_center, cached_info["debug_info"].get("bbox_center"))

            preserve_heading = (
                cached_info is not None
                and "debug_info" in cached_info
                and center_stable
                and (motion_mag is None or motion_mag < MIN_MOTION_MAGNITUDE)
            )

            if preserve_heading:
                debug_info = {**cached_info["debug_info"], "bbox_center": bbox_center, "detection_id": index, "lane_overlap_info": lane_overlap_info}
                corrected_heading_info = debug_info.get("corrected_heading")
                if raw_heading_info is None:
                    raw_heading_info = {
                        "raw_heading": debug_info.get("raw_heading"),
                        "motion_magnitude": debug_info.get("motion_magnitude"),
                    }
            else:
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

            if cached_info is None:
                cached_info = {
                    "accident_score": 0.0,
                    "congestion_score": 0.0,
                    "stationary_frames": 0,
                    "accident_detected": False,
                    "congestion_detected": False,
                }

            if corrected_heading_info is not None and last_heading_by_track is not None:
                cached_info["debug_info"] = debug_info
                last_heading_by_track[track_id] = cached_info
            elif last_heading_by_track is not None and cached_info is not None:
                if "debug_info" in cached_info and is_center_stable(bbox_center, cached_info["debug_info"].get("bbox_center")):
                    debug_info = {
                        **cached_info["debug_info"],
                        "bbox_center": bbox_center,
                        "detection_id": index,
                        "lane_overlap_info": lane_overlap_info,
                    }
                else:
                    debug_info = {
                        "track_id": track_id,
                        "detection_id": index,
                        "bbox_center": bbox_center,
                        "lane_group": lane_group,
                        "lane_group_name": LANE_LABELS_KO.get(lane_group, "unknown"),
                        "lane_overlap_info": lane_overlap_info,
                        "reference_direction": reference_direction,
                        "candidate_infos": candidate_infos,
                        "primary_info": None,
                        "raw_heading": raw_heading_info["raw_heading"] if raw_heading_info else None,
                        "motion_magnitude": raw_heading_info["motion_magnitude"] if raw_heading_info else None,
                        "corrected_heading": None,
                        "dot_product": None,
                        "flipped": False,
                    }

            motion_mag = None
            if raw_heading_info is not None:
                motion_mag = raw_heading_info.get("motion_magnitude", 0.0)

            if motion_mag is None or motion_mag < MIN_MOTION_MAGNITUDE:
                cached_info["stationary_frames"] = cached_info.get("stationary_frames", 0) + 1
            else:
                cached_info["stationary_frames"] = 0

            arrow_segment = None
            if debug_info is not None and debug_info.get("corrected_heading") is not None:
                start_point = debug_info.get("bbox_center")
                ch = debug_info.get("corrected_heading")
                arrow_segment = (
                    start_point,
                    (start_point[0] + ch[0] * ARROW_LENGTH_PIXELS, start_point[1] + ch[1] * ARROW_LENGTH_PIXELS),
                )

            frame_entries.append(
                {
                    "detection_index": index,
                    "track_id": track_id,
                    "xyxy": xyxy,
                    "debug_info": debug_info,
                    "cached_info": cached_info,
                    "lane_overlap_info": lane_overlap_info,
                    "arrow_segment": arrow_segment,
                }
            )
        except Exception as error:
            log_heading_status(f"track_id={track_id} heading 계산 실패, 해당 객체는 건너뜁니다: {error}")

    congestion_track_ids = get_congestion_candidate_track_ids(frame_entries)
    accident_events = []
    if accident_detector is not None:
        accident_timestamp = ""
        if frame_time_seconds is not None:
            accident_timestamp = f"{frame_time_seconds:.3f}s"
        accident_events = accident_detector.update(
            build_accident_detections(frame_entries),
            frame_id=frame_index or 0,
            cctv_name=source_name,
            timestamp=accident_timestamp,
            now=frame_time_seconds,
        )
    accident_event_by_entry = map_accident_events_to_entries(frame_entries, accident_events)

    for entry_index, entry in enumerate(frame_entries):
        debug_info = entry["debug_info"]
        cached_info = entry["cached_info"]
        lane_overlap_info = entry["lane_overlap_info"]
        if debug_info is None:
            debug_info = {"bbox_center": get_bbox_center(entry["xyxy"])}
            entry["debug_info"] = debug_info

        accident_event = accident_event_by_entry.get(entry_index)
        accident_candidate_info = build_accident_candidate_info(accident_event, entry)
        accident_candidate = accident_event is not None
        congestion_candidate = entry["track_id"] in congestion_track_ids and not accident_candidate
        previous_accident = cached_info.get("accident_detected", False)
        previous_congestion = cached_info.get("congestion_detected", False)

        lane_conflict = is_lane_straddling(lane_overlap_info)
        if debug_info.get("lane_group") in (None, LANE_UNKNOWN):
            lane_conflict = True

        if congestion_candidate:
            cached_info["congestion_score"] = min(
                MAX_CONGESTION_SCORE,
                cached_info.get("congestion_score", 0.0) + CONGESTION_SCORE_INC,
            )
        else:
            cached_info["congestion_score"] = max(
                0.0,
                cached_info.get("congestion_score", 0.0) - CONGESTION_SCORE_DEC,
            )

        congestion_now = cached_info.get("congestion_score", 0.0) >= CONGESTION_ON_THRESHOLD
        if cached_info.get("congestion_score", 0.0) <= CONGESTION_OFF_THRESHOLD:
            congestion_now = False

        if accident_candidate:
            cached_info["accident_score"] = max(
                cached_info.get("accident_score", 0.0),
                float(accident_event.get("score", ACCIDENT_ON_THRESHOLD)),
            )
        else:
            cached_info["accident_score"] = max(0.0, cached_info.get("accident_score", 0.0) - ACCIDENT_SCORE_DEC)

        accident_now = accident_candidate

        if accident_now:
            congestion_now = False

        cached_info["accident_detected"] = accident_now
        cached_info["congestion_detected"] = congestion_now
        if last_heading_by_track is not None:
            last_heading_by_track[entry["track_id"]] = cached_info

        state_changed = previous_accident != accident_now or previous_congestion != congestion_now
        periodic_log = frame_index is not None and frame_index % DEBUG_LOG_EVERY_FRAMES == 0
        debug_info["accident_detected"] = accident_now
        debug_info["congestion_detected"] = congestion_now
        debug_info["accident_candidate_info"] = accident_candidate_info
        debug_info["congestion_candidate"] = congestion_candidate
        debug_info["lane_conflict"] = lane_conflict
        debug_info["stationary_frames"] = cached_info.get("stationary_frames", 0)
        debug_info["accident_score"] = cached_info.get("accident_score", 0.0)
        debug_info["congestion_score"] = cached_info.get("congestion_score", 0.0)
        debug_info["frame_index"] = frame_index
        debug_info["should_log"] = accident_candidate or congestion_candidate or state_changed or periodic_log

        if debug_info is not None and (
            debug_info.get("corrected_heading") is not None
            or debug_info.get("accident_detected")
            or debug_info.get("congestion_detected")
            or debug_info.get("should_log")
        ):
            log_vehicle_heading_debug(debug_info)
            draw_debug_overlay(annotated, entry["xyxy"], debug_info)

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


def reset_loop_tracking_state(model, track_histories, last_heading_by_track):
    track_histories.clear()
    last_heading_by_track.clear()

    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None)
    if trackers:
        for tracker in trackers:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    if predictor is not None and hasattr(predictor, "vid_path"):
        predictor.vid_path = [None] * len(getattr(predictor, "vid_path", []))


def build_video_playlist(requested_video):
    video_paths = find_saved_videos()

    if requested_video:
        requested_path = resolve_video_path(requested_video)
        requested_key = requested_path.resolve(strict=False)
        for video_index, video_path in enumerate(video_paths):
            if video_path.resolve(strict=False) == requested_key:
                return video_paths, video_index
        return [requested_path], 0

    if not video_paths:
        return [resolve_video_path("")], 0

    return video_paths, len(video_paths) - 1


def open_playlist_capture(video_paths, video_index, model, track_histories, last_heading_by_track, accident_detector):
    video_path = video_paths[video_index]
    capture = open_video_capture(video_path)
    delay_ms = get_video_delay_ms(capture)
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    if video_fps is None or video_fps <= 0:
        video_fps = 1000.0 / max(delay_ms, 1)

    reset_loop_tracking_state(model, track_histories, last_heading_by_track)
    accident_detector.reset()
    print(f"비디오 입력 [{video_index + 1}/{len(video_paths)}]: {video_path}", flush=True)
    print("단축키: a=이전 비디오, d=다음 비디오, q/Esc=종료", flush=True)
    return video_path, capture, delay_ms, video_fps


def switch_playlist_video(
    capture,
    video_paths,
    video_index,
    step,
    model,
    track_histories,
    last_heading_by_track,
    accident_detector,
):
    if len(video_paths) <= 1:
        print("전환할 다른 비디오가 없습니다.", flush=True)
        return video_index, capture, None

    previous_path = video_paths[video_index]
    capture.release()
    video_index = (video_index + step) % len(video_paths)
    video_path, capture, delay_ms, video_fps = open_playlist_capture(
        video_paths,
        video_index,
        model,
        track_histories,
        last_heading_by_track,
        accident_detector,
    )
    print(f"비디오 전환: {previous_path.name} -> {video_path.name}", flush=True)
    return video_index, capture, (video_path, delay_ms, video_fps)


def parse_args():
    parser = argparse.ArgumentParser(description="Run crash/congestion detection on a saved video")
    parser.add_argument("--video", default="", help="입력 비디오 경로; 생략하면 saved_videos의 최신 파일")
    parser.add_argument("--no-window", action="store_true", help="OpenCV 화면 없이 터미널 로그만 출력")
    parser.add_argument("--max-frames", type=int, default=0, help="처리할 최대 프레임 수; 0이면 무제한")
    return parser.parse_args()


def main():
    args = parse_args()
    log_heading_status("crash/congestion detection 시작")
    model = load_model()
    heading_state = build_heading_state()
    accident_detector = AccidentLogDetector()
    track_histories = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LENGTH))
    last_heading_by_track = {}
    video_paths, video_index = build_video_playlist(args.video)
    video_path, capture, delay_ms, video_fps = open_playlist_capture(
        video_paths,
        video_index,
        model,
        track_histories,
        last_heading_by_track,
        accident_detector,
    )
    frame_index = 0
    total_processed_frames = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            print("비디오가 끝났습니다. 처음부터 다시 재생합니다.", flush=True)
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            reset_loop_tracking_state(model, track_histories, last_heading_by_track)
            accident_detector.reset()
            frame_index = 0
            continue

        frame_index += 1
        total_processed_frames += 1
        frame_time_seconds = frame_index / video_fps

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
            frame_index=frame_index,
            accident_detector=accident_detector,
            source_name=video_path.name,
            frame_time_seconds=frame_time_seconds,
        )

        if not args.no_window:
            cv2.imshow("Crash Detect", annotated_frame)

            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key in (ord("a"), ord("d")):
                step = -1 if key == ord("a") else 1
                video_index, capture, switched = switch_playlist_video(
                    capture,
                    video_paths,
                    video_index,
                    step,
                    model,
                    track_histories,
                    last_heading_by_track,
                    accident_detector,
                )
                if switched is not None:
                    video_path, delay_ms, video_fps = switched
                    frame_index = 0
                continue

        if args.max_frames > 0 and total_processed_frames >= args.max_frames:
            print(f"max_frames={args.max_frames} 처리 완료. 종료합니다.", flush=True)
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
