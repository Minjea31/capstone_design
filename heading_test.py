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
ROI_DIRECTION_LABELS_KO = {
    0: "상행",
    1: "하행",
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


def load_heading_rois(cctv_index):
    roi_path = resolve_roi_path(cctv_index)
    if not roi_path.exists():
        log_heading_status(f"ROI 파일이 없어 ROI 기반 방향 라벨을 건너뜁니다: {roi_path}")
        return []

    try:
        with roi_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception as error:
        log_heading_status(f"ROI 파일 로드 실패로 ROI 기반 방향 라벨을 건너뜁니다: {error}")
        return []

    rois = []
    for item in payload:
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
            rois.append((points, int(label)))

    if rois:
        log_heading_status(f"ROI {len(rois)}개를 로드했습니다: {roi_path}")
    return rois


def resolve_heading_group_json_path(cctv_index):
    current_path = Path(HEADING_GROUP_JSON_PATH)
    if current_path.exists():
        return current_path
    return MODULE_DIR / f"heading_{int(cctv_index)}.json"


def load_heading_groups(cctv_index):
    json_path = resolve_heading_group_json_path(cctv_index)
    if not json_path.exists():
        log_heading_status(f"heading group json 파일이 없습니다: {json_path}")
        return []

    try:
        with json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception as error:
        log_heading_status(f"heading group json 로드 실패: {error}")
        return []

    groups = payload.get("groups", []) if isinstance(payload, dict) else []
    if not isinstance(groups, list):
        return []

    normalized_groups = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_id = group.get("group_id")
        points = group.get("points", [])
        if group_id is None or not isinstance(points, list) or len(points) < 2:
            continue

        normalized_points = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                continue
            normalized_points.append((float(point[0]), float(point[1])))

        if len(normalized_points) < 2:
            continue

        normalized_groups.append(
            {
                "group_id": int(group_id),
                "points": normalized_points,
            }
        )

    if normalized_groups:
        log_heading_status(f"heading group {len(normalized_groups)}개를 로드했습니다: {json_path}")
    return normalized_groups


def get_heading_label_from_group_id(group_id):
    if group_id in (0, 1):
        return 1
    if group_id in (2, 3):
        return 0
    return None


def get_roi_heading_label(point, rois):
    if not rois:
        return None

    px, py = point
    for points, label in rois:
        polygon = np.array(points, dtype=np.int32)
        if cv2.pointPolygonTest(polygon, (float(px), float(py)), False) >= 0:
            return int(label)

    return None


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


def build_heading_group_state(groups):
    segments = []
    group_paths = []

    for group in groups:
        group_id = int(group["group_id"])
        direction_label = get_heading_label_from_group_id(group_id)
        if direction_label is None:
            continue

        points = [(int(round(x)), int(round(y))) for x, y in group["points"]]
        if len(points) < 2:
            continue

        group_paths.append(
            {
                "group_id": group_id,
                "label": direction_label,
                "points": points,
            }
        )

        for start_point, end_point in zip(points[:-1], points[1:]):
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            direction = normalize_vector((dx, dy))
            if direction is None:
                continue
            segments.append(
                {
                    "group_id": group_id,
                    "label": direction_label,
                    "start": start_point,
                    "end": end_point,
                    "dir": direction,
                    "length_sq": dx * dx + dy * dy,
                }
            )

    if not segments:
        raise ValueError("heading group segment를 만들 수 없습니다.")

    return {
        "mode": "heading_groups",
        "group_paths": group_paths,
        "segments": segments,
    }


def get_bottom_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, float(y2))


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


def get_local_road_direction(point, road_state):
    best_segment = None
    best_distance_sq = None
    px, py = point

    for segment in road_state["segments"]:
        sx, sy = segment["start"]
        ex, ey = segment["end"]
        seg_dx = ex - sx
        seg_dy = ey - sy
        length_sq = segment["length_sq"]
        projection = ((px - sx) * seg_dx + (py - sy) * seg_dy) / length_sq
        projection = max(0.0, min(1.0, projection))
        closest_x = sx + projection * seg_dx
        closest_y = sy + projection * seg_dy
        distance_sq = (px - closest_x) ** 2 + (py - closest_y) ** 2
        if best_distance_sq is None or distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_segment = segment

    if best_segment is None:
        return None
    return best_segment["dir"]


def get_local_heading_label(point, road_state):
    best_segment = None
    best_distance_sq = None
    px, py = point

    for segment in road_state["segments"]:
        sx, sy = segment["start"]
        ex, ey = segment["end"]
        seg_dx = ex - sx
        seg_dy = ey - sy
        length_sq = segment["length_sq"]
        projection = ((px - sx) * seg_dx + (py - sy) * seg_dy) / length_sq
        projection = max(0.0, min(1.0, projection))
        closest_x = sx + projection * seg_dx
        closest_y = sy + projection * seg_dy
        distance_sq = (px - closest_x) ** 2 + (py - closest_y) ** 2
        if best_distance_sq is None or distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_segment = segment

    return best_segment["label"] if best_segment is not None else None


def compute_heading_against_road(motion_vec, road_dir):
    motion_magnitude = math.hypot(motion_vec[0], motion_vec[1])
    if motion_magnitude < MIN_MOTION_MAGNITUDE:
        return None

    normal = (-road_dir[1], road_dir[0])
    parallel = motion_vec[0] * road_dir[0] + motion_vec[1] * road_dir[1]
    perpendicular = motion_vec[0] * normal[0] + motion_vec[1] * normal[1]
    heading_angle = math.degrees(math.atan2(perpendicular, parallel))

    motion_dir = (motion_vec[0] / motion_magnitude, motion_vec[1] / motion_magnitude)
    return {
        "parallel": parallel,
        "perpendicular": perpendicular,
        "heading_angle": heading_angle,
        "motion_dir": motion_dir,
        "motion_magnitude": motion_magnitude,
        "road_dir": road_dir,
    }


def draw_reference_lines(frame, road_state):
    for group_path in road_state["group_paths"]:
        color = (0, 165, 255) if group_path["label"] == 1 else (0, 255, 0)
        cv2.polylines(frame, [np.array(group_path["points"], dtype=np.int32)], False, color, 2)
        start_point = tuple(group_path["points"][0])
        cv2.putText(
            frame,
            f"group_{group_path['group_id']}",
            start_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_heading_info(frame, track_id, xyxy, heading_info, heading_label=None):
    del track_id, heading_label
    bottom_center = get_bottom_center(xyxy)
    start_point = (int(round(bottom_center[0])), int(round(bottom_center[1])))

    motion_end = (
        int(round(start_point[0] + heading_info["motion_dir"][0] * 40)),
        int(round(start_point[1] + heading_info["motion_dir"][1] * 40)),
    )

    cv2.arrowedLine(frame, start_point, motion_end, (0, 0, 255), 2, tipLength=0.25)
    cv2.circle(frame, start_point, 4, (255, 255, 255), -1)


def draw_tracking_results(frame, result, heading_state=None, track_histories=None):
    annotated = frame.copy()
    boxes = result.boxes

    if heading_state and heading_state.get("enabled"):
        draw_reference_lines(annotated, heading_state["road_state"])

    if boxes is None or len(boxes) == 0:
        return annotated

    xyxy_list = boxes.xyxy.cpu().tolist()
    conf_list = boxes.conf.cpu().tolist()
    track_ids = None
    if getattr(boxes, "id", None) is not None:
        track_ids = boxes.id.int().cpu().tolist()

    for index, (xyxy, conf) in enumerate(zip(xyxy_list, conf_list)):
        x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
        bbox_center = get_bbox_center(xyxy)
        roi_heading_label = get_roi_heading_label(bbox_center, heading_state.get("heading_rois", []))
        if roi_heading_label is None:
            continue

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
            if motion_vec is None:
                continue

            local_road_dir = get_local_road_direction(bbox_center, heading_state["road_state"])
            if local_road_dir is None:
                continue

            heading_label = roi_heading_label
            heading_info = compute_heading_against_road(motion_vec, local_road_dir)
            if heading_info is None:
                continue

            draw_heading_info(annotated, track_id, xyxy, heading_info, heading_label=heading_label)
        except Exception as error:
            log_heading_status(f"track_id={track_id} heading 계산 실패, 해당 객체는 건너뜁니다: {error}")

    return annotated


def build_heading_state(cctv_index):
    if not ENABLE_HEADING:
        log_heading_status("ENABLE_HEADING=False 이므로 heading 기능을 비활성화합니다.")
        return {"enabled": False, "reason": "disabled_by_config"}

    heading_groups = load_heading_groups(cctv_index)
    if not heading_groups:
        return {"enabled": False, "reason": "missing_heading_groups"}
    heading_rois = load_heading_rois(cctv_index)

    try:
        road_state = build_heading_group_state(heading_groups)

        log_heading_status(
            f"index={cctv_index} heading 활성화 완료 [heading_groups_json]"
        )
        return {
            "enabled": True,
            "road_state": road_state,
            "heading_rois": heading_rois,
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
        history.append(get_bottom_center(xyxy))


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
        )

        cv2.imshow("ITS CCTV Live", annotated_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
