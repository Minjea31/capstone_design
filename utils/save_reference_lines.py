import argparse
import json
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

import cv2
import numpy as np
import pandas as pd
import requests

WINDOW_NAME = "Save Heading Reference Groups"
MODULE_DIR = Path(__file__).resolve().parent
API_KEY = "f742615880e442b4a0e549771bbef99d"
API_URL = "https://openapi.its.go.kr:9443/cctvInfo"
TARGET_NAME = "청주"
CCTV_INDEX = 0
DIRECT_CCTV_NAME = ""
DIRECT_CCTV_URL = ""
LIST_LIMIT = 20
FRAME_WAIT_TIMEOUT_SECONDS = 10
FRAME_WAIT_INTERVAL_SECONDS = 0.2
POINT_SELECT_RADIUS = 18
GROUP_COLORS = [
    (0, 255, 255),
    (255, 255, 0),
    (0, 165, 255),
    (255, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
]
ROI_LABEL_COLORS = {
    0: (0, 255, 0),
    1: (0, 165, 255),
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

STATE = {
    "display_frame": None,
    "roi_points": [],
    "groups": [],
    "current_group": [],
}


def fetch_cctv_dataframe():
    query = "&".join(f"{key}={value}" for key, value in REQUEST_PARAMS.items())
    url = f"{API_URL}?{query}"
    response = urlopen(url)
    payload = json.loads(response.read().decode("utf-8"))

    data = payload.get("response", {}).get("data", [])
    if not data:
        raise RuntimeError("CCTV 목록을 불러오지 못했습니다.")

    return pd.json_normalize(data)


def select_cctv(cctv_df, selected_cctv_index):
    print("조회된 CCTV 목록:")
    for index, row in cctv_df.head(LIST_LIMIT).iterrows():
        print(f"[{index}] {row['cctvname']}")

    if selected_cctv_index is not None:
        if selected_cctv_index < 0 or selected_cctv_index >= len(cctv_df):
            raise RuntimeError(f"CCTV_INDEX 범위가 잘못되었습니다: {selected_cctv_index}")

        row = cctv_df.iloc[selected_cctv_index]
        return row["cctvname"], row["cctvurl"], int(selected_cctv_index)

    matched = cctv_df[cctv_df["cctvname"].astype(str).str.contains(TARGET_NAME, na=False)]
    if matched.empty:
        names = cctv_df["cctvname"].astype(str).dropna().tolist()
        sample_names = ", ".join(names[:10]) if names else "없음"
        raise RuntimeError(
            f"이름으로 CCTV를 찾지 못했습니다: {TARGET_NAME}\n"
            f"현재 조회된 CCTV 예시: {sample_names}"
        )

    row = matched.iloc[0]
    return row["cctvname"], row["cctvurl"], int(row.name)


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


def get_first_frame_from_direct_source(source):
    source_path = Path(source)
    if source_path.exists() and source_path.is_file():
        image = cv2.imread(str(source_path))
        if image is not None:
            return image, None, source

    capture_source = source
    if isinstance(source, str) and source.isdigit():
        capture_source = int(source)

    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"입력을 열 수 없습니다: {source}")

    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise RuntimeError(f"첫 프레임을 읽을 수 없습니다: {source}")

    return frame, capture, str(source)


def get_first_frame_from_cctv_api(selected_cctv_index):
    if DIRECT_CCTV_URL:
        cctv_name = DIRECT_CCTV_NAME or "직접 지정 CCTV"
        cctv_url = DIRECT_CCTV_URL
        resolved_index = selected_cctv_index if selected_cctv_index is not None else CCTV_INDEX
    else:
        cctv_df = fetch_cctv_dataframe()
        cctv_name, cctv_url, resolved_index = select_cctv(cctv_df, selected_cctv_index)

    print("선택 CCTV:", cctv_name)
    print("cctvurl:", cctv_url)

    stream_url = resolve_stream_url(cctv_url)
    print("stream url:", stream_url)

    capture = connect_capture(stream_url, cctv_url)
    if capture is None:
        raise RuntimeError("OpenCV에서 CCTV 스트림을 열지 못했습니다.")

    ok, frame = read_frame_with_timeout(capture, FRAME_WAIT_TIMEOUT_SECONDS)
    if not ok or frame is None:
        capture.release()
        raise RuntimeError("CCTV 첫 프레임을 읽지 못했습니다.")

    return frame, capture, cctv_name, resolved_index


def resolve_roi_json_path(requested_path, save_index):
    if requested_path:
        return Path(requested_path)
    return MODULE_DIR / "main" / "roi" / f"roi_data_{int(save_index)}.json"


def resolve_output_json_path(requested_path, save_index):
    if requested_path:
        return Path(requested_path)
    return MODULE_DIR / f"heading_{int(save_index)}.json"


def load_roi_points(roi_json_path):
    if not roi_json_path.exists():
        raise FileNotFoundError(f"ROI JSON 파일을 찾지 못했습니다: {roi_json_path}")

    with roi_json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    roi_points = []
    point_id = 0
    for roi_index, item in enumerate(payload):
        if not isinstance(item, list) or len(item) != 2:
            continue
        points_data, label = item
        if not isinstance(points_data, list):
            continue

        normalized_points = []
        for point_index, point in enumerate(points_data):
            if not isinstance(point, list) or len(point) != 2:
                continue
            x = int(point[0])
            y = int(point[1])
            normalized_points.append((x, y))
            roi_points.append(
                {
                    "id": point_id,
                    "roi_index": int(roi_index),
                    "roi_label": int(label),
                    "point_index": int(point_index),
                    "point": (x, y),
                }
            )
            point_id += 1

    if not roi_points:
        raise RuntimeError(f"선택 가능한 ROI 점이 없습니다: {roi_json_path}")

    return roi_points


def build_save_payload(index, name, roi_json_path, groups):
    serializable_groups = []
    for group_index, group in enumerate(groups):
        points = [[int(item["point"][0]), int(item["point"][1])] for item in group]
        selected_ids = [int(item["id"]) for item in group]
        roi_indexes = [int(item["roi_index"]) for item in group]
        roi_labels = [int(item["roi_label"]) for item in group]
        serializable_groups.append(
            {
                "group_id": int(group_index),
                "point_ids": selected_ids,
                "points": points,
                "roi_indexes": roi_indexes,
                "roi_labels": roi_labels,
            }
        )

    return {
        "index": int(index),
        "name": name or f"CCTV {index}",
        "source_roi_json": str(roi_json_path),
        "groups": serializable_groups,
    }


def save_group_json(output_json_path, index, name, roi_json_path, groups):
    payload = build_save_payload(index, name, roi_json_path, groups)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def get_nearest_roi_point(x, y, roi_points, excluded_ids=None):
    excluded_ids = excluded_ids or set()
    best_item = None
    best_distance_sq = None

    for item in roi_points:
        if item["id"] in excluded_ids:
            continue
        px, py = item["point"]
        distance_sq = (px - x) ** 2 + (py - y) ** 2
        if best_distance_sq is None or distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_item = item

    if best_item is None:
        return None

    if best_distance_sq is None or best_distance_sq > POINT_SELECT_RADIUS * POINT_SELECT_RADIUS:
        return None

    return best_item


def draw_group(overlay, group, color, thickness):
    if len(group) >= 2:
        cv2.polylines(
            overlay,
            [np.array([item["point"] for item in group], dtype=np.int32)],
            False,
            color,
            thickness,
        )

    for item in group:
        x, y = item["point"]
        cv2.circle(overlay, (x, y), 7, color, -1)
        cv2.circle(overlay, (x, y), 10, (255, 255, 255), 1)


def draw_overlay(frame, roi_points, groups, current_group):
    overlay = frame.copy()

    points_by_roi = {}
    for item in roi_points:
        points_by_roi.setdefault(item["roi_index"], []).append(item)

    for items in points_by_roi.values():
        polygon = np.array([item["point"] for item in items], dtype=np.int32)
        roi_label = items[0]["roi_label"]
        color = ROI_LABEL_COLORS.get(roi_label, (255, 255, 255))
        if len(items) >= 2:
            cv2.polylines(overlay, [polygon], True, color, 1)

    selected_ids = {item["id"] for group in groups for item in group}
    selected_ids.update(item["id"] for item in current_group)

    for item in roi_points:
        x, y = item["point"]
        roi_label = item["roi_label"]
        point_color = ROI_LABEL_COLORS.get(roi_label, (220, 220, 220))
        radius = 5 if item["id"] not in selected_ids else 3
        cv2.circle(overlay, (x, y), radius, point_color, -1)

    for group_index, group in enumerate(groups):
        color = GROUP_COLORS[group_index % len(GROUP_COLORS)]
        draw_group(overlay, group, color, 2)

    active_group_index = len(groups)
    active_color = GROUP_COLORS[active_group_index % len(GROUP_COLORS)]
    draw_group(overlay, current_group, active_color, 3)

    return overlay


def mouse_callback(event, x, y, flags, param):
    del flags, param
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    current_ids = {item["id"] for item in STATE["current_group"]}
    nearest_point = get_nearest_roi_point(x, y, STATE["roi_points"], excluded_ids=current_ids)
    if nearest_point is None:
        print(f"선택 가능한 ROI 점이 없습니다. 클릭 위치=({x}, {y})")
        return

    STATE["current_group"].append(nearest_point)
    print(
        "선택 점:",
        f"id={nearest_point['id']}",
        f"roi={nearest_point['roi_index']}",
        f"label={nearest_point['roi_label']}",
        f"point_index={nearest_point['point_index']}",
        f"point={nearest_point['point']}",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Select ROI points and save grouped heading reference JSON")
    parser.add_argument("--source", default="", help="image/video/stream/camera source; 비우면 CCTV API 사용")
    parser.add_argument("--roi-json", default="", help="입력 ROI JSON 경로; 비우면 main/roi/roi_data_<index>.json")
    parser.add_argument("--output-json", default="", help="출력 JSON 경로; 비우면 heading_<index>.json")
    parser.add_argument("--index", type=int, default=None, help="저장할 CCTV index; 기본값은 API 선택 index")
    parser.add_argument("--name", default="", help="저장할 CCTV name; 기본값은 API 선택 name")
    parser.add_argument("--cctv-index", type=int, default=None, help="API에서 선택할 CCTV index; 예: --cctv-index 1")
    return parser.parse_args()


def print_usage(index, name, source_label, roi_json_path, output_json_path, roi_points):
    print("reference line 작업을 시작합니다. 입력은 CCTV/API 또는 --source에서 가져온 첫 프레임입니다.")
    print(f"index: {index}")
    print(f"name: {name or f'CCTV {index}'}")
    print(f"source: {source_label}")
    print(f"ROI JSON: {roi_json_path}")
    print(f"저장 경로: {output_json_path}")
    print(f"선택 가능한 ROI 점 개수: {len(roi_points)}")
    print("사용법:")
    print("  좌클릭 : 가까운 ROI 점 선택")
    print("  n      : 현재 선택한 점들을 reference line 그룹으로 확정")
    print("  u      : 현재 그룹에서 마지막 점 취소")
    print("  d      : 마지막으로 확정한 그룹 삭제")
    print("  r      : 모든 그룹/현재 선택 초기화")
    print(f"  s      : {output_json_path} 저장")
    print("  q/Esc  : 종료")
    print("권장 순서: 하행 line 1 -> 하행 line 2 -> 상행 line 1 -> 상행 line 2")
    print("점 클릭 방향: 하행은 위에서 아래, 상행은 아래에서 위")


def main():
    args = parse_args()

    if args.source:
        frame, capture, source_label = get_first_frame_from_direct_source(args.source)
        save_index = args.index if args.index is not None else 0
        save_name = args.name or f"CCTV {save_index}"
    else:
        selected_cctv_index = args.cctv_index if args.cctv_index is not None else CCTV_INDEX
        frame, capture, selected_name, selected_index = get_first_frame_from_cctv_api(selected_cctv_index)
        save_index = args.index if args.index is not None else selected_index
        save_name = args.name or selected_name
        source_label = f"API CCTV {selected_name}"

    roi_json_path = resolve_roi_json_path(args.roi_json, save_index)
    output_json_path = resolve_output_json_path(args.output_json, save_index)

    try:
        STATE["display_frame"] = frame.copy()
        STATE["roi_points"] = load_roi_points(roi_json_path)
        STATE["groups"] = []
        STATE["current_group"] = []

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
        print_usage(save_index, save_name, source_label, roi_json_path, output_json_path, STATE["roi_points"])

        while True:
            overlay = draw_overlay(
                STATE["display_frame"],
                STATE["roi_points"],
                STATE["groups"],
                STATE["current_group"],
            )
            cv2.imshow(WINDOW_NAME, overlay)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("u"):
                if STATE["current_group"]:
                    removed = STATE["current_group"].pop()
                    print(f"현재 그룹에서 점 제거: id={removed['id']} point={removed['point']}")
            elif key == ord("n"):
                if len(STATE["current_group"]) < 2:
                    print("그룹을 확정하려면 현재 그룹에 최소 2개 점이 필요합니다.")
                    continue
                STATE["groups"].append(list(STATE["current_group"]))
                print(f"그룹 확정 완료: group_id={len(STATE['groups']) - 1}, points={len(STATE['current_group'])}")
                STATE["current_group"] = []
            elif key == ord("d"):
                if STATE["groups"]:
                    removed_group = STATE["groups"].pop()
                    print(f"마지막 저장 그룹 삭제: points={len(removed_group)}")
            elif key == ord("r"):
                STATE["groups"] = []
                STATE["current_group"] = []
                print("모든 그룹과 현재 선택을 초기화했습니다.")
            elif key == ord("s"):
                groups_to_save = list(STATE["groups"])
                if len(STATE["current_group"]) >= 2:
                    groups_to_save.append(list(STATE["current_group"]))

                if not groups_to_save:
                    print("저장하려면 최소 한 개 그룹에 2개 이상 점이 필요합니다.")
                    continue

                save_group_json(output_json_path, save_index, save_name, roi_json_path, groups_to_save)
                print(f"저장 완료: {output_json_path} (groups={len(groups_to_save)})")
            elif key == ord("q") or key == 27:
                break
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
