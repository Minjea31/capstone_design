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
import yaml

WINDOW_NAME = "Save Reference Path"
API_KEY = "f742615880e442b4a0e549771bbef99d"
API_URL = "https://openapi.its.go.kr:9443/cctvInfo"
TARGET_NAME = "청주"
CCTV_INDEX = 2
DIRECT_CCTV_NAME = ""
DIRECT_CCTV_URL = ""
LIST_LIMIT = 20
FRAME_WAIT_TIMEOUT_SECONDS = 10
FRAME_WAIT_INTERVAL_SECONDS = 0.2
LANE_COLORS = [
    (0, 255, 255),
    (255, 255, 0),
    (0, 165, 255),
    (255, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
]

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
    "lanes": [[]],
    "display_frame": None,
}


def load_yaml_config(yaml_path):
    config_path = Path(yaml_path)
    if not config_path.exists():
        return {"cctvs": []}

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        return {"cctvs": []}

    if not isinstance(data.get("cctvs"), list):
        data["cctvs"] = []
    return data


def save_yaml_config(yaml_path, config):
    config_path = Path(yaml_path)
    if not config_path.parent.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


def normalize_lane_paths(lane_paths):
    normalized = []
    for lane in lane_paths:
        if len(lane) >= 2:
            normalized.append([[int(x), int(y)] for x, y in lane])
    return normalized


def upsert_cctv_entry(config, index, name, lane_paths):
    normalized_lane_paths = normalize_lane_paths(lane_paths)
    entry = {
        "index": index,
        "name": name or f"CCTV {index}",
        "reference_paths": {
            "lane_paths": normalized_lane_paths,
        },
    }

    if len(normalized_lane_paths) == 1:
        entry["reference_path"] = {"center_path": normalized_lane_paths[0]}

    cctvs = config.setdefault("cctvs", [])
    for position, current in enumerate(cctvs):
        if isinstance(current, dict) and current.get("index") == index:
            cctvs[position] = entry
            return config

    cctvs.append(entry)
    return config


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


def mouse_callback(event, x, y, flags, param):
    del flags, param
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    STATE["lanes"][-1].append((int(x), int(y)))


def draw_lane(overlay, lane_points, lane_index, active_lane_index):
    color = LANE_COLORS[lane_index % len(LANE_COLORS)]
    thickness = 3 if lane_index == active_lane_index else 2
    if len(lane_points) >= 2:
        cv2.polylines(overlay, [np.array(lane_points, dtype=np.int32)], False, color, thickness)
    for point_index, point in enumerate(lane_points, start=1):
        cv2.circle(overlay, point, 5, color, -1)
        cv2.putText(
            overlay,
            f"{lane_index + 1}:{point_index}",
            (point[0] + 6, point[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_overlay(frame, lanes, index, name, source_label, yaml_path):
    overlay = frame.copy()
    active_lane_index = len(lanes) - 1

    saved_lane_count = sum(1 for lane in lanes if len(lane) >= 2)
    for lane_index, lane_points in enumerate(lanes):
        draw_lane(overlay, lane_points, lane_index, active_lane_index)

    current_lane_points = len(lanes[-1]) if lanes else 0
    guide_lines = [
        f"index: {index}",
        f"name: {name or f'CCTV {index}'}",
        f"source: {source_label}",
        f"yaml: {yaml_path}",
        f"lanes: {saved_lane_count} | active lane: {active_lane_index + 1} | points: {current_lane_points}",
        "Left click: add point to current lane",
        "n: next lane | u: undo point | d: delete lane | r: reset all | s: save | q: quit",
    ]

    for row_index, text in enumerate(guide_lines):
        cv2.putText(
            overlay,
            text,
            (20, 30 + row_index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay


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


def resolve_output_yaml_path(requested_yaml_path, save_index):
    if requested_yaml_path:
        return requested_yaml_path
    return f"{save_index}.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Save CCTV lane paths to YAML")
    parser.add_argument("--source", default="", help="image/video/stream/camera source; 비우면 CCTV API 사용")
    parser.add_argument("--yaml", default="", help="output YAML path; 비우면 index.yaml 자동 생성")
    parser.add_argument("--index", type=int, default=None, help="YAML에 저장할 CCTV index; 기본값은 API 선택 index")
    parser.add_argument("--name", default="", help="YAML에 저장할 CCTV name; 기본값은 API 선택 name")
    parser.add_argument("--cctv-index", type=int, default=None, help="API에서 선택할 CCTV index; 예: --cctv-index 1")
    return parser.parse_args()


def ensure_current_lane_exists():
    if not STATE["lanes"]:
        STATE["lanes"] = [[]]


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

    output_yaml_path = resolve_output_yaml_path(args.yaml, save_index)

    try:
        STATE["lanes"] = [[]]
        STATE["display_frame"] = frame.copy()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

        while True:
            ensure_current_lane_exists()
            overlay = draw_overlay(
                STATE["display_frame"],
                STATE["lanes"],
                save_index,
                save_name,
                source_label,
                output_yaml_path,
            )
            cv2.imshow(WINDOW_NAME, overlay)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("u"):
                if STATE["lanes"] and STATE["lanes"][-1]:
                    STATE["lanes"][-1].pop()
            elif key == ord("n"):
                if len(STATE["lanes"][-1]) >= 2:
                    STATE["lanes"].append([])
                else:
                    print("다음 차선으로 넘어가려면 현재 차선에 최소 2개 점이 필요합니다.")
            elif key == ord("d"):
                if STATE["lanes"]:
                    STATE["lanes"].pop()
                ensure_current_lane_exists()
            elif key == ord("r"):
                STATE["lanes"] = [[]]
            elif key == ord("s"):
                lane_paths = normalize_lane_paths(STATE["lanes"])
                if not lane_paths:
                    print("저장하려면 최소 한 개 차선 path에 2개 이상 점이 필요합니다.")
                    continue

                config = load_yaml_config(output_yaml_path)
                config = upsert_cctv_entry(config, save_index, save_name, lane_paths)
                save_yaml_config(output_yaml_path, config)
                print(f"저장 완료: {output_yaml_path} (index={save_index}, name={save_name}, lanes={len(lane_paths)})")
            elif key == ord("q") or key == 27:
                break
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
