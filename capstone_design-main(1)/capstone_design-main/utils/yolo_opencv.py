import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

# ultralytics가 홈 디렉터리에 설정 파일을 만들지 못하는 환경 대응
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# 학습에 사용한 워크스페이스 내 Ultralytics 구현을 우선 사용합니다.
VENDORED_ULTRALYTICS_ROOT = Path(__file__).resolve().parent / "yolov12" / "yolov12"
if VENDORED_ULTRALYTICS_ROOT.exists():
    sys.path.insert(0, str(VENDORED_ULTRALYTICS_ROOT))

import cv2
import pandas as pd
import requests
import torch
from ultralytics import YOLO

API_KEY = "f742615880e442b4a0e549771bbef99d"
API_URL = "https://openapi.its.go.kr:9443/cctvInfo"
MODEL_PATH = "../best.pt"
DIRECT_CCTV_URL = ""
DIRECT_CCTV_NAME = ""

# 청주 인근 고속도로 CCTV를 조회합니다.
# 보고 싶은 화면으로 바꾸려면 아래 CCTV_INDEX 값을 수정하세요.
TARGET_NAME = "청주"
CCTV_INDEX = 0
VEHICLE_CLASSES = [0]
CONFIDENCE_THRESHOLD = 0.3
LIST_LIMIT = 20
FRAME_WAIT_TIMEOUT_SECONDS = 10
FRAME_WAIT_INTERVAL_SECONDS = 0.2

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cctv-index",
        type=int,
        default=CCTV_INDEX,
        help=f"선택할 CCTV 인덱스입니다. 기본값: {CCTV_INDEX}",
    )
    return parser.parse_args()


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
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일을 찾지 못했습니다: {MODEL_PATH}")

    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = patched_torch_load
        model = YOLO(MODEL_PATH)
    finally:
        torch.load = original_torch_load

    print("YOLO model:", MODEL_PATH)
    print("tracking classes:", VEHICLE_CLASSES)
    return model


def draw_tracking_results(frame, result):
    annotated = frame.copy()
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return annotated

    names = result.names
    xyxy_list = boxes.xyxy.int().cpu().tolist()
    cls_list = boxes.cls.int().cpu().tolist()
    conf_list = boxes.conf.cpu().tolist()

    for xyxy, cls_id, conf in zip(xyxy_list, cls_list, conf_list):
        x1, y1, x2, y2 = xyxy
        color = (0, 255, 0)
        text = f"{conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            annotated,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def main():
    global CCTV_INDEX

    args = parse_args()
    CCTV_INDEX = args.cctv_index

    model = load_model()
    if DIRECT_CCTV_URL:
        cctv_name = DIRECT_CCTV_NAME or "직접 지정 CCTV"
        cctv_url = DIRECT_CCTV_URL
    else:
        cctv_df = fetch_cctv_dataframe()
        cctv_name, cctv_url = select_cctv(cctv_df)

    print("선택 CCTV:", cctv_name)
    print("cctvurl:", cctv_url)

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
        annotated_frame = draw_tracking_results(frame, results[0])

        cv2.imshow("ITS CCTV Live", annotated_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
