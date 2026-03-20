import json
import os
import time
from urllib.parse import urljoin
from urllib.request import urlopen

# ultralytics가 홈 디렉터리에 설정 파일을 만들지 못하는 환경 대응
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import pandas as pd
import requests
import torch
try:
    from ultralytics import YOLO
    ULTRALYTICS_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc

API_KEY = "f742615880e442b4a0e549771bbef99d"
API_URL = "https://openapi.its.go.kr:9443/cctvInfo"
MODEL_PATH = "yolo12n.pt"
FALLBACK_MODEL_PATHS = [
    "/home/a/yolov8_prune/ultralytics/yolov8n.pt",
]

# 다른 구간을 보고 싶으면 TARGET_NAME을 바꾸거나 CCTV_INDEX를 지정하면 됩니다.
TARGET_NAME = "성남"
CCTV_INDEX = None
VEHICLE_CLASSES = [2]
CONFIDENCE_THRESHOLD = 0.3
LIST_LIMIT = 20
FRAME_WAIT_TIMEOUT_SECONDS = 10
FRAME_WAIT_INTERVAL_SECONDS = 0.2

REQUEST_PARAMS = {
    "apiKey": API_KEY,
    "type": "all",
    "cctvType": "1",
    "minX": 127.0,
    "maxX": 127.3,
    "minY": 37.3,
    "maxY": 37.55,
    "getType": "json",
}

NETWORK_OPEN_TIMEOUT_MSEC = 5000
NETWORK_READ_TIMEOUT_MSEC = 5000


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
    try:
        response = requests.get(cctv_url, allow_redirects=True, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"스트림 URL 해석 실패, 원본 URL로 시도합니다: {exc}")
        return cctv_url

    playlist_url = response.url
    playlist_text = response.text

    for line in playlist_text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return urljoin(playlist_url, line)

    return cctv_url


def open_capture(url):
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if capture is not None:
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, NETWORK_OPEN_TIMEOUT_MSEC)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, NETWORK_READ_TIMEOUT_MSEC)
    if capture.isOpened():
        return capture

    capture.release()
    capture = cv2.VideoCapture(url)
    if capture is not None:
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, NETWORK_OPEN_TIMEOUT_MSEC)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, NETWORK_READ_TIMEOUT_MSEC)
    if capture.isOpened():
        return capture

    capture.release()
    return None


def read_frame_with_timeout(capture, timeout_seconds):
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        ok, frame = capture.read()
        if ok:
            return True, frame

        time.sleep(FRAME_WAIT_INTERVAL_SECONDS)

    return False, None


def load_model():
    if YOLO is None:
        raise FileNotFoundError(
            "ultralytics 패키지가 설치되어 있지 않아 탐지 모델을 비활성화합니다. "
            f"원인: {ULTRALYTICS_IMPORT_ERROR}"
        )

    candidate_paths = [MODEL_PATH] + FALLBACK_MODEL_PATHS
    existing_paths = [path for path in candidate_paths if os.path.exists(path)]

    if not existing_paths:
        fallback_text = ", ".join(FALLBACK_MODEL_PATHS)
        raise FileNotFoundError(
            f"모델 파일을 찾지 못했습니다: {MODEL_PATH}\n"
            f"확인한 폴백 경로: {fallback_text}"
        )

    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    last_error = None

    try:
        torch.load = patched_torch_load

        for model_path in existing_paths:
            try:
                model = YOLO(model_path)
                print("YOLO model:", model_path)
                print("tracking classes:", VEHICLE_CLASSES)
                return model
            except Exception as exc:
                last_error = exc
                print(f"모델 로딩 실패: {model_path}")
                print(f"원인: {exc}")

                if "C3k2" in str(exc):
                    print(
                        "현재 ultralytics 버전이 이 체크포인트 구조를 지원하지 않습니다. "
                        "호환 가능한 모델로 재시도합니다."
                    )

    finally:
        torch.load = original_torch_load

    raise RuntimeError(f"사용 가능한 모델을 로드하지 못했습니다: {last_error}")


def draw_tracking_results(frame, result):
    annotated = frame.copy()
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return annotated

    names = result.names
    xyxy_list = boxes.xyxy.int().cpu().tolist()
    cls_list = boxes.cls.int().cpu().tolist()
    conf_list = boxes.conf.cpu().tolist()

    if boxes.id is not None:
        track_id_list = boxes.id.int().cpu().tolist()
    else:
        track_id_list = [None] * len(xyxy_list)

    for xyxy, cls_id, conf, track_id in zip(xyxy_list, cls_list, conf_list, track_id_list):
        x1, y1, x2, y2 = xyxy
        color = (0, 255, 0)
        if isinstance(names, dict):
            label = names.get(cls_id, str(cls_id))
        elif 0 <= cls_id < len(names):
            label = names[cls_id]
        else:
            label = str(cls_id)

        if track_id is not None:
            text = f"{label} ID:{track_id} {conf:.2f}"
        else:
            text = f"{label} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated


def main():
    model = load_model()
    cctv_df = fetch_cctv_dataframe()
    cctv_name, cctv_url = select_cctv(cctv_df)

    print("선택 CCTV:", cctv_name)
    print("cctvurl:", cctv_url)

    stream_url = resolve_stream_url(cctv_url)
    print("stream url:", stream_url)

    capture = open_capture(stream_url)

    if capture is None:
        print("실제 스트림 URL 연결 실패, 원본 cctvurl로 다시 시도합니다.")
        capture = open_capture(cctv_url)

    if capture is None:
        raise RuntimeError("OpenCV에서 CCTV 스트림을 열지 못했습니다.")

    while capture.isOpened():
        ok, frame = read_frame_with_timeout(capture, FRAME_WAIT_TIMEOUT_SECONDS)
        if not ok:
            print(f"[프레임 수신 불가] {FRAME_WAIT_TIMEOUT_SECONDS}초 동안 기다렸지만 프레임이 오지 않아 종료합니다")
            break

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
