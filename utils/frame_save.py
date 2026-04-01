import json
import os
import time
from datetime import datetime
from urllib.parse import urljoin
from urllib.request import urlopen

import cv2
import pandas as pd
import requests

API_KEY = "f742615880e442b4a0e549771bbef99d"
API_URL = "https://openapi.its.go.kr:9443/cctvInfo"

# 청주 인근 고속도로 CCTV를 조회합니다.
# 보고 싶은 화면으로 바꾸려면 아래 CCTV_INDEX 값을 수정하세요.
TARGET_NAME = "청주"
CCTV_INDEX = 10
DIRECT_CCTV_NAME = ""
DIRECT_CCTV_URL = ""
LIST_LIMIT = 20
OUTPUT_DIR = "saved_frames"
SAVE_INTERVAL_SECONDS = 1
WINDOW_NAME = "ITS CCTV Frame Save"
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


def read_frame_with_timeout(capture, timeout_seconds):
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        ok, frame = capture.read()
        if ok:
            return True, frame
        time.sleep(FRAME_WAIT_INTERVAL_SECONDS)

    return False, None


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def save_frame(frame, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"frame_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    ok = cv2.imwrite(filepath, frame)
    if not ok:
        raise RuntimeError(f"프레임 저장 실패: {filepath}")
    return filepath


def main():
    ensure_output_dir(OUTPUT_DIR)

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

    capture = open_capture(stream_url)
    if capture is None:
        print("실제 스트림 URL 연결 실패, 원본 cctvurl로 다시 시도합니다.")
        capture = open_capture(cctv_url)

    if capture is None:
        raise RuntimeError("OpenCV에서 CCTV 스트림을 열지 못했습니다.")

    next_save_at = time.time() + SAVE_INTERVAL_SECONDS

    while capture.isOpened():
        ok, frame = read_frame_with_timeout(capture, FRAME_WAIT_TIMEOUT_SECONDS)
        if not ok:
            print(
                f"[프레임 수신 불가] {FRAME_WAIT_TIMEOUT_SECONDS}초 동안 기다렸지만 "
                "프레임이 오지 않아 종료합니다"
            )
            break

        cv2.imshow(WINDOW_NAME, frame)

        now = time.time()
        if now >= next_save_at:
            filepath = save_frame(frame, OUTPUT_DIR)
            print("saved:", filepath)
            while next_save_at <= now:
                next_save_at += SAVE_INTERVAL_SECONDS

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
