from pathlib import Path
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
# 조회된 목록의 0번부터 19번까지 순서대로 저장합니다.
START_INDEX = 0
END_INDEX = 19
OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "data" / "videos" / "weather")
WINDOW_NAME = "ITS CCTV Video Save"
FRAME_WAIT_TIMEOUT_SECONDS = 10
FRAME_WAIT_INTERVAL_SECONDS = 0.2
OUTPUT_FPS = 20.0
CLIP_DURATION_SECONDS = 90
OUTPUT_EXTENSION = "avi"
VIDEO_CODECS = ("MJPG", "XVID")

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


def list_cctvs(cctv_df):
    print("조회된 CCTV 목록:")
    for index, row in cctv_df.head(END_INDEX + 1).iterrows():
        print(f"[{index}] {row['cctvname']}")


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


def log_frame_dimensions(frame):
    height, width = frame.shape[:2]
    print(f"raw frame size (H, W): ({height}, {width})")


def create_video_writer(frame, output_dir, cctv_index):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    height, width = frame.shape[:2]
    print(f"writer frame size (H, W): ({height}, {width})")

    for codec in VIDEO_CODECS:
        filename = f"{cctv_index}_{timestamp}.{OUTPUT_EXTENSION}"
        filepath = os.path.join(output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(filepath, fourcc, OUTPUT_FPS, (width, height))
        if writer.isOpened():
            print(f"video codec: {codec}, format: .{OUTPUT_EXTENSION}")
            return writer, filepath
        writer.release()

    raise RuntimeError(
        "영상 저장기 생성 실패: "
        f"{OUTPUT_EXTENSION}, codecs={', '.join(VIDEO_CODECS)}"
    )


def save_cctv_clip(cctv_index, cctv_name, cctv_url):
    print(f"\n[{cctv_index}] 저장 시작: {cctv_name}")
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

    writer = None
    output_path = None
    clip_started_at = None

    try:
        while capture.isOpened():
            ok, frame = read_frame_with_timeout(capture, FRAME_WAIT_TIMEOUT_SECONDS)
            if not ok:
                print(
                    f"[프레임 수신 불가] {FRAME_WAIT_TIMEOUT_SECONDS}초 동안 기다렸지만 "
                    "프레임이 오지 않아 종료합니다"
                )
                break

            if writer is None:
                log_frame_dimensions(frame)
                writer, output_path = create_video_writer(frame, OUTPUT_DIR, cctv_index)
                clip_started_at = time.time()
                print("saving video:", output_path)

            writer.write(frame)
            cv2.imshow(WINDOW_NAME, frame)

            if time.time() - clip_started_at >= CLIP_DURATION_SECONDS:
                print(
                    f"[{cctv_index}] {CLIP_DURATION_SECONDS}초 저장 완료, 다음 CCTV로 넘어갑니다."
                )
                break

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
            print("saved:", output_path)
        cv2.destroyAllWindows()


def main():
    ensure_output_dir(OUTPUT_DIR)

    cctv_df = fetch_cctv_dataframe()
    if len(cctv_df) <= END_INDEX:
        raise RuntimeError(
            f"조회된 CCTV 수가 부족합니다. 요청 인덱스: 0~{END_INDEX}, "
            f"실제 개수: {len(cctv_df)}"
        )

    list_cctvs(cctv_df)

    for cctv_index in range(START_INDEX, END_INDEX + 1):
        row = cctv_df.iloc[cctv_index]
        cctv_name = row["cctvname"]
        cctv_url = row["cctvurl"]

        try:
            save_cctv_clip(cctv_index, cctv_name, cctv_url)
        except Exception as error:
            print(f"[{cctv_index}] 저장 실패: {error}")
            print(f"[{cctv_index}] 다음 CCTV로 넘어갑니다.")


if __name__ == "__main__":
    main()
