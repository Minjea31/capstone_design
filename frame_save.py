import os
import time
from datetime import datetime
from urllib.parse import urljoin

import cv2
import requests

DIRECT_CCTV_NAME = "[국도 3호선] 성남 대원IC"
DIRECT_CCTV_URL = (
    "https://cctvsec.ktict.co.kr/4297/"
    "j5V7v7TE3R2TZcaHilS1x+9Bikk0MUc0VUpuz0IeKWaGii0I6A9Jt+nUQm61PQY5"
    "pBj7m63lZungyjLPluodVGEe5J4y+oUQDmm3gisEWk8="
)
OUTPUT_DIR = "saved_frames"
SAVE_INTERVAL_SECONDS = 1.0
FRAME_WAIT_TIMEOUT_SECONDS = 10
FRAME_WAIT_INTERVAL_SECONDS = 0.2
WINDOW_NAME = "ITS CCTV Save"


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

    print("선택 CCTV:", DIRECT_CCTV_NAME)
    print("cctvurl:", DIRECT_CCTV_URL)

    stream_url = resolve_stream_url(DIRECT_CCTV_URL)
    print("stream url:", stream_url)

    capture = open_capture(stream_url)
    if capture is None:
        print("실제 스트림 URL 연결 실패, 원본 cctvurl로 다시 시도합니다.")
        capture = open_capture(DIRECT_CCTV_URL)

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
