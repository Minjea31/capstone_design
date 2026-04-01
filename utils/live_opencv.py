import json
import os
import threading
import time
from datetime import datetime
from urllib.parse import urljoin
from urllib.request import urlopen

import cv2
import numpy as np
import pandas as pd
import requests

API_KEY = "f742615880e442b4a0e549771bbef99d"
API_URL = "https://openapi.its.go.kr:9443/cctvInfo"

# 청주 인근 고속도로 CCTV를 조회합니다.
# 보고 싶은 화면으로 바꾸려면 아래 CCTV_INDEX 값을 수정하세요.
TARGET_NAME = "청주"
CCTV_INDEX = 0
DIRECT_CCTV_NAME = ""
DIRECT_CCTV_URL = ""
VEHICLE_CLASSES = [0]
CONFIDENCE_THRESHOLD = 0.6
LIST_LIMIT = 20
FRAME_WAIT_TIMEOUT_SECONDS = 10
FRAME_WAIT_INTERVAL_SECONDS = 0.2
PLAYLIST_POLL_INTERVAL_SECONDS = 2.0

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

# OpenCV FFmpeg가 지원하는 경우 저지연 옵션을 우선 시도합니다.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "fflags;nobuffer|flags;low_delay|reorder_queue_size;0",
)


def get_cctv_url(lat, lng):
    # CCTV 탐색 범위 지정을 위해 임의로 ±1 만큼 가감
    min_x = str(lng - 1)
    max_x = str(lng + 1)
    min_y = str(lat - 1)
    max_y = str(lat + 1)

    api_call = (
        f"{API_URL}?"
        f"apiKey={API_KEY}"
        "&type=ex&cctvType=2"
        f"&minX={min_x}"
        f"&maxX={max_x}"
        f"&minY={min_y}"
        f"&maxY={max_y}"
        "&getType=json"
    )

    w_dataset = requests.get(api_call, timeout=10).json()
    cctv_data = w_dataset.get("response", {}).get("data", [])
    if not cctv_data:
        raise RuntimeError("해당 좌표 주변 CCTV를 찾지 못했습니다.")

    coordx_list = []
    for data in cctv_data:
        xy_couple = (float(data["coordy"]), float(data["coordx"]))
        coordx_list.append(xy_couple)

    # 입력한 위경도 좌표에서 가장 가까운 위치에 있는 CCTV를 찾는 과정
    coordx_array = np.array(coordx_list)
    target_point = np.array((lat, lng))
    distances = np.linalg.norm(coordx_array - target_point, axis=1)
    min_index = np.argmin(distances)

    return cctv_data[min_index]


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
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        return capture

    capture.release()
    capture = cv2.VideoCapture(url)
    if capture.isOpened():
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        return capture

    capture.release()
    return None


class LatestFrameReader:
    def __init__(self, capture):
        self.capture = capture
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._latest_frame = None
        self._latest_received_at = None
        self._read_error = False
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def _reader_loop(self):
        while not self._stop_event.is_set():
            ok, frame = self.capture.read()
            if not ok:
                self._read_error = True
                time.sleep(FRAME_WAIT_INTERVAL_SECONDS)
                continue

            received_at = datetime.now().astimezone()
            with self._lock:
                self._latest_frame = frame
                self._latest_received_at = received_at
                self._read_error = False

    def read_latest(self, timeout_seconds):
        deadline = time.time() + timeout_seconds

        while time.time() < deadline:
            with self._lock:
                if self._latest_frame is not None:
                    return True, self._latest_frame.copy(), self._latest_received_at

            if self._read_error:
                time.sleep(FRAME_WAIT_INTERVAL_SECONDS)
            else:
                time.sleep(0.01)

        return False, None, None


class PlaylistTimestampTracker:
    def __init__(self, playlist_url):
        self.playlist_url = playlist_url
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._latest_program_time = None
        self._last_updated_at = None
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                response = requests.get(self.playlist_url, timeout=5)
                response.raise_for_status()

                latest_program_time = None
                for line in response.text.splitlines():
                    line = line.strip()
                    if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
                        iso_value = line.split(":", 1)[1].strip()
                        latest_program_time = parse_program_datetime(iso_value)

                if latest_program_time is not None:
                    with self._lock:
                        self._latest_program_time = latest_program_time
                        self._last_updated_at = datetime.now().astimezone()
            except requests.RequestException:
                pass
            except ValueError:
                pass

            self._stop_event.wait(PLAYLIST_POLL_INTERVAL_SECONDS)

    def get_latest_program_time(self):
        with self._lock:
            if self._latest_program_time is None or self._last_updated_at is None:
                return None

            # 플레이리스트의 최신 세그먼트 시각을 마지막 갱신 이후 경과 시간만큼 보정합니다.
            elapsed = datetime.now().astimezone() - self._last_updated_at
            return self._latest_program_time + elapsed


def parse_program_datetime(value):
    if "." not in value:
        return datetime.fromisoformat(value)

    prefix, suffix = value.split(".", 1)
    fraction_part = suffix
    timezone_part = ""

    for sign in ("+", "-"):
        sign_index = suffix.find(sign)
        if sign_index > 0:
            fraction_part = suffix[:sign_index]
            timezone_part = suffix[sign_index:]
            break

    normalized_fraction = (fraction_part + "000000")[:6]
    normalized_value = f"{prefix}.{normalized_fraction}{timezone_part}"
    return datetime.fromisoformat(normalized_value)


def format_timestamp(timestamp):
    local_time = timestamp.astimezone()
    return local_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def draw_text_with_outline(frame, text, origin, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2

    cv2.putText(frame, text, origin, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, origin, font, font_scale, color, thickness, cv2.LINE_AA)


def log_gpu_status():
    if not hasattr(cv2, "cuda"):
        print("OpenCV CUDA 모듈이 없어 CPU 디코딩으로 실행합니다.")
        return

    gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
    if gpu_count <= 0:
        print("사용 가능한 CUDA GPU가 없어 CPU 디코딩으로 실행합니다.")
        return

    print(f"CUDA GPU {gpu_count}개 감지됨")


def main():
    log_gpu_status()

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

    frame_reader = LatestFrameReader(capture)
    timestamp_tracker = PlaylistTimestampTracker(stream_url)
    frame_reader.start()
    timestamp_tracker.start()

    try:
        while capture.isOpened():
            ok, frame, _ = frame_reader.read_latest(FRAME_WAIT_TIMEOUT_SECONDS)
            if not ok:
                print(f"[프레임 수신 불가] {FRAME_WAIT_TIMEOUT_SECONDS}초 동안 기다렸지만 최신 프레임이 오지 않아 종료합니다")
                break

            cctv_time = timestamp_tracker.get_latest_program_time()
            displayed_at = datetime.now().astimezone()

            if cctv_time is not None:
                draw_text_with_outline(frame, f"cctv_time: {format_timestamp(cctv_time)}", (20, 35))
                latency_ms = max(0, int((displayed_at - cctv_time.astimezone()).total_seconds() * 1000))
                draw_text_with_outline(frame, f"latency_ms: {latency_ms}", (20, 65))
            else:
                draw_text_with_outline(frame, "cctv_time: loading...", (20, 35))

            draw_text_with_outline(frame, f"displayed_at: {format_timestamp(displayed_at)}", (20, 95))

            cv2.imshow("ITS CCTV Live", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        frame_reader.stop()
        timestamp_tracker.stop()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
