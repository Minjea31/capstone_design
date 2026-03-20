"""Flask 기반 CCTV 탐지 웹 UI."""

import atexit
from collections import deque
import sys
import threading
import time
from urllib.parse import urljoin

import cv2
import numpy as np
try:
    from flask import Flask, Response, abort, jsonify, render_template, request, url_for
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Flask가 현재 실행 중인 파이썬 환경에 설치되어 있지 않습니다.\n"
        f"현재 인터프리터: {sys.executable}\n"
        "IDE 인터프리터를 /home/user/anaconda3/envs/capstone/bin/python 으로 바꾸거나,\n"
        "해당 인터프리터에 Flask를 설치한 뒤 다시 실행하세요."
    ) from exc

from yolo_opencv import (
    CONFIDENCE_THRESHOLD,
    VEHICLE_CLASSES,
    draw_tracking_results,
    fetch_cctv_dataframe,
    load_model,
    open_capture,
    read_frame_with_timeout,
    resolve_stream_url,
)
import requests
from werkzeug.serving import WSGIRequestHandler

app = Flask(__name__)
STREAM_LOOP_DELAY_SECONDS = 0.03
WEB_FRAME_TIMEOUT_SECONDS = 3.0


class QuietRequestHandler(WSGIRequestHandler):
    def log_request(self, code="-", size="-"):
        if self.path.startswith("/api/status") or self.path.startswith("/video_feed"):
            return
        super().log_request(code, size)


class VideoService:
    def __init__(self):
        try:
            self.model = load_model()
            self.yolo_enabled = True
        except FileNotFoundError as exc:
            print(f"경고: {exc}")
            self.model = None
            self.yolo_enabled = False

        self.cctv_df = fetch_cctv_dataframe()
        self.cctv_names = self.cctv_df["cctvname"].astype(str).fillna("").tolist()
        self.lock = threading.Lock()
        self.capture = None
        self.current_stream_url = None
        self.selected_index = 0
        self.selected_name = "-"
        self.traffic_count = 0
        self.traffic_up = 0
        self.traffic_down = 0
        self.accident_probability = 0
        self.accident_status = "-"
        self.stream_status = "준비 중"
        self.traffic_history_limit = 36
        self.traffic_up_history = deque([0], maxlen=self.traffic_history_limit)
        self.traffic_down_history = deque([0], maxlen=self.traffic_history_limit)
        self.accident_probability_history = deque([0], maxlen=self.traffic_history_limit)
        self.last_history_update = 0.0
        self.last_frame = self._build_placeholder("loading...")
        self.last_jpeg = self._encode_jpeg(self.last_frame)
        self.stop_event = threading.Event()
        self._open_first_available_stream()
        self.reader_thread = threading.Thread(target=self._frame_loop, daemon=True)
        self.reader_thread.start()

    def _build_placeholder(self, message):
        frame = np.full((540, 960, 3), (18, 24, 34), dtype=np.uint8)
        cv2.putText(frame, "Detection Screen", (28, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 236, 245), 2, cv2.LINE_AA)
        cv2.putText(frame, message, (28, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (120, 182, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (22, 22), (938, 518), (62, 78, 104), 2)
        return frame

    def _encode_jpeg(self, frame):
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        return encoded.tobytes()

    def _open_capture_for_index(self, index):
        row = self.cctv_df.iloc[index]
        name = str(row["cctvname"])
        url = row["cctvurl"]

        stream_url = resolve_stream_url(url)
        capture = open_capture(stream_url)
        if capture is None:
            capture = open_capture(url)
            stream_url = url
        if capture is None:
            raise RuntimeError(f"CCTV 스트림을 열 수 없습니다: {name}")
        return capture, name, stream_url

    def _open_first_available_stream(self):
        last_error = None
        for index in range(len(self.cctv_df)):
            try:
                capture, name, stream_url = self._open_capture_for_index(index)
                self.capture = capture
                self.current_stream_url = stream_url
                self.selected_index = index
                self.selected_name = name
                self.stream_status = "연결됨"
                return
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError(f"열 수 있는 CCTV 스트림이 없습니다: {last_error}")

    def _reopen_selected_stream_locked(self):
        capture, name, stream_url = self._open_capture_for_index(self.selected_index)
        if self.capture is not None:
            self.capture.release()
        self.capture = capture
        self.current_stream_url = stream_url
        self.selected_name = name
        self.stream_status = "재연결됨"

    def _set_frame_locked(self, frame, traffic_count=None, status=None):
        self.last_frame = frame
        encoded = self._encode_jpeg(frame)
        if encoded is not None:
            self.last_jpeg = encoded
        if traffic_count is not None:
            self.traffic_count = traffic_count
            self._update_traffic_metrics_locked(traffic_count)
        if status is not None:
            self.stream_status = status

    def _update_traffic_metrics_locked(self, traffic_count):
        if traffic_count <= 0:
            self.traffic_up = 0
            self.traffic_down = 0
            self.accident_probability = 0
        else:
            wave = ((int(time.time()) + self.selected_index) % 5) - 2
            split_ratio = min(0.7, max(0.3, 0.5 + (wave * 0.05)))
            self.traffic_up = int(round(traffic_count * split_ratio))
            self.traffic_down = max(0, traffic_count - self.traffic_up)
            imbalance = abs(self.traffic_up - self.traffic_down)
            self.accident_probability = int(min(99, 12 + (traffic_count * 8) + (imbalance * 3)))

        self.accident_status = f"{self.accident_probability}%"

        now = time.time()
        if now - self.last_history_update >= 1.0:
            self.traffic_up_history.append(self.traffic_up)
            self.traffic_down_history.append(self.traffic_down)
            self.accident_probability_history.append(self.accident_probability)
            self.last_history_update = now

    def select_index(self, index):
        if index < 0 or index >= len(self.cctv_names):
            raise IndexError("CCTV 인덱스 범위를 벗어났습니다.")

        with self.lock:
            capture, name, stream_url = self._open_capture_for_index(index)
            if self.capture is not None:
                self.capture.release()
            self.capture = capture
            self.current_stream_url = stream_url
            self.selected_index = index
            self.selected_name = name
            self.traffic_count = 0
            self.traffic_up = 0
            self.traffic_down = 0
            self.accident_probability = 0
            self.accident_status = "-"
            self.traffic_up_history = deque([0], maxlen=self.traffic_history_limit)
            self.traffic_down_history = deque([0], maxlen=self.traffic_history_limit)
            self.accident_probability_history = deque([0], maxlen=self.traffic_history_limit)
            self.last_history_update = 0.0
            self.stream_status = "연결됨"

    def get_status(self):
        with self.lock:
            player_url = url_for("video_playlist")
            if self.current_stream_url and ".m3u8" not in self.current_stream_url:
                player_url = url_for("video_proxy", target=self.current_stream_url)
            return {
                "selected_index": self.selected_index,
                "selected_name": self.selected_name,
                "traffic_count": self.traffic_count,
                "traffic_up": self.traffic_up,
                "traffic_down": self.traffic_down,
                "traffic_up_history": list(self.traffic_up_history),
                "traffic_down_history": list(self.traffic_down_history),
                "accident_probability": self.accident_probability,
                "accident_probability_history": list(self.accident_probability_history),
                "accident_status": self.accident_status,
                "stream_status": self.stream_status,
                "player_url": player_url,
                "cctv_count": len(self.cctv_names),
                "yolo_enabled": self.yolo_enabled,
            }

    def get_cctv_items(self):
        return [
            {
                "index": index,
                "name": name,
                "selected": index == self.selected_index,
            }
            for index, name in enumerate(self.cctv_names)
        ]

    def _read_processed_frame(self):
        with self.lock:
            capture = self.capture
            if capture is None or not capture.isOpened():
                self.stream_status = "스트림 없음"
                self._set_frame_locked(self._build_placeholder("stream not connected"), traffic_count=0, status="스트림 없음")
                return self.last_frame

            ok, frame = read_frame_with_timeout(capture, WEB_FRAME_TIMEOUT_SECONDS)
            if not ok:
                try:
                    self._reopen_selected_stream_locked()
                    capture = self.capture
                    ok, frame = read_frame_with_timeout(capture, max(1.0, WEB_FRAME_TIMEOUT_SECONDS / 2))
                except RuntimeError:
                    ok = False

                if not ok:
                    self._set_frame_locked(self._build_placeholder("frame timeout"), traffic_count=0, status="프레임 수신 실패")
                    return self.last_frame

            if self.model is not None:
                results = self.model.track(
                    frame,
                    persist=True,
                    classes=VEHICLE_CLASSES,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                )
                annotated = draw_tracking_results(frame, results[0])
                traffic_count = len(results[0].boxes) if results and results[0].boxes is not None else 0
            else:
                annotated = frame
                traffic_count = 0

            self._set_frame_locked(annotated, traffic_count=traffic_count, status="연결됨")
            return annotated

    def _frame_loop(self):
        while not self.stop_event.is_set():
            try:
                self._read_processed_frame()
            except Exception as exc:
                with self.lock:
                    self.accident_status = "stream error"
                    self._set_frame_locked(self._build_placeholder("stream error"), traffic_count=0, status=str(exc)[:60])
            time.sleep(STREAM_LOOP_DELAY_SECONDS)

    def get_jpeg_bytes(self):
        with self.lock:
            return self.last_jpeg

    def release(self):
        self.stop_event.set()
        if hasattr(self, "reader_thread") and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0)
        with self.lock:
            if self.capture is not None:
                self.capture.release()
                self.capture = None


service = VideoService()
atexit.register(service.release)


@app.route("/")
def index():
    return render_template("index.html", cctvs=service.get_cctv_items(), status=service.get_status())


@app.route("/api/status")
def api_status():
    return jsonify(service.get_status())


@app.route("/api/cctvs")
def api_cctvs():
    return jsonify({"items": service.get_cctv_items()})


@app.route("/api/select/<int:index>", methods=["POST"])
def api_select(index):
    try:
        service.select_index(index)
    except (RuntimeError, IndexError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "status": service.get_status()})


def mjpeg_stream():
    while True:
        jpg = service.get_jpeg_bytes()
        if jpg is None:
            time.sleep(0.1)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


def _rewrite_hls_playlist(playlist_text, base_url):
    rewritten_lines = []
    for line in playlist_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            rewritten_lines.append(line)
            continue

        absolute_url = urljoin(base_url, stripped)
        proxied_url = url_for("video_proxy", target=absolute_url)
        rewritten_lines.append(proxied_url)

    return "\n".join(rewritten_lines)


@app.route("/video_playlist.m3u8")
def video_playlist():
    with service.lock:
        stream_url = service.current_stream_url

    if not stream_url:
        abort(404)

    try:
        response = requests.get(stream_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        return Response(f"#EXTM3U\n# stream error: {exc}\n", mimetype="application/vnd.apple.mpegurl", status=502)

    playlist_body = _rewrite_hls_playlist(response.text, response.url)
    return Response(playlist_body, mimetype="application/vnd.apple.mpegurl")


@app.route("/video_proxy")
def video_proxy():
    target = request.args.get("target", type=str)
    if not target or not target.startswith(("http://", "https://")):
        abort(400)

    try:
        upstream = requests.get(target, stream=True, timeout=10)
        upstream.raise_for_status()
    except requests.RequestException as exc:
        return Response(f"proxy error: {exc}", status=502)

    content_type = upstream.headers.get("Content-Type", "")
    if ".m3u8" in target or "mpegurl" in content_type.lower():
        try:
            playlist_text = upstream.text
        finally:
            upstream.close()
        playlist_body = _rewrite_hls_playlist(playlist_text, upstream.url)
        return Response(playlist_body, mimetype="application/vnd.apple.mpegurl")

    def generate():
        try:
            for chunk in upstream.iter_content(chunk_size=64 * 1024):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    return Response(generate(), content_type=content_type or "application/octet-stream")


def main():
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, request_handler=QuietRequestHandler)


if __name__ == "__main__":
    main()
