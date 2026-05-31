import base64
import json
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import cv2
import requests

from utils.app_config import (
    DETECTION_FRAME_WIDTH,
    DETECTION_JPEG_QUALITY,
    DETECTION_REQUEST_TIMEOUT_SECONDS,
    DETECTION_WS_PATH,
    DETECTION_WS_RECONNECT_MAX_SECONDS,
    DETECTION_WS_RECONNECT_MIN_SECONDS,
)

try:
    import websocket
except ImportError:
    websocket = None

_DETECTION_THREAD_LOCAL = threading.local()


def get_detection_session():
    session = getattr(_DETECTION_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _DETECTION_THREAD_LOCAL.session = session
    return session


def derive_backend_url_from_ws_url(ws_url):
    if not ws_url:
        return ""

    parsed = urlparse(ws_url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        return ""

    scheme = "https" if parsed.scheme == "wss" else "http"
    return f"{scheme}://{parsed.netloc}"


def derive_ws_url_from_backend_url(backend_url):
    if not backend_url:
        return ""

    parsed = urlparse(backend_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""

    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = DETECTION_WS_PATH if DETECTION_WS_PATH.startswith("/") else f"/{DETECTION_WS_PATH}"
    return f"{scheme}://{parsed.netloc}{path}"


def build_detection_payload(
    frame,
    detections,
    traffic_count,
    selected_name,
    selected_index,
    stream_status,
    roi_enabled,
    roi_path,
):
    if DETECTION_FRAME_WIDTH > 0 and frame.shape[1] > DETECTION_FRAME_WIDTH:
        scale = DETECTION_FRAME_WIDTH / float(frame.shape[1])
        frame = cv2.resize(
            frame,
            (DETECTION_FRAME_WIDTH, int(frame.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )

    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), DETECTION_JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("JPEG 인코딩 실패")

    encoded_bytes = encoded.tobytes()
    best_detection = max(
        detections,
        key=lambda item: float(item.get("confidence", 0) or 0),
        default={"class_name": "none", "confidence": 0.0, "bbox": [0, 0, 0, 0]},
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    data = {
        "class_name": best_detection["class_name"],
        "confidence": str(best_detection["confidence"]),
        "bbox": json.dumps(best_detection["bbox"]),
        "detections": json.dumps(detections, ensure_ascii=False),
        "traffic_count": str(traffic_count),
        "selected_name": selected_name,
        "selected_index": str(selected_index),
        "stream_status": stream_status,
        "roi_enabled": str(bool(roi_enabled)).lower(),
        "roi_path": str(roi_path or ""),
        "timestamp": timestamp,
    }
    websocket_data = {
        **data,
        "confidence": float(best_detection["confidence"]),
        "bbox": best_detection["bbox"],
        "detections": detections,
        "traffic_count": int(traffic_count),
        "selected_index": int(selected_index),
        "roi_enabled": bool(roi_enabled),
        "image_mime": "image/jpeg",
        "image_base64": base64.b64encode(encoded_bytes).decode("ascii"),
    }
    return data, websocket_data, encoded_bytes


def post_detection_to_backend(
    backend_url,
    frame,
    detections,
    traffic_count,
    selected_name,
    selected_index,
    stream_status,
    roi_enabled,
    roi_path,
    session=None,
):
    data, _websocket_data, encoded_bytes = build_detection_payload(
        frame=frame,
        detections=detections,
        traffic_count=traffic_count,
        selected_name=selected_name,
        selected_index=selected_index,
        stream_status=stream_status,
        roi_enabled=roi_enabled,
        roi_path=roi_path,
    )
    files = {
        "frame": ("frame.jpg", encoded_bytes, "image/jpeg"),
    }
    client = session or requests
    response = client.post(
        f"{backend_url}/api/detection",
        data=data,
        files=files,
        timeout=DETECTION_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return len(encoded_bytes)


class WebSocketReconnectWaiting(RuntimeError):
    pass


class DetectionWebSocketSender:
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.ws = None
        self.lock = threading.Lock()
        self.last_connect_attempt_at = 0.0
        self.next_connect_at = 0.0
        self.reconnect_delay = DETECTION_WS_RECONNECT_MIN_SECONDS

    def close(self):
        if not self.lock.acquire(timeout=0.2):
            return
        try:
            self._close_locked()
        finally:
            self.lock.release()

    def _close_locked(self):
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None

    def _mark_connect_success_locked(self):
        self.next_connect_at = 0.0
        self.reconnect_delay = DETECTION_WS_RECONNECT_MIN_SECONDS

    def _mark_connect_failure_locked(self):
        delay = max(DETECTION_WS_RECONNECT_MIN_SECONDS, self.reconnect_delay)
        self.next_connect_at = time.time() + delay
        self.reconnect_delay = min(
            DETECTION_WS_RECONNECT_MAX_SECONDS,
            max(DETECTION_WS_RECONNECT_MIN_SECONDS, delay * 2),
        )

    def _connect_locked(self, force=False):
        if websocket is None:
            raise RuntimeError("websocket-client 패키지가 필요합니다. pip install websocket-client")
        if self.ws is not None:
            return self.ws

        now = time.time()
        if not force and now < self.next_connect_at:
            raise WebSocketReconnectWaiting(
                f"WebSocket reconnect waiting {self.next_connect_at - now:.1f}s"
            )

        self.last_connect_attempt_at = now
        try:
            self.ws = websocket.create_connection(
                self.ws_url,
                timeout=DETECTION_REQUEST_TIMEOUT_SECONDS,
                skip_utf8_validation=True,
            )
            self._mark_connect_success_locked()
            return self.ws
        except Exception:
            self._close_locked()
            self._mark_connect_failure_locked()
            raise

    def send(self, message):
        serialized = json.dumps(message, ensure_ascii=False)
        with self.lock:
            ws = self._connect_locked()
            try:
                ws.send(serialized)
                self._mark_connect_success_locked()
            except Exception:
                self._close_locked()
                ws = self._connect_locked(force=True)
                ws.send(serialized)
                self._mark_connect_success_locked()
