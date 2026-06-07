import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

from utils.analysis import (
    analyze_directional_congestion,
    apply_roi_mask_to_frame,
    collect_detections,
)
from utils.accident import AccidentLogDetector
from utils.app_config import (
    CCTV_INDEX,
    DETECTION_BACKEND_URL,
    DETECTION_CONTROL_ENABLED,
    DETECTION_CONTROL_INTERVAL_SECONDS,
    DETECTION_CONTROL_TIMEOUT_SECONDS,
    DETECTION_LOG_SEND,
    DETECTION_MAX_IN_FLIGHT,
    DETECTION_PLAYER_URL,
    DETECTION_PROCESS_TIMING_LOG_SECONDS,
    DETECTION_SEND_EVENTS_ONLY,
    DETECTION_SEND_FRAME_JPEG,
    DETECTION_SEND_INTERVAL_SECONDS,
    DETECTION_TRANSPORT,
    DETECTION_WS_HTTP_FALLBACK,
    DETECTION_WS_URL,
    DIRECT_CCTV_NAME,
    DIRECT_CCTV_URL,
    DOWNBOUND_LABEL,
    FRAME_WAIT_TIMEOUT_SECONDS,
    REMOTE_CCTV_NAMES,
    UPBOUND_LABEL,
)
from utils.pruned_yolo_cctv import PrunedYOLODetector
from utils.remote import (
    DetectionWebSocketSender,
    WebSocketReconnectWaiting,
    build_detection_payload,
    derive_backend_url_from_ws_url,
    derive_ws_url_from_backend_url,
    encode_frame_to_base64_jpeg,
    get_detection_session,
    post_detection_to_backend,
    summarize_payload_sizes,
)
from utils.renderer import draw_tracking_results
from utils.roi import build_roi_path, load_roi_data
from utils.stream import (
    connect_capture_with_retry,
    fetch_cctv_dataframe,
    read_frame_with_timeout,
    select_cctv,
)


def _mark_accident_detections(detections, accident_events):
    for detection in detections:
        detection["accident_detected"] = False
        detection.pop("accident_reason", None)
        detection.pop("accident_score", None)

    if not accident_events:
        return

    for event in accident_events:
        event_indexes = {
            int(index)
            for index in event.get("detection_indexes", [])
            if index is not None
        }
        event_track_ids = {
            int(track_id)
            for track_id in event.get("track_ids", [])
            if track_id is not None
        }
        event_bboxes = {
            tuple(int(value) for value in bbox)
            for bbox in (event.get("bbox_a"), event.get("bbox_b"))
            if bbox
        }

        for index, detection in enumerate(detections):
            track_id = detection.get("track_id")
            bbox = tuple(int(value) for value in detection.get("bbox", []))
            matched = index in event_indexes
            matched = matched or (track_id is not None and int(track_id) in event_track_ids)
            matched = matched or (bbox and bbox in event_bboxes)
            if not matched:
                continue

            detection["accident_detected"] = True
            detection["accident_reason"] = event.get("reason")
            detection["accident_score"] = event.get("score")


def _has_active_remote_event(congestion, accident_events):
    congestion = congestion or {}
    congestion_active = any(
        bool(item.get("is_congested", False))
        for item in congestion.values()
        if isinstance(item, dict)
    )
    return congestion_active or bool(accident_events)


class PrunedCCTVPipeline:
    """CCTV 수신, detection 실행, 분석 모듈 조합을 담당하는 메인 파이프라인."""

    def __init__(
        self,
        cctv_index=None,
        cctv_url="",
        cctv_name="",
        roi_path=None,
        use_roi=False,
    ):
        self.detector = PrunedYOLODetector()
        self.cctv_df = fetch_cctv_dataframe()
        self.cctv_items = self.cctv_df["cctvname"].astype(str).fillna("").tolist()
        self.remote_cctv_index_by_name = {
            name: index for index, name in enumerate(REMOTE_CCTV_NAMES)
        }

        self.cctv_index = CCTV_INDEX if cctv_index is None else cctv_index
        self.direct_cctv_url = cctv_url or DIRECT_CCTV_URL
        self.direct_cctv_name = cctv_name or DIRECT_CCTV_NAME
        self.use_roi = bool(use_roi)
        self._custom_roi_path = roi_path
        self.roi_path = roi_path
        self.rois = []

        self.capture = None
        self.current_stream_url = None
        self.current_cctv_name = "-"
        self.current_cctv_url = ""
        self.current_index = 0
        self.remote_backend_url = DETECTION_BACKEND_URL or derive_backend_url_from_ws_url(DETECTION_WS_URL)
        self.remote_transport = DETECTION_TRANSPORT
        self.remote_ws_url = DETECTION_WS_URL or derive_ws_url_from_backend_url(self.remote_backend_url)
        self.remote_control_enabled = (
            DETECTION_CONTROL_ENABLED
            and bool(self.remote_backend_url)
            and not bool(self.direct_cctv_url)
        )
        self.remote_control_interval = DETECTION_CONTROL_INTERVAL_SECONDS
        self._last_remote_control_check_at = 0.0
        self._seen_remote_selection_version = None
        self._remote_control_in_flight = False
        self._pending_remote_control = None
        self._remote_control_lock = threading.Lock()
        self._remote_control_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="detection-control",
        )
        self._websocket_sender = (
            DetectionWebSocketSender(self.remote_ws_url)
            if self.remote_transport == "websocket" and self.remote_ws_url
            else None
        )
        self.remote_send_interval = DETECTION_SEND_INTERVAL_SECONDS
        self.remote_max_in_flight = 1 if self.remote_transport == "websocket" else DETECTION_MAX_IN_FLIGHT
        self._last_remote_send_at = 0.0
        self._remote_send_in_flight = 0
        self._pending_remote_payload = None
        self._remote_send_lock = threading.Lock()
        self._remote_log_lock = threading.Lock()
        self._last_remote_log_at = 0.0
        self._remote_event_active = False
        self._remote_send_executor = ThreadPoolExecutor(
            max_workers=DETECTION_MAX_IN_FLIGHT,
            thread_name_prefix="detection-send",
        )
        self._frame_sequence = 0
        self._previous_direction_centers = {
            "up": [],
            "down": [],
        }
        self._accident_detector = AccidentLogDetector()

    def get_cctv_items(self):
        return [
            {
                "index": index,
                "name": name,
                "selected": index == self.current_index,
            }
            for index, name in enumerate(self.cctv_items)
        ]

    def open(self):
        if self.direct_cctv_url:
            self.current_cctv_name = self.direct_cctv_name or "직접 지정 CCTV"
            self.current_cctv_url = self.direct_cctv_url
            self.current_index = 0
        else:
            cctv_name, cctv_url, selected_index = select_cctv(self.cctv_df, self.cctv_index)
            self.current_cctv_name = str(cctv_name)
            self.current_cctv_url = str(cctv_url)
            self.current_index = int(selected_index)

        self._update_roi_path()
        self._reset_direction_history()
        self._accident_detector.reset()
        self.detector.reset_tracking()
        self._open_capture()
        return self.current_cctv_name

    def select_cctv(self, index):
        if index < 0 or index >= len(self.cctv_df):
            raise IndexError("CCTV 인덱스 범위를 벗어났습니다.")

        previous_index = self.current_index
        previous_name = self.current_cctv_name
        previous_url = self.current_cctv_url
        previous_roi_path = self.roi_path
        previous_rois = list(self.rois)
        previous_stream_url = self.current_stream_url
        previous_capture = self.capture

        row = self.cctv_df.iloc[index]
        self.current_index = int(index)
        self.current_cctv_name = str(row["cctvname"])
        self.current_cctv_url = str(row["cctvurl"])
        self._update_roi_path()
        self.capture = None
        self._reset_direction_history()
        self._accident_detector.reset()
        self.detector.reset_tracking()

        try:
            self._open_capture()
        except Exception:
            self.current_index = previous_index
            self.current_cctv_name = previous_name
            self.current_cctv_url = previous_url
            self.roi_path = previous_roi_path
            self.rois = previous_rois
            self.current_stream_url = previous_stream_url
            self.capture = previous_capture
            raise

        if previous_capture is not None:
            previous_capture.release()

    def reload_rois(self):
        if self.roi_path and self.use_roi:
            self.rois = load_roi_data(self.roi_path)
        else:
            self.rois = []

    def _update_roi_path(self):
        if self._custom_roi_path:
            self.roi_path = self._custom_roi_path
        else:
            self.roi_path = build_roi_path(self.current_index)
        self.reload_rois()

    def _open_capture(self):
        if not self.current_cctv_url:
            raise RuntimeError("현재 CCTV URL이 비어 있습니다.")

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.current_stream_url, self.capture = connect_capture_with_retry(self.current_cctv_url)

    def _reset_direction_history(self):
        self._previous_direction_centers = {
            "up": {},
            "down": {},
        }

    def _get_remote_display_index(self):
        return self.remote_cctv_index_by_name.get(self.current_cctv_name, self.current_index)

    def _find_raw_cctv_index_by_name(self, selected_name):
        if not selected_name:
            return None

        for index, name in enumerate(self.cctv_items):
            if str(name) == str(selected_name):
                return index

        return None

    def _schedule_remote_control_if_needed(self):
        if not self.remote_control_enabled:
            return

        now = time.time()
        if now - self._last_remote_control_check_at < self.remote_control_interval:
            return
        self._last_remote_control_check_at = now

        with self._remote_control_lock:
            if self._remote_control_in_flight:
                return
            self._remote_control_in_flight = True

        future = self._remote_control_executor.submit(self._fetch_remote_control)
        future.add_done_callback(self._remote_control_done)

    def _fetch_remote_control(self):
        try:
            response = get_detection_session().get(
                f"{self.remote_backend_url}/api/control",
                timeout=DETECTION_CONTROL_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            if DETECTION_LOG_SEND:
                print(f"remote control fetch failed: {exc}")
            return None

    def _remote_control_done(self, future):
        try:
            control = future.result()
        except Exception as exc:
            control = None
            if DETECTION_LOG_SEND:
                print(f"remote control fetch failed: {exc}")

        with self._remote_control_lock:
            if control is not None:
                self._pending_remote_control = control
            self._remote_control_in_flight = False

    def _apply_remote_control_if_available(self):
        with self._remote_control_lock:
            control = self._pending_remote_control
            self._pending_remote_control = None

        if control is None:
            return

        version = control.get("selection_version")
        if self._seen_remote_selection_version is None:
            self._seen_remote_selection_version = version
            if not version:
                return
        elif version == self._seen_remote_selection_version:
            return

        self._seen_remote_selection_version = version
        target_index = self._find_raw_cctv_index_by_name(control.get("selected_name"))
        if target_index is None:
            try:
                fallback_index = int(control.get("selected_index"))
            except (TypeError, ValueError):
                return
            if 0 <= fallback_index < len(self.cctv_df):
                target_index = fallback_index

        if target_index is None or target_index == self.current_index:
            return

        try:
            print(
                "remote CCTV selection:",
                f"{self.current_cctv_name} -> {self.cctv_items[target_index]}",
            )
            self.select_cctv(target_index)
        except Exception as exc:
            print(f"remote CCTV selection failed: {exc}")

    def _build_remote_payload(
        self,
        frame,
        detections,
        rois,
        frame_id,
        timestamp,
        congestion,
        accident_events,
        event_active,
        event_type,
    ):
        frame_height, frame_width = frame.shape[:2]
        return {
            "backend_url": self.remote_backend_url,
            "detections": [dict(item) for item in detections],
            "rois": [
                {
                    "points": [[int(x), int(y)] for x, y in points],
                    "label": int(label),
                }
                for points, label in rois
            ],
            "frame_id": int(frame_id),
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "timestamp": str(timestamp),
            "selected_name": self.current_cctv_name,
            "selected_index": int(self._get_remote_display_index()),
            "cctv_url": self.current_cctv_url,
            "stream_url": self.current_stream_url,
            "player_url": DETECTION_PLAYER_URL or self.current_stream_url or self.current_cctv_url,
            "stream_status": "연결됨",
            "roi_enabled": bool(rois),
            "roi_path": self.roi_path,
            "congestion": congestion or {},
            "accident_events": [dict(item) for item in (accident_events or [])],
            "event_active": bool(event_active),
            "event_type": str(event_type),
            "frame": frame,
        }

    def _submit_remote_payload_locked(self, payload):
        self._remote_send_in_flight += 1
        self._last_remote_send_at = time.time()
        future = self._remote_send_executor.submit(
            self._send_detection_worker,
            **payload,
        )
        future.add_done_callback(self._remote_send_done)
        self._remote_event_active = bool(payload.get("event_active", False))

    def _send_detection_if_needed(
        self,
        frame,
        detections,
        rois,
        frame_id,
        timestamp,
        congestion=None,
        accident_events=None,
    ):
        if self.remote_transport == "websocket":
            if not self._websocket_sender:
                return
        elif not self.remote_backend_url:
            return

        event_active = _has_active_remote_event(congestion, accident_events)
        event_type = "event" if event_active else "normal"
        if DETECTION_SEND_EVENTS_ONLY and not event_active and self._remote_event_active:
            event_type = "clear"
        if DETECTION_SEND_EVENTS_ONLY and not event_active and not self._remote_event_active:
            return

        now = time.time()
        if now - self._last_remote_send_at < self.remote_send_interval:
            return

        payload = self._build_remote_payload(
            frame,
            detections,
            rois,
            frame_id,
            timestamp,
            congestion,
            accident_events,
            event_active,
            event_type,
        )
        with self._remote_send_lock:
            if self._remote_send_in_flight >= self.remote_max_in_flight:
                self._pending_remote_payload = payload
                return
            self._submit_remote_payload_locked(payload)

    def _remote_send_done(self, _future):
        with self._remote_send_lock:
            self._remote_send_in_flight = max(0, self._remote_send_in_flight - 1)
            if (
                self._pending_remote_payload is not None
                and self._remote_send_in_flight < self.remote_max_in_flight
            ):
                payload = self._pending_remote_payload
                self._pending_remote_payload = None
                self._submit_remote_payload_locked(payload)

    def _send_detection_worker(self, **payload):
        try:
            started_at = time.time()
            transport = self.remote_transport
            frame_jpeg = encode_frame_to_base64_jpeg(payload.get("frame")) if DETECTION_SEND_FRAME_JPEG else ""
            if self.remote_transport == "websocket":
                _data, websocket_data, payload_size = build_detection_payload(
                    detections=payload["detections"],
                    rois=payload["rois"],
                    frame_id=payload["frame_id"],
                    frame_width=payload["frame_width"],
                    frame_height=payload["frame_height"],
                    timestamp=payload["timestamp"],
                    selected_name=payload["selected_name"],
                    selected_index=payload["selected_index"],
                    cctv_url=payload["cctv_url"],
                    stream_url=payload["stream_url"],
                    player_url=payload["player_url"],
                    stream_status=payload["stream_status"],
                    roi_enabled=payload["roi_enabled"],
                    roi_path=payload["roi_path"],
                    congestion=payload.get("congestion"),
                    accident_events=payload.get("accident_events"),
                    event_active=payload.get("event_active", False),
                    event_type=payload.get("event_type", "normal"),
                    frame_jpeg=frame_jpeg,
                )
                try:
                    self._websocket_sender.send(websocket_data)
                    encoded_size = payload_size
                except WebSocketReconnectWaiting:
                    return
                except Exception as ws_exc:
                    if not DETECTION_WS_HTTP_FALLBACK or not self.remote_backend_url:
                        raise

                    encoded_size = post_detection_to_backend(
                        **payload,
                        frame_jpeg=frame_jpeg,
                        session=get_detection_session(),
                    )
                    transport = "http-fallback"
                    if DETECTION_LOG_SEND:
                        print(f"websocket send failed, used HTTP fallback: {ws_exc}")
            else:
                encoded_size = post_detection_to_backend(
                    **payload,
                    frame_jpeg=frame_jpeg,
                    session=get_detection_session(),
                )
            if DETECTION_LOG_SEND:
                completed_at = time.time()
                elapsed = completed_at - started_at
                with self._remote_log_lock:
                    interval = completed_at - self._last_remote_log_at if self._last_remote_log_at else 0.0
                    self._last_remote_log_at = completed_at
                send_fps = (1.0 / interval) if interval > 0 else 0.0
                print(
                    "remote detection sent:",
                    f"send={elapsed:.3f}s",
                    f"interval={interval:.2f}s",
                    f"fps={send_fps:.1f}",
                    f"{encoded_size / 1024:.1f}KB",
                    f"transport={transport}",
                    f"frame_jpeg={'on' if bool(frame_jpeg) else 'off'}",
                    f"detections={len(payload['detections'])}",
                    f"event={payload.get('event_type', 'normal')}",
                    "top_fields="
                    + ",".join(
                        f"{key}:{size / 1024:.1f}KB"
                        for key, size in summarize_payload_sizes(websocket_data)
                    ),
                )
        except requests.RequestException as exc:
            print(f"remote detection send failed: {exc}")
        except Exception as exc:
            print(f"remote detection send failed: {exc}")

    def process_next_frame(self):
        process_started_at = time.time()
        self._schedule_remote_control_if_needed()
        self._apply_remote_control_if_available()

        if self.capture is None or not self.capture.isOpened():
            self._open_capture()

        read_started_at = time.time()
        ok, frame = read_frame_with_timeout(self.capture, FRAME_WAIT_TIMEOUT_SECONDS)
        if not ok:
            if self.capture is not None:
                self.capture.release()
            time.sleep(1)
            self._open_capture()
            ok, frame = read_frame_with_timeout(self.capture, FRAME_WAIT_TIMEOUT_SECONDS)
            if not ok:
                raise RuntimeError("프레임 수신 실패")
        read_elapsed = time.time() - read_started_at

        rois = self.rois if self.use_roi else []
        inference_frame = apply_roi_mask_to_frame(frame, rois) if rois else frame
        self._frame_sequence += 1
        frame_id = self._frame_sequence
        frame_timestamp = datetime.now(timezone.utc).isoformat()
        inference_started_at = time.time()
        results = self.detector.predict(inference_frame)
        inference_elapsed = time.time() - inference_started_at
        analysis_started_at = time.time()
        detections = collect_detections(results[0], rois=rois)
        congestion, current_direction_centers = analyze_directional_congestion(
            frame.shape,
            detections,
            rois,
            self._previous_direction_centers,
        )
        self._previous_direction_centers = current_direction_centers
        accident_events = self._accident_detector.update(
            detections,
            frame_id=frame_id,
            cctv_name=self.current_cctv_name,
            timestamp=frame_timestamp,
        )
        _mark_accident_detections(detections, accident_events)
        analysis_elapsed = time.time() - analysis_started_at
        draw_started_at = time.time()
        annotated_frame = draw_tracking_results(frame, detections, rois=rois)
        draw_elapsed = time.time() - draw_started_at
        traffic_count = len(detections)
        traffic_up = int(congestion["up"]["count"])
        traffic_down = int(congestion["down"]["count"])
        send_started_at = time.time()
        self._send_detection_if_needed(
            annotated_frame,
            detections,
            rois,
            frame_id,
            frame_timestamp,
            congestion=congestion,
            accident_events=accident_events,
        )
        send_elapsed = time.time() - send_started_at
        total_elapsed = time.time() - process_started_at
        if DETECTION_LOG_SEND and total_elapsed >= DETECTION_PROCESS_TIMING_LOG_SECONDS:
            print(
                "process frame timing:",
                f"total={total_elapsed:.2f}s",
                f"read={read_elapsed:.2f}s",
                f"infer={inference_elapsed:.2f}s",
                f"analysis={analysis_elapsed:.2f}s",
                f"draw={draw_elapsed:.2f}s",
                f"send_schedule={send_elapsed:.3f}s",
            )

        return {
            "frame": annotated_frame,
            "frame_id": frame_id,
            "timestamp": frame_timestamp,
            "traffic_count": traffic_count,
            "traffic_up": traffic_up,
            "traffic_down": traffic_down,
            "detections": detections,
            "congestion": congestion,
            "rois": [
                {
                    "points": [[int(x), int(y)] for x, y in points],
                    "label": int(label),
                }
                for points, label in rois
            ],
            "frame_width": int(frame.shape[1]),
            "frame_height": int(frame.shape[0]),
            "selected_name": self.current_cctv_name,
            "selected_index": self.current_index,
            "stream_url": self.current_stream_url,
            "stream_status": "연결됨",
            "yolo_enabled": True,
            "roi_enabled": bool(rois),
            "roi_path": self.roi_path,
        }

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self._websocket_sender is not None:
            self._websocket_sender.close()
        self._remote_control_executor.shutdown(wait=False, cancel_futures=True)
        self._remote_send_executor.shutdown(wait=False, cancel_futures=True)
