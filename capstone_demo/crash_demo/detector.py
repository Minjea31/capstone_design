import os
import threading
import time
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np

from .utils.accident import AccidentLogDetector
from .utils.video_source import find_saved_videos, get_video_delay_ms, open_video_capture


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_PATH = MODULE_DIR / "crash.pt"
DEFAULT_SAVED_VIDEOS_DIR = MODULE_DIR / "saved_videos"
DEFAULT_ROI_PATH = MODULE_DIR / "referenceline" / "crash.json"
DEFAULT_REFERENCE_PATH = MODULE_DIR / "referenceline" / "crash_reference.json"
VEHICLE_CLASSES = [0]
CONFIDENCE_THRESHOLD = 0.7
ACCIDENT_STATE_HOLD_SECONDS = float(os.environ.get("WEB_ACCIDENT_HOLD_SECONDS", "5.0"))
ROI_COLORS = {
    0: (0, 178, 255),
    1: (54, 242, 138),
    -1: (180, 180, 180),
}


def _load_yolo_model(weights_path, device=None):
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("crash_demo 실행에 torch와 ultralytics가 필요합니다.") from exc

    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"사고 감지 모델 파일을 찾을 수 없습니다: {path}")

    original_torch_load = torch.load

    def patched_torch_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    try:
        torch.load = patched_torch_load
        model = YOLO(str(path))
    finally:
        torch.load = original_torch_load

    if device:
        model.to(device)
    return model


def _reset_tracker_state(model):
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return
    trackers = getattr(predictor, "trackers", None)
    if trackers:
        for tracker in trackers:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()


def _bbox_roi_overlap_ratio(bbox, polygon_points):
    x1, y1, x2, y2 = [int(value) for value in bbox]
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    if width <= 0 or height <= 0:
        return 0.0

    mask = np.zeros((height, width), dtype=np.uint8)
    shifted_points = np.array(
        [[int(px - x1), int(py - y1)] for px, py in polygon_points],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [shifted_points], 1)
    overlap = int(np.count_nonzero(mask))
    return overlap / float(width * height)


def _direction_label_for_bbox(bbox, roi_polygons):
    if not roi_polygons:
        return None

    x1, y1, x2, y2 = [int(value) for value in bbox]
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    best_label = None
    best_overlap = 0.0

    for polygon in roi_polygons:
        points = np.array(polygon["points"], dtype=np.int32)
        if cv2.pointPolygonTest(points, center, False) >= 0:
            return int(polygon["label"])

        overlap = _bbox_roi_overlap_ratio(bbox, polygon["points"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = int(polygon["label"])

    return best_label if best_overlap > 0 else None


def _event_detection_indexes(events):
    indexes = set()
    for event in events or []:
        for index in event.get("detection_indexes", []):
            if index is not None:
                indexes.add(int(index))
    return indexes


def _collect_tracked_detections(result, roi_polygons=None):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy_list = boxes.xyxy.cpu().tolist()
    track_ids = None
    if getattr(boxes, "id", None) is not None:
        track_ids = boxes.id.int().cpu().tolist()

    detections = []
    for index, xyxy in enumerate(xyxy_list):
        x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
        track_id = int(track_ids[index]) if track_ids is not None else None
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                "detection_index": int(index),
                "track_id": track_id,
                "direction_label": _direction_label_for_bbox([x1, y1, x2, y2], roi_polygons or []),
            }
        )
    return detections


def _event_flags(events):
    if not events:
        return False, False

    up = False
    down = False
    for event in events:
        directions = [value for value in event.get("directions", []) if value is not None]
        if not directions:
            up = True
            down = True
            continue
        down = down or any(int(value) == 0 for value in directions)
        up = up or any(int(value) == 1 for value in directions)
    return up, down


def _load_json(path):
    import json

    try:
        with Path(path).open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        print(f"[CRASH] reference load failed: {path} ({exc})", flush=True)
        return None


def _load_roi_polygons(path=DEFAULT_ROI_PATH):
    payload = _load_json(path)
    if not isinstance(payload, list):
        return []

    polygons = []
    for item in payload:
        if not isinstance(item, list) or len(item) != 2:
            continue
        points_data, label = item
        if not isinstance(points_data, list):
            continue
        points = []
        for point in points_data:
            if isinstance(point, list) and len(point) == 2:
                points.append((int(point[0]), int(point[1])))
        if len(points) >= 3:
            polygons.append({"points": points, "label": int(label)})
    return polygons


def _load_reference_lines(path=DEFAULT_REFERENCE_PATH):
    payload = _load_json(path)
    groups = payload.get("groups", []) if isinstance(payload, dict) else []
    if not isinstance(groups, list):
        return []

    lines = []
    for group in groups:
        points_data = group.get("points", []) if isinstance(group, dict) else []
        roi_labels = group.get("roi_labels", []) if isinstance(group, dict) else []
        if not isinstance(points_data, list) or len(points_data) < 2:
            continue
        points = []
        for point in points_data:
            if isinstance(point, list) and len(point) == 2:
                points.append((int(point[0]), int(point[1])))
        if len(points) < 2:
            continue
        label = int(roi_labels[0]) if roi_labels else -1
        lines.append({"points": points, "label": label})
    return lines


def _draw_reference_overlay(frame, roi_polygons, reference_lines):
    overlay = frame.copy()
    for polygon in roi_polygons:
        points = np.array(polygon["points"], dtype=np.int32)
        color = ROI_COLORS.get(int(polygon["label"]), ROI_COLORS[-1])
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    for line in reference_lines:
        points = np.array(line["points"], dtype=np.int32)
        color = ROI_COLORS.get(int(line["label"]), (255, 255, 255))
        cv2.polylines(frame, [points], False, color, 3, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, tuple(int(value) for value in point), 4, color, -1, cv2.LINE_AA)


class CrashVideoMonitor:
    def __init__(self, weights_path=DEFAULT_WEIGHTS_PATH, videos_dir=DEFAULT_SAVED_VIDEOS_DIR, device=None):
        self.weights_path = Path(weights_path)
        self.videos_dir = Path(videos_dir)
        self.device = device
        self._lock = threading.Lock()
        self._frame_condition = threading.Condition()
        self._latest_jpeg = None
        self._thread = None
        self._stop_event = threading.Event()
        self._state = {
            "enabled": True,
            "running": False,
            "source": "",
            "frame_id": 0,
            "accident_up": False,
            "accident_down": False,
            "last_event": None,
            "error": "",
        }
        self._active_until = 0.0
        self._active_detection_indexes = set()
        self._roi_polygons = _load_roi_polygons()
        self._reference_lines = _load_reference_lines()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="crash-video-monitor", daemon=True)
        self._thread.start()

    def snapshot(self):
        with self._lock:
            state = dict(self._state)

        if time.time() > self._active_until:
            state["accident_up"] = False
            state["accident_down"] = False
        return state

    def frame_generator(self):
        while not self._stop_event.is_set():
            with self._frame_condition:
                self._frame_condition.wait(timeout=1.0)
                jpeg = self._latest_jpeg
            if jpeg is None:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")

    def _set_state(self, **updates):
        with self._lock:
            self._state.update(updates)

    def _set_accident_state(self, events, video_path, frame_id):
        accident_up, accident_down = _event_flags(events)
        if accident_up or accident_down:
            self._active_until = max(self._active_until, time.time() + ACCIDENT_STATE_HOLD_SECONDS)
            self._active_detection_indexes = _event_detection_indexes(events)
            self._set_state(
                source=str(video_path),
                frame_id=int(frame_id),
                accident_up=accident_up,
                accident_down=accident_down,
                last_event=events[0] if events else None,
                error="",
            )
            return

        if time.time() > self._active_until:
            self._active_detection_indexes.clear()
            self._set_state(accident_up=False, accident_down=False)

    def _publish_frame(self, frame, detections, events):
        annotated = frame.copy()
        _draw_reference_overlay(annotated, self._roi_polygons, self._reference_lines)
        has_accident = bool(events) or time.time() <= self._active_until
        accident_indexes = _event_detection_indexes(events) or self._active_detection_indexes

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            direction_label = detection.get("direction_label")
            color = ROI_COLORS.get(int(direction_label), ROI_COLORS[-1]) if direction_label is not None else ROI_COLORS[-1]
            if int(detection.get("detection_index", -1)) in accident_indexes:
                color = (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        if has_accident:
            cv2.putText(
                annotated,
                "ACCIDENT",
                (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        ok, jpeg = cv2.imencode(".jpg", annotated)
        if not ok:
            return

        with self._frame_condition:
            self._latest_jpeg = jpeg.tobytes()
            self._frame_condition.notify_all()

    def _run(self):
        try:
            videos = find_saved_videos(self.videos_dir)
            if not videos:
                self._set_state(running=False, error=f"saved_videos에 영상이 없습니다: {self.videos_dir}")
                return

            model = _load_yolo_model(self.weights_path, self.device)
            accident_detector = AccidentLogDetector(log_enabled=True)
            self._set_state(running=True, source=str(videos[-1]), error="")

            while not self._stop_event.is_set():
                for video_path in videos:
                    if self._stop_event.is_set():
                        break
                    self._process_video(model, accident_detector, video_path)
        except Exception as exc:
            self._set_state(running=False, error=str(exc))
            print(f"[CRASH] monitor stopped: {exc}", flush=True)

    def _process_video(self, model, accident_detector, video_path):
        capture = open_video_capture(video_path)
        delay_seconds = get_video_delay_ms(capture) / 1000.0
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 1.0 / max(delay_seconds, 0.001)

        _reset_tracker_state(model)
        accident_detector.reset()
        self._set_state(source=str(video_path), frame_id=0, error="")
        print(f"[CRASH] video input: {video_path}", flush=True)

        try:
            frame_id = 0
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                frame_id += 1
                frame_time_seconds = frame_id / fps
                results = model.track(
                    frame,
                    persist=True,
                    classes=VEHICLE_CLASSES,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                    device=self.device,
                )
                detections = _collect_tracked_detections(results[0], self._roi_polygons) if results else []
                events = accident_detector.update(
                    detections,
                    frame_id=frame_id,
                    cctv_name=video_path.name,
                    timestamp=f"{frame_time_seconds:.3f}s",
                    now=frame_time_seconds,
                )
                self._set_accident_state(events, video_path, frame_id)
                self._publish_frame(frame, detections, events)
                time.sleep(min(delay_seconds, 0.05))
        finally:
            capture.release()
