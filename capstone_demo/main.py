import argparse
from collections import deque
import glob
import json
import os
from pathlib import Path
import threading

try:
    from flask import Flask, Response, render_template, send_from_directory
except ImportError:
    raise SystemExit("Flask가 설치되어 있지 않습니다. python -m pip install flask 를 실행하세요.")

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
ROI_DIR = ROOT_DIR / "referenceline"
DEFAULT_MODEL_PATH = ROOT_DIR / "model" / "raw_model.pt"
DEFAULT_ROI_GLOB = "roi_data_*.json"
DEFAULT_EXTERNAL_CAMERA = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0"
USB_CAMERA_KEYWORDS = ("c920", "logitech", "pro webcam", "usb")
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DOWNBOUND_LABEL = 0
UPBOUND_LABEL = 1
ROI_BBOX_OVERLAP_THRESHOLD = 0.1
CONGESTION_OVERLAP_THRESHOLD = 0.5
CONGESTION_CENTER_MOVEMENT_THRESHOLD = 1000
HISTORY_LIMIT = 36

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

app = Flask(__name__, static_folder=str(WEB_DIR), template_folder=str(WEB_DIR))

state_lock = threading.Lock()
traffic_up_history = deque([0], maxlen=HISTORY_LIMIT)
traffic_down_history = deque([0], maxlen=HISTORY_LIMIT)
latest_detection = None
latest_counts = {
    "traffic_count": 0,
    "traffic_up": 0,
    "traffic_down": 0,
}
latest_congestion = {
    "up": {
        "count": 0,
        "roi_coverage_ratio": 0.0,
        "mean_center_movement": None,
        "is_congested": False,
    },
    "down": {
        "count": 0,
        "roi_coverage_ratio": 0.0,
        "mean_center_movement": None,
        "is_congested": False,
    },
}
previous_direction_centers = {"up": [], "down": []}
loaded_roi_path = None
_yolo_model = None
_yolo_names = {}
_last_roi_scale_log = None


def camera_name(device_path):
    device_name = Path(os.path.realpath(str(device_path))).name
    name_path = Path("/sys/class/video4linux") / device_name / "name"
    try:
        return name_path.read_text().strip()
    except OSError:
        return device_name


def source_to_opencv_candidates(camera_source):
    candidates = []
    source_text = str(camera_source)

    if source_text.isdigit():
        candidates.append(int(source_text))
        candidates.append(f"/dev/video{source_text}")
        return candidates

    candidates.append(source_text)
    resolved = os.path.realpath(source_text)
    if resolved != source_text:
        candidates.append(resolved)

    device_name = Path(resolved).name
    if device_name.startswith("video"):
        candidates.append(int(device_name.removeprefix("video")))

    return candidates


def configure_camera(cap, width, height, fps):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)


def open_camera(camera_source, width, height, fps):
    for opencv_source in source_to_opencv_candidates(camera_source):
        cap = cv2.VideoCapture(opencv_source, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue

        configure_camera(cap, width, height, fps)
        ok, _frame = cap.read()
        if ok:
            return cap

        cap.release()

    return None


def camera_candidates():
    candidates = set(glob.glob("/dev/video*"))
    if os.path.exists(DEFAULT_EXTERNAL_CAMERA):
        candidates.add(DEFAULT_EXTERNAL_CAMERA)

    def sort_key(device_path):
        name = camera_name(device_path).lower()
        resolved = Path(os.path.realpath(device_path)).name
        index = int(resolved.removeprefix("video")) if resolved.startswith("video") else 999
        is_usb_camera = any(keyword in name for keyword in USB_CAMERA_KEYWORDS)
        is_by_id = str(device_path).startswith("/dev/v4l/by-id/")
        return (not is_by_id, not is_usb_camera, index)

    return sorted(candidates, key=sort_key)


def open_named_camera(name_query, width, height, fps):
    for device_path in camera_candidates():
        if name_query.lower() not in camera_name(device_path).lower():
            continue

        cap = open_camera(device_path, width, height, fps)
        if cap is not None:
            return cap, device_path

    return None, None


def find_camera(width, height, fps):
    for device_path in camera_candidates():
        cap = open_camera(device_path, width, height, fps)
        if cap is not None:
            return cap, device_path
    return None, None


def resolve_camera(camera_source, width, height, fps):
    if camera_source is None:
        return find_camera(width, height, fps)

    camera_source = str(camera_source)
    if camera_source.isdigit() or os.path.exists(camera_source):
        return open_camera(camera_source, width, height, fps), camera_source

    return open_named_camera(camera_source, width, height, fps)


def load_roi_file(roi_json_path=None):
    global loaded_roi_path

    if roi_json_path is not None:
        path = Path(roi_json_path)
        if not path.exists():
            raise FileNotFoundError(f"ROI 파일을 찾을 수 없습니다: {path}")
    else:
        candidates = sorted(ROI_DIR.glob(DEFAULT_ROI_GLOB))
        if not candidates:
            raise FileNotFoundError(f"referenceline 폴더에 ROI 파일이 없습니다: {ROI_DIR}")
        path = candidates[0]

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    print(f"[ROI] Loaded ROI file: {path} ({len(payload)} items)")
    loaded_roi_path = str(path)

    # Keep original point coordinates from ROI file (assumed pixel coords),
    # but don't assume they match current camera resolution. We'll scale per-frame.
    rois = []
    for item in payload:
        if not isinstance(item, list) or len(item) != 2:
            continue
        points, label = item
        if not isinstance(points, list) or len(points) < 3:
            continue
        # store raw points as list of (x,y)
        pts = [(int(p[0]), int(p[1])) for p in points]
        rois.append({"points": pts, "label": int(label)})
    return rois


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def bbox_center_inside_roi(bbox, rois):
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return any(point_in_polygon((cx, cy), roi["polygon"]) for roi in rois)


def direction_color(label):
    if int(label) == UPBOUND_LABEL:
        return (0, 255, 0)
    if int(label) == DOWNBOUND_LABEL:
        return (0, 165, 255)
    return (255, 255, 0)


def draw_rois(frame, rois):
    overlay = frame.copy()
    for roi in rois:
        color = direction_color(roi["label"])
        cv2.fillPoly(overlay, [roi["polygon"]], color)
        cv2.polylines(frame, [roi["polygon"]], True, color, 1)
    return cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)


def scale_rois_for_frame(rois, frame_w, frame_h):
    """Scale stored ROI points to the current frame resolution.

    rois: list of {"points": [(x,y),...], "label": int}
    returns: list of {"polygon": np.array([...], np.int32), "label": int}
    """
    global _last_roi_scale_log

    scaled = []
    # compute max coords from ROI to infer source coordinate space
    max_x = 0
    max_y = 0
    for r in rois:
        for (x, y) in r["points"]:
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

    # Determine if scaling is needed: only scale when ROI coords exceed current frame
    need_scale = (max_x + 1 > frame_w) or (max_y + 1 > frame_h)
    scale_x = 1.0
    scale_y = 1.0
    if need_scale:
        src_w = max_x + 1
        src_h = max_y + 1
        scale_x = frame_w / src_w
        scale_y = frame_h / src_h

    for r in rois:
        pts = []
        for (x, y) in r["points"]:
            if need_scale:
                sx = int(x * scale_x)
                sy = int(y * scale_y)
            else:
                sx = int(x)
                sy = int(y)
            pts.append((sx, sy))
        polygon = np.array(pts, np.int32)
        scaled.append({"polygon": polygon, "label": r["label"]})
    scale_log = (frame_w, frame_h, max_x, max_y, need_scale, round(scale_x, 3), round(scale_y, 3))
    if scale_log != _last_roi_scale_log:
        print(f"[ROI] frame=({frame_w}x{frame_h}) max_roi=({max_x}x{max_y}) need_scale={need_scale} scale=({scale_x if need_scale else 1:.3f},{scale_y if need_scale else 1:.3f})")
        _last_roi_scale_log = scale_log
    return scaled


def scaled_rois_as_points(rois):
    return [
        {
            "points": [[int(x), int(y)] for x, y in roi["polygon"].tolist()],
            "label": int(roi["label"]),
        }
        for roi in rois
    ]


def get_bbox_roi_overlap_ratio(bbox, rois):
    x1, y1, x2, y2 = [int(value) for value in bbox]
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    if width <= 0 or height <= 0 or not rois:
        return 0.0

    roi_mask = np.zeros((height, width), dtype=np.uint8)
    for roi in rois:
        shifted_points = np.array(
            [[int(px - x1), int(py - y1)] for px, py in roi["polygon"]],
            dtype=np.int32,
        )
        cv2.fillPoly(roi_mask, [shifted_points], 1)

    overlap_pixels = int(np.count_nonzero(roi_mask))
    return min(1.0, overlap_pixels / float(width * height))


def get_bbox_direction_label(bbox, rois):
    x1, y1, x2, y2 = [int(value) for value in bbox]
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    for roi in rois:
        if point_in_polygon(center, roi["polygon"]):
            return int(roi["label"])
    return None


def build_roi_mask(frame_shape, rois, label):
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for roi in rois:
        if int(roi["label"]) != int(label):
            continue
        cv2.fillPoly(mask, [roi["polygon"]], 1)

    return mask


def calculate_roi_coverage_ratio(frame_shape, detections, rois, label):
    if not rois:
        return 0.0

    roi_mask = build_roi_mask(frame_shape, rois, label)
    roi_area = int(np.count_nonzero(roi_mask))
    if roi_area <= 0:
        return 0.0

    detection_mask = np.zeros_like(roi_mask)
    for detection in detections:
        if detection.get("direction_label") != int(label):
            continue

        x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
        x1 = max(0, min(x1, detection_mask.shape[1]))
        x2 = max(0, min(x2, detection_mask.shape[1]))
        y1 = max(0, min(y1, detection_mask.shape[0]))
        y2 = max(0, min(y2, detection_mask.shape[0]))
        if x2 <= x1 or y2 <= y1:
            continue
        detection_mask[y1:y2, x1:x2] = 1

    overlap_pixels = int(np.count_nonzero(cv2.bitwise_and(roi_mask, detection_mask)))
    return min(1.0, overlap_pixels / float(roi_area))


def calculate_mean_center_movement(current_detections, previous_centers):
    if not current_detections or not previous_centers:
        return None

    unmatched_previous = [tuple(center) for center in previous_centers if center is not None]
    movements = []

    for detection in current_detections:
        center = detection.get("center")
        if center is None or not unmatched_previous:
            continue

        current_center = tuple(center)
        best_index = min(
            range(len(unmatched_previous)),
            key=lambda index: float(
                np.hypot(
                    current_center[0] - unmatched_previous[index][0],
                    current_center[1] - unmatched_previous[index][1],
                )
            ),
        )
        previous_center = unmatched_previous.pop(best_index)
        movements.append(
            float(np.hypot(current_center[0] - previous_center[0], current_center[1] - previous_center[1]))
        )

    if not movements:
        return None

    return float(sum(movements) / len(movements))


def analyze_directional_congestion(frame_shape, detections, rois, previous_centers_by_direction):
    labels = {
        "up": UPBOUND_LABEL,
        "down": DOWNBOUND_LABEL,
    }
    analysis = {}
    current_centers_by_direction = {key: [] for key in labels}

    for direction_name, label in labels.items():
        direction_detections = [
            item
            for item in detections
            if item.get("direction_label") == int(label)
        ]
        current_centers = [
            tuple(item["center"])
            for item in direction_detections
            if item.get("center") is not None
        ]
        current_centers_by_direction[direction_name] = current_centers
        coverage_ratio = calculate_roi_coverage_ratio(frame_shape, direction_detections, rois, label)
        mean_center_movement = calculate_mean_center_movement(
            direction_detections,
            previous_centers_by_direction.get(direction_name, []),
        )
        is_congested = (
            coverage_ratio >= CONGESTION_OVERLAP_THRESHOLD
            and mean_center_movement is not None
            and mean_center_movement <= CONGESTION_CENTER_MOVEMENT_THRESHOLD
        )
        analysis[direction_name] = {
            "count": len(direction_detections),
            "roi_coverage_ratio": round(float(coverage_ratio), 4),
            "mean_center_movement": (
                round(float(mean_center_movement), 4)
                if mean_center_movement is not None
                else None
            ),
            "is_congested": bool(is_congested),
        }

    return analysis, current_centers_by_direction


def parse_classes(value):
    value = str(value).strip()
    if not value or value.lower() in {"all", "none"}:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_model(model_path, device):
    global _yolo_model, _yolo_names

    if _yolo_model is not None:
        return _yolo_model

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics가 설치되어 있지 않습니다. python -m pip install ultralytics 를 실행하세요.") from exc

    path = Path(model_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"YOLO 모델 파일을 찾을 수 없습니다: {path}")

    _yolo_model = YOLO(str(path), task="detect")
    if device:
        _yolo_model.to(device)
    _yolo_names = getattr(_yolo_model, "names", {}) or {}
    print(f"YOLO model: {path}")
    print(f"classes: {_yolo_names}")
    return _yolo_model


def run_yolo(frame):
    model = load_model(args.model, args.device)
    classes = parse_classes(args.classes)
    return model.predict(frame, conf=args.conf, classes=classes, device=args.device, verbose=False)


def collect_detections(result, rois=None):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy_list = boxes.xyxy.int().cpu().tolist()
    cls_list = boxes.cls.int().cpu().tolist()
    conf_list = boxes.conf.cpu().tolist()
    names = getattr(result, "names", None) or _yolo_names or {}

    detections = []
    for xyxy, cls_id, conf in zip(xyxy_list, cls_list, conf_list):
        x1, y1, x2, y2 = [int(value) for value in xyxy]
        center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
        detection = {
            "class_name": str(names.get(int(cls_id), cls_id)),
            "class_id": int(cls_id),
            "confidence": round(float(conf), 4),
            "bbox": [x1, y1, x2, y2],
            "center": center,
        }

        if rois:
            overlap_ratio = get_bbox_roi_overlap_ratio((x1, y1, x2, y2), rois)
            direction_label = get_bbox_direction_label((x1, y1, x2, y2), rois)
            if overlap_ratio < args.roi_overlap or direction_label is None:
                continue
            detection["roi_overlap_ratio"] = round(float(overlap_ratio), 4)
            detection["direction_label"] = int(direction_label)

        detections.append(detection)

    return detections


def draw_detections(frame, detections):
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        direction_label = detection.get("direction_label", -1)
        color = direction_color(direction_label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    return frame


def update_detection_state(frame_shape, detections, rois):
    global latest_detection, latest_counts, latest_congestion, previous_direction_centers

    traffic_up = sum(1 for item in detections if item.get("direction_label") == UPBOUND_LABEL)
    traffic_down = sum(1 for item in detections if item.get("direction_label") == DOWNBOUND_LABEL)
    traffic_count = len(detections)
    congestion, current_direction_centers = analyze_directional_congestion(
        frame_shape,
        detections,
        rois,
        previous_direction_centers,
    )

    payload = {
        "frame_width": int(frame_shape[1]),
        "frame_height": int(frame_shape[0]),
        "detections": detections,
        "rois": scaled_rois_as_points(rois),
        "congestion": congestion,
    }

    with state_lock:
        latest_detection = payload
        latest_congestion = congestion
        previous_direction_centers = current_direction_centers
        latest_counts = {
            "traffic_count": traffic_count,
            "traffic_up": traffic_up,
            "traffic_down": traffic_down,
        }
        traffic_up_history.append(traffic_up)
        traffic_down_history.append(traffic_down)

def build_status():
    roi_path = loaded_roi_path or args.roi_json or ("ROI 비활성화" if not args.roi_use else "ROI 로딩 대기 중")
    with state_lock:
        counts = dict(latest_counts)
        up_history = list(traffic_up_history)
        down_history = list(traffic_down_history)
        detection_payload = latest_detection
        congestion_payload = dict(latest_congestion)

    return {
        "selected_name": "USB Camera",
        "stream_status": "연결됨",
        "yolo_enabled": True,
        "roi_path": roi_path,
        "traffic_count": counts["traffic_count"],
        "traffic_up": counts["traffic_up"],
        "traffic_down": counts["traffic_down"],
        "traffic_up_history": up_history,
        "traffic_down_history": down_history,
        "congestion_up": bool(congestion_payload.get("up", {}).get("is_congested", False)),
        "congestion_down": bool(congestion_payload.get("down", {}).get("is_congested", False)),
        "congestion": congestion_payload,
        "accident_up": False,
        "accident_down": False,
        "latest_detection": detection_payload,
    }


def list_cameras():
    items = []
    for index, device_path in enumerate(camera_candidates()):
        items.append({
            "index": index,
            "name": camera_name(device_path),
            "path": device_path,
            "selected": index == 0,
        })
    if not items:
        items.append({"index": 0, "name": "USB Camera", "path": "unknown", "selected": True})
    return items


@app.route("/api/status")
def api_status():
    status = build_status()
    return Response(json.dumps(status), mimetype="application/json")


@app.route("/api/cctvs")
def api_cctvs():
    return Response(json.dumps({"items": list_cameras()}), mimetype="application/json")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(WEB_DIR, path)


def frame_generator():
    cap, device_path = resolve_camera(None, args.width, args.height, args.fps)
    if cap is None:
        raise RuntimeError("USB 카메라를 열 수 없습니다.")
    print(f"[CAM] opened device: {device_path}")
    print(
        f"[CAM] mode: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
        f"{cap.get(cv2.CAP_PROP_FPS):.1f} FPS"
    )

    rois = []
    if args.roi_use:
        rois = load_roi_file(args.roi_json)

    try:
        printed_size = False
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                continue

            raw_frame = frame.copy()

            # print camera capture size once
            if not printed_size:
                h, w = frame.shape[:2]
                print(f"[CAM] capture frame size: {w}x{h}")
                printed_size = True

            scaled_rois = []

            if args.roi_use and rois:
                scaled_rois = scale_rois_for_frame(rois, raw_frame.shape[1], raw_frame.shape[0])

            results = run_yolo(raw_frame)
            detections = collect_detections(results[0], rois=scaled_rois)
            update_detection_state(raw_frame.shape, detections, scaled_rois)

            if args.roi_use and scaled_rois:
                frame = draw_rois(raw_frame.copy(), scaled_rois)
            else:
                frame = raw_frame.copy()

            frame = draw_detections(frame, detections)

            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    finally:
        cap.release()


def parse_args():
    parser = argparse.ArgumentParser(description="USB 카메라 웹 스트리밍 서버")
    parser.add_argument("--roi-use", action="store_true", help="ROI 영역을 함께 표시합니다")
    parser.add_argument("--roi-json", default=None, help="ROI JSON 파일 경로")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="카메라 요청 너비")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="카메라 요청 높이")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="카메라 요청 FPS")
    parser.add_argument("--model", "--model-path", dest="model", default=str(DEFAULT_MODEL_PATH), help="YOLO .pt 모델 경로")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold")
    parser.add_argument("--classes", default="0", help="감지 클래스 필터. 예: 0 또는 0,1. 전체 클래스는 all")
    parser.add_argument("--roi-overlap", type=float, default=ROI_BBOX_OVERLAP_THRESHOLD, help="bbox 면적 중 ROI와 겹쳐야 하는 최소 비율")
    parser.add_argument("--device", default=None, help="YOLO 추론 장치. 예: cuda:0 또는 cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT, threaded=True)
