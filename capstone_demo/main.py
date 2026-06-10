import argparse
import glob
import json
import os
from pathlib import Path

try:
    from flask import Flask, Response, render_template, send_from_directory
except ImportError:
    raise SystemExit("Flask가 설치되어 있지 않습니다. python -m pip install flask 를 실행하세요.")

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
ROI_DIR = ROOT_DIR / "referenceline"
DEFAULT_ROI_GLOB = "roi_data_*.json"

app = Flask(__name__, static_folder=str(WEB_DIR), template_folder=str(WEB_DIR))


def find_camera(camera_source=None):
    """Open a camera and return (cap, device_path)."""
    if camera_source is not None:
        if str(camera_source).isdigit():
            source = int(camera_source)
        else:
            source = camera_source
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap, str(source)
        cap.release()
        return None, None

    candidates = sorted(glob.glob("/dev/video*"), key=lambda x: int(Path(x).name.removeprefix("video")))
    for device_path in candidates:
        cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap, device_path
        cap.release()
    return None, None


def load_roi_file(roi_json_path=None):
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


def draw_rois(frame, rois):
    overlay = frame.copy()
    for roi in rois:
        color = (0, 165, 255) if roi["label"] == 0 else (0, 255, 0)
        cv2.fillPoly(overlay, [roi["polygon"]], color)
        cv2.polylines(frame, [roi["polygon"]], True, color, 1)
    return cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)


def scale_rois_for_frame(rois, frame_w, frame_h):
    """Scale stored ROI points to the current frame resolution.

    rois: list of {"points": [(x,y),...], "label": int}
    returns: list of {"polygon": np.array([...], np.int32), "label": int}
    """
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
    # debug: print scaling info once
    try:
        print(f"[ROI] frame=({frame_w}x{frame_h}) max_roi=({max_x}x{max_y}) need_scale={need_scale} scale=({scale_x if need_scale else 1:.3f},{scale_y if need_scale else 1:.3f})")
    except Exception:
        pass
    return scaled


def detect_objects(frame):
    # TODO: YOLO 추가 시 여기에 실제 감지 로직을 넣으세요.
    # 현재는 아무 bbox도 반환하지 않습니다.
    return []


def camera_name(device_path):
    name_path = Path("/sys/class/video4linux") / Path(device_path).name / "name"
    try:
        return name_path.read_text().strip()
    except OSError:
        return Path(device_path).name


def build_status():
    selected_name = str(args.camera or "USB Camera")
    roi_path = args.roi_json or "None"
    return {
        "selected_name": selected_name,
        "stream_status": "연결됨",
        "yolo_enabled": bool(args.yolo),
        "roi_path": roi_path,
        "traffic_count": 4,
        "traffic_up": 2,
        "traffic_down": 2,
        "traffic_up_history": [0, 1, 2, 3, 2, 1, 2, 3, 4, 2, 3, 2, 4, 2, 3, 4, 3, 2, 4, 2],
        "traffic_down_history": [0, 2, 3, 2, 4, 3, 2, 4, 3, 2, 3, 4, 2, 3, 4, 3, 2, 3, 2, 4],
        "congestion_up": False,
        "congestion_down": False,
    }


def list_cameras():
    candidates = sorted(glob.glob("/dev/video*"), key=lambda x: int(Path(x).name.removeprefix("video")))
    items = []
    for index, device_path in enumerate(candidates):
        items.append({
            "index": index,
            "name": camera_name(device_path),
            "path": device_path,
            "selected": args.camera is None or str(args.camera) == str(index) or str(args.camera) == device_path,
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
    cap, device_path = find_camera(args.camera)
    if cap is None:
        raise RuntimeError("USB 카메라를 열 수 없습니다.")
    print(f"[CAM] opened device: {device_path}")

    if args.width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    rois = []
    if args.roi_use:
        rois = load_roi_file(args.roi_json)

    try:
        printed_size = False
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                continue

            # print camera capture size once
            if not printed_size:
                h, w = frame.shape[:2]
                print(f"[CAM] capture frame size: {w}x{h}")
                printed_size = True

            scaled_rois = []

            if args.roi_use and rois:
                # scale ROI definitions to current frame size before drawing/checking
                scaled_rois = scale_rois_for_frame(rois, frame.shape[1], frame.shape[0])
                frame = draw_rois(frame, scaled_rois)

            if args.roi_use and args.yolo:
                boxes = detect_objects(frame)
                for bbox in boxes:
                    if bbox_center_inside_roi(bbox, scaled_rois):
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

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
    parser.add_argument("--camera", default=None, help="카메라 소스 번호 또는 장치 경로")
    parser.add_argument("--host", default="0.0.0.0", help="웹 서버 호스트")
    parser.add_argument("--port", default=8000, type=int, help="웹 서버 포트")
    parser.add_argument("--width", type=int, default=1280, help="카메라 캡처 너비")
    parser.add_argument("--height", type=int, default=720, help="카메라 캡처 높이")
    parser.add_argument("--yolo", action="store_true", help="YOLO 바운딩박스 표시 (나중에 연결)" )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.run(host=args.host, port=args.port, threaded=True)
