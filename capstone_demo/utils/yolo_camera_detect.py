#!/usr/bin/env python3
import argparse
import glob
import os
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "model" / "raw_model.pt"
DEFAULT_EXTERNAL_CAMERA = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0"
USB_CAMERA_KEYWORDS = ("c920", "logitech", "pro webcam", "usb")
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def camera_name(device_path):
    device_name = Path(os.path.realpath(device_path)).name
    name_path = Path("/sys/class/video4linux") / device_name / "name"
    try:
        return name_path.read_text().strip()
    except OSError:
        return device_name


def camera_candidates():
    devices = set(glob.glob("/dev/video*"))
    if os.path.exists(DEFAULT_EXTERNAL_CAMERA):
        devices.add(DEFAULT_EXTERNAL_CAMERA)

    def sort_key(device_path):
        name = camera_name(device_path).lower()
        resolved = Path(os.path.realpath(device_path)).name
        index = int(resolved.removeprefix("video")) if resolved.startswith("video") else 999
        is_usb_camera = any(keyword in name for keyword in USB_CAMERA_KEYWORDS)
        is_by_id = str(device_path).startswith("/dev/v4l/by-id/")
        return (not is_by_id, not is_usb_camera, index)

    return sorted(devices, key=sort_key)


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

    if camera_source.isdigit() or os.path.exists(camera_source):
        return open_camera(camera_source, width, height, fps), camera_source

    return open_named_camera(camera_source, width, height, fps)


def load_model(model_path, device):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics가 없습니다. python3 -m pip install ultralytics 로 설치하세요.") from exc

    path = Path(model_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        raise SystemExit(f"모델 파일을 찾지 못했습니다: {path}")

    model = YOLO(str(path), task="detect")
    if device:
        model.to(device)
    print(f"YOLO model: {path}")
    print(f"classes: {getattr(model, 'names', {})}")
    return model


def draw_boxes(frame, result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return frame

    names = result.names or {}
    xyxy_list = boxes.xyxy.int().cpu().tolist()
    cls_list = boxes.cls.int().cpu().tolist()
    conf_list = boxes.conf.cpu().tolist()

    for xyxy, cls_id, conf in zip(xyxy_list, cls_list, conf_list):
        x1, y1, x2, y2 = [int(value) for value in xyxy]
        label = f"{names.get(int(cls_id), cls_id)} {float(conf):.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame


def parse_args():
    parser = argparse.ArgumentParser(description="USB 카메라 YOLO 단독 detect 뷰어")
    parser.add_argument("--camera", default=None, help="카메라 번호, 경로, 또는 이름. 예: 2, /dev/video2, c920")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="YOLO .pt 모델 경로")
    parser.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    parser.add_argument("--classes", default="0", help="감지 클래스. 예: 0 또는 0,1. 전체는 all")
    parser.add_argument("--device", default=None, help="추론 장치. 예: cuda:0 또는 cpu")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="카메라 요청 너비")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="카메라 요청 높이")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="카메라 요청 FPS")
    return parser.parse_args()


def parse_classes(value):
    value = str(value).strip()
    if not value or value.lower() in {"all", "none"}:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    args = parse_args()
    cap, camera_source = resolve_camera(args.camera, args.width, args.height, args.fps)
    if cap is None:
        requested = args.camera or "USB camera"
        print(f"카메라를 열지 못했습니다: {requested}")
        print("사용 가능한 장치:")
        for device_path in camera_candidates():
            print(f"  {device_path}: {camera_name(device_path)}")
        raise SystemExit(1)

    model = load_model(args.model, args.device)
    classes = parse_classes(args.classes)
    window_name = f"YOLO Camera Detect - {camera_name(camera_source)} ({camera_source})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"camera: {camera_name(camera_source)} ({camera_source})")
    print(f"mode: {width}x{height} @ {fps:.1f} FPS")
    print("quit: q 또는 Esc")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("프레임을 읽지 못했습니다.")
                break

            results = model.predict(frame, conf=args.conf, classes=classes, device=args.device, verbose=False)
            display = draw_boxes(frame.copy(), results[0])
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
