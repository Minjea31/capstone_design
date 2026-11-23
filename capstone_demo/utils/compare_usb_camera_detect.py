#!/usr/bin/env python3
import argparse
import glob
import inspect
import os
from pathlib import Path
import sys
import time


os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = DEMO_ROOT.parent
VENDORED_ULTRALYTICS_ROOT = PROJECT_ROOT / "yolov12" / "yolov12"
if VENDORED_ULTRALYTICS_ROOT.exists():
    sys.path.insert(0, str(VENDORED_ULTRALYTICS_ROOT))

import cv2
import torch
from ultralytics import YOLO


WINDOW_NAME = "YOLO USB Camera Compare"
DEFAULT_LEFT_MODEL = DEMO_ROOT / "model" / "raw_model.pt"
DEFAULT_LEFT_YAML = None
DEFAULT_RIGHT_MODEL = DEMO_ROOT / "model" / "pruned.pt"
DEFAULT_RIGHT_YAML = DEMO_ROOT / "model" / "pruned.yaml"
DEFAULT_EXTERNAL_CAMERA = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0"
USB_CAMERA_KEYWORDS = ("c920", "logitech", "pro webcam", "usb")
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30


def camera_name(device_path):
    device_name = Path(os.path.realpath(str(device_path))).name
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

    camera_source = str(camera_source)
    if camera_source.isdigit() or os.path.exists(camera_source):
        return open_camera(camera_source, width, height, fps), camera_source

    return open_named_camera(camera_source, width, height, fps)


def validate_file(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


def validate_optional_file(path, label):
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


def validate_device(device):
    if device is None:
        return

    normalized = str(device).strip().lower()
    if normalized in ("", "none", "cpu"):
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"GPU device was requested (--device {device}), but PyTorch cannot use CUDA "
            "in this environment. Use a CUDA-enabled Python environment or pass --device cpu."
        )

    if normalized.isdigit() and int(normalized) >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU device index is out of range: {device}. "
            f"Available CUDA device count: {torch.cuda.device_count()}"
        )


def load_model(model_path, yaml_path=None):
    original_torch_load = torch.load
    torch_load_params = inspect.signature(original_torch_load).parameters

    def patched_torch_load(*load_args, **load_kwargs):
        if "weights_only" in torch_load_params:
            load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    try:
        torch.load = patched_torch_load
        if yaml_path is not None:
            return YOLO(str(yaml_path), task="detect").load(str(model_path))
        return YOLO(str(model_path), task="detect")
    finally:
        torch.load = original_torch_load


def parse_classes(value):
    value = str(value).strip()
    if not value or value.lower() in {"all", "none"}:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def predict(model, frame, args, confidence):
    kwargs = {
        "conf": confidence,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.classes is not None:
        kwargs["classes"] = args.classes
    if args.device:
        kwargs["device"] = args.device

    result = model.predict(frame, **kwargs)[0]
    speed = getattr(result, "speed", {}) or {}
    inference_ms = float(speed.get("inference", 0.0) or 0.0)
    return result, inference_ms


def extract_boxes(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    return boxes.xyxy.int().cpu().tolist()


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area == 0:
        return 0.0

    return inter_area / union_area


def unmatched_box_indices(source_boxes, target_boxes, match_iou):
    unmatched = []
    for source_index, source_box in enumerate(source_boxes):
        matched = any(box_iou(source_box, target_box) >= match_iou for target_box in target_boxes)
        if not matched:
            unmatched.append(source_index)
    return unmatched


def draw_boxes(frame, boxes, color, thickness):
    annotated = frame.copy()
    for xyxy in boxes:
        x1, y1, x2, y2 = [int(value) for value in xyxy]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
    return annotated


def resize_panel(frame, panel_width):
    if panel_width <= 0 or frame.shape[1] == panel_width:
        return frame

    scale = panel_width / frame.shape[1]
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (panel_width, height), interpolation=cv2.INTER_AREA)


def add_header(frame, title):
    header_h = 42
    header = cv2.copyMakeBorder(
        frame,
        header_h,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )
    cv2.putText(
        header,
        title,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return header


def add_difference_graph(panel, diff_history, speed_history, left_name, right_name):
    header_h = 128
    output = cv2.copyMakeBorder(
        panel,
        header_h,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(14, 14, 14),
    )
    width = output.shape[1]
    graph_x = 72
    graph_y = 20
    graph_w = max(160, width - 96)
    graph_h = 68
    zero_y = graph_y + graph_h // 2
    max_abs = max(1, max((abs(value) for value in diff_history), default=0))
    left_color = (0, 255, 0)
    right_color = (0, 180, 255)
    zero_color = (185, 185, 185)
    grid_color = (52, 52, 52)
    speed_color = (210, 210, 210)

    cv2.putText(output, f"+ {right_name}", (12, graph_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, right_color, 1, cv2.LINE_AA)
    cv2.putText(output, f"- {left_name}", (12, graph_y + graph_h), cv2.FONT_HERSHEY_SIMPLEX, 0.48, left_color, 1, cv2.LINE_AA)
    cv2.rectangle(output, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (70, 70, 70), 1)
    for index in range(1, 4):
        grid_y = graph_y + int(graph_h * index / 4)
        cv2.line(output, (graph_x, grid_y), (graph_x + graph_w, grid_y), grid_color, 1)
    cv2.line(output, (graph_x, zero_y), (graph_x + graph_w, zero_y), zero_color, 1)
    cv2.putText(output, "0", (graph_x - 18, zero_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, zero_color, 1, cv2.LINE_AA)

    if diff_history:
        usable_w = max(1, graph_w - 8)
        usable_h = max(1, graph_h // 2 - 8)
        points = []
        for index, value in enumerate(diff_history):
            x = graph_x + 4 + int(usable_w * index / max(1, len(diff_history) - 1))
            y = zero_y - int((value / max_abs) * usable_h)
            points.append((x, y, value))
        for (x1, y1, _value1), (x2, y2, value2) in zip(points, points[1:]):
            color = right_color if value2 >= 0 else left_color
            cv2.line(output, (x1, y1), (x2, y2), color, 2)
        x, y, value = points[-1]
        point_color = right_color if value >= 0 else left_color
        cv2.circle(output, (x, y), 4, point_color, -1)

    if speed_history:
        left_ms = sum(item[0] for item in speed_history) / len(speed_history)
        right_ms = sum(item[1] for item in speed_history) / len(speed_history)
        if left_ms < right_ms:
            speed_summary = f"inference avg: {left_name} {left_ms:.1f}ms | {right_name} {right_ms:.1f}ms | faster {left_name} {right_ms - left_ms:.1f}ms"
            speed_color = left_color
        elif right_ms < left_ms:
            speed_summary = f"inference avg: {left_name} {left_ms:.1f}ms | {right_name} {right_ms:.1f}ms | faster {right_name} {left_ms - right_ms:.1f}ms"
            speed_color = right_color
        else:
            speed_summary = f"inference avg: {left_name} {left_ms:.1f}ms | {right_name} {right_ms:.1f}ms | same"
        cv2.putText(output, speed_summary, (12, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.55, speed_color, 1, cv2.LINE_AA)
    return output


def make_panel(frame, left_model, right_model, args, camera_label, diff_history, speed_history):
    left_conf = args.conf if args.left_conf is None else args.left_conf
    right_conf = max(0.0, args.conf) if args.right_conf is None else args.right_conf
    left_result, left_elapsed_ms = predict(left_model, frame, args, left_conf)
    right_result, right_elapsed_ms = predict(right_model, frame, args, right_conf)

    left_boxes = extract_boxes(left_result)
    right_boxes = extract_boxes(right_result)
    left_only_indices = unmatched_box_indices(left_boxes, right_boxes, args.match_iou)
    right_only_indices = unmatched_box_indices(right_boxes, left_boxes, args.match_iou)
    left_only_boxes = [left_boxes[index] for index in left_only_indices]
    right_only_boxes = [right_boxes[index] for index in right_only_indices]

    left_frame = draw_boxes(frame, left_boxes, (0, 255, 0), 1)
    left_frame = draw_boxes(left_frame, left_only_boxes, (255, 0, 0), 2)
    right_frame = draw_boxes(frame, right_boxes, (0, 180, 255), 1)
    right_frame = draw_boxes(right_frame, right_only_boxes, (0, 0, 255), 2)
    left_count = len(left_boxes)
    right_count = len(right_boxes)

    left_frame = resize_panel(left_frame, args.panel_width)
    right_frame = resize_panel(right_frame, args.panel_width)
    left_frame = add_header(left_frame, args.left_name)
    right_frame = add_header(right_frame, args.right_name)

    diff_history.append(right_count - left_count)
    if len(diff_history) > max(2, args.graph_history):
        del diff_history[: len(diff_history) - max(2, args.graph_history)]
    speed_history.append((left_elapsed_ms, right_elapsed_ms))
    if len(speed_history) > max(2, args.graph_history):
        del speed_history[: len(speed_history) - max(2, args.graph_history)]

    panel = cv2.hconcat([left_frame, right_frame])
    return add_difference_graph(panel, diff_history, speed_history, args.left_name, args.right_name)


def parse_args():
    parser = argparse.ArgumentParser(description="USB 카메라에서 raw/pruned YOLO 모델을 한 화면에서 비교합니다.")
    parser.add_argument(
        "--camera",
        default=DEFAULT_EXTERNAL_CAMERA,
        help=(
            "카메라 번호, 경로, 또는 이름 "
            f"(기본값: Logitech C920의 {DEFAULT_EXTERNAL_CAMERA})"
        ),
    )
    parser.add_argument("--left-model", type=Path, default=DEFAULT_LEFT_MODEL)
    parser.add_argument("--left-yaml", type=Path, default=DEFAULT_LEFT_YAML)
    parser.add_argument("--right-model", type=Path, default=DEFAULT_RIGHT_MODEL)
    parser.add_argument("--right-yaml", type=Path, default=DEFAULT_RIGHT_YAML)
    parser.add_argument("--left-name", default="left:raw")
    parser.add_argument("--right-name", default="right:ours model")
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--left-conf", type=float, default=None, help="왼쪽 모델 confidence.")
    parser.add_argument("--right-conf", type=float, default=None, help="오른쪽 모델 confidence.")
    parser.add_argument("--classes", default="0", help="감지 클래스. 예: 0 또는 0,1. 전체는 all")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Ultralytics device 값. 예: 0 또는 cpu.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="카메라 요청 너비")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="카메라 요청 높이")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="카메라 요청 FPS")
    parser.add_argument("--panel-width", type=int, default=640, help="각 모델 패널 너비. 0이면 원본 크기 유지")
    parser.add_argument("--wait-ms", type=int, default=1)
    parser.add_argument("--match-iou", type=float, default=0.5, help="두 모델 bbox 매칭 IoU 기준")
    parser.add_argument("--graph-history", type=int, default=180, help="카운트 차이 그래프에 유지할 최근 프레임 수")
    return parser.parse_args()


def main():
    args = parse_args()
    args.left_model = args.left_model.resolve()
    if args.left_yaml is not None:
        args.left_yaml = args.left_yaml.resolve()
    args.right_model = args.right_model.resolve()
    if args.right_yaml is not None:
        args.right_yaml = args.right_yaml.resolve()
    args.classes = parse_classes(args.classes)

    validate_file(args.left_model, "left model")
    validate_optional_file(args.left_yaml, "left yaml")
    validate_file(args.right_model, "right model")
    validate_optional_file(args.right_yaml, "right yaml")

    print("left model:", args.left_model)
    print("left yaml:", args.left_yaml or "-")
    print("right model:", args.right_model)
    print("right yaml:", args.right_yaml or "-")
    print("device:", args.device)
    validate_device(args.device)

    cap, camera_source = resolve_camera(args.camera, args.width, args.height, args.fps)
    if cap is None:
        requested = args.camera or "USB camera"
        print(f"카메라를 열지 못했습니다: {requested}")
        print("사용 가능한 장치:")
        for device_path in camera_candidates():
            print(f"  {device_path}: {camera_name(device_path)}")
        raise SystemExit(1)

    camera_label = f"{camera_name(camera_source)} ({camera_source})"
    print("loading models...")
    left_model = load_model(args.left_model, args.left_yaml)
    right_model = load_model(args.right_model, args.right_yaml)
    print(f"camera: {camera_label}")
    print(
        f"mode: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
        f"{cap.get(cv2.CAP_PROP_FPS):.1f} FPS"
    )
    print("keys: q/ESC=quit")

    diff_history = []
    speed_history = []
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("프레임을 읽지 못했습니다.")
                break

            panel = make_panel(frame, left_model, right_model, args, camera_label, diff_history, speed_history)
            cv2.imshow(WINDOW_NAME, panel)

            key = cv2.waitKey(max(1, args.wait_ms)) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
