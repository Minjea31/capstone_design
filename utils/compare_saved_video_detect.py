import argparse
import inspect
import os
import re
import sys
import time
from pathlib import Path


os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
CAPSTONE_ROOT = SCRIPT_DIR.parent
VENDORED_ULTRALYTICS_ROOT = CAPSTONE_ROOT / "yolov12" / "yolov12"
if VENDORED_ULTRALYTICS_ROOT.exists():
    sys.path.insert(0, str(VENDORED_ULTRALYTICS_ROOT))

import cv2
import torch
from ultralytics import YOLO


WINDOW_NAME = "YOLO Compare"
DEFAULT_VIDEO_DIR = SCRIPT_DIR / "saved_videos"
DEFAULT_LEFT_MODEL = CAPSTONE_ROOT / "model" / "clear_only_test_severe_raw.pt"
DEFAULT_RIGHT_MODEL = CAPSTONE_ROOT / "model" / "last_model.pt"
DEFAULT_DEVICE = "0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two YOLO models on saved AVI videos in one side-by-side panel."
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--left-model", type=Path, default=DEFAULT_LEFT_MODEL)
    parser.add_argument("--right-model", type=Path, default=DEFAULT_RIGHT_MODEL)
    parser.add_argument("--left-name", default="clear_only_test_severe_raw.pt")
    parser.add_argument("--right-name", default="last_model.pt")
    parser.add_argument("--conf", type=float, default=0.8)
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Detect only these class ids. Omit to detect every class.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Ultralytics device value, for example '0' for GPU or 'cpu'. Default: 0.",
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=640,
        help="Width of each model panel. Use 0 to keep the original video size.",
    )
    parser.add_argument("--wait-ms", type=int, default=1)
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=30.0,
        help="FPS to use when OpenCV cannot read FPS from the video.",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold used to decide whether left and right boxes match.",
    )
    return parser.parse_args()


def natural_key(path):
    parts = re.split(r"(\d+)", path.stem)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_videos(video_dir):
    videos = sorted(video_dir.glob("*.avi"), key=natural_key)
    if not videos:
        raise FileNotFoundError(f"AVI video files not found: {video_dir}")
    return videos


def validate_file(path, label):
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


def load_model(model_path):
    original_torch_load = torch.load
    torch_load_params = inspect.signature(original_torch_load).parameters

    def patched_torch_load(*args, **kwargs):
        if "weights_only" in torch_load_params:
            kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = patched_torch_load
        return YOLO(str(model_path))
    finally:
        torch.load = original_torch_load


def open_video(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Video open failed: {video_path}")
    return capture


def get_video_fps(capture, fallback_fps):
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        return fallback_fps
    return fps


def read_realtime_frame(capture, playback_started_at, frame_interval):
    elapsed_seconds = time.monotonic() - playback_started_at
    target_frame_index = int(elapsed_seconds / frame_interval)
    current_frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))

    while current_frame_index < target_frame_index:
        if not capture.grab():
            return False, None, current_frame_index
        current_frame_index += 1

    ok, frame = capture.read()
    shown_frame_index = max(0, int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
    return ok, frame, shown_frame_index


def wait_for_video_time(playback_started_at, shown_frame_index, frame_interval, min_wait_ms):
    target_time = playback_started_at + ((shown_frame_index + 1) * frame_interval)
    remaining_ms = int((target_time - time.monotonic()) * 1000)
    return max(min_wait_ms, remaining_ms)


def predict(model, frame, args):
    kwargs = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.classes is not None:
        kwargs["classes"] = args.classes
    if args.device:
        kwargs["device"] = args.device

    return model.predict(frame, **kwargs)[0]


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
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

    return annotated


def resize_panel(frame, panel_width):
    if panel_width <= 0 or frame.shape[1] == panel_width:
        return frame

    scale = panel_width / frame.shape[1]
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (panel_width, height), interpolation=cv2.INTER_AREA)


def add_header(frame, model_name, video_name, video_index, video_count, detect_count):
    header_h = 42
    header = frame.copy()
    cv2.rectangle(header, (0, 0), (header.shape[1], header_h), (20, 20, 20), -1)
    title = f"{model_name} | {video_name} ({video_index + 1}/{video_count}) | det: {detect_count}"
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


def make_panel(frame, left_model, right_model, args, video_path, video_index, video_count):
    left_result = predict(left_model, frame, args)
    right_result = predict(right_model, frame, args)

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

    left_frame = add_header(
        left_frame, args.left_name, video_path.name, video_index, video_count, left_count
    )
    right_frame = add_header(
        right_frame, args.right_name, video_path.name, video_index, video_count, right_count
    )

    return cv2.hconcat([left_frame, right_frame])


def print_status(video_path, video_index, video_count, fps):
    print(f"[video {video_index + 1}/{video_count}] {video_path}")
    print(f"fps: {fps:.2f}")
    print("keys: a=previous video, d=next video, q/ESC=quit")


def main():
    args = parse_args()
    args.video_dir = args.video_dir.resolve()
    args.left_model = args.left_model.resolve()
    args.right_model = args.right_model.resolve()

    validate_file(args.left_model, "left model")
    validate_file(args.right_model, "right model")
    videos = list_videos(args.video_dir)

    print("left model:", args.left_model)
    print("right model:", args.right_model)
    print("video dir:", args.video_dir)
    print("device:", args.device)
    validate_device(args.device)
    print("loading models...")
    left_model = load_model(args.left_model)
    right_model = load_model(args.right_model)

    video_index = 0
    capture = open_video(videos[video_index])
    fps = get_video_fps(capture, args.fallback_fps)
    frame_interval = 1.0 / fps
    playback_started_at = time.monotonic()
    print_status(videos[video_index], video_index, len(videos), fps)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame, shown_frame_index = read_realtime_frame(
                capture, playback_started_at, frame_interval
            )
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                playback_started_at = time.monotonic()
                continue

            panel = make_panel(
                frame, left_model, right_model, args, videos[video_index], video_index, len(videos)
            )
            cv2.imshow(WINDOW_NAME, panel)

            wait_ms = wait_for_video_time(
                playback_started_at, shown_frame_index, frame_interval, max(1, args.wait_ms)
            )
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("a"), ord("d")):
                capture.release()
                if key == ord("a"):
                    video_index = (video_index - 1) % len(videos)
                else:
                    video_index = (video_index + 1) % len(videos)
                capture = open_video(videos[video_index])
                fps = get_video_fps(capture, args.fallback_fps)
                frame_interval = 1.0 / fps
                playback_started_at = time.monotonic()
                print_status(videos[video_index], video_index, len(videos), fps)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
