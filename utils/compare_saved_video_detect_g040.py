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
DEFAULT_LEFT_YAML = None
DEFAULT_RIGHT_MODEL = CAPSTONE_ROOT / "model" / "g040_i048.pt"
DEFAULT_RIGHT_YAML = CAPSTONE_ROOT / "model" / "g040_i048.yaml"
DEFAULT_DEVICE = "0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two YOLO models on saved AVI videos in one side-by-side panel."
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--left-model", type=Path, default=DEFAULT_LEFT_MODEL)
    parser.add_argument("--left-yaml", type=Path, default=DEFAULT_LEFT_YAML)
    parser.add_argument("--right-model", type=Path, default=DEFAULT_RIGHT_MODEL)
    parser.add_argument("--right-yaml", type=Path, default=DEFAULT_RIGHT_YAML)
    parser.add_argument("--left-name", default="base_model")
    parser.add_argument("--right-name", default="g040_i048")
    parser.add_argument("--conf", type=float, default=0.8)
    parser.add_argument(
        "--left-conf",
        type=float,
        default=None,
        help="Confidence threshold for the left model. Defaults to --conf.",
    )
    parser.add_argument(
        "--right-conf",
        type=float,
        default=None,
        help="Confidence threshold for the right model. Defaults to --conf.",
    )
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
    parser.add_argument(
        "--graph-history",
        type=int,
        default=180,
        help="Number of recent frames to keep in the count-difference graph.",
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

    def patched_torch_load(*args, **kwargs):
        if "weights_only" in torch_load_params:
            kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = patched_torch_load
        if yaml_path is not None:
            return YOLO(str(yaml_path), task="detect").load(str(model_path))
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
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

    return annotated


def resize_panel(frame, panel_width):
    if panel_width <= 0 or frame.shape[1] == panel_width:
        return frame

    scale = panel_width / frame.shape[1]
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (panel_width, height), interpolation=cv2.INTER_AREA)


def add_header(frame, model_name, video_name, video_index, video_count):
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
    title = f"{model_name} | {video_name} ({video_index + 1}/{video_count})"
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
    cv2.putText(
        output,
        f"+ {right_name}",
        (12, graph_y + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        right_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"- {left_name}",
        (12, graph_y + graph_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        left_color,
        1,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        output,
        (graph_x, graph_y),
        (graph_x + graph_w, graph_y + graph_h),
        (70, 70, 70),
        1,
    )
    for index in range(1, 4):
        grid_y = graph_y + int(graph_h * index / 4)
        cv2.line(output, (graph_x, grid_y), (graph_x + graph_w, grid_y), grid_color, 1)
    cv2.line(output, (graph_x, zero_y), (graph_x + graph_w, zero_y), zero_color, 1)
    cv2.putText(
        output,
        "0",
        (graph_x - 18, zero_y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        zero_color,
        1,
        cv2.LINE_AA,
    )

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
        cv2.putText(
            output,
            speed_summary,
            (12, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            speed_color,
            1,
            cv2.LINE_AA,
        )
    return output


def make_panel(
    frame,
    left_model,
    right_model,
    args,
    video_path,
    video_index,
    video_count,
    diff_history,
    speed_history,
):
    left_conf = args.conf if args.left_conf is None else args.left_conf
    right_conf = args.conf if args.right_conf is None else args.right_conf
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

    left_frame = add_header(
        left_frame,
        args.left_name,
        video_path.name,
        video_index,
        video_count,
    )
    right_frame = add_header(
        right_frame,
        args.right_name,
        video_path.name,
        video_index,
        video_count,
    )

    diff_history.append(right_count - left_count)
    if len(diff_history) > max(2, args.graph_history):
        del diff_history[: len(diff_history) - max(2, args.graph_history)]
    speed_history.append((left_elapsed_ms, right_elapsed_ms))
    if len(speed_history) > max(2, args.graph_history):
        del speed_history[: len(speed_history) - max(2, args.graph_history)]
    panel = cv2.hconcat([left_frame, right_frame])
    return add_difference_graph(panel, diff_history, speed_history, args.left_name, args.right_name)


def print_status(video_path, video_index, video_count, fps):
    print(f"[video {video_index + 1}/{video_count}] {video_path}")
    print(f"fps: {fps:.2f}")
    print("keys: a=previous video, d=next video, q/ESC=quit")


def main():
    args = parse_args()
    args.video_dir = args.video_dir.resolve()
    args.left_model = args.left_model.resolve()
    if args.left_yaml is not None:
        args.left_yaml = args.left_yaml.resolve()
    args.right_model = args.right_model.resolve()
    if args.right_yaml is not None:
        args.right_yaml = args.right_yaml.resolve()

    validate_file(args.left_model, "left model")
    validate_optional_file(args.left_yaml, "left yaml")
    validate_file(args.right_model, "right model")
    validate_optional_file(args.right_yaml, "right yaml")
    videos = list_videos(args.video_dir)

    print("left model:", args.left_model)
    print("left yaml:", args.left_yaml or "-")
    print("right model:", args.right_model)
    print("right yaml:", args.right_yaml or "-")
    print("video dir:", args.video_dir)
    print("device:", args.device)
    print("left conf:", args.conf if args.left_conf is None else args.left_conf)
    print("right conf:", args.conf if args.right_conf is None else args.right_conf)
    validate_device(args.device)
    print("loading models...")
    left_model = load_model(args.left_model, args.left_yaml)
    right_model = load_model(args.right_model, args.right_yaml)

    video_index = 0
    capture = open_video(videos[video_index])
    fps = get_video_fps(capture, args.fallback_fps)
    frame_interval = 1.0 / fps
    playback_started_at = time.monotonic()
    diff_history = []
    speed_history = []
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
                diff_history = []
                speed_history = []
                continue

            panel = make_panel(
                frame,
                left_model,
                right_model,
                args,
                videos[video_index],
                video_index,
                len(videos),
                diff_history,
                speed_history,
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
                diff_history = []
                speed_history = []
                print_status(videos[video_index], video_index, len(videos), fps)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
