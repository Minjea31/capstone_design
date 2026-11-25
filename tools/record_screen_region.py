import argparse
import inspect
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
CAPSTONE_ROOT = SCRIPT_DIR.parent
DEFAULT_COMPARE_SCRIPT = SCRIPT_DIR / "compare_saved_video_detect_g040.py"
DEFAULT_VIDEO = CAPSTONE_ROOT / "data" / "videos" / "weather" / "3.avi"
DEFAULT_OUTPUT_DIR = CAPSTONE_ROOT / "data" / "videos" / "weather"
DEFAULT_LEFT_MODEL = CAPSTONE_ROOT / "models" / "experiments" / "clear_only_test_severe_raw.pt"
DEFAULT_LEFT_YAML = None
DEFAULT_RIGHT_MODEL = CAPSTONE_ROOT / "models" / "experiments" / "g040_i048.pt"
DEFAULT_RIGHT_YAML = CAPSTONE_ROOT / "models" / "experiments" / "g040_i048.yaml"
VIDEO_CODECS = ("MJPG", "XVID")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render the same panel as compare_saved_video_detect_g040.py --conf 0.8 "
            "and save it directly as an AVI recording."
        )
    )
    parser.add_argument("--compare-script", type=Path, default=DEFAULT_COMPARE_SCRIPT)
    parser.add_argument("--video", type=Path, nargs="+", default=[DEFAULT_VIDEO])
    parser.add_argument("--conf", type=float, default=0.8)
    parser.add_argument("--left-conf", type=float, default=None)
    parser.add_argument("--right-conf", type=float, default=None)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--left-model", type=Path, default=DEFAULT_LEFT_MODEL)
    parser.add_argument("--left-yaml", type=Path, default=DEFAULT_LEFT_YAML)
    parser.add_argument("--right-model", type=Path, default=DEFAULT_RIGHT_MODEL)
    parser.add_argument("--right-yaml", type=Path, default=DEFAULT_RIGHT_YAML)
    parser.add_argument("--left-name", default="base_model")
    parser.add_argument("--right-name", default="g040_i048")
    parser.add_argument("--classes", type=int, nargs="*", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--graph-history", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .avi path. Only valid when one --video is provided.",
    )
    return parser.parse_args()


def validate_file(path, label):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def validate_optional_file(path, label):
    if path is None:
        return None
    return validate_file(path, label)


def load_compare_module(script_path):
    spec = importlib.util.spec_from_file_location("compare_saved_video_detect", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load compare script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_compare_model(compare, model_path, yaml_path):
    load_model_params = inspect.signature(compare.load_model).parameters
    if "yaml_path" in load_model_params or len(load_model_params) > 1:
        return compare.load_model(model_path, yaml_path)
    return compare.load_model(model_path)


def make_output_path(args, video_path):
    if args.output is not None:
        if len(args.video) != 1:
            raise ValueError("--output can only be used with one --video")
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conf_text = str(args.conf).replace(".", "")
        output_path = args.output_dir / f"compare_{video_path.stem}_conf{conf_text}_{timestamp}.avi"

    if output_path.suffix.lower() != ".avi":
        output_path = output_path.with_suffix(".avi")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.resolve()


def make_compare_args(args):
    return SimpleNamespace(
        conf=args.conf,
        left_conf=args.left_conf,
        right_conf=args.right_conf,
        classes=args.classes,
        imgsz=args.imgsz,
        device=args.device,
        panel_width=args.panel_width,
        fallback_fps=args.fallback_fps,
        match_iou=args.match_iou,
        graph_history=args.graph_history,
        left_name=args.left_name,
        right_name=args.right_name,
    )


def create_video_writer(output_path, fps, frame_size):
    width, height = frame_size
    for codec in VIDEO_CODECS:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"video codec: {codec}, format: .avi")
            return writer
        writer.release()

    raise RuntimeError(f"VideoWriter open failed: {output_path}")


def read_source_frame(capture, frame_index, source_fps, output_fps):
    source_index = int(frame_index * source_fps / output_fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
    ok, frame = capture.read()
    if ok:
        return frame

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError("failed to read a frame from source video")
    return frame


def render_recording(compare, compare_args, left_model, right_model, video_path, output_path, args):
    capture = compare.open_video(video_path)
    source_fps = compare.get_video_fps(capture, args.fallback_fps)
    frame_total = max(1, int(round(args.duration * args.fps)))
    diff_history = []
    speed_history = []
    make_panel_params = inspect.signature(compare.make_panel).parameters
    uses_speed_history = "speed_history" in make_panel_params
    writer = None

    print("source video:", video_path)
    print("source fps:", f"{source_fps:.2f}")
    print("recording duration:", f"{args.duration:.1f}s")
    print("output fps:", f"{args.fps:.2f}")
    print("saving video:", output_path)

    try:
        for frame_index in range(frame_total):
            frame = read_source_frame(capture, frame_index, source_fps, args.fps)
            panel_args = [
                frame,
                left_model,
                right_model,
                compare_args,
                video_path,
                0,
                1,
                diff_history,
            ]
            if uses_speed_history:
                panel_args.append(speed_history)
            panel = compare.make_panel(*panel_args)

            height, width = panel.shape[:2]
            if writer is None:
                writer = create_video_writer(output_path, args.fps, (width, height))
                print(f"writer frame size (H, W): ({height}, {width})")

            writer.write(panel)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    print("saved:", output_path)


def main():
    args = parse_args()
    if args.conf < 0:
        raise ValueError("--conf must be greater than or equal to 0")
    if args.duration <= 0:
        raise ValueError("--duration must be greater than 0")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0")

    args.compare_script = validate_file(args.compare_script, "compare script")
    args.left_model = validate_file(args.left_model, "left model")
    args.left_yaml = validate_optional_file(args.left_yaml, "left yaml")
    args.right_model = validate_file(args.right_model, "right model")
    args.right_yaml = validate_optional_file(args.right_yaml, "right yaml")
    args.video = [validate_file(path, "video") for path in args.video]
    args.output_dir = args.output_dir.resolve()

    compare = load_compare_module(args.compare_script)
    compare.validate_device(args.device)

    print("left model:", args.left_model)
    print("left yaml:", args.left_yaml or "-")
    print("right model:", args.right_model)
    print("right yaml:", args.right_yaml or "-")
    print("loading models...")
    left_model = load_compare_model(compare, args.left_model, args.left_yaml)
    right_model = load_compare_model(compare, args.right_model, args.right_yaml)
    compare_args = make_compare_args(args)

    for video_path in args.video:
        output_path = make_output_path(args, video_path)
        render_recording(compare, compare_args, left_model, right_model, video_path, output_path, args)


if __name__ == "__main__":
    main()
