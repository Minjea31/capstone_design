import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent

VENDORED_ULTRALYTICS_ROOT = PROJECT_ROOT / "yolov12" / "yolov12"
if VENDORED_ULTRALYTICS_ROOT.exists():
    sys.path.insert(0, str(VENDORED_ULTRALYTICS_ROOT))

import cv2
import torch
from ultralytics import YOLO

from utils.video_source import get_video_delay_ms, open_video_capture, resolve_video_path

MODEL_PATH = MODULE_DIR / "crash_pruned.pt"
MODEL_YAML_PATH = MODULE_DIR / "crash_pruned.yaml"
WINDOW_NAME = "Crash Detection"
CONFIDENCE_THRESHOLD = 0.7


def parse_args():
    parser = argparse.ArgumentParser(description="Run crash detection on a saved video")
    parser.add_argument("--video", default="", help="입력 비디오 경로; 생략하면 saved_videos의 최신 파일")
    parser.add_argument("--model", default=str(MODEL_PATH), help="YOLO .pt 모델 경로")
    parser.add_argument("--model-yaml", default=str(MODEL_YAML_PATH), help="프루닝 모델 구조 YAML 경로")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="YOLO confidence threshold")
    return parser.parse_args()


def load_model(model_path, model_yaml_path):
    weights_path = Path(model_path)
    yaml_path = Path(model_yaml_path) if model_yaml_path else None

    if not weights_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {weights_path}")
    if yaml_path is not None and not yaml_path.exists():
        raise FileNotFoundError(f"모델 구조 YAML 파일을 찾을 수 없습니다: {yaml_path}")

    original_torch_load = torch.load

    def patched_torch_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    try:
        torch.load = patched_torch_load
        if yaml_path is not None:
            return YOLO(str(yaml_path), task="detect").load(str(weights_path))
        return YOLO(str(weights_path), task="detect")
    finally:
        torch.load = original_torch_load


def main():
    args = parse_args()
    model = load_model(args.model, args.model_yaml)
    video_path = resolve_video_path(args.video)
    cap = open_video_capture(video_path)
    delay_ms = get_video_delay_ms(cap)

    print(f"비디오 추론을 시작합니다: {video_path}")
    print("종료하려면 q를 누르세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("비디오가 끝났습니다.")
            break

        results = model.predict(frame, conf=args.conf, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow(WINDOW_NAME, annotated_frame)

        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
