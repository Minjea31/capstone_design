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
from ultralytics import YOLO

from utils.video_source import get_video_delay_ms, open_video_capture, resolve_video_path

MODEL_PATH = MODULE_DIR / "crash.pt"
WINDOW_NAME = "Crash Detection"
CONFIDENCE_THRESHOLD = 0.7


def parse_args():
    parser = argparse.ArgumentParser(description="Run crash detection on a saved video")
    parser.add_argument("--video", default="", help="입력 비디오 경로; 생략하면 saved_videos의 최신 파일")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="YOLO confidence threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    if not MODEL_PATH.exists():
        print(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        raise SystemExit(1)

    model = YOLO(str(MODEL_PATH))
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
