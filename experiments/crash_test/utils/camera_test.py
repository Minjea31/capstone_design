import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2

from utils.video_source import get_video_delay_ms, open_video_capture, resolve_video_path


SAVE_INTERVAL_SECONDS = 0.5
OUTPUT_DIR = Path("saved_frames")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Preview a saved video and save frames periodically")
    parser.add_argument("--video", default="", help="입력 비디오 경로; 생략하면 saved_videos의 최신 파일")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = resolve_video_path(args.video)
    cap = open_video_capture(video_path)
    delay_ms = get_video_delay_ms(cap)

    print(f"비디오 실행 중: {video_path}")
    print("0.5초마다 프레임을 저장합니다. 종료하려면 q를 누르세요.")

    last_saved_at = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("비디오가 끝났습니다.")
            break

        cv2.imshow("Saved Video", frame)

        now = time.monotonic()
        if now - last_saved_at >= SAVE_INTERVAL_SECONDS:
            filename = OUTPUT_DIR / f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"저장 완료: {filename}")
            last_saved_at = now

        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
