from datetime import datetime
from pathlib import Path

import cv2

CAMERA_INDEX = 2
OUTPUT_DIR = Path("saved_videos")
OUTPUT_FPS = 20.0
VIDEO_CODECS = ("MJPG", "XVID")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_video_writer(frame, output_path_base: Path):
    height, width = frame.shape[:2]
    print(f"writer frame size (H, W): ({height}, {width})")

    for codec in VIDEO_CODECS:
        output_path = output_path_base.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, OUTPUT_FPS, (width, height))
        if writer.isOpened():
            print(f"video codec: {codec}, format: .avi")
            return writer, output_path
        writer.release()

    raise RuntimeError(
        "영상 저장기 생성 실패: "
        f".avi, codecs={', '.join(VIDEO_CODECS)}"
    )


def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"카메라를 열 수 없습니다: index={CAMERA_INDEX}")
        raise SystemExit(1)

    output_path_base = OUTPUT_DIR / f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = None
    output_path = None

    print("카메라 실행 중...")
    print("종료하려면 d를 누르세요.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            if writer is None:
                writer, output_path = create_video_writer(frame, output_path_base)
                print(f"saving video: {output_path}")

            writer.write(frame)
            cv2.imshow("Camera Video Save", frame)

            if cv2.waitKey(1) & 0xFF == ord("d"):
                break
    finally:
        if writer is not None:
            writer.release()
            print(f"saved: {output_path}")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
