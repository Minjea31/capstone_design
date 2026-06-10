import time
from datetime import datetime
from pathlib import Path

import cv2


EXTERNAL_CAMERA = (
    "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0"
)
OUTPUT_ROOT = Path("recordings")
FRAME_INTERVAL_SECONDS = 0.2
DEFAULT_FPS = 30.0
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS = 30.0


def configure_camera(cap):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera mode: {width}x{height} at {fps:.1f} FPS")

    if (width, height) != (CAPTURE_WIDTH, CAPTURE_HEIGHT):
        print(
            f"Warning: requested {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}, "
            f"but the camera selected {width}x{height}."
        )

    return width, height, fps


def main():
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_ROOT / session_name
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    camera_path = Path(EXTERNAL_CAMERA).resolve()
    camera_index = int(camera_path.name.removeprefix("video"))
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera: {EXTERNAL_CAMERA} -> {camera_path}"
        )

    width, height, fps = configure_camera(cap)
    if fps <= 0:
        fps = DEFAULT_FPS

    video_path = session_dir / "recording.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not create video: {video_path}")

    window_name = "C920 Recording - q or Esc to stop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    start_time = time.monotonic()
    next_frame_time = start_time
    saved_frame_count = 0

    print(f"Recording video: {video_path.resolve()}")
    print(f"Saving frames every 0.2 seconds: {frames_dir.resolve()}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read a frame from the camera.")
                break

            writer.write(frame)

            now = time.monotonic()
            if now >= next_frame_time:
                elapsed = now - start_time
                frame_path = frames_dir / (
                    f"frame_{saved_frame_count:06d}_{elapsed:010.3f}s.jpg"
                )
                cv2.imwrite(str(frame_path), frame)
                saved_frame_count += 1

                while next_frame_time <= now:
                    next_frame_time += FRAME_INTERVAL_SECONDS

            elapsed = now - start_time
            display_frame = frame.copy()
            cv2.putText(
                display_frame,
                f"REC {elapsed:0.1f}s | saved frames: {saved_frame_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Finished: {video_path.resolve()}")
    print(f"Saved {saved_frame_count} frames.")


if __name__ == "__main__":
    main()
