import glob
import os
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Error: OpenCV is not installed.")
    print("Install it with: python3 -m pip install opencv-python")
    raise SystemExit(1)

EXTERNAL_CAMERA = (
    "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0"
)


def open_camera(camera_source):
    cap = cv2.VideoCapture(camera_source, cv2.CAP_V4L2)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            return cap

    cap.release()
    return None


def camera_name(device_path):
    device_name = Path(os.path.realpath(device_path)).name
    name_path = Path("/sys/class/video4linux") / device_name / "name"

    try:
        return name_path.read_text().strip()
    except OSError:
        return device_name


def camera_candidates():
    devices = glob.glob("/dev/video*")

    def sort_key(device_path):
        name = camera_name(device_path).lower()
        index = int(Path(device_path).name.removeprefix("video"))
        is_usb_camera = any(
            keyword in name for keyword in ("c920", "logitech", "usb")
        )
        return (not is_usb_camera, index)

    return sorted(devices, key=sort_key)


def open_named_camera(name_query):
    for device_path in camera_candidates():
        if name_query.lower() not in camera_name(device_path).lower():
            continue

        cap = open_camera(device_path)
        if cap is not None:
            return cap, device_path

    return None, None


def find_camera():
    candidates = camera_candidates()
    for device_path in candidates:
        cap = open_camera(device_path)
        if cap is not None:
            print(f"Opened {camera_name(device_path)} at {device_path}")
            return cap, device_path

    if not candidates:
        print("No /dev/video* devices are available in this environment.")

    return None, None


def resolve_camera(camera_source):
    if camera_source is None:
        if os.path.exists(EXTERNAL_CAMERA):
            return open_camera(EXTERNAL_CAMERA), EXTERNAL_CAMERA
        return find_camera()

    if camera_source.isdigit():
        device_path = f"/dev/video{camera_source}"
        return open_camera(device_path), device_path

    if os.path.exists(camera_source):
        return open_camera(camera_source), camera_source

    return open_named_camera(camera_source)


def main():
    requested_source = sys.argv[1] if len(sys.argv) > 1 else None
    cap, camera_source = resolve_camera(requested_source)

    if cap is None:
        source_description = requested_source or "any camera"
        print(f"Error: could not open {source_description}.")
        print("Available camera devices:")
        for device_path in camera_candidates():
            print(f"  {device_path}: {camera_name(device_path)}")
        return

    window_name = f"{camera_name(camera_source)} ({camera_source})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    is_fullscreen = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: could not read frame from camera.")
                break

            cv2.putText(
                frame,
                f"EXTERNAL USB: {camera_name(camera_source)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("f"):
                is_fullscreen = not is_fullscreen
                window_mode = (
                    cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                )
                cv2.setWindowProperty(
                    window_name, cv2.WND_PROP_FULLSCREEN, window_mode
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
