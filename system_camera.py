import cv2

WINDOW_NAME = "System Camera"
CAMERA_INDEX = 0


def main():
    capture = cv2.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        raise RuntimeError(f"카메라를 열지 못했습니다: index={CAMERA_INDEX}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("프레임을 읽지 못했습니다.")
                break

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
