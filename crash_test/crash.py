from pathlib import Path

import cv2
from ultralytics import YOLO


CAMERA_INDEX = 2
MODEL_PATH = Path(__file__).with_name("crash.pt")
WINDOW_NAME = "Crash Detection"
CONFIDENCE_THRESHOLD = 0.7


def main():
    if not MODEL_PATH.exists():
        print(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        raise SystemExit(1)

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"카메라를 열 수 없습니다: index={CAMERA_INDEX}")
        raise SystemExit(1)

    print("실시간 추론을 시작합니다. 종료하려면 q를 누르세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow(WINDOW_NAME, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
