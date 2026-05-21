from datetime import datetime
from pathlib import Path
import time

import cv2

CAMERA_INDEX = 2
SAVE_INTERVAL_SECONDS = 0.5
OUTPUT_DIR = Path("saved_frames")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"카메라를 열 수 없습니다: index={CAMERA_INDEX}")
    raise SystemExit(1)

print("카메라 실행 중... 0.5초마다 프레임을 저장합니다. 종료하려면 q를 누르세요.")

last_saved_at = 0.0

while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    cv2.imshow("Camera", frame)

    now = time.monotonic()
    if now - last_saved_at >= SAVE_INTERVAL_SECONDS:
        filename = OUTPUT_DIR / f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"저장 완료: {filename}")
        last_saved_at = now

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
