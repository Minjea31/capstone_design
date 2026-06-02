# Capstone Design CCTV Detection

고속도로 CCTV 영상을 수신하고 Pruned YOLO 모델로 차량을 탐지한 뒤, WebRTC 기반 웹 대시보드에서 영상과 교통량 정보를 확인하는 프로젝트입니다.

## Web Dashboard

배포된 웹사이트:

```text
https://detection-demo-ho836p2rk-capstone11.vercel.app/
```

## Main Pipeline

현재 실행 흐름은 아래와 같습니다.

```text
CCTV 원본 스트림 수신
-> Pruned YOLO 차량 탐지
-> ROI 기반 상행/하행 교통량 계산
-> 영상은 WebRTC로 송출
-> 탐지 결과/상태/통계는 Railway WebSocket으로 전송
-> Vercel 웹 대시보드에서 표시
```

## Run

YOLO PC에서 `main/` 폴더로 이동한 뒤 실행합니다.

```bash
cd /home/user/capstone_design-seperate/main

DETECTION_BACKEND_URL=https://detection-demo-production.up.railway.app \
DETECTION_TRANSPORT=websocket \
DETECTION_WEBRTC_ENABLED=1 \
DETECTION_WEBRTC_FPS=20 \
DETECTION_WEBRTC_FRAME_WIDTH=640 \
DETECTION_SEND_FRAME_JPEG=0 \
DETECTION_SEND_INTERVAL=0.7 \
DETECTION_FRAME_WIDTH=640 \
DETECTION_JPEG_QUALITY=40 \
DETECTION_CONTROL_ENABLED=1 \
DETECTION_CONTROL_INTERVAL=3 \
DETECTION_CONTROL_TIMEOUT=2 \
DETECTION_PROCESS_TIMING_LOG=1.0 \
DETECTION_LOG_SEND=1 \
python3 main.py --use-roi
```

## Key Options

- `DETECTION_BACKEND_URL`: Railway 백엔드 주소
- `DETECTION_TRANSPORT=websocket`: 탐지 결과를 WebSocket으로 전송
- `DETECTION_WEBRTC_ENABLED=1`: 영상을 WebRTC로 송출
- `DETECTION_WEBRTC_FPS=20`: WebRTC 영상 목표 FPS
- `DETECTION_SEND_FRAME_JPEG=0`: Railway로 JPEG 프레임 전송 비활성화
- `DETECTION_SEND_INTERVAL=0.7`: 탐지 결과/상태 전송 간격
- `DETECTION_CONTROL_ENABLED=1`: 웹에서 CCTV 전환 가능
- `DETECTION_PROCESS_TIMING_LOG=1.0`: 1초 이상 걸린 프레임 처리 시간 로그 출력

## Project Structure

```text
main/
  main.py                 # 전체 파이프라인 실행 진입점
  backend/                # Railway 백엔드
  frontend/               # Vercel 프론트엔드
  utils/                  # CCTV 수신, YOLO 추론, ROI, WebRTC 송출
  pruned.pt               # Pruned YOLO 가중치
  pruned.yaml             # Pruned YOLO 모델 설정

utils/
  live_opencv.py          # OpenCV 기반 CCTV 테스트/보조 코드

model/                    # 실험/학습 모델 파일
crash_test/               # 사고/위험도 관련 실험 코드
yolov12/                  # YOLOv12 관련 코드
yolov12_pruning/          # Pruning 관련 코드
```

## Deployment Targets

- Frontend: Vercel
- Backend: Railway
- YOLO inference/video sender: local YOLO PC

영상은 YOLO PC에서 WebRTC로 송출하고, 숫자/차트/상태 데이터는 Railway를 통해 웹으로 전달합니다.
