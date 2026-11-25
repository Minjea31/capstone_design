# Capstone Main

## 악천후 저장 영상으로 웹 실행

프로젝트의 `utils/saved_videos`에 있는 영상을 기존 CCTV 대시보드에서 선택하고 반복 재생할 수 있습니다.
오른쪽 CCTV 선택 목록 아래에 `2.avi`, `3.avi`, `6.avi`, `10.avi` 선택 버튼이 표시됩니다.
기존 YOLO 탐지 결과도 함께 표시하며, 실시간 CCTV와 저장 영상을 같은 화면에서 전환할 수 있습니다.

```bash
cd /home/a/capstone_design/main
/home/a/anaconda3/envs/yolo/bin/python main.py \
  --video-dir ../utils/saved_videos --port 5000
```

브라우저에서 `http://localhost:5000`에 접속한 뒤 오른쪽의 **악천후 저장 영상** 목록을 선택하세요.
위 명령은 첫 저장 영상으로 시작하고, 실시간 CCTV를 선택할 때 ITS API로 URL을 조회합니다.
영상이 끝나면 처음부터 반복합니다. 기본 실행(`python main.py`)도 저장 영상 목록을 자동으로 추가합니다.
다른 PC에서는 실행 PC의 IP 주소와 포트로 접속합니다.

`2.avi`, `3.avi`, `10.avi`는 각각 기존 `roi_data_2.json`, `roi_data_3.json`, `roi_data_10.json`을 자동 적용합니다.
`6.avi`는 기존 `roi_data_6.json`과 촬영 방향이 달라 전체 화면에서 탐지합니다. 이 영상의 ROI는 다시 지정해야 합니다.
자동 연결은 프로젝트의 `utils/saved_videos` 안에 있는 확인된 파일에만 적용합니다.
저장 영상과 일치하는 ROI가 있으면 `--roi-path <JSON 경로>`를 지정하세요.
처리 속도는 추론 성능에 따라 달라지며, CUDA를 사용할 수 없으면 CPU에서 실행합니다.

배포된 Vercel/Railway 화면에 이 버튼을 표시하려면 `frontend`와 `backend`를 함께 배포하고,
저장 영상 파일이 있는 PC에서 수정된 `main.py` 송신기를 실행해야 합니다.

## UI

UI는 아래 주소에서 확인할 수 있습니다.

`https://detection-demo-q926dgjow-capstone11.vercel.app/`

## Railway 전송 권장값

Railway 한 인스턴스에서 프레임, 상태, WebSocket을 같이 처리하므로 너무 높은 FPS/화질은 지연을 만듭니다.

```bash
DETECTION_BACKEND_URL=https://detection-demo-production.up.railway.app \
DETECTION_TRANSPORT=websocket \
DETECTION_WS_HTTP_FALLBACK=1 \
DETECTION_LOG_SEND=1 \
DETECTION_SEND_EVENTS_ONLY=0 \
DETECTION_SEND_INTERVAL=0.45 \
DETECTION_FRAME_WIDTH=720 \
DETECTION_JPEG_QUALITY=35 \
DETECTION_CONTROL_INTERVAL=3 \
DETECTION_CONTROL_TIMEOUT=2 \
python main.py --use-roi
```

더 안정성이 필요하면 `DETECTION_SEND_INTERVAL=0.7`, `DETECTION_FRAME_WIDTH=640`, `DETECTION_JPEG_QUALITY=30`으로 낮추세요.

## WebRTC 영상 송출

영상은 WebRTC로 직접 보내고, 숫자/차트/status만 Railway로 보내면 Railway 부하가 줄어듭니다.

YOLO PC에서 한 번 설치:

```bash
pip install -r requirements-webrtc.txt
```

실행:

```bash
DETECTION_BACKEND_URL=https://detection-demo-production.up.railway.app \
DETECTION_TRANSPORT=websocket \
DETECTION_WEBRTC_ENABLED=1 \
DETECTION_WEBRTC_FPS=12 \
DETECTION_SEND_FRAME_JPEG=0 \
DETECTION_SEND_INTERVAL=0.7 \
DETECTION_FRAME_WIDTH=640 \
DETECTION_JPEG_QUALITY=30 \
DETECTION_CONTROL_INTERVAL=3 \
DETECTION_CONTROL_TIMEOUT=2 \
python3 main.py --use-roi
```

WebRTC가 연결되면 영상은 브라우저 `<video>`로 표시되고, `/frame.jpg` 폴링은 기본적으로 꺼집니다.

NAT/방화벽 때문에 연결이 안 되면 TURN 서버가 필요합니다. 이때 YOLO PC와 Vercel에 같은 ICE 서버를 설정하세요.

```bash
DETECTION_WEBRTC_ICE_SERVERS='[{"urls":["stun:stun.l.google.com:19302"]}]'
```

```env
NEXT_PUBLIC_WEBRTC_ICE_SERVERS=[{"urls":["stun:stun.l.google.com:19302"]}]
```
