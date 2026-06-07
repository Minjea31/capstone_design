# Capstone Main

## UI

UI는 아래 주소에서 확인할 수 있습니다.

`https://detection-demo-q926dgjow-capstone11.vercel.app/`

## Railway 전송 권장값

Railway 한 인스턴스에서 프레임, 상태, WebSocket을 같이 처리하므로 너무 높은 FPS/화질은 지연을 만듭니다.

```bash
DETECTION_BACKEND_URL=https://detection-demo-production.up.railway.app \
DETECTION_TRANSPORT=websocket \
DETECTION_WS_HTTP_FALLBACK=1 \
DETECTION_LOG_SEND=1 /
DETECTION_SEND_INTERVAL=0.45 \
DETECTION_FRAME_WIDTH=720 \
DETECTION_JPEG_QUALITY=35 \
DETECTION_CONTROL_INTERVAL=3 \
DETECTION_CONTROL_TIMEOUT=2 \
python3 main.py --use-roi
```

```bash

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
