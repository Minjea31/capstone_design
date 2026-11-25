# 원격 탐지 백엔드

Flask API가 분석 PC의 탐지 결과·프레임·상태를 수신하고 Next.js 화면에 전달합니다. 원격 CCTV/저장 영상 선택 제어, WebSocket 전송, WebRTC 시그널링도 처리합니다. 모델 추론은 분석 PC에서 실행하므로 백엔드에 PyTorch나 YOLO가 필요하지 않습니다.

## 환경과 실행

```bash
# 저장소 루트에서
conda create -n capstone-backend python=3.10 pip -y
conda activate capstone-backend
python -m pip install -r main/backend/requirements.txt
python main/backend/main.py
```

기본 주소는 `http://localhost:5000`입니다. `PORT=5001 python main/backend/main.py`로 포트를 바꿀 수 있습니다. 공통 설치·프론트엔드 설정은 [환경 구축 문서](../../docs/setup.md)를 참고하세요.

## 주요 경로

| 경로 | 역할 |
| --- | --- |
| `GET /health` | 상태 확인 |
| `POST /api/detection` | 분석 PC 결과 수신 |
| `GET /api/latest` | 최근 탐지 결과 |
| `GET /api/status` | 집계·시연 상태 |
| `GET /api/cctvs` | CCTV·저장 영상 목록 |
| `POST /api/select/<index>` | 분석할 입력 선택 |
| `GET /api/control` | 분석 PC가 선택 제어 조회 |
| `/ws/sender`, `/ws/viewer` | 분석 PC·뷰어 WebSocket |
| `/ws/webrtc/sender`, `/ws/webrtc/viewer` | WebRTC 시그널링 |

## Railway

서비스 루트는 `main/backend`입니다. `railway.json`의 실행 명령은 다음과 같습니다.

```bash
gunicorn --workers 1 --worker-class gthread --threads 100 --timeout 0 --bind 0.0.0.0:$PORT main:app
```

메모리에 최신 상태와 시그널링 세션을 보관하므로 기존처럼 worker 하나를 사용합니다. healthcheck 경로는 `/health`입니다. 저장소 구조 정리에서도 이 배포 경로와 설정은 유지했습니다.
