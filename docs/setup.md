# 환경 구축

모든 명령은 별도 안내가 없으면 저장소 루트에서 실행합니다. 로컬 분석 PC와 웹 백엔드는 각각 다른 환경에서 실행해도 됩니다.

## 1. 시스템 도구와 코드 다운로드

Ubuntu/Linux 기준입니다. 이미 설치된 패키지는 다시 설치할 필요가 없습니다.

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs libgl1 libglib2.0-0

git clone https://github.com/Minjea31/capstone_design.git
cd capstone_design
git lfs install
git lfs pull
```

`.pt`, `.avi`, `.mp4` 파일은 Git LFS로 관리합니다. GitHub의 ZIP 다운로드만으로는 모델·영상이 들어오지 않을 수 있으므로 `git lfs pull`을 실행합니다.

## 2. Python 추론 환경

Python 3.10을 사용합니다. 다음 중 한 방법만 선택합니다.

Conda:

```bash
conda create -n capstone python=3.10 pip -y
conda activate capstone
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.10이 설치되어 있다면 venv:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt`에는 검증에 사용한 PyTorch 2.2.2, torchvision 0.17.2, NumPy 1.26.4 및 YOLOv12 소스 커밋이 기록돼 있습니다. YOLOv12는 `sunsmarterjie/yolov12`의 `64d5ffc6db35a1bde446689ab8450ecd43475e00` 커밋을 일반 패키지로 설치합니다. `pip install -e _deps/yolov12` 방식은 사용하지 않습니다.

설치 확인:

```bash
python -c "import cv2, flask, torch, ultralytics; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('YOLO:', ultralytics.__file__)"
python main/main.py --help
python tools/verify_runtime.py
```

YOLO 모듈 위치가 삭제된 `_deps`를 가리키거나 `ModuleNotFoundError: ultralytics`가 발생하면 이전 editable 설치를 제거하고 다시 설치합니다.

```bash
python -m pip uninstall -y ultralytics
python -m pip install -r requirements.txt
```

기존 환경에 `opencv-python`과 `opencv-python-headless`가 함께 설치되어 있으면 충돌할 수 있습니다. 새 환경에서는 위 requirements만 설치합니다. OpenCV 창을 여는 ROI 도구 때문에 추론 PC에서는 `opencv-python`을 사용합니다.

## 3. CPU와 GPU

CCTV 송신기는 CUDA 사용 가능 여부에 따라 GPU/CPU를 자동 선택합니다. USB 시연은 `--device cpu`, 모델 비교 도구는 `--device cpu`로 CPU를 명시할 수 있습니다.

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

이 저장소의 회귀 검증은 CPU에서 수행했습니다. GPU 사용 시 설치한 PyTorch와 NVIDIA 드라이버가 호환되어야 합니다. Jetson ARM 장치에는 이 x86_64 환경을 그대로 적용하지 말고 해당 JetPack에 맞는 PyTorch/torchvision을 설치해야 합니다.

FlashAttention은 기본 설치에 포함하지 않습니다. 지원하지 않는 장치에서는 YOLOv12의 scaled dot product attention 경로를 사용합니다.

## 4. 실행

저장 영상으로 로컬 확인:

```bash
python main/main.py --video-dir data/videos/weather --use-roi --port 5000
```

- 주소: `http://localhost:5000`
- 영상: `data/videos/weather/{2,3,6,10}.avi`
- 모델: `models/cctv/pruned.pt`, `models/cctv/pruned.yaml`
- 실시간 CCTV 선택에는 인터넷과 ITS API 응답이 필요합니다.

USB 카메라 시연:

```bash
python capstone_demo/main.py --roi-use --device cpu
```

- 주소: `http://localhost:8000`
- 카메라 변경: `--camera 0` 또는 `--camera /dev/video2`
- 사고 영상 모니터 끄기: `--no-crash-demo`

## 5. 별도 웹 백엔드

추론 모델을 설치하지 않는 별도 환경에서 실행할 수 있습니다.

```bash
python3.10 -m venv .venv-backend
source .venv-backend/bin/activate
python -m pip install -r main/backend/requirements.txt
python main/backend/main.py
```

`http://localhost:5000/health`에서 확인합니다. 로컬 CCTV 대시보드와 동시에 실행하려면 다른 포트를 지정합니다.

```bash
PORT=5001 python main/backend/main.py
```

운영 서버는 `main/backend/railway.json`의 Gunicorn 실행 설정을 사용합니다. Railway 서비스 루트는 `main/backend`입니다.

## 6. 프론트엔드

검증에 사용한 Node.js는 22입니다. Node.js와 npm을 설치한 뒤:

```bash
cd main/frontend
npm ci
NEXT_PUBLIC_API_URL=http://localhost:5000 npm run dev
```

`http://localhost:3000`에 접속합니다. 백엔드를 5001 포트로 실행했다면 API URL도 5001로 맞춥니다. 배포할 때는 Vercel 환경변수 `NEXT_PUBLIC_API_URL`에 실제 백엔드 주소를 넣고 프로젝트 루트를 `main/frontend`로 지정합니다.

프로덕션 빌드 확인:

```bash
cd main/frontend
npm ci
npm run build
```

## 7. 선택 기능

WebRTC 송신을 사용하려면 활성화한 추론 환경에서:

```bash
python -m pip install -r main/requirements-webrtc.txt
```

실행 환경변수와 ICE/TURN 설정은 [main 안내](../main/README.md#webrtc-영상-송출)를 참고합니다.

학습·GA 실험을 실행하려면:

```bash
python -m pip install -r requirements-training.txt
```

학습 데이터셋은 저장소에 포함하지 않습니다. `training/yolov12`와 `training/yolov12_pruning`의 데이터 생성 도구로 구성하고 `--data`에 YAML 경로를 전달합니다. 자세한 경로와 실행 예시는 [학습 안내](../training/README.md)에 있습니다. Depth Anything 실험은 별도로 `transformers`와 모델 다운로드가 필요하며 기본 시연에는 사용하지 않습니다.
