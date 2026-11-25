# 폴더 구조와 정리 내역

실행 앱, 입력 데이터, 모델, 실험, 발표 자료를 구분했습니다. 배포 설정에 사용되는 `main/backend`, `main/frontend`와 기존 로컬 실행 명령 `cd main && python main.py`는 유지했습니다. 파이썬의 `main/utils`는 실행 앱 내부 모듈이며, 루트의 도구 모음은 `tools`입니다.

## 이동 경로

| 이전 경로 | 현재 경로 | 용도 |
| --- | --- | --- |
| `main/pruned.pt`, `main/pruned.yaml` | `models/cctv/` | CCTV 추론 모델 |
| `capstone_demo/model/` | `models/camera/` | USB 발표 시연 모델 |
| `crash_test/crash*.pt`, `crash_pruned.yaml` | `models/crash/` | 사고 영상 모델 |
| `model/` | `models/experiments/` | 악천후 학습·GA 비교 가중치 |
| `yolov12/best.pt`, `yolov12/yolov12/final_demo/` | `models/experiments/` | 기존 실험 결과 |
| 루트 및 학습 폴더의 사전학습 `.pt` | `models/pretrained/` | YOLO 기본 가중치 |
| `utils/saved_videos/` | `data/videos/weather/` | 악천후 영상 4개 |
| 두 시연 폴더의 사고 `saved_videos/` | `data/videos/crash/` | 공용 사고 영상 3개 |
| 루트 `utils/` | `tools/` | 수집·ROI·비교·녹화 도구 |
| `crash_test/` | `experiments/crash_test/` | 사고·방향·깊이 실험 |
| `yolov12/` | `training/yolov12/` | 데이터 증강·학습·평가 |
| `yolov12_pruning/` | `training/yolov12_pruning/` | FPGM·GA 경량화 |
| `yolov12_pruning/yolov12/prune.py` | `training/pruning.py` | 공용 사용자 수정 pruning |
| `yolov12_pruning/drawio/` | `docs/diagrams/` | 모델 구조 설명 그림 |
| 루트 QR 이미지 | `assets/demo-qr.png` | 기존 데모 QR |

모델·영상 기본 경로는 각 스크립트의 `__file__`을 기준으로 해석합니다. CCTV ROI의 카메라 이름 매핑과 저장 영상 ROI 매핑, 추론 임계값, 차량 클래스, 웹 화면 및 API 구현은 유지했습니다.

## 삭제·추적 제외한 항목

- 셸 명령을 잘못 붙여넣어 생성된 빈 파일 `main/python3`, `main/DETECTION_*.txt`를 삭제했습니다.
- 루트 `.gitignore`와 중복되는 `main/.gitignore`를 통합했습니다.
- 사고 영상 3개의 중복 사본과 동일한 `yolov12n.pt` 사본 하나를 정리했습니다. SHA-256이 일치하는 파일만 합쳤습니다.
- 실제 패키지 소스가 없는 Ultralytics `pyproject.toml` 2개와 오래된 requirements 2개를 제거했습니다. Python 3.11 전용 FlashAttention 휠 파일을 요구하던 설정 대신 루트의 실행·학습 requirements를 사용합니다.
- 참조되지 않는 수동 디버거 `model_structure_pdb.py`와 제목만 있던 그림 README를 제거했습니다.
- 기존 사용자 수정 pruning 구현은 `training/pruning.py`로 옮기고 두 `compress/`에서 함께 불러옵니다. 연산 구현은 그대로 보존했으며, 설치된 PyTorch 파일을 직접 수정하지 않습니다.
- `.vscode/settings.json`은 로컬에 남기고 Git 추적만 해제했습니다.
- 캐시·추가 녹화 영상·검증 출력은 `.gitignore`로 제외합니다. `_deps`, `.codex`, `.agents`도 계속 제외합니다.

학습 결과와 실험 코드는 실행 앱에서 호출하지 않는다는 이유만으로 삭제하지 않았습니다. CCTV·카메라·사고 모델은 서로 다른 가중치이므로 각각 보존했습니다. 두 `compress` 구현 역시 GA 레이어별 비율 지원 등 내용이 달라 합치지 않았습니다.

## 출력과 데이터 관리

추가 녹화 영상과 검증 결과는 `outputs/`에 저장하는 것을 권장합니다. Git에는 시연용 영상 7개만 명시적으로 포함합니다. 모델·영상은 `.gitattributes`의 확장자 규칙으로 Git LFS에 저장하며 이동 후에도 같은 규칙이 적용됩니다.

학습 데이터셋은 별도 준비해야 합니다. 기존 데이터 생성기의 상대 폴더 관계는 `training/` 아래에서 유지했습니다. 임의 경로를 CLI 옵션으로 전달하던 스크립트는 기존처럼 명시적으로 지정할 수 있습니다.
