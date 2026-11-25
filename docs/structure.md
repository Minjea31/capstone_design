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
