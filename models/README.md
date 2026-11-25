# 모델

`.pt`는 Git LFS로 관리합니다. 실행 전 `git lfs pull`로 내려받습니다.

| 폴더 | 용도 |
| --- | --- |
| `cctv/` | `main/main.py`의 CCTV·악천후 영상 추론 |
| `camera/` | USB 발표 시연 모델과 원본 비교 모델 |
| `crash/` | 사고 영상 시연·실험 모델 |
| `experiments/` | 데이터 증강·학습·GA 실험의 기존 모델 |
| `pretrained/` | YOLO12n·YOLOv12n·YOLOv8n 초기 가중치 |

기존 모델별 학습 데이터 설명은 [experiments/info.txt](experiments/info.txt)에 있습니다. `experiments/final_demo`는 별도의 기존 학습 결과이며 카메라 시연 모델과 가중치가 달라 보존했습니다.
