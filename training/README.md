# 학습·경량화 실험

`yolov12/`는 악천후 데이터 구성·기본 학습·평가, `yolov12_pruning/`은 FPGM/GA 기반 경량화와 재학습 코드입니다. 두 폴더의 `compress/`는 서로 다른 실험 구현이므로 각각 유지합니다.

```bash
python -m pip install -r requirements-training.txt
```

데이터셋은 저장소에 포함하지 않습니다. 데이터 YAML의 train/val/test 경로를 실제 파일 위치에 맞춰 준비해야 합니다. 기본 YOLO 가중치는 `models/pretrained/yolov12n.pt`, 기존 학습·GA 가중치는 `models/experiments/`에 있습니다.

## 기본 학습

저장소 루트에서 절대 데이터 경로를 만든 후 실행합니다.

```bash
DATA_YAML="$(pwd)/data/datasets/traffic.yaml"
cd training/yolov12/yolov12
python train_baseline_model.py --data "$DATA_YAML" --device 0 --epoch 600 --bs 8
```

## GA 경량화

```bash
DATA_YAML="$(pwd)/data/datasets/traffic.yaml"
cd training/yolov12_pruning/yolov12
python GA.py --data "$DATA_YAML" --help
```

`GA.py`의 전체 옵션을 확인한 뒤 `--help`를 빼고 필요한 학습·탐색 설정으로 실행합니다. `run_severe_train_then_prune.sh`는 `DATA`, `DEVICE`, `MODEL_PT` 등 환경변수로 입력을 받습니다. 자동화 스크립트의 기본 가중치 경로도 `models/pretrained`로 변경했습니다.

기존 `prune.py`의 사용자 수정 구현은 `training/pruning.py`에서 공유합니다. 두 `compress` 패키지가 이 모듈을 직접 사용하므로 site-packages의 PyTorch를 덮어쓸 필요가 없습니다. `scikit-learn`과 `pygad`는 학습 requirements에 포함돼 있습니다.

```bash
python -m unittest discover -s training/tests -p 'test_*.py'
```

위 테스트는 두 경량화 구현에서 채널 마스크, forward 실행, BatchNorm pruning과 마스크 제거를 확인합니다.

기존 실험 데이터 생성기는 각 학습 폴더 아래의 데이터셋 구조를 사용합니다. `--data`와 출력 옵션으로 경로를 지정할 수 있습니다. 오래된 실험의 데이터·체크포인트는 별도로 준비해야 하며 이번 구조 정리에서는 장시간 재학습이나 정확도 재평가를 실행하지 않았습니다.
