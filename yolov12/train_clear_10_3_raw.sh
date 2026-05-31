#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$ROOT_DIR/yolov12"

EPOCHS=600
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-0}"
MODEL_PT="${MODEL_PT:-yolov12n.pt}"
PYTHON_BIN="${PYTHON_BIN:-python}"

declare -a RUNS=(
  "clear_haze_10_3 clear_haze_10_3.yaml"
  "clear_rain_10_3 clear_rain_10_3.yaml"
  "clear_raindrop_10_3 clear_raindrop_10_3.yaml"
)

cd "$TRAIN_DIR"

for run in "${RUNS[@]}"; do
  dataset_name="${run%% *}"
  yaml_file="${run##* }"
  model_name="${dataset_name}_raw"

  echo "Starting training: ${model_name} (${yaml_file})"

  "$PYTHON_BIN" train_baseline_model.py \
    --name "$model_name" \
    --data "../${yaml_file}" \
    --epoch "$EPOCHS" \
    --bs "$BATCH_SIZE" \
    --device "$DEVICE" \
    --model_pt "$MODEL_PT"
done
