#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

DATA="${DATA:-../mix_data.yaml}"
DEVICE="${DEVICE:-0}"
BS="${BS:-8}"
EPOCH="${EPOCH:-600}"
MODEL_PT="${MODEL_PT:-yolov12n.pt}"

BASELINE_NAME="${BASELINE_NAME:-demo_final}"
BEST_PT="${BEST_PT:-checkpoints/${BASELINE_NAME}/weights/best.pt}"

PRUNE_RATIO="${PRUNE_RATIO:-0.6}"
PRUNE_EPOCH="${PRUNE_EPOCH:-${EPOCH}}"
PRUNE_NAME="${PRUNE_NAME:-${BASELINE_NAME}_prune_r06}"
PRUNE_CFG_DIR="${PRUNE_CFG_DIR:-./severe_prune}"

echo "[info] baseline training start: name=${BASELINE_NAME}, data=${DATA}, epochs=${EPOCH}"
"${PYTHON_BIN}" train_baseline_model.py \
  --name "${BASELINE_NAME}" \
  --data "${DATA}" \
  --epoch "${EPOCH}" \
  --bs "${BS}" \
  --device "${DEVICE}" \
  --model_pt "${MODEL_PT}"

if [[ ! -f "${BEST_PT}" ]]; then
  echo "[error] baseline best.pt not found: ${BEST_PT}" >&2
  exit 1
fi

echo "[info] pruning fine-tuning start: bmodel=${BEST_PT}, ratio=${PRUNE_RATIO}, data=${DATA}"
"${PYTHON_BIN}" pruning_finetuning.py \
  --bmodel "${BEST_PT}" \
  --data "${DATA}" \
  --compress_ratio "${PRUNE_RATIO}" \
  --epoch "${PRUNE_EPOCH}" \
  --name "${PRUNE_NAME}" \
  --bs "${BS}" \
  --device "${DEVICE}" \
  --cfg_output_path "${PRUNE_CFG_DIR}" \
  --ratio_map ""

echo "[done] baseline training and pruning fine-tuning completed"
