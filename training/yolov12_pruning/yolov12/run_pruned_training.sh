#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SCRIPT="${ROOT_DIR}/pruned_training.py"
RUN_ROOT="${ROOT_DIR}/ga_out/ft_runs"

RUNS=(
  "g001_i030"
  "g004_i048"
)

for run_name in "${RUNS[@]}"; do
  echo "[info] start training: ${run_name}"
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" --resume_path "${RUN_ROOT}/${run_name}"
  echo "[info] finished training: ${run_name}"
done

echo "[done] all requested pruned models have been trained"
