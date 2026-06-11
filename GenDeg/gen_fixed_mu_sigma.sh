#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
SCRIPT="run_parallel_mu_sigma.py"
INPUT="./raw/frames"
BATCH=4
STEPS=100
CFG_TEXT=7.5
CFG_IMAGE=1.5
MU=0.1
SIGMA=0.07

# Create outputs for haze, rain, raindrop
# $PYTHON $SCRIPT --resolution 512 --output ./raw --deg_type haze --input "$INPUT" --batch-size $BATCH --steps $STEPS --cfg-text $CFG_TEXT --cfg-image $CFG_IMAGE --mu $MU --sigma $SIGMA

# $PYTHON $SCRIPT --resolution 512 --output ./raw --deg_type rain --input "$INPUT" --batch-size $BATCH --steps $STEPS --cfg-text $CFG_TEXT --cfg-image $CFG_IMAGE --mu $MU --sigma $SIGMA

$PYTHON $SCRIPT --resolution 512 --output ./raw --deg_type raindrop --input "$INPUT" --batch-size $BATCH --steps $STEPS --cfg-text $CFG_TEXT --cfg-image $CFG_IMAGE --mu $MU --sigma $SIGMA
