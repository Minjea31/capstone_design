#!/bin/bash

# 기본 설정
INPUT_PATH="/share/dataset/all_images"
BASE_OUTPUT="./gendeg_dataset"
RES=512
BATCH=4

DEG_TYPES=("raindrop" "rain" "haze")

for DEG in "${DEG_TYPES[@]}"
do
    # 각 타입별로 mu와 sigma 설정
    case $DEG in
        "raindrop")
            MU=0.01
            SIGMA=0.06
            ;;
        "rain")
            MU=0.02
            SIGMA=0.02
            ;;
        "haze")
            MU=0.1
            SIGMA=0.07
            ;;
    esac

    echo "------------------------------------------------"
    echo "Starting generation for: $DEG (mu: $MU, sigma: $SIGMA)"
    echo "------------------------------------------------"

    CURRENT_OUTPUT="${BASE_OUTPUT}/output_${DEG}_final"
    mkdir -p $CURRENT_OUTPUT

    # python 실행 시 --mu와 --sigma 인자 추가 (실제 코드의 인자 명칭에 맞춰 수정 필요)
    python run_parallel_mu_sigma.py \
        --input "$INPUT_PATH" \
        --output "$CURRENT_OUTPUT" \
        --resolution $RES \
        --deg_type "$DEG" \
        --batch-size $BATCH \
        --mu $MU \
        --sigma $SIGMA \
        --replace_dir "${INPUT_PATH}/"

    echo "Finished $DEG. Results saved in $CURRENT_OUTPUT"
done

echo "All tasks completed!"