#!/bin/bash

# 기본 설정
INPUT_PATH="/share/dataset/all_images"
BASE_OUTPUT="./gendeg_dataset"
RES=512
BATCH=4

# 생성할 변조(Degradation) 리스트
# raindrop: 빗방울, rain: 비, haze: 안개
DEG_TYPES=("raindrop" "rain" "haze")

for DEG in "${DEG_TYPES[@]}"
do
    echo "------------------------------------------------"
    echo "Starting generation for: $DEG"
    echo "------------------------------------------------"

    # 출력 폴더 설정 (예: ./gendeg_dataset/output_raindrop)
    CURRENT_OUTPUT="${BASE_OUTPUT}/output_${DEG}"

    # 폴더가 없으면 생성
    mkdir -p $CURRENT_OUTPUT

    # 명령어 실행
    python run_parallel_mu_sigma.py \
        --input "$INPUT_PATH" \
        --output "$CURRENT_OUTPUT" \
        --resolution $RES \
        --deg_type "$DEG" \
        --batch-size $BATCH \
        --auto_s \
        --replace_dir "${INPUT_PATH}/"

    echo "Finished $DEG. Results saved in $CURRENT_OUTPUT"
done

echo "All tasks completed!"