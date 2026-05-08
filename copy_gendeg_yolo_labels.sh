#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/home/a/capstone}"
SOURCE_LABEL_ROOT="$ROOT_DIR/yolov12/dataset"
TARGET_ROOT="$ROOT_DIR/gendeg_dataset"

if [[ ! -d "$SOURCE_LABEL_ROOT" ]]; then
  echo "Source dataset not found: $SOURCE_LABEL_ROOT" >&2
  exit 1
fi

if [[ ! -d "$TARGET_ROOT" ]]; then
  echo "Target dataset not found: $TARGET_ROOT" >&2
  exit 1
fi

declare -A LABEL_MAP=()
declare -A DUPLICATE_MAP=()

while IFS= read -r label_path; do
  label_name="$(basename "$label_path")"
  if [[ -n "${LABEL_MAP[$label_name]:-}" && "${LABEL_MAP[$label_name]}" != "$label_path" ]]; then
    DUPLICATE_MAP["$label_name"]=1
  fi
  LABEL_MAP["$label_name"]="$label_path"
done < <(find "$SOURCE_LABEL_ROOT" -type f -name '*.txt' | sort)

matched=0
missing=0
skipped=0

while IFS= read -r image_path; do
  image_name="$(basename "$image_path")"
  image_stem="${image_name%.*}"

  if [[ "$image_stem" == *"__"* ]]; then
    source_stem="${image_stem##*__}"
  else
    source_stem="$image_stem"
  fi

  source_label_name="$source_stem.txt"
  target_label_path="${image_path%.*}.txt"

  if [[ -f "$target_label_path" ]]; then
    ((skipped+=1))
    continue
  fi

  if [[ -n "${DUPLICATE_MAP[$source_label_name]:-}" ]]; then
    echo "Duplicate source label name, skipped: $source_label_name" >&2
    ((missing+=1))
    continue
  fi

  source_label_path="${LABEL_MAP[$source_label_name]:-}"
  if [[ -z "$source_label_path" ]]; then
    echo "Missing source label for: $image_name" >&2
    ((missing+=1))
    continue
  fi

  cp "$source_label_path" "$target_label_path"
  ((matched+=1))
done < <(find "$TARGET_ROOT" -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) | sort)

echo "Copied labels: $matched"
echo "Missing labels: $missing"
echo "Skipped existing: $skipped"
