import os
import json

root_dir = "/share/GenDeg/GenDeg/raw/frame"
output_seed_path = os.path.join(root_dir, "seed.json")

new_seeds = []

for img_name in sorted(os.listdir(root_dir)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(root_dir, img_name)

    seed_entry = {
        "image_path": img_path,
        "target_path": img_path,
        "dataset": "Custom",
        "degradation": "none",
        "category": "all",
        "split": "test"
    }

    new_seeds.append(seed_entry)

with open(output_seed_path, "w", encoding="utf-8") as f:
    json.dump(new_seeds, f, indent=4, ensure_ascii=False)

print(f"seed.json 생성 완료: {output_seed_path}")
print(f"총 {len(new_seeds)}개 이미지 등록")
