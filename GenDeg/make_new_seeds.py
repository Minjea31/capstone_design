import os
import json
from pathlib import Path

# 설정: 유저님의 실제 이미지 데이터셋 경로
root_dir = "/share/dataset/all_images"
output_seed_path = os.path.join(root_dir, "seed.json")

new_seeds = []

# root_dir 아래의 1,183개 폴더를 탐색합니다.
subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
print(f"총 {len(subdirs)}개의 폴더를 발견했습니다. seeds.json 생성을 시작합니다...")

for subdir in subdirs:
    folder_path = os.path.join(root_dir, subdir)
    
    # 폴더 이름 자체가 이미지 파일명인 구조이므로 내부에서 실제 이미지를 찾습니다.
    # 보통 폴더 안에 .jpg나 .png 파일이 하나 있을 것입니다.
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not files:
        continue
        
    # 첫 번째 이미지를 대상으로 삼습니다.
    img_name = files[0]
    # GenDeg 모델이 기대하는 경로 형식으로 구성 (보통 /data/ 경로를 기준으로 작성)
    # 나중에 --replace_dir 인자로 보정되므로 절대 경로의 끝부분 위주로 작성합니다.
    img_relative_path = os.path.join(subdir, img_name)
    
    # 젠데그 모델의 seeds.json 표준 형식에 맞게 딕셔너리 생성
    seed_entry = {
        "image_path": img_relative_path, # 변환될 이미지 (보통 target과 동일하게 설정 후 생성)
        "target_path": img_relative_path, # 원본(Clean) 이미지
        "dataset": "Custom",             # 데이터셋 이름 (임의 지정)
        "degradation": "none",           # 현재 상태
        "category": "all",               # 카테고리
        "split": "test"                  # 실행 시 split="test"로 불러오므로 test로 지정
    }
    new_seeds.append(seed_entry)

# 파일 저장
with open(output_seed_path, 'w', encoding='utf-8') as f:
    json.dump(new_seeds, f, indent=4)

print(f"새로운 seeds.json이 생성되었습니다: {output_seed_path}")
print(f"총 {len(new_seeds)}개의 엔트리가 포함되었습니다.")