import os
import shutil
from pathlib import Path
from tqdm import tqdm

# --- 경로 설정 (본인의 환경에 맞게 확인) ---
base_dir = "/share/dataset/all_images"
original_seeds_path = "/share/GenDeg/GenDeg/data/seeds.json"
# 추가하고 싶은 하위 폴더 리스트
sub_folders = ["raindrop", "haze", "low-light", "snow"] 

def reorganize():
    # 1. 파일 목록 가져오기 (이미지 파일만)
    extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
    files = [f for f in os.listdir(base_dir) if f.endswith(extensions)]
    
    print(f"총 {len(files)}개의 이미지를 정리합니다...")

    for filename in tqdm(files):
        # 파일명에서 확장자 제거한 이름 (폴더명이 됨)
        folder_name = os.path.splitext(filename)[0]
        new_folder_path = os.path.join(base_dir, folder_name)
        
        try:
            # 2. 폴더 생성
            os.makedirs(new_folder_path, exist_ok=True)
            
            # 3. 이미지 이동
            old_file_path = os.path.join(base_dir, filename)
            new_file_path = os.path.join(new_folder_path, filename)
            shutil.move(old_file_path, new_file_path)
            
            # 4. 하위 폴더 생성 (raindrop, snow 등)
            for sub in sub_folders:
                os.makedirs(os.path.join(new_folder_path, sub), exist_ok=True)
            
            # 5. seeds.json 복사
            if os.path.exists(original_seeds_path):
                shutil.copy(original_seeds_path, os.path.join(new_folder_path, "seeds.json"))
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\n✅ 모든 데이터 정리가 완료되었습니다!")

if __name__ == "__main__":
    reorganize()