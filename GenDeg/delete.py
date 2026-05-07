import os
import shutil
from pathlib import Path

# 설정: 데이터셋 최상위 폴더 경로
root_dir = "/share/dataset/all_images"
target_seed_name = "seeds.json"
main_seed_path = os.path.join(root_dir, target_seed_name)

print(f"작업을 시작합니다: {root_dir}")

# 1. 먼저 메인 seed 파일 하나를 루트로 복사/이동
# 하위 폴더들을 탐색하며 첫 번째로 발견되는 seeds.json을 루트로 가져옵니다.
seed_found = False
subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

for subdir in subdirs:
    source_path = os.path.join(root_dir, subdir, target_seed_name)
    if os.path.exists(source_path):
        print(f"메인 seed 파일 발견: {source_path}")
        shutil.copy2(source_path, main_seed_path)
        print(f"복사 완료: {main_seed_path}")
        seed_found = True
        break

if not seed_found:
    print("하위 폴더에서 seeds.json 파일을 찾을 수 없습니다. 작업을 중단합니다.")
    exit()

# 2. 하위 폴더의 모든 seeds.json 삭제
print("\n하위 폴더의 중복된 seeds.json 삭제를 시작합니다...")
delete_count = 0

for subdir in subdirs:
    file_to_delete = os.path.join(root_dir, subdir, target_seed_name)
    if os.path.exists(file_to_delete):
        try:
            os.remove(file_to_delete)
            delete_count += 1
            if delete_count % 100 == 0:
                print(f"{delete_count}개 파일 삭제 중...")
        except Exception as e:
            print(f"삭제 실패 ({file_to_delete}): {e}")

print(f"\n작업 완료!")
print(f"- 메인 파일 저장 위치: {main_seed_path}")
print(f"- 삭제된 하위 파일 개수: {delete_count}개")