import os
import shutil
from ultralytics import YOLO
from compress.Compress import PruneHandler

# ─── 설정 ─────────────────────────────────────────────────────────
# 데이터셋 YAML 절대경로
DATA_YAML        = '/home/minjae/edge_fire_detection/merged_dataset.yaml'

# 다운로드된 사전 학습 모델(.pt) 절대경로
BASE_MODEL_PATH  = '/home/minjae/edge_fire_detection/yolov12/yolov12n.pt'

# .pt 파일 저장 폴더 절대경로
PT_DIR           = '/home/minjae/edge_fire_detection/real_pt'

os.makedirs(PT_DIR, exist_ok=True)

# 학습·Pruning 파라미터
PRETRAIN_EPOCHS  = 600
PRUNE_EPOCHS     = 600
BATCH_SIZE       = 32
DEVICE           = '0'
COMPRESS_RATIO   = 0.5
PRUNE_METHOD     = 'GM'    # 'L1', 'L2' 또는 'GM'
PRUNE_TYPE       = 'ALL'   # 'ALL', 'H' 또는 'B'

# 결과 저장 절대경로
RUNS_DIR         = '/home/minjae/edge_fire_detection/yolov12/runs'
PRETRAIN_PROJECT = os.path.join(RUNS_DIR, 'detect')
PRUNE_PROJECT    = os.path.join(RUNS_DIR, 'prune')
BEST_DST         = os.path.join(PT_DIR, 'best.pt')

def main():
    # 1️⃣ 사전 학습
    print("🟡 [1/3] YOLOv8 사전 학습 시작")
    model = YOLO(BASE_MODEL_PATH, verbose=True)
    model.train(
        data=DATA_YAML,
        epochs=PRETRAIN_EPOCHS,
        imgsz=640,
        batch=BATCH_SIZE,
        device=DEVICE,
        name='pretrain',
        project=PRETRAIN_PROJECT
    )
    # 학습 결과 디렉토리 직접 구성
    save_dir = str(model.trainer.save_dir)
    best_src = os.path.join(save_dir, 'weights', 'best.pt')
    print(f"[INFO] 학습 완료된 best.pt 위치: {best_src}")

    # 2️⃣ best.pt 복사
    print(f"[INFO] 복사: {best_src} → {BEST_DST}")
    shutil.copy(best_src, BEST_DST)

    # 3️⃣ Pruning
    print("🟢 [2/3] Pruning 시작")
    base_model = YOLO(BEST_DST, verbose=True)
    pruner = PruneHandler(
        base_model,
        compression_ratio=COMPRESS_RATIO,
        method=PRUNE_METHOD,
        cfg_output_path=PT_DIR,
        prune_type=PRUNE_TYPE
    )
    pruned = pruner.compress_yolov12()  # best_prune.pt + best.yaml 생성

    # 4️⃣ Pruned 모델 재학습
    print("🔵 [3/3] Pruned 모델 재학습 시작")
    pruned.train(
        data=DATA_YAML,
        epochs=PRUNE_EPOCHS,
        imgsz=640,
        batch=BATCH_SIZE,
        device=DEVICE,
        name='pruned',
        project=PRUNE_PROJECT,
        workers=8,
        resume=False
    )

    print("\n✅ 전체 파이프라인 완료!")

if __name__ == '__main__':
    main()

