from ultralytics import YOLO
from compress.Compress import PruneHandler as ph
import argparse

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

parser = argparse.ArgumentParser()
parser.add_argument('--model_pt', type=str, default=str(REPOSITORY_ROOT / 'training/yolov12_pruning/yolov12/checkpoints/demo_raw/weights/best.pt'))
parser.add_argument('--model_yaml', type=str, default=None)

# parser.add_argument('--model_pt', type=str, default=str(REPOSITORY_ROOT / 'models/experiments/g040_i048.pt'))
# parser.add_argument('--model_yaml', type=str, default=str(REPOSITORY_ROOT / 'models/experiments/g040_i048.yaml'))

parser.add_argument('--data', type=str, default=str(REPOSITORY_ROOT / 'training/yolov12_pruning/mix_data.yaml'))
parser.add_argument('--split', type=str, default="test", choices=["train", "val", "test"])
parser.add_argument('--imgsz', type=int, default=640)
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('--workers', type=int, default=8)

args = parser.parse_args()
if args.model_yaml:
    model = YOLO(args.model_yaml).load(args.model_pt)
else:
    model = YOLO(args.model_pt)
model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch, workers=args.workers)
# model.val(data="/home/minjae/edge_fire_detection/data/Fire_other_smoke/Fire_other_smoke_data.yaml", imgsz=640, batch=32, workers=8)
