from ultralytics import YOLO
from compress.Compress import PruneHandler as ph
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--model_pt', type=str, default='/home/a/capstone/yolov12_pruning/yolov12/ga_after_training/g002_i072/weights/best.pt')

parser.add_argument('--model_pt', type=str, default='/home/a/capstone/yolov12_pruning/raw_best.pt')
# parser.add_argument('--model_yaml', type=str, default='/home/a/capstone/yolov12_pruning/yolov12/ga_after_training/g002_i072/best.yaml')

parser.add_argument('--data', type=str, default="/home/a/capstone/yolov12_pruning/severe.yaml")

args = parser.parse_args()
# model = YOLO(args.model_yaml).load(args.model_pt)
model = YOLO(args.model_pt)
model.val(data=args.data, imgsz=640, batch=8, workers=8)
# model.val(data="/home/minjae/edge_fire_detection/data/Fire_other_smoke/Fire_other_smoke_data.yaml", imgsz=640, batch=32, workers=8)
