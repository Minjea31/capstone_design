from ultralytics import YOLO
import torch
import pdb

# 모델 로드
model = YOLO("/home/minjae/edge_fire_detection/yolov12/ultralytics/cfg/models/v12/yolov12.yaml")

# 더미 입력 생성
x = torch.zeros(1, 3, 640, 640)

# 이 지점에서 멈추면 x가 이미 정의되어 있습니다
pdb.set_trace()
