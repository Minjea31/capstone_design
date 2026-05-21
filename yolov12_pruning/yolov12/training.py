
from ultralytics import YOLO

model = YOLO('yolov12n.yaml')


results = model.train(
    data='../merged_dataset.yaml',  # 수정한 yaml 파일 경로
    epochs=600,
    batch=32,
    imgsz=640,
    scale=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.1,
    device="0",
)
