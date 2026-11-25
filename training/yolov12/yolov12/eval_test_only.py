import argparse
from pathlib import Path

from ultralytics import YOLO

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO .pt model on the test split only."
    )
    parser.add_argument(
        "--model-pt",
        type=str,
        default=str(REPOSITORY_ROOT / 'training/yolov12/yolov12/checkpoints/test11/weights/best.pt'),
        help="Path to YOLO weights (.pt).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "dataset.yaml"),
        help="Path to dataset YAML.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help='CUDA device like "0" or "cpu". Empty string lets Ultralytics choose.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model_pt).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
    )

    print("\n[test evaluation summary]")
    print(f"model: {model_path}")
    print(f"data: {data_path}")
    print(f"precision(B): {metrics.box.mp:.6f}")
    print(f"recall(B):    {metrics.box.mr:.6f}")
    print(f"mAP50(B):     {metrics.box.map50:.6f}")
    print(f"mAP50-95(B):  {metrics.box.map:.6f}")


if __name__ == "__main__":
    main()
