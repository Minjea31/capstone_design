import argparse
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_VIEW_WIDTH = 1600
MAX_VIEW_HEIGHT = 900


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize CCTV dataset images side by side with and without YOLO bounding boxes."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("cctv_raw_dataset"),
        help="Dataset root path. Default: cctv_raw_dataset",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to visualize. Default: train",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=0,
        help="cv2.waitKey delay in ms. Default: 0",
    )
    return parser.parse_args()


def load_image_paths(images_dir: Path):
    image_paths = [path for path in sorted(images_dir.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_paths:
        raise FileNotFoundError(f"No image files found in: {images_dir}")
    return image_paths


def yolo_to_xyxy(values, width, height):
    class_id, cx, cy, bw, bh = values
    x_center = cx * width
    y_center = cy * height
    box_width = bw * width
    box_height = bh * height

    x1 = int(round(x_center - box_width / 2))
    y1 = int(round(y_center - box_height / 2))
    x2 = int(round(x_center + box_width / 2))
    y2 = int(round(y_center + box_height / 2))

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return int(class_id), x1, y1, x2, y2


def draw_boxes(image, label_path: Path):
    canvas = image.copy()
    if not label_path.exists():
        cv2.putText(canvas, "No label file", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return canvas, 0

    lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
    if not lines:
        cv2.putText(canvas, "Empty label file", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        return canvas, 0

    height, width = canvas.shape[:2]
    count = 0

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        values = list(map(float, parts))
        _, x1, y1, x2, y2 = yolo_to_xyxy(values, width, height)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
        count += 1

    return canvas, count


def scale_to_fit(image, max_width, max_height):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image

    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def build_view(image_path: Path, label_path: Path, index: int, total: int):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    boxed, box_count = draw_boxes(image, label_path)
    merged = cv2.hconcat([image, boxed])
    return scale_to_fit(merged, MAX_VIEW_WIDTH, MAX_VIEW_HEIGHT)


def main():
    args = parse_args()
    dataset_root = args.dataset.resolve()
    split_root = dataset_root / args.split
    images_dir = split_root / "images"
    labels_dir = split_root / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    image_paths = load_image_paths(images_dir)
    index = 0
    window_name = f"CCTV Visualizer - {args.split}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        image_path = image_paths[index]
        label_path = labels_dir / f"{image_path.stem}.txt"
        canvas = build_view(image_path, label_path, index, len(image_paths))

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(args.delay) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("a"):
            index = (index - 1) % len(image_paths)
        elif key == ord("d"):
            index = (index + 1) % len(image_paths)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
