import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_VIEW_WIDTH = 1600
MAX_VIEW_HEIGHT = 1000
PANEL_GAP = 12
TITLE_HEIGHT = 36
TITLE_BG = (30, 30, 30)
TITLE_FG = (255, 255, 255)
VARIANTS = [
    ("Original", None),
    ("Haze", "haze"),
    ("Rain", "rain"),
    ("Raindrop", "raindrop"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize original and generated degradation images as a panel."
    )
    parser.add_argument(
        "--raw-dataset",
        type=Path,
        default=Path("cctv_raw_dataset"),
        help="Original dataset root path. Default: cctv_raw_dataset",
    )
    parser.add_argument(
        "--gendeg-dataset",
        type=Path,
        default=Path("add_gendeg_dataset"),
        help="Generated degradation dataset root path. Default: add_gendeg_dataset",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to visualize. Default: train",
    )
    return parser.parse_args()


def load_image_paths(images_dir: Path):
    image_paths = [path for path in sorted(images_dir.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_paths:
        raise FileNotFoundError(f"No image files found in: {images_dir}")
    return image_paths


def scale_to_fit(image, max_width, max_height):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image

    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def build_variant_map(gendeg_images_dir: Path):
    variant_map = {}

    for path in load_image_paths(gendeg_images_dir):
        stem = path.stem
        variant_name = None

        if "__" in stem:
            left, right = stem.split("__", 1)
            if left.startswith("haze_"):
                variant_name = "haze"
            elif left.startswith("rain_"):
                variant_name = "rain"
            elif left.startswith("raindrop_"):
                variant_name = "raindrop"

            if variant_name:
                original_key = right
                variant_map.setdefault(original_key, {})[variant_name] = path

    return variant_map


def read_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def make_placeholder(reference_shape, title):
    placeholder = np.zeros(reference_shape, dtype=np.uint8)
    placeholder[:] = (20, 20, 20)
    text = f"{title} missing"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    x = max(20, (placeholder.shape[1] - text_size[0]) // 2)
    y = placeholder.shape[0] // 2
    cv2.putText(placeholder, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
    return placeholder


def add_title(image, title):
    panel = np.full((image.shape[0] + TITLE_HEIGHT, image.shape[1], 3), 0, dtype=np.uint8)
    panel[:TITLE_HEIGHT, :] = TITLE_BG
    panel[TITLE_HEIGHT:, :] = image
    cv2.putText(panel, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TITLE_FG, 2)
    return panel


def pad_to_size(image, width, height):
    top = 0
    left = 0
    bottom = height - image.shape[0]
    right = width - image.shape[1]
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def make_row(panels):
    max_height = max(panel.shape[0] for panel in panels)
    max_width = max(panel.shape[1] for panel in panels)
    normalized = [pad_to_size(panel, max_width, max_height) for panel in panels]

    spacer = np.zeros((max_height, PANEL_GAP, 3), dtype=np.uint8)
    row = normalized[0]
    for panel in normalized[1:]:
        row = cv2.hconcat([row, spacer, panel])
    return row


def build_view(raw_image_path: Path, variant_paths):
    original = read_image(raw_image_path)
    tiles = []

    for title, variant_key in VARIANTS:
        if variant_key is None:
            image = original
        else:
            variant_path = variant_paths.get(variant_key)
            image = read_image(variant_path) if variant_path else make_placeholder(original.shape, title)
        tiles.append(add_title(image, title))

    top_row = make_row(tiles[:2])
    bottom_row = make_row(tiles[2:])
    row_width = max(top_row.shape[1], bottom_row.shape[1])
    top_row = pad_to_size(top_row, row_width, top_row.shape[0])
    bottom_row = pad_to_size(bottom_row, row_width, bottom_row.shape[0])
    spacer = np.zeros((PANEL_GAP, row_width, 3), dtype=np.uint8)
    canvas = cv2.vconcat([top_row, spacer, bottom_row])
    return scale_to_fit(canvas, MAX_VIEW_WIDTH, MAX_VIEW_HEIGHT)


def main():
    args = parse_args()

    raw_images_dir = (args.raw_dataset.resolve() / args.split / "images")
    gendeg_images_dir = (args.gendeg_dataset.resolve() / args.split / "images")

    if not raw_images_dir.exists():
        raise FileNotFoundError(f"Raw images directory not found: {raw_images_dir}")
    if not gendeg_images_dir.exists():
        raise FileNotFoundError(f"Generated images directory not found: {gendeg_images_dir}")

    raw_image_paths = load_image_paths(raw_images_dir)
    variant_map = build_variant_map(gendeg_images_dir)

    index = 0
    window_name = f"GenDeg Visualizer - {args.split}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        raw_image_path = raw_image_paths[index]
        variant_paths = variant_map.get(raw_image_path.stem, {})
        canvas = build_view(raw_image_path, variant_paths)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("a"):
            index = (index - 1) % len(raw_image_paths)
        elif key == ord("d"):
            index = (index + 1) % len(raw_image_paths)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
