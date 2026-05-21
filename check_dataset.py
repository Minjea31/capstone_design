#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Box:
    class_id: int
    x_center: float
    y_center: float
    box_width: float
    box_height: float


@dataclass
class ViewerState:
    boxes: list
    selected_index: int | None = None
    deleted_boxes: list | None = None
    scale: float = 1.0
    offset_x: int = 0
    offset_y: int = 0
    display_width: int = 0
    display_height: int = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO labels with keyboard navigation."
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to a dataset folder that contains images/ and labels/.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1280,
        help="OpenCV window width in pixels.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=720,
        help="OpenCV window height in pixels.",
    )
    return parser.parse_args()


def resolve_dirs(dataset_path: Path):
    dataset_path = dataset_path.expanduser().resolve()
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected images/ and labels/ under: {dataset_path}"
        )

    return images_dir, labels_dir


def load_images(images_dir: Path):
    image_paths = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No image files found in: {images_dir}")
    return image_paths


def yolo_to_xyxy(values, width, height):
    class_id, x_center, y_center, box_width, box_height = values

    x_center *= width
    y_center *= height
    box_width *= width
    box_height *= height

    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)

    return int(class_id), x1, y1, x2, y2


def load_boxes(label_path: Path):
    if not label_path.exists():
        return []

    boxes = []
    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            try:
                values = [float(value) for value in parts]
            except ValueError:
                continue

            boxes.append(
                Box(
                    class_id=int(values[0]),
                    x_center=values[1],
                    y_center=values[2],
                    box_width=values[3],
                    box_height=values[4],
                )
            )

    return boxes


def save_boxes(label_path: Path, boxes):
    with label_path.open("w", encoding="utf-8") as file:
        for box in boxes:
            file.write(
                f"{box.class_id} "
                f"{box.x_center:.6f} {box.y_center:.6f} "
                f"{box.box_width:.6f} {box.box_height:.6f}\n"
            )


def draw_boxes(image, boxes, selected_index):
    height, width = image.shape[:2]

    for index, box in enumerate(boxes):
        class_id, x1, y1, x2, y2 = yolo_to_xyxy(
            [
                box.class_id,
                box.x_center,
                box.y_center,
                box.box_width,
                box.box_height,
            ],
            width,
            height,
        )
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        color = (0, 0, 255) if index == selected_index else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    return image


def make_display_image(image, window_width, window_height, state: ViewerState):
    height, width = image.shape[:2]
    scale = min(window_width / width, window_height / height)
    scale = max(scale, 1e-6)

    display_width = max(1, int(width * scale))
    display_height = max(1, int(height * scale))
    resized = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_LINEAR)

    canvas = image.copy()
    if display_width != width or display_height != height:
        canvas = 255 * (cv2.cvtColor(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) * 0)
        canvas = cv2.copyMakeBorder(
            resized,
            top=max(0, (window_height - display_height) // 2),
            bottom=max(0, window_height - display_height - max(0, (window_height - display_height) // 2)),
            left=max(0, (window_width - display_width) // 2),
            right=max(0, window_width - display_width - max(0, (window_width - display_width) // 2)),
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    else:
        canvas = resized

    state.scale = scale
    state.display_width = display_width
    state.display_height = display_height
    state.offset_x = max(0, (window_width - display_width) // 2)
    state.offset_y = max(0, (window_height - display_height) // 2)

    return canvas


def select_box_at_point(x, y, image_width, image_height, state: ViewerState):
    if not (
        state.offset_x <= x < state.offset_x + state.display_width
        and state.offset_y <= y < state.offset_y + state.display_height
    ):
        state.selected_index = None
        return

    image_x = int((x - state.offset_x) / state.scale)
    image_y = int((y - state.offset_y) / state.scale)

    image_x = max(0, min(image_x, image_width - 1))
    image_y = max(0, min(image_y, image_height - 1))

    selected = None
    selected_area = None
    for index, box in enumerate(state.boxes):
        _, x1, y1, x2, y2 = yolo_to_xyxy(
            [
                box.class_id,
                box.x_center,
                box.y_center,
                box.box_width,
                box.box_height,
            ],
            image_width,
            image_height,
        )
        if x1 <= image_x <= x2 and y1 <= image_y <= y2:
            area = max(1, (x2 - x1) * (y2 - y1))
            if selected_area is None or area < selected_area:
                selected = index
                selected_area = area

    state.selected_index = selected


def build_mouse_callback(state: ViewerState, image_width: int, image_height: int):
    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            select_box_at_point(x, y, image_width, image_height, state)

    return on_mouse


def main():
    args = parse_args()
    images_dir, labels_dir = resolve_dirs(args.dataset_path)
    image_paths = load_images(images_dir)
    total_images = len(image_paths)

    index = 0
    window_name = "Dataset Checker"
    viewer_state = ViewerState(boxes=[], deleted_boxes=[])
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)

    while True:
        image_path = image_paths[index]
        label_path = labels_dir / f"{image_path.stem}.txt"
        cv2.setWindowTitle(window_name, f"{window_name} ({index + 1} / {total_images})")

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        viewer_state.boxes = load_boxes(label_path)
        viewer_state.selected_index = None
        viewer_state.deleted_boxes = []
        cv2.setMouseCallback(
            window_name,
            build_mouse_callback(viewer_state, image.shape[1], image.shape[0]),
        )

        while True:
            frame = image.copy()
            frame = draw_boxes(frame, viewer_state.boxes, viewer_state.selected_index)
            display = make_display_image(
                frame,
                args.window_width,
                args.window_height,
                viewer_state,
            )
            cv2.imshow(window_name, display)

            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue

            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return
            if key == ord("a"):
                index = (index - 1) % len(image_paths)
                break
            if key == ord("d"):
                index = (index + 1) % len(image_paths)
                break
            if key == ord("x") and viewer_state.selected_index is not None:
                deleted_index = viewer_state.selected_index
                deleted_box = viewer_state.boxes[deleted_index]
                viewer_state.deleted_boxes.append((deleted_index, deleted_box))
                del viewer_state.boxes[deleted_index]
                viewer_state.selected_index = None
                save_boxes(label_path, viewer_state.boxes)
            if key == ord("z") and viewer_state.deleted_boxes:
                restore_index, restore_box = viewer_state.deleted_boxes.pop()
                restore_index = max(0, min(restore_index, len(viewer_state.boxes)))
                viewer_state.boxes.insert(restore_index, restore_box)
                viewer_state.selected_index = restore_index
                save_boxes(label_path, viewer_state.boxes)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
