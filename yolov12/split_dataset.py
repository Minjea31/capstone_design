#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a YOLO-style flat dataset into train/val/test directories."
    )
    parser.add_argument("dataset_dir", type=Path, help="Dataset root containing images/ and labels/")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.7, 0.2, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios for train, val, and test",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them",
    )
    return parser.parse_args()


def collect_pairs(images_dir: Path, labels_dir: Path) -> list[str]:
    image_stems = {
        path.stem
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    label_stems = {path.stem for path in labels_dir.iterdir() if path.is_file() and path.suffix == ".txt"}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)
    if missing_labels or missing_images:
        problems = []
        if missing_labels:
            problems.append(f"missing labels for {len(missing_labels)} images")
        if missing_images:
            problems.append(f"missing images for {len(missing_images)} labels")
        raise SystemExit("Dataset pair mismatch: " + ", ".join(problems))

    stems = sorted(image_stems)
    if not stems:
        raise SystemExit("No image/label pairs found to split.")
    return stems


def prepare_output_dirs(dataset_dir: Path) -> None:
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            split_dir = dataset_dir / split / kind
            split_dir.mkdir(parents=True, exist_ok=True)
            leftovers = list(split_dir.iterdir())
            if leftovers:
                raise SystemExit(f"Refusing to overwrite non-empty directory: {split_dir}")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"{dataset_dir} must contain flat images/ and labels/ directories")

    ratios = args.ratios
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise SystemExit("Ratios must be positive and sum to 1.0")

    stems = collect_pairs(images_dir, labels_dir)
    prepare_output_dirs(dataset_dir)

    rng = random.Random(args.seed)
    rng.shuffle(stems)

    total = len(stems)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    split_map = {
        "train": stems[:train_end],
        "val": stems[train_end:val_end],
        "test": stems[val_end:],
    }

    transfer = shutil.move if args.move else shutil.copy2

    image_lookup = {path.stem: path for path in images_dir.iterdir() if path.is_file()}
    label_lookup = {path.stem: path for path in labels_dir.iterdir() if path.is_file() and path.suffix == ".txt"}

    for split, split_stems in split_map.items():
        for stem in split_stems:
            transfer(image_lookup[stem], dataset_dir / split / "images" / image_lookup[stem].name)
            transfer(label_lookup[stem], dataset_dir / split / "labels" / label_lookup[stem].name)

    if args.move:
        for source_dir in (images_dir, labels_dir):
            for leftover in source_dir.iterdir():
                raise SystemExit(f"Unexpected leftover file after move: {leftover}")

    print(
        f"Split {total} pairs with seed={args.seed}: "
        f"train={len(split_map['train'])}, val={len(split_map['val'])}, test={len(split_map['test'])}"
    )


if __name__ == "__main__":
    main()
