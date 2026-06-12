#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}


@dataclass(frozen=True)
class Pair:
    image: Path
    label: Path
    relative_image: Path


def image_files(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def collect_pairs(image_dir: Path, label_dir: Path) -> list[Pair]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    pairs: list[Pair] = []
    missing_labels: list[Path] = []
    for image in image_files(image_dir):
        relative_image = image.relative_to(image_dir)
        label = (label_dir / relative_image).with_suffix(".txt")
        if label.exists():
            pairs.append(Pair(image=image, label=label, relative_image=relative_image))
        else:
            missing_labels.append(label)

    if missing_labels:
        preview = "\n".join(str(path) for path in missing_labels[:5])
        raise FileNotFoundError(
            f"{len(missing_labels)} labels are missing for images under {image_dir}.\n"
            f"First missing labels:\n{preview}"
        )

    image_stems = {
        path.relative_to(image_dir).with_suffix("")
        for path in image_files(image_dir)
    }
    labels_without_images = [
        path
        for path in label_dir.rglob("*.txt")
        if path.relative_to(label_dir).with_suffix("") not in image_stems
    ]
    if labels_without_images:
        preview = "\n".join(str(path) for path in labels_without_images[:5])
        raise FileNotFoundError(
            f"{len(labels_without_images)} labels do not have matching images under {image_dir}.\n"
            f"First unmatched labels:\n{preview}"
        )

    return pairs


def split_counts(total: int) -> dict[str, int]:
    floors = {split: int(total * RATIOS[split]) for split in SPLITS}
    remaining = total - sum(floors.values())
    fractions = {
        split: total * RATIOS[split] - floors[split]
        for split in SPLITS
    }

    counts = dict(floors)
    for split in sorted(SPLITS, key=lambda item: (-fractions[item], SPLITS.index(item)))[:remaining]:
        counts[split] += 1
    return counts


def split_pairs(pairs: list[Pair], seed: int) -> dict[str, list[Pair]]:
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    counts = split_counts(len(shuffled))
    start = 0
    split_map: dict[str, list[Pair]] = {}
    for split in SPLITS:
        end = start + counts[split]
        split_map[split] = shuffled[start:end]
        start = end
    return split_map


def collect_raw_split(root: Path) -> dict[str, list[Pair]]:
    return {
        split: collect_pairs(root / split / "images", root / split / "labels")
        for split in SPLITS
    }


def prepare_output_dirs(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_root} already exists. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    for split in SPLITS:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_pair(pair: Pair, output_root: Path, split: str, source_name: str) -> None:
    image_dst = output_root / split / "images" / source_name / pair.relative_image
    label_dst = (output_root / split / "labels" / source_name / pair.relative_image).with_suffix(".txt")
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pair.image, image_dst)
    shutil.copy2(pair.label, label_dst)


def copy_source(
    split_map: dict[str, list[Pair]],
    output_root: Path,
    source_name: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split, pairs in split_map.items():
        for pair in pairs:
            copy_pair(pair, output_root, split, source_name)
        counts[split] = len(pairs)
    return counts


def write_yaml(output_root: Path, yaml_path: Path) -> None:
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.name}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "",
                "nc: 1",
                "names: ['car']",
                "",
            ]
        ),
        encoding="utf-8",
    )


def verify_dataset(output_root: Path) -> dict[str, int]:
    totals: dict[str, int] = {}
    for split in SPLITS:
        image_dir = output_root / split / "images"
        label_dir = output_root / split / "labels"
        images = image_files(image_dir)
        labels = sorted(path for path in label_dir.rglob("*.txt") if path.is_file())

        image_stems = {
            path.relative_to(image_dir).with_suffix("")
            for path in images
        }
        label_stems = {
            path.relative_to(label_dir).with_suffix("")
            for path in labels
        }
        if image_stems != label_stems:
            missing_labels = sorted(image_stems - label_stems)
            missing_images = sorted(label_stems - image_stems)
            raise RuntimeError(
                f"{split} stem mismatch: "
                f"missing_labels={len(missing_labels)}, missing_images={len(missing_images)}"
            )
        totals[split] = len(images)
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create mix_dataset by combining raw_data_dataset with a 6:2:2 haze_demo split."
    )
    parser.add_argument("--raw", type=Path, default=PROJECT_ROOT / "raw_data_dataset")
    parser.add_argument("--haze", type=Path, default=PROJECT_ROOT / "haze_demo")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "mix_dataset")
    parser.add_argument("--yaml", type=Path, default=PROJECT_ROOT / "mix_data.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_split = collect_raw_split(args.raw)
    haze_pairs = collect_pairs(args.haze / "train" / "images", args.haze / "train" / "labels")
    haze_split = split_pairs(haze_pairs, args.seed)

    prepare_output_dirs(args.output, overwrite=args.overwrite)
    raw_counts = copy_source(raw_split, args.output, "raw_data_dataset")
    haze_counts = copy_source(haze_split, args.output, "haze_demo")
    write_yaml(args.output, args.yaml)
    totals = verify_dataset(args.output)

    print(f"raw source: {args.raw}")
    print(f"haze source: {args.haze}")
    print(f"output: {args.output}")
    print(f"seed: {args.seed}")
    for split in SPLITS:
        print(
            f"{split}: raw={raw_counts[split]}, "
            f"haze={haze_counts[split]}, total={totals[split]}"
        )
    print(f"yaml: {args.yaml}")


if __name__ == "__main__":
    main()
