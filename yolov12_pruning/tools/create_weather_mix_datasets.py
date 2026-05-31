#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENDEG_ROOT = Path("/home/a/capstone/yolov12/ys_gendeg_ Preprocessing")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}


@dataclass(frozen=True)
class Pair:
    image: Path
    label: Path


@dataclass(frozen=True)
class Source:
    name: str
    pairs: list[Pair]


def image_files(image_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def collect_pairs(image_dir: Path, label_dir: Path) -> list[Pair]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    pairs: list[Pair] = []
    missing: list[Path] = []
    for image in image_files(image_dir):
        rel = image.relative_to(image_dir)
        label = (label_dir / rel).with_suffix(".txt")
        if label.exists():
            pairs.append(Pair(image=image, label=label))
        else:
            missing.append(label)

    if missing:
        preview = "\n".join(str(p) for p in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} labels are missing for images under {image_dir}.\n"
            f"First missing labels:\n{preview}"
        )
    return pairs


def collect_clear() -> Source:
    root = PROJECT_ROOT / "cctv_raw_dataset"
    pairs: list[Pair] = []
    for split in SPLITS:
        pairs.extend(collect_pairs(root / split / "images", root / split / "labels"))
    return Source(name="clear", pairs=pairs)


def collect_weather(weather: str) -> Source:
    roots = {
        "rain": GENDEG_ROOT / "output_rain_final",
        "raindrop": GENDEG_ROOT / "output_raindrop_final",
        "haze": GENDEG_ROOT / "output_haze_final" / "output_haze_final",
    }
    root = roots[weather]
    return Source(
        name=weather,
        pairs=collect_pairs(root / "images", root / "labels"),
    )


def split_counts(total: int) -> dict[str, int]:
    floors = {split: int(total * RATIOS[split]) for split in SPLITS}
    remaining = total - sum(floors.values())
    fractions = {
        split: (total * RATIOS[split]) - floors[split]
        for split in SPLITS
    }

    counts = dict(floors)
    for split in sorted(SPLITS, key=lambda s: (-fractions[s], SPLITS.index(s)))[:remaining]:
        counts[split] += 1
    return counts


def split_pairs(source: Source, seed: int) -> dict[str, list[Pair]]:
    rng = random.Random(seed)
    pairs = list(source.pairs)
    rng.shuffle(pairs)

    counts = split_counts(len(pairs))
    start = 0
    output: dict[str, list[Pair]] = {}
    for split in SPLITS:
        end = start + counts[split]
        output[split] = pairs[start:end]
        start = end
    return output


def copy_pair(pair: Pair, output_root: Path, split: str, source_name: str, index: int) -> None:
    image_name = f"{source_name}_{index:06d}__{pair.image.name}"
    label_name = f"{Path(image_name).stem}.txt"

    image_dst = output_root / split / "images" / image_name
    label_dst = output_root / split / "labels" / label_name

    shutil.copy2(pair.image, image_dst)
    shutil.copy2(pair.label, label_dst)


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


def write_yaml(dataset_name: str) -> None:
    yaml_path = PROJECT_ROOT / f"{dataset_name}.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_name}_dataset",
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


def build_dataset(clear: Source, weather: Source, seed: int, overwrite: bool) -> None:
    dataset_name = f"clear_{weather.name}"
    output_root = PROJECT_ROOT / f"{dataset_name}_dataset"
    prepare_output_dirs(output_root, overwrite=overwrite)

    split_sources = {
        "clear": split_pairs(clear, seed),
        weather.name: split_pairs(weather, seed),
    }

    summary: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    for source_name, split_map in split_sources.items():
        for split in SPLITS:
            for index, pair in enumerate(split_map[split], start=1):
                copy_pair(pair, output_root, split, source_name, index)
            summary[split][source_name] = len(split_map[split])

    write_yaml(dataset_name)

    print(f"{dataset_name}_dataset")
    for split in SPLITS:
        parts = ", ".join(
            f"{source_name}={count}"
            for source_name, count in summary[split].items()
        )
        print(f"  {split}: {parts}, total={sum(summary[split].values())}")
    print(f"  yaml: {dataset_name}.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create clear+weather YOLO datasets with independent 6:2:2 splits."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clear = collect_clear()
    print(f"clear source: {len(clear.pairs)} images")

    for weather_name in ("rain", "raindrop", "haze"):
        weather = collect_weather(weather_name)
        print(f"{weather_name} source: {len(weather.pairs)} images")
        build_dataset(clear, weather, seed=args.seed, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
