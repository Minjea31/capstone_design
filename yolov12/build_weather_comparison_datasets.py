#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


@dataclass(frozen=True)
class SamplePair:
    stem: str
    image_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build comparison YOLO datasets by copying from existing sources."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional dataset names to build. If omitted, builds all configured datasets.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_nc_and_names(yaml_path: Path) -> tuple[int, list[str]]:
    nc: int | None = None
    names: list[str] | None = None

    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("nc:"):
            nc = int(line.split(":", 1)[1].strip())
        elif line.startswith("names:"):
            names = ast.literal_eval(line.split(":", 1)[1].strip())

    if nc is None or names is None:
        raise ValueError(f"Failed to parse nc/names from {yaml_path}")
    return nc, names


def find_image_paths(images_dir: Path) -> dict[str, Path]:
    image_paths: dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            image_paths[path.stem] = path
    return image_paths


def find_label_paths(labels_dir: Path) -> dict[str, Path]:
    label_paths: dict[str, Path] = {}
    for path in sorted(labels_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".txt":
            label_paths[path.stem] = path
    return label_paths


def collect_pairs(images_dir: Path, labels_dir: Path, source_name: str) -> list[SamplePair]:
    image_paths = find_image_paths(images_dir)
    label_paths = find_label_paths(labels_dir)

    missing_image_labels = sorted(set(label_paths) - set(image_paths))
    for stem in missing_image_labels:
        logging.warning(
            "[%s] Ignoring label without image: %s",
            source_name,
            label_paths[stem],
        )

    missing_label_images = sorted(set(image_paths) - set(label_paths))
    for stem in missing_label_images:
        logging.warning(
            "[%s] Missing label for image: %s",
            source_name,
            image_paths[stem],
        )

    valid_stems = sorted(set(image_paths) & set(label_paths))
    if not valid_stems:
        raise ValueError(f"[{source_name}] No valid image/label pairs found in {images_dir}")

    return [
        SamplePair(
            stem=stem,
            image_path=image_paths[stem],
            label_path=label_paths[stem],
        )
        for stem in valid_stems
    ]


def split_severe_pairs(severe_pairs: list[SamplePair]) -> tuple[list[SamplePair], list[SamplePair]]:
    shuffled = list(severe_pairs)
    random.Random(SEED).shuffle(shuffled)
    midpoint = len(shuffled) // 2
    severe_val = sorted(shuffled[:midpoint], key=lambda pair: pair.stem)
    severe_test = sorted(shuffled[midpoint:], key=lambda pair: pair.stem)
    return severe_val, severe_test


def ensure_empty_dataset_root(dataset_root: Path) -> None:
    if dataset_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset directory: {dataset_root}")

    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (dataset_root / split / kind).mkdir(parents=True, exist_ok=False)


def copy_pairs(
    pairs: list[SamplePair],
    dataset_root: Path,
    split: str,
    prefix: str,
    desc: str,
) -> None:
    image_dest_dir = dataset_root / split / "images"
    label_dest_dir = dataset_root / split / "labels"

    for pair in tqdm(pairs, desc=desc, unit="file"):
        dest_stem = f"{prefix}{pair.stem}"
        dest_image = image_dest_dir / f"{dest_stem}{pair.image_path.suffix.lower()}"
        dest_label = label_dest_dir / f"{dest_stem}.txt"

        try:
            shutil.copy2(pair.image_path, dest_image)
            shutil.copy2(pair.label_path, dest_label)
        except Exception:
            logging.exception(
                "Failed while copying stem '%s' from image=%s label=%s",
                pair.stem,
                pair.image_path,
                pair.label_path,
            )
            raise


def write_data_yaml(dataset_root: Path, nc: int, names: list[str]) -> None:
    yaml_text = "\n".join(
        [
            "train: train/images",
            "val: val/images",
            "test: test/images",
            "",
            f"nc: {nc}",
            f"names: {names}",
            "",
        ]
    )
    (dataset_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def write_root_yaml(dataset_root: Path, project_root: Path, nc: int, names: list[str]) -> None:
    yaml_text = "\n".join(
        [
            f"train: {dataset_root.name}/train/images",
            f"val: {dataset_root.name}/val/images",
            f"test: {dataset_root.name}/test/images",
            "",
            f"nc: {nc}",
            f"names: {names}",
            "",
        ]
    )
    (project_root / f"{dataset_root.name}.yaml").write_text(yaml_text, encoding="utf-8")


def count_split(dataset_root: Path, split: str) -> tuple[int, int]:
    images = sum(
        1
        for path in (dataset_root / split / "images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    labels = sum(
        1
        for path in (dataset_root / split / "labels").iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    )
    return images, labels


def write_severe_manifest(output_dir: Path, severe_val: list[SamplePair], severe_test: list[SamplePair]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": SEED,
        "severe_val": [pair.stem for pair in severe_val],
        "severe_test": [pair.stem for pair in severe_test],
    }
    (output_dir / "severe_split_seed42.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (output_dir / "severe_val_stems.txt").write_text(
        "\n".join(manifest["severe_val"]) + "\n",
        encoding="utf-8",
    )
    (output_dir / "severe_test_stems.txt").write_text(
        "\n".join(manifest["severe_test"]) + "\n",
        encoding="utf-8",
    )


def resolve_gendeg_variant(root: Path, variant: str) -> tuple[Path, Path]:
    candidate_dirs = {
        "haze": [
            root / "output_haze_final" / "output_haze_final",
            root / "output_haze_final",
        ],
        "rain": [root / "output_rain_final"],
        "raindrop": [root / "output_raindrop_final"],
    }[variant]

    for candidate in candidate_dirs:
        images_dir = candidate / "images"
        labels_dir = candidate / "labels"
        if images_dir.is_dir() and labels_dir.is_dir():
            return images_dir, labels_dir

    checked = ", ".join(str(path) for path in candidate_dirs)
    raise FileNotFoundError(f"Could not resolve GenDeg '{variant}' dataset. Checked: {checked}")


def build_dataset(
    dataset_root: Path,
    project_root: Path,
    nc: int,
    names: list[str],
    split_groups: dict[str, list[tuple[str, str, list[SamplePair]]]],
) -> dict[str, tuple[int, int]]:
    ensure_empty_dataset_root(dataset_root)

    for split, groups in split_groups.items():
        for source_name, prefix, pairs in groups:
            copy_pairs(
                pairs=pairs,
                dataset_root=dataset_root,
                split=split,
                prefix=prefix,
                desc=f"{dataset_root.name}:{split}:{source_name}",
            )

    write_data_yaml(dataset_root, nc, names)
    write_root_yaml(dataset_root, project_root, nc, names)
    return {split: count_split(dataset_root, split) for split in ("train", "val", "test")}


def build_dataset_definitions(
    clear_train: list[SamplePair],
    clear_val: list[SamplePair],
    clear_test: list[SamplePair],
    severe_val: list[SamplePair],
    severe_test: list[SamplePair],
    gendeg_haze: list[SamplePair],
    gendeg_rain: list[SamplePair],
    gendeg_raindrop: list[SamplePair],
) -> dict[str, dict[str, list[tuple[str, str, list[SamplePair]]]]]:
    gendeg_train_all = [
        ("clear_train", "clear_", clear_train),
        ("gendeg_haze", "gendeg_haze_", gendeg_haze),
        ("gendeg_rain", "gendeg_rain_", gendeg_rain),
        ("gendeg_raindrop", "gendeg_raindrop_", gendeg_raindrop),
    ]
    gendeg_train_no_raindrop = [
        ("clear_train", "clear_", clear_train),
        ("gendeg_haze", "gendeg_haze_", gendeg_haze),
        ("gendeg_rain", "gendeg_rain_", gendeg_rain),
    ]

    return {
        "baseline_test_severe": {
            "train": [("clear_train", "clear_", clear_train)],
            "val": [("clear_val", "clear_", clear_val), ("severe_val", "severe_", severe_val)],
            "test": [("severe_test", "severe_", severe_test)],
        },
        "baseline_test_mixed": {
            "train": [("clear_train", "clear_", clear_train)],
            "val": [("clear_val", "clear_", clear_val), ("severe_val", "severe_", severe_val)],
            "test": [("clear_test", "clear_", clear_test), ("severe_test", "severe_", severe_test)],
        },
        "gendeg_test_severe": {
            "train": gendeg_train_all,
            "val": [("clear_val", "clear_", clear_val), ("severe_val", "severe_", severe_val)],
            "test": [("severe_test", "severe_", severe_test)],
        },
        "gendeg_test_mixed": {
            "train": gendeg_train_all,
            "val": [("clear_val", "clear_", clear_val), ("severe_val", "severe_", severe_val)],
            "test": [("clear_test", "clear_", clear_test), ("severe_test", "severe_", severe_test)],
        },
        "gendeg_no_raindrop_test_severe": {
            "train": gendeg_train_no_raindrop,
            "val": [("clear_val", "clear_", clear_val), ("severe_val", "severe_", severe_val)],
            "test": [("severe_test", "severe_", severe_test)],
        },
        "gendeg_no_raindrop_test_mixed": {
            "train": gendeg_train_no_raindrop,
            "val": [("clear_val", "clear_", clear_val), ("severe_val", "severe_", severe_val)],
            "test": [("clear_test", "clear_", clear_test), ("severe_test", "severe_", severe_test)],
        },
    }


def main() -> None:
    args = parse_args()
    configure_logging()

    project_root = Path(__file__).resolve().parent
    cctv_root = project_root / "cctv_raw_dataset"
    severe_root = project_root / "severe" / "test"
    gendeg_root = project_root / "ys_gendeg_ Preprocessing"
    yaml_path = project_root / "dataset.yaml"
    manifest_dir = project_root / "generated_dataset_manifests"

    nc, names = load_nc_and_names(yaml_path)

    clear_train = collect_pairs(cctv_root / "train" / "images", cctv_root / "train" / "labels", "clear_train")
    clear_val = collect_pairs(cctv_root / "val" / "images", cctv_root / "val" / "labels", "clear_val")
    clear_test = collect_pairs(cctv_root / "test" / "images", cctv_root / "test" / "labels", "clear_test")
    severe_pairs = collect_pairs(severe_root / "images", severe_root / "labels", "severe")

    haze_images, haze_labels = resolve_gendeg_variant(gendeg_root, "haze")
    rain_images, rain_labels = resolve_gendeg_variant(gendeg_root, "rain")
    raindrop_images, raindrop_labels = resolve_gendeg_variant(gendeg_root, "raindrop")
    gendeg_haze = collect_pairs(haze_images, haze_labels, "gendeg_haze")
    gendeg_rain = collect_pairs(rain_images, rain_labels, "gendeg_rain")
    gendeg_raindrop = collect_pairs(raindrop_images, raindrop_labels, "gendeg_raindrop")

    severe_val, severe_test = split_severe_pairs(severe_pairs)
    write_severe_manifest(manifest_dir, severe_val, severe_test)

    dataset_definitions = build_dataset_definitions(
        clear_train=clear_train,
        clear_val=clear_val,
        clear_test=clear_test,
        severe_val=severe_val,
        severe_test=severe_test,
        gendeg_haze=gendeg_haze,
        gendeg_rain=gendeg_rain,
        gendeg_raindrop=gendeg_raindrop,
    )

    selected_names = args.datasets or list(dataset_definitions)
    unknown_names = sorted(set(selected_names) - set(dataset_definitions))
    if unknown_names:
        raise ValueError(f"Unknown dataset names: {', '.join(unknown_names)}")

    results: dict[str, dict[str, tuple[int, int]]] = {}
    for dataset_name in selected_names:
        split_groups = dataset_definitions[dataset_name]
        results[dataset_name] = build_dataset(
            dataset_root=project_root / dataset_name,
            project_root=project_root,
            nc=nc,
            names=names,
            split_groups=split_groups,
        )

    print("1. severe split result")
    print(f"severe_val images: {len(severe_val)}")
    print(f"severe_test images: {len(severe_test)}")
    print()

    for index, dataset_name in enumerate(selected_names, start=2):
        print(f"{index}. {dataset_name}")
        for split in ("train", "val", "test"):
            image_count, label_count = results[dataset_name][split]
            print(f"{split} images: {image_count}")
            print(f"{split} labels: {label_count}")
        print()


if __name__ == "__main__":
    main()
