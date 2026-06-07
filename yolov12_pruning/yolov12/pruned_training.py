import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


DATASET_YAML = Path("/home/a/capstone/yolov12/clear_dg_severe_train.yaml")
DEFAULT_OUTPUT_ROOT = Path("./ga_after_training")


def parse_args():
    parser = argparse.ArgumentParser(description="Retrain pruned YOLO models from GA outputs.")
    parser.add_argument("--epoch", type=int, default=600)
    parser.add_argument("--name", type=str, default="real_pruned")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--resume_path", type=str)
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


def is_run_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and path.name.startswith("g")
        and "_i" in path.name
        and (path / "best.yaml").is_file()
        and (path / "weights" / "best.pt").is_file()
    )


def collect_run_dirs(resume_path: Path):
    if is_run_dir(resume_path):
        return [resume_path]

    run_dirs = sorted(path for path in resume_path.iterdir() if is_run_dir(path))
    if run_dirs:
        return run_dirs

    raise FileNotFoundError(
        f"No valid pruned run directories found in: {resume_path}"
    )


def train_one(run_dir: Path, args):
    best_yaml = run_dir / "best.yaml"
    best_pt = run_dir / "weights" / "best.pt"
    output_dir = DEFAULT_OUTPUT_ROOT / run_dir.name

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(best_yaml)).load(str(best_pt))
    model.train(
        data=str(DATASET_YAML),
        epochs=args.epoch,
        batch=args.bs,
        device=args.device,
        project=str(DEFAULT_OUTPUT_ROOT),
        name=run_dir.name,
        imgsz=640,
        workers=8,
        resume=False,
        exist_ok=True,
    )

    shutil.copy2(best_yaml, output_dir / "best.yaml")
    print(f"[done] {run_dir.name} -> {output_dir}")


def main():
    args = parse_args()

    if not args.resume_path:
        raise ValueError("--resume_path is required.")
    if not DATASET_YAML.is_file():
        raise FileNotFoundError(f"Dataset yaml not found: {DATASET_YAML}")

    resume_path = Path(args.resume_path).resolve()
    if not resume_path.exists():
        raise FileNotFoundError(f"resume_path not found: {resume_path}")

    run_dirs = collect_run_dirs(resume_path)
    print(f"[info] found {len(run_dirs)} run(s) to train")

    for run_dir in run_dirs:
        print(f"[info] training {run_dir}")
        train_one(run_dir, args)


if __name__ == "__main__":
    main()
