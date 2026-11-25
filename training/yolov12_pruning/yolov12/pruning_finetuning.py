from ultralytics import YOLO
from compress.Compress import PruneHandler as ph
import argparse, os, json, warnings

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

parser = argparse.ArgumentParser()
parser.add_argument('--bmodel', type=str, default=str(REPOSITORY_ROOT / 'models/experiments/raw_best.pt'))
parser.add_argument('--cfg_output_path', type=str, default=str(REPOSITORY_ROOT / 'training/yolov12_pruning/yolov12/layer_wise'))
parser.add_argument('--data', type=str, default=str(REPOSITORY_ROOT / 'training/yolov12_pruning/dataset.yaml'))

# GA를 안쓸경우 전역 프루닝 목표 값.
parser.add_argument('--compress_ratio', type=float, default='0.5')
parser.add_argument('--prune_type', type=str, default='ALL', help="H or B or ALL")
parser.add_argument('--method', type=str, default='GM', help="L1 or L2 or GM")
parser.add_argument('--ratio_map', type=str, default=str(REPOSITORY_ROOT / 'training/yolov12_pruning/ga_out/best_ratio_map.json'), help="(optional) path to best_ratio_map.json from GA")


parser.add_argument('--epoch', type=int, default=600)
parser.add_argument('--name', type=str, default='test')
parser.add_argument('--bs', type=int, default=8)
parser.add_argument('--resume_path', type=str)
parser.add_argument("--device", type=str, default='0')

args = parser.parse_args()

def load_ratio_map(args):
    if args.ratio_map and os.path.isfile(args.ratio_map):
        path = args.ratio_map
    else:
        auto = os.path.join(args.cfg_output_path, "best_ratio_map.json")
        path = auto if os.path.isfile(auto) else None

    if path:
        with open(path, "r") as f:
            rm = json.load(f)
        print(f"[info] loaded ratio_map: {path} (layers={len(rm)})")
        return rm
    print("[info] ratio_map not found. Use global compress_ratio only.")
    return None


if not args.resume_path:
    base = YOLO(args.bmodel)
    ratio_map = load_ratio_map(args)

    try:
        handler = ph(base,
                     compression_ratio=args.compress_ratio,
                     method=args.method,
                     cfg_output_path=args.cfg_output_path,
                     prune_type=args.prune_type,
                     ratio_map=ratio_map)
    except TypeError:
        warnings.warn("[warn] PruneHandler에 ratio_map 인자가 없네요. 전역 compress_ratio로만 동작합니다.")
        handler = ph(base,
                     compression_ratio=args.compress_ratio,
                     method=args.method,
                     cfg_output_path=args.cfg_output_path,
                     prune_type=args.prune_type)

    cmodel = handler.compress_yolov12()

    # 항상 파인튜닝
    cmodel.train(data=args.data, epochs=args.epoch, batch=args.bs,
                 device=args.device, name=args.name, project='checkpoints',
                 imgsz=640, workers=8, resume=False)

else:
    cmodel = YOLO(args.resume_path)
    cmodel.train(data=args.data, epochs=50, batch=args.bs,
                 device=args.device, name=args.name, project='checkpoints',
                 imgsz=640, workers=8, resume=True)