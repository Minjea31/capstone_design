#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv12 layerwise pruning ratio search with Genetic Algorithm.

- Gene: per-layer pruning ratio in [0,1] for each Conv2d slated for pruning
- Fitness: quick eval (mAP50) - optional compression penalty (params/FLOPs)
- Output: best_ratio_map.json (and optionally best-pruned model)
"""

import os
import json
import copy
import csv
import argparse
import warnings
import sys
from datetime import datetime
from typing import List, Dict

warnings.simplefilter("ignore", UserWarning)


import numpy as np
import torch
import pygad
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.torch_utils import get_flops

# robust import whether run as "python compress/GA.py" or "python -m compress.GA"
from compress.Compress import PruneHandler


def parse_args():
    p = argparse.ArgumentParser(description="GA-based layerwise pruning search for YOLOv12")

    # Required paths
    p.add_argument("--bmodel", type=str, default='/home/a/capstone/model/last_model.pt', help="Base YOLOv12 .pt path (unpruned)")
    p.add_argument("--data", type=str, default='/home/a/capstone/yolov12/clear_dg_severe_train.yaml', help="data.yaml (should include a val set)")

    # Outputs
    p.add_argument("--cfg_out", type=str, default="./ga_out", help="Dir for intermediate outputs & ratio map")

    # Pruning settings
    p.add_argument("--method", type=str, default="GM", choices=["GM", "L1", "L2"])
    p.add_argument("--prune_type", type=str, default="ALL", choices=["ALL", "H", "B"])

    # GA hyperparams
    
    p.add_argument("--generations", type=int, default=20) # 세대
    p.add_argument("--parents", type=int, default=10) # 다음 세대에 몇개를 고정할지
    p.add_argument("--pop", type=int, default=50) # 후보군의 갯수
    p.add_argument("--init_low", type=float, default=0.2)
    p.add_argument("--init_high", type=float, default=0.8)
    p.add_argument("--mutation_prob", type=float, default=0.2)
    p.add_argument("--crossover_prob", type=float, default=0.9)

    p.add_argument("--ft_epochs", type=int, default=5,
               help="After pruning, run YOLO.train() for N epochs before val(). "
                    "Set 0 to skip finetuning.")

    # Eval / resources
    p.add_argument("--device", type=str, default="0", help="device for YOLO.val(), e.g., '0' or 'cpu'")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8,
                   help="Number of DataLoader workers for train/val. Set 0 to avoid multiprocessing contention.")
    # 매 객체마다 새로운 객치를 불러올건지 / 아니면 deepcopy를 할건지 
    p.add_argument("--reload_each", action="store_true",
                   help="Reload base model for each fitness call (safer, slower). Default: deepcopy base once.")

    # Compression target (optional)
    p.add_argument("--target_cr", type=float, default=0.5,
                   help="If set, add penalty by |target_cr - current_cr| using params/FLOPs")
    p.add_argument("--cr_lambda", type=float, default=0.8,
                   help="Penalty weight lambda")

    # Save best pruned model (optional)
    p.add_argument("--save_best_model", action="store_true",
                   help="After GA ends, prune once with best ratio & save")

    # FLOPs penalty (optional)
    p.add_argument("--use_flops", action="store_true",
                   help="Try to use THOP to measure FLOPs; falls back to params if unavailable")
    return p.parse_args()

# Conv 모듈을 뽑아옴.
def list_prunable_convs(yolo_model: YOLO) -> List[str]:
    """
    Mirror Compress.py's pruning loop rule for Conv selection.
    Current rule excludes names containing '21'.
    """
    names = []
    for name, m in yolo_model.model.model.named_modules():
        if isinstance(m, torch.nn.Conv2d) and ('21' not in name):
            names.append(name)
    return names


def genes_to_ratio_map(solution: np.ndarray, layer_names: List[str], args) -> Dict[str, float]:
    # 0.3~0.8 까지만 허용
    g = np.clip(np.array(solution, dtype=np.float32), args.init_low, args.init_high)
    g = np.round(g, 2)
    return {layer_names[i]: round(float(g[i]), 2) for i in range(len(layer_names))}


def count_params(yolo_model: YOLO) -> int:
    # return int(sum(p.numel() for p in yolo_model.model.parameters() if p.requires_grad))
    return int(sum(p.numel() for p in yolo_model.model.parameters()))  # ← 이걸로


def init_results_csv(csv_path: str):
    if os.path.isfile(csv_path):
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "mAP50", "inferenceTime_ms", "FLOPs_GFLOPs"])


def append_result_row(csv_path: str, name: str, map50: float, inference_ms: float, flops_g: float):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, f"{map50:.2f}", f"{inference_ms:.1f}", f"{flops_g:.1f}"])


# 파인튜닝과 검증을 한 번에 수행
def evaluate_model(yolo_model, args, gen: int, ind: int) -> Dict[str, float]:
    """
    Pruned YOLO 모델을 args.ft_epochs만큼 학습한 뒤, 저장된 YAML/PT를 다시 로드해서
    eval.py와 같은 방식으로 val()을 돌려 mAP50을 반환한다.
    학습/검증이 실패하면 0.0을 리턴한다.
    """
    run_root = os.path.join(args.cfg_out, "ft_runs")
    run_name = f"g{gen:03d}_i{ind:03d}"
    run_dir = os.path.join(run_root, run_name)
    model_yaml = os.path.join(run_dir, "best.yaml")
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    last_pt = os.path.join(run_dir, "weights", "last.pt")
    eval_model = yolo_model
    default_result = {"map50": 0.0, "inference_ms": 0.0, "flops_g": 0.0}

    if args.ft_epochs and args.ft_epochs > 0:
        original_validate = BaseTrainer.validate
        original_final_eval = BaseTrainer.final_eval

        def _skip_validate(self):
            fitness = 0.0
            if self.metrics is None:
                self.metrics = {}
            if self.best_fitness is None or self.best_fitness < fitness:
                self.best_fitness = fitness
            return self.metrics, fitness

        def _skip_final_eval(self):
            return None

        try:
            BaseTrainer.validate = _skip_validate
            BaseTrainer.final_eval = _skip_final_eval
            try:
                yolo_model.train(
                    data=args.data,
                    epochs=args.ft_epochs,
                    batch=args.batch,
                    device=args.device,
                    workers=args.workers,
                    project=run_root,
                    name=run_name,
                    exist_ok=True,
                    verbose=False,
                    plots=False,
                    val=False,
                )
            except TypeError:
                yolo_model.train(
                    data=args.data,
                    epochs=args.ft_epochs,
                    batch=args.batch,
                    device=args.device,
                    workers=args.workers,
                    project=run_root,
                    name=run_name,
                    exist_ok=True,
                    val=False,
                )
        except Exception as e:
            print(f"[warn] finetune failed: {e}")
            return default_result
        finally:
            BaseTrainer.validate = original_validate
            BaseTrainer.final_eval = original_final_eval

        model_pt = best_pt if os.path.isfile(best_pt) else last_pt
        if not os.path.isfile(model_yaml):
            print(f"[warn] eval yaml not found: {model_yaml}")
            return default_result
        if not os.path.isfile(model_pt):
            print(f"[warn] eval weights not found: {model_pt}")
            return default_result

        try:
            eval_model = YOLO(model_yaml).load(model_pt)
        except Exception as e:
            print(f"[warn] reload for eval failed: {e}")
            return default_result

    try:
        r = eval_model.val(
            data=args.data,
            imgsz=640,
            batch=args.batch,
            plots=False,
            verbose=False,
            workers=args.workers,
        )
    except Exception as e:
        print(f"[warn] val failed: {e}")
        return default_result

    inference_ms = float(getattr(r, "speed", {}).get("inference", 0.0))
    try:
        flops_g = float(get_flops(eval_model.model, imgsz=640) or 0.0)
    except Exception:
        flops_g = 0.0

    rd = getattr(r, "results_dict", None)
    if isinstance(rd, dict):
        for k in ("metrics/mAP50", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/mAP50-95"):
            if k in rd:
                return {"map50": float(rd[k]), "inference_ms": inference_ms, "flops_g": flops_g}
    return {"map50": 0.0, "inference_ms": inference_ms, "flops_g": flops_g}



def make_fitness(base_loader, args, layer_names: List[str], base_params: int):
    record_map: Dict[tuple, Dict[str, object]] = {}
    solution_map: Dict[tuple, Dict[str, object]] = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(args.cfg_out, f"ga_results_{timestamp}.csv")
    init_results_csv(results_csv)

    def fitness_func(ga_instance, solution, solution_idx):
        gens_done = getattr(ga_instance, "generations_completed", 0)
        gen_cur   = min(gens_done + 1, ga_instance.num_generations)
        ind_cur   = solution_idx + 1
        ind_total = ga_instance.sol_per_pop
        kp        = getattr(ga_instance, "keep_parents", 0)

        # ----- (추가) 개체별 run 경로 구성 -----
        run_root = os.path.join(args.cfg_out, "ft_runs")
        run_name = f"g{gen_cur:03d}_i{ind_cur:03d}"
        run_dir  = os.path.join(run_root, run_name)
        os.makedirs(run_dir, exist_ok=True)  # YAML이 저장될 상위 폴더(=weights의 부모)

        # ===== 유전자 -> 레이어별 비율 맵 =====
        ratio_map = genes_to_ratio_map(solution, layer_names, args)

        # ===== 베이스 모델 로드 =====
        model = base_loader()

        # ===== PruneHandler 생성 =====
        try:
            handler = PruneHandler(
                model,
                compression_ratio=0.0,
                method=args.method,
                # ★ 변경: 공용 폴더가 아니라 '그 개체의 run 폴더'로 지정
                cfg_output_path=run_dir,
                prune_type=args.prune_type,
                ratio_map=ratio_map
            )
        except TypeError:
            warnings.warn(
                "[GA] PruneHandler doesn't accept ratio_map. "
                "Falling back to global ratio (NOT layerwise). Patch Compress.py to support ratio_map!"
            )
            handler = PruneHandler(
                model,
                compression_ratio=0.5,
                method=args.method,
                # ★ 동일하게 run_dir로
                cfg_output_path=run_dir,
                prune_type=args.prune_type
            )

        # ===== 프루닝 + 재구성 =====
        pruned = handler.compress_yolov12()

        # ===== 정확도 =====
        # mAP (train()은 기존처럼 run_root/name으로 저장 -> weights/*.pt)
        eval_result = evaluate_model(
            pruned, args,
            gen=gen_cur,
            ind=ind_cur
        )
        acc = eval_result["map50"]
        inference_ms = eval_result["inference_ms"]
        flops_g = eval_result["flops_g"]

        # ===== 압축비 패널티(옵션) =====
        cur_cr = None
        score = acc
        if args.target_cr is not None:
            cur_cr = 1.0 - (count_params(pruned) / base_params)
            score = acc - args.cr_lambda * abs(args.target_cr - cur_cr)

        cr_str = f"{cur_cr:.4f}" if cur_cr is not None else "N/A"
        print(f"[GA]     acc={acc:.6f} | CR={cr_str} | fitness={score:.6f}\n", flush=True)
        append_result_row(results_csv, run_name, acc, inference_ms, flops_g)

        record = {
            "score": float(score),
            "acc": float(acc),
            "inference_ms": float(inference_ms),
            "flops_g": float(flops_g),
            "cur_cr": cur_cr,
            "gen": gen_cur,
            "ind": ind_cur,
            "solution": np.array(solution, dtype=np.float32).copy(),
        }
        idx_key = (gen_cur, ind_cur)
        prev = record_map.get(idx_key)
        if prev is None or record["score"] > prev["score"]:
            record_map[idx_key] = record

        sol_key = tuple(np.round(record["solution"], 4))
        prev_sol = solution_map.get(sol_key)
        if prev_sol is None or record["score"] > prev_sol["score"]:
            solution_map[sol_key] = record

        return float(score)

    fitness_func.record_map = record_map
    fitness_func.solution_map = solution_map
    return fitness_func

def main():
    args = parse_args()
    os.makedirs(args.cfg_out, exist_ok=True)

    gene_values = np.round(np.arange(args.init_low, args.init_high + 1e-6, 0.01), 2).tolist()

    # Load base model once
    print("[info] loading base model...")
    base = YOLO(args.bmodel)

    # Determine prunable Conv layers (order must remain stable)
    layer_names = list_prunable_convs(base)
    if not layer_names:
        raise RuntimeError("No prunable Conv2d layers found. Check pruning filter (e.g., '21' exclusion).")
    print(f"[info] #prunable convs: {len(layer_names)}")

    base_params = count_params(base)
    print(f"[info] base params: {base_params}")

    # Base loaders
    def load_by_deepcopy():
        return copy.deepcopy(base)

    def load_by_reload():
        return YOLO(args.bmodel)
    # reload_each가 true이면 새로운 모델, false이면 deepcopy
    base_loader = load_by_reload if args.reload_each else load_by_deepcopy

    # Build GA
    fitness_func = make_fitness(base_loader, args, layer_names, base_params)
    num_genes = len(layer_names)

    print("[info] GA search start")
    ga = pygad.GA(
    num_generations=args.generations,
    num_parents_mating=5,   # 교차에 참여할 부모 수
    sol_per_pop=args.pop, # 후보군 갯수
    num_genes=num_genes, # conv 모듈의 갯수
    fitness_func=fitness_func,
    
    gene_space = gene_values,

    # 유전자 초기 범위
    init_range_low=args.init_low,
    init_range_high=args.init_high,
    
    parent_selection_type="rank",
    
    # 부모 세대 유지 갯수
    keep_elitism = args.parents,
    
    
    # 변이/교차
    crossover_type="uniform",
    crossover_probability=args.crossover_prob,
    
    mutation_type="random",
    mutation_probability=args.mutation_prob,
    mutation_percent_genes = 20,
    
    allow_duplicate_genes=True
    )

    ga.run()

    """
    # === 플로팅 ===
    """
    # 1) 세대별 best fitness 변화
    ga.plot_fitness()

    # # 2) 세대별 유전자 값 변화
    # ga.plot_genes()

    # # 3) 세대별 새로운 solution 탐색률
    # ga.plot_new_solution_rate()
    


    best_sol, best_fit, _ = ga.best_solution()
    solution_map = getattr(fitness_func, "solution_map", {})
    all_records = sorted(solution_map.values(), key=lambda r: r["score"], reverse=True)
    best_sol_arr = np.array(best_sol, dtype=np.float32)
    best_record = None
    for rec in all_records:
        if np.allclose(rec["solution"], best_sol_arr):
            best_record = rec
            break
    if best_record is None and all_records:
        best_record = all_records[0]

    # ✅ 여기서 GA 끝난 직후 결과 출력
    print("[info] GA finished", flush=True)
    if best_record is not None:
        best_line = (
            f"[info] best candidate -> gen {best_record['gen']:03d}, idx {best_record['ind']:03d}, "
            f"fitness {best_record['score']:.6f}, mAP {best_record['acc']:.6f}"
        )
        if best_record["cur_cr"] is not None:
            best_line += f", CR {best_record['cur_cr']:.4f}"
        print(best_line, flush=True)
    print(f"[info] best fitness: {best_fit:.6f}", flush=True)
    if all_records:
        print("[info] top-5 candidates (fitness desc):", flush=True)
        for rank, rec in enumerate(all_records[:5], start=1):
            line = (
                f"        {rank}위 -> gen {rec['gen']:03d}, idx {rec['ind']:03d}, "
                f"fitness {rec['score']:.6f}, mAP {rec['acc']:.6f}"
            )
            if rec["cur_cr"] is not None:
                line += f", CR {rec['cur_cr']:.4f}"
            print(line, flush=True)
    print(f"[info] saved ratio map: {args.cfg_out}/best_ratio_map.json", flush=True)
    print("[info] pruning once more with best ratio and saving...", flush=True)


    # Save best ratio map
    best_ratio = genes_to_ratio_map(best_sol, layer_names, args)
    ratio_path = os.path.join(args.cfg_out, "best_ratio_map.json")
    with open(ratio_path, "w") as f:
        json.dump(best_ratio, f, indent=2)
    print("[info] saved ratio map:", ratio_path)

    # Optionally prune once more with best ratio (to save pt/yaml cleanly)
    if args.save_best_model:
        print("[info] pruning once more with best ratio and saving...")
        y = YOLO(args.bmodel)
        try:
            handler = PruneHandler(
                y,
                compression_ratio=0.0,  # dummy
                method=args.method,
                cfg_output_path=args.cfg_out,
                prune_type=args.prune_type,
                ratio_map=best_ratio
            )
        except TypeError:
            warnings.warn(
                "[GA] PruneHandler doesn't accept ratio_map. "
                "Saving best model with global ratio (NOT layerwise)."
            )
            handler = PruneHandler(
                y,
                compression_ratio=0.5,  # fallback only
                method=args.method,
                cfg_output_path=args.cfg_out,
                prune_type=args.prune_type
            )
        _ = handler.compress_yolov12()
        print("[info] best pruned model saved under:", args.cfg_out)


if __name__ == "__main__":
    main()
