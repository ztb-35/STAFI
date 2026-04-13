#!/usr/bin/env python3
"""Evaluate op_089 candidate bit flips for acceleration and curvature changes."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Sequence

import torch
from tqdm import tqdm

from recurrent_bit_utils import (
    build_eval_loader,
    cache_eval_batches,
    evaluate_recurrent_metrics,
    flip_scalar_bit_fast_,
    live_params,
    load_flip_rows_json,
    load_model,
    validate_target_mode,
)


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser("Evaluate op_089 candidate bit flips on accel/curvature metrics")
    ap.add_argument("--data-root", default="data/comma2k19/")
    ap.add_argument("--val-index", default=os.path.join(root, "data", "comma2k19_val_non_overlap.txt"))
    ap.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1)
    ap.add_argument("--recurrent-num", "--recurrent_num", dest="recurrent_num", type=int, default=100)
    ap.add_argument("--num-val-batches", "--num_val_batches", dest="num_val_batches", type=int, default=2)
    ap.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=0)
    ap.add_argument("--ckpt", default=os.path.join(root, "openpilot_model", "supercombo_torch_weights.pth"))
    ap.add_argument("--input-json", "--input_json", dest="input_json", required=True)
    ap.add_argument("--flip-count", "--flip_count", dest="flip_count", type=int, default=10)
    ap.add_argument("--eval-mode", "--mode", dest="eval_mode", choices=["independent", "cumulative"], default="independent")
    ap.add_argument("--steer-actuator-delay", "--steer_actuator_delay", dest="steer_actuator_delay", type=float, default=0.0)
    ap.add_argument("--target-mode", "--target_mode", dest="target_mode", default="pseudo_controls")
    ap.add_argument("--out", default=os.path.join(root, "op_089", "out", "candidate_bits_eval_089.json"))
    ap.add_argument("--amp", action="store_true")
    return ap.parse_args()


def evaluate_independent(
    model: torch.nn.Module,
    cached_batches: Sequence[Dict[str, torch.Tensor]],
    param_map: Dict[str, torch.Tensor],
    flips: Sequence[Dict[str, Any]],
    baseline: Dict[str, float],
    device: torch.device,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for flip in tqdm(flips, desc="Evaluate flips", leave=False):
        name = str(flip["name"])
        flat_idx = int(flip["index_flat"])
        bit = int(flip["bit"])
        if name not in param_map:
            continue
        param = param_map[name]
        old_val, new_val = flip_scalar_bit_fast_(param, flat_idx, bit)
        scores = evaluate_recurrent_metrics(
            model,
            cached_batches,
            device,
            recurrent_num=args.recurrent_num,
            use_amp=args.amp,
            steer_actuator_delay=args.steer_actuator_delay,
            target_mode=args.target_mode,
        )
        flip_scalar_bit_fast_(param, flat_idx, bit)
        rows.append(
            {
                "name": name,
                "index_flat": flat_idx,
                "bit": bit,
                "old": float(old_val),
                "new": float(new_val),
                "input_score": float(flip.get("score", 0.0)),
                "baseline_metrics": dict(baseline),
                "flipped_metrics": dict(scores),
                "delta": {k: float(scores[k] - baseline[k]) for k in baseline},
            }
        )
    return rows


def evaluate_cumulative(
    model: torch.nn.Module,
    cached_batches: Sequence[Dict[str, torch.Tensor]],
    param_map: Dict[str, torch.Tensor],
    flips: Sequence[Dict[str, Any]],
    baseline: Dict[str, float],
    device: torch.device,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    applied: List[Dict[str, Any]] = []
    current_baseline = dict(baseline)
    for flip in tqdm(flips, desc="Evaluate cumulative", leave=False):
        name = str(flip["name"])
        flat_idx = int(flip["index_flat"])
        bit = int(flip["bit"])
        if name not in param_map:
            continue
        param = param_map[name]
        old_val, new_val = flip_scalar_bit_fast_(param, flat_idx, bit)
        applied.append({"name": name, "index_flat": flat_idx, "bit": bit})
        scores = evaluate_recurrent_metrics(
            model,
            cached_batches,
            device,
            recurrent_num=args.recurrent_num,
            use_amp=args.amp,
            steer_actuator_delay=args.steer_actuator_delay,
            target_mode=args.target_mode,
        )
        rows.append(
            {
                "name": name,
                "index_flat": flat_idx,
                "bit": bit,
                "old": float(old_val),
                "new": float(new_val),
                "input_score": float(flip.get("score", 0.0)),
                "applied_count": len(applied),
                "applied": list(applied),
                "baseline_metrics": dict(current_baseline),
                "flipped_metrics": dict(scores),
                "delta": {k: float(scores[k] - current_baseline[k]) for k in current_baseline},
            }
        )
        current_baseline = dict(scores)
    return rows


def save_results(args: argparse.Namespace, baseline: Dict[str, float], rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "meta": {
            "timestamp": timestamp_id(),
            "ckpt": args.ckpt,
            "data_root": args.data_root,
            "val_index": args.val_index,
            "batch_size": args.batch_size,
            "recurrent_num": args.recurrent_num,
            "num_val_batches": args.num_val_batches,
            "input_json": args.input_json,
            "flip_count": args.flip_count,
            "eval_mode": args.eval_mode,
            "steer_actuator_delay": args.steer_actuator_delay,
            "target_mode": args.target_mode,
        },
        "baseline_metrics": baseline,
        "results": list(rows),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Save] wrote {args.out}")


def main() -> str:
    args = parse_args()
    args.target_mode = validate_target_mode(args.target_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    loader = build_eval_loader(
        args.data_root,
        args.val_index,
        args.batch_size,
        device,
        num_workers=args.num_workers,
    )
    cached_batches = cache_eval_batches(loader, args.num_val_batches)
    print(f"[Data] cached {len(cached_batches)} validation batches")

    model = load_model(args.ckpt, device)
    baseline = evaluate_recurrent_metrics(
        model,
        cached_batches,
        device,
        recurrent_num=args.recurrent_num,
        use_amp=args.amp,
        steer_actuator_delay=args.steer_actuator_delay,
        target_mode=args.target_mode,
    )
    print("[Baseline] " + ", ".join(f"{k}={v:.6f}" for k, v in baseline.items()))

    flips = load_flip_rows_json(args.input_json, limit=args.flip_count)
    print(f"[Config] flips={len(flips)} mode={args.eval_mode}")
    param_map = live_params(model)

    if args.eval_mode == "cumulative":
        rows = evaluate_cumulative(model, cached_batches, param_map, flips, baseline, device, args)
    else:
        rows = evaluate_independent(model, cached_batches, param_map, flips, baseline, device, args)

    save_results(args, baseline, rows)
    if rows:
        top = rows[0]
        delta = top["delta"]
        print(
            f"[Top-1] name={top['name']} flat={top['index_flat']} bit={top['bit']} "
            f"dAccel={delta['action.desiredAcceleration']:+.6f} "
            f"dCurv={delta['action.desiredCurvature']:+.6f} "
            f"dCurvDelta={delta['action.curvatureDelta']:+.6f}"
        )
    return args.out


if __name__ == "__main__":
    main()
