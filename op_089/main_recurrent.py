#!/usr/bin/env python3
"""Rank fp32 bit flips for op_089 recurrent evaluation using accel/curvature metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from recurrent_bit_utils import (
    build_eval_loader,
    cache_eval_batches,
    evaluate_recurrent_metrics,
    flip_scalar_bit_fast_,
    live_params,
    load_candidates_json,
    load_model,
    metric_base_name,
    parse_bitset,
    score_metric_delta,
    score_metric_percent_delta,
    validate_target_mode,
    validate_metric,
)


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser("Rank recurrent op_089 bit flips by action accel/curvature metrics")
    ap.add_argument("--data-root", default="data/comma2k19/")
    ap.add_argument("--val-index", default=os.path.join(root, "data", "comma2k19_val_non_overlap.txt"))
    ap.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1)
    ap.add_argument("--recurrent-num", "--recurrent_num", dest="recurrent_num", type=int, default=100)
    ap.add_argument("--num-val-batches", "--num_val_batches", dest="num_val_batches", type=int, default=2)
    ap.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=0)
    ap.add_argument("--ckpt", default=os.path.join(root, "openpilot_model", "supercombo_torch_weights.pth"))
    ap.add_argument("--weights-in", "--topW-json", dest="weights_in", required=True)
    ap.add_argument("--top-w", "--topW", dest="top_w", type=int, default=100)
    ap.add_argument("--top-b", "--topB", dest="top_b", type=int, default=10)
    ap.add_argument("--bitset", default=">=5")
    ap.add_argument("--eval-metric", "--metric", dest="eval_metric", default="action.desiredAcceleration")
    ap.add_argument("--max-bits-per-scalar", "--max_bits_per_scalar", dest="max_bits_per_scalar", type=int, default=1)
    ap.add_argument("--steer-actuator-delay", "--steer_actuator_delay", dest="steer_actuator_delay", type=float, default=0.0)
    ap.add_argument("--target-mode", "--target_mode", dest="target_mode", default="pseudo_controls")
    ap.add_argument("--out", default=os.path.join(root, "op_089", "out", "important_bits_089_recurrent.json"))
    ap.add_argument("--amp", action="store_true")
    return ap.parse_args()


@torch.no_grad()
def rank_bits_progressive(
    model: torch.nn.Module,
    cached_batches: Sequence[Dict[str, torch.Tensor]],
    candidates: Sequence[Tuple[str, int]],
    bitset: Sequence[int],
    device: torch.device,
    *,
    top_b: int,
    eval_metric: str,
    recurrent_num: int,
    use_amp: bool,
    max_bits_per_scalar: int,
    steer_actuator_delay: float,
    target_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    param_map = live_params(model)
    pending = [(name, flat_idx, bit) for (name, flat_idx) in candidates for bit in bitset]
    committed_counts: Dict[Tuple[str, int], int] = {}
    records: List[Dict[str, Any]] = []

    baseline_scores = evaluate_recurrent_metrics(
        model,
        cached_batches,
        device,
        recurrent_num=recurrent_num,
        use_amp=use_amp,
        steer_actuator_delay=steer_actuator_delay,
        target_mode=target_mode,
    )
    if not math.isfinite(baseline_scores[metric_base_name(eval_metric)]):
        raise RuntimeError(f"Baseline metric is non-finite: {baseline_scores}")

    print(
        "[Baseline] "
        + ", ".join(f"{k}={v:.6f}" for k, v in baseline_scores.items())
    )

    progress = tqdm(range(top_b), desc="Rank bits", leave=False)
    for iteration in progress:
        best_row: Optional[Dict[str, Any]] = None
        best_scores: Optional[Dict[str, float]] = None

        for name, flat_idx, bit in list(pending):
            if name not in param_map:
                pending.remove((name, flat_idx, bit))
                continue
            if committed_counts.get((name, flat_idx), 0) >= max_bits_per_scalar:
                pending.remove((name, flat_idx, bit))
                continue

            param = param_map[name]
            old_val, new_val = flip_scalar_bit_fast_(param, flat_idx, bit)
            flipped_scores = evaluate_recurrent_metrics(
                model,
                cached_batches,
                device,
                recurrent_num=recurrent_num,
                use_amp=use_amp,
                steer_actuator_delay=steer_actuator_delay,
                target_mode=target_mode,
            )
            flip_scalar_bit_fast_(param, flat_idx, bit)

            tracked_value = flipped_scores[metric_base_name(eval_metric)]
            if not math.isfinite(tracked_value):
                pending.remove((name, flat_idx, bit))
                continue

            abs_delta = score_metric_delta(eval_metric, baseline_scores, flipped_scores)
            percent_delta = score_metric_percent_delta(eval_metric, baseline_scores, flipped_scores)
            row = {
                "iter": iteration,
                "name": name,
                "index_flat": int(flat_idx),
                "bit": int(bit),
                "old": float(old_val),
                "new": float(new_val),
                "metric": eval_metric,
                "score": float(percent_delta),
                "score_percent": float(percent_delta),
                "score_abs_delta": float(abs_delta),
                "baseline_metrics": dict(baseline_scores),
                "flipped_metrics": dict(flipped_scores),
            }
            if best_row is None or percent_delta > float(best_row["score"]):
                best_row = row
                best_scores = dict(flipped_scores)

        if best_row is None or best_scores is None:
            break

        param = param_map[best_row["name"]]
        flip_scalar_bit_fast_(param, int(best_row["index_flat"]), int(best_row["bit"]))
        records.append(best_row)
        baseline_scores = best_scores

        key = (best_row["name"], int(best_row["index_flat"]))
        committed_counts[key] = committed_counts.get(key, 0) + 1
        pending = [
            cand
            for cand in pending
            if not (
                cand[0] == best_row["name"]
                and cand[1] == int(best_row["index_flat"])
                and cand[2] == int(best_row["bit"])
            )
        ]
        if committed_counts[key] >= max_bits_per_scalar:
            pending = [cand for cand in pending if not (cand[0] == key[0] and cand[1] == key[1])]
        if not pending:
            break

    return records, baseline_scores


def save_results(
    out_path: str,
    args: argparse.Namespace,
    records: Sequence[Dict[str, Any]],
    final_metrics: Dict[str, float],
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "meta": {
            "timestamp": timestamp_id(),
            "ckpt": args.ckpt,
            "data_root": args.data_root,
            "val_index": args.val_index,
            "batch_size": args.batch_size,
            "recurrent_num": args.recurrent_num,
            "num_val_batches": args.num_val_batches,
            "top_w": args.top_w,
            "top_b": args.top_b,
            "bitset": args.bitset,
            "eval_metric": args.eval_metric,
            "score_type": "percent_change_vs_previous_baseline",
            "score_unit": "percent",
            "max_bits_per_scalar": args.max_bits_per_scalar,
            "steer_actuator_delay": args.steer_actuator_delay,
            "target_mode": args.target_mode,
            "weights_in": args.weights_in,
        },
        "final_metrics": final_metrics,
        "flips": list(records),
        "plan": [
            {
                "name": row["name"],
                "index_flat": row["index_flat"],
                "bit": row["bit"],
                "score": row["score"],
                "score_percent": row["score_percent"],
                "score_abs_delta": row["score_abs_delta"],
            }
            for row in records
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Save] wrote {out_path}")


def main() -> str:
    args = parse_args()
    args.eval_metric = validate_metric(args.eval_metric)
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
    candidates = load_candidates_json(args.weights_in, topn=args.top_w)
    bitset = parse_bitset(args.bitset)
    print(f"[Config] candidates={len(candidates)} bitset={bitset} metric={args.eval_metric}")

    records, final_metrics = rank_bits_progressive(
        model,
        cached_batches,
        candidates,
        bitset,
        device,
        top_b=args.top_b,
        eval_metric=args.eval_metric,
        recurrent_num=args.recurrent_num,
        use_amp=args.amp,
        max_bits_per_scalar=args.max_bits_per_scalar,
        steer_actuator_delay=args.steer_actuator_delay,
        target_mode=args.target_mode,
    )

    if not records:
        raise RuntimeError("No viable bit candidates produced.")
    save_results(args.out, args, records, final_metrics)
    if records:
        top = records[0]
        print(
            f"[Top-1] name={top['name']} flat={top['index_flat']} bit={top['bit']} "
            f"score={top['score']:.6f}"
        )
    return args.out


if __name__ == "__main__":
    main()
