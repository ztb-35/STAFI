#!/usr/bin/env python3
"""Rank single bit-flip candidates by multiple loss metrics for op_097."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from op_097 import important_bits_onnx as ib  # noqa: E402


SUPPORTED_METRICS = {
    "loss_total",
    "loss_cls",
    "loss_reg",
    "+diffx",
    "-diffx",
    "+diffy",
    "-diffy",
}


def parse_metrics(s: str) -> List[str]:
    metrics = [x.strip() for x in s.split(",") if x.strip()]
    if not metrics:
        raise ValueError("metrics cannot be empty")
    bad = [m for m in metrics if m not in SUPPORTED_METRICS]
    if bad:
        raise ValueError(f"Unsupported metrics: {bad}. Supported: {sorted(SUPPORTED_METRICS)}")
    return metrics


def decode_plan_losses(
    raw_outputs: np.ndarray,
    output_slices: Dict[str, slice],
    gt: torch.Tensor,
) -> Tuple[float, float, float, torch.Tensor]:
    out = torch.from_numpy(raw_outputs.astype(np.float32))
    plan_slice = output_slices["plan"]
    plan = out[:, plan_slice].view(out.shape[0], 5, 991)

    pred_cls = plan[:, :, -1]
    params_flat = plan[:, :, :-1]
    pred_traj = params_flat.view(out.shape[0], 5, 2, 33, 15)[:, :, 0, :, :3]

    pred_end = pred_traj[:, :, 32, :]
    gt_end = gt[:, 32:33, :].expand(-1, 5, -1)
    distances = 1.0 - F.cosine_similarity(pred_end, gt_end, dim=2)
    gt_cls = distances.argmin(dim=1)
    row = torch.arange(gt_cls.shape[0])
    best_traj = pred_traj[row, gt_cls, :, :]

    cls = F.cross_entropy(pred_cls, gt_cls)
    reg = F.smooth_l1_loss(best_traj, gt, reduction="mean")
    total = cls + reg
    return float(total.item()), float(cls.item()), float(reg.item()), best_traj


def eval_metrics_for_session(
    session: "ib.ort.InferenceSession",
    cached_batches: Sequence[ib.CachedBatch],
    output_slices: Dict[str, slice],
    input_shapes: Dict[str, Tuple[int, ...]],
    metrics: Sequence[str],
) -> Dict[str, float]:
    vals: Dict[str, List[float]] = {m: [] for m in metrics}
    want_diff = any(m in {"+diffx", "-diffx", "+diffy", "-diffy"} for m in metrics)
    for cb in cached_batches:
        bsz, t, _, _, _ = cb.seq_imgs12.shape
        feature_buffer = np.zeros((bsz, 99, 512), dtype=np.float16)
        for i in range(t):
            inputs = {
                "input_imgs": cb.seq_imgs12[:, i],
                "big_input_imgs": cb.seq_imgs12[:, i],
            }
            inputs.update(ib.build_static_onnx_inputs(input_shapes, bsz, feature_buffer))
            raw = session.run(["outputs"], inputs)[0]

            total, cls, reg, best_traj = decode_plan_losses(raw, output_slices, cb.seq_gt[:, i])
            if "loss_total" in vals:
                vals["loss_total"].append(total)
            if "loss_cls" in vals:
                vals["loss_cls"].append(cls)
            if "loss_reg" in vals:
                vals["loss_reg"].append(reg)
            if want_diff:
                bt = best_traj.detach().cpu()
                for m in ("+diffx", "-diffx", "+diffy", "-diffy"):
                    if m in vals:
                        vals[m].append(ib.traj_metric(bt, m))

            if "features_buffer" in input_shapes:
                hs = raw[:, output_slices["hidden_state"]].astype(np.float16)
                feature_buffer = np.concatenate([feature_buffer[:, 1:, :], hs[:, None, :]], axis=1)

    return {m: (float(np.mean(v)) if v else float("nan")) for m, v in vals.items()}


def eval_metrics_with_fallback(
    model_bytes_or_path: object,
    providers: List[str],
    cached_batches: Sequence[ib.CachedBatch],
    output_slices: Dict[str, slice],
    input_shapes: Dict[str, Tuple[int, ...]],
    metrics: Sequence[str],
) -> Tuple[Dict[str, float], List[str]]:
    active = list(providers)
    sess = ib.make_session(model_bytes_or_path, providers=active)
    try:
        out = eval_metrics_for_session(sess, cached_batches, output_slices, input_shapes, metrics)
        return out, active
    except Exception as exc:
        if ("CUDAExecutionProvider" in active) and ib.is_cuda_runtime_error(exc):
            print(f"[ORT] CUDA runtime error, fallback to CPU: {exc}")
            active = ["CPUExecutionProvider"]
            sess = ib.make_session(model_bytes_or_path, providers=active)
            out = eval_metrics_for_session(sess, cached_batches, output_slices, input_shapes, metrics)
            return out, active
        raise


def with_timestamp(path: str, ts: str, out_dir: str) -> str:
    base = os.path.basename(path)
    root, ext = os.path.splitext(base)
    if not ext:
        ext = ".json"
    return os.path.join(out_dir, f"{root}_{ts}{ext}")


def get_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    split_dir = root / "op_097" / "splits"
    p = argparse.ArgumentParser(description="Rank single bitflip top bits by multiple losses (op_097)")
    p.add_argument("--onnx", default=str(root / "op_097" / "models" / "supercombo.onnx"))
    p.add_argument("--metadata", default=str(root / "op_097" / "models" / "supercombo_metadata.pkl"))
    p.add_argument("--data-root", default="/home/zx/Projects/comma2k19")
    p.add_argument("--train-index", default=str(split_dir / "comma2k19_train_non_overlap.txt"))
    p.add_argument("--val-index", default=str(split_dir / "comma2k19_val_non_overlap.txt"))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-val-batches", type=int, default=2)
    p.add_argument("--eval-seq-len", type=int, default=20)
    p.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--weights-in", default="", help="Optional precomputed weight candidates JSON")
    p.add_argument("--top-w", type=int, default=500)
    p.add_argument("--per-tensor-k", type=int, default=1)
    p.add_argument("--allow-bias", action="store_true")
    p.add_argument("--restrict", default="")
    p.add_argument("--bitset", default="exponent_sign")
    p.add_argument("--metrics", default="loss_total,loss_cls,loss_reg")
    p.add_argument("--top-k", type=int, default=50, help="Top-K flips to keep per metric")
    p.add_argument("--max-flips", type=int, default=0, help="Debug cap for evaluated flips; <=0 means all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(root / "op_097" / "out" / "single_flip_top_bits_losses_097.json"))
    p.add_argument("--save-all-records", action="store_true", help="Store all flip records (can be large)")
    p.add_argument(
        "--dataloader-inner-progress",
        action="store_true",
        help="Show fine-grained progress inside each val sample loading/preprocess step",
    )
    return p.parse_args()


def main() -> None:
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    metrics = parse_metrics(args.metrics)
    providers = ib.resolve_providers(args.provider)
    print(f"[ORT] providers={providers}")

    with open(args.metadata, "rb") as f:
        metadata = pickle.load(f)
    output_slices: Dict[str, slice] = metadata["output_slices"]
    input_shapes: Dict[str, Tuple[int, ...]] = metadata["input_shapes"]
    restrict = [x.strip() for x in args.restrict.split(",") if x.strip()] or None
    bitset = ib.parse_bitset(args.bitset)
    bitset_mode = ib.canonical_bitset_mode(args.bitset)

    _, val_loader = ib.build_loaders(args)
    cached_batches = ib.collect_cached_batches(
        val_loader,
        input_shapes=input_shapes,
        num_batches=args.num_val_batches,
        eval_seq_len=args.eval_seq_len,
    )
    if not cached_batches:
        raise RuntimeError("No validation batches cached. Check dataset path/splits.")

    baseline_metrics, providers = eval_metrics_with_fallback(
        args.onnx,
        providers,
        cached_batches,
        output_slices,
        input_shapes,
        metrics,
    )
    print("[Baseline] " + ", ".join(f"{k}={v:.6f}" for k, v in baseline_metrics.items()))

    base_model = onnx.load(args.onnx)
    if args.weights_in:
        selected = ib.load_weight_candidates(args.weights_in)
        if args.top_w > 0:
            selected = selected[: min(args.top_w, len(selected))]
        print(f"[Weights] loaded {len(selected)} candidates from {args.weights_in}")
        weights_source = os.path.abspath(args.weights_in)
    else:
        selected = ib.select_weight_candidates(
            base_model,
            top_w=args.top_w,
            per_tensor_k=args.per_tensor_k,
            allow_bias=args.allow_bias,
            restrict=restrict,
        )
        print(f"[Weights] selected {len(selected)} candidates by magnitude")
        weights_source = ""
    if not selected:
        raise RuntimeError("No candidates available")

    pending: List[Tuple[str, int, int, float]] = []
    for name, flat, score in selected:
        for bit in bitset:
            pending.append((name, int(flat), int(bit), float(score)))
    if args.max_flips > 0:
        pending = pending[: min(args.max_flips, len(pending))]
    if not pending:
        raise RuntimeError("No flip combinations to evaluate")
    print(f"[Flips] evaluate {len(pending)} single flips")

    base_model_bytes = base_model.SerializeToString()
    records: List[Dict[str, object]] = []
    pbar = tqdm(pending, desc="Single-flip scan", unit="flip", dynamic_ncols=True)
    for name, flat, bit, weight_score in pbar:
        flipped_bytes, old, new = ib.make_flipped_model_bytes(base_model_bytes, name, flat, bit)
        if flipped_bytes is None:
            continue
        flipped_metrics, providers = eval_metrics_with_fallback(
            flipped_bytes,
            providers,
            cached_batches,
            output_slices,
            input_shapes,
            metrics,
        )
        if not all(np.isfinite(v) for v in flipped_metrics.values()):
            continue
        delta = {m: float(flipped_metrics[m] - baseline_metrics[m]) for m in metrics}
        records.append(
            {
                "name": name,
                "flat": int(flat),
                "bit": int(bit),
                "old": float(old),
                "new": float(new),
                "weight_score": float(weight_score),
                "flipped_metrics": flipped_metrics,
                "delta": delta,
            }
        )

    top_by_metric: Dict[str, List[Dict[str, object]]] = {}
    for metric in metrics:
        rows = sorted(records, key=lambda r: float(r["delta"][metric]), reverse=True)  # type: ignore[index]
        top_by_metric[metric] = rows[: min(args.top_k, len(rows))]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "op_097", "out")
    out_path = with_timestamp(args.out, ts, out_dir)
    payload: Dict[str, object] = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "onnx": os.path.abspath(args.onnx),
            "metadata": os.path.abspath(args.metadata),
            "data_root": os.path.abspath(args.data_root),
            "train_index": os.path.abspath(args.train_index),
            "val_index": os.path.abspath(args.val_index),
            "provider": args.provider,
            "active_providers_end": providers,
            "num_val_batches": int(args.num_val_batches),
            "eval_seq_len": int(args.eval_seq_len),
            "top_w": int(args.top_w),
            "per_tensor_k": int(args.per_tensor_k),
            "bitset_mode": bitset_mode,
            "bitset": bitset,
            "metrics": metrics,
            "top_k": int(args.top_k),
            "max_flips": int(args.max_flips),
            "weights_source": weights_source,
            "num_candidates": int(len(selected)),
            "num_flips_evaluated": int(len(records)),
        },
        "baseline_metrics": baseline_metrics,
        "top_by_metric": top_by_metric,
    }
    if args.save_all_records:
        payload["all_records"] = records

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[Done] saved: {out_path}")
    for m in metrics:
        tops = top_by_metric.get(m, [])
        if tops:
            t = tops[0]
            print(
                f"[Top-1 {m}] name={t['name']} flat={t['flat']} bit={t['bit']} "
                f"delta={t['delta'][m]:.6f} value={t['flipped_metrics'][m]:.6f}"
            )


if __name__ == "__main__":
    main()
