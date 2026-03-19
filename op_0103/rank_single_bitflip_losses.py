#!/usr/bin/env python3
"""Rank single bit-flip candidates by multiple losses for openpilot 0.10.3."""

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

from op_0103 import important_bits_onnx as ib  # noqa: E402


SUPPORTED_METRICS = {
    "loss_smooth_l1",
    "loss_l1",
    "loss_mse",
    "action.desiredCurvature",
    "-action.desiredCurvature",
    "action.desiredAcceleration",
    "-action.desiredAcceleration",
    "modelV2.position.x",
    "modelV2.velocity.x",
    "modelV2.acceleration.x",
    "+diffx",
    "-diffx",
    "+diffy",
    "-diffy",
    "+endx",
    "-endx",
    "+endy",
    "-endy",
    "+lanex",
    "-lanex",
    "+laney",
    "-laney",
    "+leadx",
    "-leadx",
    "+leady",
    "-leady",
    "+leadprob",
    "-leadprob",
}


def parse_metrics(s: str) -> List[str]:
    metrics = [x.strip() for x in s.split(",") if x.strip()]
    if not metrics:
        raise ValueError("metrics cannot be empty")
    bad = [m for m in metrics if m not in SUPPORTED_METRICS]
    if bad:
        raise ValueError(f"Unsupported metrics: {bad}. Supported: {sorted(SUPPORTED_METRICS)}")
    return metrics


def decode_policy_losses(
    policy_raw_outputs: np.ndarray,
    policy_output_slices: Dict[str, slice],
    gt: torch.Tensor,
) -> Tuple[Dict[str, float], torch.Tensor]:
    out = torch.from_numpy(policy_raw_outputs.astype(np.float32))
    plan_slice = policy_output_slices["plan"]
    plan_raw = out[:, plan_slice]
    if plan_raw.shape[1] % 2 != 0:
        raise ValueError(f"Unexpected policy plan width: {plan_raw.shape[1]}")

    n_values = plan_raw.shape[1] // 2
    if n_values % 33 != 0:
        raise ValueError(f"Policy plan width is not divisible by 33: {n_values}")
    plan_width = n_values // 33
    if plan_width < 3:
        raise ValueError(f"Policy plan width too small: {plan_width}")

    pred_mu = plan_raw[:, :n_values].view(out.shape[0], 33, plan_width)
    pred_pos = pred_mu[:, :, :3]

    l_smooth_l1 = float(F.smooth_l1_loss(pred_pos, gt, reduction="mean").item())
    l_l1 = float(F.l1_loss(pred_pos, gt, reduction="mean").item())
    l_mse = float(F.mse_loss(pred_pos, gt, reduction="mean").item())
    return {
        "loss_smooth_l1": l_smooth_l1,
        "loss_l1": l_l1,
        "loss_mse": l_mse,
    }, pred_mu


def decode_policy_losses_torch(
    policy_raw_outputs: torch.Tensor,
    policy_output_slices: Dict[str, slice],
    gt: torch.Tensor,
) -> Tuple[Dict[str, float], torch.Tensor]:
    out = policy_raw_outputs.float()
    plan_slice = policy_output_slices["plan"]
    plan_raw = out[:, plan_slice]
    if plan_raw.shape[1] % 2 != 0:
        raise ValueError(f"Unexpected policy plan width: {plan_raw.shape[1]}")

    n_values = plan_raw.shape[1] // 2
    if n_values % 33 != 0:
        raise ValueError(f"Policy plan width is not divisible by 33: {n_values}")
    plan_width = n_values // 33
    if plan_width < 3:
        raise ValueError(f"Policy plan width too small: {plan_width}")

    pred_mu = plan_raw[:, :n_values].view(out.shape[0], 33, plan_width)
    pred_pos = pred_mu[:, :, :3]

    l_smooth_l1 = float(F.smooth_l1_loss(pred_pos, gt, reduction="mean").item())
    l_l1 = float(F.l1_loss(pred_pos, gt, reduction="mean").item())
    l_mse = float(F.mse_loss(pred_pos, gt, reduction="mean").item())
    return {
        "loss_smooth_l1": l_smooth_l1,
        "loss_l1": l_l1,
        "loss_mse": l_mse,
    }, pred_mu


def eval_metrics_for_session(
    vision_session: "ib.ort.InferenceSession",
    policy_session: "ib.ort.InferenceSession",
    cached_batches: Sequence[ib.CachedBatch],
    clean_vision_refs: Optional[Sequence[ib.CachedVisionRefs]],
    vision_output_slices: Dict[str, slice],
    policy_output_slices: Dict[str, slice],
    vision_input_dtype: np.dtype,
    policy_input_dtype: np.dtype,
    policy_input_shapes: Dict[str, Tuple[int, ...]],
    metrics: Sequence[str],
) -> Dict[str, float]:
    vals: Dict[str, List[float]] = {m: [] for m in metrics}

    for batch_idx, cb in enumerate(cached_batches):
        bsz, t, _, _, _ = cb.seq_imgs12.shape

        desire_shape = policy_input_shapes["desire_pulse"]
        features_shape = policy_input_shapes["features_buffer"]
        traffic_shape = policy_input_shapes["traffic_convention"]
        desire_len = ib.shape_dim(desire_shape, 1, 25)
        desire_dim = ib.shape_dim(desire_shape, 2, 8)
        feat_len = ib.shape_dim(features_shape, 1, 25)
        feat_dim = ib.shape_dim(features_shape, 2, 512)
        traffic_dim = ib.shape_dim(traffic_shape, 1, 2)

        desire_pulse = np.zeros((bsz, desire_len, desire_dim), dtype=policy_input_dtype)
        traffic = np.zeros((bsz, traffic_dim), dtype=policy_input_dtype)
        if traffic_dim >= 2:
            traffic[:, 0] = 1.0
        features_buffer = np.zeros((bsz, feat_len, feat_dim), dtype=policy_input_dtype)
        prev_desired_accel = np.zeros((bsz,), dtype=np.float32)
        prev_desired_curvature = np.zeros((bsz,), dtype=np.float32)

        for ti in range(t):
            vision_inputs = {
                "img": ib.cast_vision_input_for_ort(cb.seq_imgs12[:, ti], vision_input_dtype),
                "big_img": ib.cast_vision_input_for_ort(cb.seq_imgs12[:, ti], vision_input_dtype),
            }
            vision_raw = vision_session.run(["outputs"], vision_inputs)[0]
            hidden = vision_raw[:, vision_output_slices["hidden_state"]].astype(policy_input_dtype, copy=False)
            features_buffer = np.concatenate([features_buffer[:, 1:, :], hidden[:, None, :]], axis=1)

            policy_inputs = {
                "desire_pulse": desire_pulse,
                "traffic_convention": traffic,
                "features_buffer": features_buffer,
            }
            policy_raw = policy_session.run(["outputs"], policy_inputs)[0]
            losses, pred_pos = decode_policy_losses(policy_raw, policy_output_slices, cb.seq_gt[:, ti])
            action_targets = None
            if any(m.startswith("action.") for m in metrics):
                action_targets, prev_desired_accel, prev_desired_curvature = ib._compute_action_targets_from_plan(
                    pred_pos,
                    prev_desired_accel,
                    prev_desired_curvature,
                )

            for m in metrics:
                if m in losses:
                    vals[m].append(losses[m])
                else:
                    vals[m].append(
                        ib.output_metric_torch(
                            pred_pos=pred_pos,
                            vision_raw_outputs=torch.from_numpy(vision_raw.astype(np.float32)),
                            vision_output_slices=vision_output_slices,
                            metric=m,
                            clean_vision_ref=None if clean_vision_refs is None else clean_vision_refs[batch_idx],
                            step_idx=ti,
                            action_targets=action_targets,
                        )
                    )

    return {m: (float(np.mean(v)) if v else float("nan")) for m, v in vals.items()}


def eval_metrics_for_torch_models(
    vision_model_torch: torch.nn.Module,
    policy_model_torch: torch.nn.Module,
    cached_batches: Sequence[ib.CachedBatch],
    clean_vision_refs: Optional[Sequence[ib.CachedVisionRefs]],
    vision_output_slices: Dict[str, slice],
    policy_output_slices: Dict[str, slice],
    policy_input_shapes: Dict[str, Tuple[int, ...]],
    vision_input_order: Sequence[str],
    policy_input_order: Sequence[str],
    device: torch.device,
    vision_input_dtype_torch: torch.dtype,
    policy_input_dtype_torch: torch.dtype,
    metrics: Sequence[str],
) -> Dict[str, float]:
    vals: Dict[str, List[float]] = {m: [] for m in metrics}
    with torch.no_grad():
        for batch_idx, cb in enumerate(cached_batches):
            imgs = torch.from_numpy(cb.seq_imgs12).to(device=device, dtype=vision_input_dtype_torch)
            gt = cb.seq_gt.to(device=device, dtype=torch.float32)
            bsz, t, _, _, _ = imgs.shape

            desire_shape = policy_input_shapes["desire_pulse"]
            features_shape = policy_input_shapes["features_buffer"]
            traffic_shape = policy_input_shapes["traffic_convention"]
            desire_len = ib.shape_dim(desire_shape, 1, 25)
            desire_dim = ib.shape_dim(desire_shape, 2, 8)
            feat_len = ib.shape_dim(features_shape, 1, 25)
            feat_dim = ib.shape_dim(features_shape, 2, 512)
            traffic_dim = ib.shape_dim(traffic_shape, 1, 2)

            desire_pulse = torch.zeros((bsz, desire_len, desire_dim), device=device, dtype=policy_input_dtype_torch)
            traffic = torch.zeros((bsz, traffic_dim), device=device, dtype=policy_input_dtype_torch)
            if traffic_dim >= 2:
                traffic[:, 0] = 1.0
            features_buffer = torch.zeros((bsz, feat_len, feat_dim), device=device, dtype=policy_input_dtype_torch)
            prev_desired_accel = np.zeros((bsz,), dtype=np.float32)
            prev_desired_curvature = np.zeros((bsz,), dtype=np.float32)

            for ti in range(t):
                vision_inputs = {"img": imgs[:, ti], "big_img": imgs[:, ti]}
                vision_feed = [vision_inputs[k] for k in vision_input_order]
                vision_raw = vision_model_torch(*vision_feed)
                if not isinstance(vision_raw, torch.Tensor):
                    raise RuntimeError(f"Unexpected vision torch output type: {type(vision_raw)}")

                hidden = vision_raw[:, vision_output_slices["hidden_state"]].to(policy_input_dtype_torch)
                features_buffer = torch.cat([features_buffer[:, 1:, :], hidden[:, None, :]], dim=1)

                policy_inputs = {
                    "desire_pulse": desire_pulse,
                    "traffic_convention": traffic,
                    "features_buffer": features_buffer,
                }
                policy_feed = [policy_inputs[k] for k in policy_input_order]
                policy_raw = policy_model_torch(*policy_feed)
                if not isinstance(policy_raw, torch.Tensor):
                    raise RuntimeError(f"Unexpected policy torch output type: {type(policy_raw)}")

                losses, pred_pos = decode_policy_losses_torch(policy_raw, policy_output_slices, gt[:, ti])
                action_targets = None
                if any(m.startswith("action.") for m in metrics):
                    action_targets, prev_desired_accel, prev_desired_curvature = ib._compute_action_targets_from_plan(
                        pred_pos.detach().cpu(),
                        prev_desired_accel,
                        prev_desired_curvature,
                    )
                for m in metrics:
                    if m in losses:
                        vals[m].append(losses[m])
                    else:
                        vals[m].append(
                            ib.output_metric_torch(
                                pred_pos=pred_pos.detach().cpu(),
                                vision_raw_outputs=vision_raw.detach().cpu().float(),
                                vision_output_slices=vision_output_slices,
                                metric=m,
                                clean_vision_ref=None if clean_vision_refs is None else clean_vision_refs[batch_idx],
                                step_idx=ti,
                                action_targets=action_targets,
                            )
                        )

    return {m: (float(np.mean(v)) if v else float("nan")) for m, v in vals.items()}


def eval_metrics_with_fallback(
    vision_model: object,
    policy_model: object,
    providers: List[str],
    cached_batches: Sequence[ib.CachedBatch],
    clean_vision_refs: Optional[Sequence[ib.CachedVisionRefs]],
    vision_output_slices: Dict[str, slice],
    policy_output_slices: Dict[str, slice],
    policy_input_shapes: Dict[str, Tuple[int, ...]],
    metrics: Sequence[str],
) -> Tuple[Dict[str, float], List[str]]:
    active = list(providers)
    vision_sess = ib.make_session(vision_model, providers=active)
    policy_sess = ib.make_session(policy_model, providers=active)
    try:
        vision_dtype = ib.ort_type_to_numpy_dtype(vision_sess.get_inputs()[0].type)
        policy_dtype = ib.ort_type_to_numpy_dtype(policy_sess.get_inputs()[0].type)
        out = eval_metrics_for_session(
            vision_sess,
            policy_sess,
            cached_batches,
            clean_vision_refs,
            vision_output_slices,
            policy_output_slices,
            vision_dtype,
            policy_dtype,
            policy_input_shapes,
            metrics,
        )
        return out, active
    except Exception as exc:
        if ("CUDAExecutionProvider" in active) and ib.is_cuda_runtime_error(exc):
            print(f"[ORT] CUDA runtime error, fallback to CPU: {exc}")
            active = ["CPUExecutionProvider"]
            vision_sess = ib.make_session(vision_model, providers=active)
            policy_sess = ib.make_session(policy_model, providers=active)
            vision_dtype = ib.ort_type_to_numpy_dtype(vision_sess.get_inputs()[0].type)
            policy_dtype = ib.ort_type_to_numpy_dtype(policy_sess.get_inputs()[0].type)
            out = eval_metrics_for_session(
                vision_sess,
                policy_sess,
                cached_batches,
                clean_vision_refs,
                vision_output_slices,
                policy_output_slices,
                vision_dtype,
                policy_dtype,
                policy_input_shapes,
                metrics,
            )
            return out, active
        raise


def build_torch_flip_targets(
    base_vision_model: onnx.ModelProto,
    base_policy_model: onnx.ModelProto,
    vision_model_torch: torch.nn.Module,
    policy_model_torch: torch.nn.Module,
    candidates: Sequence[Tuple[str, str, int, float]],
    allow_bias: bool,
    restrict: Optional[List[str]],
) -> Dict[Tuple[str, str, int], ib.TorchFlipTarget]:
    refs_by_model = {
        "vision": ib.build_onnx_to_torch_refs(base_vision_model, vision_model_torch, allow_bias=allow_bias, restrict=restrict),
        "policy": ib.build_onnx_to_torch_refs(base_policy_model, policy_model_torch, allow_bias=allow_bias, restrict=restrict),
    }
    flip_targets: Dict[Tuple[str, str, int], ib.TorchFlipTarget] = {}
    for model_key, name, flat_idx, _score in candidates:
        tref = refs_by_model.get(model_key, {}).get(name)
        if tref is None:
            continue
        if flat_idx < 0 or flat_idx >= int(tref.numel()):
            continue
        idx = tuple(np.unravel_index(int(flat_idx), tuple(tref.shape)))
        flip_targets[(model_key, name, int(flat_idx))] = ib.TorchFlipTarget(tensor=tref, index=idx)
    return flip_targets


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def with_timestamp(path: str, ts: str, out_dir: str) -> str:
    base = os.path.basename(path)
    root, ext = os.path.splitext(base)
    if not ext:
        ext = ".json"
    return os.path.join(out_dir, f"{root}_{ts}{ext}")


def get_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    split_dir = root / "op_0103" / "splits"
    p = argparse.ArgumentParser(description="Rank single bitflip top bits by multiple losses (op_0103)")
    p.add_argument("--vision-onnx", default=str(root / "op_0103" / "models" / "driving_vision.onnx"))
    p.add_argument("--vision-metadata", default=str(root / "op_0103" / "models" / "driving_vision_metadata.pkl"))
    p.add_argument("--policy-onnx", default=str(root / "op_0103" / "models" / "driving_policy.onnx"))
    p.add_argument("--policy-metadata", default=str(root / "op_0103" / "models" / "driving_policy_metadata.pkl"))
    p.add_argument("--data-root", default="/home/zx/Projects/comma2k19")
    p.add_argument("--train-index", default=str(split_dir / "comma2k19_train_non_overlap.txt"))
    p.add_argument("--val-index", default=str(split_dir / "comma2k19_val_non_overlap.txt"))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-val-batches", type=int, default=2)
    p.add_argument("--eval-seq-len", type=int, default=20)
    p.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--eval-backend", choices=["ort", "torch"], default="torch")
    p.add_argument("--target-model", choices=["vision", "policy", "both"], default="both")
    p.add_argument("--weights-in", default="", help="Optional precomputed weight candidates JSON")
    p.add_argument("--top-w", type=int, default=500)
    p.add_argument("--per-tensor-k", type=int, default=1)
    p.add_argument("--allow-bias", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--restrict", default="")
    p.add_argument("--bitset", default="exponent_sign")
    p.add_argument("--metrics", default="loss_smooth_l1,loss_l1,loss_mse")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-flips", type=int, default=0, help="Debug cap; <=0 means evaluate all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(root / "op_0103" / "out" / "single_flip_top_bits_losses_0103.json"))
    p.add_argument("--save-all-records", action="store_true", help="Store all per-flip records (can be large)")
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

    with open(args.vision_metadata, "rb") as f:
        vision_metadata = pickle.load(f)
    with open(args.policy_metadata, "rb") as f:
        policy_metadata = pickle.load(f)
    vision_output_slices: Dict[str, slice] = vision_metadata["output_slices"]
    policy_output_slices: Dict[str, slice] = policy_metadata["output_slices"]
    policy_input_shapes: Dict[str, Tuple[int, ...]] = policy_metadata["input_shapes"]

    restrict = [x.strip() for x in args.restrict.split(",") if x.strip()] or None
    bitset, bitset_mode = ib.parse_bitset_with_mode(args.bitset)

    _, val_loader = ib.build_loaders(args)
    cached_batches = ib.collect_cached_batches(val_loader, num_batches=args.num_val_batches, eval_seq_len=args.eval_seq_len)
    if not cached_batches:
        raise RuntimeError("No validation batches cached. Check dataset path/splits.")
    needs_clean_vision_refs = any(ib._metric_uses_clean_vision_ref(m) for m in metrics)
    clean_vision_refs: Optional[List[ib.CachedVisionRefs]] = None

    torch_eval_device = torch.device("cuda" if (args.provider == "cuda" and torch.cuda.is_available()) else "cpu")
    if args.eval_backend == "torch":
        print(f"[Eval] using torch backend on device={torch_eval_device}")
        v_torch, p_torch, v_order, p_order, v_dtype, p_dtype = ib.build_torch_models_for_eval(
            onnx.load(args.vision_onnx),
            onnx.load(args.policy_onnx),
            device=torch_eval_device,
        )
        if needs_clean_vision_refs:
            clean_vision_refs = ib.collect_clean_vision_refs_torch(
                v_torch,
                cached_batches,
                vision_output_slices,
                v_order,
                torch_eval_device,
                v_dtype,
            )
        baseline_metrics = eval_metrics_for_torch_models(
            v_torch,
            p_torch,
            cached_batches,
            clean_vision_refs,
            vision_output_slices,
            policy_output_slices,
            policy_input_shapes,
            v_order,
            p_order,
            torch_eval_device,
            v_dtype,
            p_dtype,
            metrics,
        )
    else:
        baseline_metrics, providers = eval_metrics_with_fallback(
            args.vision_onnx,
            args.policy_onnx,
            providers,
            cached_batches,
            None,
            vision_output_slices,
            policy_output_slices,
            policy_input_shapes,
            metrics,
        )
        if needs_clean_vision_refs:
            vision_sess = ib.make_session(args.vision_onnx, providers=providers)
            vision_dtype = ib.ort_type_to_numpy_dtype(vision_sess.get_inputs()[0].type)
            clean_vision_refs = ib.collect_clean_vision_refs_ort(
                vision_sess,
                cached_batches,
                vision_output_slices,
                vision_dtype,
            )
            baseline_metrics, providers = eval_metrics_with_fallback(
                args.vision_onnx,
                args.policy_onnx,
                providers,
                cached_batches,
                clean_vision_refs,
                vision_output_slices,
                policy_output_slices,
                policy_input_shapes,
                metrics,
            )
    print("[Baseline] " + ", ".join(f"{k}={v:.6f}" for k, v in baseline_metrics.items()))

    base_vision_model = onnx.load(args.vision_onnx)
    base_policy_model = onnx.load(args.policy_onnx)

    if args.weights_in:
        default_model_key = "policy" if args.target_model == "policy" else "vision"
        selected = ib.load_weight_candidates(args.weights_in, default_model_key=default_model_key)
        if args.top_w > 0:
            selected = selected[: min(args.top_w, len(selected))]
        weights_source = os.path.abspath(args.weights_in)
        print(f"[Weights] loaded {len(selected)} candidates from {args.weights_in}")
    else:
        models_for_selection: Dict[str, onnx.ModelProto] = {}
        if args.target_model in ("vision", "both"):
            models_for_selection["vision"] = base_vision_model
        if args.target_model in ("policy", "both"):
            models_for_selection["policy"] = base_policy_model
        selected = ib.select_weight_candidates(
            models=models_for_selection,
            target_model=args.target_model,
            top_w=args.top_w,
            per_tensor_k=args.per_tensor_k,
            allow_bias=args.allow_bias,
            restrict=restrict,
        )
        weights_source = ""
        print(f"[Weights] selected {len(selected)} candidates by magnitude")

    if not selected:
        raise RuntimeError("No candidates available")

    pending: List[Tuple[str, str, int, int, float]] = []
    for model_key, name, flat_idx, score in selected:
        for bit in bitset:
            pending.append((model_key, name, int(flat_idx), int(bit), float(score)))
    if args.max_flips > 0:
        pending = pending[: min(args.max_flips, len(pending))]
    if not pending:
        raise RuntimeError("No flip combinations to evaluate")
    print(f"[Flips] evaluate {len(pending)} single flips")

    base_model_bytes = {
        "vision": base_vision_model.SerializeToString(),
        "policy": base_policy_model.SerializeToString(),
    }
    if args.eval_backend == "torch":
        flip_targets = build_torch_flip_targets(
            base_vision_model,
            base_policy_model,
            v_torch,
            p_torch,
            selected,
            allow_bias=args.allow_bias,
            restrict=restrict,
        )

    records: List[Dict[str, object]] = []
    pbar = tqdm(pending, desc="Single-flip scan", unit="flip", dynamic_ncols=True)
    for model_key, name, flat_idx, bit, weight_score in pbar:
        if args.eval_backend == "torch":
            target = flip_targets.get((model_key, name, int(flat_idx)))
            if target is None:
                continue
            applied, old, new = ib.apply_torch_fp16_flip_inplace(target, bit)
            if not applied:
                continue
            try:
                flipped_metrics = eval_metrics_for_torch_models(
                    v_torch,
                    p_torch,
                    cached_batches,
                    clean_vision_refs,
                    vision_output_slices,
                    policy_output_slices,
                    policy_input_shapes,
                    v_order,
                    p_order,
                    torch_eval_device,
                    v_dtype,
                    p_dtype,
                    metrics,
                )
            finally:
                ib.restore_torch_scalar_inplace(target, old)
        else:
            flipped_bytes, old, new = ib.make_flipped_model_bytes(base_model_bytes[model_key], name, flat_idx, bit)
            if flipped_bytes is None:
                continue
            cand_vision_bytes = flipped_bytes if model_key == "vision" else base_model_bytes["vision"]
            cand_policy_bytes = flipped_bytes if model_key == "policy" else base_model_bytes["policy"]
            flipped_metrics, providers = eval_metrics_with_fallback(
                cand_vision_bytes,
                cand_policy_bytes,
                providers,
                cached_batches,
                clean_vision_refs,
                vision_output_slices,
                policy_output_slices,
                policy_input_shapes,
                metrics,
            )
        if not all(np.isfinite(v) for v in flipped_metrics.values()):
            continue
        delta = {m: float(flipped_metrics[m] - baseline_metrics[m]) for m in metrics}
        records.append(
            {
                "model": model_key,
                "name": name,
                "flat": int(flat_idx),
                "bit": int(bit),
                "old": float(old),
                "new": float(new),
                "weight_score": float(weight_score),
                "flipped_metrics": flipped_metrics,
                "delta": delta,
            }
        )

    top_by_metric: Dict[str, List[Dict[str, object]]] = {}
    for m in metrics:
        rows = sorted(records, key=lambda r: float(r["delta"][m]), reverse=True)  # type: ignore[index]
        top_by_metric[m] = rows[: min(args.top_k, len(rows))]

    run_ts = timestamp_id()
    out_dir = os.path.join(REPO_ROOT, "op_0103", "out")
    out_path = with_timestamp(args.out, run_ts, out_dir)
    payload: Dict[str, object] = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "vision_onnx": os.path.abspath(args.vision_onnx),
            "vision_metadata": os.path.abspath(args.vision_metadata),
            "policy_onnx": os.path.abspath(args.policy_onnx),
            "policy_metadata": os.path.abspath(args.policy_metadata),
            "data_root": os.path.abspath(args.data_root),
            "train_index": os.path.abspath(args.train_index),
            "val_index": os.path.abspath(args.val_index),
            "target_model": args.target_model,
            "provider": args.provider,
            "eval_backend": args.eval_backend,
            "active_providers_end": providers,
            "torch_device": str(torch_eval_device) if args.eval_backend == "torch" else "",
            "batch_size": int(args.batch_size),
            "num_val_batches": int(args.num_val_batches),
            "eval_seq_len": int(args.eval_seq_len),
            "top_w": int(args.top_w),
            "top_k": int(args.top_k),
            "per_tensor_k": int(args.per_tensor_k),
            "bitset_mode": bitset_mode,
            "bitset": bitset,
            "allow_bias": bool(args.allow_bias),
            "restrict": restrict,
            "metrics": metrics,
            "max_flips": int(args.max_flips),
            "weights_source_file": weights_source,
            "num_candidates": int(len(selected)),
            "num_selected_weights": int(len(selected)),
            "num_requested_flip_combinations": int(len(pending)),
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
                f"[Top-1 {m}] model={t['model']} name={t['name']} flat={t['flat']} bit={t['bit']} "
                f"delta={t['delta'][m]:.6f} value={t['flipped_metrics'][m]:.6f}"
            )


if __name__ == "__main__":
    main()
