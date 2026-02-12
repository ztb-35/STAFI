#!/usr/bin/env python3
"""Search important bits for openpilot 0.9.7 supercombo ONNX.

Pipeline:
1) Auto-build comma2k19 split files when missing.
2) Build mini train/val loaders from Comma2k19SequenceDataset.
3) Observe 0.9.7 output slices and compute a baseline planning loss.
4) Select candidate scalar weights by absolute value.
5) Rank bit flips independently by delta loss.
6) Export a JSON plan compatible with downstream bit-flip scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx import TensorProto, numpy_helper
from torch.utils.data import DataLoader
from tqdm import tqdm

# # Reuse the existing comma2k19 loader logic from 0.8.9 implementation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# OP089_DIR = REPO_ROOT / "op_089"
# if str(OP089_DIR) not in sys.path:
#     sys.path.insert(0, str(OP089_DIR))
from data import Comma2k19SequenceDataset
from tools.common_dataset import SafeDataset, ensure_non_overlap_split_files, make_loader, normalize_data_root


@dataclass
class CachedBatch:
    seq_imgs12: np.ndarray
    seq_gt: torch.Tensor


def make_session(model_bytes_or_path: Any, providers: Sequence[str]) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model_bytes_or_path, sess_options=so, providers=list(providers))


def resolve_providers(provider_mode: str) -> List[str]:
    avail = set(ort.get_available_providers())
    if provider_mode == "cpu":
        return ["CPUExecutionProvider"]
    if provider_mode == "cuda":
        if "CUDAExecutionProvider" not in avail:
            raise RuntimeError(
                "CUDAExecutionProvider is not available in current onnxruntime. "
                "Install onnxruntime-gpu or use --provider cpu/auto."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def parse_bitset(name: str) -> List[int]:
    key = name.strip().lower()
    if key in ("all", "full"):
        return list(range(16))
    if key == "mantissa":
        return list(range(10))
    if key == "exponent":
        return list(range(10, 15))
    if key in ("exponent_sign", "exp_sign", "exponent&sign"):
        return list(range(10, 16))
    if key == "sign":
        return [15]
    bits = [int(x) for x in key.split(",") if x.strip()]
    for b in bits:
        if b < 0 or b > 15:
            raise ValueError(f"fp16 bit out of range: {b}")
    return bits


def canonical_bitset_mode(name: str) -> str:
    key = name.strip().lower()
    if key in ("all", "full"):
        return "all"
    if key == "mantissa":
        return "mantissa"
    if key == "exponent":
        return "exp"
    if key in ("exponent_sign", "exp_sign", "exponent&sign"):
        return "sign_exp"
    if key == "sign":
        return "sign"
    return "custom"


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    tr_n, va_n = ensure_non_overlap_split_files(args.data_root, args.train_index, args.val_index, seed=args.seed)
    if not (os.path.isfile(args.train_index) and os.path.isfile(args.val_index)):
        raise RuntimeError("Failed to build split files.")
    print(f"[Split] train={tr_n} val={va_n}")
    data_root = normalize_data_root(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = Comma2k19SequenceDataset(args.train_index, data_root, "train", use_memcache=False)
    val_ds = Comma2k19SequenceDataset(args.val_index, data_root, "val", use_memcache=False)
    train_ds = SafeDataset(train_ds)
    val_ds = SafeDataset(val_ds)

    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, device=device, num_workers=0)
    val_loader = make_loader(val_ds, batch_size=1, shuffle=False, device=device, num_workers=0)
    return train_loader, val_loader


def to_eval_sequence(
    batch: Dict[str, torch.Tensor],
    input_shapes: Dict[str, Tuple[int, ...]],
    eval_seq_len: int,
) -> Tuple[np.ndarray, torch.Tensor]:
    seq_imgs = batch["seq_input_img"]  # (B, T, C, H, W), C=6 usually
    seq_labels = batch["seq_future_poses"]  # (B, T, 33, 3)
    bsz, t, c, h, w = seq_imgs.shape
    if eval_seq_len > 0:
        t = min(t, eval_seq_len)
        seq_imgs = seq_imgs[:, :t]
        seq_labels = seq_labels[:, :t]

    imgs = seq_imgs.float()  # (B, T, C, H, W)
    if c == 6:
        imgs = torch.cat([imgs, imgs], dim=2)
    if imgs.shape[2] != 12:
        raise ValueError(f"Expected 12 channels for input_imgs, got {imgs.shape[2]}")

    gt = seq_labels[:, :, :, :].float()  # (B, T, 33, 3)
    if gt.shape[2] < 33:
        # This branch should never happen for comma2k19, keep as guard.
        raise ValueError(f"Need at least 33 trajectory points per frame, got {gt.shape}")
    gt = gt[:, :, :33, :]

    return imgs.numpy().astype(np.float16), gt


def build_static_onnx_inputs(
    input_shapes: Dict[str, Tuple[int, ...]],
    bsz: int,
    feature_buffer: np.ndarray,
) -> Dict[str, np.ndarray]:
    static_inputs: Dict[str, np.ndarray] = {}
    if "desire" in input_shapes:
        static_inputs["desire"] = np.zeros((bsz, 100, 8), dtype=np.float16)  # desire all zero
    if "traffic_convention" in input_shapes:
        static_inputs["traffic_convention"] = np.tile(np.array([[1.0, 0.0]], dtype=np.float16), (bsz, 1))  # right-hand
    if "lateral_control_params" in input_shapes:
        static_inputs["lateral_control_params"] = np.zeros((bsz, 2), dtype=np.float16)
    if "prev_desired_curv" in input_shapes:
        static_inputs["prev_desired_curv"] = np.zeros((bsz, 100, 1), dtype=np.float16)
    if "features_buffer" in input_shapes:
        static_inputs["features_buffer"] = feature_buffer
    return static_inputs


def plan_loss(raw_outputs: np.ndarray, output_slices: Dict[str, slice], gt: torch.Tensor) -> float:
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
    return float((cls + reg).item())


def decode_plan(raw_outputs: np.ndarray, output_slices: Dict[str, slice], gt: torch.Tensor) -> Tuple[float, torch.Tensor]:
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
    return float((cls + reg).item()), best_traj


def eval_loss(
    session: ort.InferenceSession,
    cached_batches: Sequence[CachedBatch],
    output_slices: Dict[str, slice],
    input_shapes: Dict[str, Tuple[int, ...]],
) -> float:
    vals: List[float] = []
    for cb in cached_batches:
        bsz, t, _, _, _ = cb.seq_imgs12.shape
        feature_buffer = np.zeros((bsz, 99, 512), dtype=np.float16)
        per_step_losses: List[float] = []
        for i in range(t):
            inputs = {
                "input_imgs": cb.seq_imgs12[:, i],
                # requested: image stream and wide image stream use the same input
                "big_input_imgs": cb.seq_imgs12[:, i],
            }
            inputs.update(build_static_onnx_inputs(input_shapes, bsz, feature_buffer))
            raw = session.run(["outputs"], inputs)[0]
            loss, _ = decode_plan(raw, output_slices, cb.seq_gt[:, i])
            per_step_losses.append(loss)
            if "features_buffer" in input_shapes:
                hs = raw[:, output_slices["hidden_state"]].astype(np.float16)
                feature_buffer = np.concatenate([feature_buffer[:, 1:, :], hs[:, None, :]], axis=1)
        vals.append(float(np.mean(per_step_losses)))
    return float(np.mean(vals)) if vals else float("nan")


def traj_metric(best_traj: torch.Tensor, metric: str) -> float:
    if metric == "+diffx":
        return float(best_traj[:, :, 0].mean().item())
    if metric == "-diffx":
        return float((-best_traj[:, :, 0]).mean().item())
    if metric == "+diffy":
        return float(best_traj[:, :, 1].mean().item())
    if metric == "-diffy":
        return float((-best_traj[:, :, 1]).mean().item())
    raise ValueError(f"Unknown trajectory metric: {metric}")


def eval_metric_value(
    session: ort.InferenceSession,
    cached_batches: Sequence[CachedBatch],
    output_slices: Dict[str, slice],
    input_shapes: Dict[str, Tuple[int, ...]],
    metric: str,
) -> float:
    if metric == "loss":
        return eval_loss(session, cached_batches, output_slices, input_shapes)
    if metric not in {"+diffx", "-diffx", "+diffy", "-diffy"}:
        raise ValueError(f"Unknown metric: {metric}")
    vals: List[float] = []
    for cb in cached_batches:
        bsz, t, _, _, _ = cb.seq_imgs12.shape
        feature_buffer = np.zeros((bsz, 99, 512), dtype=np.float16)
        for ti in range(t):
            inputs = {
                "input_imgs": cb.seq_imgs12[:, ti],
                "big_input_imgs": cb.seq_imgs12[:, ti],
            }
            inputs.update(build_static_onnx_inputs(input_shapes, bsz, feature_buffer))
            raw = session.run(["outputs"], inputs)[0]
            _, flipped_best = decode_plan(raw, output_slices, cb.seq_gt[:, ti])
            vals.append(traj_metric(flipped_best, metric))

            if "features_buffer" in input_shapes:
                hs = raw[:, output_slices["hidden_state"]].astype(np.float16)
                feature_buffer = np.concatenate([feature_buffer[:, 1:, :], hs[:, None, :]], axis=1)
    return float(np.mean(vals)) if vals else float("nan")


def collect_cached_batches(
    val_loader: DataLoader,
    input_shapes: Dict[str, Tuple[int, ...]],
    num_batches: int,
    eval_seq_len: int,
) -> List[CachedBatch]:
    cached: List[CachedBatch] = []
    it = iter(val_loader)
    for _ in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        seq_imgs12, gt = to_eval_sequence(batch, input_shapes, eval_seq_len=eval_seq_len)
        cached.append(CachedBatch(seq_imgs12=seq_imgs12, seq_gt=gt))
    return cached


def inspect_outputs(
    session: ort.InferenceSession,
    batch: CachedBatch,
    output_slices: Dict[str, slice],
    input_shapes: Dict[str, Tuple[int, ...]],
) -> None:
    bsz = batch.seq_imgs12.shape[0]
    feature_buffer = np.zeros((bsz, 99, 512), dtype=np.float16)
    inputs = {
        "input_imgs": batch.seq_imgs12[:, 0],
        "big_input_imgs": batch.seq_imgs12[:, 0],
    }
    inputs.update(build_static_onnx_inputs(input_shapes, bsz, feature_buffer))
    raw = session.run(["outputs"], inputs)[0][0]  # (6504,)
    print("\n[Inspect] Output slices from metadata:")
    for name, sl in output_slices.items():
        arr = raw[sl].astype(np.float32)
        print(f"  {name:<24} [{sl.start:>4}:{str(sl.stop):>4}] len={arr.size:<4} mean={arr.mean(): .5f} std={arr.std(): .5f}")


def iter_fp16_initializers(model: onnx.ModelProto, allow_bias: bool, restrict: Optional[List[str]]) -> Iterable[Tuple[str, np.ndarray]]:
    for t in model.graph.initializer:
        if t.data_type != TensorProto.FLOAT16:
            continue
        name = t.name
        if (not allow_bias) and name.endswith(".bias"):
            continue
        if restrict and not any(tag in name for tag in restrict):
            continue
        arr = numpy_helper.to_array(t)
        if arr.size == 0:
            continue
        yield name, arr


def select_weight_candidates(
    model: onnx.ModelProto,
    top_w: int,
    per_tensor_k: int,
    allow_bias: bool,
    restrict: Optional[List[str]],
) -> List[Tuple[str, int, float]]:
    items: List[Tuple[str, int, float]] = []
    for name, arr in iter_fp16_initializers(model, allow_bias=allow_bias, restrict=restrict):
        flat_abs = np.abs(arr.reshape(-1).astype(np.float32))
        k = min(per_tensor_k, flat_abs.size)
        if k <= 0:
            continue
        idx = np.argpartition(flat_abs, -k)[-k:]
        for i in idx:
            items.append((name, int(i), float(flat_abs[i])))
    items.sort(key=lambda x: x[2], reverse=True)
    return items[: min(top_w, len(items))]


def select_weight_candidates_by_gradient(
    model: onnx.ModelProto,
    cached_batches: Sequence[CachedBatch],
    output_slices: Dict[str, slice],
    input_shapes: Dict[str, Tuple[int, ...]],
    top_w: int,
    per_tensor_k: int,
    allow_bias: bool,
    restrict: Optional[List[str]],
    providers: Sequence[str] = ("CPUExecutionProvider",),
    epsilon: float = 1e-3,
    sample_all: bool = False,
) -> List[Tuple[str, int, float]]:
    """Select weight candidates based on gradient magnitude using numerical differentiation.

    For each weight parameter, compute numerical gradient:
        grad ≈ (loss(w + ε) - loss(w)) / ε

    Then select top-k weights per tensor by |grad|.
    """
    print("[Gradient] Computing baseline loss...")
    base_sess = make_session(model.SerializeToString(), providers=providers)
    base_loss = eval_loss(base_sess, cached_batches, output_slices, input_shapes)
    print(f"[Gradient] Baseline loss: {base_loss:.6f}")

    base_bytes = model.SerializeToString()
    items: List[Tuple[str, int, float]] = []

    # Iterate over all fp16 initializers
    tensor_list = list(iter_fp16_initializers(model, allow_bias=allow_bias, restrict=restrict))
    print(f"[Gradient] Computing gradients for {len(tensor_list)} tensors...")

    for tensor_idx, (name, arr) in enumerate(tqdm(tensor_list, desc="Computing gradients", unit="tensor")):
        flat = arr.reshape(-1)
        n = flat.size

        # Sample indices if tensor is too large
        if sample_all:
            # Test ALL weights in this tensor
            sample_idx = np.arange(n)
        elif per_tensor_k > 0 and per_tensor_k < n:
            # First sample by magnitude to focus on larger weights
            flat_abs = np.abs(flat.astype(np.float32))
            sample_k = min(per_tensor_k * 10, n)  # Sample 10x more for gradient computation
            sample_idx = np.argpartition(flat_abs, -sample_k)[-sample_k:]
        else:
            sample_idx = np.arange(n)

        gradients = np.zeros(len(sample_idx), dtype=np.float32)

        # Compute numerical gradient for sampled indices
        for i, flat_idx in enumerate(sample_idx):
            # Perturb weight by epsilon
            m = onnx.load_from_string(base_bytes)
            for t in m.graph.initializer:
                if t.name != name:
                    continue
                arr_copy = numpy_helper.to_array(t).copy()
                flat_arr = arr_copy.reshape(-1)

                original_val = float(flat_arr[flat_idx])
                # Add perturbation
                flat_arr[flat_idx] = np.float16(original_val + epsilon)

                t.CopyFrom(numpy_helper.from_array(arr_copy.astype(np.float16), name=t.name))
                break

            # Compute loss with perturbed weight
            perturbed_bytes = m.SerializeToString()
            perturbed_sess = make_session(perturbed_bytes, providers=providers)
            perturbed_loss = eval_loss(perturbed_sess, cached_batches, output_slices, input_shapes)

            # Numerical gradient (forward difference)
            grad = (perturbed_loss - base_loss) / epsilon
            gradients[i] = abs(grad)

        # Select top-k by gradient magnitude
        k = min(per_tensor_k, len(sample_idx))
        if k <= 0:
            continue

        top_k_idx = np.argpartition(gradients, -k)[-k:]
        for i in top_k_idx:
            flat_idx = int(sample_idx[i])
            grad_mag = float(gradients[i])
            items.append((name, flat_idx, grad_mag))

    # Sort by gradient magnitude and return top_w
    items.sort(key=lambda x: x[2], reverse=True)
    print(f"[Gradient] Selected {len(items[:top_w])} candidates by gradient magnitude")
    return items[: min(top_w, len(items))]


def flip_fp16_value(arr: np.ndarray, flat_idx: int, bit: int) -> Tuple[float, float]:
    flat = arr.reshape(-1)
    raw = flat.view(np.uint16)
    old = np.float16(flat[flat_idx])
    raw[flat_idx] ^= np.uint16(1 << bit)
    new = np.float16(flat[flat_idx])
    if not np.isfinite(new):
        raw[flat_idx] ^= np.uint16(1 << bit)
        return float(old), float(old)
    return float(old), float(new)


def make_flipped_model_bytes(base_bytes: bytes, name: str, flat_idx: int, bit: int) -> Tuple[Optional[bytes], float, float]:
    m = onnx.load_from_string(base_bytes)
    for t in m.graph.initializer:
        if t.name != name:
            continue
        arr = numpy_helper.to_array(t).copy()
        old, new = flip_fp16_value(arr, flat_idx, bit)
        if new == old:
            return None, old, new
        t.CopyFrom(numpy_helper.from_array(arr.astype(np.float16), name=t.name))
        return m.SerializeToString(), old, new
    raise KeyError(f"Initializer not found: {name}")


def rank_bits_progressive(
    base_model: onnx.ModelProto,
    original_score: float,
    cached_batches: Sequence[CachedBatch],
    output_slices: Dict[str, slice],
    input_shapes: Dict[str, Tuple[int, ...]],
    candidates: Sequence[Tuple[str, int]],
    bitset: Sequence[int],
    top_b: int,
    eval_metric: str,
    providers: Sequence[str] = ("CPUExecutionProvider",),
) -> List[Dict[str, Any]]:
    if not np.isfinite(original_score):
        raise RuntimeError(f"Baseline metric score is non-finite: {original_score}")

    current_model_bytes = base_model.SerializeToString()
    current_score = float(original_score)
    records: List[Dict[str, Any]] = []

    pending: List[Tuple[str, int, int]] = [(n, fi, b) for (n, fi) in candidates for b in bitset]
    if not pending:
        return records

    total_steps = min(top_b, len(pending))
    skipped_non_finite = 0
    outer = tqdm(total=total_steps, desc="Progressive rank", unit="step", dynamic_ncols=True)
    for step in range(total_steps):
        if not pending:
            break

        best_idx = -1
        best_item: Optional[Dict[str, Any]] = None
        inner = tqdm(total=len(pending), desc=f"Scan step {step+1}", unit="flip", leave=False, dynamic_ncols=True)
        for idx, (name, flat_idx, bit) in enumerate(pending):
            model_bytes, old, new = make_flipped_model_bytes(current_model_bytes, name, flat_idx, int(bit))
            if model_bytes is None:
                inner.update(1)
                continue

            sess = make_session(model_bytes, providers=providers)
            flipped_score = eval_metric_value(
                sess,
                cached_batches,
                output_slices,
                input_shapes,
                eval_metric,
            )
            if not np.isfinite(flipped_score):
                skipped_non_finite += 1
                inner.update(1)
                continue

            score = float(flipped_score - current_score)
            if (best_item is None) or (score > best_item["score"]):
                best_idx = idx
                best_item = {
                    "name": name,
                    "flat": int(flat_idx),
                    "bit": int(bit),
                    "metric": eval_metric,
                    "score": float(score),
                    "original_score": float(current_score),
                    "flipped_score": float(flipped_score),
                    "old": float(old),
                    "new": float(new),
                    "step": int(step + 1),
                    "_model_bytes": model_bytes,
                }
                inner.set_postfix(best_score=f"{score:.4f}")
            inner.update(1)
        inner.close()

        if best_item is None:
            print("[Rank] stop early: no finite candidate remained.")
            break

        current_model_bytes = best_item.pop("_model_bytes")
        current_score = float(best_item["flipped_score"])
        records.append(best_item)

        pending.pop(best_idx)
        outer.update(1)
        outer.set_postfix(current_score=f"{current_score:.4f}", kept=len(records), pending=len(pending), skipped_nan=skipped_non_finite)
    outer.close()

    return records


def save_weight_candidates(
    path: str,
    candidates: Sequence[Tuple[str, int, float]],
    *,
    onnx_path: str,
    restrict: Optional[List[str]],
    allow_bias: bool,
    top_w: int,
    per_tensor_k: int,
    selection_method: str = "magnitude",
    gradient_epsilon: Optional[float] = None,
) -> None:
    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "onnx": os.path.abspath(onnx_path),
            "restrict": restrict,
            "allow_bias": bool(allow_bias),
            "top_w": int(top_w),
            "per_tensor_k": int(per_tensor_k),
            "num_candidates": int(len(candidates)),
            "selection_method": selection_method,
        },
        "candidates": [
            {"name": name, "flat": int(flat), "score": float(score)}
            for name, flat, score in candidates
        ],
    }
    if gradient_epsilon is not None:
        payload["meta"]["gradient_epsilon"] = float(gradient_epsilon)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_weight_candidates(path: str) -> List[Tuple[str, int, float]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("candidates", [])
    out: List[Tuple[str, int, float]] = []
    for r in rows:
        out.append((str(r["name"]), int(r["flat"]), float(r.get("score", 0.0))))
    return out


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
    default_split_dir = root / "op_097" / "splits"

    p = argparse.ArgumentParser(description="Important bit search for openpilot 0.9.7 ONNX")
    p.add_argument("--onnx", default=str(root / "op_097" / "models" / "supercombo.onnx"))
    p.add_argument("--metadata", default=str(root / "op_097" / "models" / "supercombo_metadata.pkl"))
    p.add_argument("--data-root", default="/home/xzha135/work/comma2k19")
    p.add_argument("--train-index", default=str(default_split_dir / "comma2k19_train_non_overlap.txt"))
    p.add_argument("--val-index", default=str(default_split_dir / "comma2k19_val_non_overlap.txt"))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-val-batches", type=int, default=2)
    p.add_argument("--top-w", type=int, default=50)
    p.add_argument("--per-tensor-k", type=int, default=1)
    p.add_argument("--top-b", type=int, default=1)
    p.add_argument("--bitset", default="exponent_sign", help="fp16 bit set: mantissa/exponent/sign/exponent_sign/all or csv")
    p.add_argument("--allow-bias", action="store_true")
    p.add_argument("--restrict", default="", help="comma-separated substrings to keep, e.g. vision,policy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", choices=["all", "select-weights", "rank-bits"], default="all")
    p.add_argument("--weights-in", default="", help="JSON file produced by select-weights stage")
    p.add_argument("--weights-out", default=str(root / "op_097" / "weights_candidates_097.json"))
    p.add_argument(
        "--eval-metric",
        choices=["loss", "+diffx", "-diffx", "+diffy", "-diffy"],
        default="loss",
        help="ranking score: loss delta, or signed trajectory delta between flipped and clean",
    )
    p.add_argument("--out", default=str(root / "op_097" / "important_bits_097.json"))
    p.add_argument("--eval-seq-len", type=int, default=20, help="timesteps used per cached sequence; <=0 means full sequence")
    p.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto", help="ONNX Runtime provider")
    p.add_argument("--inspect-only", action="store_true", help="Only print observed output stats and baseline loss")
    p.add_argument(
        "--weight-selection-method",
        choices=["magnitude", "gradient"],
        default="magnitude",
        help="Weight selection method: magnitude (absolute value) or gradient (numerical gradient)"
    )
    p.add_argument("--gradient-epsilon", type=float, default=1e-3, help="Epsilon for numerical gradient computation")
    p.add_argument("--sample-all-weights", action="store_true", help="Compute gradients for ALL weights in each tensor (ignores per-tensor-k)")
    return p.parse_args()


def main() -> None:
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_ts = timestamp_id()
    out_dir = os.path.join(REPO_ROOT, "op_097", "out")
    weights_out_path = with_timestamp(args.weights_out, run_ts, out_dir)
    result_out_path = with_timestamp(args.out, run_ts, out_dir)
    providers = resolve_providers(args.provider)
    print(f"[ORT] providers={providers}")

    import pickle

    with open(args.metadata, "rb") as f:
        metadata = pickle.load(f)

    output_slices: Dict[str, slice] = metadata["output_slices"]
    input_shapes: Dict[str, Tuple[int, ...]] = metadata["input_shapes"]
    restrict = [x.strip() for x in args.restrict.split(",") if x.strip()] or None
    bitset = parse_bitset(args.bitset)
    bitset_mode = canonical_bitset_mode(args.bitset)

    if args.stage == "select-weights":
        base_model = onnx.load(args.onnx)

        if args.weight_selection_method == "gradient":
            # Need to load data for gradient computation
            train_loader, val_loader = build_loaders(args)
            cached_batches = collect_cached_batches(
                val_loader,
                input_shapes=input_shapes,
                num_batches=args.num_val_batches,
                eval_seq_len=args.eval_seq_len,
            )
            if not cached_batches:
                raise RuntimeError("No validation batches cached. Check dataset path/splits.")

            mode_str = "ALL weights" if args.sample_all_weights else f"sampled (per_tensor_k={args.per_tensor_k})"
            print(f"[Weights] Using gradient-based selection (epsilon={args.gradient_epsilon}, mode={mode_str})")
            selected = select_weight_candidates_by_gradient(
                base_model,
                cached_batches=cached_batches,
                output_slices=output_slices,
                input_shapes=input_shapes,
                top_w=args.top_w,
                per_tensor_k=args.per_tensor_k,
                allow_bias=args.allow_bias,
                restrict=restrict,
                providers=providers,
                epsilon=args.gradient_epsilon,
                sample_all=args.sample_all_weights,
            )
        else:
            print(f"[Weights] Using magnitude-based selection")
            selected = select_weight_candidates(
                base_model,
                top_w=args.top_w,
                per_tensor_k=args.per_tensor_k,
                allow_bias=args.allow_bias,
                restrict=restrict,
            )

        if not selected:
            raise RuntimeError("No fp16 initializer candidates selected.")
        save_weight_candidates(
            weights_out_path,
            selected,
            onnx_path=args.onnx,
            restrict=restrict,
            allow_bias=args.allow_bias,
            top_w=args.top_w,
            per_tensor_k=args.per_tensor_k,
            selection_method=args.weight_selection_method,
            gradient_epsilon=args.gradient_epsilon if args.weight_selection_method == "gradient" else None,
        )
        print(f"[Done] Saved {len(selected)} weight candidates to: {weights_out_path}")
        return

    train_loader, val_loader = build_loaders(args)
    cached_batches = collect_cached_batches(
        val_loader,
        input_shapes=input_shapes,
        num_batches=args.num_val_batches,
        eval_seq_len=args.eval_seq_len,
    )
    if not cached_batches:
        raise RuntimeError("No validation batches cached. Check dataset path/splits.")

    base_sess = make_session(args.onnx, providers=providers)
    providers = list(base_sess.get_providers())
    print(f"[ORT] active_providers={providers}")
    inspect_outputs(base_sess, cached_batches[0], output_slices, input_shapes)
    base_loss = eval_loss(base_sess, cached_batches, output_slices, input_shapes)
    print(f"\n[Eval] Baseline planning loss on cached val batches: {base_loss:.6f}")
    original_metric_score = eval_metric_value(
        base_sess,
        cached_batches,
        output_slices,
        input_shapes,
        args.eval_metric,
    )
    print(f"[Eval] Baseline {args.eval_metric} score on cached val batches: {original_metric_score:.6f}")

    if args.inspect_only:
        print("[Done] inspect-only mode")
        return

    base_model = onnx.load(args.onnx)
    if args.weights_in:
        selected = load_weight_candidates(args.weights_in)
        if args.top_w > 0:
            selected = selected[: min(args.top_w, len(selected))]
        print(f"[Weights] loaded {len(selected)} candidates from {args.weights_in}")
    else:
        if args.weight_selection_method == "gradient":
            mode_str = "ALL weights" if args.sample_all_weights else f"sampled (per_tensor_k={args.per_tensor_k})"
            print(f"[Weights] Using gradient-based selection (epsilon={args.gradient_epsilon}, mode={mode_str})")
            selected = select_weight_candidates_by_gradient(
                base_model,
                cached_batches=cached_batches,
                output_slices=output_slices,
                input_shapes=input_shapes,
                top_w=args.top_w,
                per_tensor_k=args.per_tensor_k,
                allow_bias=args.allow_bias,
                restrict=restrict,
                providers=providers,
                epsilon=args.gradient_epsilon,
                sample_all=args.sample_all_weights,
            )
        else:
            print(f"[Weights] Using magnitude-based selection")
            selected = select_weight_candidates(
                base_model,
                top_w=args.top_w,
                per_tensor_k=args.per_tensor_k,
                allow_bias=args.allow_bias,
                restrict=restrict,
            )

        if args.stage == "all":
            save_weight_candidates(
                weights_out_path,
                selected,
                onnx_path=args.onnx,
                restrict=restrict,
                allow_bias=args.allow_bias,
                top_w=args.top_w,
                per_tensor_k=args.per_tensor_k,
                selection_method=args.weight_selection_method,
                gradient_epsilon=args.gradient_epsilon if args.weight_selection_method == "gradient" else None,
            )
            print(f"[Weights] saved {len(selected)} candidates to {weights_out_path}")
    if not selected:
        raise RuntimeError("No fp16 initializer candidates selected.")

    candidates = [(n, fi) for (n, fi, _score) in selected]
    ranked = rank_bits_progressive(
        base_model=base_model,
        original_score=original_metric_score,
        cached_batches=cached_batches,
        output_slices=output_slices,
        input_shapes=input_shapes,
        candidates=candidates,
        bitset=bitset,
        top_b=args.top_b,
        eval_metric=args.eval_metric,
        providers=providers,
    )

    plan = [{"name": r["name"], "flat": r["flat"], "bit": r["bit"]} for r in ranked]
    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "onnx": os.path.abspath(args.onnx),
            "metadata": os.path.abspath(args.metadata),
            "data_root": os.path.abspath(args.data_root),
            "train_index": os.path.abspath(args.train_index),
            "val_index": os.path.abspath(args.val_index),
            "num_val_batches": args.num_val_batches,
            "eval_seq_len": args.eval_seq_len,
            "top_w": args.top_w,
            "top_b": args.top_b,
            "per_tensor_k": args.per_tensor_k,
            "bitset_mode": bitset_mode,
            "bitset": bitset,
            "allow_bias": bool(args.allow_bias),
            "restrict": restrict,
            "original_metric_score": original_metric_score,
            "eval_metric": args.eval_metric,
            "search_mode": "progressive",
            "weight_selection_method": args.weight_selection_method,
        },
        "ranked": ranked,
        "plan": plan,
    }
    if args.weight_selection_method == "gradient":
        payload["meta"]["gradient_epsilon"] = args.gradient_epsilon

    os.makedirs(os.path.dirname(result_out_path) or ".", exist_ok=True)
    with open(result_out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[Done] Saved ranked bit plan to: {result_out_path}")
    if ranked:
        top = ranked[0]
        print(
            f"[Top-1] name={top['name']} flat={top['flat']} bit={top['bit']} "
            f"metric={top['metric']} score={top['score']:.6f} "
            f"({top['original_score']:.6f} -> {top['flipped_score']:.6f})"
        )


if __name__ == "__main__":
    main()
