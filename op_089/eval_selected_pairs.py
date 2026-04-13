#!/usr/bin/env python3
"""Evaluate selected op_089 ONNX/JSON pairs on acceleration and curvature metrics."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from scipy.interpolate import interp1d
from tqdm import tqdm

from recurrent_bit_utils import (
    compute_action_targets_from_output,
    evaluate_recurrent_metrics,
    flip_scalar_bit_fast_,
    live_params,
    load_flip_rows_json,
    load_model,
    validate_target_mode,
)
from data import Comma2k19SequenceDataset
import utils_comma2k19.orientation as orient


TIMESTAMP_RE = re.compile(r"\d{8}-\d{2}:\d{2}:\d{2}")


@dataclass
class Pair:
    json_path: Path
    onnx_path: Path


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser("Evaluate selected op_089 ONNX/JSON pairs")
    ap.add_argument("--selected-dir", default=str(root / "op_089" / "selected"))
    ap.add_argument("--out-dir", default=str(root / "op_089" / "out" / "selected"))
    ap.add_argument("--data-root", default="/home/zx/Projects/comma2k19")
    ap.add_argument("--val-index", default=str(root / "op_097" / "splits" / "comma2k19_val_non_overlap.txt"))
    ap.add_argument(
        "--base-ckpt",
        default="/home/zx/Projects/openpilot_0.8.9/openpilot/flipped_models/supercombo_torch_weights.pth",
    )
    ap.add_argument("--batch-sizes", default="1,2")
    ap.add_argument("--recurrent-num", type=int, default=100)
    ap.add_argument("--num-val-batches", type=int, default=2)
    ap.add_argument("--flip-count", type=int, default=2)
    ap.add_argument("--torch-micro-batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--target-mode", "--target_mode", dest="target_mode", default="pseudo_controls")
    ap.add_argument("--amp", action="store_true")
    return ap.parse_args()


def parse_batch_sizes(spec: str) -> List[int]:
    out = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not out:
        raise ValueError("batch sizes cannot be empty")
    return out


def _stem_suffix(stem: str, prefix: str) -> str:
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def _extract_timestamp(text: str) -> Optional[str]:
    m = TIMESTAMP_RE.search(text)
    return m.group(0) if m else None


def pair_selected_files(selected_dir: Path) -> List[Pair]:
    jsons = sorted(selected_dir.glob("flipped_bits*.json"))
    onnxs = sorted(selected_dir.glob("*.onnx"))
    if not jsons or not onnxs:
        raise RuntimeError(f"No selected json/onnx files found in {selected_dir}")

    pairs: List[Pair] = []
    used_onnx: set[Path] = set()
    for json_path in jsons:
        json_ts = _extract_timestamp(json_path.stem)
        candidates: List[Path] = []
        if json_ts is not None:
            candidates = [p for p in onnxs if _extract_timestamp(p.stem) == json_ts and p not in used_onnx]
        if not candidates:
            json_suffix = _stem_suffix(json_path.stem, "flipped_bits_")
            candidates = [
                p
                for p in onnxs
                if _stem_suffix(p.stem, "supercombo_2_") == json_suffix and p not in used_onnx
            ]
        if len(candidates) != 1:
            raise RuntimeError(f"Could not uniquely pair {json_path.name}: candidates={candidates}")
        onnx_path = candidates[0]
        used_onnx.add(onnx_path)
        pairs.append(Pair(json_path=json_path, onnx_path=onnx_path))
    return pairs


def evaluate_independent_top_bits(
    model: torch.nn.Module,
    cached_batches: Sequence[Dict[str, torch.Tensor]],
    device: torch.device,
    flips: Sequence[Dict[str, Any]],
    baseline: Dict[str, float],
    *,
    recurrent_num: int,
    use_amp: bool,
    target_mode: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    param_map = live_params(model)
    for flip in flips:
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
            recurrent_num=recurrent_num,
            use_amp=use_amp,
            target_mode=target_mode,
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
                "flipped_metrics": scores,
                "delta": {k: float(scores[k] - baseline[k]) for k in baseline},
            }
        )
    return rows


def evaluate_cumulative_top_bits(
    model: torch.nn.Module,
    cached_batches: Sequence[Dict[str, torch.Tensor]],
    device: torch.device,
    flips: Sequence[Dict[str, Any]],
    baseline: Dict[str, float],
    *,
    recurrent_num: int,
    use_amp: bool,
    target_mode: str,
) -> Dict[str, Any]:
    param_map = live_params(model)
    applied: List[Dict[str, Any]] = []
    for flip in flips:
        name = str(flip["name"])
        flat_idx = int(flip["index_flat"])
        bit = int(flip["bit"])
        if name not in param_map:
            continue
        flip_scalar_bit_fast_(param_map[name], flat_idx, bit)
        applied.append({"name": name, "index_flat": flat_idx, "bit": bit})

    scores = evaluate_recurrent_metrics(
        model,
        cached_batches,
        device,
        recurrent_num=recurrent_num,
        use_amp=use_amp,
        target_mode=target_mode,
    )
    return {
        "applied": applied,
        "metrics": scores,
        "delta": {k: float(scores[k] - baseline[k]) for k in baseline},
    }


def save_pair_result(out_path: Path, payload: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def split_cached_batches(
    cached_batches: Sequence[Dict[str, torch.Tensor]],
    max_batch_size: int,
) -> List[Dict[str, torch.Tensor]]:
    out: List[Dict[str, torch.Tensor]] = []
    for batch in cached_batches:
        bsz = int(batch["seq_input_img"].shape[0])
        if bsz <= max_batch_size:
            out.append(batch)
            continue
        for start in range(0, bsz, max_batch_size):
            end = min(start + max_batch_size, bsz)
            out.append(
                {
                    key: value[start:end].clone() if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
            )
    return out


def evaluate_recurrent_metrics_onnx(
    onnx_path: str,
    cached_batches: Sequence[Dict[str, torch.Tensor]],
    *,
    recurrent_num: int,
    target_mode: str,
) -> Dict[str, float]:
    def _run_with_providers(providers: List[str]) -> Dict[str, float]:
        session = ort.InferenceSession(onnx_path, providers=providers)
        input_meta = {x.name: x for x in session.get_inputs()}
        totals: Dict[str, List[np.ndarray]] = {
            "action.desiredAcceleration": [],
            "action.desiredCurvature": [],
            "action.curvatureDelta": [],
        }
        for batch in cached_batches:
            seq_imgs12 = batch["seq_input_img"].float()
            seq_gt = batch["seq_future_poses"].float()
            bsz = seq_imgs12.shape[0]

            if seq_imgs12.shape[2] == 6:
                seq_imgs12 = torch.cat([seq_imgs12, seq_imgs12], dim=2)
            seq_imgs12 = seq_imgs12[:, :recurrent_num]
            seq_gt = seq_gt[:, :recurrent_num]

            desire = np.zeros((bsz, 8), dtype=np.float32)
            traffic = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (bsz, 1))
            hidden = np.zeros((bsz, 512), dtype=np.float32)

            prev_desired_accel = np.zeros((bsz,), dtype=np.float32)
            prev_desired_curvature = np.zeros((bsz,), dtype=np.float32)

            for step in range(seq_imgs12.shape[1]):
                imgs_np = seq_imgs12[:, step].cpu().numpy().astype(np.float32)

                def _batch_is_fixed_one(name: str) -> bool:
                    shape = getattr(input_meta.get(name), "shape", None)
                    return bool(shape) and shape[0] == 1

                if any(_batch_is_fixed_one(name) for name in ("input_imgs", "desire", "traffic_convention", "initial_state")):
                    outs = []
                    next_hidden = []
                    for i in range(bsz):
                        out_i = session.run(
                            ["outputs"],
                            {
                                "input_imgs": imgs_np[i : i + 1],
                                "desire": desire[i : i + 1],
                                "traffic_convention": traffic[i : i + 1],
                                "initial_state": hidden[i : i + 1],
                            },
                        )[0]
                        outs.append(out_i)
                        next_hidden.append(out_i[:, -512:])
                    out = np.concatenate(outs, axis=0)
                    hidden = np.concatenate(next_hidden, axis=0).astype(np.float32, copy=False)
                else:
                    out = session.run(
                        ["outputs"],
                        {
                            "input_imgs": imgs_np,
                            "desire": desire,
                            "traffic_convention": traffic,
                            "initial_state": hidden,
                        },
                    )[0]
                    hidden = out[:, -512:].astype(np.float32, copy=False)
                out_t = torch.from_numpy(out.astype(np.float32))
                action_targets, prev_desired_accel, prev_desired_curvature = compute_action_targets_from_output(
                    out_t,
                    prev_desired_accel,
                    prev_desired_curvature,
                    target_mode=target_mode,
                )
                for key in totals:
                    totals[key].append(action_targets[key])

        scores: Dict[str, float] = {}
        for key, values in totals.items():
            merged = np.concatenate(values, axis=0)
            finite = merged[np.isfinite(merged)]
            scores[key] = float(np.mean(finite)) if finite.size else float("nan")
        return scores

    try:
        return _run_with_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
    except Exception as exc:
        print(f"[ONNX] CUDA provider failed for {onnx_path}: {exc}. Retrying on CPU.")
        return _run_with_providers(["CPUExecutionProvider"])


def load_quick_cached_batches(
    data_root: str,
    val_index: str,
    batch_size: int,
    num_val_batches: int,
    recurrent_num: int,
) -> List[Dict[str, torch.Tensor]]:
    ds = Comma2k19SequenceDataset(val_index, data_root.rstrip("/") + "/", "val", use_memcache=False, inner_progress=False)
    rel_paths = [line.strip() for line in Path(val_index).read_text(encoding="utf-8").splitlines() if line.strip()]
    needed = batch_size * num_val_batches

    samples: List[Dict[str, torch.Tensor]] = []
    for rel in rel_paths:
        if len(samples) >= needed:
            break
        seg = Path(data_root) / rel
        cap = ds._get_cv2_vid(str(seg / "video.hevc"))
        frames = []
        target_frames = recurrent_num + 1
        for _ in range(target_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if len(frames) < target_frames:
            continue

        proc = []
        for frame in frames:
            warped = cv2.warpPerspective(src=frame, M=ds.warp_matrix, dsize=(512, 256), flags=cv2.WARP_INVERSE_MAP)
            rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            proc.append(ds.transforms(Image.fromarray(rgb))[None])
        img_stack = torch.cat(proc, dim=0)
        seq_input_img = torch.cat((img_stack[:-1], img_stack[1:]), dim=1)

        pose_need = recurrent_num + ds.num_pts
        frame_positions = ds._get_numpy(str(seg / "global_pose" / "frame_positions"))[:pose_need]
        frame_orientations = ds._get_numpy(str(seg / "global_pose" / "frame_orientations"))[:pose_need]
        if len(frame_positions) < pose_need or len(frame_orientations) < pose_need:
            continue

        future_poses = []
        for i in range(recurrent_num):
            ecef_from_local = orient.rot_from_quat(frame_orientations[i])
            local_from_ecef = ecef_from_local.T
            frame_positions_local = np.einsum("ij,kj->ki", local_from_ecef, frame_positions - frame_positions[i]).astype(np.float32)
            fs = [interp1d(ds.t_idx, frame_positions_local[i : i + ds.num_pts, j]) for j in range(3)]
            interp_positions = [fs[j](ds.t_anchors)[:, None] for j in range(3)]
            future_poses.append(np.concatenate(interp_positions, axis=1))

        samples.append(
            {
                "seq_input_img": seq_input_img,
                "seq_future_poses": torch.tensor(np.array(future_poses), dtype=torch.float32),
            }
        )

    if len(samples) < needed:
        raise RuntimeError(f"Need {needed} valid validation segments, got {len(samples)}")

    cached_batches: List[Dict[str, torch.Tensor]] = []
    for start in range(0, needed, batch_size):
        chunk = samples[start : start + batch_size]
        cached_batches.append(
            {
                "seq_input_img": torch.stack([x["seq_input_img"] for x in chunk], dim=0),
                "seq_future_poses": torch.stack([x["seq_future_poses"] for x in chunk], dim=0),
            }
        )
    return cached_batches


def main() -> None:
    args = parse_args()
    args.target_mode = validate_target_mode(args.target_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_dir = Path(args.selected_dir)
    out_dir = Path(args.out_dir)
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    pairs = pair_selected_files(selected_dir)

    print(f"[Device] {device}")
    print(f"[Pairs] {len(pairs)} selected pairs found")

    summary: Dict[str, Any] = {
        "meta": {
            "timestamp": timestamp_id(),
            "selected_dir": str(selected_dir),
            "out_dir": str(out_dir),
            "data_root": args.data_root,
            "val_index": args.val_index,
            "base_ckpt": args.base_ckpt,
            "batch_sizes": batch_sizes,
            "recurrent_num": args.recurrent_num,
            "num_val_batches": args.num_val_batches,
            "flip_count": args.flip_count,
        },
        "results": [],
    }

    for batch_size in batch_sizes:
        print(f"[Batch] batch_size={batch_size}: caching validation batches (quick path)")
        cached_batches = load_quick_cached_batches(
            args.data_root,
            args.val_index,
            batch_size,
            args.num_val_batches,
            args.recurrent_num,
        )
        torch_eval_batches = split_cached_batches(cached_batches, args.torch_micro_batch)
        base_model = load_model(args.base_ckpt, device)
        baseline = evaluate_recurrent_metrics(
            base_model,
            torch_eval_batches,
            device,
            recurrent_num=args.recurrent_num,
            use_amp=args.amp,
            target_mode=args.target_mode,
        )
        print("[Baseline] " + ", ".join(f"{k}={v:.6f}" for k, v in baseline.items()))

        for pair in tqdm(pairs, desc=f"Evaluate bs={batch_size}", leave=False):
            flips = load_flip_rows_json(str(pair.json_path), limit=args.flip_count)

            selected_scores = evaluate_recurrent_metrics_onnx(
                str(pair.onnx_path),
                cached_batches,
                recurrent_num=args.recurrent_num,
                target_mode=args.target_mode,
            )

            independent_model = load_model(args.base_ckpt, device)
            independent_rows = evaluate_independent_top_bits(
                independent_model,
                torch_eval_batches,
                device,
                flips,
                baseline,
                recurrent_num=args.recurrent_num,
                use_amp=args.amp,
                target_mode=args.target_mode,
            )

            cumulative_model = load_model(args.base_ckpt, device)
            cumulative = evaluate_cumulative_top_bits(
                cumulative_model,
                torch_eval_batches,
                device,
                flips,
                baseline,
                recurrent_num=args.recurrent_num,
                use_amp=args.amp,
                target_mode=args.target_mode,
            )

            pair_key = f"{pair.json_path.stem}_bs{batch_size}"
            out_path = out_dir / f"{pair_key}.json"
            payload = {
                "meta": {
                    "timestamp": timestamp_id(),
                    "batch_size": batch_size,
                    "recurrent_num": args.recurrent_num,
                    "num_val_batches": args.num_val_batches,
                    "flip_count": args.flip_count,
                    "target_mode": args.target_mode,
                    "json_path": str(pair.json_path),
                    "onnx_path": str(pair.onnx_path),
                    "base_ckpt": args.base_ckpt,
                },
                "baseline_metrics": baseline,
                "selected_model_metrics": selected_scores,
                "selected_model_delta": {k: float(selected_scores[k] - baseline[k]) for k in baseline},
                "candidate_top_bits": independent_rows,
                "cumulative_top_bits": cumulative,
            }
            save_pair_result(out_path, payload)
            summary["results"].append(
                {
                    "batch_size": batch_size,
                    "json_path": str(pair.json_path),
                    "onnx_path": str(pair.onnx_path),
                    "out_path": str(out_path),
                }
            )
            print(f"[Save] {out_path}")

    summary_path = out_dir / "selected_pairs_summary.json"
    save_pair_result(summary_path, summary)
    print(f"[Save] {summary_path}")


if __name__ == "__main__":
    main()
