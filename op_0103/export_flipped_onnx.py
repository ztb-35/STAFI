#!/usr/bin/env python3
"""Export flipped op_0103 ONNX model(s) from a ranked bit-plan JSON."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VISION_ONNX = REPO_ROOT / "op_0103" / "models" / "driving_vision.onnx"
DEFAULT_POLICY_ONNX = REPO_ROOT / "op_0103" / "models" / "driving_policy.onnx"
DEFAULT_OUT_DIR = REPO_ROOT / "op_0103" / "out"
TIMESTAMP_RE = re.compile(r"(?P<ts>\d{8}-\d{6})")


@dataclass
class FlipSpec:
    model: str
    name: str
    flat: int
    bit: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export op_0103 ONNX with top-N bit flips from plan JSON")
    p.add_argument("--plan-json", required=True, help="JSON produced by op_0103/important_bits_onnx.py")
    p.add_argument("--target-model", choices=["vision", "policy", "both"], default="vision")
    p.add_argument("--source-key", choices=["plan", "ranked"], default="plan", help="Which JSON list to consume")
    p.add_argument("--topn", type=int, default=1, help="Apply first N flips after filtering by target model")
    p.add_argument("--vision-onnx", default=str(DEFAULT_VISION_ONNX))
    p.add_argument("--policy-onnx", default=str(DEFAULT_POLICY_ONNX))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument(
        "--onnx-out",
        default="",
        help="Optional explicit output path. Only valid when --target-model is vision or policy.",
    )
    p.add_argument("--strict", action="store_true", help="Fail on missing initializer or out-of-range index")
    return p.parse_args()


def parse_timestamp(plan_json: str, payload: Dict[str, Any]) -> str:
    match = TIMESTAMP_RE.search(os.path.basename(plan_json))
    if match:
        return match.group("ts")

    created_at = payload.get("meta", {}).get("created_at")
    if isinstance(created_at, str):
        compact = re.sub(r"[^0-9]", "", created_at)
        if len(compact) >= 14:
            return f"{compact[:8]}-{compact[8:14]}"

    raise ValueError(f"Could not infer timestamp from JSON filename or meta.created_at: {plan_json}")


def metric_suffix(payload: Dict[str, Any]) -> str:
    raw = payload.get("meta", {}).get("eval_metric")
    if not isinstance(raw, str) or not raw.strip():
        return ""

    metric = raw.strip().lower()
    metric = metric.replace("+", "pos")
    metric = metric.replace("-", "neg")
    metric = re.sub(r"[^a-z0-9]+", "_", metric).strip("_")
    return metric


def load_specs(plan_json: str, source_key: str) -> Tuple[Dict[str, Any], List[FlipSpec]]:
    with open(plan_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get(source_key)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No valid entries under '{source_key}' in {plan_json}")

    specs: List[FlipSpec] = []
    for row in rows:
        try:
            specs.append(
                FlipSpec(
                    model=str(row["model"]),
                    name=str(row["name"]),
                    flat=int(row["flat"]),
                    bit=int(row["bit"]),
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Bad row in '{source_key}': {row}") from exc

    return payload, specs


def filter_specs(specs: Sequence[FlipSpec], target_model: str, topn: int) -> Dict[str, List[FlipSpec]]:
    if topn <= 0:
        raise ValueError("--topn must be >= 1")

    wanted = ("vision", "policy") if target_model == "both" else (target_model,)
    out: Dict[str, List[FlipSpec]] = {}
    for model_key in wanted:
        rows = [spec for spec in specs if spec.model == model_key][:topn]
        if not rows:
            raise ValueError(f"No flips found for target model '{model_key}'")
        out[model_key] = rows
    return out


def flip_scalar(arr: np.ndarray, flat_idx: int, bit: int) -> Tuple[float, float]:
    flat = arr.reshape(-1)
    if arr.dtype == np.float16:
        if not (0 <= bit <= 15):
            raise ValueError(f"fp16 bit out of range: {bit}")
        raw = flat.view(np.uint16)
        old = np.float16(flat[flat_idx])
        raw[flat_idx] ^= np.uint16(1 << bit)
        new = np.float16(flat[flat_idx])
        return float(old), float(new)
    if arr.dtype == np.float32:
        if not (0 <= bit <= 31):
            raise ValueError(f"fp32 bit out of range: {bit}")
        raw = flat.view(np.uint32)
        old = np.float32(flat[flat_idx])
        raw[flat_idx] ^= np.uint32(1 << bit)
        new = np.float32(flat[flat_idx])
        return float(old), float(new)
    raise TypeError(f"Unsupported dtype for bit flip: {arr.dtype}")


def apply_flips(
    model: onnx.ModelProto,
    specs: Sequence[FlipSpec],
    strict: bool,
) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    init_by_name = {t.name: t for t in model.graph.initializer}

    applied = 0
    skipped = 0
    non_finite = 0
    records: List[Dict[str, Any]] = []

    for idx, spec in enumerate(specs):
        tensor = init_by_name.get(spec.name)
        if tensor is None:
            msg = f"[{idx}] missing initializer: {spec.name}"
            if strict:
                raise KeyError(msg)
            print(f"[Skip] {msg}")
            skipped += 1
            continue

        arr = numpy_helper.to_array(tensor).copy()
        if spec.flat < 0 or spec.flat >= arr.size:
            msg = f"[{idx}] flat index out of range: {spec.name} flat={spec.flat} numel={arr.size}"
            if strict:
                raise IndexError(msg)
            print(f"[Skip] {msg}")
            skipped += 1
            continue

        old, new = flip_scalar(arr, spec.flat, spec.bit)
        if not np.isfinite(new):
            non_finite += 1

        tensor.CopyFrom(numpy_helper.from_array(arr, name=tensor.name))
        applied += 1
        records.append(
            {
                "idx": idx,
                "model": spec.model,
                "name": spec.name,
                "flat": spec.flat,
                "bit": spec.bit,
                "old": old,
                "new": new,
                "finite": bool(np.isfinite(new)),
            }
        )

    return applied, skipped, non_finite, records


def default_output_path(out_dir: str, model_key: str, ts: str, metric: str, topn: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    parts = [f"{model_key}_model", ts]
    if metric:
        parts.append(metric)
    parts.append(f"top{topn}")
    return os.path.join(out_dir, "_".join(parts) + ".onnx")


def export_one(
    *,
    model_key: str,
    onnx_in: str,
    onnx_out: str,
    specs: Sequence[FlipSpec],
    strict: bool,
) -> None:
    model = onnx.load(onnx_in)
    applied, skipped, non_finite, records = apply_flips(model, specs, strict=strict)
    onnx.save(model, onnx_out)

    print(f"[Done] target_model={model_key}")
    print(f"[Done] input={onnx_in}")
    print(f"[Done] output={onnx_out}")
    print(f"[Done] requested={len(specs)} applied={applied} skipped={skipped} non_finite_new={non_finite}")
    for record in records[: min(10, len(records))]:
        print(
            f"  idx={record['idx']} name={record['name']} flat={record['flat']} "
            f"bit={record['bit']} old={record['old']:.8g} new={record['new']:.8g} finite={record['finite']}"
        )


def main() -> None:
    args = parse_args()
    payload, specs = load_specs(args.plan_json, args.source_key)
    ts = parse_timestamp(args.plan_json, payload)
    metric = metric_suffix(payload)
    by_model = filter_specs(specs, args.target_model, args.topn)

    if args.onnx_out and args.target_model == "both":
        raise ValueError("--onnx-out cannot be used with --target-model both")

    inputs = {
        "vision": os.path.abspath(args.vision_onnx),
        "policy": os.path.abspath(args.policy_onnx),
    }

    for model_key, model_specs in by_model.items():
        onnx_out = (
            os.path.abspath(args.onnx_out)
            if args.onnx_out
            else default_output_path(os.path.abspath(args.out_dir), model_key, ts, metric, len(model_specs))
        )
        export_one(
            model_key=model_key,
            onnx_in=inputs[model_key],
            onnx_out=onnx_out,
            specs=model_specs,
            strict=args.strict,
        )


if __name__ == "__main__":
    main()
