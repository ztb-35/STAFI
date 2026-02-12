#!/usr/bin/env python3
"""Apply bit-flip plan JSON to ONNX initializers and export a new ONNX file."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


@dataclass
class FlipSpec:
    name: str
    flat: int
    bit: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flip ONNX model bits based on plan JSON")
    p.add_argument("--onnx-in", required=True, help="Input ONNX path")
    p.add_argument("--plan-json", required=True, help="JSON from important_bits_onnx.py")
    p.add_argument("--onnx-out", required=True, help="Output ONNX path")
    p.add_argument("--source-key", choices=["plan", "ranked"], default="plan", help="Which JSON key to consume")
    p.add_argument("--topk", type=int, default=0, help="Only apply first K entries; <=0 means all")
    p.add_argument("--strict", action="store_true", help="Fail on missing initializer or out-of-range index")
    return p.parse_args()


def load_specs(plan_json: str, source_key: str, topk: int) -> List[FlipSpec]:
    with open(plan_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get(source_key)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No valid entries under '{source_key}' in {plan_json}")

    specs: List[FlipSpec] = []
    for r in rows:
        try:
            specs.append(FlipSpec(name=str(r["name"]), flat=int(r["flat"]), bit=int(r["bit"])))
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Bad row in '{source_key}': {r}") from e

    if topk > 0:
        specs = specs[: min(topk, len(specs))]
    return specs


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


def apply_flips(model: onnx.ModelProto, specs: Sequence[FlipSpec], strict: bool) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    init_by_name = {t.name: t for t in model.graph.initializer}

    applied = 0
    skipped = 0
    non_finite = 0
    records: List[Dict[str, Any]] = []

    for i, s in enumerate(specs):
        t = init_by_name.get(s.name)
        if t is None:
            msg = f"[{i}] missing initializer: {s.name}"
            if strict:
                raise KeyError(msg)
            print(f"[Skip] {msg}")
            skipped += 1
            continue

        arr = numpy_helper.to_array(t).copy()
        if s.flat < 0 or s.flat >= arr.size:
            msg = f"[{i}] flat index out of range: {s.name} flat={s.flat} numel={arr.size}"
            if strict:
                raise IndexError(msg)
            print(f"[Skip] {msg}")
            skipped += 1
            continue

        old, new = flip_scalar(arr, s.flat, s.bit)
        if not np.isfinite(new):
            non_finite += 1

        t.CopyFrom(numpy_helper.from_array(arr, name=t.name))
        applied += 1
        rec = {
            "idx": i,
            "name": s.name,
            "flat": s.flat,
            "bit": s.bit,
            "old": old,
            "new": new,
            "finite": bool(np.isfinite(new)),
        }
        records.append(rec)

    return applied, skipped, non_finite, records


def main() -> None:
    args = parse_args()
    specs = load_specs(args.plan_json, args.source_key, args.topk)

    model = onnx.load(args.onnx_in)
    applied, skipped, non_finite, records = apply_flips(model, specs, strict=args.strict)

    os.makedirs(os.path.dirname(args.onnx_out) or ".", exist_ok=True)
    onnx.save(model, args.onnx_out)

    print(f"[Done] input={args.onnx_in}")
    print(f"[Done] output={args.onnx_out}")
    print(f"[Done] requested={len(specs)} applied={applied} skipped={skipped} non_finite_new={non_finite}")

    preview = records[: min(10, len(records))]
    if preview:
        print("[Preview] first flips:")
        for r in preview:
            print(
                f"  idx={r['idx']} name={r['name']} flat={r['flat']} bit={r['bit']} "
                f"old={r['old']:.8g} new={r['new']:.8g} finite={r['finite']}"
            )


if __name__ == "__main__":
    main()
