#!/usr/bin/env python3
"""Re-export selected flipped op_089 models to ONNX with a target IR/opset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import onnx
import torch

from recurrent_bit_utils import flip_scalar_bit_fast_, live_params, load_flip_rows_json, load_model


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser("Re-export selected flipped op_089 ONNX models")
    ap.add_argument("--selected-dir", default=str(root / "op_089" / "selected"))
    ap.add_argument("--out-dir", default=str(root / "op_089" / "selected_ir6_opset9"))
    ap.add_argument(
        "--ckpt",
        default="/home/zx/Projects/openpilot_0.8.9/openpilot/flipped_models/supercombo_torch_weights.pth",
    )
    ap.add_argument("--flip-count", type=int, default=2)
    ap.add_argument("--opset", type=int, default=9)
    ap.add_argument("--ir-version", "--ir_version", dest="ir_version", type=int, default=6)
    return ap.parse_args()


def pair_selected_files(selected_dir: Path) -> List[Path]:
    jsons = sorted(selected_dir.glob("flipped_bits*.json"))
    if not jsons:
        raise RuntimeError(f"No selected json files found in {selected_dir}")
    return jsons


def infer_onnx_name(json_path: Path) -> str:
    stem = json_path.stem
    if stem == "flipped_bits_left_steering_9":
        return "supercombo_2_left_steering_9.onnx"
    if stem == "flipped_bits_lead_prob_20251114-17:33:26":
        return "supercombo_2_lead_prob_20251114-17:33:26.onnx"
    if stem == "flipped_bits_20251114-21:04:09":
        return "supercombo_2_speed_up15_20251114-21:04:09.onnx"
    suffix = stem.removeprefix("flipped_bits_")
    return f"supercombo_2_{suffix}.onnx"


def export_model(model: torch.nn.Module, out_path: Path, opset: int, ir_version: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to(torch.device("cpu"))
    model_cpu.eval()

    imgs = torch.randn(1, 12, 128, 256, dtype=torch.float32)
    desire = torch.zeros(1, 8, dtype=torch.float32)
    traffic = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    h0 = torch.zeros(1, 512, dtype=torch.float32)

    torch.onnx.export(
        model_cpu,
        (imgs, desire, traffic, h0),
        str(out_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=False,
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input_imgs", "desire", "traffic_convention", "initial_state"],
        output_names=["outputs"],
        dynamo=False,
    )
    onnx_model = onnx.load(str(out_path))
    onnx_model.ir_version = int(ir_version)
    onnx.save(onnx_model, str(out_path))


def apply_flips(model: torch.nn.Module, flips: List[Dict[str, int]]) -> None:
    param_map = live_params(model)
    for flip in flips:
        name = str(flip["name"])
        flat_idx = int(flip["index_flat"])
        bit = int(flip["bit"])
        if name not in param_map:
            raise KeyError(f"Parameter not found for flip: {name}")
        flip_scalar_bit_fast_(param_map[name], flat_idx, bit)


def main() -> None:
    args = parse_args()
    selected_dir = Path(args.selected_dir)
    out_dir = Path(args.out_dir)
    json_paths = pair_selected_files(selected_dir)
    print(f"[Pairs] {len(json_paths)} selected json files")

    for json_path in json_paths:
        flips = load_flip_rows_json(str(json_path), limit=args.flip_count)
        model = load_model(args.ckpt, torch.device("cpu"))
        apply_flips(model, flips)

        onnx_name = infer_onnx_name(json_path)
        out_onnx = out_dir / onnx_name
        export_model(model, out_onnx, opset=args.opset, ir_version=args.ir_version)

        payload = {
            "meta": {
                "source_json": str(json_path),
                "source_ckpt": args.ckpt,
                "flip_count": len(flips),
                "opset": int(args.opset),
                "ir_version": int(args.ir_version),
            },
            "flips": flips,
        }
        out_json = out_dir / json_path.name
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[Save] {out_onnx}")
        print(f"[Save] {out_json}")


if __name__ == "__main__":
    main()
