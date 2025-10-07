#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bit-flip search -> JSON plan only (no ONNX export).

Workflow:
  1) Accumulate importance |dL/dw| on proxy training batches.
  2) Select Top-W weights (default 50).
  3) For each candidate bit in those weights:
       flip once -> eval loss on cached val batches -> revert
     Rank by Δloss and return Top-B bits (default 50).
  4) Save a JSON file with:
       - meta (settings, timestamp)
       - ranked (top-B with dloss/old/new)
       - plan   (just {name, flat, bit} for your flipper script)

Assumes:
  - from openpilot_torch import OpenPilotModel
  - from data import Comma2k19SequenceDataset
Adjust imports if paths differ.
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import json, torch
from datetime import datetime
import onnxruntime as ort
# --------- replace if your module paths differ ----------
from openpilot_torch import OpenPilotModel, load_weights_from_onnx
from data import Comma2k19SequenceDataset

# ---------------- Loss pieces ----------------
distance_func = nn.CosineSimilarity(dim=2)
cls_loss = nn.CrossEntropyLoss()
reg_loss = nn.SmoothL1Loss(reduction='none')

def make_supercombo_inputs(batch, device):
    """
    batch:
      - 'seq_input_img': (B,T,C,H,W) (C=6 or 12)
      - 'seq_future_poses': (B,T,33,3)
    returns:
      imgs12 (B,12,H,W), desire (B,8), traffic (B,2), h0 (B,512), traj_gt (B,33,3)
    """
    seq_imgs   = batch['seq_input_img'].to(device, non_blocking=True)
    seq_labels = batch['seq_future_poses'].to(device, non_blocking=True)
    B, T, C, H, W = seq_imgs.shape
    if C == 6:
        seq_imgs = torch.cat([seq_imgs, seq_imgs], dim=2)  # -> (B,T,12,H,W)
    imgs12  = seq_imgs[:, -1]                              # (B,12,H,W)
    desire  = torch.zeros((B, 8),  device=device)
    traffic = torch.tensor([[1., 0.]], device=device).repeat(B, 1)
    h0      = torch.zeros((B, 512), device=device)
    traj_gt = seq_labels[:, -1, :, :]                      # (B,33,3)
    return imgs12, desire, traffic, h0, traj_gt

@torch.no_grad()
def plan_loss_on_batch(model, pack, use_amp=True) -> float:
    imgs12, desire, traffic, h0, gt = pack
    with torch.cuda.amp.autocast(enabled=use_amp):
        out  = model(imgs12, desire, traffic, h0)       # (B, N)
        B    = out.shape[0]
        plan = out[:, :5 * 991].view(B, 5, 991)
        pred_cls = plan[:, :, -1]                       # (B,5)
        params_flat = plan[:, :, :-1]                   # (B,5,990)
        pred_traj = params_flat.view(B, 5, 2, 33, 15)[:, :, 0, :, :3]  # mean xyz

        pred_end = pred_traj[:, :, 32, :]                              # (B,5,3)
        gt_end   = gt[:, 32:33, :].expand(-1, 5, -1)                   # (B,5,3)
        distances = 1 - distance_func(pred_end, gt_end)                # (B,5)
        index = distances.argmin(dim=1)                                # (B,)

        gt_cls = index
        row_idx = torch.arange(len(gt_cls), device=gt_cls.device)
        best_traj = pred_traj[row_idx, gt_cls, :, :]                   # (B,33,3)

        cls_loss_ = cls_loss(pred_cls, gt_cls)
        reg_loss_ = reg_loss(best_traj, gt).mean(dim=(0,1))
        loss = cls_loss_ + reg_loss_.mean()
    return float(loss.item())

@torch.no_grad()
def eval_loss_cached(model, cached_batches, use_amp=True) -> float:
    model.eval()
    vals = [plan_loss_on_batch(model, cb, use_amp) for cb in cached_batches]
    return float(np.mean(vals)) if vals else float('nan')

# --------------- Importance accumulation ---------------
def score_tensor(p: torch.Tensor, mode: str) -> torch.Tensor:
    with torch.no_grad():
        if mode == 'w':       return p.detach().abs()
        if p.grad is None:    return torch.zeros_like(p, dtype=torch.float32)
        if mode == 'grad':    return p.grad.detach().abs()
        if mode in ('taylor1','gradxw'): return (p.grad.detach() * p.detach()).abs()
        if mode == 'fisher':  return (p.grad.detach() ** 2)
        raise ValueError(f"unknown mode: {mode}")

def accumulate_importance(model, loader, device, num_batches=4, mode='gradxw', use_amp=True) -> Dict[str, torch.Tensor]:
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model.train()
    it = iter(loader)
    out: Dict[str, torch.Tensor] = {}
    processed = 0
    for _ in range(num_batches):
        try:
            b = next(it)
        except StopIteration:
            break
        imgs12, desire, traffic, h0, gt = make_supercombo_inputs(b, device)
        model.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            y   = model(imgs12, desire, traffic, h0)
            B   = y.shape[0]
            pl  = y[:, :5*991].view(B, 5, 991)
            cls = pl[:, :, -1]
            pf  = pl[:, :, :-1]
            traj= pf.view(B,5,2,33,15)[:, :, 0, :, :3]
            with torch.no_grad():
                pend = traj[:, :, 32, :]
                gend = gt[:, 32:33, :].expand(-1,5,-1)
                d    = 1 - distance_func(pend, gend)
                idx  = d.argmin(dim=1)
            gt_cls = idx
            row = torch.arange(len(gt_cls), device=gt_cls.device)
            best_traj = traj[row, gt_cls, :, :]
            loss = cls_loss(cls, gt_cls) + reg_loss(best_traj, gt).mean(dim=(0,1)).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(torch.optim.SGD(model.parameters(), lr=0.0))
        scaler.update()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and p.dtype == torch.float32:
                    s = score_tensor(p, mode)
                    out[n] = s.clone() if n not in out else out[n].add_(s)
        processed += 1
    if processed:
        for k in out:
            out[k].div_(processed)
    return out

# --------------- Utilities for mapping & flipping ---------------
def unravel(fi: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(int(i) for i in np.unravel_index(int(fi), shape, order='C'))

def live_params(model) -> Dict[str, torch.Tensor]:
    return dict(model.named_parameters())

@torch.no_grad()
def flip_scalar_bit_(param: torch.Tensor, flat_idx: int, bit: int) -> Tuple[float, float]:
    """
    Toggle one bit in-place and return (old, new). Revert internally if new is non-finite.
    Uses torch.int32 XOR (no Python-int overflow).
    """
    assert param.dtype == torch.float32
    assert 0 <= int(bit) <= 31
    cpu = param.detach().cpu().contiguous()
    f = cpu.view(torch.float32).view(-1)
    i = cpu.view(torch.int32).view(-1)
    flat_idx = int(flat_idx)
    old = float(f[flat_idx].item())
    mask = (torch.ones((), dtype=torch.int32) << int(bit))
    i[flat_idx] = torch.bitwise_xor(i[flat_idx], mask)
    new = float(f[flat_idx].item())
    if not math.isfinite(new):
        i[flat_idx] = torch.bitwise_xor(i[flat_idx], mask)
        param.copy_(cpu.to(param.device))
        return old, old
    param.copy_(cpu.to(param.device))
    return old, new

@torch.no_grad()
def revert_scalar_bit_(param: torch.Tensor, flat_idx: int, bit: int) -> None:
    """Toggle again to restore original value (also torch.int32 XOR)."""
    assert param.dtype == torch.float32
    assert 0 <= int(bit) <= 31
    cpu = param.detach().cpu().contiguous()
    i = cpu.view(torch.int32).view(-1)
    flat_idx = int(flat_idx)
    mask = (torch.ones((), dtype=torch.int32) << int(bit))
    i[flat_idx] = torch.bitwise_xor(i[flat_idx], mask)
    param.copy_(cpu.to(param.device))

# Bit sets
MANTISSA      = list(range(0, 23))
MANTISSA_LOW  = list(range(0, 7))
EXPONENT      = list(range(23, 31))
SIGN          = [31]

def parse_bitset(s: str) -> List[int]:
    s = s.strip().lower()
    if s == 'mantissa':      return MANTISSA
    if s == 'mantissa_low':  return MANTISSA_LOW
    if s == 'exponent':      return EXPONENT
    if s == 'sign':          return SIGN
    if s in ('full','all'):  return list(range(32))
    bits = [int(x) for x in s.split(',') if x.strip()!='']
    for b in bits:
        if b < 0 or b > 31: raise ValueError(f"bit {b} out of range")
    return bits

# --------------- Data helpers ---------------
def build_loaders(data_root, train_idx, val_idx, batch_size, device):
    tr = Comma2k19SequenceDataset(train_idx, data_root, 'train', use_memcache=False)
    va = Comma2k19SequenceDataset(val_idx,   data_root, 'val',   use_memcache=False)
    tr_loader = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                            num_workers=0, pin_memory=(device.type=='cuda'))
    va_loader = torch.utils.data.DataLoader(va, batch_size=1, shuffle=False,
                                            num_workers=0, pin_memory=(device.type=='cuda'))
    return tr_loader, va_loader

@torch.no_grad()
def cache_eval_batches(val_loader, device, num_batches=3):
    cached = []
    it = iter(val_loader)
    for _ in range(num_batches):
        try:
            b = next(it)
        except StopIteration:
            break
        imgs12, desire, traffic, h0, gt = make_supercombo_inputs(b, device)
        cached.append((imgs12, desire, traffic, h0, gt))
    return cached

# --------------- Selection ---------------
def select_topW_weights(importance: Dict[str, torch.Tensor],
                        W: int,
                        allow_bias: bool,
                        restrict: Optional[List[str]]) -> List[Tuple[str, int, float]]:
    """
    Returns list of (param_name, local_flat_idx, score) sorted descending.
    """
    items: List[Tuple[str,int,float]] = []
    for n, s in importance.items():
        if s.dtype != torch.float32: continue
        if not allow_bias and n.endswith(".bias"): continue
        if restrict and not any(tag in n for tag in restrict): continue
        flat = s.reshape(-1).cpu()
        if flat.numel() == 0: continue
        val, idx = torch.max(flat, dim=0)
        items.append((n, int(idx), float(val)))
    items.sort(key=lambda x: x[2], reverse=True)
    return items[:W]

@torch.no_grad()
def rank_bits_independent(model,
                          cached_batches,
                          candidates: List[Tuple[str, int]],
                          bitset: List[int],
                          use_amp: bool = True,
                          max_iters: Optional[int] = None,
                          per_weight_once: bool = True,
                          value_guard: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    PROGRESSIVE bit selection (one bit per iteration), with two fixes:
      - per_weight_once: once we flip a scalar at (name, flat_idx), remove *all* other bits of that scalar.
      - skip non-finite Δloss: do not break the loop if the best candidate is inf/NaN; just ignore it.

    value_guard (optional): if set (e.g. 1e6), skip a flip that makes |new| > value_guard.
    """
    records: List[Dict[str, Any]] = []
    P = live_params(model)

    # Expand candidate space: (name, flat_idx, bit)
    pending: List[Tuple[str, int, int]] = [(n, fi, b) for (n, fi) in candidates for b in bitset]
    if not pending:
        return records

    base = eval_loss_cached(model, cached_batches, use_amp)
    total_iters = len(pending) if max_iters is None else min(max_iters, len(pending))
    # Optionally keep track of which scalar locations were used already
    used_keys: set[Tuple[str, int]] = set()

    for it in range(total_iters):
        best_pos = -1
        best_delta = -math.inf
        best_triplet: Optional[Tuple[str, int, int]] = None
        best_old_new: Optional[Tuple[float, float]] = None

        # Evaluate each remaining candidate vs CURRENT baseline
        for pos, (name, fi, bit) in enumerate(pending):
            # optional: enforce one flip per scalar
            if per_weight_once and (name, fi) in used_keys:
                continue

            t = P[name]
            old, new = flip_scalar_bit_(t, fi, bit)

            # guard bad flips
            bad_flip = (new == old) or (value_guard is not None and abs(new) > float(value_guard))
            if bad_flip:
                # revert if we actually changed (we didn't if new==old)
                if new != old:
                    revert_scalar_bit_(t, fi, bit)
                continue

            loss = eval_loss_cached(model, cached_batches, use_amp)
            revert_scalar_bit_(t, fi, bit)
            cand_delta = loss - base

            # **skip** non-finite deltas instead of letting them win and halting search
            if not math.isfinite(cand_delta):
                continue

            if cand_delta > best_delta:
                best_delta = cand_delta
                best_pos = pos
                best_triplet = (name, fi, bit)
                best_old_new = (old, new)

        # No valid candidate this round → stop
        if best_triplet is None or best_pos < 0:
            break

        # Permanently apply the winner and update baseline
        name, fi, bit = best_triplet
        t = P[name]
        flip_scalar_bit_(t, fi, bit)  # keep
        base += best_delta

        records.append({
            "name": name,
            "index_flat": int(fi),
            "bit": int(bit),
            "old": float(best_old_new[0]),
            "new": float(best_old_new[1]),
            "dloss": float(best_delta)
        })

        # Remove candidates to avoid reusing this scalar
        if per_weight_once:
            used_keys.add((name, fi))
            pending = [c for c in pending if not (c[0] == name and c[1] == fi)]
        else:
            # only remove the exact (name, fi, bit)
            pending.pop(best_pos)

        # Stop when we have collected max_iters bits
        if max_iters is not None and len(records) >= max_iters:
            break

    return records


# --------------- JSON export ---------------
def timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def save_plan_json(records: List[Dict[str,Any]],
                   path: str,
                   topB: int,
                   meta: Dict[str,Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    top = records[:topB]
    payload = {
        "meta": {**meta, "timestamp": timestamp(), "num_candidates": len(records), "num_top": len(top)},
        "flips": top,
        "plan": [{"name": r["name"], "index_flat": r["index_flat"], "bit": r["bit"]} for r in top]
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Save] plan -> {path}")

# --------------- CLI / Main ---------------
def parse_args():
    ap = argparse.ArgumentParser("Bit-flip plan only (no ONNX export)")
    ap.add_argument("--data-root", default="data/comma2k19/")
    ap.add_argument("--train-index", default="data/comma2k19_train_non_overlap.txt")
    ap.add_argument("--val-index",   default="data/comma2k19_val_non_overlap.txt")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--ckpt", type=str, default="openpilot_model/supercombo_torch_weights.pth")

    ap.add_argument("--imp-batches", type=int, default=6, help="batches for importance accumulation")
    ap.add_argument("--eval-batches", type=int, default=3, help="cached val batches for loss eval")
    ap.add_argument("--imp-mode", choices=["w","grad","gradxw","taylor1","fisher"], default="gradxw")

    ap.add_argument("--topW", type=int, default=20, help="number of important weights to consider")
    ap.add_argument("--topB", type=int, default=20, help="number of best bits to return")
    ap.add_argument("--bitset", type=str, default="full",
                    help='mantissa_low|mantissa|exponent|sign|full or comma list "0,1,2,3"')
    ap.add_argument("--restrict", type=str, default="", help='comma substrings to filter params (e.g., "vision_net,plan_head")')
    ap.add_argument("--allow-bias", type=int, default=1)
    ap.add_argument('--save_dir', type=str, default='flipped_models')
    ap.add_argument('--save_prefix', type=str, default='model')
    ap.add_argument("--amp", action="store_true")
    return ap.parse_args()

def timestamp_id() -> str:
    # e.g., "20250927-130542"
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    tr_loader = torch.utils.data.DataLoader(
        Comma2k19SequenceDataset(args.train_index, args.data_root, 'train', use_memcache=False),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type=='cuda')
    )
    va_loader = torch.utils.data.DataLoader(
        Comma2k19SequenceDataset(args.val_index,   args.data_root, 'val',   use_memcache=False),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=(device.type=='cuda')
    )
    cached = cache_eval_batches(va_loader, device, num_batches=args.eval_batches)

    # model + weights
    model = OpenPilotModel().to(device)
    if os.path.isfile(args.ckpt):
        sd = torch.load(args.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError:
            from collections import OrderedDict
            new_sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
            model.load_state_dict(new_sd, strict=False)
    model.eval()

    # 1) Importance
    print(f"[Importance] mode={args.imp_mode}, batches={args.imp_batches}")
    imp = accumulate_importance(model, tr_loader, device, num_batches=args.imp_batches,
                                mode=args.imp_mode, use_amp=args.amp)

    # 2) Top-W weights
    restrict = [s.strip() for s in args.restrict.split(",") if s.strip()] or None
    topW = select_topW_weights(imp, args.topW, allow_bias=bool(args.allow_bias), restrict=restrict)
    if not topW:
        print("No important weights found.")
        return
    print(f"[Select] Top {len(topW)} weights ready.")

    # 3) Rank bits independently (no permanent flips here)
    candidates = [(name, fi) for (name, fi, _) in topW]
    bitset = parse_bitset(args.bitset)
    print(f"[Bitset] {bitset}  | candidates={len(candidates)}")
    records = rank_bits_independent(model, cached, candidates, bitset, use_amp=args.amp)

    if not records:
        print("No viable bit candidates produced.")
        return

    # 4) Save plan JSON only
    meta = {
        "ckpt": args.ckpt,
        "imp_mode": args.imp_mode,
        "imp_batches": args.imp_batches,
        "eval_batches": args.eval_batches,
        "topW": args.topW,
        "topB": args.topB,
        "bitset": args.bitset,
        "restrict": args.restrict,
        "allow_bias": bool(args.allow_bias),
    }
    # build base name (seconds only) & collision guard
    ts = timestamp_id()  # e.g., "20250927-130542"
    base = f"{args.save_prefix}_{ts}"
    json_path = os.path.join(args.save_dir, base + ".json")
    suffix = 0
    while os.path.exists(json_path):
        suffix += 1
        json_path = os.path.join(args.save_dir, f"{base}-{suffix:03d}.json")
    save_plan_json(records, json_path, args.topB, meta)
    return json_path

def flip_bit(t: torch.Tensor, index_tuple, bit_idx=0):
    # 取出元素
    val = t[index_tuple].item()
    # 转成 int32 raw bits
    int_view = torch.tensor([val], dtype=torch.float32).view(torch.int32)
    # 翻转
    mask = 1 << bit_idx
    int_flipped = int_view ^ mask
    # 转回 float
    flipped_val = int_flipped.view(torch.float32).item()
    return flipped_val

if __name__ == "__main__":
    json_path = main()

    onnx_path = 'openpilot_model/supercombo.onnx'
    # Load the bit-flip record
    with open(json_path, "r") as f:
        flips = json.load(f)["flips"]
    session = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']  # 如需 GPU: ['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    inputs_meta = session.get_inputs()
    outputs_meta = session.get_outputs()

    # 4) 加载 PyTorch 模型 & 从 ONNX 权重迁移
    model = OpenPilotModel()
    load_weights_from_onnx(model, onnx_path)
    model.eval()
    model.cpu()

    # 5) 保存权重
    torch.save(model.state_dict(), 'flipped_models/supercombo_torch_weights.pth')

    # 6）导出 onnx

    ## 准备 dummy 输入（以 openpilot supercombo 为例）
    # 用 torch 张量作为 dummy 输入
    imgs = torch.zeros(1, 12, 128, 256, dtype=torch.float32)  # input_imgs
    desire = torch.zeros(1, 8, dtype=torch.float32)  # desire
    tc = torch.tensor([[1., 0.]], dtype=torch.float32)  # traffic_convention
    state = torch.zeros(1, 512, dtype=torch.float32)  # initial_state

    # bitflip
    sd = model.state_dict()
    for flip in flips:
        name = flip["name"]
        bit = flip["bit"]
        flat_idx = flip["index_flat"]

        param = sd[name]
        was_cuda = param.is_cuda
        p_cpu = param.detach().cpu().contiguous()
        fview = p_cpu.view(torch.float32).view(-1)
        iview = p_cpu.view(torch.int32).view(-1)

        # Flip the bit
        mask = torch.tensor(1 << bit, dtype=torch.int32)
        iview[flat_idx] = iview[flat_idx] ^ mask

        if was_cuda:
            sd[name].copy_(p_cpu.to(param.device))
        else:
            sd[name].copy_(p_cpu)

    model.load_state_dict(sd)

    # create timestamp string like "20251006-094604"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    out_dir = "flipped_models"
    os.makedirs(out_dir, exist_ok=True)

    # export to ONNX with timestamped filename
    onnx_path = os.path.join(out_dir, f"model_{ts}.onnx")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (imgs, desire, tc, state),
            onnx_path,
            input_names=["input_imgs", "desire", "traffic_convention", "initial_state"],
            output_names=["outputs"],
            do_constant_folding=False,
            opset_version=17,
            training=torch.onnx.TrainingMode.EVAL
        )

    print(f"saved: {onnx_path}")
