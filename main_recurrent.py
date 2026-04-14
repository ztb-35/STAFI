import os
import sys
import json
import math
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import json, torch
from datetime import datetime
from tqdm import tqdm, trange
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------- replace if your module paths differ ----------
from openpilot_torch import OpenPilotModel, load_weights_from_onnx
from data import Comma2k19SequenceRecurrentDataset
import glob
from collections import Counter, defaultdict

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
    seq_imgs = batch['seq_input_img'].to(device, non_blocking=True)
    seq_labels = batch['seq_future_poses'].to(device, non_blocking=True)
    B, T, C, H, W = seq_imgs.shape#T is the leangth of the frame stream
    if C == 6:
        seq_imgs = torch.cat([seq_imgs, seq_imgs], dim=2)  # -> (B,T,12,H,W)
    imgs12 = seq_imgs  # (B,T, 12,H,W)
    desire = torch.zeros((B, 8), device=device)
    traffic = torch.tensor([[1., 0.]], device=device).repeat(B, 1)
    h0 = torch.zeros((B, 512), device=device)
    traj_gt = seq_labels  # (B,T,33,3)
    return imgs12, desire, traffic, h0, traj_gt


import torch.multiprocessing as mp

try:
    mp.set_start_method("fork", force=True)  # or "spawn" on some clusters
except RuntimeError:
    pass
mp.set_sharing_strategy("file_system")



# --------------- Importance accumulation ---------------
def score_tensor(p: torch.Tensor, mode: str) -> torch.Tensor:
    with torch.no_grad():
        if mode == 'w':       return p.detach().abs()
        if p.grad is None:    return torch.zeros_like(p, dtype=torch.float32)
        if mode == 'grad':    return p.grad.detach().abs()
        if mode in ('taylor1', 'gradxw'): return (p.grad.detach() * p.detach()).abs()
        if mode == 'fisher':  return (p.grad.detach() ** 2)
        raise ValueError(f"unknown mode: {mode}")


# --------------- Utilities for mapping & flipping ---------------
def unravel(fi: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(int(i) for i in np.unravel_index(int(fi), shape, order='C'))


def live_params(model) -> Dict[str, torch.Tensor]:
    return dict(model.named_parameters())


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
MANTISSA = list(range(0, 23))
MANTISSA_LOW = list(range(0, 7))
EXPONENT = list(range(23, 31))
EXPONENT_SIGN = list(range(23, 32))
SIGN = [31]


def parse_bitset(s: str) -> List[int]:
    s = s.strip().lower()
    if s == 'mantissa':        return MANTISSA
    if s == 'mantissa_low':    return MANTISSA_LOW
    if s == 'exponent':        return EXPONENT
    if s == 'exponent&sign':        return EXPONENT_SIGN
    if s == 'sign':            return SIGN
    if s in ('full', 'all'):    return list(range(16, 32))
    bits = [int(x) for x in s.split(',') if x.strip() != '']
    for b in bits:
        if b < 0 or b > 31: raise ValueError(f"bit {b} out of range")
    return bits


@torch.no_grad()
def flip_scalar_bit_fast_(param: torch.Tensor, flat_idx: int, bit: int) -> Tuple[float, float]:
    """
    Toggle one bit of ONE scalar (float32) directly on device without full tensor copies.
    Returns (old, new). Call again to revert (XOR twice).
    """
    assert param.dtype == torch.float32
    assert 0 <= bit <= 31

    flat = param.data.view(-1)  # use .data to avoid autograd bookkeeping
    # Pull value to cpu as float32
    old_val = float(flat[flat_idx].detach().cpu().numpy().astype(np.float32).item())

    # reinterpret bits via numpy uint32 view
    old_uint = np.frombuffer(np.float32(old_val).tobytes(), dtype=np.uint32)[0]
    new_uint = old_uint ^ (np.uint32(1) << np.uint32(bit))
    new_val = np.frombuffer(np.uint32(new_uint).tobytes(), dtype=np.float32)[0]

    # write back (create scalar tensor on same device)
    new_tensor = torch.tensor(new_val, dtype=torch.float32, device=param.device)
    flat[flat_idx] = new_tensor

    return old_val, float(new_val)


def float_to_bits32(x: float) -> str:
    import numpy as np
    ui = np.frombuffer(np.float32(x).tobytes(), dtype=np.uint32)[0]
    return f"{ui:032b}"


def tensor_scalar_bits(param: torch.Tensor, flat_idx: int) -> str:
    v = float(param.data.view(-1)[flat_idx].detach().cpu().numpy().astype(np.float32).item())
    return float_to_bits32(v)


@torch.no_grad()
def progressive_bit_search(model,
                           val_loader,
                           candidates,
                           bitset,
                           device,
                           iters: int,
                           use_amp: bool = True,
                           max_bits_per_scalar: int = 2,  # <-- NEW
                           per_bit_once: bool = True,
                           value_guard: Optional[float] = None,
                           max_cached_batches: int = 2,
                           progress: bool = True):
    P = live_params(model)
    model.eval()

    pending = [(n, fi, b) for (n, fi) in candidates for b in bitset]
    used_bits = set()  # (name, fi, bit)
    committed_counts = {}  # (name, fi) -> count
    committed = []

    def process_batch(b_idx, batch, model, device, use_amp, distance_func):
        imgs12_, desire, traffic, h0, gt_ = make_supercombo_inputs(batch, device)
        for count in range(args.frame_stream_length):

            imgs12 = imgs12_[:,count,:,:,:]
            gt = gt_[:,count,:,:]
            with torch.amp.autocast('cuda', enabled=use_amp):
                y_clean = model(imgs12, desire, traffic, h0)

            B = y_clean.shape[0]
            h0 = y_clean[:,-512:]
            pl = y_clean[:, :5 * 991].view(B, 5, 991)
            prob = pl[:,:,-1]#probability of output 5 traj
            pf = pl[:, :, :-1]
            lead_p = y_clean[:,6064]
            traj = pf.view(B, 5, 2, 33, 15)[:, :, 0, :, :3]
            velocity = pf.view(B, 5, 2, 33, 15)[:, :, 1, :, 3:6]
            with torch.no_grad():
                if count>20:
                    print("")
                pend = traj[:, :, 17, 0]  # forward distance only
                gend = gt[:, 17, 0].unsqueeze(1).expand(-1, 5)
                d = ((pend - gend) / (abs(gend) + (1)))
                idx = d.argmin(dim=1)
            rows = torch.arange(len(idx), device=idx.device)
            best_traj = traj[rows, idx, :, :].detach().cpu()
            best_velocity = velocity[rows, idx, :, :].detach().cpu()


        return {
            "imgs12": imgs12.detach().cpu(),
            "desire": desire.detach().cpu(),
            "traffic": traffic.detach().cpu(),
            "h0": h0.detach().cpu(),
            "gt": gt.detach().cpu(),
            "baseline_best_traj": best_traj,
            "lead_p": lead_p.detach().cpu(),
            "baseline_best_velocity": best_velocity,
        }

    def make_cache_current_baseline(val_loader, model, device, distance_func, use_amp=True, max_cached_batches=max_cached_batches,
                                    num_workers=1):
        print("🚀 Begin to make baseline cache")
        cache = []

        # 用 trange 显示进度
        batches = []
        for b_idx, batch in enumerate(val_loader):
            if b_idx >= max_cached_batches:
                break
            batches.append((b_idx, batch))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_batch, b_idx, batch, model, device, use_amp, distance_func)
                       for b_idx, batch in batches]

            for f in tqdm(as_completed(futures), total=len(futures), desc="Caching baseline"):
                cache.append(f.result())

        print(f"✅ Done. Cached {len(cache)} batches.")
        return cache

    def _mad(x: torch.Tensor, eps=1e-6):
        # x: any shape; returns scalar MAD
        med = x.median()
        return (x - med).abs().median().clamp_min(eps)

    def score_one_bit(name, fi, bit, cache):
        if name not in P:
            return None
        t = P[name]
        # print(f"OLD {t.data.view(-1)[fi].detach().cpu().numpy().astype(np.float32).item()}")
        old, new = flip_scalar_bit_fast_(t, fi, bit)

        total = 0.0
        cnt = 0
        
        for cm in cache:
            imgs12 = cm["imgs12"].to(device, non_blocking=True)
            desire = cm["desire"].to(device, non_blocking=True)
            traffic = cm["traffic"].to(device, non_blocking=True)
            h0 = cm["h0"].to(device, non_blocking=True)
            gt = cm["gt"].to(device, non_blocking=True)
            base_traj = cm["baseline_best_traj"].to(device, non_blocking=True)
            base_vel = cm["baseline_best_velocity"].to(device, non_blocking=True)
            base_lead_p = cm["lead_p"].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                y = model(imgs12, desire, traffic, h0)

            B = y.shape[0]
            pl = y[:, :5 * 991].view(B, 5, 991)
            pf = pl[:, :, :-1]
            lead_p = y[:,6064]
            traj = pf.view(B, 5, 2, 33, 15)[:, :, 0, :, :3]
            velocity = pf.view(B, 5, 2, 33, 15)[:, :, 1, :, 3:6]
            with torch.no_grad():
                pend = traj[:, :, 17, 0]  # forward distance only
                gend = gt[:, 17, 0].unsqueeze(1).expand(-1, 5)
                d = ((pend - gend) / (abs(gend) + (1)))
                idx = d.argmin(dim=1)#select the best predicted trajectory
            rows = torch.arange(len(idx), device=idx.device)
            flipped_traj = traj[rows, idx, :, :]
            flipped_vel = velocity[rows, idx, :, :]
            if args.metric == 'speed up':
                #delta = flipped_traj[:, 17, 0] - base_traj[:, 17, 0]#enlarge forward distance(speed up)
                delta = (flipped_traj[:, 10, 0] - base_traj[:, 10, 0]) - 2*(flipped_traj[:, 10, 1] - base_traj[:, 10, 1]).abs()
            elif args.metric == 'speed up 2':
                delta = base_lead_p - lead_p
            elif args.metric == 'velocity':
                delta = base_vel[:, 10, 0] - flipped_vel[:, 10, 0]#decrease predicted ego car velocity(speed up)
            if args.metric == 'slowing down':
                delta = base_traj[:, 10, 0] - flipped_traj[:, 10, 0]#enlarge forward distance(speed up)
            elif args.metric == 'left steering':
                delta = (flipped_traj[:, 10, 1] - base_traj[:, 10, 1])
            elif args.metric == 'right steering':
                delta = (base_traj[:, 10, 1] - flipped_traj[:, 10, 1])
            elif args.metric == 'speed up & left steering':
                delta = (flipped_traj[:, 10, 0] - base_traj[:, 10, 0])/9.94 + (flipped_traj[:, 10, 1] - base_traj[:, 10, 1]).mean()
            elif args.metric == 'speed up & right steering':
                delta = (flipped_traj[:, 10, 0] - base_traj[:, 10, 0])/9.94 + (base_traj[:, 10, 1] - flipped_traj[:, 10, 1]).mean()
            mask = torch.isfinite(delta)
            
            if mask.any():
                total += float(delta[mask].mean().item())
                cnt += 1

        flip_scalar_bit_fast_(t, fi, bit)  # revert trial
        if cnt == 0:
            return None

        return (total / cnt, float(old), float(new))

    time_list = []

    cache = make_cache_current_baseline(val_loader, model, device, distance_func)

    iter_bar = trange(iters, desc="Progressive bit search", disable=not progress)
    print("progressive bit search start time:", timestamp_id())

    for it in iter_bar:
        best = None
        any_viable = False

        print("cur time:", timestamp_id())
        for (name, fi, bit) in list(pending):
            # enforce per-bit uniqueness
            if per_bit_once and (name, fi, bit) in used_bits:
                continue
            # enforce at most K bits per scalar
            if committed_counts.get((name, fi), 0) >= max_bits_per_scalar:
                continue

            res = score_one_bit(name, fi, bit, cache)

            if res is None:
                continue
            any_viable = True
            score, old, new = res
            # print(f"CUR BIT {bit} score {score}")

            if (best is None) or (score > best[0]):
                if score > 0:
                    best = (score, name, fi, bit, old, new)
        print("cur_end time:", timestamp_id())

        if not any_viable or best is None:
            break

        score, name, fi, bit, old_eval, new_eval = best
        # # commit this bit
        t = P[name]
        old_c, new_c = flip_scalar_bit_fast_(t, fi, bit)

        committed.append({
            "iter": it,
            "name": name,
            "index_flat": int(fi),
            "bit": int(bit),
            "old": float(old_eval),
            "new": float(new_eval),
            "score": float(score),
        })

        used_bits.add((name, fi, bit))
        committed_counts[(name, fi)] = committed_counts.get((name, fi), 0) + 1

        # prune only this exact bit from pending; keep others until count hits the cap
        pending = [(n, f, b) for (n, f, b) in pending if not (n == name and f == fi and b == bit)]

        # if we’ve now reached the per-scalar cap, drop all remaining bits for this scalar
        if committed_counts[(name, fi)] >= max_bits_per_scalar:
            pending = [(n, f, b) for (n, f, b) in pending if not (n == name and f == fi)]

        if not pending:
            break
    print("progressive bit search end time:", timestamp_id())
    return committed


def timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def save_plan_json(records: List[Dict[str, Any]],
                   path: str,
                   topB: int,
                   meta: Dict[str, Any]):
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


def load_topW_json(path: str, topn: int = 10) -> List[Tuple[str, int]]:
    with open(path, "r") as f:
        payload = json.load(f)
    recs = payload.get("summary", [])
    recs = recs[:topn]
    return [(r["name"], int(r["index_flat"])) for r in recs]


# --------------- CLI / Main ---------------
def parse_args():
    ap = argparse.ArgumentParser("Bit-flip plan only (no ONNX export)")
    ap.add_argument("--data-root", default="data/comma2k19/")
    ap.add_argument("--train-index", default="data/comma2k19_train_non_overlap.txt")
    ap.add_argument("--val-index", default="data/comma2k19_val_non_overlap.txt")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--ckpt", type=str, default="openpilot_model/supercombo_torch_weights.pth")
    ap.add_argument("--data_length", type=int, default=100, help="batches for importance accumulation")
    ap.add_argument("--topW", type=int, default=100, help="number of best weights to search bits")
    ap.add_argument("--topB", type=int, default=10, help="number of best bits to return")
    ap.add_argument("--bitset", type=str, default="full",
                    help='mantissa_low|mantissa|exponent|sign|full|exponent&sign')
    ap.add_argument("--restrict", type=str, default="",
                    help='comma substrings to filter params (e.g., "vision_net,plan_head")')
    ap.add_argument("--allow-bias", type=int, default=1)
    ap.add_argument('--save_dir', type=str, default='flipped_models')
    ap.add_argument('--save_prefix', type=str, default='flipped_bits')  ##flipped bits json file name prefix

    ap.add_argument("--topW-json", type=str, default="flipped_models/topW_100_20251031-131544_score_summary.json",
                    help="If provided, load Top-W (name,index_flat) from this JSON and skip importance.")
    ap.add_argument("--max_cached_batches", type=int, default=2, 
                    help="max_cached_batches for bit flip metrics computation")
    ap.add_argument("--frame_stream_length", type=int, default=8,
                    help="frame stream length in our recurrent pipeline, to initial supercombo input state get a more reliable output")
    ap.add_argument("--metric", type=str, default="right steering",
                help="metrics for different actions, options: speed up, velocity, slowing down, left steering, right steering, "
                     "speed up & left steering, speed up & right steering")
    ap.add_argument("--amp", action="store_true")
    return ap.parse_args()


def timestamp_id() -> str:
    # e.g., "20250927-130542"
    return datetime.now().strftime("%Y%m%d-%H:%M:%S")


def resolve_device() -> torch.device:
    """Prefer CUDA when supported by the current PyTorch build, otherwise CPU."""
    if not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        return torch.device("cpu")

    try:
        major, minor = torch.cuda.get_device_capability(0)
    except (AssertionError, RuntimeError) as exc:
        warnings.warn(f"Unable to query CUDA capability ({exc}); defaulting to CPU.")
        return torch.device("cpu")

    compiled_arches = []
    get_arch_list = getattr(torch.cuda, "get_arch_list", None)
    if callable(get_arch_list):
        compiled_arches = get_arch_list()

    device_arch = f"sm_{major}{minor}"
    if compiled_arches and device_arch not in compiled_arches:
        warnings.warn(
            "GPU compute capability "
            f"{device_arch} is not supported by this PyTorch build ({compiled_arches}); "
            "falling back to CPU."
        )
        return torch.device("cpu")

    return torch.device("cuda")


def main():
    print("clipped sequence length in comma2k19:", args.data_length)
    #device = resolve_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # data
    val_loader = torch.utils.data.DataLoader(
        Comma2k19SequenceRecurrentDataset(args.val_index,   args.data_root, args.data_length, args.frame_stream_length,
                                          'val',   use_memcache=False),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type=='cuda')
    )
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
    candidates = load_topW_json(args.topW_json, args.topW)

    bitset = parse_bitset(args.bitset)
    print(f"[Bitset] {bitset}  | candidates={len(candidates)}")
    records = progressive_bit_search(model, val_loader, candidates, bitset, device,
                                     iters=args.topB, max_cached_batches=args.max_cached_batches)

    if not records:
        print("No viable bit candidates produced.")
        return

    # 4) Save plan JSON only
    meta = {
        "ckpt": args.ckpt,
        "topW": args.topW,
        "topB": args.topB,
        "bitset": args.bitset,
        "metric": args.metric,
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
    # set a for loop, during iterations, find which bits appears most times
    args = parse_args()
    time_0 = time.time()
    import gc, torch
    print("start time:", timestamp_id())
    for dl in range(20,69):  # data_length from 200 to 800
        print("start time with data_length as", (dl+1)*10, timestamp_id())
        args.data_length = int(dl + 1) * 10
        outp = main()  # main() returns the JSON path
        # delete local references if you kept them anywhere (outp is small, but main may create big objects)
        # gc.collect()
        # torch.cuda.empty_cache()
        # print("cleanup done, mem allocated:", torch.cuda.memory_allocated(), "reserved:", torch.cuda.memory_reserved())
    print("end time:", timestamp_id())
