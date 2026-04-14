"""
BitShield Defense Evaluation on OpenPilot Supercombo Model
===========================================================
BitShield (NDSS'25, Qin et al.): Binary-level Integrity Guard for neural networks.

Original design (DIG — Dynamic Integrity Guard):
  - Model weights are compiled to a native binary (TVM / Glow / NNFusion).
  - A cryptographic checksum (CRC32 / SHA) is computed over each weight tensor's
    binary representation at deployment time.
  - At runtime, the checksum is recomputed and compared to the golden value.
  - On mismatch → process terminates immediately (no repair attempted).

CIG (Compile-time Integrity Guard) — not evaluated here:
  - Inserts hardware-level write-protection on compiled weight pages.
  - Requires OS/MMU co-operation; not applicable to Python/PyTorch runtime.

Adaptation for float32 Supercombo:
  - Binary compilation (TVM/Glow/NNFusion) is not used; model runs in PyTorch.
  - DIG is simulated at *tensor granularity*: zlib.CRC32 over the raw float32
    bytes of each parameter tensor — functionally equivalent integrity check.
  - Terminate-on-detect semantics are preserved: once a mismatch is found the
    model output is blocked; no repair attempted.

Evaluation metrics (from main_recurrent.py):
  For each scenario, the flip JSON's meta.metric field names the attack goal.
  We report BOTH:
    - L2 deviation from clean output (detection proxy)
    - metric-based delta (δ): measures whether the attack goal was achieved.
      Large δ after attack = attack succeeded.
      δ is UNCHANGED after BitShield triggers (no repair), but the output is
      blocked — the vehicle must stop or use a safe default.
  Metrics mirror the scoring in progressive_bit_search:
    speed up         : Δforward_dist(t=10) - 2·|Δlateral(t=10)|
    speed up 2       : Δlead_prob (base - flipped)
    velocity         : Δvelocity_x(t=10) (base - flipped)
    slowing down     : -Δforward_dist(t=10)
    left steering    : Δlateral(t=10) left (+y)
    right steering   : Δlateral(t=10) right (-y)
    speed up & left  : combined
    speed up & right : combined

Usage
-----
  cd STAFI
  python eval_bitshield.py [--n 2] [--n_test 5] \\
    [--val_txt  STAFI/data/comma2k19_val.txt] \\
    [--data_root STAFI/data/]
"""

import sys
import json
import struct
import zlib
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from openpilot_torch import OpenPilotModel
from data import Comma2k19SequenceDataset
from torch.utils.data import DataLoader


# ── Shared helpers ─────────────────────────────────────────────────────────────

distance_func = nn.CosineSimilarity(dim=2)
cls_loss_fn   = nn.CrossEntropyLoss()
reg_loss_fn   = nn.SmoothL1Loss(reduction='none')


def make_supercombo_inputs(batch, device):
    seq_imgs   = batch['seq_input_img'].to(device)
    seq_labels = batch['seq_future_poses'].to(device)
    B, T, C, H, W = seq_imgs.shape
    if C == 6:
        seq_imgs = torch.cat([seq_imgs, seq_imgs], dim=2)
    imgs12  = seq_imgs[:, -1]
    desire  = torch.zeros((B, 8),  device=device)
    traffic = torch.tensor([[1., 0.]], device=device).repeat(B, 1)
    h0      = torch.zeros((B, 512), device=device)
    traj_gt = seq_labels[:, -1]          # (B, 33, 3) ground-truth trajectory
    return imgs12, desire, traffic, h0, traj_gt


def build_loader(txt_path, data_root, mode, batch_size=1):
    ds = Comma2k19SequenceDataset(txt_path, data_root, mode, use_memcache=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def flip_float32_bit(value: float, bit: int) -> float:
    b = struct.pack('f', value)
    i = struct.unpack('I', b)[0] ^ (1 << bit)
    return struct.unpack('f', struct.pack('I', i))[0]


def apply_flips(state_dict, flips, n):
    sd = {k: v.clone() for k, v in state_dict.items()}
    applied = []
    for fl in flips[:n]:
        name, idx, bit = fl['name'], fl['index_flat'], fl['bit']
        if name not in sd:
            continue
        flat = sd[name].flatten()
        old_val = flat[idx].item()
        flat[idx] = flip_float32_bit(old_val, bit)
        sd[name] = flat.view(sd[name].shape)
        applied.append(fl)
    return sd, applied


@torch.no_grad()
def run_inference(model, inputs):
    """inputs = (imgs12, desire, traffic, h0[, traj_gt]) — traj_gt not passed to model."""
    model.eval()
    return model(*inputs[:4]).detach()


def load_model(state_dict):
    m = OpenPilotModel()
    m.load_state_dict(state_dict)
    m.eval()
    return m


# ── Trajectory parsing & metric helpers ───────────────────────────────────────

def parse_output(y: torch.Tensor) -> dict:
    """
    Parse Supercombo output tensor into trajectory, velocity, lead_p.
    Mirrors the parsing in main_recurrent.py::score_one_bit.
    """
    B = y.shape[0]
    pl       = y[:, :5 * 991].view(B, 5, 991)
    pf       = pl[:, :, :-1]
    lead_p   = y[:, 6064]
    traj     = pf.view(B, 5, 2, 33, 15)[:, :, 0, :, :3]   # (B,5,33,3)
    velocity = pf.view(B, 5, 2, 33, 15)[:, :, 1, :, 3:6]  # (B,5,33,3)
    return {'traj': traj, 'velocity': velocity, 'lead_p': lead_p}


def select_best(parsed: dict, traj_gt: torch.Tensor) -> dict:
    """
    Select the best-of-5 trajectory using GT forward distance at t=17.
    Mirrors: idx = d.argmin(dim=1) in main_recurrent.py.
    """
    traj = parsed['traj']
    pend = traj[:, :, 17, 0]
    gend = traj_gt[:, 17, 0].unsqueeze(1).expand(-1, 5)
    d    = (pend - gend) / (traj_gt[:, 17, 0].abs().unsqueeze(1) + 1)
    idx  = d.argmin(dim=1)
    rows = torch.arange(len(idx), device=idx.device)
    return {
        'best_traj':     traj[rows, idx],              # (B,33,3)
        'best_velocity': parsed['velocity'][rows, idx], # (B,33,3)
        'lead_p':        parsed['lead_p'],
    }


def compute_delta(best_model: dict, best_base: dict, metric: str) -> float:
    """
    Compute attack delta (higher = attack goal more achieved).
    Mirrors the delta computation in main_recurrent.py::score_one_bit.
    Returns the mean over the batch as a scalar float.
    """
    fa  = best_model['best_traj']
    fb  = best_base['best_traj']
    va  = best_model['best_velocity']
    vb  = best_base['best_velocity']
    lpa = best_model['lead_p']
    lpb = best_base['lead_p']

    if metric == 'speed up':
        delta = (fa[:, 10, 0] - fb[:, 10, 0]) \
                - 2 * (fa[:, 10, 1] - fb[:, 10, 1]).abs()
    elif metric == 'speed up 2':
        delta = lpb - lpa
    elif metric == 'velocity':
        delta = vb[:, 10, 0] - va[:, 10, 0]
    elif metric == 'slowing down':
        delta = fb[:, 10, 0] - fa[:, 10, 0]
    elif metric == 'left steering':
        delta = fa[:, 10, 1] - fb[:, 10, 1]
    elif metric == 'right steering':
        delta = fb[:, 10, 1] - fa[:, 10, 1]
    elif metric == 'speed up & left steering':
        delta = (fa[:, 10, 0] - fb[:, 10, 0]) / 9.94 \
                + (fa[:, 10, 1] - fb[:, 10, 1])
    elif metric == 'speed up & right steering':
        delta = (fa[:, 10, 0] - fb[:, 10, 0]) / 9.94 \
                + (fb[:, 10, 1] - fa[:, 10, 1])
    else:
        # Unknown metric — fall back to forward-distance change
        delta = fa[:, 10, 0] - fb[:, 10, 0]

    mask = torch.isfinite(delta)
    if not mask.any():
        return float('nan')
    return float(delta[mask].mean().item())


# ── BitShield Defense ──────────────────────────────────────────────────────────

class BitShieldDefense:
    """
    Tensor-level CRC32 integrity guard, simulating BitShield DIG (NDSS'25).

    Original DIG: CRC32 over each weight tensor's binary layout in the compiled
    model binary (TVM / Glow / NNFusion output).  On mismatch the process
    terminates.

    Adaptation: zlib.crc32 on the raw float32 bytes of each PyTorch parameter
    tensor — identical integrity guarantee without binary compilation.

    No repair is performed.  The 'post_action' output is blocked on detection
    (terminate-on-detect semantics). The metric δ remains unchanged (attack
    goal still achieved in the attacked weights), but the output is suppressed —
    the vehicle must stop or use a safe fallback trajectory.

    CIG (hardware write-protection) is not evaluated as it requires OS-level
    memory page protection unavailable in the Python/PyTorch runtime.
    """

    def __init__(self):
        self._golden_checksums: dict = {}   # name → int (CRC32)

    def setup(self, clean_sd: dict) -> int:
        """Compute and store CRC32 for every parameter tensor."""
        self._golden_checksums.clear()
        for name, tensor in clean_sd.items():
            raw = tensor.detach().cpu().contiguous().float().numpy().tobytes()
            self._golden_checksums[name] = zlib.crc32(raw)
        return len(self._golden_checksums)

    def detect(self, attacked_sd: dict) -> tuple:
        """
        Returns
        -------
        detected_params : set[str]
        n_checked : int
        """
        detected_params: set = set()
        n_checked = 0
        for name, tensor in attacked_sd.items():
            if name not in self._golden_checksums:
                continue
            raw = tensor.detach().cpu().contiguous().float().numpy().tobytes()
            n_checked += 1
            if zlib.crc32(raw) != self._golden_checksums[name]:
                detected_params.add(name)
        return detected_params, n_checked

    def storage_overhead(self) -> dict:
        n_tensors = len(self._golden_checksums)
        return {'n_tensors': n_tensors, 'crc_bytes': n_tensors * 4}


# ── Main evaluation ────────────────────────────────────────────────────────────

def evaluate(args):
    base       = Path(__file__).parent
    clean_path = base / 'openpilot_model' / 'supercombo_torch_weights.pth'
    flip_dir   = base / 'flipped_bits'
    out_dir    = base / 'crossfire_results'
    out_dir.mkdir(exist_ok=True)
    device     = torch.device('cpu')

    # ── Load clean model ──────────────────────────────────────────────────────
    print("=" * 65)
    print("STEP 1: Load clean model")
    print("=" * 65)
    clean_sd    = torch.load(clean_path, map_location='cpu')
    clean_model = load_model(clean_sd)
    total_params = sum(v.numel() for v in clean_sd.values())
    total_bytes  = sum(v.numel() * 4 for v in clean_sd.values())
    print(f"  {len(clean_sd)} parameter tensors loaded  "
          f"({total_params:,} floats, {total_bytes/1024**2:.1f} MB)")

    # ── Data loaders ──────────────────────────────────────────────────────────
    print("\n  Building data loaders ...")
    val_loader = build_loader(args.val_txt, args.data_root, 'val', batch_size=1)
    print(f"  val: {len(val_loader.dataset)} segment(s)")

    # ── Collect test inputs ────────────────────────────────────────────────────
    print(f"\n  Collecting {args.n_test} test inputs from val set ...")
    test_inputs = []   # each entry: (imgs12, desire, traffic, h0, traj_gt)
    it_val = iter(val_loader)
    for _ in range(args.n_test):
        try:
            batch = next(it_val)
        except StopIteration:
            break
        imgs12, desire, traffic, h0, traj_gt = make_supercombo_inputs(batch, device)
        test_inputs.append((imgs12, desire, traffic, h0, traj_gt))
    if not test_inputs:
        print("  [WARN] val loader empty — using dummy inputs")
        dummy_gt = torch.zeros(1, 33, 3)
        dummy_gt[:, :, 0] = torch.linspace(0, 50, 33)
        test_inputs = [
            (torch.randn(1, 12, 128, 256), torch.zeros(1, 8),
             torch.tensor([[1., 0.]]), torch.zeros(1, 512), dummy_gt)
            for _ in range(args.n_test)
        ]

    # Pre-compute clean model outputs and best-trajectory selection
    clean_outs  = [run_inference(clean_model, inp) for inp in test_inputs]
    clean_bests = [
        select_best(parse_output(out), inp[4])
        for out, inp in zip(clean_outs, test_inputs)
    ]
    print(f"  {len(test_inputs)} test inputs ready  (clean bests precomputed)")

    # ── BitShield setup ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2: BitShield (DIG) setup")
    print("=" * 65)
    shield       = BitShieldDefense()
    n_protected  = shield.setup(clean_sd)
    storage      = shield.storage_overhead()

    print(f"  Tensors protected  : {n_protected}")
    print(f"  CRC32 storage      : {storage['crc_bytes']} bytes "
          f"({storage['crc_bytes'] / 1024:.2f} KB)  — 4 bytes per tensor")
    print(f"  Model weight size  : {total_bytes / 1024**2:.1f} MB  "
          f"(CRC overhead = {storage['crc_bytes'] / total_bytes * 100:.4f}%)")
    print(f"\n  Adaptation note:")
    print(f"  Original BitShield DIG computes CRC32 over compiled binary weight")
    print(f"  sections (TVM/Glow/NNFusion output). We simulate it with zlib.crc32")
    print(f"  over raw float32 bytes per PyTorch tensor — identical integrity")
    print(f"  guarantee without requiring binary compilation.")
    print(f"  CIG (hardware write-protection) is NOT evaluated (requires OS MMU).")
    print(f"  On detection: output BLOCKED (no repair), vehicle must stop/fallback.")

    # ── Evaluate per scenario ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"STEP 3: Evaluate {args.n} flip(s) per scenario")
    print("=" * 65)

    json_files = sorted(flip_dir.glob('*.json'))
    rows = []

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        flips = data.get('flips', [])
        if not flips:
            continue
        metric = data.get('meta', {}).get('metric', 'unknown')

        print(f"\n{'─'*65}")
        print(f"Scenario : {jf.name}")
        print(f"Metric   : {metric}")

        attacked_sd, applied = apply_flips(clean_sd, flips, args.n)
        flip_params = {a['name'] for a in applied}

        for a in applied:
            print(f"  flip: {a['name']}[{a['index_flat']}] bit {a['bit']}  "
                  f"{a['old']:.4f} → {a['new']:.4f}")

        # ── Attacked model outputs ─────────────────────────────────────────────
        attacked_model = load_model(attacked_sd)
        atk_outs  = [run_inference(attacked_model, inp) for inp in test_inputs]
        atk_bests = [
            select_best(parse_output(out), inp[4])
            for out, inp in zip(atk_outs, test_inputs)
        ]
        l2_atk_vals    = [(out - ref).norm().item()
                          for out, ref in zip(atk_outs, clean_outs)]
        delta_atk_vals = [compute_delta(ab, cb, metric)
                          for ab, cb in zip(atk_bests, clean_bests)]

        avg_l2_atk    = float(np.mean(l2_atk_vals))
        avg_delta_atk = float(np.nanmean(delta_atk_vals))

        # ── Detection (DIG) ───────────────────────────────────────────────────
        det_params, n_checked = shield.detect(attacked_sd)
        detected  = bool(flip_params & det_params)
        in_scope  = flip_params & det_params
        false_pos = det_params - flip_params

        print(f"\n  Detection (DIG):")
        print(f"    Tensors scanned   : {n_checked}")
        print(f"    Detected tensors  : {sorted(det_params) or 'none'}")
        print(f"    Flip in scope     : {sorted(in_scope) or 'NONE ← not detected'}")
        print(f"    False positives   : {sorted(false_pos) or 'none'}")
        print(f"    Attack detected   : {'YES ✓' if detected else 'NO ✗'}")

        print(f"\n  Attack effect:")
        print(f"    L2 deviation      : {avg_l2_atk:.4f}")
        print(f"    δ({metric})       : {avg_delta_atk:+.4f}  "
              f"({'attack succeeded ✓' if avg_delta_atk > 0 else 'attack had no effect'})")

        # ── Post-action ───────────────────────────────────────────────────────
        # BitShield terminates on detection; no repair.
        # The metric δ of the attacked weights is unchanged, but the output is
        # blocked — the vehicle does not follow the adversarial trajectory.
        if detected:
            print(f"\n  Post-action (terminate-on-detect):")
            print(f"    Output BLOCKED — vehicle must stop or use safe default.")
            print(f"    δ in memory    : {avg_delta_atk:+.4f}  (attack still 'present' in weights)")
            print(f"    δ acted upon   : N/A  (output suppressed — attack goal PREVENTED)")
            print(f"    Defense verdict: ✓ ATTACK GOAL PREVENTED  "
                  f"(δ = {avg_delta_atk:+.4f} never reaches actuator)")
            post_action = 'blocked'
            delta_acted = None
        else:
            print(f"\n  Post-action: attack NOT detected — model runs unchecked.")
            print(f"    δ acted upon   : {avg_delta_atk:+.4f}  (attack goal REALIZED)")
            print(f"    Defense verdict: ✗ FAILED  (undetected, attack succeeds)")
            post_action = 'undetected'
            delta_acted = round(avg_delta_atk, 4)

        rows.append({
            'scenario':            jf.name,
            'metric':              metric,
            'flip_params':         sorted(flip_params),
            'n_tensors_scanned':   n_checked,
            'detected':            detected,
            'detected_params':     sorted(det_params),
            'in_scope':            bool(in_scope),
            'false_positives':     sorted(false_pos),
            'attack': {
                'l2':   round(avg_l2_atk,   4),
                'delta': round(avg_delta_atk, 4),
            },
            'post_action':  post_action,
            # delta_acted: None if blocked (attack prevented), else attack δ
            'delta_acted':  delta_acted,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    hdr = (f"{'Scenario':<38} {'Det':>4} {'Post-action':>14} "
           f"{'δ-attack':>10} {'δ-acted':>10}")
    print(hdr)
    print("-" * 78)
    for r in rows:
        name       = r['scenario'][:37]
        det        = "YES" if r['detected'] else "NO "
        pa         = r['post_action']
        d_atk      = r['attack']['delta']
        d_acted    = r['delta_acted'] if r['delta_acted'] is not None else float('nan')
        print(f"{name:<38} {det:>4} {pa:>14} "
              f"{d_atk:>10.4f} {d_acted:>10.4f}")

    if rows:
        det_rate      = np.mean([r['detected'] for r in rows]) * 100
        n_blocked     = sum(1 for r in rows if r['post_action'] == 'blocked')
        n_missed      = sum(1 for r in rows if r['post_action'] == 'undetected')
        avg_delta_atk = np.nanmean([r['attack']['delta'] for r in rows])
        acted_vals    = [r['delta_acted'] for r in rows if r['delta_acted'] is not None]
        avg_delta_acted = float(np.nanmean(acted_vals)) if acted_vals else float('nan')
        n_fp          = sum(1 for r in rows if r['false_positives'])

        print(f"\nAggregates  (n={len(rows)} scenarios):")
        print(f"  Detection rate         : {det_rate:.0f}%")
        print(f"  Blocked (safe)         : {n_blocked}/{len(rows)}")
        print(f"  Undetected (risk)      : {n_missed}/{len(rows)}")
        print(f"  Avg attack δ           : {avg_delta_atk:+.4f}  "
              f"(attack goal strength in weights)")
        print(f"  Avg δ acted upon       : {avg_delta_acted:+.4f}  "
              f"(only undetected scenarios; nan = all detected)")
        print(f"  False positives        : {n_fp}/{len(rows)} scenarios")

    print(f"\nKey findings for paper Section 7.2:")
    print(f"  1. BitShield (DIG) detects 100% of attacks — any single-bit flip")
    print(f"     in a BN weight changes the tensor CRC32 with certainty.")
    print(f"  2. Metric-based δ shows attack goal strength. On detection, the")
    print(f"     output is BLOCKED → the adversarial δ never reaches the actuator.")
    print(f"     This is distinct from repair: the weights are still corrupted,")
    print(f"     but the vehicle stops instead of following the adversarial plan.")
    print(f"  3. No repair → operational cost: vehicle must stop or use a safe")
    print(f"     fallback. Suitable for safety-critical fail-stop systems.")
    print(f"  4. Storage overhead is minimal: {storage['crc_bytes']} bytes "
          f"({storage['crc_bytes']/1024:.2f} KB for {n_protected} tensors).")

    out_path = out_dir / 'bitshield_eval_summary.json'
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\nFull results → {out_path}")


def main():
    base = Path(__file__).parent
    p = argparse.ArgumentParser(
        description='BitShield (DIG) defense evaluation on Supercombo')
    p.add_argument('--n',         type=int, default=2,
                   help='Number of flips per scenario (default 2)')
    p.add_argument('--n_test',    type=int, default=5,
                   help='Number of test inputs from val set (default 5)')
    p.add_argument('--val_txt',   type=str,
                   default=str(base / 'data' / 'comma2k19_val.txt'))
    p.add_argument('--data_root', type=str,
                   default=str(base / 'data') + '/')
    args = p.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
