"""
defense_eval.py
===============
Evaluation of RADAR, Aegis, and BitShield defenses against bit-flip attacks
on the OpenPilot Supercombo float32 model.

Each defense is adapted from its original design, which targets 8-bit quantized
classification models. Adaptations and inherent limitations for float32 are
documented inline.

Design Summary
--------------
RADAR (NDSS'21, Ye et al.)
  Original : Grain-wise codebook checksum on 8-bit quantized weights.
             Quick-recovery zeroes the flipped grain.
  Adapted  : Grain-wise CRC32 on raw float32 bytes (grain_size=8).
             Two recovery modes: zero-grain (original) and oracle-restore (upper bound).

Aegis (NDSS'21, Li et al.)
  Original : CSB (Coding Sparse Binary) weight clustering + trigger-based detection
             on 8-bit quantized models. Cluster-centroid recovery.
  Adapted  : (a) Output-fingerprint detection: reference outputs on fixed inputs
             serve as the "trigger response"; deviation > threshold → detected.
             (b) CSB clustering approximated as k-bins over float32 values.
             Recovery uses stored golden values (equivalent to oracle restore).
  Limitation: CSB requires integer quantization levels; float32 has no natural
             codebook, so the detection relies on output divergence instead.

BitShield (NDSS'25, Cao et al.)
  Original : Binary-level DIG (Dynamic Integrity Guard) checksum + CIG (Control
             Integrity Guard) for compiled model binaries (TVM/Glow/NNFusion).
             Detection triggers process termination; no weight repair.
  Adapted  : Tensor-level CRC32 checksum (analogous to DIG, grain = entire tensor).
             CIG and binary compilation are not applicable to PyTorch inference.
  Limitation: Without binary compilation, hardware-level memory templates cannot
             be used. Detection granularity is per-tensor, not per-byte.

Evaluation Metrics (consistent with eval_aspis.py / eval_crossfire.py)
  - detected      : whether the attack was flagged
  - l2_attacked   : || attacked_output  - clean_output ||₂
  - l2_repaired   : || repaired_output  - clean_output ||₂
  - reduction_pct : (1 - l2_repaired / l2_attacked) × 100

Usage
-----
  cd STAFI
  python defense_eval.py [--n 2] [--n_test 5] [--grain_size 8] [--fp_inputs 10]
"""

import sys
import json
import struct
import zlib
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from openpilot_torch import OpenPilotModel
from data import Comma2k19SequenceDataset
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")


# ── Shared helpers (same as eval_aspis.py) ────────────────────────────────────

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
    traj_gt = seq_labels[:, -1]
    return imgs12, desire, traffic, h0, traj_gt


def build_loader(txt_path, data_root, mode, batch_size=1):
    ds = Comma2k19SequenceDataset(txt_path, data_root, mode, use_memcache=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=0)


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
    model.eval()
    out = model(*inputs)
    return out.detach()


def load_model(state_dict):
    m = OpenPilotModel()
    m.load_state_dict(state_dict)
    m.eval()
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Defense 1: RADAR
# ══════════════════════════════════════════════════════════════════════════════

class RADARDefense:
    """
    Grain-wise CRC32 checksum detection, adapted from RADAR (NDSS'21).

    Original RADAR uses 8-bit integer grain sums + codebook-based signatures.
    Here we use zlib.crc32 on the raw float32 bytes of each grain.

    Two recovery strategies:
      'zero'   - zero out the affected grain (RADAR's "quick recovery")
      'oracle' - restore exact golden values from stored backup
                 (requires full copy storage; oracle upper bound only)
    """

    def __init__(self, grain_size: int = 8):
        self.grain_size = grain_size
        self._golden_checksums: dict = {}   # name → list[int]
        self._golden_sd: dict = {}          # name → tensor (oracle backup)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _tensor_to_grains(self, t: torch.Tensor):
        """Flatten tensor, split into grains of self.grain_size."""
        flat = t.detach().cpu().flatten().float()
        n = flat.numel()
        pad = (-n) % self.grain_size
        if pad:
            flat = torch.cat([flat, torch.zeros(pad)])
        return flat.view(-1, self.grain_size)   # (n_grains, grain_size)

    @staticmethod
    def _grain_crc(grain: torch.Tensor) -> int:
        return zlib.crc32(grain.numpy().tobytes())

    def setup(self, clean_sd: dict):
        """Compute golden grain checksums for every parameter tensor."""
        self._golden_checksums.clear()
        self._golden_sd = {k: v.clone() for k, v in clean_sd.items()}
        for name, tensor in clean_sd.items():
            grains = self._tensor_to_grains(tensor)
            self._golden_checksums[name] = [
                self._grain_crc(grains[i]) for i in range(grains.shape[0])
            ]

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, attacked_sd: dict):
        """
        Returns (detected_params, affected_grain_map).
          detected_params   : set of param names where any grain checksum differs
          affected_grain_map: {name: [grain_idx, ...]}
        """
        detected_params = set()
        affected_grains: dict = defaultdict(list)

        for name, tensor in attacked_sd.items():
            if name not in self._golden_checksums:
                continue
            grains = self._tensor_to_grains(tensor)
            golden = self._golden_checksums[name]
            for gi in range(min(grains.shape[0], len(golden))):
                if self._grain_crc(grains[gi]) != golden[gi]:
                    detected_params.add(name)
                    affected_grains[name].append(gi)

        return detected_params, dict(affected_grains)

    # ── Recovery ──────────────────────────────────────────────────────────────

    def repair_zero(self, attacked_sd: dict, affected_grains: dict) -> dict:
        """Zero-out all elements in detected grains (RADAR quick recovery)."""
        sd = {k: v.clone() for k, v in attacked_sd.items()}
        for name, grain_idxs in affected_grains.items():
            if name not in sd:
                continue
            orig_shape = sd[name].shape
            n = sd[name].numel()
            pad = (-n) % self.grain_size
            flat = sd[name].flatten()
            if pad:
                flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
            grains = flat.view(-1, self.grain_size)
            for gi in grain_idxs:
                if gi < grains.shape[0]:
                    grains[gi] = 0.0
            sd[name] = grains.view(-1)[:n].view(orig_shape)
        return sd

    def repair_oracle(self, attacked_sd: dict, affected_grains: dict) -> dict:
        """Restore affected grains from golden backup (oracle upper bound)."""
        sd = {k: v.clone() for k, v in attacked_sd.items()}
        for name, grain_idxs in affected_grains.items():
            if name not in sd or name not in self._golden_sd:
                continue
            orig_shape = sd[name].shape
            n = sd[name].numel()
            pad = (-n) % self.grain_size

            atk_flat = sd[name].flatten()
            gld_flat = self._golden_sd[name].flatten()
            if pad:
                atk_flat = torch.cat([atk_flat, torch.zeros(pad, dtype=atk_flat.dtype)])
                gld_flat = torch.cat([gld_flat, torch.zeros(pad, dtype=gld_flat.dtype)])
            atk_grains = atk_flat.view(-1, self.grain_size)
            gld_grains = gld_flat.view(-1, self.grain_size)
            for gi in grain_idxs:
                if gi < atk_grains.shape[0]:
                    atk_grains[gi] = gld_grains[gi]
            sd[name] = atk_grains.view(-1)[:n].view(orig_shape)
        return sd


# ══════════════════════════════════════════════════════════════════════════════
# Defense 2: Aegis
# ══════════════════════════════════════════════════════════════════════════════

class AegisDefense:
    """
    Output-fingerprint detection + backup repair, adapted from Aegis (NDSS'21).

    Original Aegis uses:
      (a) CSB (Coding Sparse Binary): 8-bit quantized weights must match a
          pre-defined codebook; any weight falling outside valid levels is detected.
      (b) Trigger-based detection: a backdoor trigger is baked into the model
          during quantization-aware training; model outputs on trigger inputs
          change if weights are attacked.

    Adaptation for float32 Supercombo:
      CSB cannot be applied (no integer codebook for float32), so we implement
      (b) as output-fingerprint detection: reference outputs on fixed validation
      inputs are stored during setup; significant L2 divergence after attack
      indicates detection. Threshold calibrated as mean + 3σ over clean runs.

    Recovery: restore all changed weight values from a golden backup copy.
    This is equivalent to oracle restoration — Aegis's cluster-centroid restoration
    is subsumed by exact backup for float32 weights.

    Limitation: Aegis's core CSB mechanism requires 8-bit quantization. The
    output-fingerprint variant applied here detects coarse-grained attacks that
    affect model outputs but may miss very targeted flips that land outside
    the output distribution used for fingerprinting.
    """

    def __init__(self, fp_threshold_sigma: float = 3.0):
        self.sigma = fp_threshold_sigma
        self._fingerprints: list = []    # list of clean output tensors
        self._fp_inputs: list = []       # reference inputs
        self._threshold: float = 0.0    # detection threshold
        self._golden_sd: dict = {}       # backup for repair

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, model, val_loader, n_fp_inputs: int = 10, device='cpu'):
        """
        Record output fingerprints on n_fp_inputs reference val inputs.
        Calibrate detection threshold from inter-run variability (should be ~0
        for a deterministic model; we add a small buffer).
        """
        model.eval()
        self._fp_inputs.clear()
        self._fingerprints.clear()

        it = iter(val_loader)
        for _ in range(n_fp_inputs):
            try:
                batch = next(it)
            except StopIteration:
                break
            imgs12, desire, traffic, h0, _ = make_supercombo_inputs(batch, device)
            inp = (imgs12, desire, traffic, h0)
            out = run_inference(model, inp)
            self._fp_inputs.append(inp)
            self._fingerprints.append(out)

        # Calibrate threshold: model is deterministic, so same-input L2 = 0.
        # We use a small epsilon + sigma * std_across_inputs as threshold.
        if len(self._fingerprints) > 1:
            # Compute pairwise L2 norms among fingerprints (cross-input variation)
            fp_norms = torch.stack([f.norm() for f in self._fingerprints])
            mean_norm = fp_norms.mean().item()
            std_norm  = fp_norms.std().item() if len(self._fingerprints) > 1 else 0.0
            self._threshold = mean_norm * 0.05 + self.sigma * std_norm + 1e-3
        else:
            self._threshold = 1.0   # fallback

    def setup_backup(self, clean_sd: dict):
        """Store golden weight backup for repair."""
        self._golden_sd = {k: v.clone() for k, v in clean_sd.items()}

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, attacked_model) -> tuple:
        """
        Compare attacked model outputs against stored fingerprints.
        Returns (detected: bool, max_l2: float).
        """
        attacked_model.eval()
        max_l2 = 0.0
        for inp, ref in zip(self._fp_inputs, self._fingerprints):
            out = run_inference(attacked_model, inp)
            l2 = (out - ref).norm().item()
            max_l2 = max(max_l2, l2)
        detected = max_l2 > self._threshold
        return detected, max_l2

    # ── Recovery ──────────────────────────────────────────────────────────────

    def repair(self, attacked_sd: dict) -> dict:
        """
        Restore all weights that differ from golden backup.
        (Cluster-centroid restoration from Aegis's CSB is approximated
        by exact oracle restoration for float32 weights.)
        """
        sd = {}
        for name, atk_v in attacked_sd.items():
            if name in self._golden_sd:
                gld_v = self._golden_sd[name]
                diff = (atk_v - gld_v).abs().max().item()
                sd[name] = gld_v.clone() if diff > 1e-6 else atk_v.clone()
            else:
                sd[name] = atk_v.clone()
        return sd


# ══════════════════════════════════════════════════════════════════════════════
# Defense 3: BitShield
# ══════════════════════════════════════════════════════════════════════════════

class BitShieldDefense:
    """
    Tensor-level DIG checksum detection, adapted from BitShield (NDSS'25).

    Original BitShield:
      - Compiles model to binary (TVM/Glow/NNFusion)
      - Instruments binary with DIG (Dynamic Integrity Guard): per-group CRC
        checksums in the compiled weight memory
      - CIG (Control Integrity Guard): protects instruction control flow
      - On detection: terminates the process (no weight repair)

    Adaptation for float32 Supercombo:
      Binary compilation is not applicable to PyTorch inference. We simulate
      DIG by computing tensor-level CRC32 checksums (grain = entire tensor).
      This provides coarser detection granularity than BitShield's byte-level
      groups but preserves the detection-without-repair semantics.

    CIG and memory-template-based flip restrictions are not implemented
    (require compiled binary + hardware-level DRAM profiling).

    Limitation: Without binary compilation, BitShield's core hardware memory
    protection cannot be applied. The simulation shows what DIG detection would
    achieve if the model were compiled — detection is 100% for any flip, but
    the deployed (compiled) model would differ from PyTorch due to compiler
    optimizations and quantization, affecting task performance.
    """

    def __init__(self):
        self._golden_checksums: dict = {}   # name → int (tensor-level CRC32)

    # ── Setup ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _tensor_crc(t: torch.Tensor) -> int:
        return zlib.crc32(t.detach().cpu().contiguous().float().numpy().tobytes())

    def setup(self, clean_sd: dict):
        """Compute tensor-level CRC32 for all parameters (simulated DIG)."""
        self._golden_checksums = {
            name: self._tensor_crc(tensor)
            for name, tensor in clean_sd.items()
        }

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, attacked_sd: dict) -> tuple:
        """
        Returns (detected_params: set, detail: dict).
        detected_params: names of tensors whose CRC32 changed.
        """
        detected_params = set()
        detail = {}
        for name, tensor in attacked_sd.items():
            if name not in self._golden_checksums:
                continue
            current_crc = self._tensor_crc(tensor)
            if current_crc != self._golden_checksums[name]:
                detected_params.add(name)
                detail[name] = {'golden': self._golden_checksums[name],
                                'current': current_crc}
        return detected_params, detail

    # ── Recovery ──────────────────────────────────────────────────────────────

    # BitShield has no weight repair — it terminates on detection.
    # For evaluation purposes, we compute what the loss would be if
    # the system simply ran with the attacked weights (no repair).
    # We also report "terminate" as the recovery action.


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ══════════════════════════════════════════════════════════════════════════════

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
    clean_sd = torch.load(clean_path, map_location='cpu')
    clean_model = load_model(clean_sd)
    print(f"  {len(clean_sd)} parameter tensors loaded")

    # ── Data loaders ──────────────────────────────────────────────────────────
    print("\n  Building data loaders ...")
    train_loader = build_loader(args.train_txt, args.data_root, 'train', batch_size=1)
    val_loader   = build_loader(args.val_txt,   args.data_root, 'val',   batch_size=1)
    print(f"  train: {len(train_loader.dataset)} segment(s),  val: {len(val_loader.dataset)} segment(s)")

    # ── Collect test inputs (val) ─────────────────────────────────────────────
    print(f"\n  Collecting {args.n_test} test inputs from val set ...")
    test_inputs = []
    it_val = iter(val_loader)
    for _ in range(args.n_test):
        try:
            batch = next(it_val)
        except StopIteration:
            break
        imgs12, desire, traffic, h0, _ = make_supercombo_inputs(batch, device)
        test_inputs.append((imgs12, desire, traffic, h0))
    if not test_inputs:
        print("  [WARN] val loader empty — using dummy inputs")
        test_inputs = [(torch.randn(1, 12, 128, 256), torch.zeros(1, 8),
                        torch.tensor([[1., 0.]]), torch.zeros(1, 512))
                       for _ in range(args.n_test)]
    clean_outs = [run_inference(clean_model, inp) for inp in test_inputs]
    print(f"  {len(test_inputs)} test inputs ready")

    # ── Defense setup ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2: Defense setup")
    print("=" * 65)

    # RADAR
    print(f"\n  [RADAR] grain_size={args.grain_size}")
    radar = RADARDefense(grain_size=args.grain_size)
    radar.setup(clean_sd)
    total_grains = sum(len(v) for v in radar._golden_checksums.values())
    print(f"    {total_grains:,} grains across {len(radar._golden_checksums)} tensors")

    # Aegis
    print(f"\n  [Aegis] fingerprint inputs={args.fp_inputs}, threshold_sigma={args.fp_sigma}")
    aegis = AegisDefense(fp_threshold_sigma=args.fp_sigma)
    aegis.setup(clean_model, val_loader, n_fp_inputs=args.fp_inputs, device=device)
    aegis.setup_backup(clean_sd)
    print(f"    {len(aegis._fp_inputs)} fingerprint inputs stored")
    print(f"    detection threshold: {aegis._threshold:.4f}")
    print(f"    NOTE: Aegis CSB requires 8-bit quantization (not applicable to float32).")
    print(f"          Output-fingerprint detection is used as adaptation.")

    # BitShield
    print(f"\n  [BitShield] tensor-level DIG (simulated)")
    bitshield = BitShieldDefense()
    bitshield.setup(clean_sd)
    print(f"    {len(bitshield._golden_checksums)} tensor CRC32 checksums stored")
    print(f"    NOTE: Binary compilation (TVM/Glow/NNFusion) not applicable to PyTorch.")
    print(f"          DIG simulated at tensor granularity; CIG not implemented.")

    # ── Evaluate each flip scenario ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"STEP 3: Evaluate {args.n} flip(s) per scenario")
    print("=" * 65)

    json_files = sorted(flip_dir.glob('*.json'))
    all_rows = []

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        flips = data.get('flips', [])
        if not flips:
            continue

        print(f"\n{'─'*65}")
        print(f"Scenario: {jf.name}")

        attacked_sd, applied = apply_flips(clean_sd, flips, args.n)
        flip_params = {a['name'] for a in applied}
        for a in applied:
            print(f"  flip: {a['name']}[{a['index_flat']}] bit {a['bit']}  "
                  f"{a['old']:.4f} → {a['new']:.4f}")

        attacked_model = load_model(attacked_sd)

        # L2 of attacked model
        l2_attacked_vals = [(run_inference(attacked_model, inp) - ref).norm().item()
                            for inp, ref in zip(test_inputs, clean_outs)]
        avg_l2_atk = float(np.mean(l2_attacked_vals))

        row = {'scenario': jf.name, 'flip_params': sorted(flip_params),
               'l2_attacked': round(avg_l2_atk, 4)}

        # ── RADAR ────────────────────────────────────────────────────────────
        radar_det_params, affected_grains = radar.detect(attacked_sd)
        radar_detected = bool(flip_params & radar_det_params)
        in_scope_radar = flip_params & radar_det_params

        # zero-grain recovery
        sd_zero = radar.repair_zero(attacked_sd, affected_grains)
        m_zero  = load_model(sd_zero)
        l2_zero = float(np.mean([(run_inference(m_zero, inp) - ref).norm().item()
                                  for inp, ref in zip(test_inputs, clean_outs)]))

        # oracle recovery (upper bound)
        sd_orc = radar.repair_oracle(attacked_sd, affected_grains)
        m_orc  = load_model(sd_orc)
        l2_orc = float(np.mean([(run_inference(m_orc, inp) - ref).norm().item()
                                  for inp, ref in zip(test_inputs, clean_outs)]))

        def reduction(l2_rep):
            return (1 - l2_rep / max(avg_l2_atk, 1e-9)) * 100

        print(f"\n  [RADAR  ] detected params  : {sorted(radar_det_params) or 'none'}")
        print(f"            flip_params in scope : {sorted(in_scope_radar) or 'NONE'}")
        print(f"            detected attack  : {'YES' if radar_detected else 'NO'}")
        print(f"            L2 (attacked)    : {avg_l2_atk:.4f}")
        print(f"            L2 (zero-repair) : {l2_zero:.4f}  ({reduction(l2_zero):+.1f}%)")
        print(f"            L2 (oracle-rep)  : {l2_orc:.4f}  ({reduction(l2_orc):+.1f}%)")
        print(f"            n_affected_grains: {sum(len(v) for v in affected_grains.values())}")

        row['RADAR'] = {
            'detected':         radar_detected,
            'in_scope':         bool(in_scope_radar),
            'l2_attacked':      round(avg_l2_atk, 4),
            'l2_zero_repair':   round(l2_zero,    4),
            'l2_oracle_repair': round(l2_orc,     4),
            'reduc_zero_pct':   round(reduction(l2_zero), 1),
            'reduc_oracle_pct': round(reduction(l2_orc),  1),
            'n_grains_affected': sum(len(v) for v in affected_grains.values()),
        }

        # ── Aegis ─────────────────────────────────────────────────────────────
        aegis_detected, max_fp_l2 = aegis.detect(attacked_model)
        sd_rep_aegis = aegis.repair(attacked_sd) if aegis_detected else attacked_sd
        m_rep_aegis  = load_model(sd_rep_aegis)
        l2_aegis_rep = float(np.mean([(run_inference(m_rep_aegis, inp) - ref).norm().item()
                                       for inp, ref in zip(test_inputs, clean_outs)]))

        print(f"\n  [Aegis  ] fingerprint L2     : {max_fp_l2:.4f} (threshold={aegis._threshold:.4f})")
        print(f"            detected attack  : {'YES' if aegis_detected else 'NO'}")
        print(f"            L2 (attacked)    : {avg_l2_atk:.4f}")
        print(f"            L2 (repaired)    : {l2_aegis_rep:.4f}  ({reduction(l2_aegis_rep):+.1f}%)")
        print(f"            recovery method  : oracle-restore (CSB centroid N/A for float32)")

        row['Aegis'] = {
            'detected':          aegis_detected,
            'max_fp_l2':         round(max_fp_l2,      4),
            'threshold':         round(aegis._threshold, 4),
            'l2_attacked':       round(avg_l2_atk,     4),
            'l2_repaired':       round(l2_aegis_rep,   4),
            'reduc_pct':         round(reduction(l2_aegis_rep), 1),
        }

        # ── BitShield ─────────────────────────────────────────────────────────
        bs_det_params, bs_detail = bitshield.detect(attacked_sd)
        bs_detected = bool(flip_params & bs_det_params)
        in_scope_bs  = flip_params & bs_det_params
        # BitShield terminates on detection — no repair, process would abort.
        # We report the attacked L2 as the effective post-"repair" loss.

        print(f"\n  [BitShield] detected tensors : {sorted(bs_det_params) or 'none'}")
        print(f"              flip_params in scope : {sorted(in_scope_bs) or 'NONE'}")
        print(f"              detected attack  : {'YES' if bs_detected else 'NO'}")
        print(f"              L2 (attacked)    : {avg_l2_atk:.4f}")
        print(f"              recovery         : TERMINATE (no repair — BitShield design)")
        print(f"              Note: DIG simulated at tensor granularity; CIG not implemented.")

        row['BitShield'] = {
            'detected':        bs_detected,
            'in_scope':        bool(in_scope_bs),
            'detected_tensors': sorted(bs_det_params),
            'l2_attacked':     round(avg_l2_atk, 4),
            'l2_post_action':  round(avg_l2_atk, 4),  # terminate = no functional repair
            'reduc_pct':       0.0,                    # termination ≠ recovery
        }

        all_rows.append(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")

    for defense_name in ['RADAR', 'Aegis', 'BitShield']:
        print(f"\n  ── {defense_name} ──")
        hdr = (f"  {'Scenario':<40} {'Det':>5} {'L2-atk':>7} "
               f"{'L2-rep':>7} {'Reduc%':>7}")
        print(hdr)
        print("  " + "-" * 62)
        det_rates = []
        for r in all_rows:
            if defense_name not in r:
                continue
            d = r[defense_name]
            det_str = "YES" if d['detected'] else "NO "
            if defense_name == 'RADAR':
                l2_rep = d['l2_oracle_repair']
                reduc  = d['reduc_oracle_pct']
            elif defense_name == 'Aegis':
                l2_rep = d['l2_repaired']
                reduc  = d['reduc_pct']
            else:  # BitShield
                l2_rep = d['l2_post_action']
                reduc  = d['reduc_pct']
            name = r['scenario'][:39]
            print(f"  {name:<40} {det_str:>5} {d['l2_attacked']:>7.4f} "
                  f"{l2_rep:>7.4f} {reduc:>6.1f}%")
            det_rates.append(1 if d['detected'] else 0)
        if det_rates:
            avg_det = np.mean(det_rates) * 100
            print(f"\n  Average detection rate: {avg_det:.0f}%")

    # Detection rate table across all three defenses
    print(f"\n{'='*65}")
    print("DETECTION RATE COMPARISON")
    print(f"{'='*65}")
    hdr2 = f"  {'Scenario':<40} {'RADAR':>8} {'Aegis':>8} {'BitShield':>10}"
    print(hdr2)
    print("  " + "-" * 68)
    for r in all_rows:
        name = r['scenario'][:39]
        rdr  = "YES" if r.get('RADAR', {}).get('detected', False) else "NO"
        aeg  = "YES" if r.get('Aegis', {}).get('detected', False) else "NO"
        bts  = "YES" if r.get('BitShield', {}).get('detected', False) else "NO"
        print(f"  {name:<40} {rdr:>8} {aeg:>8} {bts:>10}")

    print(f"\n  Notes:")
    print(f"  RADAR     : grain_size={args.grain_size}, zero-recovery zeroes surrounding clean weights")
    print(f"  Aegis     : CSB not applicable to float32; fingerprint detection only;")
    print(f"              oracle-restore used instead of cluster-centroid restoration")
    print(f"  BitShield : Binary compilation not applicable; DIG simulated at tensor level;")
    print(f"              process-terminate action means no functional recovery")

    # Save JSON
    out_path = out_dir / 'defense_eval_summary.json'
    with open(out_path, 'w') as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nFull results → {out_path}")


def main():
    base = Path(__file__).parent
    p = argparse.ArgumentParser(description='Evaluate RADAR, Aegis, BitShield on Supercombo')
    p.add_argument('--n',         type=int,   default=2,
                   help='Number of flips per scenario (default 2)')
    p.add_argument('--n_test',    type=int,   default=5,
                   help='Number of test inputs from val set (default 5)')
    p.add_argument('--grain_size',type=int,   default=8,
                   help='RADAR grain size (default 8, matching paper)')
    p.add_argument('--fp_inputs', type=int,   default=10,
                   help='Aegis: number of fingerprint reference inputs (default 10)')
    p.add_argument('--fp_sigma',  type=float, default=3.0,
                   help='Aegis: detection threshold = mean + fp_sigma * std (default 3.0)')
    p.add_argument('--train_txt', type=str,   default=str(base / 'data' / 'comma2k19_train.txt'))
    p.add_argument('--val_txt',   type=str,   default=str(base / 'data' / 'comma2k19_val.txt'))
    p.add_argument('--data_root', type=str,   default=str(base / 'data') + '/')
    args = p.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
