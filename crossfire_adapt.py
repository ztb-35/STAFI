#!/usr/bin/env python3
"""
CrossFire defense adapted for float32 PyTorch models (e.g., openpilot supercombo).

Original CrossFire (AAAI 2025) assumes INT8 quantized GNNs. This adaptation:
  - Same Blake2b hashing for detection & verification
  - Layer-wise [min, max] range instead of INT8 [-128,127] for OOD detection
  - Gradient-based neuropot selection (same concept, no quantization needed)
  - Works on plain float32 state_dicts without any quantization framework

Two modes:
  Mode A (Proactive):  crossfire_setup() on clean model  → get protected model + metadata
                       crossfire_detect_repair() on attacked model → detect & repair
  Mode B (Reactive):   You already have clean + attacked → Mode A still works, just pass
                       clean_state_dict to crossfire_detect_repair() for oracle repair.

Usage:
    python crossfire_adapt.py \
        --clean  openpilot_model/supercombo_torch_weights.pth \
        --attacked path/to/attacked.pth \
        --output  repaired.pth \
        --npc 0.1 --npg 1.3
"""

import hashlib
import json
import argparse
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


# ─── Hashing utilities ────────────────────────────────────────────────────────

def layer_hash(tensor: torch.Tensor, digest_size: int = 4) -> str:
    """Blake2b hash of the entire tensor (4-byte digest → high reliability)."""
    return hashlib.blake2b(
        tensor.detach().cpu().contiguous().numpy().tobytes(),
        digest_size=digest_size
    ).hexdigest()


def row_col_checksums(tensor: torch.Tensor, digest_size: int = 2) -> Tuple[List[str], List[str]]:
    """
    Compute per-row and per-column Blake2b checksums for attack localization.
    For 1D tensors (BN weight/bias), row_hashes = per-element hashes, col_hashes = [].
    """
    t = tensor.detach().cpu().contiguous()

    if t.dim() == 1:
        row_hashes = [
            hashlib.blake2b(t[i:i+1].numpy().tobytes(), digest_size=digest_size).hexdigest()
            for i in range(t.shape[0])
        ]
        return row_hashes, []

    # Reshape to 2D for Conv / Linear
    t2 = t.view(t.shape[0], -1)
    row_hashes = [
        hashlib.blake2b(t2[i].numpy().tobytes(), digest_size=digest_size).hexdigest()
        for i in range(t2.shape[0])
    ]
    col_hashes = [
        hashlib.blake2b(t2[:, j].numpy().tobytes(), digest_size=digest_size).hexdigest()
        for j in range(t2.shape[1])
    ]
    return row_hashes, col_hashes


# ─── Phase 1: Setup (run on clean model BEFORE deployment) ───────────────────

def crossfire_setup(
    clean_sd: Dict[str, torch.Tensor],
    neuropots_pct: float = 0.1,
    gamma: float = 1.3,
    depth_lambda: float = 1.1,
    pruning_ratio: float = 0.75,
    model: Optional[torch.nn.Module] = None,
    data_samples: Optional[List[torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    CrossFire initialization on a clean model state_dict.

    Steps (following the paper):
      1. L1 pruning to induce sparsity (steers attacker toward zero weights)
      2. Gradient-based neuropot selection (top-k neurons by |gradient|)
      3. Scale neuropot columns by 1/gamma_hat (activations scaled up at runtime via hook)
      4. Compute Blake2b layer hashes + row/col checksums
      5. Store metadata securely

    Returns:
      protected_sd  : modified state_dict (pruned + neuropot-scaled) for deployment
      metadata      : dict with hashes, checksums, neuropot info (store securely)
    """
    # ── Compute gradients for neuropot selection ──
    grad_sums: Dict[str, torch.Tensor] = {}
    if model is not None and data_samples:
        model.load_state_dict(clean_sd, strict=False)
        model.eval()
        for name, param in model.named_parameters():
            if param.requires_grad and param.data.dtype == torch.float32:
                grad_sums[name] = torch.zeros_like(param.data)

        print(f"Computing gradients from {len(data_samples)} samples for neuropot selection...")
        for sample in data_samples:
            model.zero_grad()
            try:
                out = model(sample)
                pseudo_target = out.detach()
                loss = torch.nn.MSELoss()(out.float(), pseudo_target.float())
                loss.backward()
                for name, param in model.named_parameters():
                    if name in grad_sums and param.grad is not None:
                        grad_sums[name] += param.grad.data.abs()
            except Exception as e:
                print(f"  Warning: gradient pass failed on sample: {e}")

    protected_sd: Dict[str, torch.Tensor] = {}
    metadata: Dict[str, Any] = {}
    depth_idx = 0

    for name, tensor in clean_sd.items():
        if not isinstance(tensor, torch.Tensor) or tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            protected_sd[name] = tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
            continue

        t = tensor.detach().cpu().float().clone()

        # ── Step 1: L1 pruning (induces sparsity, steers attacker to zero weights) ──
        # pruning_ratio = fraction of weights to KEEP (75% keep = 25% zeroed)
        if t.numel() > 4:
            keep_threshold = torch.quantile(t.abs().reshape(-1).float(), 1.0 - pruning_ratio)
            t[t.abs() < keep_threshold] = 0.0

        # ── Step 2+3: Neuropot selection & scaling (only for 2D+ tensors) ──
        neuropot_indices = None
        neuropot_values = None
        gamma_hat = None

        if t.dim() >= 2:
            t2 = t.view(t.shape[0], -1)
            num_cols = t2.shape[1]
            num_pots = max(int(num_cols * neuropots_pct), 2)

            depth_scale = gamma * (depth_lambda ** depth_idx)

            # Select by gradient magnitude (or random if no gradients)
            if name in grad_sums:
                g2 = grad_sums[name].view(t.shape[0], -1)
                col_scores = g2.abs().sum(dim=0)
                _, neuropot_indices = torch.topk(col_scores, num_pots)

                # Individualized scaling (saliency-weighted, from paper §"Choice of Scaling Parameters")
                scores = col_scores[neuropot_indices]
                min_s, max_s = scores.min(), scores.max()
                if max_s > min_s:
                    gamma_hat = 1 + (scores - min_s) * (depth_scale - 1) / (max_s - min_s)
                else:
                    gamma_hat = torch.ones(num_pots) * depth_scale
            else:
                # No gradients: random selection, uniform scaling
                neuropot_indices = torch.randperm(num_cols)[:num_pots]
                gamma_hat = torch.ones(num_pots) * depth_scale

            # Store original neuropot column values (BEFORE scaling)
            neuropot_values = t2[:, neuropot_indices].clone()

            # Scale down neuropot columns by 1/gamma_hat
            t2[:, neuropot_indices] /= gamma_hat
            t = t2.view(tensor.shape)

            depth_idx += 1

        # ── Step 4: Compute hashes ──
        lh = layer_hash(t)
        rh, ch = row_col_checksums(t)

        metadata[name] = {
            'layer_hash':       lh,
            'row_hashes':       rh,
            'col_hashes':       ch,
            'weight_min':       t.min().item(),
            'weight_max':       t.max().item(),
            'shape':            list(t.shape),
            'neuropot_indices': neuropot_indices.tolist() if neuropot_indices is not None else None,
            'neuropot_values':  neuropot_values.tolist() if neuropot_values is not None else None,
            'gamma_hat':        gamma_hat.tolist() if gamma_hat is not None else None,
        }

        protected_sd[name] = t

    print(f"CrossFire setup complete. Protected {len(metadata)} float parameters.")
    return protected_sd, metadata


# ─── Phase 2+3: Detect & Repair ───────────────────────────────────────────────

def crossfire_detect_repair(
    attacked_sd: Dict[str, torch.Tensor],
    metadata: Dict[str, Any],
    clean_sd: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """
    CrossFire detection and repair on an attacked model state_dict.

    Detection:  Compare Blake2b layer hash → if mismatch → attacked
    Localize:   Row/col checksum intersection → pinpoint changed elements
    Repair priority (per paper):
      1. Neuropot column  → restore from stored original value
      2. OOD (outside [min,max]) → unset MSBs iteratively (or zero)
      3. Clean model available → oracle restore
      4. Fallback → zero out

    Args:
        attacked_sd : state dict of the attacked model
        metadata    : output of crossfire_setup()
        clean_sd    : (optional) clean model state dict for oracle repair

    Returns:
        (repaired_sd, report)
    """
    repaired_sd = {k: v.clone() for k, v in attacked_sd.items()}
    report: Dict[str, Any] = {
        'attacked_params':   [],
        'fully_repaired':    [],
        'partially_repaired': [],
        'total_detected':    0,
        'total_repaired':    0,
        'detail':            [],
    }

    for name, tensor in attacked_sd.items():
        if name not in metadata:
            continue
        if not isinstance(tensor, torch.Tensor) or tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            continue

        meta = metadata[name]
        t_atk = tensor.detach().cpu().float().clone()

        # ── Detection: layer hash ──
        if layer_hash(t_atk) == meta['layer_hash']:
            continue   # Layer untouched

        report['attacked_params'].append(name)
        print(f"\n[DETECTED] {name}")

        t_rep = t_atk.clone()

        # ── Localize via row/col checksums ──
        curr_rh, curr_ch = row_col_checksums(t_atk)
        changed_rows = [i for i, (a, b) in enumerate(zip(meta['row_hashes'], curr_rh)) if a != b]
        changed_cols = [j for j, (a, b) in enumerate(zip(meta['col_hashes'], curr_ch)) if a != b]

        print(f"  Changed rows: {changed_rows}")
        if changed_cols:
            print(f"  Changed cols: {changed_cols}")

        neuropot_idx = (torch.tensor(meta['neuropot_indices']) if meta['neuropot_indices'] else None)
        neuropot_vals = (torch.tensor(meta['neuropot_values']) if meta['neuropot_values'] else None)

        # ── Repair ──
        if t_rep.dim() == 1:
            # 1D (BN weight / bias): row_hashes = per-element hashes
            for i in changed_rows:
                orig_val = t_atk[i].item()
                report['total_detected'] += 1
                new_val, method = _repair_scalar(
                    i_flat=i, orig_val=orig_val,
                    w_min=meta['weight_min'], w_max=meta['weight_max'],
                    clean_val=clean_sd[name][i].item() if (clean_sd and name in clean_sd) else None,
                )
                t_rep[i] = new_val
                report['total_repaired'] += 1
                report['detail'].append({'param': name, 'index': i, 'old': orig_val,
                                         'new': new_val, 'method': method})
                print(f"  [{i}] {orig_val:.6f} → {new_val:.6f}  ({method})")

        elif changed_rows and changed_cols:
            # 2D+ tensor: repair at (row, col) intersection
            t2 = t_rep.view(t_rep.shape[0], -1)
            clean_2d = (clean_sd[name].view(t_rep.shape[0], -1).float()
                        if (clean_sd and name in clean_sd) else None)

            for row in changed_rows:
                for col in changed_cols:
                    orig_val = t2[row, col].item()
                    report['total_detected'] += 1
                    method = None
                    new_val = None

                    # Priority 1: neuropot column → restore original value
                    if neuropot_idx is not None:
                        pot_match = (neuropot_idx == col).nonzero(as_tuple=True)[0]
                        if len(pot_match) > 0:
                            pos = pot_match[0]
                            new_val = neuropot_vals[row, pos].item()
                            method = 'neuropot_restore'

                    # Priority 2: clean oracle
                    if new_val is None and clean_2d is not None:
                        new_val = clean_2d[row, col].item()
                        method = 'oracle_clean'

                    # Priority 3: OOD bit-level correction
                    if new_val is None:
                        new_val, method = _repair_scalar(
                            i_flat=None, orig_val=orig_val,
                            w_min=meta['weight_min'], w_max=meta['weight_max'],
                            clean_val=None,
                        )

                    t2[row, col] = new_val
                    report['total_repaired'] += 1
                    report['detail'].append({'param': name, 'index': (row, col),
                                             'old': orig_val, 'new': new_val, 'method': method})
                    print(f"  ({row},{col}) {orig_val:.6f} → {new_val:.6f}  ({method})")

            t_rep = t2.view(tensor.shape)

        else:
            # Checksum localization failed → OOD range fallback
            ood_mask = (t_rep < meta['weight_min']) | (t_rep > meta['weight_max'])
            if ood_mask.any():
                count = int(ood_mask.sum().item())
                if clean_sd and name in clean_sd:
                    t_rep[ood_mask] = clean_sd[name].float()[ood_mask]
                    method = 'oracle_ood'
                else:
                    t_rep[ood_mask] = 0.0
                    method = 'zero_ood'
                print(f"  OOD fallback: fixed {count} weights ({method})")
                report['total_repaired'] += count

        repaired_sd[name] = t_rep.to(tensor.dtype)

        # ── Verify ──
        if layer_hash(t_rep) == meta['layer_hash']:
            report['fully_repaired'].append(name)
            print(f"  FULLY RECONSTRUCTED (hash match)")
        else:
            report['partially_repaired'].append(name)
            print(f"  Partially repaired (hash still differs)")

    return repaired_sd, report


def _repair_scalar(
    i_flat: Optional[int],
    orig_val: float,
    w_min: float,
    w_max: float,
    clean_val: Optional[float],
) -> Tuple[float, str]:
    """
    Repair a single float32 scalar following CrossFire repair priority:
      1. Oracle (clean value)
      2. OOD: iteratively unset MSBs in float32 bits until value in [w_min, w_max]
      3. Zero out
    """
    # Priority 1: oracle
    if clean_val is not None:
        return clean_val, 'oracle_clean'

    # Priority 2: OOD bit-level correction (adapted for float32)
    if orig_val < w_min or orig_val > w_max:
        bits = int.from_bytes(np.float32(orig_val).tobytes(), 'little')
        # Iteratively unset bits from bit 30 down (skip sign bit 31 first attempt)
        for bit_pos in range(30, -1, -1):
            candidate_bits = bits & ~(1 << bit_pos)
            candidate_val = np.frombuffer(
                candidate_bits.to_bytes(4, 'little'), dtype=np.float32
            )[0].item()
            if w_min <= candidate_val <= w_max:
                return candidate_val, 'ood_bit_correction'
        # If sign bit flip made it OOD, try flipping it
        candidate_bits = bits ^ (1 << 31)
        candidate_val = np.frombuffer(
            candidate_bits.to_bytes(4, 'little'), dtype=np.float32
        )[0].item()
        if w_min <= candidate_val <= w_max:
            return candidate_val, 'ood_sign_correction'
        return 0.0, 'zero_ood'

    # Priority 3: in-range flip → zero (conservative fallback, same as RADAR)
    return 0.0, 'zero_inrange'


# ─── Metadata serialization ───────────────────────────────────────────────────

def save_metadata(metadata: Dict, path: str):
    """Save CrossFire metadata to JSON (store securely, e.g., TEE)."""
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    serializable = {}
    for name, meta in metadata.items():
        serializable[name] = {k: convert(v) for k, v in meta.items()}
    with open(path, 'w') as f:
        json.dump(serializable, f)
    print(f"Metadata saved to {path}")


def load_metadata(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


# ─── Main ─────────────────────────────────────────────────────────────────────

def load_sd(path: str, device='cpu') -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        # Could be plain state_dict or {'Model': model, ...}
        if 'Model' in obj and hasattr(obj['Model'], 'state_dict'):
            return obj['Model'].state_dict()
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    if hasattr(obj, 'state_dict'):
        return obj.state_dict()
    raise ValueError(f"Cannot extract state_dict from {path}")


def main():
    parser = argparse.ArgumentParser(description='CrossFire defense for float32 PyTorch models')
    parser.add_argument('--clean',    required=True,  help='Path to clean model (.pth)')
    parser.add_argument('--attacked', required=True,  help='Path to attacked model (.pth)')
    parser.add_argument('--output',   default='repaired.pth', help='Output path for repaired model')
    parser.add_argument('--metadata', default='crossfire_metadata.json', help='Path to save/load metadata')
    parser.add_argument('--npc',      type=float, default=0.10, help='Neuropot percentage (default 0.10)')
    parser.add_argument('--npg',      type=float, default=1.30, help='Base gamma (default 1.30)')
    parser.add_argument('--lambda_d', type=float, default=1.10, help='Depth lambda (default 1.10)')
    parser.add_argument('--prune',    type=float, default=0.75, help='L1 prune keep-ratio (default 0.75)')
    args = parser.parse_args()

    print("=== Loading models ===")
    clean_sd   = load_sd(args.clean)
    attacked_sd = load_sd(args.attacked)
    print(f"Clean model:   {len(clean_sd)} parameters")
    print(f"Attacked model:{len(attacked_sd)} parameters")

    print("\n=== Phase 1: CrossFire Setup (on clean model) ===")
    protected_sd, metadata = crossfire_setup(
        clean_sd        = clean_sd,
        neuropots_pct   = args.npc,
        gamma           = args.npg,
        depth_lambda    = args.lambda_d,
        pruning_ratio   = args.prune,
        model           = None,       # Pass your model class here for gradient-based neuropots
        data_samples    = None,       # Pass sample inputs here for gradient-based neuropots
    )
    save_metadata(metadata, args.metadata)

    print("\n=== Phase 2+3: Detect & Repair (on attacked model) ===")
    repaired_sd, report = crossfire_detect_repair(
        attacked_sd = attacked_sd,
        metadata    = metadata,
        clean_sd    = clean_sd,    # Enables oracle repair (best mode since you have clean model)
    )

    torch.save(repaired_sd, args.output)
    print(f"\nRepaired model saved → {args.output}")

    print("\n=== Summary ===")
    print(f"Attacked parameters:     {report['attacked_params']}")
    print(f"Fully reconstructed:     {report['fully_repaired']}")
    print(f"Partially repaired:      {report['partially_repaired']}")
    print(f"Total flips detected:    {report['total_detected']}")
    print(f"Total flips repaired:    {report['total_repaired']}")
    reconstruction_rate = len(report['fully_repaired']) / max(len(report['attacked_params']), 1)
    print(f"Layer reconstruction rate: {reconstruction_rate*100:.1f}%")


if __name__ == '__main__':
    main()
