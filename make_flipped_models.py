"""
Apply first N bit flips from each JSON file in flipped_bits/ to the clean model,
saving each result as a separate .pth file in openpilot_model/.

Usage:
    python make_flipped_models.py [--n 2]
"""

import json
import struct
import argparse
from pathlib import Path

import torch


def flip_float32_bit(value: float, bit: int) -> float:
    """Flip a single bit (0=LSB, 31=sign) in a float32."""
    packed = struct.pack('f', value)
    int_val = struct.unpack('I', packed)[0]
    int_val ^= (1 << bit)
    packed = struct.pack('I', int_val)
    return struct.unpack('f', packed)[0]


def apply_flips(state_dict: dict, flips: list, n: int) -> dict:
    """Apply the first n flips from the flip list to a copy of state_dict."""
    sd = {k: v.clone() for k, v in state_dict.items()}

    for flip in flips[:n]:
        name = flip['name']
        idx = flip['index_flat']
        bit = flip['bit']
        expected_old = flip['old']
        expected_new = flip['new']

        if name not in sd:
            print(f"  [WARN] param '{name}' not found in model, skipping")
            continue

        tensor = sd[name]
        flat = tensor.flatten()

        actual_old = flat[idx].item()
        if abs(actual_old - expected_old) > 1e-5:
            print(f"  [WARN] {name}[{idx}]: expected old={expected_old:.6f}, "
                  f"got {actual_old:.6f} — applying flip anyway")

        flipped = flip_float32_bit(actual_old, bit)
        flat[idx] = flipped

        # Write back (flatten is a view when tensor is contiguous)
        sd[name] = flat.view(tensor.shape)

        if abs(flipped - expected_new) > 1e-3:
            print(f"  [WARN] {name}[{idx}] bit {bit}: flipped={flipped:.6f}, "
                  f"expected new={expected_new:.6f}")

    return sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=2,
                        help='Number of flips to apply from each JSON (default: 2)')
    parser.add_argument('--clean', type=str,
                        default='openpilot_model/supercombo_torch_weights.pth',
                        help='Path to clean model state_dict (relative to script dir)')
    parser.add_argument('--flip_dir', type=str,
                        default='flipped_bits',
                        help='Directory containing flip JSON files (relative to script dir)')
    parser.add_argument('--out_dir', type=str,
                        default='openpilot_model',
                        help='Output directory for flipped models (relative to script dir)')
    args = parser.parse_args()

    base = Path(__file__).parent
    clean_path = base / args.clean
    flip_dir = base / args.flip_dir
    out_dir = base / args.out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading clean model from {clean_path}")
    clean_sd = torch.load(clean_path, map_location='cpu')
    print(f"  {len(clean_sd)} parameters loaded")

    json_files = sorted(flip_dir.glob('*.json'))
    print(f"\nFound {len(json_files)} JSON file(s), applying first {args.n} flip(s) each\n")

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        flips = data.get('flips', [])
        if not flips:
            print(f"[SKIP] {jf.name}: no flips found")
            continue

        n_apply = min(args.n, len(flips))
        print(f"Processing {jf.name}  ({n_apply} flip(s))")
        for i, fl in enumerate(flips[:n_apply]):
            print(f"  flip {i}: {fl['name']}[{fl['index_flat']}] bit {fl['bit']}  "
                  f"{fl['old']:.6f} → {fl['new']:.6f}")

        flipped_sd = apply_flips(clean_sd, flips, n_apply)

        # Output filename: replace .json suffix with _flipped.pth
        out_name = jf.stem + '_flipped.pth'
        out_path = out_dir / out_name
        torch.save(flipped_sd, out_path)
        print(f"  Saved → {out_path}\n")

    print("Done.")


if __name__ == '__main__':
    main()
