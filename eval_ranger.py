"""
RANGER defense evaluation on pre-flipped Supercombo models using recurrent dataset.
Computes delta between defense-clamped and clean models for all metrics.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from openpilot_torch import OpenPilotModel
from data import Comma2k19SequenceRecurrentDataset
from torch.utils.data import DataLoader


distance_func = nn.CosineSimilarity(dim=2)
cls_loss_fn = nn.CrossEntropyLoss()
reg_loss_fn = nn.SmoothL1Loss(reduction='none')


def make_supercombo_inputs(batch, device):
    """
    batch:
      - 'seq_input_img': (B,T,C,H,W) (C=6 or 12)
      - 'seq_future_poses': (B,T,33,3)
    returns:
      imgs12 (B,T,12,H,W), desire (B,8), traffic (B,2), h0 (B,512), traj_gt (B,T,33,3)
    """
    seq_imgs = batch['seq_input_img'].to(device, non_blocking=True)
    seq_labels = batch['seq_future_poses'].to(device, non_blocking=True)
    B, T, C, H, W = seq_imgs.shape
    if C == 6:
        seq_imgs = torch.cat([seq_imgs, seq_imgs], dim=2)
    imgs12 = seq_imgs
    desire = torch.zeros((B, 8), device=device)
    traffic = torch.tensor([[1., 0.]], device=device).repeat(B, 1)
    h0 = torch.zeros((B, 512), device=device)
    traj_gt = seq_labels
    return imgs12, desire, traffic, h0, traj_gt


def build_loader(txt_path, data_root, data_length, frame_stream_length, mode, batch_size, device):
    ds = Comma2k19SequenceRecurrentDataset(txt_path, data_root, data_length, frame_stream_length,
                                           mode, use_memcache=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                      pin_memory=(device.type == 'cuda'))


@torch.no_grad()
def run_inference(model, inputs):
    model.eval()
    return model(*inputs[:4]).detach()


def load_model(state_dict):
    m = OpenPilotModel()
    m.load_state_dict(state_dict)
    m.eval()
    return m


# ── Trajectory parsing & metric helpers ─────────────────────────────────────────────

def parse_output(y: torch.Tensor) -> dict:
    B = y.shape[0]
    pl = y[:, :5 * 991].view(B, 5, 991)
    pf = pl[:, :, :-1]
    lead_p = y[:, 6064]
    traj = pf.view(B, 5, 2, 33, 15)[:, :, 0, :, :3]
    velocity = pf.view(B, 5, 2, 33, 15)[:, :, 1, :, 3:6]
    return {'traj': traj, 'velocity': velocity, 'lead_p': lead_p}


def select_best(parsed: dict, traj_gt: torch.Tensor) -> dict:
    traj = parsed['traj']
    pend = traj[:, :, 17, 0]
    gend = traj_gt[:, 17, 0].unsqueeze(1).expand(-1, 5)
    d = (pend - gend) / (traj_gt[:, 17, 0].abs().unsqueeze(1) + 1)
    idx = d.argmin(dim=1)
    rows = torch.arange(len(idx), device=idx.device)
    return {
        'best_traj': traj[rows, idx],
        'best_velocity': parsed['velocity'][rows, idx],
        'lead_p': parsed['lead_p'],
    }


def compute_delta(best_model: dict, best_base: dict, metric: str) -> float:
    fa = best_model['best_traj']
    fb = best_base['best_traj']
    va = best_model['best_velocity']
    vb = best_base['best_velocity']
    lpa = best_model['lead_p']
    lpb = best_base['lead_p']

    if metric == 'speed up':
        delta = (fa[:, 10, 0] - fb[:, 10, 0]) - 2 * (fa[:, 10, 1] - fb[:, 10, 1]).abs()
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
        delta = (fa[:, 10, 0] - fb[:, 10, 0]) / 9.94 + (fa[:, 10, 1] - fb[:, 10, 1])
    elif metric == 'speed up & right steering':
        delta = (fa[:, 10, 0] - fb[:, 10, 0]) / 9.94 + (fb[:, 10, 1] - fa[:, 10, 1])
    else:
        delta = fa[:, 10, 0] - fb[:, 10, 0]

    mask = torch.isfinite(delta)
    if not mask.any():
        return float('nan')
    return float(delta[mask].mean().item())


ALL_METRICS = [
    'speed up', 'speed up 2', 'velocity', 'slowing down',
    'left steering', 'right steering', 'speed up & left steering', 'speed up & right steering'
]


# ── RANGER Defense ──────────────────────────────────────────────────────────────

class RangerDefense:
    def __init__(self, margin: float = 0.0):
        self.margin = margin
        self._golden_range = {}
        self._golden_sd = {}

    def setup(self, clean_sd: dict):
        self._golden_range.clear()
        self._golden_sd = {k: v.clone() for k, v in clean_sd.items()}
        for name, tensor in clean_sd.items():
            low = tensor.min().item() - self.margin
            high = tensor.max().item() + self.margin
            self._golden_range[name] = (torch.tensor(low), torch.tensor(high))
        total_params = len(self._golden_range)
        return total_params

    def detect(self, attacked_sd: dict):
        detected_params = set()
        affected_ranges = {}
        for name, tensor in attacked_sd.items():
            if name not in self._golden_range:
                continue
            low, high = self._golden_range[name]
            mask = (tensor < low) | (tensor > high)
            count = int(mask.sum().item())
            if count > 0:
                detected_params.add(name)
                affected_ranges[name] = count
        return detected_params, affected_ranges

    def repair_clamp(self, attacked_sd: dict) -> dict:
        sd = {k: v.clone() for k, v in attacked_sd.items()}
        for name, tensor in sd.items():
            if name not in self._golden_range:
                continue
            low, high = self._golden_range[name]
            sd[name] = torch.clamp(tensor, low, high)
        return sd

    def repair_oracle(self, attacked_sd: dict) -> dict:
        sd = {k: v.clone() for k, v in attacked_sd.items()}
        for name, tensor in attacked_sd.items():
            if name not in self._golden_sd:
                continue
            low, high = self._golden_range[name]
            mask = (tensor < low) | (tensor > high)
            if mask.any():
                sd[name][mask] = self._golden_sd[name][mask]
        return sd


def collect_test_inputs(val_loader, device):
    test_inputs = []
    for batch in val_loader:
        imgs12, desire, traffic, h0, traj_gt = make_supercombo_inputs(batch, device)
        test_inputs.append((imgs12, desire, traffic, h0, traj_gt))
    return test_inputs


def process_sequence(model, test_inputs, device):
    all_bests = []
    for imgs12, desire, traffic, h0, traj_gt in test_inputs:
        B, T, C, H, W = imgs12.shape
        h0_current = h0.clone()
        batch_bests = []
        for t in range(T):
            imgs_t = imgs12[:, t, :, :, :]
            gt_t = traj_gt[:, t, :, :]
            with torch.no_grad():
                out_t = run_inference(model, (imgs_t, desire, traffic, h0_current))
                parsed = parse_output(out_t)
                best = select_best(parsed, gt_t)
            batch_bests.append(best)
            h0_current = out_t[:, -512:]
        all_bests.append(batch_bests)
    return all_bests


def compute_metric_deltas(model_bests, clean_bests, metric):
    delta_vals = []
    for batch_idx in range(len(clean_bests)):
        for t in range(len(clean_bests[batch_idx])):
            delta = compute_delta(model_bests[batch_idx][t], clean_bests[batch_idx][t], metric)
            if not np.isnan(delta):
                delta_vals.append(delta)
    return float(np.nanmean(delta_vals)) if delta_vals else float('nan')


def evaluate(args):
    base = Path(__file__).parent
    clean_path = base / 'openpilot_model' / 'supercombo_torch_weights.pth'
    model_dir = base / 'openpilot_model'
    out_dir = base / 'crossfire_results'
    out_dir.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 65)
    print('STEP 1: Load clean model')
    print('=' * 65)
    clean_sd = torch.load(clean_path, map_location='cpu')
    clean_model = load_model(clean_sd)
    print(f'  {len(clean_sd)} parameter tensors loaded')

    print('\nSTEP 2: Build recurrent val loader')
    val_loader = build_loader(args.val_txt, args.data_root, args.data_length,
                              args.frame_stream_length, 'val', args.batch_size, device)
    print(f'  val loader samples: {len(val_loader.dataset)}')

    print('\nSTEP 3: Collect full recurrent test inputs')
    test_inputs = collect_test_inputs(val_loader, device)
    print(f'  collected {len(test_inputs)} video sample(s)')
    if not test_inputs:
        print('  [WARN] val loader is empty; aborting.')
        return

    print('\nSTEP 4: RANGER setup')
    ranger = RangerDefense(margin=args.range_margin)
    total_params = ranger.setup(clean_sd)
    print(f'  Parameters protected : {total_params}')
    print(f'  Range margin         : {args.range_margin}')

    print('\nSTEP 5: Process clean baselines')
    clean_bests = process_sequence(clean_model, test_inputs, device)
    print(f'  Processed {len(test_inputs)} sample(s), {len(clean_bests[0])} frames each')

    print('\nSTEP 6: Evaluate attacked models')
    model_files = sorted([f for f in model_dir.glob('*.pth') if f.name != 'supercombo_torch_weights.pth'])
    rows = []

    for model_path in model_files:
        model_name = model_path.stem
        print('\n' + '─' * 65)
        print(f'Attacked Model : {model_name}')

        attacked_sd = torch.load(model_path, map_location='cpu')
        attacked_model = load_model(attacked_sd)

        atk_bests = process_sequence(attacked_model, test_inputs, device)
        attack_deltas = {metric: compute_metric_deltas(atk_bests, clean_bests, metric)
                         for metric in ALL_METRICS}

        det_params, affected_ranges = ranger.detect(attacked_sd)
        n_affected = sum(affected_ranges.values())
        detected = len(det_params) > 0

        print(f'  Detection: {"YES ✓" if detected else "NO ✗"} ({n_affected} elements outside range)')
        for metric in ALL_METRICS[:3]:
            print(f'    {metric:30s}: {attack_deltas[metric]:+.4f}')

        sd_clamp = ranger.repair_clamp(attacked_sd)
        m_clamp = load_model(sd_clamp)
        clamp_bests = process_sequence(m_clamp, test_inputs, device)
        clamp_deltas = {metric: compute_metric_deltas(clamp_bests, clean_bests, metric)
                        for metric in ALL_METRICS}

        sd_orc = ranger.repair_oracle(attacked_sd)
        m_orc = load_model(sd_orc)
        orc_bests = process_sequence(m_orc, test_inputs, device)
        oracle_deltas = {metric: compute_metric_deltas(orc_bests, clean_bests, metric)
                         for metric in ALL_METRICS}

        rows.append({
            'model_name': model_name,
            'detected': detected,
            'n_elements_out_of_range': n_affected,
            'attack_deltas': attack_deltas,
            'clamp_repair_deltas': clamp_deltas,
            'oracle_repair_deltas': oracle_deltas,
        })

    print('\n' + '=' * 65)
    print('SUMMARY')
    print('=' * 65)
    if rows:
        det_rate = np.mean([r['detected'] for r in rows]) * 100
        print(f'  Detection rate: {det_rate:.0f}%')
        for metric in ALL_METRICS:
            avg_attack = np.nanmean([r['attack_deltas'][metric] for r in rows])
            avg_clamp = np.nanmean([r['clamp_repair_deltas'][metric] for r in rows])
            avg_oracle = np.nanmean([r['oracle_repair_deltas'][metric] for r in rows])
            print(f'  {metric:30s}: Attack {avg_attack:+.4f} | Clamp {avg_clamp:+.4f} | Oracle {avg_oracle:+.4f}')

    out_path = out_dir / 'ranger_eval_all_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'\nFull results → {out_path}')


def main():
    base = Path(__file__).parent
    p = argparse.ArgumentParser(description='RANGER defense evaluation (recurrent)')
    p.add_argument('--batch_size', type=int, default=8, help='Batch size')
    p.add_argument('--data_length', type=int, default=100, help='Data length per sample')
    p.add_argument('--frame_stream_length', type=int, default=8, help='Frame stream length')
    p.add_argument('--range_margin', type=float, default=0.0, help='Margin added to clean parameter min/max during Ranger clamping')
    p.add_argument('--val_txt', type=str,
                   default=str(base / 'data' / 'comma2k19_val_non_overlap.txt'))
    p.add_argument('--data_root', type=str,
                   default=str(base / 'data') + '/')
    args = p.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
