#!/usr/bin/env python3
"""Shared recurrent bit-search helpers for openpilot 0.8.9."""

from __future__ import annotations

import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "op_089") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "op_089"))

from data import Comma2k19SequenceDataset
from openpilot_torch import OpenPilotModel, load_weights_from_onnx
from tools.common_dataset import SafeDataset, make_loader, normalize_data_root


MODEL_DT = 1.0 / 20.0
LAT_SMOOTH_SECONDS = 0.1
LONG_SMOOTH_SECONDS = 0.3
MIN_LAT_CONTROL_SPEED = 0.3
CONTROL_N = 17
TRAJECTORY_SIZE = 33
DEFAULT_LONG_LAG = 0.15
LON_MPC_STEP = 0.2
ACCEL_MIN = -4.0
ACCEL_MAX = 2.0
COMFORT_BRAKE = 2.5
LEAD_TIME_OFFSETS = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float32)
MAX_CURVATURE_RATES = np.array([0.03762194918267951, 0.003441203371932992], dtype=np.float32)
MAX_CURVATURE_RATE_SPEEDS = np.array([0.0, 35.0], dtype=np.float32)
T_IDXS = 10.0 * (np.arange(33, dtype=np.float32) / 32.0) ** 2
X_IDXS = 192.0 * (np.arange(33, dtype=np.float32) / 32.0) ** 2

PLAN_SIZE = 4955
LANE_GROUP1_SIZE = 4 * 132
LANE_LINES_PROB_SIZE = 8
LANE_GROUP2_SIZE = 2 * 132
LEAD_SIZE = 255
LEAD_PROB_SIZE = 3
DESIRE_STATE_SIZE = 8
META_SIZE = 32
DESIRE_PRED_SIZE = 32
POSE_SIZE = 12
RNN_STATE_SIZE = 512

LANE_GROUP1_START = PLAN_SIZE
LANE_LINES_PROB_START = LANE_GROUP1_START + LANE_GROUP1_SIZE
LANE_GROUP2_START = LANE_LINES_PROB_START + LANE_LINES_PROB_SIZE
LEAD_START = LANE_GROUP2_START + LANE_GROUP2_SIZE
LEAD_PROB_START = LEAD_START + LEAD_SIZE
DESIRE_STATE_START = LEAD_PROB_START + LEAD_PROB_SIZE
META_START = DESIRE_STATE_START + DESIRE_STATE_SIZE
DESIRE_PRED_START = META_START + META_SIZE
POSE_START = DESIRE_PRED_START + DESIRE_PRED_SIZE

SUPPORTED_METRICS = {
    "action.desiredAcceleration",
    "-action.desiredAcceleration",
    "action.desiredCurvature",
    "-action.desiredCurvature",
    "action.curvatureDelta",
    "-action.curvatureDelta",
}

SUPPORTED_TARGET_MODES = {"pseudo_controls", "raw_plan_fit"}

MANTISSA = list(range(0, 23))
MANTISSA_LOW = list(range(0, 7))
EXPONENT = list(range(23, 31))
EXPONENT_SIGN = list(range(23, 32))
SIGN = [31]


def validate_metric(metric: str) -> str:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: {metric}. Supported: {sorted(SUPPORTED_METRICS)}")
    return metric


def metric_base_name(metric: str) -> str:
    return metric[1:] if metric.startswith("-") else metric


def metric_direction(metric: str) -> float:
    return -1.0 if metric.startswith("-") else 1.0


def score_metric_delta(metric: str, baseline_scores: Dict[str, float], flipped_scores: Dict[str, float]) -> float:
    base = metric_base_name(metric)
    return metric_direction(metric) * (float(flipped_scores[base]) - float(baseline_scores[base]))


def score_metric_percent_delta(
    metric: str,
    baseline_scores: Dict[str, float],
    flipped_scores: Dict[str, float],
    *,
    eps: float = 1e-6,
) -> float:
    base = metric_base_name(metric)
    baseline_value = float(baseline_scores[base])
    flipped_value = float(flipped_scores[base])
    abs_delta = flipped_value - baseline_value
    denom = max(abs(baseline_value), float(eps))
    return metric_direction(metric) * (abs_delta / denom) * 100.0


def validate_target_mode(mode: str) -> str:
    if mode not in SUPPORTED_TARGET_MODES:
        raise ValueError(f"Unsupported target mode: {mode}. Supported: {sorted(SUPPORTED_TARGET_MODES)}")
    return mode


def parse_bitset(spec: str) -> List[int]:
    s = spec.strip().lower()
    if s == "mantissa":
        return MANTISSA
    if s == "mantissa_low":
        return MANTISSA_LOW
    if s == "exponent":
        return EXPONENT
    if s in {"exponent_sign", "exponent&sign", "exp_sign"}:
        return EXPONENT_SIGN
    if s == "sign":
        return SIGN
    if s in {"full", "all"}:
        return list(range(32))
    if s.startswith(">="):
        start = int(s[2:])
        return list(range(max(0, start), 32))
    if s.startswith(">"):
        start = int(s[1:]) + 1
        return list(range(max(0, start), 32))
    if s.startswith("<="):
        end = int(s[2:])
        return list(range(0, min(31, end) + 1))
    if s.startswith("<"):
        end = int(s[1:]) - 1
        return list(range(0, min(31, end) + 1))
    if ":" in s:
        lo_s, hi_s = s.split(":", 1)
        lo = int(lo_s)
        hi = int(hi_s)
        if lo > hi:
            raise ValueError(f"Invalid bit range: {spec}")
        return list(range(max(0, lo), min(31, hi) + 1))
    bits = [int(x) for x in s.split(",") if x.strip()]
    if not bits:
        raise ValueError(f"Invalid bitset: {spec}")
    for bit in bits:
        if bit < 0 or bit > 31:
            raise ValueError(f"Bit {bit} out of range [0, 31]")
    return bits


def load_candidates_json(path: str, topn: Optional[int] = None) -> List[Tuple[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = []
    for key in ("summary", "flips", "plan", "ranked", "records"):
        if isinstance(payload.get(key), list):
            rows = payload[key]
            break
    if not rows and isinstance(payload, list):
        rows = payload
    if not rows:
        raise ValueError(f"No candidate rows found in {path}")

    out: List[Tuple[str, int]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        flat = row.get("index_flat", row.get("flat", row.get("index")))
        if name is None or flat is None:
            continue
        out.append((str(name), int(flat)))
        if topn is not None and len(out) >= topn:
            break
    if not out:
        raise ValueError(f"No valid candidates found in {path}")
    return out


def load_flip_rows_json(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = []
    for key in ("flips", "plan", "summary", "ranked", "records"):
        if isinstance(payload.get(key), list):
            rows = payload[key]
            break
    if not rows and isinstance(payload, list):
        rows = payload

    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        flat = row.get("index_flat", row.get("flat", row.get("index")))
        bit = row.get("bit")
        if name is None or flat is None or bit is None:
            continue
        out.append(
            {
                "name": str(name),
                "index_flat": int(flat),
                "bit": int(bit),
                "score": float(row.get("score", 0.0)),
            }
        )
        if limit is not None and len(out) >= limit:
            break
    if not out:
        raise ValueError(f"No valid flips found in {path}")
    return out


def build_eval_loader(
    data_root: str,
    val_index: str,
    batch_size: int,
    device: torch.device,
    *,
    num_workers: int = 0,
    inner_progress: bool = False,
):
    data_root = normalize_data_root(data_root)
    dataset = Comma2k19SequenceDataset(
        val_index,
        data_root,
        "val",
        use_memcache=False,
        inner_progress=bool(inner_progress),
    )
    dataset = SafeDataset(dataset)
    return make_loader(dataset, batch_size=batch_size, shuffle=False, device=device, num_workers=num_workers)


def cache_eval_batches(loader: Iterable[Dict[str, torch.Tensor]], num_batches: int) -> List[Dict[str, torch.Tensor]]:
    cached: List[Dict[str, torch.Tensor]] = []
    for idx, batch in enumerate(loader):
        if idx >= num_batches:
            break
        cached.append(
            {
                key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
        )
    if not cached:
        raise RuntimeError("Validation loader produced no batches.")
    return cached


def prepare_sequence_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    recurrent_num: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_imgs = batch["seq_input_img"].float()
    seq_gt = batch["seq_future_poses"].float()

    if recurrent_num > 0:
        seq_len = min(recurrent_num, seq_imgs.shape[1])
        seq_imgs = seq_imgs[:, :seq_len]
        seq_gt = seq_gt[:, :seq_len]

    if seq_imgs.shape[2] == 6:
        seq_imgs = torch.cat([seq_imgs, seq_imgs], dim=2)
    if seq_imgs.shape[2] != 12:
        raise ValueError(f"Expected 12-channel recurrent input, got {tuple(seq_imgs.shape)}")

    return seq_imgs.to(device, non_blocking=True), seq_gt.to(device, non_blocking=True)


def load_model(ckpt: str, device: torch.device) -> OpenPilotModel:
    model = OpenPilotModel().to(device)
    if not ckpt:
        model.eval()
        return model
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if ckpt.endswith(".onnx"):
        load_weights_from_onnx(model, ckpt)
        model.to(device)
        model.eval()
        return model

    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        cleaned = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
        model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model


def live_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return dict(model.named_parameters())


@torch.no_grad()
def flip_scalar_bit_fast_(param: torch.Tensor, flat_idx: int, bit: int) -> Tuple[float, float]:
    assert param.dtype == torch.float32
    assert 0 <= int(bit) <= 31
    flat = param.data.view(-1)
    old_val = float(flat[flat_idx].detach().cpu().numpy().astype(np.float32).item())
    old_bits = np.frombuffer(np.float32(old_val).tobytes(), dtype=np.uint32)[0]
    new_bits = old_bits ^ (np.uint32(1) << np.uint32(bit))
    new_val = np.frombuffer(np.uint32(new_bits).tobytes(), dtype=np.float32)[0]
    flat[flat_idx] = torch.tensor(new_val, dtype=torch.float32, device=param.device)
    return old_val, float(new_val)


def make_autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=(enabled and device.type == "cuda"))


def decode_best_plan(output: torch.Tensor, gt_frame: torch.Tensor | None = None) -> torch.Tensor:
    bsz = output.shape[0]
    plan = output[:, : 5 * 991].view(bsz, 5, 991)
    plan_modes = plan[:, :, :-1].view(bsz, 5, 2, 33, 15)
    plan_mean = plan_modes[:, :, 0, :, :]
    plan_logits = plan[:, :, -1]
    best_idx = plan_logits.argmax(dim=1)
    rows = torch.arange(bsz, device=output.device)
    return plan_mean[rows, best_idx]


def _smooth_value_np(val: np.ndarray, prev_val: np.ndarray, tau: float, dt: float = MODEL_DT) -> np.ndarray:
    alpha = 1.0 - np.exp(-dt / tau) if tau > 0 else 1.0
    return alpha * val + (1.0 - alpha) * prev_val


def _interp_rows(x: float, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return np.array([np.interp(x, xp, row) for row in fp], dtype=np.float32)


def _interp_rows_var(xs: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return np.array([np.interp(float(xs[i]), xp, fp[i]) for i in range(fp.shape[0])], dtype=np.float32)


def _estimate_curvature_rate(curvatures: np.ndarray, t_idxs: np.ndarray) -> np.ndarray:
    if curvatures.shape[1] < 2:
        return np.zeros((curvatures.shape[0],), dtype=np.float32)
    rates = np.gradient(curvatures, t_idxs, axis=1)
    return rates[:, 0].astype(np.float32, copy=False)


def _estimate_curvature_rates(curvatures: np.ndarray, t_idxs: np.ndarray) -> np.ndarray:
    if curvatures.shape[1] < 2:
        return np.zeros_like(curvatures, dtype=np.float32)
    return np.gradient(curvatures, t_idxs, axis=1).astype(np.float32, copy=False)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _plan_t_from_best_plan_x(best_plan_np: np.ndarray) -> np.ndarray:
    bsz = best_plan_np.shape[0]
    plan_x = best_plan_np[:, :, 0]
    plan_t = np.full((bsz, TRAJECTORY_SIZE), T_IDXS[-1], dtype=np.float32)
    plan_t[:, 0] = 0.0
    for b in range(bsz):
        tidx = 0
        for xidx in range(1, TRAJECTORY_SIZE):
            while tidx < TRAJECTORY_SIZE - 1 and plan_x[b, tidx + 1] < X_IDXS[xidx]:
                tidx += 1
            current_x = plan_x[b, tidx]
            next_x = plan_x[b, min(tidx + 1, TRAJECTORY_SIZE - 1)]
            if next_x < X_IDXS[xidx] or next_x <= current_x:
                plan_t[b, xidx] = T_IDXS[-1]
                continue
            p = (X_IDXS[xidx] - current_x) / (next_x - current_x)
            plan_t[b, xidx] = np.float32(p * T_IDXS[min(tidx + 1, TRAJECTORY_SIZE - 1)] + (1.0 - p) * T_IDXS[tidx])
    return plan_t


def _decode_output_components(output: torch.Tensor) -> Dict[str, np.ndarray]:
    out_np = output.detach().cpu().numpy().astype(np.float32, copy=False)
    bsz = out_np.shape[0]

    plan = out_np[:, :PLAN_SIZE].reshape(bsz, 5, 991)
    plan_modes = plan[:, :, :-1].reshape(bsz, 5, 2, TRAJECTORY_SIZE, 15)
    plan_mean = plan_modes[:, :, 0, :, :]
    plan_logits = plan[:, :, -1]
    best_idx = plan_logits.argmax(axis=1)
    rows = np.arange(bsz)
    best_plan = plan_mean[rows, best_idx]

    lane_probs_raw = out_np[:, LANE_LINES_PROB_START : LANE_LINES_PROB_START + LANE_LINES_PROB_SIZE]
    lane_probs = _sigmoid_np(lane_probs_raw[:, 1::2]).astype(np.float32, copy=False)

    lane_group1 = out_np[:, LANE_GROUP1_START : LANE_GROUP1_START + LANE_GROUP1_SIZE]
    lane_group1 = lane_group1.reshape(bsz, 4, 2, TRAJECTORY_SIZE, 2)
    inner_left = lane_group1[:, 1]
    inner_right = lane_group1[:, 2]

    lead = out_np[:, LEAD_START : LEAD_START + LEAD_SIZE].reshape(bsz, 5, 51)
    lead_sel_logits = lead[:, :, 48:51]
    lead_best_idx = lead_sel_logits[:, :, 0].argmax(axis=1)
    lead_best = lead[rows, lead_best_idx]
    lead_mean = lead_best[:, :24].reshape(bsz, 6, 4)
    lead_prob = _sigmoid_np(out_np[:, LEAD_PROB_START : LEAD_PROB_START + LEAD_PROB_SIZE]).astype(np.float32, copy=False)

    pose = out_np[:, POSE_START : POSE_START + POSE_SIZE]
    pose_mean = pose[:, :6]

    return {
        "best_plan": best_plan,
        "plan_t": _plan_t_from_best_plan_x(best_plan),
        "lane_probs": lane_probs,
        "lane_inner_left": inner_left,
        "lane_inner_right": inner_right,
        "lead_mean": lead_mean,
        "lead_prob": lead_prob,
        "pose_mean": pose_mean,
    }


def compute_action_targets_from_best_plan_fit(
    best_plan: torch.Tensor,
    prev_desired_accel: np.ndarray,
    prev_desired_curvature: np.ndarray,
    *,
    action_t: float = MODEL_DT,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    del prev_desired_accel, prev_desired_curvature

    plan_np = best_plan.detach().cpu().numpy().astype(np.float32, copy=False)
    speeds = plan_np[:, :, 3]
    accels = plan_np[:, :, 6]
    yaws = plan_np[:, :, 11]
    yaw_rates = plan_np[:, :, 14]

    v_now = speeds[:, 0]
    a_now = accels[:, 0]
    v_abs = np.maximum(np.abs(v_now), 1e-1).astype(np.float32, copy=False)

    v_target = _interp_rows(action_t, T_IDXS, speeds)
    desired_accel = ((2.0 * (v_target - v_now) / action_t) - a_now).astype(np.float32, copy=False)

    current_curvature = (yaw_rates[:, 0] / v_abs).astype(np.float32, copy=False)
    psi_target = _interp_rows(action_t, T_IDXS, yaws)
    desired_curvature = (2.0 * (psi_target / (v_abs * action_t)) - current_curvature).astype(np.float32, copy=False)
    desired_curvature = np.where(np.abs(v_now) > MIN_LAT_CONTROL_SPEED, desired_curvature, current_curvature).astype(
        np.float32, copy=False
    )
    curvature_delta = (desired_curvature - current_curvature).astype(np.float32, copy=False)

    metrics = {
        "action.desiredAcceleration": desired_accel,
        "action.desiredCurvature": desired_curvature,
        "action.curvatureDelta": curvature_delta,
        "action.desiredCurvatureRate": np.zeros_like(desired_curvature, dtype=np.float32),
        "plan.currentCurvature": current_curvature,
    }
    return metrics, desired_accel, desired_curvature


def compute_action_targets_from_output(
    output: torch.Tensor,
    prev_desired_accel: np.ndarray,
    prev_desired_curvature: np.ndarray,
    *,
    steer_actuator_delay: float = 0.0,
    target_mode: str = "pseudo_controls",
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    target_mode = validate_target_mode(target_mode)
    decoded = _decode_output_components(output)
    plan_np = decoded["best_plan"]
    if target_mode == "raw_plan_fit":
        best_plan_t = torch.from_numpy(plan_np)
        return compute_action_targets_from_best_plan_fit(
            best_plan_t,
            prev_desired_accel,
            prev_desired_curvature,
        )

    plan_t = decoded["plan_t"]
    lane_probs = decoded["lane_probs"]
    lane_inner_left = decoded["lane_inner_left"]
    lane_inner_right = decoded["lane_inner_right"]
    lead_mean = decoded["lead_mean"]
    lead_prob = decoded["lead_prob"][:, 0]
    pose_mean = decoded["pose_mean"]

    long_plan_speeds = plan_np[:, :CONTROL_N, 3]
    long_plan_accels = plan_np[:, :CONTROL_N, 6]
    path_xyz = plan_np[:, :, :3].copy()
    plan_yaw = plan_np[:, :, 11]

    pose_vx = pose_mean[:, 0]
    pose_yaw_rate = pose_mean[:, 5]
    plan_v0 = long_plan_speeds[:, 0]
    v_now = np.where(np.isfinite(pose_vx) & (np.abs(pose_vx) < 100.0), pose_vx, plan_v0).astype(np.float32, copy=False)
    a_now = long_plan_accels[:, 0]
    v_abs = np.maximum(np.abs(v_now), 1e-1).astype(np.float32, copy=False)

    # Pseudo longitudinal planner: combine cruise-like target from the best plan
    # with a lead-constrained cap, then feed the result through LongControl's target formula.
    cruise_v_target = _interp_rows(DEFAULT_LONG_LAG, T_IDXS[:CONTROL_N], long_plan_speeds)
    cruise_a_target = (2.0 * (cruise_v_target - v_now) / DEFAULT_LONG_LAG) - a_now

    lead_x = lead_mean[:, :, 0]
    lead_v = lead_mean[:, :, 2]
    lead_x_now = _interp_rows(LON_MPC_STEP, LEAD_TIME_OFFSETS, lead_x)
    lead_v_now = _interp_rows(LON_MPC_STEP, LEAD_TIME_OFFSETS, lead_v)
    desired_gap = (5.0 + 1.8 * np.maximum(v_abs, 0.0)).astype(np.float32, copy=False)
    free_distance = np.maximum(lead_x_now - desired_gap, 0.0).astype(np.float32, copy=False)
    lead_safe_v = np.sqrt(np.maximum(lead_v_now * lead_v_now + 2.0 * COMFORT_BRAKE * free_distance, 0.0)).astype(
        np.float32, copy=False
    )
    lead_v_target = np.minimum(cruise_v_target, lead_safe_v).astype(np.float32, copy=False)
    lead_a_target = (2.0 * (lead_v_target - v_now) / DEFAULT_LONG_LAG) - a_now
    a_target = cruise_a_target + lead_prob * (np.minimum(cruise_a_target, lead_a_target) - cruise_a_target)
    desired_accel = np.clip(a_target, ACCEL_MIN, ACCEL_MAX).astype(np.float32, copy=False)

    # Pseudo lateral planner: blend the predicted path with lane center estimates
    # using a simplified lane-planner weighting, then derive psis/curvatures from that path.
    lll_y = lane_inner_left[:, 0, :, 0]
    rll_y = lane_inner_right[:, 0, :, 0]
    lll_std = np.exp(lane_inner_left[:, 1, 0, 0]).astype(np.float32, copy=False)
    rll_std = np.exp(lane_inner_right[:, 1, 0, 0]).astype(np.float32, copy=False)
    l_prob = lane_probs[:, 1].copy()
    r_prob = lane_probs[:, 2].copy()
    width_pts = rll_y - lll_y

    for t_check in (0.0, 1.5, 3.0):
        width_at_t = _interp_rows_var(t_check * (v_abs + 7.0), X_IDXS, width_pts)
        mod = np.interp(width_at_t, [4.0, 5.0], [1.0, 0.0]).astype(np.float32, copy=False)
        l_prob *= mod
        r_prob *= mod
    l_prob *= np.interp(lll_std, [0.15, 0.3], [1.0, 0.0]).astype(np.float32, copy=False)
    r_prob *= np.interp(rll_std, [0.15, 0.3], [1.0, 0.0]).astype(np.float32, copy=False)

    speed_lane_width = np.interp(v_abs, [0.0, 31.0], [2.8, 3.5]).astype(np.float32, copy=False)
    current_lane_width = np.abs(width_pts[:, 0]).astype(np.float32, copy=False)
    lane_certainty = (l_prob * r_prob).astype(np.float32, copy=False)
    lane_width = (lane_certainty * current_lane_width + (1.0 - lane_certainty) * speed_lane_width).astype(
        np.float32, copy=False
    )
    clipped_lane_width = np.minimum(4.0, lane_width).astype(np.float32, copy=False)
    path_from_left = lll_y + clipped_lane_width[:, None] / 2.0
    path_from_right = rll_y - clipped_lane_width[:, None] / 2.0
    d_prob = (l_prob + r_prob - l_prob * r_prob).astype(np.float32, copy=False)
    lane_path_y = (l_prob[:, None] * path_from_left + r_prob[:, None] * path_from_right) / (
        l_prob[:, None] + r_prob[:, None] + 1e-4
    )
    lane_path_y_interp = np.array(
        [np.interp(T_IDXS, plan_t[i], lane_path_y[i]) for i in range(plan_np.shape[0])],
        dtype=np.float32,
    )
    path_xyz[:, :, 1] = d_prob[:, None] * lane_path_y_interp + (1.0 - d_prob[:, None]) * path_xyz[:, :, 1]

    sample_dists = (v_abs[:, None] * T_IDXS[:CONTROL_N][None, :]).astype(np.float32, copy=False)
    path_norm = np.linalg.norm(path_xyz, axis=2).astype(np.float32, copy=False)
    d_path_y = path_xyz[:, :, 1]
    psis = np.array(
        [np.interp(sample_dists[i], path_norm[i], plan_yaw[i]) for i in range(plan_np.shape[0])],
        dtype=np.float32,
    )
    d_path_y_pts = np.array(
        [np.interp(sample_dists[i], path_norm[i], d_path_y[i]) for i in range(plan_np.shape[0])],
        dtype=np.float32,
    )
    curvatures = (_estimate_curvature_rates(psis, T_IDXS[:CONTROL_N]) / v_abs[:, None]).astype(np.float32, copy=False)
    current_curvature = (pose_yaw_rate / v_abs).astype(np.float32, copy=False)
    curvatures[:, 0] = current_curvature
    curvature_rates = _estimate_curvature_rates(curvatures, T_IDXS[:CONTROL_N])
    desired_curvature_rate = curvature_rates[:, 0]

    delay = float(steer_actuator_delay) + 0.2
    psi_target = _interp_rows(delay, T_IDXS[:CONTROL_N], psis)
    desired_curvature = current_curvature + 2.0 * (psi_target / (v_abs * delay) - current_curvature)

    max_curvature_rate = np.interp(v_abs, MAX_CURVATURE_RATE_SPEEDS, MAX_CURVATURE_RATES).astype(np.float32, copy=False)
    safe_desired_curvature_rate = np.clip(desired_curvature_rate, -max_curvature_rate, max_curvature_rate)
    safe_desired_curvature = np.clip(
        desired_curvature,
        current_curvature - max_curvature_rate / MODEL_DT,
        current_curvature + max_curvature_rate / MODEL_DT,
    ).astype(np.float32, copy=False)
    safe_desired_curvature = np.where(np.abs(v_now) > MIN_LAT_CONTROL_SPEED, safe_desired_curvature, current_curvature).astype(
        np.float32, copy=False
    )

    curvature_delta = (safe_desired_curvature - current_curvature).astype(np.float32, copy=False)

    metrics = {
        "action.desiredAcceleration": desired_accel,
        "action.desiredCurvature": safe_desired_curvature,
        "action.curvatureDelta": curvature_delta,
        "action.desiredCurvatureRate": safe_desired_curvature_rate.astype(np.float32, copy=False),
        "plan.currentCurvature": current_curvature.astype(np.float32, copy=False),
        "plan.pathLateralOffset": d_path_y_pts[:, 0].astype(np.float32, copy=False),
    }
    return metrics, desired_accel, safe_desired_curvature


@torch.no_grad()
def evaluate_recurrent_metrics(
    model: torch.nn.Module,
    cached_batches: Sequence[Dict[str, torch.Tensor]],
    device: torch.device,
    *,
    recurrent_num: int,
    use_amp: bool,
    steer_actuator_delay: float = 0.0,
    target_mode: str = "pseudo_controls",
) -> Dict[str, float]:
    target_mode = validate_target_mode(target_mode)
    totals: Dict[str, List[np.ndarray]] = {
        "action.desiredAcceleration": [],
        "action.desiredCurvature": [],
        "action.curvatureDelta": [],
    }

    model.eval()
    for batch in cached_batches:
        seq_imgs12, seq_gt = prepare_sequence_batch(batch, device, recurrent_num)
        bsz = seq_imgs12.shape[0]
        seq_len = seq_imgs12.shape[1]

        desire = torch.zeros((bsz, 8), device=device)
        traffic = torch.tensor([[1.0, 0.0]], device=device).repeat(bsz, 1)
        hidden = torch.zeros((bsz, 512), device=device)

        prev_desired_accel = np.zeros((bsz,), dtype=np.float32)
        prev_desired_curvature = np.zeros((bsz,), dtype=np.float32)

        for step in range(seq_len):
            imgs12 = seq_imgs12[:, step]
            with make_autocast(device, use_amp):
                output = model(imgs12, desire, traffic, hidden)
            hidden = output[:, -512:].detach()
            action_targets, prev_desired_accel, prev_desired_curvature = compute_action_targets_from_output(
                output,
                prev_desired_accel,
                prev_desired_curvature,
                steer_actuator_delay=steer_actuator_delay,
                target_mode=target_mode,
            )
            for key in totals:
                totals[key].append(action_targets[key])

    scores: Dict[str, float] = {}
    for key, values in totals.items():
        if not values:
            scores[key] = float("nan")
            continue
        merged = np.concatenate(values, axis=0)
        finite = merged[np.isfinite(merged)]
        scores[key] = float(np.mean(finite)) if finite.size else float("nan")
    return scores
