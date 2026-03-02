#!/usr/bin/env python3
"""Search important bits for openpilot 0.10.3 split ONNX models.

Pipeline:
1) Auto-build comma2k19 split files when missing.
2) Build mini train/val loaders from Comma2k19SequenceDataset.
3) Evaluate end-to-end image -> vision -> policy trajectory output.
4) Select candidate scalar weights by absolute value or gradient magnitude.
5) Rank bit flips progressively by delta metric on final policy output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
import cv2
from onnx import TensorProto, numpy_helper
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import Comma2k19SequenceDataset
from tools.common_dataset import SafeDataset, ensure_non_overlap_split_files, make_loader, normalize_data_root


@dataclass
class CachedBatch:
    seq_imgs12: np.ndarray
    seq_gt: torch.Tensor


def make_session(model_bytes_or_path: Any, providers: Sequence[str]) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model_bytes_or_path, sess_options=so, providers=list(providers))


def is_cuda_runtime_error(exc: BaseException) -> bool:
    msg = str(exc).upper()
    return ("CUDA" in msg) or ("CUBLAS" in msg) or ("CUDNN" in msg)


def sanitize_onnx_for_onnx2torch(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fix ONNX optional-input placeholders that may break onnx2torch conversion."""
    m = copy.deepcopy(model)
    for node in m.graph.node:
        if node.op_type != "Clip":
            continue
        inputs = list(node.input)
        while inputs and (inputs[-1] == ""):
            inputs.pop()
        del node.input[:]
        node.input.extend(inputs)
    return m


def tensor_hash_key(arr: np.ndarray) -> Tuple[str, Tuple[int, ...], str]:
    arr_c = np.ascontiguousarray(arr)
    return str(arr_c.dtype), tuple(arr_c.shape), hashlib.sha1(arr_c.tobytes()).hexdigest()


def patch_onnx2torch_cuda_matmul() -> None:
    """Patch onnx2torch MatMul for CUDA stability and dtype mismatch robustness."""
    try:
        from onnx2torch.node_converters.matmul import OnnxMatMul
    except Exception:
        return

    if getattr(OnnxMatMul, "_stafi_cuda_patch_applied", False):
        return

    def safe_forward(self: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            if x.is_floating_point() and y.is_floating_point() and (x.dtype != y.dtype):
                y = y.to(x.dtype)
            if x.is_cuda and y.is_cuda and x.is_floating_point() and y.is_floating_point() and x.dtype in (torch.float16, torch.bfloat16):
                out32 = torch.matmul(x.float(), y.float())
                return out32.to(x.dtype)
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.is_cuda and y.is_cuda and x.dim() > 2 and y.dim() > 2:
            x2 = x.contiguous().reshape(-1, x.shape[-2], x.shape[-1])
            y2 = y.contiguous().reshape(-1, y.shape[-2], y.shape[-1])
            outs = [x2[i] @ y2[i] for i in range(x2.shape[0])]
            z = torch.stack(outs, dim=0)
            return z.reshape(*x.shape[:-2], x.shape[-2], y.shape[-1])
        return torch.matmul(x, y)

    OnnxMatMul.forward = safe_forward
    OnnxMatMul._stafi_cuda_patch_applied = True


def patch_torch_linear_dtype_mismatch() -> None:
    """Patch torch.nn.Linear to tolerate floating dtype mismatch at runtime."""
    if getattr(torch.nn.Linear, "_stafi_dtype_patch_applied", False):
        return

    def safe_linear_forward(self: torch.nn.Linear, input_tensor: torch.Tensor) -> torch.Tensor:
        x = input_tensor
        if (
            isinstance(x, torch.Tensor)
            and isinstance(self.weight, torch.Tensor)
            and x.is_floating_point()
            and self.weight.is_floating_point()
            and x.dtype != self.weight.dtype
        ):
            x = x.to(self.weight.dtype)
        return F.linear(x, self.weight, self.bias)

    torch.nn.Linear.forward = safe_linear_forward  # type: ignore[assignment]
    torch.nn.Linear._stafi_dtype_patch_applied = True  # type: ignore[attr-defined]


def convert_onnx_to_torch_with_compat(model: onnx.ModelProto) -> torch.nn.Module:
    """Convert ONNX to torch, auto-aliasing unsupported higher opset versions."""
    from onnx2torch import convert as onnx2torch_convert
    from onnx2torch.node_converters import registry
    from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node

    msg_re = re.compile(
        r"OperationDescription\(domain='(?P<domain>.*)', operation_type='(?P<op>.*)', version=(?P<ver>\d+)\)"
    )

    gelu_desc = registry.OperationDescription(domain="", operation_type="Gelu", version=20)
    if gelu_desc not in registry._CONVERTER_REGISTRY:  # pylint: disable=protected-access
        class OnnxGelu(torch.nn.Module):
            def __init__(self, approximate: str = "none"):
                super().__init__()
                self.approximate = approximate if approximate in ("none", "tanh") else "none"

            def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
                return F.gelu(input_tensor, approximate=self.approximate)

        def gelu_converter(node: Any, graph: Any) -> OperationConverterResult:  # noqa: ANN401
            approximate = node.attributes.get("approximate", "none")
            if isinstance(approximate, bytes):
                approximate = approximate.decode("utf-8")
            return OperationConverterResult(
                torch_module=OnnxGelu(str(approximate)),
                onnx_mapping=onnx_mapping_from_node(node=node),
            )

        registry._CONVERTER_REGISTRY[gelu_desc] = gelu_converter  # pylint: disable=protected-access
        print("[Gradient] onnx2torch custom converter registered: Gelu@20")

    reshape_desc = registry.OperationDescription(domain="", operation_type="Reshape", version=19)
    if reshape_desc not in registry._CONVERTER_REGISTRY:  # pylint: disable=protected-access
        class OnnxReshapeCompat(torch.nn.Module):
            def __init__(self, allowzero: bool):
                super().__init__()
                self.allowzero = bool(allowzero)

            def forward(self, input_tensor: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
                shape_list = shape.detach().cpu().tolist()
                if not self.allowzero:
                    shape_list = [
                        (input_tensor.shape[i] if int(dim_size) == 0 else int(dim_size))
                        for i, dim_size in enumerate(shape_list)
                    ]
                else:
                    shape_list = [int(dim_size) for dim_size in shape_list]
                return torch.reshape(input_tensor, torch.Size(shape_list))

        def reshape_converter(node: Any, graph: Any) -> OperationConverterResult:  # noqa: ANN401
            allowzero = int(node.attributes.get("allowzero", 0)) == 1
            return OperationConverterResult(
                torch_module=OnnxReshapeCompat(allowzero=allowzero),
                onnx_mapping=onnx_mapping_from_node(node=node),
            )

        registry._CONVERTER_REGISTRY[reshape_desc] = reshape_converter  # pylint: disable=protected-access
        print("[Gradient] onnx2torch custom converter registered: Reshape@19")

    def _register_reduce_dynamic(op_type: str, version: int) -> None:
        desc = registry.OperationDescription(domain="", operation_type=op_type, version=version)
        if desc in registry._CONVERTER_REGISTRY:  # pylint: disable=protected-access
            return

        class OnnxReduceDynamic(torch.nn.Module):
            def __init__(self, operation_type: str, keepdims: int, noop_with_empty_axes: int):
                super().__init__()
                self.operation_type = operation_type
                self.keepdims = bool(keepdims)
                self.noop_with_empty_axes = bool(noop_with_empty_axes)

            def forward(self, input_tensor: torch.Tensor, axes: Optional[torch.Tensor] = None) -> torch.Tensor:
                if axes is None or axes.numel() == 0:
                    if self.noop_with_empty_axes:
                        return input_tensor
                    fixed_axes = list(range(input_tensor.dim()))
                else:
                    fixed_axes = torch.sort(axes).values.tolist()

                if self.operation_type == "ReduceMean":
                    return torch.mean(input_tensor, dim=fixed_axes, keepdim=self.keepdims)
                if self.operation_type == "ReduceL2":
                    return torch.norm(input_tensor, p=2, dim=fixed_axes, keepdim=self.keepdims)
                raise NotImplementedError(f"Unsupported reduce op: {self.operation_type}")

        def reduce_converter(node: Any, graph: Any) -> OperationConverterResult:  # noqa: ANN401
            keepdims = int(node.attributes.get("keepdims", 1))
            noop_with_empty_axes = int(node.attributes.get("noop_with_empty_axes", 0))
            return OperationConverterResult(
                torch_module=OnnxReduceDynamic(
                    operation_type=op_type,
                    keepdims=keepdims,
                    noop_with_empty_axes=noop_with_empty_axes,
                ),
                onnx_mapping=onnx_mapping_from_node(node=node),
            )

        registry._CONVERTER_REGISTRY[desc] = reduce_converter  # pylint: disable=protected-access
        print(f"[Gradient] onnx2torch custom converter registered: {op_type}@{version}")

    _register_reduce_dynamic("ReduceMean", 18)
    _register_reduce_dynamic("ReduceL2", 18)

    for _ in range(32):
        try:
            return onnx2torch_convert(model)
        except NotImplementedError as exc:
            m = msg_re.search(str(exc))
            if m is None:
                raise
            domain = m.group("domain")
            op_type = m.group("op")
            ver = int(m.group("ver"))

            versions = sorted(
                d.version
                for d in registry._CONVERTER_REGISTRY.keys()  # pylint: disable=protected-access
                if (d.domain == domain and d.operation_type == op_type and d.version < ver)
            )
            if not versions:
                raise RuntimeError(
                    f"onnx2torch missing converter for {op_type}@{ver} and no lower-version fallback exists."
                ) from exc

            base_ver = versions[-1]
            new_desc = registry.OperationDescription(domain=domain, operation_type=op_type, version=ver)
            old_desc = registry.OperationDescription(domain=domain, operation_type=op_type, version=base_ver)
            registry._CONVERTER_REGISTRY[new_desc] = registry._CONVERTER_REGISTRY[old_desc]  # pylint: disable=protected-access
            print(f"[Gradient] onnx2torch compat alias: {op_type}@{ver} -> @{base_ver}")

    raise RuntimeError("onnx2torch conversion failed after too many compatibility alias retries.")


def resolve_providers(provider_mode: str) -> List[str]:
    avail = set(ort.get_available_providers())
    if provider_mode == "cpu":
        return ["CPUExecutionProvider"]
    if provider_mode == "cuda":
        if "CUDAExecutionProvider" not in avail:
            raise RuntimeError(
                "CUDAExecutionProvider is not available in current onnxruntime. "
                "Install onnxruntime-gpu or use --provider cpu/auto."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def parse_bitset(name: str) -> List[int]:
    key = name.strip().lower()
    if key in ("all", "full"):
        return list(range(16))
    if key == "mantissa":
        return list(range(10))
    if key == "exponent":
        return list(range(10, 15))
    if key in ("exponent_sign", "exp_sign", "exponent&sign"):
        return list(range(10, 16))
    if key == "sign":
        return [15]
    bits = [int(x) for x in key.split(",") if x.strip()]
    for b in bits:
        if b < 0 or b > 15:
            raise ValueError(f"fp16 bit out of range: {b}")
    return bits


def canonical_bitset_mode(name: str) -> str:
    key = name.strip().lower()
    if key in ("all", "full"):
        return "all"
    if key == "mantissa":
        return "mantissa"
    if key == "exponent":
        return "exp"
    if key in ("exponent_sign", "exp_sign", "exponent&sign"):
        return "sign_exp"
    if key == "sign":
        return "sign"
    return "custom"


def requested_model_keys(target_model: str) -> List[str]:
    if target_model == "vision":
        return ["vision"]
    if target_model == "policy":
        return ["policy"]
    return ["vision", "policy"]


def ort_type_to_numpy_dtype(type_str: str) -> np.dtype:
    if "uint8" in type_str:
        return np.uint8
    if "float16" in type_str:
        return np.float16
    return np.float32


def cast_vision_input_for_ort(arr: np.ndarray, vision_input_dtype: np.dtype) -> np.ndarray:
    if vision_input_dtype != np.uint8:
        return arr.astype(vision_input_dtype, copy=False)

    # Comma2k19 loader returns normalized RGB-like channels; approximate back to uint8 for uint8 ONNX inputs.
    rgb_mean = np.array([0.3890, 0.3937, 0.3851], dtype=np.float32)
    rgb_std = np.array([0.2172, 0.2141, 0.2209], dtype=np.float32)

    x = arr.astype(np.float32, copy=False)
    c = x.shape[1]
    if c % 3 == 0:
        mean = np.tile(rgb_mean, c // 3)[None, :, None, None]
        std = np.tile(rgb_std, c // 3)[None, :, None, None]
        x = (x * std + mean) * 255.0
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def shape_dim(shape: Sequence[Any], axis: int, fallback: int) -> int:
    if axis >= len(shape):
        return fallback
    v = shape[axis]
    if isinstance(v, int) and v > 0:
        return int(v)
    return fallback


def _rgb_chw_norm_to_uint8_hwc(rgb_chw: np.ndarray) -> np.ndarray:
    """Convert normalized CHW RGB (3x128x256) to uint8 HWC resized to 256x512."""
    rgb_mean = np.array([0.3890, 0.3937, 0.3851], dtype=np.float32)[:, None, None]
    rgb_std = np.array([0.2172, 0.2141, 0.2209], dtype=np.float32)[:, None, None]
    x = rgb_chw.astype(np.float32, copy=False)
    x = (x * rgb_std + rgb_mean) * 255.0
    x = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    x_hwc = np.transpose(x, (1, 2, 0))
    # openpilot model input path is based on 256x512 frames.
    return cv2.resize(x_hwc, (512, 256), interpolation=cv2.INTER_LINEAR)


def _rgb_hwc_to_yuv420_6ch(rgb_hwc: np.ndarray) -> np.ndarray:
    """Pack a 256x512 RGB frame into openpilot-style YUV420 6-plane tensor (6x128x256)."""
    if rgb_hwc.shape[0] != 256 or rgb_hwc.shape[1] != 512 or rgb_hwc.shape[2] != 3:
        raise ValueError(f"Expected RGB frame shape (256,512,3), got {rgb_hwc.shape}")
    yuv = cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2YUV)
    y = yuv[:, :, 0]
    u = cv2.resize(yuv[:, :, 1], (256, 128), interpolation=cv2.INTER_AREA)
    v = cv2.resize(yuv[:, :, 2], (256, 128), interpolation=cv2.INTER_AREA)
    out = np.empty((6, 128, 256), dtype=np.uint8)
    out[0] = y[0::2, 0::2]
    out[1] = y[0::2, 1::2]
    out[2] = y[1::2, 0::2]
    out[3] = y[1::2, 1::2]
    out[4] = u
    out[5] = v
    return out


def convert_seq_rgb6_to_yuv12(seq_imgs: torch.Tensor) -> np.ndarray:
    """Convert (B,T,6,128,256) normalized RGB-pair tensor to (B,T,12,128,256) uint8 YUV planes."""
    arr = seq_imgs.detach().cpu().numpy()
    bsz, t, c, _, _ = arr.shape
    if c != 6:
        raise ValueError(f"Expected C=6 RGB pair input, got C={c}")
    out = np.empty((bsz, t, 12, 128, 256), dtype=np.uint8)
    for bi in range(bsz):
        for ti in range(t):
            rgb0 = _rgb_chw_norm_to_uint8_hwc(arr[bi, ti, 0:3])
            rgb1 = _rgb_chw_norm_to_uint8_hwc(arr[bi, ti, 3:6])
            yuv0 = _rgb_hwc_to_yuv420_6ch(rgb0)
            yuv1 = _rgb_hwc_to_yuv420_6ch(rgb1)
            out[bi, ti] = np.concatenate([yuv0, yuv1], axis=0)
    return out


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    tr_n, va_n = ensure_non_overlap_split_files(args.data_root, args.train_index, args.val_index, seed=args.seed)
    if not (os.path.isfile(args.train_index) and os.path.isfile(args.val_index)):
        raise RuntimeError("Failed to build split files.")
    print(f"[Split] train={tr_n} val={va_n}")

    data_root = normalize_data_root(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = Comma2k19SequenceDataset(args.train_index, data_root, "train", use_memcache=False, inner_progress=False)
    val_ds = Comma2k19SequenceDataset(
        args.val_index,
        data_root,
        "val",
        use_memcache=False,
        inner_progress=bool(args.dataloader_inner_progress),
    )
    train_ds = SafeDataset(train_ds)
    val_ds = SafeDataset(val_ds)

    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, device=device, num_workers=0)
    val_loader = make_loader(val_ds, batch_size=1, shuffle=False, device=device, num_workers=0)
    return train_loader, val_loader


def to_eval_sequence(batch: Dict[str, torch.Tensor], eval_seq_len: int) -> Tuple[np.ndarray, torch.Tensor]:
    seq_imgs = batch["seq_input_img"]  # (B, T, C, H, W), C=6 usually
    seq_labels = batch["seq_future_poses"]  # (B, T, 200, 3)
    _, t, c, _, _ = seq_imgs.shape
    if eval_seq_len > 0:
        t = min(t, eval_seq_len)
        seq_imgs = seq_imgs[:, :t]
        seq_labels = seq_labels[:, :t]

    if c == 6:
        imgs12 = convert_seq_rgb6_to_yuv12(seq_imgs)
    elif c == 12:
        imgs12 = seq_imgs.detach().cpu().numpy().astype(np.uint8)
    else:
        raise ValueError(f"Expected C=6 (RGB pair) or C=12 (packed), got C={c}")

    gt = seq_labels[:, :, :33, :].float()
    if gt.shape[2] != 33:
        raise ValueError(f"Need 33 trajectory points for policy output compare, got {gt.shape}")

    return imgs12, gt


def decode_policy_plan(
    policy_raw_outputs: np.ndarray,
    policy_output_slices: Dict[str, slice],
    gt: torch.Tensor,
) -> Tuple[float, torch.Tensor]:
    out = torch.from_numpy(policy_raw_outputs.astype(np.float32))
    plan_slice = policy_output_slices["plan"]
    plan_raw = out[:, plan_slice]
    if plan_raw.shape[1] % 2 != 0:
        raise ValueError(f"Unexpected policy plan width: {plan_raw.shape[1]}")

    n_values = plan_raw.shape[1] // 2
    if n_values % 33 != 0:
        raise ValueError(f"Policy plan width is not divisible by 33: {n_values}")
    plan_width = n_values // 33
    if plan_width < 3:
        raise ValueError(f"Policy plan width too small: {plan_width}")

    pred_mu = plan_raw[:, :n_values].view(out.shape[0], 33, plan_width)
    pred_pos = pred_mu[:, :, :3]
    reg = F.smooth_l1_loss(pred_pos, gt, reduction="mean")
    return float(reg.item()), pred_pos


def decode_policy_plan_torch(
    policy_raw_outputs: torch.Tensor,
    policy_output_slices: Dict[str, slice],
    gt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = policy_raw_outputs.float()
    plan_slice = policy_output_slices["plan"]
    plan_raw = out[:, plan_slice]
    if plan_raw.shape[1] % 2 != 0:
        raise ValueError(f"Unexpected policy plan width: {plan_raw.shape[1]}")

    n_values = plan_raw.shape[1] // 2
    if n_values % 33 != 0:
        raise ValueError(f"Policy plan width is not divisible by 33: {n_values}")
    plan_width = n_values // 33
    if plan_width < 3:
        raise ValueError(f"Policy plan width too small: {plan_width}")

    pred_mu = plan_raw[:, :n_values].view(out.shape[0], 33, plan_width)
    pred_pos = pred_mu[:, :, :3]
    reg = F.smooth_l1_loss(pred_pos, gt, reduction="mean")
    return reg, pred_pos


def traj_metric(pred_pos: torch.Tensor, metric: str) -> float:
    if metric == "+diffx":
        return float(pred_pos[:, :, 0].mean().item())
    if metric == "-diffx":
        return float((-pred_pos[:, :, 0]).mean().item())
    if metric == "+diffy":
        return float(pred_pos[:, :, 1].mean().item())
    if metric == "-diffy":
        return float((-pred_pos[:, :, 1]).mean().item())
    raise ValueError(f"Unknown trajectory metric: {metric}")


def eval_metric_for_sequence(
    vision_session: ort.InferenceSession,
    policy_session: ort.InferenceSession,
    batch: CachedBatch,
    vision_output_slices: Dict[str, slice],
    policy_output_slices: Dict[str, slice],
    vision_input_dtype: np.dtype,
    policy_input_dtype: np.dtype,
    policy_input_shapes: Dict[str, Tuple[int, ...]],
    metric: str,
) -> float:
    bsz, t, _, _, _ = batch.seq_imgs12.shape

    desire_shape = policy_input_shapes["desire_pulse"]
    features_shape = policy_input_shapes["features_buffer"]
    traffic_shape = policy_input_shapes["traffic_convention"]
    desire_len = shape_dim(desire_shape, 1, 25)
    desire_dim = shape_dim(desire_shape, 2, 8)
    feat_len = shape_dim(features_shape, 1, 25)
    feat_dim = shape_dim(features_shape, 2, 512)
    traffic_dim = shape_dim(traffic_shape, 1, 2)

    desire_pulse = np.zeros((bsz, desire_len, desire_dim), dtype=policy_input_dtype)
    traffic = np.zeros((bsz, traffic_dim), dtype=policy_input_dtype)
    if traffic_dim >= 2:
        traffic[:, 0] = 1.0  # right-hand traffic
    features_buffer = np.zeros((bsz, feat_len, feat_dim), dtype=policy_input_dtype)

    values: List[float] = []
    for ti in range(t):
        vision_inputs = {
            "img": cast_vision_input_for_ort(batch.seq_imgs12[:, ti], vision_input_dtype),
            "big_img": cast_vision_input_for_ort(batch.seq_imgs12[:, ti], vision_input_dtype),
        }
        vision_raw = vision_session.run(["outputs"], vision_inputs)[0]

        hidden = vision_raw[:, vision_output_slices["hidden_state"]].astype(policy_input_dtype, copy=False)
        features_buffer = np.concatenate([features_buffer[:, 1:, :], hidden[:, None, :]], axis=1)

        policy_inputs = {
            "desire_pulse": desire_pulse,
            "traffic_convention": traffic,
            "features_buffer": features_buffer,
        }
        policy_raw = policy_session.run(["outputs"], policy_inputs)[0]
        loss, pred_pos = decode_policy_plan(policy_raw, policy_output_slices, batch.seq_gt[:, ti])

        if metric == "loss":
            values.append(loss)
        else:
            values.append(traj_metric(pred_pos, metric))

    return float(np.mean(values)) if values else float("nan")


def eval_metric_value(
    vision_session: ort.InferenceSession,
    policy_session: ort.InferenceSession,
    cached_batches: Sequence[CachedBatch],
    vision_output_slices: Dict[str, slice],
    policy_output_slices: Dict[str, slice],
    vision_input_dtype: np.dtype,
    policy_input_dtype: np.dtype,
    policy_input_shapes: Dict[str, Tuple[int, ...]],
    metric: str,
) -> float:
    vals: List[float] = []
    for cb in cached_batches:
        score = eval_metric_for_sequence(
            vision_session,
            policy_session,
            cb,
            vision_output_slices,
            policy_output_slices,
            vision_input_dtype,
            policy_input_dtype,
            policy_input_shapes,
            metric,
        )
        vals.append(score)
    return float(np.mean(vals)) if vals else float("nan")


def collect_cached_batches(
    val_loader: DataLoader,
    num_batches: int,
    eval_seq_len: int,
) -> List[CachedBatch]:
    cached: List[CachedBatch] = []
    it = iter(val_loader)
    for _ in tqdm(range(num_batches), desc="Caching val batches", unit="batch", dynamic_ncols=True):
        try:
            batch = next(it)
        except StopIteration:
            break
        seq_imgs12, gt = to_eval_sequence(batch, eval_seq_len=eval_seq_len)
        cached.append(CachedBatch(seq_imgs12=seq_imgs12, seq_gt=gt))
    return cached


def inspect_outputs(
    vision_session: ort.InferenceSession,
    policy_session: ort.InferenceSession,
    batch: CachedBatch,
    vision_output_slices: Dict[str, slice],
    policy_output_slices: Dict[str, slice],
    vision_input_dtype: np.dtype,
    policy_input_dtype: np.dtype,
    policy_input_shapes: Dict[str, Tuple[int, ...]],
) -> None:
    bsz = batch.seq_imgs12.shape[0]
    desire_shape = policy_input_shapes["desire_pulse"]
    features_shape = policy_input_shapes["features_buffer"]
    traffic_shape = policy_input_shapes["traffic_convention"]
    desire_len = shape_dim(desire_shape, 1, 25)
    desire_dim = shape_dim(desire_shape, 2, 8)
    feat_len = shape_dim(features_shape, 1, 25)
    feat_dim = shape_dim(features_shape, 2, 512)
    traffic_dim = shape_dim(traffic_shape, 1, 2)

    desire_pulse = np.zeros((bsz, desire_len, desire_dim), dtype=policy_input_dtype)
    traffic = np.zeros((bsz, traffic_dim), dtype=policy_input_dtype)
    if traffic_dim >= 2:
        traffic[:, 0] = 1.0
    features_buffer = np.zeros((bsz, feat_len, feat_dim), dtype=policy_input_dtype)

    vision_inputs = {
        "img": cast_vision_input_for_ort(batch.seq_imgs12[:, 0], vision_input_dtype),
        "big_img": cast_vision_input_for_ort(batch.seq_imgs12[:, 0], vision_input_dtype),
    }
    vision_raw = vision_session.run(["outputs"], vision_inputs)[0]
    hidden = vision_raw[:, vision_output_slices["hidden_state"]].astype(policy_input_dtype, copy=False)
    features_buffer = np.concatenate([features_buffer[:, 1:, :], hidden[:, None, :]], axis=1)

    policy_inputs = {
        "desire_pulse": desire_pulse,
        "traffic_convention": traffic,
        "features_buffer": features_buffer,
    }
    policy_raw = policy_session.run(["outputs"], policy_inputs)[0]

    print("\n[Inspect] Vision output slices from metadata:")
    for name, sl in vision_output_slices.items():
        arr = vision_raw[0, sl].astype(np.float32)
        print(f"  {name:<24} len={arr.size:<4} mean={arr.mean(): .5f} std={arr.std(): .5f}")

    print("\n[Inspect] Policy output slices from metadata:")
    for name, sl in policy_output_slices.items():
        arr = policy_raw[0, sl].astype(np.float32)
        print(f"  {name:<24} len={arr.size:<4} mean={arr.mean(): .5f} std={arr.std(): .5f}")


def iter_fp16_initializers(model: onnx.ModelProto, allow_bias: bool, restrict: Optional[List[str]]) -> Iterable[Tuple[str, np.ndarray]]:
    for t in model.graph.initializer:
        if t.data_type != TensorProto.FLOAT16:
            continue
        name = t.name
        if (not allow_bias) and name.endswith(".bias"):
            continue
        if restrict and not any(tag in name for tag in restrict):
            continue
        arr = numpy_helper.to_array(t)
        if arr.size == 0:
            continue
        yield name, arr


def get_model_input_order(model: onnx.ModelProto) -> List[str]:
    initializer_names = {t.name for t in model.graph.initializer}
    return [i.name for i in model.graph.input if i.name not in initializer_names]


def get_named_input_elem_type(model: onnx.ModelProto, input_name: str) -> int:
    for i in model.graph.input:
        if i.name == input_name:
            return int(i.type.tensor_type.elem_type)
    raise KeyError(f"Input not found in model graph: {input_name}")


def onnx_elem_type_to_torch_dtype(elem_type: int, default: torch.dtype = torch.float32) -> torch.dtype:
    if elem_type == TensorProto.UINT8:
        return torch.uint8
    if elem_type == TensorProto.FLOAT16:
        return torch.float16
    if elem_type == TensorProto.FLOAT:
        return torch.float32
    return default


def build_onnx_to_torch_refs(
    model: onnx.ModelProto,
    module: torch.nn.Module,
    allow_bias: bool,
    restrict: Optional[List[str]],
) -> Dict[str, torch.Tensor]:
    param_dict = dict(module.named_parameters())
    buffer_dict = dict(module.named_buffers())

    key_to_locs: Dict[Tuple[str, Tuple[int, ...], str], List[Tuple[str, str]]] = {}
    shape_to_locs: Dict[Tuple[int, ...], List[Tuple[str, str]]] = {}

    for p_name, p in param_dict.items():
        arr = p.detach().cpu().numpy()
        key_to_locs.setdefault(tensor_hash_key(arr), []).append(("param", p_name))
        shape_to_locs.setdefault(tuple(arr.shape), []).append(("param", p_name))
    for b_name, b in buffer_dict.items():
        arr = b.detach().cpu().numpy()
        key_to_locs.setdefault(tensor_hash_key(arr), []).append(("buffer", b_name))
        shape_to_locs.setdefault(tuple(arr.shape), []).append(("buffer", b_name))

    refs: Dict[str, torch.Tensor] = {}
    used_locs: set[Tuple[str, str]] = set()
    skipped = 0

    for onnx_name, arr in iter_fp16_initializers(model, allow_bias=allow_bias, restrict=restrict):
        loc: Optional[Tuple[str, str]] = None
        for cand in key_to_locs.get(tensor_hash_key(arr), []):
            if cand not in used_locs:
                loc = cand
                break

        if loc is None:
            for cand in shape_to_locs.get(tuple(arr.shape), []):
                if cand in used_locs:
                    continue
                if cand[0] == "param":
                    carr = param_dict[cand[1]].detach().cpu().numpy()
                else:
                    carr = buffer_dict[cand[1]].detach().cpu().numpy()
                if np.allclose(carr.astype(np.float32), arr.astype(np.float32), rtol=1e-3, atol=1e-3):
                    loc = cand
                    break

        if loc is None:
            skipped += 1
            continue
        used_locs.add(loc)
        if loc[0] == "param":
            refs[onnx_name] = param_dict[loc[1]]
        else:
            refs[onnx_name] = buffer_dict[loc[1]]
    if skipped > 0:
        print(f"[Gradient] skipped {skipped} unmapped initializers (likely folded constants)")
    return refs


def eval_loss_torch_and_backward_split(
    vision_model_torch: torch.nn.Module,
    policy_model_torch: torch.nn.Module,
    cached_batches: Sequence[CachedBatch],
    vision_output_slices: Dict[str, slice],
    policy_output_slices: Dict[str, slice],
    policy_input_shapes: Dict[str, Tuple[int, ...]],
    vision_input_order: Sequence[str],
    policy_input_order: Sequence[str],
    device: torch.device,
    model_dtype: torch.dtype,
    vision_input_dtype_torch: torch.dtype,
) -> float:
    vision_model_torch.zero_grad(set_to_none=True)
    policy_model_torch.zero_grad(set_to_none=True)
    vals: List[float] = []

    desire_shape = policy_input_shapes["desire_pulse"]
    features_shape = policy_input_shapes["features_buffer"]
    traffic_shape = policy_input_shapes["traffic_convention"]
    desire_len = shape_dim(desire_shape, 1, 25)
    desire_dim = shape_dim(desire_shape, 2, 8)
    feat_len = shape_dim(features_shape, 1, 25)
    feat_dim = shape_dim(features_shape, 2, 512)
    traffic_dim = shape_dim(traffic_shape, 1, 2)

    for cb in cached_batches:
        imgs = torch.from_numpy(cb.seq_imgs12).to(device=device, dtype=vision_input_dtype_torch)
        gt = cb.seq_gt.to(device=device, dtype=torch.float32)
        bsz, t, _, _, _ = imgs.shape

        desire_pulse = torch.zeros((bsz, desire_len, desire_dim), device=device, dtype=model_dtype)
        traffic = torch.zeros((bsz, traffic_dim), device=device, dtype=model_dtype)
        if traffic_dim >= 2:
            traffic[:, 0] = 1.0
        features_buffer = torch.zeros((bsz, feat_len, feat_dim), device=device, dtype=model_dtype)

        per_step_losses: List[torch.Tensor] = []
        for i in range(t):
            vision_inputs = {"img": imgs[:, i], "big_img": imgs[:, i]}
            vision_feed = [vision_inputs[k] for k in vision_input_order]
            vision_raw = vision_model_torch(*vision_feed)
            if not isinstance(vision_raw, torch.Tensor):
                raise RuntimeError(f"Unexpected vision torch output type: {type(vision_raw)}")

            hidden = vision_raw[:, vision_output_slices["hidden_state"]].to(model_dtype)
            features_buffer = torch.cat([features_buffer[:, 1:, :], hidden[:, None, :]], dim=1)

            policy_inputs = {
                "desire_pulse": desire_pulse,
                "traffic_convention": traffic,
                "features_buffer": features_buffer,
            }
            policy_feed = [policy_inputs[k] for k in policy_input_order]
            policy_raw = policy_model_torch(*policy_feed)
            if not isinstance(policy_raw, torch.Tensor):
                raise RuntimeError(f"Unexpected policy torch output type: {type(policy_raw)}")

            loss_t, _ = decode_policy_plan_torch(policy_raw, policy_output_slices, gt[:, i])
            per_step_losses.append(loss_t)

        if not per_step_losses:
            continue
        batch_loss = torch.stack(per_step_losses).mean()
        batch_loss.backward()
        vals.append(float(batch_loss.detach().cpu().item()))

    return float(np.mean(vals)) if vals else float("nan")


def eval_metric_value_torch(
    vision_model_torch: torch.nn.Module,
    policy_model_torch: torch.nn.Module,
    cached_batches: Sequence[CachedBatch],
    vision_output_slices: Dict[str, slice],
    policy_output_slices: Dict[str, slice],
    policy_input_shapes: Dict[str, Tuple[int, ...]],
    vision_input_order: Sequence[str],
    policy_input_order: Sequence[str],
    device: torch.device,
    vision_input_dtype_torch: torch.dtype,
    policy_input_dtype_torch: torch.dtype,
    metric: str,
) -> float:
    vals: List[float] = []
    for cb in cached_batches:
        imgs = torch.from_numpy(cb.seq_imgs12).to(device=device, dtype=vision_input_dtype_torch)
        gt = cb.seq_gt.to(device=device, dtype=torch.float32)
        bsz, t, _, _, _ = imgs.shape

        desire_shape = policy_input_shapes["desire_pulse"]
        features_shape = policy_input_shapes["features_buffer"]
        traffic_shape = policy_input_shapes["traffic_convention"]
        desire_len = shape_dim(desire_shape, 1, 25)
        desire_dim = shape_dim(desire_shape, 2, 8)
        feat_len = shape_dim(features_shape, 1, 25)
        feat_dim = shape_dim(features_shape, 2, 512)
        traffic_dim = shape_dim(traffic_shape, 1, 2)

        desire_pulse = torch.zeros((bsz, desire_len, desire_dim), device=device, dtype=policy_input_dtype_torch)
        traffic = torch.zeros((bsz, traffic_dim), device=device, dtype=policy_input_dtype_torch)
        if traffic_dim >= 2:
            traffic[:, 0] = 1.0
        features_buffer = torch.zeros((bsz, feat_len, feat_dim), device=device, dtype=policy_input_dtype_torch)

        per_step_vals: List[float] = []
        for i in range(t):
            vision_inputs = {"img": imgs[:, i], "big_img": imgs[:, i]}
            vision_feed = [vision_inputs[k] for k in vision_input_order]
            vision_raw = vision_model_torch(*vision_feed)
            if not isinstance(vision_raw, torch.Tensor):
                raise RuntimeError(f"Unexpected vision torch output type: {type(vision_raw)}")

            hidden = vision_raw[:, vision_output_slices["hidden_state"]].to(policy_input_dtype_torch)
            features_buffer = torch.cat([features_buffer[:, 1:, :], hidden[:, None, :]], dim=1)

            policy_inputs = {
                "desire_pulse": desire_pulse,
                "traffic_convention": traffic,
                "features_buffer": features_buffer,
            }
            policy_feed = [policy_inputs[k] for k in policy_input_order]
            policy_raw = policy_model_torch(*policy_feed)
            if not isinstance(policy_raw, torch.Tensor):
                raise RuntimeError(f"Unexpected policy torch output type: {type(policy_raw)}")

            loss_t, pred_pos = decode_policy_plan_torch(policy_raw, policy_output_slices, gt[:, i])
            if metric == "loss":
                per_step_vals.append(float(loss_t.detach().cpu().item()))
            else:
                per_step_vals.append(traj_metric(pred_pos.detach().cpu(), metric))

        if per_step_vals:
            vals.append(float(np.mean(per_step_vals)))

    return float(np.mean(vals)) if vals else float("nan")


def build_torch_models_for_eval(
    vision_model: onnx.ModelProto,
    policy_model: onnx.ModelProto,
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.nn.Module, List[str], List[str], torch.dtype, torch.dtype]:
    patch_torch_linear_dtype_mismatch()
    if device.type == "cuda":
        patch_onnx2torch_cuda_matmul()
    vision_input_order = get_model_input_order(vision_model)
    policy_input_order = get_model_input_order(policy_model)
    vision_torch = convert_onnx_to_torch_with_compat(sanitize_onnx_for_onnx2torch(vision_model)).to(device)
    policy_torch = convert_onnx_to_torch_with_compat(sanitize_onnx_for_onnx2torch(policy_model)).to(device)
    vision_torch.train(False)
    policy_torch.train(False)
    vision_torch = vision_torch.float()
    policy_torch = policy_torch.float()

    v_elem_type = get_named_input_elem_type(vision_model, "img")
    p_elem_type = get_named_input_elem_type(policy_model, "desire_pulse")
    v_dtype = onnx_elem_type_to_torch_dtype(v_elem_type, default=torch.float32)
    p_dtype = onnx_elem_type_to_torch_dtype(p_elem_type, default=torch.float32)
    if p_dtype not in (torch.float16, torch.float32):
        p_dtype = torch.float32
    return vision_torch, policy_torch, vision_input_order, policy_input_order, v_dtype, p_dtype


def select_weight_candidates(
    models: Dict[str, onnx.ModelProto],
    target_model: str,
    top_w: int,
    per_tensor_k: int,
    allow_bias: bool,
    restrict: Optional[List[str]],
) -> List[Tuple[str, str, int, float]]:
    items: List[Tuple[str, str, int, float]] = []
    for model_key in requested_model_keys(target_model):
        model = models[model_key]
        for name, arr in iter_fp16_initializers(model, allow_bias=allow_bias, restrict=restrict):
            flat_abs = np.abs(arr.reshape(-1).astype(np.float32))
            k = flat_abs.size if per_tensor_k <= 0 else min(per_tensor_k, flat_abs.size)
            if k <= 0:
                continue
            idx = np.argpartition(flat_abs, -k)[-k:]
            for i in idx:
                items.append((model_key, name, int(i), float(flat_abs[i])))
    items.sort(key=lambda x: x[3], reverse=True)
    if top_w <= 0:
        return items
    return items[: min(top_w, len(items))]


def rank_bits_progressive_torch(
    base_vision_model: onnx.ModelProto,
    base_policy_model: onnx.ModelProto,
    original_score: float,
    cached_batches: Sequence[CachedBatch],
    vision_output_slices: Dict[str, slice],
    policy_output_slices: Dict[str, slice],
    policy_input_shapes: Dict[str, Tuple[int, ...]],
    candidates: Sequence[Tuple[str, str, int]],
    bitset: Sequence[int],
    top_b: int,
    eval_metric: str,
    device: torch.device,
) -> List[Dict[str, Any]]:
    if not np.isfinite(original_score):
        raise RuntimeError(f"Baseline metric score is non-finite: {original_score}")

    current_model_bytes = {
        "vision": base_vision_model.SerializeToString(),
        "policy": base_policy_model.SerializeToString(),
    }
    current_score = float(original_score)
    records: List[Dict[str, Any]] = []

    pending: List[Tuple[str, str, int, int]] = [(mk, n, fi, b) for (mk, n, fi) in candidates for b in bitset]
    if not pending:
        return records

    total_steps = min(top_b, len(pending))
    skipped_non_finite = 0

    outer = tqdm(total=total_steps, desc="Progressive rank (torch)", unit="step", dynamic_ncols=True)
    for step in range(total_steps):
        if not pending:
            break

        best_idx = -1
        best_item: Optional[Dict[str, Any]] = None
        inner = tqdm(total=len(pending), desc=f"Scan step {step + 1}", unit="flip", leave=False, dynamic_ncols=True)
        for idx, (model_key, name, flat_idx, bit) in enumerate(pending):
            model_bytes, old, new = make_flipped_model_bytes(current_model_bytes[model_key], name, flat_idx, int(bit))
            if model_bytes is None:
                inner.update(1)
                continue

            cand_vision_bytes = model_bytes if model_key == "vision" else current_model_bytes["vision"]
            cand_policy_bytes = model_bytes if model_key == "policy" else current_model_bytes["policy"]

            cand_vision_model = onnx.load_from_string(cand_vision_bytes)
            cand_policy_model = onnx.load_from_string(cand_policy_bytes)
            try:
                v_torch, p_torch, v_order, p_order, v_dtype, p_dtype = build_torch_models_for_eval(
                    cand_vision_model,
                    cand_policy_model,
                    device=device,
                )
                flipped_score = eval_metric_value_torch(
                    v_torch,
                    p_torch,
                    cached_batches,
                    vision_output_slices,
                    policy_output_slices,
                    policy_input_shapes,
                    v_order,
                    p_order,
                    device,
                    v_dtype,
                    p_dtype,
                    eval_metric,
                )
            except Exception:
                skipped_non_finite += 1
                inner.update(1)
                continue

            if not np.isfinite(flipped_score):
                skipped_non_finite += 1
                inner.update(1)
                continue

            score = float(flipped_score - current_score)
            if (best_item is None) or (score > best_item["score"]):
                best_idx = idx
                best_item = {
                    "model": model_key,
                    "name": name,
                    "flat": int(flat_idx),
                    "bit": int(bit),
                    "metric": eval_metric,
                    "score": float(score),
                    "original_score": float(current_score),
                    "flipped_score": float(flipped_score),
                    "old": float(old),
                    "new": float(new),
                    "step": int(step + 1),
                    "_vision_bytes": cand_vision_bytes,
                    "_policy_bytes": cand_policy_bytes,
                }
                inner.set_postfix(best_score=f"{score:.4f}")
            inner.update(1)
        inner.close()

        if best_item is None:
            print("[Rank] stop early: no finite candidate remained.")
            break

        current_model_bytes["vision"] = best_item.pop("_vision_bytes")
        current_model_bytes["policy"] = best_item.pop("_policy_bytes")
        current_score = float(best_item["flipped_score"])
        records.append(best_item)

        pending.pop(best_idx)
        outer.update(1)
        outer.set_postfix(current_score=f"{current_score:.4f}", kept=len(records), pending=len(pending), skipped_nan=skipped_non_finite)
    outer.close()
    return records


def select_weight_candidates_by_gradient(
    models: Dict[str, onnx.ModelProto],
    target_model: str,
    cached_batches: Sequence[CachedBatch],
    vision_output_slices: Dict[str, slice],
    policy_output_slices: Dict[str, slice],
    policy_input_shapes: Dict[str, Tuple[int, ...]],
    top_w: int,
    per_tensor_k: int,
    allow_bias: bool,
    restrict: Optional[List[str]],
    providers: Sequence[str],
    sample_all: bool = False,
    torch_dtype_mode: str = "auto",
) -> List[Tuple[str, str, int, float]]:
    try:
        import onnx2torch  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"onnx2torch is required for gradient selection but unavailable: {exc}") from exc

    use_cuda = ("CUDAExecutionProvider" in providers) and torch.cuda.is_available()
    if use_cuda:
        patch_onnx2torch_cuda_matmul()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"[Gradient] backend=pytorch_autograd device={device}")
    vision_model = models["vision"]
    policy_model = models["policy"]
    vision_input_order = get_model_input_order(vision_model)
    policy_input_order = get_model_input_order(policy_model)

    vision_torch = convert_onnx_to_torch_with_compat(sanitize_onnx_for_onnx2torch(vision_model)).to(device)
    policy_torch = convert_onnx_to_torch_with_compat(sanitize_onnx_for_onnx2torch(policy_model)).to(device)
    vision_torch.train(False)
    policy_torch.train(False)

    if torch_dtype_mode == "fp16":
        model_dtype = torch.float16
    elif torch_dtype_mode == "fp32":
        model_dtype = torch.float32
    else:
        model_dtype = torch.float32
    if (device.type == "cpu") and (model_dtype == torch.float16):
        print("[Gradient] fp16 backward on CPU is unsupported on many platforms, switching to fp32.")
        model_dtype = torch.float32
    if model_dtype == torch.float32:
        vision_torch = vision_torch.float()
        policy_torch = policy_torch.float()
    img_elem_type = get_named_input_elem_type(vision_model, "img")
    vision_input_dtype_torch = torch.uint8 if img_elem_type == TensorProto.UINT8 else model_dtype

    refs_by_model = {
        "vision": build_onnx_to_torch_refs(vision_model, vision_torch, allow_bias=allow_bias, restrict=restrict),
        "policy": build_onnx_to_torch_refs(policy_model, policy_torch, allow_bias=allow_bias, restrict=restrict),
    }
    for model_key in requested_model_keys(target_model):
        for tref in refs_by_model[model_key].values():
            tref.requires_grad_(True)
            tref.grad = None

    print(f"[Gradient] Computing baseline loss + backward (dtype={str(model_dtype).replace('torch.', '')})...")
    base_loss = eval_loss_torch_and_backward_split(
        vision_torch,
        policy_torch,
        cached_batches,
        vision_output_slices,
        policy_output_slices,
        policy_input_shapes,
        vision_input_order,
        policy_input_order,
        device=device,
        model_dtype=model_dtype,
        vision_input_dtype_torch=vision_input_dtype_torch,
    )
    print(f"[Gradient] Baseline loss: {base_loss:.6f}")

    items: List[Tuple[str, str, int, float]] = []
    for model_key in requested_model_keys(target_model):
        tensor_list = list(iter_fp16_initializers(models[model_key], allow_bias=allow_bias, restrict=restrict))
        print(f"[Gradient] Collecting gradients for {model_key}: {len(tensor_list)} tensors...")
        for name, arr in tqdm(tensor_list, desc=f"Grad {model_key}", unit="tensor", dynamic_ncols=True):
            tref = refs_by_model[model_key].get(name)
            if tref is None or tref.grad is None:
                continue

            grad_abs = np.abs(tref.grad.detach().cpu().numpy().reshape(-1).astype(np.float32))
            flat = arr.reshape(-1)
            n = flat.size

            if sample_all or per_tensor_k <= 0:
                sample_idx = np.arange(n)
            elif per_tensor_k < n:
                flat_abs = np.abs(flat.astype(np.float32))
                sample_k = min(per_tensor_k * 10, n)
                sample_idx = np.argpartition(flat_abs, -sample_k)[-sample_k:]
            else:
                sample_idx = np.arange(n)

            k = len(sample_idx) if per_tensor_k <= 0 else min(per_tensor_k, len(sample_idx))
            if k <= 0:
                continue
            sample_grad_abs = grad_abs[sample_idx]
            top_k_local = np.argpartition(sample_grad_abs, -k)[-k:]
            for local_i in top_k_local:
                flat_idx = int(sample_idx[local_i])
                grad_mag = float(sample_grad_abs[local_i])
                items.append((model_key, name, flat_idx, grad_mag))

    items.sort(key=lambda x: x[3], reverse=True)
    if top_w <= 0:
        print(f"[Gradient] Selected {len(items)} candidates by gradient magnitude (all)")
        return items
    print(f"[Gradient] Selected {len(items[:top_w])} candidates by gradient magnitude")
    return items[: min(top_w, len(items))]


def flip_fp16_value(arr: np.ndarray, flat_idx: int, bit: int) -> Tuple[float, float]:
    flat = arr.reshape(-1)
    raw = flat.view(np.uint16)
    old = np.float16(flat[flat_idx])
    raw[flat_idx] ^= np.uint16(1 << bit)
    new = np.float16(flat[flat_idx])
    if not np.isfinite(new):
        raw[flat_idx] ^= np.uint16(1 << bit)
        return float(old), float(old)
    return float(old), float(new)


def make_flipped_model_bytes(base_bytes: bytes, name: str, flat_idx: int, bit: int) -> Tuple[Optional[bytes], float, float]:
    m = onnx.load_from_string(base_bytes)
    for t in m.graph.initializer:
        if t.name != name:
            continue
        arr = numpy_helper.to_array(t).copy()
        old, new = flip_fp16_value(arr, flat_idx, bit)
        if new == old:
            return None, old, new
        t.CopyFrom(numpy_helper.from_array(arr.astype(np.float16), name=t.name))
        return m.SerializeToString(), old, new
    raise KeyError(f"Initializer not found: {name}")


def rank_bits_progressive(
    base_vision_model: onnx.ModelProto,
    base_policy_model: onnx.ModelProto,
    original_score: float,
    cached_batches: Sequence[CachedBatch],
    vision_output_slices: Dict[str, slice],
    policy_output_slices: Dict[str, slice],
    policy_input_shapes: Dict[str, Tuple[int, ...]],
    candidates: Sequence[Tuple[str, str, int]],
    bitset: Sequence[int],
    top_b: int,
    eval_metric: str,
    providers: Sequence[str],
) -> List[Dict[str, Any]]:
    if not np.isfinite(original_score):
        raise RuntimeError(f"Baseline metric score is non-finite: {original_score}")

    current_model_bytes = {
        "vision": base_vision_model.SerializeToString(),
        "policy": base_policy_model.SerializeToString(),
    }
    current_score = float(original_score)
    records: List[Dict[str, Any]] = []

    pending: List[Tuple[str, str, int, int]] = [(mk, n, fi, b) for (mk, n, fi) in candidates for b in bitset]
    if not pending:
        return records

    active_providers = list(providers)
    total_steps = min(top_b, len(pending))
    skipped_non_finite = 0

    outer = tqdm(total=total_steps, desc="Progressive rank", unit="step", dynamic_ncols=True)
    for step in range(total_steps):
        if not pending:
            break

        best_idx = -1
        best_item: Optional[Dict[str, Any]] = None
        inner = tqdm(total=len(pending), desc=f"Scan step {step + 1}", unit="flip", leave=False, dynamic_ncols=True)
        for idx, (model_key, name, flat_idx, bit) in enumerate(pending):
            model_bytes, old, new = make_flipped_model_bytes(current_model_bytes[model_key], name, flat_idx, int(bit))
            if model_bytes is None:
                inner.update(1)
                continue

            cand_vision_bytes = model_bytes if model_key == "vision" else current_model_bytes["vision"]
            cand_policy_bytes = model_bytes if model_key == "policy" else current_model_bytes["policy"]

            try:
                vision_sess = make_session(cand_vision_bytes, providers=active_providers)
                policy_sess = make_session(cand_policy_bytes, providers=active_providers)
                vision_dtype = ort_type_to_numpy_dtype(vision_sess.get_inputs()[0].type)
                policy_dtype = ort_type_to_numpy_dtype(policy_sess.get_inputs()[0].type)
                flipped_score = eval_metric_value(
                    vision_sess,
                    policy_sess,
                    cached_batches,
                    vision_output_slices,
                    policy_output_slices,
                    vision_dtype,
                    policy_dtype,
                    policy_input_shapes,
                    eval_metric,
                )
            except Exception as exc:
                if ("CUDAExecutionProvider" in active_providers) and is_cuda_runtime_error(exc):
                    print(f"[ORT] CUDA runtime error during rank scan, fallback to CPU: {exc}")
                    active_providers = ["CPUExecutionProvider"]
                    vision_sess = make_session(cand_vision_bytes, providers=active_providers)
                    policy_sess = make_session(cand_policy_bytes, providers=active_providers)
                    vision_dtype = ort_type_to_numpy_dtype(vision_sess.get_inputs()[0].type)
                    policy_dtype = ort_type_to_numpy_dtype(policy_sess.get_inputs()[0].type)
                    flipped_score = eval_metric_value(
                        vision_sess,
                        policy_sess,
                        cached_batches,
                        vision_output_slices,
                        policy_output_slices,
                        vision_dtype,
                        policy_dtype,
                        policy_input_shapes,
                        eval_metric,
                    )
                else:
                    raise

            if not np.isfinite(flipped_score):
                skipped_non_finite += 1
                inner.update(1)
                continue

            score = float(flipped_score - current_score)
            if (best_item is None) or (score > best_item["score"]):
                best_idx = idx
                best_item = {
                    "model": model_key,
                    "name": name,
                    "flat": int(flat_idx),
                    "bit": int(bit),
                    "metric": eval_metric,
                    "score": float(score),
                    "original_score": float(current_score),
                    "flipped_score": float(flipped_score),
                    "old": float(old),
                    "new": float(new),
                    "step": int(step + 1),
                    "_vision_bytes": cand_vision_bytes,
                    "_policy_bytes": cand_policy_bytes,
                }
                inner.set_postfix(best_score=f"{score:.4f}")
            inner.update(1)
        inner.close()

        if best_item is None:
            print("[Rank] stop early: no finite candidate remained.")
            break

        current_model_bytes["vision"] = best_item.pop("_vision_bytes")
        current_model_bytes["policy"] = best_item.pop("_policy_bytes")
        current_score = float(best_item["flipped_score"])
        records.append(best_item)

        pending.pop(best_idx)
        outer.update(1)
        outer.set_postfix(current_score=f"{current_score:.4f}", kept=len(records), pending=len(pending), skipped_nan=skipped_non_finite)
    outer.close()

    return records


def save_weight_candidates(
    path: str,
    candidates: Sequence[Tuple[str, str, int, float]],
    *,
    vision_onnx_path: str,
    policy_onnx_path: str,
    target_model: str,
    restrict: Optional[List[str]],
    allow_bias: bool,
    top_w: int,
    per_tensor_k: int,
    selection_method: str = "magnitude",
) -> None:
    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "vision_onnx": os.path.abspath(vision_onnx_path),
            "policy_onnx": os.path.abspath(policy_onnx_path),
            "target_model": target_model,
            "restrict": restrict,
            "allow_bias": bool(allow_bias),
            "top_w": int(top_w),
            "per_tensor_k": int(per_tensor_k),
            "num_candidates": int(len(candidates)),
            "selection_method": selection_method,
        },
        "candidates": [
            {"model": model_key, "name": name, "flat": int(flat), "score": float(score)}
            for model_key, name, flat, score in candidates
        ],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_weight_candidates(path: str, default_model_key: str) -> List[Tuple[str, str, int, float]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("candidates", [])
    out: List[Tuple[str, str, int, float]] = []
    for r in rows:
        out.append(
            (
                str(r.get("model", default_model_key)),
                str(r["name"]),
                int(r["flat"]),
                float(r.get("score", 0.0)),
            )
        )
    return out


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def with_timestamp(path: str, ts: str, out_dir: str) -> str:
    base = os.path.basename(path)
    root, ext = os.path.splitext(base)
    if not ext:
        ext = ".json"
    return os.path.join(out_dir, f"{root}_{ts}{ext}")


def get_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_split_dir = root / "op_0103" / "splits"

    p = argparse.ArgumentParser(description="Important bit search for openpilot 0.10.3 split ONNX models")
    p.add_argument("--vision-onnx", default=str(root / "op_0103" / "models" / "driving_vision.onnx"))
    p.add_argument("--vision-metadata", default=str(root / "op_0103" / "models" / "driving_vision_metadata.pkl"))
    p.add_argument("--policy-onnx", default=str(root / "op_0103" / "models" / "driving_policy.onnx"))
    p.add_argument("--policy-metadata", default=str(root / "op_0103" / "models" / "driving_policy_metadata.pkl"))
    p.add_argument("--data-root", default="/home/zx/Projects/comma2k19")
    p.add_argument("--train-index", default=str(default_split_dir / "comma2k19_train_non_overlap.txt"))
    p.add_argument("--val-index", default=str(default_split_dir / "comma2k19_val_non_overlap.txt"))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-val-batches", type=int, default=2)
    p.add_argument("--top-w", type=int, default=50, help="global candidate cap; <=0 means keep all selected scalars")
    p.add_argument("--per-tensor-k", type=int, default=1, help="max scalars per tensor; <=0 means keep all scalars in each tensor")
    p.add_argument("--top-b", type=int, default=1)
    p.add_argument("--bitset", default="exponent_sign", help="fp16 bit set: mantissa/exponent/sign/exponent_sign/all or csv")
    p.add_argument("--allow-bias", action="store_true")
    p.add_argument("--restrict", default="", help="comma-separated substrings to keep, e.g. Conv,Gemm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", choices=["all", "select-weights", "rank-bits"], default="all")
    p.add_argument("--weights-in", default="", help="JSON file produced by select-weights stage")
    p.add_argument("--weights-out", default=str(root / "op_0103" / "weights_candidates_0103.json"))
    p.add_argument(
        "--eval-metric",
        choices=["loss", "+diffx", "-diffx", "+diffy", "-diffy"],
        default="loss",
        help="ranking score: loss delta, or signed trajectory delta on final policy trajectory",
    )
    p.add_argument("--out", default=str(root / "op_0103" / "important_bits_0103.json"))
    p.add_argument("--eval-seq-len", type=int, default=20, help="timesteps used per cached sequence; <=0 means full sequence")
    p.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto", help="ONNX Runtime provider")
    p.add_argument(
        "--eval-backend",
        choices=["auto", "ort", "torch"],
        default="auto",
        help="Backend for baseline/rank-bits eval. auto: use torch when provider=cuda, else ort",
    )
    p.add_argument("--target-model", choices=["vision", "policy", "both"], default="both", help="Which model(s) to search weights/bits on")
    p.add_argument(
        "--weight-selection-method",
        choices=["magnitude", "gradient"],
        default="magnitude",
        help="Weight selection method: magnitude (absolute value) or gradient (PyTorch autograd)",
    )
    p.add_argument(
        "--gradient-torch-dtype",
        choices=["auto", "fp16", "fp32"],
        default="auto",
        help="Torch dtype for autograd backend (auto uses fp32 for stability)",
    )
    p.add_argument("--sample-all-weights", action="store_true", help="Compute gradients for ALL weights in each tensor")
    p.add_argument("--inspect-only", action="store_true", help="Only print observed output stats and baseline loss")
    p.add_argument(
        "--dataloader-inner-progress",
        action="store_true",
        help="Show fine-grained progress inside each val sample loading/preprocess step",
    )
    return p.parse_args()


def main() -> None:
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_ts = timestamp_id()
    out_dir = os.path.join(REPO_ROOT, "op_0103", "out")
    weights_out_path = with_timestamp(args.weights_out, run_ts, out_dir)
    result_out_path = with_timestamp(args.out, run_ts, out_dir)
    providers = resolve_providers(args.provider)
    print(f"[ORT] providers={providers}")

    with open(args.vision_metadata, "rb") as f:
        vision_metadata = pickle.load(f)
    with open(args.policy_metadata, "rb") as f:
        policy_metadata = pickle.load(f)

    vision_output_slices: Dict[str, slice] = vision_metadata["output_slices"]
    policy_output_slices: Dict[str, slice] = policy_metadata["output_slices"]
    policy_input_shapes: Dict[str, Tuple[int, ...]] = policy_metadata["input_shapes"]

    restrict = [x.strip() for x in args.restrict.split(",") if x.strip()] or None
    bitset = parse_bitset(args.bitset)
    bitset_mode = canonical_bitset_mode(args.bitset)

    models_for_selection: Dict[str, onnx.ModelProto] = {}
    need_both_models = args.weight_selection_method == "gradient"
    if need_both_models or args.target_model in ("vision", "both"):
        models_for_selection["vision"] = onnx.load(args.vision_onnx)
    if need_both_models or args.target_model in ("policy", "both"):
        models_for_selection["policy"] = onnx.load(args.policy_onnx)

    if args.stage == "select-weights":
        if args.weight_selection_method == "gradient":
            _, val_loader = build_loaders(args)
            cached_batches = collect_cached_batches(val_loader, num_batches=args.num_val_batches, eval_seq_len=args.eval_seq_len)
            if not cached_batches:
                raise RuntimeError("No validation batches cached. Check dataset path/splits.")
            mode_str = "ALL weights" if args.sample_all_weights else f"sampled (per_tensor_k={args.per_tensor_k})"
            print(f"[Weights] Using gradient-based selection (mode={mode_str})")
            selected = select_weight_candidates_by_gradient(
                models=models_for_selection,
                target_model=args.target_model,
                cached_batches=cached_batches,
                vision_output_slices=vision_output_slices,
                policy_output_slices=policy_output_slices,
                policy_input_shapes=policy_input_shapes,
                top_w=args.top_w,
                per_tensor_k=args.per_tensor_k,
                allow_bias=args.allow_bias,
                restrict=restrict,
                providers=providers,
                sample_all=args.sample_all_weights,
                torch_dtype_mode=args.gradient_torch_dtype,
            )
        else:
            print("[Weights] Using magnitude-based selection")
            selected = select_weight_candidates(
                models=models_for_selection,
                target_model=args.target_model,
                top_w=args.top_w,
                per_tensor_k=args.per_tensor_k,
                allow_bias=args.allow_bias,
                restrict=restrict,
            )
        if not selected:
            raise RuntimeError("No fp16 initializer candidates selected.")
        save_weight_candidates(
            weights_out_path,
            selected,
            vision_onnx_path=args.vision_onnx,
            policy_onnx_path=args.policy_onnx,
            target_model=args.target_model,
            restrict=restrict,
            allow_bias=args.allow_bias,
            top_w=args.top_w,
            per_tensor_k=args.per_tensor_k,
            selection_method=args.weight_selection_method,
        )
        print(f"[Done] Saved {len(selected)} weight candidates to: {weights_out_path}")
        return

    _, val_loader = build_loaders(args)
    cached_batches = collect_cached_batches(val_loader, num_batches=args.num_val_batches, eval_seq_len=args.eval_seq_len)
    if not cached_batches:
        raise RuntimeError("No validation batches cached. Check dataset path/splits.")

    base_vision_model = onnx.load(args.vision_onnx)
    base_policy_model = onnx.load(args.policy_onnx)

    if args.eval_backend == "auto":
        eval_backend = "torch" if args.provider == "cuda" else "ort"
    else:
        eval_backend = args.eval_backend

    torch_eval_device = torch.device("cuda" if (args.provider == "cuda" and torch.cuda.is_available()) else "cpu")
    if eval_backend == "ort":
        active_providers = list(providers)
        vision_sess = make_session(args.vision_onnx, providers=active_providers)
        policy_sess = make_session(args.policy_onnx, providers=active_providers)
        try:
            vision_dtype = ort_type_to_numpy_dtype(vision_sess.get_inputs()[0].type)
            policy_dtype = ort_type_to_numpy_dtype(policy_sess.get_inputs()[0].type)
            inspect_outputs(
                vision_sess,
                policy_sess,
                cached_batches[0],
                vision_output_slices,
                policy_output_slices,
                vision_dtype,
                policy_dtype,
                policy_input_shapes,
            )
            base_loss = eval_metric_value(
                vision_sess,
                policy_sess,
                cached_batches,
                vision_output_slices,
                policy_output_slices,
                vision_dtype,
                policy_dtype,
                policy_input_shapes,
                "loss",
            )
            original_metric_score = eval_metric_value(
                vision_sess,
                policy_sess,
                cached_batches,
                vision_output_slices,
                policy_output_slices,
                vision_dtype,
                policy_dtype,
                policy_input_shapes,
                args.eval_metric,
            )
        except Exception as exc:
            if ("CUDAExecutionProvider" in active_providers) and is_cuda_runtime_error(exc):
                print(f"[ORT] CUDA runtime error during baseline eval, fallback to CPU: {exc}")
                active_providers = ["CPUExecutionProvider"]
                vision_sess = make_session(args.vision_onnx, providers=active_providers)
                policy_sess = make_session(args.policy_onnx, providers=active_providers)
                vision_dtype = ort_type_to_numpy_dtype(vision_sess.get_inputs()[0].type)
                policy_dtype = ort_type_to_numpy_dtype(policy_sess.get_inputs()[0].type)
                inspect_outputs(
                    vision_sess,
                    policy_sess,
                    cached_batches[0],
                    vision_output_slices,
                    policy_output_slices,
                    vision_dtype,
                    policy_dtype,
                    policy_input_shapes,
                )
                base_loss = eval_metric_value(
                    vision_sess,
                    policy_sess,
                    cached_batches,
                    vision_output_slices,
                    policy_output_slices,
                    vision_dtype,
                    policy_dtype,
                    policy_input_shapes,
                    "loss",
                )
                original_metric_score = eval_metric_value(
                    vision_sess,
                    policy_sess,
                    cached_batches,
                    vision_output_slices,
                    policy_output_slices,
                    vision_dtype,
                    policy_dtype,
                    policy_input_shapes,
                    args.eval_metric,
                )
            else:
                raise
        providers = list(vision_sess.get_providers())
        print(f"[ORT] active_providers={providers}")
    else:
        print(f"[Eval] using torch backend on device={torch_eval_device}")
        v_torch, p_torch, v_order, p_order, v_dtype, p_dtype = build_torch_models_for_eval(
            base_vision_model,
            base_policy_model,
            device=torch_eval_device,
        )
        base_loss = eval_metric_value_torch(
            v_torch,
            p_torch,
            cached_batches,
            vision_output_slices,
            policy_output_slices,
            policy_input_shapes,
            v_order,
            p_order,
            torch_eval_device,
            v_dtype,
            p_dtype,
            "loss",
        )
        original_metric_score = eval_metric_value_torch(
            v_torch,
            p_torch,
            cached_batches,
            vision_output_slices,
            policy_output_slices,
            policy_input_shapes,
            v_order,
            p_order,
            torch_eval_device,
            v_dtype,
            p_dtype,
            args.eval_metric,
        )
        if args.inspect_only:
            print("[Inspect] torch backend selected; ORT slice inspect skipped.")

    print(f"\n[Eval] Baseline policy trajectory loss on cached val batches: {base_loss:.6f}")
    print(f"[Eval] Baseline {args.eval_metric} score on cached val batches: {original_metric_score:.6f}")

    if args.inspect_only:
        print("[Done] inspect-only mode")
        return

    weights_source_mode = "computed_in_run"
    weights_source_file: Optional[str] = None
    if args.weights_in:
        default_model_key = "policy" if args.target_model == "policy" else "vision"
        selected = load_weight_candidates(args.weights_in, default_model_key=default_model_key)
        if args.top_w > 0:
            selected = selected[: min(args.top_w, len(selected))]
        weights_source_mode = "loaded_from_file"
        weights_source_file = os.path.abspath(args.weights_in)
        print(f"[Weights] loaded {len(selected)} candidates from {args.weights_in}")
    else:
        if args.weight_selection_method == "gradient":
            mode_str = "ALL weights" if args.sample_all_weights else f"sampled (per_tensor_k={args.per_tensor_k})"
            print(f"[Weights] Using gradient-based selection (mode={mode_str})")
            selected = select_weight_candidates_by_gradient(
                models=models_for_selection,
                target_model=args.target_model,
                cached_batches=cached_batches,
                vision_output_slices=vision_output_slices,
                policy_output_slices=policy_output_slices,
                policy_input_shapes=policy_input_shapes,
                top_w=args.top_w,
                per_tensor_k=args.per_tensor_k,
                allow_bias=args.allow_bias,
                restrict=restrict,
                providers=providers,
                sample_all=args.sample_all_weights,
                torch_dtype_mode=args.gradient_torch_dtype,
            )
        else:
            print("[Weights] Using magnitude-based selection")
            selected = select_weight_candidates(
                models=models_for_selection,
                target_model=args.target_model,
                top_w=args.top_w,
                per_tensor_k=args.per_tensor_k,
                allow_bias=args.allow_bias,
                restrict=restrict,
            )
        if args.stage == "all":
            save_weight_candidates(
                weights_out_path,
                selected,
                vision_onnx_path=args.vision_onnx,
                policy_onnx_path=args.policy_onnx,
                target_model=args.target_model,
                restrict=restrict,
                allow_bias=args.allow_bias,
                top_w=args.top_w,
                per_tensor_k=args.per_tensor_k,
                selection_method=args.weight_selection_method,
            )
            weights_source_file = os.path.abspath(weights_out_path)
            print(f"[Weights] saved {len(selected)} candidates to {weights_out_path}")
    if not selected:
        raise RuntimeError("No fp16 initializer candidates selected.")

    candidates = [(mk, n, fi) for (mk, n, fi, _score) in selected]

    if eval_backend == "torch":
        ranked = rank_bits_progressive_torch(
            base_vision_model=base_vision_model,
            base_policy_model=base_policy_model,
            original_score=original_metric_score,
            cached_batches=cached_batches,
            vision_output_slices=vision_output_slices,
            policy_output_slices=policy_output_slices,
            policy_input_shapes=policy_input_shapes,
            candidates=candidates,
            bitset=bitset,
            top_b=args.top_b,
            eval_metric=args.eval_metric,
            device=torch_eval_device,
        )
    else:
        ranked = rank_bits_progressive(
            base_vision_model=base_vision_model,
            base_policy_model=base_policy_model,
            original_score=original_metric_score,
            cached_batches=cached_batches,
            vision_output_slices=vision_output_slices,
            policy_output_slices=policy_output_slices,
            policy_input_shapes=policy_input_shapes,
            candidates=candidates,
            bitset=bitset,
            top_b=args.top_b,
            eval_metric=args.eval_metric,
            providers=providers,
        )

    plan = [{"model": r["model"], "name": r["name"], "flat": r["flat"], "bit": r["bit"]} for r in ranked]
    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "vision_onnx": os.path.abspath(args.vision_onnx),
            "vision_metadata": os.path.abspath(args.vision_metadata),
            "policy_onnx": os.path.abspath(args.policy_onnx),
            "policy_metadata": os.path.abspath(args.policy_metadata),
            "target_model": args.target_model,
            "data_root": os.path.abspath(args.data_root),
            "train_index": os.path.abspath(args.train_index),
            "val_index": os.path.abspath(args.val_index),
            "num_val_batches": args.num_val_batches,
            "eval_seq_len": args.eval_seq_len,
            "top_w": args.top_w,
            "top_b": args.top_b,
            "per_tensor_k": args.per_tensor_k,
            "bitset_mode": bitset_mode,
            "bitset": bitset,
            "allow_bias": bool(args.allow_bias),
            "restrict": restrict,
            "original_metric_score": original_metric_score,
            "eval_metric": args.eval_metric,
            "search_mode": "progressive",
            "eval_backend": eval_backend,
            "weight_selection_method": args.weight_selection_method,
            "weights_source_mode": weights_source_mode,
            "weights_source_file": weights_source_file,
        },
        "ranked": ranked,
        "plan": plan,
    }

    os.makedirs(os.path.dirname(result_out_path) or ".", exist_ok=True)
    with open(result_out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[Done] Saved ranked bit plan to: {result_out_path}")
    if ranked:
        top = ranked[0]
        print(
            f"[Top-1] model={top['model']} name={top['name']} flat={top['flat']} bit={top['bit']} "
            f"metric={top['metric']} score={top['score']:.6f} "
            f"({top['original_score']:.6f} -> {top['flipped_score']:.6f})"
        )


if __name__ == "__main__":
    main()
