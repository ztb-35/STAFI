#!/bin/bash
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 1:00:00
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH -A loni_depedlab11

set -euo pipefail
module load cuda

# Activate conda environment
USER_NAME="${USER:-$(whoami)}"
if [ "$USER_NAME" = "zx" ]; then
    CONDA_BIN="/home/zx/miniconda3/condabin/conda"
    CONDA_ENV="op097"
    DATA_ROOT="/home/zx/Projects/comma2k19"
elif [ "$USER_NAME" = "xzha135" ]; then
    CONDA_BIN="/usr/local/packages/python/3.11.5-anaconda/bin/conda"
    CONDA_ENV="op97_py311"
    DATA_ROOT="/home/xzha135/work/comma2k19"
else
    echo "Error: Unsupported user '$USER_NAME'. Please set paths manually."
    exit 1
fi


echo "Activating conda environment: $CONDA_ENV"
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$CONDA_ENV"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="/home/xzha135/work/projects_ws/DAC/STAFI/op_0103/models/driving_vision.onnx"

echo "========================================="
echo "ONNX Runtime CUDA Environment Diagnosis"
echo "========================================="
echo "PYTHON_BIN=$PYTHON_BIN"
echo "MODEL_PATH=$MODEL_PATH"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<empty>}"
echo ""

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[FAIL] python binary not found: $PYTHON_BIN"
  exit 1
fi

echo "[Step 1] Basic GPU visibility"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
else
  echo "nvidia-smi not found"
fi
echo ""

echo "[Step 2] Python + onnxruntime metadata"
ORT_INFO="$("$PYTHON_BIN" - <<'PY'
import json
import os
import pathlib
import sys

info = {
    "python_executable": sys.executable,
}
try:
    import onnxruntime as ort
    info["onnxruntime_version"] = ort.__version__
    info["available_providers"] = ort.get_available_providers()
    capi_dir = pathlib.Path(ort.__file__).resolve().parent / "capi"
    info["ort_capi_dir"] = str(capi_dir)
    info["libonnxruntime_providers_cuda_so"] = str(capi_dir / "libonnxruntime_providers_cuda.so")
    info["libonnxruntime_providers_cuda_exists"] = (capi_dir / "libonnxruntime_providers_cuda.so").exists()
except Exception as e:
    info["onnxruntime_import_error"] = f"{type(e).__name__}: {e}"
print(json.dumps(info))
PY
)"
echo "$ORT_INFO" | "$PYTHON_BIN" -m json.tool || echo "$ORT_INFO"
echo ""

ORT_CUDA_SO="$("$PYTHON_BIN" - <<'PY'
import pathlib
try:
    import onnxruntime as ort
    capi_dir = pathlib.Path(ort.__file__).resolve().parent / "capi"
    print(str(capi_dir / "libonnxruntime_providers_cuda.so"))
except Exception:
    print("")
PY
)"

if [[ -n "$ORT_CUDA_SO" && -f "$ORT_CUDA_SO" ]]; then
  echo "[Step 3] ldd for CUDA provider library"
  ldd "$ORT_CUDA_SO" | tee /tmp/ort_cuda_ldd.txt
  echo ""
  if rg -q "not found" /tmp/ort_cuda_ldd.txt 2>/dev/null; then
    echo "[WARN] Missing shared libraries detected by ldd:"
    rg "not found" /tmp/ort_cuda_ldd.txt || true
  elif grep -q "not found" /tmp/ort_cuda_ldd.txt; then
    echo "[WARN] Missing shared libraries detected by ldd:"
    grep "not found" /tmp/ort_cuda_ldd.txt || true
  else
    echo "[OK] No missing shared libraries from ldd."
  fi
  echo ""
else
  echo "[WARN] CUDA provider shared library not found; skip ldd."
  echo ""
fi

echo "[Step 4] Locate libcublasLt.so.12"
if command -v ldconfig >/dev/null 2>&1; then
  ldconfig -p | rg "libcublasLt.so.12" || true
else
  echo "ldconfig not available"
fi
for p in \
  "${CONDA_PREFIX:-}/lib/libcublasLt.so.12" \
  "/usr/local/cuda/lib64/libcublasLt.so.12" \
  "/usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.12" \
  "/usr/lib/x86_64-linux-gnu/libcublasLt.so.12"; do
  if [[ -n "$p" && -f "$p" ]]; then
    echo "found: $p"
  fi
done
echo ""

echo "[Step 5] Try creating CUDA session"
"$PYTHON_BIN" - <<PY
import traceback
import onnxruntime as ort

model_path = "$MODEL_PATH"
print("model_path:", model_path)
print("requested providers:", ["CUDAExecutionProvider", "CPUExecutionProvider"])
try:
    sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("session providers:", sess.get_providers())
    if "CUDAExecutionProvider" in sess.get_providers():
        print("[OK] CUDAExecutionProvider active.")
    else:
        print("[WARN] CUDAExecutionProvider not active; fell back to CPU.")
except Exception as e:
    print("[FAIL] InferenceSession creation failed:")
    print(f"{type(e).__name__}: {e}")
    traceback.print_exc()
PY
echo ""

echo "[Step 6] Suggested quick fix if libcublasLt.so.12 missing"
echo "Example:"
echo "  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:/usr/local/cuda/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH"
echo "  # or install matching cuda libs in env:"
echo "  # conda install -n <env> -c nvidia cuda-cublas cuda-cudart cuda-cupti cudnn"
echo ""
echo "Done."

