#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 24:00:00
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH -A loni_depedlab11

set -euo pipefail
module load cuda

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "================ JOB INFO ================"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $SLURM_NODELIST"
echo "CPUS PER TASK: $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

echo
echo "========== CPU INFO =========="
lscpu
echo "==============================="

echo
echo "========== NUMA INFO =========="
numactl --hardware
echo "================================"

echo
echo "========== GPU INFO =========="
nvidia-smi
echo "================================"

echo
echo "========== GPU TOPOLOGY =========="
nvidia-smi topo -m
echo "=================================="

echo
echo "========== CPU BINDING =========="
taskset -p $$
echo "================================="

echo
echo "========== ENVIRONMENT =========="
env | grep SLURM
echo "================================="

USER_NAME="${USER:-$(whoami)}"
if [ "$USER_NAME" = "zx" ]; then
    CONDA_BIN="${CONDA_BIN:-/home/zx/miniconda3/condabin/conda}"
    CONDA_ENV="${CONDA_ENV:-op097}"
    DATA_ROOT_DEFAULT="/home/zx/Projects/comma2k19"
elif [ "$USER_NAME" = "xzha135" ]; then
    CONDA_BIN="${CONDA_BIN:-/usr/local/packages/python/3.11.5-anaconda/bin/conda}"
    CONDA_ENV="${CONDA_ENV:-op97_py311}"
    DATA_ROOT_DEFAULT="/home/xzha135/work/comma2k19"
else
    echo "Error: Unsupported user '$USER_NAME'. Please set CONDA_BIN/CONDA_ENV/DATA_ROOT manually."
    exit 1
fi

DATA_ROOT="${DATA_ROOT:-$DATA_ROOT_DEFAULT}"
PROVIDER="${PROVIDER:-cuda}"
TARGET_MODEL="${TARGET_MODEL:-both}"
SELECTION_METHOD="${SELECTION_METHOD:-taylor-guided}"
NUM_VAL_BATCHES="${NUM_VAL_BATCHES:-128}"
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-100}"
TOP_W="${TOP_W:-10000}"
PER_TENSOR_K="${PER_TENSOR_K:-0}"
OUT_JSON="${OUT_JSON:-op_0103/out/weights_0103_taylor_seq100.json}"

echo "Activating conda environment: $CONDA_ENV"
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$CONDA_ENV"

echo ""
echo "========================================="
echo "Rank Weight Only (op_0103)"
echo "========================================="
echo ""

cd /home/xzha135/work/projects_ws/DAC/STAFI

python op_0103/important_bits_onnx.py \
  --stage select-weights \
  --provider "$PROVIDER" \
  --weight-selection-method "$SELECTION_METHOD" \
  --target-model "$TARGET_MODEL" \
  --data-root "$DATA_ROOT" \
  --num-val-batches "$NUM_VAL_BATCHES" \
  --eval-seq-len "$EVAL_SEQ_LEN" \
  --top-w "$TOP_W" \
  --per-tensor-k "$PER_TENSOR_K" \
  --weights-out "$OUT_JSON"
