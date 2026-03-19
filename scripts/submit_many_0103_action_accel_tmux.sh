#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 24:00:00
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH -A loni_depedlab11
#SBATCH --job-name=rank_bit_accel_multi

set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module load cuda || true
fi

USER_NAME="${USER:-$(whoami)}"
if [ "$USER_NAME" = "zx" ]; then
    CONDA_BIN="/home/zx/miniconda3/condabin/conda"
    CONDA_ENV="op097"
    DATA_ROOT="/home/zx/Projects/comma2k19"
    REPO_ROOT="/home/zx/Projects/DAC/STAFI"
elif [ "$USER_NAME" = "xzha135" ]; then
    CONDA_BIN="/usr/local/packages/python/3.11.5-anaconda/bin/conda"
    CONDA_ENV="op97_py311"
    DATA_ROOT="/home/xzha135/work/comma2k19"
    REPO_ROOT="/home/xzha135/work/projects_ws/DAC/STAFI"
else
    echo "Error: Unsupported user '$USER_NAME'. Please set paths manually."
    exit 1
fi

COUNT="${COUNT:-4}"
NUM_VAL_BATCHES="${NUM_VAL_BATCHES:-1}"
TOP_W="${TOP_W:-100}"
TOP_B="${TOP_B:-5}"
BITSET="${BITSET:-">=5"}"

TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$REPO_ROOT/logs/tmux"
mkdir -p "$LOG_DIR"

echo "================ JOB INFO ================"
echo "JOB ID: ${SLURM_JOB_ID:-local}"
echo "NODE: ${SLURM_NODELIST:-local}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "COUNT: $COUNT"
echo "NUM_VAL_BATCHES: $NUM_VAL_BATCHES"
echo "TOP_W: $TOP_W"
echo "CPUS PER TASK: ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "=========================================="

cd "$REPO_ROOT"
echo "Activating conda environment: $CONDA_ENV"
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$CONDA_ENV"

pids=()

for i in $(seq 1 "$COUNT"); do
    RUN_ID="$(printf '%02d' "$i")"
    LOG_FILE="$LOG_DIR/op0103_accel_multi_${TS}_run${RUN_ID}.log"
    OUT_JSON="op_0103/out/important_bits_0103_taylor_seq100_b${NUM_VAL_BATCHES}_action_accel_top${TOP_W}_${TS}_run${RUN_ID}.json"

    echo "[Launch] run=${RUN_ID} log=${LOG_FILE} out=${OUT_JSON}"

    (
        python op_0103/important_bits_onnx.py \
          --stage rank-bits \
          --provider cuda \
          --eval-backend torch \
          --target-model both \
          --data-root "$DATA_ROOT" \
          --weights-in op_0103/saved_out/weights_0103_taylor_seq100.json \
          --num-val-batches "$NUM_VAL_BATCHES" \
          --eval-seq-len 100 \
          --top-w "$TOP_W" \
          --top-b "$TOP_B" \
          --per-tensor-k 0 \
          --bitset "$BITSET" \
          --eval-metric=action.desiredAcceleration \
          --out "$OUT_JSON"
    ) >"$LOG_FILE" 2>&1 &

    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "[Done] at least one run failed"
    exit 1
fi

echo "[Done] all runs finished successfully"
