#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 24:00:00
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH -A loni_depedlab11
#SBATCH --job-name=rank_bit_leadprob

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

echo ""
echo "========================================="
echo "Rank Bits For Lead Probability Attack"
echo "========================================="
echo ""

cd /home/xzha135/work/projects_ws/DAC/STAFI

python op_0103/important_bits_onnx.py \
  --stage rank-bits \
  --provider cuda \
  --eval-backend torch \
  --target-model both \
  --data-root "$DATA_ROOT" \
  --weights-in op_0103/saved_out/weights_0103_taylor_seq100.json \
  --num-val-batches 2 \
  --eval-seq-len 100 \
  --top-w 500 \
  --top-b 5 \
  --per-tensor-k 0 \
  --bitset ">=5" \
  --eval-metric=-leadprob \
  --out op_0103/out/important_bits_0103_taylor_seq100_b2_leadprob_top500.json
