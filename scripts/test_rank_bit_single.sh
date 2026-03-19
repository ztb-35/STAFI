#!/bin/bash
###
 # @Descripttion: 
 # @version: 1.0
 # @Author: Zx
 # @Email: ureinsecure@outlook.com
 # @Date: 2026-03-04 16:35:28
 # @LastEditors: Zx
 # @LastEditTime: 2026-03-05 18:35:18
 # @FilePath: /STAFI/scripts/test_rank_bit_single.sh
### 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 24:00:00
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH -A loni_depedlab11

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

echo ""
echo "========================================="
echo "Testing Gradient-Based Weight Selection"
echo "========================================="
echo ""

cd /home/xzha135/work/projects_ws/DAC/STAFI

#python op_0103/important_bits_onnx.py \
#  --stage all \
#  --provider cuda \
#  --eval-backend auto \
#  --weight-selection-method gradient \
#  --target-model both \
#  --data-root $DATA_ROOT \
#  --num-val-batches 1 \
#  --eval-seq-len 2 \
#  --top-w 5 \
#  --per-tensor-k 1 \
#  --top-b 1 \
#  --bitset exponent_sign \
#  --eval-metric loss \
#  --weights-out op_0103/weights_candidates_0103_smallbatch_gpu.json \
#  --out op_0103/important_bits_0103_smallbatch_gpu.json

# 2) 正式跑（示例参数，可按资源调整）
# python op_0103/important_bits_onnx.py \
#    --stage all \
#    --provider cuda \
#    --eval-backend ort \
#    --weight-selection-method gradient \
#    --target-model both \
#    --data-root $DATA_ROOT \
#    --num-val-batches 128 \
#    --eval-seq-len 20 \
#    --top-w 500 \
#    --per-tensor-k 0 \
#    --top-b 5 \
#    --bitset exponent_sign \
#    --eval-metric +diffx \
#    --weights-out op_0103/out/weights_candidates_0103_gpu.json \
#    --out op_0103/out/important_bits_0103_gpu.json

# 3) 仅 rank-bits（使用已有 weights JSON）
python op_0103/rank_single_bitflip_losses.py \
  --provider cuda \
  --eval-backend torch \
  --target-model both \
  --data-root $DATA_ROOT \
  --weights-in op_0103/saved_out/weights_0103_taylor_seq100.json \
  --num-val-batches 8 \
  --eval-seq-len 20 \
  --top-w 3000 \
  --top-k 10000 \
  --per-tensor-k 0 \
  --bitset ">=6" \
  --metrics=loss_mse,+diffx,-diffx,+diffy,-diffy \
  --out op_0103/out/single_flip_taylor_seq100.json
