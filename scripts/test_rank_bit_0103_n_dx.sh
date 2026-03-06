#!/bin/bash
###
 # @Descripttion:
 # @version: 1.0
 # @Author: Zx
 # @Email: ureinsecure@outlook.com
 # @Date: 2026-03-05 14:06:12
 # @LastEditors: Zx
 # @LastEditTime: 2026-03-05 14:49:16
 # @FilePath: /STAFI/scripts/test_rank_bit_0103_n_dx.sh
###
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 72:00:00
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

echo ""
echo "========================================="
echo "Testing Gradient-Based Weight Selection"
echo "========================================="
echo ""

cd /home/xzha135/work/projects_ws/DAC/STAFI

# 3) 仅 rank-bits（使用已有 weights JSON）
python op_0103/important_bits_onnx.py \
  --stage rank-bits \
  --provider cuda \
  --eval-backend torch \
  --target-model both \
  --data-root $DATA_ROOT \
  --weights-in op_0103/weights_3000_with_bias.json \
  --num-val-batches 8 \
  --eval-seq-len 20 \
  --top-w 100 \
  --top-b 5 \
  --bitset ">=8" \
  --eval-metric=-diffx \
  --out op_0103/out/important_bits_0103_n_dx.json
