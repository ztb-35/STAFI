#!/bin/bash
# Test script for gradient-based weight selection

set -e

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

# Test parameters (small values for quick testing)
ONNX_PATH="op_097/models/supercombo.onnx"
METADATA_PATH="op_097/models/supercombo_metadata.pkl"

# Check if model files exist
if [ ! -f "$ONNX_PATH" ]; then
    echo "Error: ONNX model not found at $ONNX_PATH"
    exit 1
fi

if [ ! -f "$METADATA_PATH" ]; then
    echo "Error: Metadata file not found at $METADATA_PATH"
    exit 1
fi

echo "Step 1: Testing gradient-based selection (select-weights stage only)"
echo "----------------------------------------------------------------------"
python op_097/important_bits_onnx.py \
    --onnx "$ONNX_PATH" \
    --metadata "$METADATA_PATH" \
    --data-root "$DATA_ROOT" \
    --weight-selection-method gradient \
    --gradient-epsilon 1e-3 \
    --top-w 5 \
    --per-tensor-k 1 \
    --num-val-batches 1 \
    --dataloader-inner-progress \
    --eval-seq-len 5 \
    --stage select-weights \
    --provider cpu \
    --restrict "vision"

echo ""
echo "Step 2: Comparing with magnitude-based selection"
echo "----------------------------------------------------------------------"
python op_097/important_bits_onnx.py \
    --onnx "$ONNX_PATH" \
    --metadata "$METADATA_PATH" \
    --data-root "$DATA_ROOT" \
    --weight-selection-method magnitude \
    --top-w 5 \
    --per-tensor-k 1 \
    --stage select-weights \
    --restrict "vision"

echo ""
echo "========================================="
echo "Test completed successfully!"
echo "========================================="
echo ""
echo "Check the output files in op_097/out/ to see the selected weights."
echo "The 'score' field represents:"
echo "  - For gradient method: |gradient| magnitude"
echo "  - For magnitude method: |weight| absolute value"
