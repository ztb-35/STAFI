#!/bin/bash
# Test script for gradient-based weight selection

set -e

# Activate conda environment
echo "Activating conda environment: op97_py311"
eval "$(/usr/local/packages/python/3.11.5-anaconda/bin/conda shell.bash hook)"
conda activate op97_py311

echo ""
echo "========================================="
echo "Testing Gradient-Based Weight Selection"
echo "========================================="
echo ""

# Test parameters (small values for quick testing)
ONNX_PATH="op_097/models/supercombo.onnx"
METADATA_PATH="op_097/models/supercombo_metadata.pkl"
DATA_ROOT="/home/xzha135/work/comma2k19"

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
