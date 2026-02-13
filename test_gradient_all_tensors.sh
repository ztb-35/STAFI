#!/bin/bash
# Test gradient selection on ALL tensors (no restriction)

set -e

# Activate conda environment
echo "Activating conda environment: op097"
eval "$(/home/zx/miniconda3/condabin/conda shell.bash hook)"
conda activate op097

echo ""
echo "========================================="
echo "Gradient Selection - ALL Tensors"
echo "========================================="
echo ""
echo "WARNING: This will process ~313 FP16 tensors"
echo "Expected time: ~30 minutes"
echo ""

# Test parameters
ONNX_PATH="op_097/models/supercombo.onnx"
METADATA_PATH="op_097/models/supercombo_metadata.pkl"
DATA_ROOT="/home/zx/Projects/comma2k19"

# Check if model files exist
if [ ! -f "$ONNX_PATH" ]; then
    echo "Error: ONNX model not found at $ONNX_PATH"
    exit 1
fi

if [ ! -f "$METADATA_PATH" ]; then
    echo "Error: Metadata file not found at $METADATA_PATH"
    exit 1
fi

echo "Starting gradient-based selection on ALL tensors..."
echo "----------------------------------------------------------------------"
# python op_097/important_bits_onnx.py \
#     --onnx "$ONNX_PATH" \
#     --metadata "$METADATA_PATH" \
#     --data-root "$DATA_ROOT" \
#     --weight-selection-method gradient \
#     --gradient-epsilon 1e-3 \
#     --top-w 500 \
#     --sample-all-weights \
#     --num-val-batches 128 \
#     --eval-seq-len 20 \
#     --stage select-weights \
#     --provider auto
python op_097/important_bits_onnx.py \
    --onnx "$ONNX_PATH" \
    --metadata "$METADATA_PATH" \
    --data-root "$DATA_ROOT" \
    --weight-selection-method gradient \
    --gradient-epsilon 1e-3 \
    --top-w 500 \
    --per-tensor-k 20\
    --num-val-batches 1 \
    --eval-seq-len 2 \
    --stage select-weights \
    --provider auto
echo ""
echo "========================================="
echo "Test completed successfully!"
echo "========================================="
echo ""
echo "Results saved in op_097/out/"
