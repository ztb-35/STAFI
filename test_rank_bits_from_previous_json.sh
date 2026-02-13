#!/bin/bash
# Rank bits using an existing weights-candidates JSON (skip select-weights stage)

set -euo pipefail

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
echo "Rank Bits From Existing Weights JSON"
echo "========================================="
echo ""

ONNX_PATH="op_097/models/supercombo.onnx"
METADATA_PATH="op_097/models/supercombo_metadata.pkl"
OUT_DIR="op_097/out"

# Optional input:
#   1) first arg: explicit JSON path
#   2) WEIGHTS_JSON env var
#   3) fallback: latest weights_candidates_097_*.json in op_097/out
WEIGHTS_JSON="${1:-${WEIGHTS_JSON:-}}"
if [ -z "$WEIGHTS_JSON" ]; then
    WEIGHTS_JSON="$(ls -1t "$OUT_DIR"/weights_candidates_097_*.json 2>/dev/null | head -n 1 || true)"
fi

if [ ! -f "$ONNX_PATH" ]; then
    echo "Error: ONNX model not found at $ONNX_PATH"
    exit 1
fi
if [ ! -f "$METADATA_PATH" ]; then
    echo "Error: Metadata file not found at $METADATA_PATH"
    exit 1
fi
if [ -z "$WEIGHTS_JSON" ] || [ ! -f "$WEIGHTS_JSON" ]; then
    echo "Error: No valid weights JSON found."
    echo "Hint: pass a path as first argument, e.g."
    echo "  ./test_rank_bits_from_previous_json.sh op_097/out/weights_candidates_097_YYYYMMDD-HHMMSS.json"
    exit 1
fi

PROVIDER="${PROVIDER:-cuda}"
NUM_VAL_BATCHES="${NUM_VAL_BATCHES:-3}"
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-20}"
TOP_B="${TOP_B:-5}"
BITSET="${BITSET:-exponent_sign}"
EVAL_METRIC="${EVAL_METRIC:-loss}"
OUT_JSON="${OUT_JSON:-op_097/important_bits_097_from_prev_weights.json}"

echo "[Input] weights_json=$WEIGHTS_JSON"
echo "[Config] provider=$PROVIDER num_val_batches=$NUM_VAL_BATCHES eval_seq_len=$EVAL_SEQ_LEN top_b=$TOP_B bitset=$BITSET eval_metric=$EVAL_METRIC"
echo "----------------------------------------------------------------------"

python op_097/important_bits_onnx.py \
    --stage rank-bits \
    --onnx "$ONNX_PATH" \
    --metadata "$METADATA_PATH" \
    --data-root "$DATA_ROOT" \
    --weights-in "$WEIGHTS_JSON" \
    --top-w 0 \
    --top-b "$TOP_B" \
    --bitset "$BITSET" \
    --eval-metric "$EVAL_METRIC" \
    --num-val-batches "$NUM_VAL_BATCHES" \
    --eval-seq-len "$EVAL_SEQ_LEN" \
    --provider "$PROVIDER" \
    --out "$OUT_JSON"

echo ""
echo "========================================="
echo "Rank bits completed successfully!"
echo "========================================="
echo ""
echo "Results saved in op_097/out/"
