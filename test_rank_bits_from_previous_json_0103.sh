#!/bin/bash
# Rank bits for openpilot 0.10.3 split models using an existing weights-candidates JSON.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/zx/Projects/comma2k19}"

VISION_ONNX="op_0103/models/driving_vision.onnx"
VISION_METADATA="op_0103/models/driving_vision_metadata.pkl"
POLICY_ONNX="op_0103/models/driving_policy.onnx"
POLICY_METADATA="op_0103/models/driving_policy_metadata.pkl"
OUT_DIR="op_0103/out"
DEFAULT_TOP500_JSON="op_0103/weights_candidates_0103_top500.json"

WEIGHTS_JSON="${1:-${WEIGHTS_JSON:-}}"
if [ -z "$WEIGHTS_JSON" ]; then
  if [ -f "$DEFAULT_TOP500_JSON" ]; then
    WEIGHTS_JSON="$DEFAULT_TOP500_JSON"
  else
    WEIGHTS_JSON="$(ls -1t "$OUT_DIR"/weights_candidates_0103_*.json 2>/dev/null | head -n 1 || true)"
  fi
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Error: python executable not found: $PYTHON_BIN"
  exit 1
fi
if [ ! -f "$VISION_ONNX" ] || [ ! -f "$VISION_METADATA" ]; then
  echo "Error: missing vision model or metadata under op_0103/models/"
  exit 1
fi
if [ ! -f "$POLICY_ONNX" ] || [ ! -f "$POLICY_METADATA" ]; then
  echo "Error: missing policy model or metadata under op_0103/models/"
  exit 1
fi
if [ -z "$WEIGHTS_JSON" ] || [ ! -f "$WEIGHTS_JSON" ]; then
  echo "Error: No valid weights JSON found."
  echo "Hint: pass one explicitly, e.g."
  echo "  ./test_rank_bits_from_previous_json_0103.sh op_0103/out/weights_candidates_0103_*.json"
  exit 1
fi

PROVIDER="${PROVIDER:-cuda}"
NUM_VAL_BATCHES="${NUM_VAL_BATCHES:-8}"
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-20}"
TOP_B="${TOP_B:-20}"
TOP_W="${TOP_W:-0}"
BITSET="${BITSET:-exponent_sign}"
EVAL_METRIC="${EVAL_METRIC:-+diffx}"
TARGET_MODEL="${TARGET_MODEL:-both}"
OUT_JSON="${OUT_JSON:-op_0103/important_bits_0103_from_prev_weights.json}"

echo "[Input] weights_json=$WEIGHTS_JSON"
echo "[Config] provider=$PROVIDER num_val_batches=$NUM_VAL_BATCHES eval_seq_len=$EVAL_SEQ_LEN top_w=$TOP_W top_b=$TOP_B bitset=$BITSET eval_metric=$EVAL_METRIC target_model=$TARGET_MODEL"
echo "----------------------------------------------------------------------"

"$PYTHON_BIN" op_0103/important_bits_onnx.py \
  --stage rank-bits \
  --vision-onnx "$VISION_ONNX" \
  --vision-metadata "$VISION_METADATA" \
  --policy-onnx "$POLICY_ONNX" \
  --policy-metadata "$POLICY_METADATA" \
  --data-root "$DATA_ROOT" \
  --weights-in "$WEIGHTS_JSON" \
  --top-w "$TOP_W" \
  --top-b "$TOP_B" \
  --bitset "$BITSET" \
  --eval-metric "$EVAL_METRIC" \
  --target-model "$TARGET_MODEL" \
  --num-val-batches "$NUM_VAL_BATCHES" \
  --eval-seq-len "$EVAL_SEQ_LEN" \
  --provider "$PROVIDER" \
  --out "$OUT_JSON"

echo ""
echo "========================================="
echo "Rank bits completed successfully!"
echo "========================================="
echo ""
echo "Results saved in op_0103/out/"
