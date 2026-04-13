#!/bin/bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/home/zx/miniconda3/condabin/conda}"
CONDA_ENV="${CONDA_ENV:-op097}"
DATA_ROOT="${DATA_ROOT:-/home/zx/Projects/comma2k19}"
REPO_ROOT="${REPO_ROOT:-/home/zx/Projects/DAC/STAFI}"

BATCH_SIZE="${BATCH_SIZE:-1}"
RECURRENT_NUM="${RECURRENT_NUM:-100}"
NUM_VAL_BATCHES="${NUM_VAL_BATCHES:-2}"
FLIP_COUNT="${FLIP_COUNT:-10}"
EVAL_MODE="${EVAL_MODE:-independent}"
INPUT_JSON="${INPUT_JSON:-}"
CKPT="${CKPT:-}"
OUT_JSON="${OUT_JSON:-$REPO_ROOT/op_089/out/candidate_bits_eval_089.json}"

if [ -z "$INPUT_JSON" ]; then
    echo "Error: set INPUT_JSON to the candidate json file."
    exit 1
fi

cd "$REPO_ROOT"
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$CONDA_ENV"

CMD=(
    python op_089/eval_candidate_bits.py
    --data-root "$DATA_ROOT"
    --batch-size "$BATCH_SIZE"
    --recurrent-num "$RECURRENT_NUM"
    --num-val-batches "$NUM_VAL_BATCHES"
    --input-json "$INPUT_JSON"
    --flip-count "$FLIP_COUNT"
    --eval-mode "$EVAL_MODE"
    --out "$OUT_JSON"
)

if [ -n "$CKPT" ]; then
    CMD+=(--ckpt "$CKPT")
fi

"${CMD[@]}"
