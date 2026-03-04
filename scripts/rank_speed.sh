#!/bin/bash
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 06:00:00
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH -A loni_depedlab11

module load cuda

set -euo pipefail

USER_NAME="${USER:-$(whoami)}"
if [ "$USER_NAME" = "zx" ]; then
    CONDA_BIN_DEFAULT="/home/zx/miniconda3/condabin/conda"
    CONDA_ENV_DEFAULT="op097"
    DATA_ROOT_DEFAULT="/home/zx/Projects/comma2k19"
    WEIGHTS_JSON_DEFAULT="/home/zx/Downloads/weights_candidates_0103_gpu_20260303-124416.json"
    REPO_ROOT_DEFAULT="/home/zx/Projects/DAC/STAFI"
elif [ "$USER_NAME" = "xzha135" ]; then
    CONDA_BIN_DEFAULT="/usr/local/packages/python/3.11.5-anaconda/bin/conda"
    CONDA_ENV_DEFAULT="cu12"
    DATA_ROOT_DEFAULT="/home/xzha135/work/comma2k19"
    WEIGHTS_JSON_DEFAULT="/home/xzha135/work/projects_ws/DAC/STAFI/op_0103/out/weights_candidates_0103_batch8.json"
    REPO_ROOT_DEFAULT="/home/xzha135/work/projects_ws/DAC/STAFI"
else
    echo "Error: Unsupported user '$USER_NAME'. Please set paths manually."
    exit 1
fi

REPO_ROOT="${REPO_ROOT:-$REPO_ROOT_DEFAULT}"
if [ ! -d "$REPO_ROOT" ]; then
    echo "Error: Repo root not found: $REPO_ROOT"
    exit 1
fi
cd "$REPO_ROOT"

CONDA_BIN="${CONDA_BIN:-$CONDA_BIN_DEFAULT}"
CONDA_ENV="${CONDA_ENV:-$CONDA_ENV_DEFAULT}"
if [ ! -x "$CONDA_BIN" ]; then
    echo "Error: conda binary not found/executable: $CONDA_BIN"
    exit 1
fi

echo "Using conda env: $CONDA_ENV"
# Resolve env prefix without shell-level 'conda activate' (works with strict HPC conda wrappers)
CONDA_ENV_PREFIX=""
if [ -d "$CONDA_ENV" ]; then
    CONDA_ENV_PREFIX="$CONDA_ENV"
else
    CONDA_ENV_PREFIX="$("$CONDA_BIN" env list | awk -v env="$CONDA_ENV" '$1==env {print $NF}' | head -n 1 || true)"
fi
if [ -z "$CONDA_ENV_PREFIX" ] || [ ! -d "$CONDA_ENV_PREFIX" ]; then
    echo "Error: Failed to resolve conda env prefix for '$CONDA_ENV'."
    echo "Hint: run '$CONDA_BIN env list' to check available envs."
    exit 1
fi

PYTHON_BIN="$CONDA_ENV_PREFIX/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: Python not found in env: $PYTHON_BIN"
    exit 1
fi
export CONDA_PREFIX="$CONDA_ENV_PREFIX"

# Prefer CUDA/cuDNN libs from current env to avoid system CUDA conflicts.
PY_PREFIX="$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')"
PY_VER="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
CUDA_LIB_PREFIX=""
for d in \
  "$PY_PREFIX/lib/python$PY_VER/site-packages/nvidia/cudnn/lib" \
  "$PY_PREFIX/lib/python$PY_VER/site-packages/nvidia/cublas/lib" \
  "$PY_PREFIX/lib/python$PY_VER/site-packages/nvidia/cuda_runtime/lib" \
  "$PY_PREFIX/lib/python$PY_VER/site-packages/nvidia/cuda_nvrtc/lib"; do
  if [ -d "$d" ]; then
    if [ -z "$CUDA_LIB_PREFIX" ]; then
      CUDA_LIB_PREFIX="$d"
    else
      CUDA_LIB_PREFIX="$CUDA_LIB_PREFIX:$d"
    fi
  fi
done
if [ -n "$CUDA_LIB_PREFIX" ]; then
  export LD_LIBRARY_PATH="$CUDA_LIB_PREFIX:${LD_LIBRARY_PATH:-}"
fi

echo ""
echo "========================================="
echo "Benchmark op_0103 rank-bits (ORT vs Torch)"
echo "========================================="
echo ""

OUT_DIR="op_0103/out"
mkdir -p "$OUT_DIR"
if [ ! -d "$OUT_DIR" ]; then
    echo "Error: failed to create output directory: $OUT_DIR"
    exit 1
fi

WEIGHTS_JSON="${WEIGHTS_JSON:-$WEIGHTS_JSON_DEFAULT}"

if [ -z "$WEIGHTS_JSON" ] || [ ! -f "$WEIGHTS_JSON" ]; then
    echo "Error: No valid weights JSON found."
    echo "Expected hardcoded path: $WEIGHTS_JSON_DEFAULT"
    echo "You can override by env, e.g.:"
    echo "  sbatch --export=ALL,WEIGHTS_JSON=/abs/path/your_weights.json scripts/rank_speed.sh"
    exit 1
fi

VISION_ONNX="${VISION_ONNX:-op_0103/models/driving_vision.onnx}"
VISION_META="${VISION_META:-op_0103/models/driving_vision_metadata.pkl}"
POLICY_ONNX="${POLICY_ONNX:-op_0103/models/driving_policy.onnx}"
POLICY_META="${POLICY_META:-op_0103/models/driving_policy_metadata.pkl}"

for f in "$VISION_ONNX" "$VISION_META" "$POLICY_ONNX" "$POLICY_META"; do
    if [ ! -f "$f" ]; then
        echo "Error: required file not found: $f"
        exit 1
    fi
done

DATA_ROOT="${DATA_ROOT:-$DATA_ROOT_DEFAULT}"
NUM_VAL_BATCHES="${NUM_VAL_BATCHES:-8}"
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-20}"
TOP_W="${TOP_W:-500}"
TOP_B="${TOP_B:-20}"
BITSET="${BITSET:-exponent_sign}"
EVAL_METRIC="${EVAL_METRIC:-+diffx}"
TARGET_MODEL="${TARGET_MODEL:-both}"
REPEATS="${REPEATS:-3}"
WARMUP="${WARMUP:-1}"
RUN_GPU="${RUN_GPU:-1}"
RUN_CPU="${RUN_CPU:-1}"

TS="$(date +%Y%m%d-%H%M%S)"
CSV_PATH="$OUT_DIR/bench_rankbits_0103_${TS}.csv"
SUMMARY_PATH="$OUT_DIR/bench_rankbits_0103_${TS}.summary.txt"

echo "provider,backend,phase,iter,exit_code,real_sec,user_sec,sys_sec,max_rss_kb,time_file,log_file,out_json" > "$CSV_PATH"

timestamp() {
    date +"%F %T"
}

run_one() {
    local provider="$1"
    local backend="$2"
    local phase="$3"
    local iter="$4"

    local out_json="$OUT_DIR/bench_${provider}_${backend}_${phase}${iter}_${TS}.json"
    local log_file="$OUT_DIR/bench_${provider}_${backend}_${phase}${iter}_${TS}.log"
    local time_file="$OUT_DIR/bench_${provider}_${backend}_${phase}${iter}_${TS}.time"

    echo "[$(timestamp)] Running provider=$provider backend=$backend phase=$phase iter=$iter"

    set +e
    /usr/bin/time -f "real_sec=%e user_sec=%U sys_sec=%S max_rss_kb=%M" -o "$time_file" \
        "$PYTHON_BIN" op_0103/important_bits_onnx.py \
            --stage rank-bits \
            --provider "$provider" \
            --eval-backend "$backend" \
            --target-model "$TARGET_MODEL" \
            --vision-onnx "$VISION_ONNX" \
            --vision-metadata "$VISION_META" \
            --policy-onnx "$POLICY_ONNX" \
            --policy-metadata "$POLICY_META" \
            --data-root "$DATA_ROOT" \
            --weights-in "$WEIGHTS_JSON" \
            --num-val-batches "$NUM_VAL_BATCHES" \
            --eval-seq-len "$EVAL_SEQ_LEN" \
            --top-w "$TOP_W" \
            --top-b "$TOP_B" \
            --bitset "$BITSET" \
            --eval-metric "$EVAL_METRIC" \
            --out "$out_json" \
            > "$log_file" 2>&1
    local exit_code=$?
    set -e

    local real_sec user_sec sys_sec max_rss
    real_sec="$(awk -F'[ =]' '/real_sec=/{print $2}' "$time_file" 2>/dev/null || true)"
    user_sec="$(awk -F'[ =]' '/user_sec=/{print $4}' "$time_file" 2>/dev/null || true)"
    sys_sec="$(awk -F'[ =]' '/sys_sec=/{print $6}' "$time_file" 2>/dev/null || true)"
    max_rss="$(awk -F'[ =]' '/max_rss_kb=/{print $8}' "$time_file" 2>/dev/null || true)"

    if [ -z "$real_sec" ]; then real_sec="NA"; fi
    if [ -z "$user_sec" ]; then user_sec="NA"; fi
    if [ -z "$sys_sec" ]; then sys_sec="NA"; fi
    if [ -z "$max_rss" ]; then max_rss="NA"; fi

    # important_bits_onnx.py appends its own timestamp to --out under op_0103/out.
    # Record the actual generated JSON path when possible.
    local actual_out_json="$out_json"
    if [ ! -f "$actual_out_json" ]; then
        actual_out_json="$(ls -1t "${out_json%.json}"_*.json 2>/dev/null | head -n 1 || true)"
        if [ -z "$actual_out_json" ]; then
            actual_out_json="NA"
        fi
    fi

    echo "$provider,$backend,$phase,$iter,$exit_code,$real_sec,$user_sec,$sys_sec,$max_rss,$time_file,$log_file,$actual_out_json" >> "$CSV_PATH"

    if [ "$exit_code" -ne 0 ]; then
        echo "[$(timestamp)] FAIL provider=$provider backend=$backend phase=$phase iter=$iter (exit=$exit_code)"
        echo "  log:  $log_file"
        return "$exit_code"
    fi

    echo "[$(timestamp)] DONE provider=$provider backend=$backend phase=$phase iter=$iter real_sec=$real_sec"
    return 0
}

has_gpu=0
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi -L >/dev/null 2>&1; then
        has_gpu=1
    fi
fi

echo "[Config] weights_json=$WEIGHTS_JSON"
echo "[Config] data_root=$DATA_ROOT"
echo "[Env] conda_bin=$CONDA_BIN"
echo "[Env] conda_env=$CONDA_ENV"
echo "[Env] python=$PYTHON_BIN"
if [ -n "$CUDA_LIB_PREFIX" ]; then
    echo "[Env] prepend_cuda_libs=$CUDA_LIB_PREFIX"
fi
echo "[Config] num_val_batches=$NUM_VAL_BATCHES eval_seq_len=$EVAL_SEQ_LEN top_w=$TOP_W top_b=$TOP_B"
echo "[Config] bitset=$BITSET eval_metric=$EVAL_METRIC target_model=$TARGET_MODEL"
echo "[Config] warmup=$WARMUP repeats=$REPEATS run_gpu=$RUN_GPU run_cpu=$RUN_CPU has_gpu=$has_gpu"
echo "[Output] csv=$CSV_PATH"
echo ""

if [ "$RUN_GPU" = "1" ]; then
    if [ "$has_gpu" = "1" ]; then
        for backend in ort torch; do
            for i in $(seq 1 "$WARMUP"); do
                run_one cuda "$backend" warmup "$i" || true
            done
            for i in $(seq 1 "$REPEATS"); do
                run_one cuda "$backend" measure "$i" || true
            done
        done
    else
        echo "[WARN] RUN_GPU=1 but no visible GPU, skip cuda benchmarks"
    fi
fi

if [ "$RUN_CPU" = "1" ]; then
    for backend in ort torch; do
        for i in $(seq 1 "$WARMUP"); do
            run_one cpu "$backend" warmup "$i" || true
        done
        for i in $(seq 1 "$REPEATS"); do
            run_one cpu "$backend" measure "$i" || true
        done
    done
fi

# Build summary with average real_sec over successful measured runs
awk -F',' '
BEGIN {
  print "rank-bits speed benchmark summary";
  print "--------------------------------";
}
NR==1 { next }
$3=="measure" && $5==0 && $6!="NA" {
  key=$1"/"$2;
  sum[key]+=$6;
  cnt[key]+=1;
}
{
  keyall=$1"/"$2;
  total[keyall]+=1;
  if ($5==0) ok[keyall]+=1;
  else fail[keyall]+=1;
}
END {
  printf("%-12s %-8s %-8s %-10s\n", "combo", "ok", "fail", "avg_sec");
  combos[1]="cuda/ort";
  combos[2]="cuda/torch";
  combos[3]="cpu/ort";
  combos[4]="cpu/torch";
  for (i=1; i<=4; i++) {
    k=combos[i];
    o=(ok[k]+0);
    f=(fail[k]+0);
    if (cnt[k]>0) a=sum[k]/cnt[k]; else a=-1;
    if (a>=0) printf("%-12s %-8d %-8d %.6f\n", k, o, f, a);
    else printf("%-12s %-8d %-8d NA\n", k, o, f);
  }
}
' "$CSV_PATH" | tee "$SUMMARY_PATH"

echo ""
echo "[Done] Benchmark finished"
echo "CSV:     $CSV_PATH"
echo "Summary: $SUMMARY_PATH"
echo ""

