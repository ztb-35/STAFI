# OpenPilot 0.9.7 Important Bits 操作手册

本目录提供三步流程：
1. 选权重候选（`select-weights`）
2. 排名 bit（`rank-bits`）
3. 按计划翻转 ONNX 并导出新模型（`export_flipped_onnx.py`）

## 1. 环境与数据

- 项目根目录：`/home/xzha135/work/projects_ws/DAC/STAFI`
- conda 环境：`/work/xzha135/conda_envs/op97_py311`
- 数据集目录：`/home/xzha135/work/comma2k19`
- 模型文件：
  - `op_097/models/supercombo.onnx`
  - `op_097/models/supercombo_metadata.pkl`

说明：
- 脚本会自动生成 split 文件（若不存在）到 `op_097/splits/`
- 输入设置已固定为：
  - `desire` 全 0
  - `traffic_convention = [1, 0]`（右手交通）
  - `input_imgs` 与 `big_input_imgs` 使用相同输入
  - 循环推理时自动滚动更新 `feature_buffer`

## 2. 第一步：选权重候选

```bash
/work/xzha135/conda_envs/op97_py311/bin/python op_097/important_bits_onnx.py \
  --stage select-weights \
  --onnx op_097/models/supercombo.onnx \
  --provider auto \
  --top-w 50 \
  --per-tensor-k 1 \
  --weights-out op_097/weights_candidates_097.json
```

输出：
- `op_097/out/weights_candidates_097_YYYYMMDD-HHMMSS.json`

## 3. 第二步：排名 bits

### 3.1 原始 loss 排名（兼容旧逻辑）

```bash
/work/xzha135/conda_envs/op97_py311/bin/python op_097/important_bits_onnx.py \
  --stage rank-bits \
  --weights-in op_097/out/weights_candidates_097_YYYYMMDD-HHMMSS.json \
  --provider auto \
  --data-root /home/xzha135/work/comma2k19 \
  --num-val-batches 3 \
  --eval-seq-len 20 \
  --bitset exponent_sign \
  --eval-metric loss \
  --top-b 50 \
  --out op_097/important_bits_097_loss.json
```

### 3.2 按轨迹差分排名（翻转后 - 翻转前）

可选 `--eval-metric`：
- `+diffx`：`mean(x_flipped - x_clean)`
- `-diffx`：`mean(-(x_flipped - x_clean))`
- `+diffy`：`mean(y_flipped - y_clean)`
- `-diffy`：`mean(-(y_flipped - y_clean))`

示例（按 `+diffx`）：

```bash
/work/xzha135/conda_envs/op97_py311/bin/python op_097/important_bits_onnx.py \
  --stage rank-bits \
  --weights-in op_097/out/weights_candidates_097_YYYYMMDD-HHMMSS.json \
  --provider auto \
  --data-root /home/xzha135/work/comma2k19 \
  --num-val-batches 3 \
  --eval-seq-len 20 \
  --bitset exponent_sign \
  --eval-metric +diffx \
  --top-b 50 \
  --out op_097/important_bits_097_diffx.json
```

输出 JSON 关键字段：
- `meta.eval_metric`
- `meta.bitset_mode`
- `ranked[].original_score / ranked[].flipped_score / ranked[].score`
- `plan[]`（导出翻转模型时直接用）

说明：
- 当前 `rank-bits` 使用 progressive 搜索：每一步会在“已翻转模型”基础上继续搜索下一个 bit。
- 非有限值（`NaN/Inf`）候选会被自动跳过，不写入最终 `ranked`。

注意：
- `important_bits_onnx.py` 会自动给 `--weights-out` 和 `--out` 增加时间戳后缀，并统一写入 `op_097/out/`。
- 例如传入 `--out op_097/important_bits_097_diffx.json`，实际输出类似：
  - `op_097/out/important_bits_097_diffx_20260212-135959.json`
- `--provider` 可选 `auto/cpu/cuda`：
  - `auto`：优先 CUDA（可用时），否则 CPU
  - `cpu`：强制 CPU
  - `cuda`：强制 CUDA（环境不支持会报错）

## 4. 第三步：导出翻转后的 ONNX

```bash
/work/xzha135/conda_envs/op97_py311/bin/python op_097/export_flipped_onnx.py \
  --onnx-in op_097/models/supercombo.onnx \
  --plan-json op_097/important_bits_097_diffx.json \
  --source-key plan \
  --onnx-out op_097/models/supercombo_flipped.onnx
```

常用参数：
- `--source-key plan|ranked`
- `--topk N`：只应用前 N 个 flips
- `--strict`：遇到无效 name/index 直接报错退出

## 5. 一步跑完（不拆分）

```bash
/work/xzha135/conda_envs/op97_py311/bin/python op_097/important_bits_onnx.py \
  --stage all \
  --provider auto \
  --data-root /home/xzha135/work/comma2k19 \
  --num-val-batches 3 \
  --eval-seq-len 20 \
  --top-w 50 \
  --top-b 50 \
  --bitset exponent_sign \
  --eval-metric loss \
  --weights-out op_097/weights_candidates_097.json \
  --out op_097/important_bits_097_loss.json
```

## 6. 关于 bit 编号（重要）

当前 ONNX 权重是 `FLOAT16`，位定义是：
- `0..9`：mantissa
- `10..14`：exponent
- `15`：sign

所以 `bitset: [15]` 表示翻转符号位（不是 fp32 的 bit31）。
