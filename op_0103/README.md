# OpenPilot 0.10.3 Important Bits（vision + policy）

`op_0103/important_bits_onnx.py` 支持：
1. `select-weights`：在 `driving_vision.onnx` / `driving_policy.onnx` 的 FP16 权重中选候选标量
2. `rank-bits`：progressive 搜索 bit flip，并按指标排序
3. `all`：先选权重再排 bit

## 评估目标

虽然 0.10.3 模型拆成了 vision / policy 两个 ONNX，但脚本做的是端到端评估：

- 输入图像 (`seq_input_img`) -> vision (`hidden_state`) -> policy (`plan`)
- 评分基于最终 `policy plan`（`33 x 15`）的轨迹位置（前三维）

支持指标：
- `loss`：`smooth_l1(pred_pos, gt_pos)`
- `+diffx/-diffx/+diffy/-diffy`：用于比较翻转前后输出趋势（脚本内部做 delta）

## 运行示例（使用 `.venv`）

### 1) 选权重

```bash
.venv/bin/python op_0103/important_bits_onnx.py \
  --stage select-weights \
  --target-model both \
  --top-w 500 \
  --per-tensor-k 1 \
  --weights-out op_0103/weights_candidates_0103.json
```

### 2) 排 bit（从已有候选 JSON）

```bash
./test_rank_bits_from_previous_json_0103.sh op_0103/weights_candidates_0103_top500.json
```

或直接调用：

```bash
.venv/bin/python op_0103/important_bits_onnx.py \
  --stage rank-bits \
  --weights-in op_0103/weights_candidates_0103_top500.json \
  --target-model both \
  --eval-metric +diffx \
  --bitset exponent_sign \
  --top-b 50 \
  --out op_0103/important_bits_0103.json
```

输出统一写到 `op_0103/out/`，文件名带时间戳。
