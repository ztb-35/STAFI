# OpenPilot 0.10.3 Important Bits（vision + policy）

`op_0103/important_bits_onnx.py` 支持：
1. `select-weights`：在 `driving_vision.onnx` / `driving_policy.onnx` 的 FP16 权重中选候选标量
2. `rank-bits`：progressive 搜索 bit flip，并按指标排序
3. `all`：先选权重再排 bit

默认 `--weight-selection-method` 现在是 `taylor-guided`，即按 `|grad * w|` 选择候选权重。

## 评估目标

虽然 0.10.3 模型拆成了 vision / policy 两个 ONNX，但脚本做的是端到端评估：

- 输入图像 (`seq_input_img`) -> vision (`hidden_state`) -> policy (`plan`)
- 评分基于最终 `policy plan`（`33 x 15`）的轨迹位置（前三维）

支持指标：
- `loss`：`smooth_l1(pred_pos, gt_pos)`
- `action.desiredCurvature/-action.desiredCurvature`：按 `modeld.py` 风格，用 `plan` 的 yaw/yaw_rate 对 `action_t` 做插值并加上 `prev_action` 平滑
- `action.desiredAcceleration/-action.desiredAcceleration`：按 `modeld.py` 风格，用 `plan` 的 velocity/acceleration 对 `action_t` 做插值并加上 `prev_action` 平滑
- `modelV2.position.x`：`fill_model_msg.py` 里的 `modelV2.position.x`，这里取 33 个 timestep 的平均值
- `modelV2.velocity.x`：`fill_model_msg.py` 里的 `modelV2.velocity.x`，这里取 33 个 timestep 的平均值
- `modelV2.acceleration.x`：`fill_model_msg.py` 里的 `modelV2.acceleration.x`，这里取 33 个 timestep 的平均值
- `+diffx/-diffx/+diffy/-diffy`：33 个 timestep 上的平均方向偏移
- `+endx/-endx/+endy/-endy`：最后一个 timestep 的方向偏移
- `+lanex/-lanex/+laney/-laney`：vision `lane_lines` 相对原始模型输出的平均位置偏移
- `+leadx/-leadx/+leady/-leady`：vision `lead` 中最高 `lead_prob` 模式，相对原始模型输出的平均位置偏移
- `+leadprob/-leadprob`：vision `lead_prob`（sigmoid 后）相对原始模型输出增大/减小；若想把 lead probability 压到 0，用 `-leadprob`

在 rank-bits 阶段，候选 bit flip 现在还会额外过滤：
- `policy plan` 不能出现 `NaN/Inf`
- `vision` 的 `meta / lane_lines / lane_lines_prob / road_edges / lead / lead_prob / hidden_state` 必须保持有限值，可继续解析

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

### 3) 按已有 `important_bits` JSON 导出翻转后的 ONNX

例如基于 `op_0103/out/important_bits_0103_taylor_seq100_b2_p_dy_20260317-163910.json`，
导出 `vision` 模型的前 3 个 bit flip，输出名会自动按 JSON 时间戳命名为
`vision_model_20260317-163910_top3.onnx`：

```bash
.venv/bin/python op_0103/export_flipped_onnx.py \
  --plan-json op_0103/out/important_bits_0103_taylor_seq100_b2_p_dy_20260317-163910.json \
  --target-model vision \
  --topn 3
```

默认输出到 `op_0103/out/`。也可用 `--target-model policy`，或 `--target-model both` 同时导出两个模型。
