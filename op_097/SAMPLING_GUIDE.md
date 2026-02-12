# 权重采样策略指南

## 三种采样模式

### 模式 1: 采样模式（推荐）⭐
```bash
--per-tensor-k 2
# 不加 --sample-all-weights
```

**工作原理**:
- 每个张量先按绝对值选择 `per_tensor_k × 10` 个候选
- 计算这些候选的梯度
- 选择梯度最大的 `per_tensor_k` 个

**示例**:
```
张量大小: 73,728 个权重
per_tensor_k = 2

步骤:
1. 按|weight|选择 20 个候选 (2 × 10)
2. 计算这 20 个的梯度 → 21 次前向传播
3. 选择梯度最大的 2 个
```

**计算量**:
```
313个张量 × 20个权重 × 21次前向 = 131,460 次前向传播
预计时间: ~30分钟 (num_val_batches=1)
```

---

### 模式 2: 全量模式（新增）🔥
```bash
--sample-all-weights
# per_tensor_k 会被忽略
```

**工作原理**:
- 计算张量中**每一个权重**的梯度
- 按梯度排序，选择 top-k

**示例**:
```
张量大小: 73,728 个权重

步骤:
1. 对所有 73,728 个权重计算梯度
2. 每个权重需要 1 次前向传播
3. 选择梯度最大的 top-k 个
```

**计算量（以 vision 模块为例）**:
```python
# 统计 vision 模块的总参数量
张量1: stem.0.weight          →     1,728
张量2: stem.1.weight          →     9,216
张量3: stages.0.blocks.0...   →    18,432
...
张量61: final_conv.weight     →    65,536
-------------------------------------------
总计: ~2,847,392 个权重参数

计算量:
2,847,392 个权重 × 1 次前向 = 2,847,392 次前向传播
预计时间: ~400 小时！(num_val_batches=1, 单CPU)
```

**⚠️ 严重警告**:
- 全量模式计算量**极其庞大**
- 单个张量可能有几万到几十万个参数
- **不推荐在完整模型上使用**

---

### 模式 3: 超大采样
```bash
--per-tensor-k 100
# 不加 --sample-all-weights
```

**工作原理**:
- 类似模式1，但采样更多候选
- 采样 `100 × 10 = 1000` 个候选
- 选择梯度最大的 100 个

**计算量**:
```
313个张量 × 1000个权重 × 1001次前向 = 313,313,000 次前向传播
预计时间: ~40 小时 (num_val_batches=1)
```

---

## 实际使用建议

### 场景 1: 快速探索（10分钟）
```bash
python op_097/important_bits_onnx.py \
    --weight-selection-method gradient \
    --top-w 50 \
    --per-tensor-k 1 \
    --num-val-batches 1 \
    --eval-seq-len 5 \
    --restrict "vision" \
    --stage select-weights
```

### 场景 2: 标准分析（30-60分钟）
```bash
python op_097/important_bits_onnx.py \
    --weight-selection-method gradient \
    --top-w 500 \
    --per-tensor-k 2 \
    --num-val-batches 1 \
    --eval-seq-len 20 \
    --stage select-weights  # 所有模块
```

### 场景 3: 精细搜索（2-4小时）
```bash
python op_097/important_bits_onnx.py \
    --weight-selection-method gradient \
    --top-w 1000 \
    --per-tensor-k 5 \
    --num-val-batches 10 \
    --eval-seq-len 20 \
    --stage select-weights
```

### 场景 4: 单张量全量分析（可行）✅
```bash
# 只分析一个特定的小张量
python op_097/important_bits_onnx.py \
    --weight-selection-method gradient \
    --sample-all-weights \
    --top-w 100 \
    --num-val-batches 1 \
    --eval-seq-len 5 \
    --restrict "stem.0"  # 限制到一个小张量
    --stage select-weights
```

---

## 性能优化建议

### 1. 使用限制（--restrict）
```bash
# 只分析特定模块
--restrict "vision"           # ~110 个张量
--restrict "policy"           # ~50 个张量
--restrict "stem"             # ~6 个张量
```

### 2. 减少验证批次
```bash
--num-val-batches 1           # 最快
--num-val-batches 10          # 平衡
--num-val-batches 128         # 最准确但极慢
```

### 3. 使用 GPU
```bash
--provider cuda               # 快 3-5 倍
```

### 4. 并行化（高级）
如果有多个 GPU，可以分块并行处理：
```bash
# GPU 0: vision 模块
CUDA_VISIBLE_DEVICES=0 python ... --restrict "vision"

# GPU 1: policy 模块
CUDA_VISIBLE_DEVICES=1 python ... --restrict "policy"
```

---

## 计算量对照表

| 模式 | 张量数 | 每张量权重数 | 总前向传播次数 | 预计时间 (CPU) |
|------|--------|-------------|--------------|---------------|
| 采样 k=1 | 313 | 10 | 34,430 | ~10分钟 |
| 采样 k=2 | 313 | 20 | 131,460 | ~30分钟 |
| 采样 k=5 | 313 | 50 | 803,300 | ~2小时 |
| 采样 k=10 | 313 | 100 | 3,161,300 | ~7小时 |
| **全量** | **313** | **~9,100平均** | **~2,848,000** | **~400小时** |

---

## 结论

**推荐使用采样模式** (`--per-tensor-k 2-5`)：
- ✅ 计算量可控
- ✅ 通过预采样已经过滤了小权重
- ✅ 梯度大的权重通常对应绝对值也较大
- ✅ 性价比最高

**仅在以下情况使用全量模式**:
- 单个小张量的详细分析
- 有充足的计算资源和时间
- 需要绝对精确的梯度排序
