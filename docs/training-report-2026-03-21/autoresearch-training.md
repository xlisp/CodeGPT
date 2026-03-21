# 用 autoresearch 训练 CodeGPT：为什么、怎么做、结果

**日期**：2026-03-21
**硬件**：NVIDIA GeForce GTX 1080 8GB (Pascal, sm_61)
**训练脚本**：`train_autoresearch.py`

---

## 为什么换到 autoresearch？

### 原问题：数据太少，架构太旧

原版 CodeGPT 有两个根本性瓶颈：

**1. 数据量严重不足**

| 数据源 | Token 数 | 问题 |
|--------|----------|------|
| 本地 `~/PyPro` Python 文件 | 28K tokens | 少到模型会迅速过拟合 |
| codeparrot/codeparrot-clean (HF) | ~100K samples 上限 | 需要手动下载 |
| climbmix-400b-shuffle (autoresearch) | **400B tokens** | 训不完也够用 |

28K tokens 是什么概念？一个 GPT-2 124M 的模型参数量是 124M，而训练数据只有 28K tokens。参数量是数据量的 **4,400 倍**，这必然导致严重过拟合。模型不是在"学习语言规律"，而是在死记硬背这几十个文件。

**2. 原架构已经过时**

原版 CodeGPT 基于 2022 年的 nanoGPT，使用：
- 标准 Multi-Head Attention（没有 RoPE，位置感知弱）
- LayerNorm（已被 RMSNorm 取代）
- AdamW 优化器（对矩阵参数不是最优）
- 学习位置编码（泛化性差）

autoresearch 集成了 2024-2025 年的最新成果：

| 技术 | 原 CodeGPT | autoresearch |
|------|-----------|--------------|
| 位置编码 | 可学习绝对位置嵌入 | **RoPE**（旋转位置编码） |
| 归一化 | LayerNorm | **RMSNorm**（更快更稳） |
| 注意力 | 标准 MHA | **滑动窗口注意力** (SSSL 模式) |
| 值嵌入 | 无 | **Value Residual**（ResFormer 技巧）|
| 优化器 | AdamW | **Muon + AdamW**（矩阵参数用 Muon）|
| LR 调度 | Cosine Decay | **WarmDown**（线性 warmdown，更好）|
| Logits | 直接 softmax | **Softcap**（tanh 截断，防溢出）|
| GC 管理 | Python 默认 | **gc.freeze() + gc.disable()**（防 500ms 停顿）|

---

## autoresearch 的核心技术详解

### 1. Muon 优化器（为什么比 AdamW 更好？）

AdamW 对所有参数一视同仁，用 element-wise 的二阶矩估计来调整学习率。但对于矩阵参数（Attention 的 Q/K/V/O 和 FFN 的权重），有更好的选择。

Muon（Momentum + Newton-Schulz）的思路：

```
标准 SGD:   g_t → 直接用梯度更新
带 Muon:    g_t → Nesterov 动量 → 极坐标正交化 → NorMuon 方差归一化 → 更新
```

**极坐标正交化**（Polar Express 算法）：把梯度矩阵投影到最近的正交矩阵。直觉上，这相当于归一化更新方向，使得每一步在参数空间中"走相同的距离"，不受梯度大小影响。用 5 步牛顿-Schulz 迭代近似：

```python
# 每步迭代：a*X + X @ (b*XᵀX + c*(XᵀX)²)
for a, b, c in polar_express_coeffs[:5]:
    A = X.mT @ X
    B = b * A + c * (A @ A)
    X = a * X + X @ B
```

**NorMuon**：在正交化后进一步做方差归一化，让所有矩阵参数的有效学习率相同，无需对不同形状矩阵调参。

结果：Muon 在相同计算量下收敛更快，loss 更低。

### 2. Value Residual（ResFormer 技巧）

标准 Transformer 的 Attention：
```
V = X @ Wv
out = softmax(QKᵀ/√d) @ V
```

Value Residual 的改进：
```
V = X @ Wv + gate * Ve    # Ve 是独立的 token embedding lookup
gate = 2 * sigmoid(X[:, :32] @ W_gate)  # 每个头独立门控
```

直觉：`Ve` 是 "token 的原始语义"，`V` 是 "经过 attention 变换后的语义"。残差连接让模型可以在两者之间平衡，减少深层网络中的表征退化。代价是额外的 value embedding 参数（约 16M），但收益显著。

### 3. 滑动窗口注意力（SSSL 模式）

```
S = short window (T/2 = 1024 tokens)
L = full window  (T   = 2048 tokens)

层分配（8层模型）：
Layer 0,4: S (1024)   Layer 1,5: S (1024)
Layer 2,6: S (1024)   Layer 3,7: L (2048)  ← 最后一层强制 full
```

**为什么有效**：大部分语言信息是局部的（词法、语法依赖通常在 512 token 内），只有少数层需要全局 context。滑动窗口减少 O(T²) 复杂度，在相同显存下可用更大 batch 或更长序列。

### 4. Logit Softcap

```python
logits = 15 * tanh(logits / 15)
```

将 logits 限制在 [-15, 15] 范围内。防止训练初期 logits 爆炸导致 softmax 数值不稳定，等价于给模型施加一个"不要太过自信"的先验。

---

## GTX 1080 适配：哪里改了，为什么

autoresearch 原版针对 H100 设计，GTX 1080 (Pascal, sm_61) 需要三处核心改动：

### 问题 1：Flash Attention 3 → PyTorch SDPA

| 方案 | 要求 | GTX 1080 |
|------|------|----------|
| Flash Attention 3 | Hopper (sm_90)，H100 专用 | ❌ 不支持 |
| Flash Attention 2 | Ampere+ (sm_80+) | ❌ 不支持 |
| PyTorch SDPA | 任意 GPU | ✅ 支持 |

实现等效的滑动窗口注意力：

```python
def sdpa_with_window(q, k, v, window_size):
    # q,k,v: (B, T, n_head, head_dim) — FA3 layout
    q, k, v = [x.transpose(1, 2) for x in [q, k, v]]  # → SDPA layout

    w = window_size[0]
    if w >= T:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        # 构造滑动窗口 causal mask
        diff = torch.arange(T).unsqueeze(0) - torch.arange(T).unsqueeze(1)
        mask = (diff >= 0) & (diff < w)          # position i attends to [i-w+1 .. i]
        attn_bias = torch.full((T,T), -inf).masked_fill(mask, 0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)

    return out.transpose(1, 2).contiguous()
```

### 问题 2：bfloat16 → float16

GTX 1080 没有原生 bfloat16 硬件支持，PyTorch 的 `is_bf16_supported()` 因 bug 返回 True，但实际走软件模拟，慢 10-30 倍。

必须用 float16，但 float16 动态范围小（最大值 65504），需要 GradScaler 防止梯度下溢：

```python
autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
scaler = torch.amp.GradScaler("cuda")

# 训练循环
with autocast_ctx:
    loss = model(x, y)
scaler.scale(loss / grad_accum_steps).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

**注意**：GradScaler 要求模型参数是 float32，只有前向计算用 float16。原版 autoresearch 将 embedding 参数转为 bfloat16，这里保持 float32。

### 问题 3：torch.compile → 关闭

`@torch.compile(dynamic=False, fullgraph=True)` 在 sm_61 上有兼容性问题。同时，原版使用 CPU scalar tensor 传参（专为 compile 优化），去掉 compile 后改为 Python float：

```python
# 原版（配合 torch.compile 用 CPU tensor 避免重编译）
self._adamw_lr_t = torch.tensor(0.0, device="cpu")
self._adamw_lr_t.fill_(group['lr'])
adamw_step_fused(..., self._adamw_lr_t, ...)

# 适配版（直接 Python float）
adamw_step_fused(..., group['lr'], ...)
```

### 问题 4：F.rms_norm → 手动实现

`F.rms_norm` 在 PyTorch 2.4 才加入，codegpt 环境是 2.3.1：

```python
def norm(x):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)
```

---

## 训练配置

```python
# train_autoresearch.py 关键超参数
DEPTH = 8                  # 8 层 Transformer
ASPECT_RATIO = 64          # n_embd = 8 × 64 = 512
HEAD_DIM = 128             # 每个 attention head 128 维
WINDOW_PATTERN = "SSSL"    # 3 层短窗口 + 1 层全窗口

TOTAL_BATCH_SIZE = 2**15   # 32768 tokens/step
DEVICE_BATCH_SIZE = 4      # 4 × 2048 = 8192 tokens/forward
# → grad_accum_steps = 32768 / 8192 = 4

MATRIX_LR = 0.04           # Muon 优化器学习率（矩阵参数）
EMBEDDING_LR = 0.6         # AdamW 学习率（embedding）
WEIGHT_DECAY = 0.2         # Cautious weight decay（随训练进度衰减到 0）
WARMDOWN_RATIO = 0.5       # 后 50% 时间线性衰减学习率到 0

TIME_BUDGET = 300          # 5 分钟训练预算
```

**模型规模**（DEPTH=8 时）：

| 组件 | 参数量 |
|------|--------|
| wte（token embedding） | 4.2M |
| value_embeds（6 层 × embedding）| 16.8M |
| lm_head（unembedding） | 4.2M |
| transformer 矩阵 | 25.2M |
| per-layer scalars | 16 |
| **总计** | **50.3M** |

---

## 训练日志（2026-03-21）

**数据准备**：

```
下载 shard_00000.parquet + shard_00001.parquet + shard_06542.parquet（验证集）
BPE tokenizer 训练（8192 词表）：19.1s
缓存位置：~/.cache/autoresearch/
```

**启动**：

```
Vocab size:  8,192
n_layer:     8
n_head:      4   (n_embd 512 / head_dim 128 = 4 heads)
n_embd:      512
sequence_len:2048
window:      SSSL
params:      50.3M
flops/token: 2.39e8
grad_accum:  4 steps
time_budget: 300s
```

**完整训练过程（27 步，300s）**：

| Step | Loss (EMA) | lrm | tok/sec | 说明 |
|------|-----------|-----|---------|------|
| 0 | 9.011 | 1.00 | 1,641 | 初始 ≈ ln(8192) = 9.012 ✓ |
| 1 | 8.908 | 1.00 | 1,686 | 快速下降 |
| 2 | 8.638 | 1.00 | 1,680 | |
| 3 | 8.220 | 1.00 | 1,674 | |
| 5 | 7.768 | 1.00 | 1,653 | |
| 7 | 7.361 | 1.00 | 1,631 | |
| 10 | 6.956 | 1.00 | 1,644 | 开始计时（warmup 10 步）|
| 14 | 6.552 | 1.00 | 1,675 | |
| 18 | 6.249 | 1.00 | 1,670 | |
| 19 | 6.179 | 0.95 | 1,674 | WarmDown 开始（进度 52.7%）|
| 20 | 6.107 | 0.82 | 1,680 | LR 快速衰减 |
| 21 | 6.014 | 0.69 | 1,675 | |
| 22 | 5.934 | 0.55 | 1,676 | |
| 23 | 5.853 | 0.42 | 1,679 | |
| 24 | 5.759 | 0.29 | 1,677 | |
| 25 | 5.658 | 0.16 | 1,680 | |
| **26** | **5.559** | 0.03 | 1,679 | **最终步，time's up** |

**最终评估结果**：

```
val_bpb:          1.724407     ← Bits Per Byte（越低越好）
training_seconds: 314.4
total_seconds:    4108.0       ← 含启动/编译时间（~68分钟总耗时含数据准备）
peak_vram_mb:     4241.6       ← 约 4.1 GB 显存
mfu_percent:      4.77
total_tokens_M:   0.9          ← 27步 × 32768 tokens = 884K tokens
num_steps:        27
num_params_M:     50.3
depth:            8
```

**性能指标**：

| 指标 | 值 | 说明 |
|------|-----|------|
| val_bpb | **1.724** | Bits Per Byte，vocab 无关指标 |
| tok/sec | ~1,660 | GTX 1080，无 FlashAttention，无 torch.compile |
| ms/step | ~19,600 | DEVICE_BS=4, T=2048, grad_accum=4 |
| 峰值显存 | 4,242 MB | 约 4.1 GB，8 GB 中用了 52% |
| MFU | 4.77% | 基于 GTX 1080 FP16 峰值 8.87 TFLOPS |
| Loss 下降 | 9.011 → 5.559 | 5 分钟内下降 38.3% |

**val_bpb 说明**：

Bits Per Byte 是与词表大小无关的评估指标（autoresearch 的标准指标）：

```
val_bpb = Σ(cross_entropy_loss) / (log(2) × Σ(target_bytes))
```

- `1.724 bpb` = 平均每个字节需要 1.724 bit 来编码
- 理论最优（完美预测）= 1.0 bpb
- 随机猜测（8192 词表）≈ ln(8192)/log(2) ≈ 9.0 bpb（对应 loss=9.0）
- 5 分钟从 9.0 → 1.72，说明模型已学到大量语言规律

**与原 CodeGPT 训练的对比**：

| 指标 | 原 CodeGPT | autoresearch 版 |
|------|-----------|----------------|
| 训练数据 | 28K tokens（本地 Python）| 400B tokens（climbmix）|
| 词表大小 | 50,257（GPT-2 BPE）| 8,192（自训 BPE）|
| 每步 tokens | 81,920 | 32,768 |
| tok/sec | ~2,926 | ~1,640 |
| ms/step | ~28,000 | ~19,500 |
| 优化器 | AdamW | Muon + AdamW |
| 位置编码 | 可学习绝对位置 | RoPE |
| 归一化 | LayerNorm | RMSNorm |
| 注意力窗口 | 全局 | 滑动窗口 SSSL |

速度略低是因为 value_embeds 的额外开销 + SDPA 无法像 FlashAttention 那样融合。但训练质量（loss 收敛速度、最终效果）应当更好。

---

## 如何运行

### 前置步骤（一次性）

```bash
# 安装依赖
~/miniconda3/envs/codegpt/bin/pip install pyarrow requests rustbpe tiktoken

# 下载数据 + 训练 tokenizer（约 5-10 分钟）
cd /home/xlisp/PyPro/autoresearch
~/miniconda3/envs/codegpt/bin/python prepare.py --num-shards 2

# 或下载更多数据（10 shards，更好的训练效果）
~/miniconda3/envs/codegpt/bin/python prepare.py --num-shards 10
```

### 训练

```bash
cd /home/xlisp/PyPro/CodeGPT
~/miniconda3/envs/codegpt/bin/python -W ignore train_autoresearch.py
```

训练 5 分钟后自动停止，输出：

```
---
val_bpb:          x.xxxxxx      # bits per byte（越低越好）
training_seconds: 300.x
total_seconds:    xxx.x
peak_vram_mb:     xxxx.x
mfu_percent:      x.xx
total_tokens_M:   xx.x
num_steps:        xx
num_params_M:     50.3
depth:            8
```

### 调整模型大小

通过修改 `train_autoresearch.py` 顶部的超参数：

```python
# 更小的模型（显存不足时）
DEPTH = 6              # 6 层，n_embd = 384
DEVICE_BATCH_SIZE = 8

# 更大的模型（追求更好效果）
DEPTH = 12             # 12 层，n_embd = 768（接近 GPT-2 124M）
DEVICE_BATCH_SIZE = 2  # 需要减小 batch

# 更多训练数据
# cd autoresearch && python prepare.py --num-shards 50
```

---

## 为什么选择 autoresearch

1. **数据问题是根本**：从 28K → 400B tokens，这是 1400 万倍的提升。任何架构改进都不如数据量带来的收益大。

2. **现代架构有实证支撑**：RoPE、RMSNorm、Muon 都有论文和大规模实验证明比旧版本更好，autoresearch 已经把这些打包好了。

3. **5 分钟实验循环**：autoresearch 的设计哲学是"固定 5 分钟时间预算，每次只改一个变量"。这让在 GTX 1080 上快速验证想法成为可能——不需要等 64 天。

4. **BPB 指标更科学**：原版用 val_loss（依赖词表大小），autoresearch 用 Bits Per Byte（vocab-size-independent），不同配置的结果可以直接比较。

---

## 参考文献

- [nanoGPT](https://github.com/karpathy/nanoGPT) — CodeGPT 原始基础
- [autoresearch](https://github.com/karpathy/autoresearch) — 本次训练基础框架
- [Muon optimizer](https://github.com/KellerJordan/Muon) — 矩阵参数的 Newton-Schulz 正交化优化器
- [ResFormer / Value Residual](https://arxiv.org/abs/2410.17897) — 值残差连接
- [RoPE](https://arxiv.org/abs/2104.09864) — 旋转位置编码
- [Flash Attention](https://arxiv.org/abs/2205.14135) — IO-aware 注意力（GTX 1080 不支持 FA2/3，本项目用 PyTorch SDPA）
