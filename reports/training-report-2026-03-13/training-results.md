# 训练结果

## 验证性训练（50 iterations）

本次为验证训练流程可行性的短期运行，参数：

```
eval_iters    = 5
max_iters     = 50
eval_interval = 50
log_interval  = 5
```

### Loss 曲线

| Iteration | Train Loss | Val Loss | 备注 |
|-----------|-----------|----------|------|
| 0 | 10.9359 | 10.9277 | 初始（随机权重，接近 ln(50304) ≈ 10.83）|
| 5 | 9.8182 | — | |
| 10 | 8.5885 | — | |
| 15 | 6.7352 | — | |
| 20 | 6.2798 | — | |
| 25 | 6.8333 | — | |
| 30 | 6.7521 | — | |
| 35 | 7.6974 | — | |
| 40 | 6.2691 | — | |
| 45 | 5.7495 | — | |
| **50** | **6.2724** | **5.3586** | 保存检查点 |

初始 loss ≈ ln(vocab_size) = ln(50304) ≈ 10.83，与实测 10.93 吻合，说明模型初始化正确（均匀分布预测）。50 步后 val loss 降至 5.36，模型开始学习代码模式。

### 训练速度

| 指标 | 值 |
|------|-----|
| 每 iter 耗时 | ~28 秒 |
| 每 forward pass 耗时 | ~335 ms |
| GPU 显存占用 | 5062 MiB / 8192 MiB |
| GPU 利用率 | 100% |
| MFU（模型算力利用率）| 0.75%（以 A100 为基准）|
| 每 iter tokens | 81,920 |
| 实际吞吐量 | ~2,926 tokens/sec |

> MFU 0.75% 基于 A100 312 TFLOPS 计算，换算为 GTX 1080 实际性能约为：
> GTX 1080 FP16 ≈ 8.87 TFLOPS，实际利用率约 26%。

### 检查点

```
路径: out-codegpt/ckpt.pt
大小: 1.4 GB
内容:
  - model state dict（float16 权重）
  - optimizer state dict（AdamW）
  - iter_num: 50
  - best_val_loss: 5.3586
  - model_args: {n_layer:12, n_head:12, n_embd:768, block_size:512, ...}
  - config: CodeGPTConfig(...)
```

## 性能分析

### GTX 1080 的限制

- **无 Tensor Core**（Pascal 架构）：FP16 计算走 CUDA core，吞吐约 8.87 TFLOPS
- **无原生 bfloat16**：必须用 float16（bfloat16 为软件模拟，慢 10-30 倍）
- **无 FlashAttention2**：需要 sm_75+（Turing），GTX 1080 使用 PyTorch 内置的 SDPA，通过 memory-efficient attention 实现
- **显存 8GB**：限制 batch_size 和 block_size，有效 batch size 较小

### 若使用更好的 GPU 预期性能

| GPU | 架构 | FP16 TFLOPS | 预估 tokens/sec | 相对提速 |
|-----|------|-------------|-----------------|---------|
| GTX 1080 | Pascal sm_61 | ~8.87 | ~2,926 | 1× |
| RTX 3090 | Ampere sm_86 | ~35.6 | ~11,726 | 4× |
| A100 40GB | Ampere sm_80 | ~312 | ~102,800 | 35× |
| H100 80GB | Hopper sm_90 | ~989 | ~326,000 | 111× |

## 完整训练估算

按当前配置（max_iters=200,000，每 iter 28 秒）：

- 总训练时间 ≈ 200,000 × 28s = **5,600,000 秒 ≈ 64.8 天**

实际不可行，正式训练建议：
1. 使用 `train_codegpt_small.py`（10M 参数小模型，约 3–5 天）
2. 或更换更强 GPU（RTX 3090 约 16 天）
3. 或减少 max_iters（如 10,000 iter ≈ 3.2 天，可得初步收敛结果）
