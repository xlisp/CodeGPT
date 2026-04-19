# 模型配置

## 模型架构：CodeGPT (GPT-2 124M 规模)

CodeGPT 是基于 nanoGPT 扩展的代码生成语言模型，核心是标准 GPT-2 Decoder-only Transformer，加入了代码专用特性。

### 架构参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 参数量 | **123.59M** | 约 1.24 亿参数 |
| `n_layer` | 12 | Transformer 层数 |
| `n_head` | 12 | 注意力头数 |
| `n_embd` | 768 | 隐藏维度 |
| `block_size` | 512 | 最大序列长度（单卡适配，原始 1024）|
| `vocab_size` | 50304 | GPT-2 词表 + 代码特殊 token，填充至 64 的倍数 |
| `bias` | False | 无偏置，遵循 GPT-2 best practice |
| `dropout` | 0.0 | 训练时不 dropout（数据量小时可考虑调大）|

### 特殊 Token（代码扩展词表）

在 GPT-2 原始 50257 个 token 基础上新增：

| Token | ID | 用途 |
|-------|----|------|
| `<\|endoftext\|>` | 50256 | 文档结束（GPT-2 原有）|
| `<\|fim_prefix\|>` | 50257 | FIM 前缀标记 |
| `<\|fim_middle\|>` | 50258 | FIM 中间标记 |
| `<\|fim_suffix\|>` | 50259 | FIM 后缀标记 |
| `<\|fim_pad\|>` | 50260 | FIM 填充（loss 中 mask 掉）|
| `<\|code_start\|>` | 50261 | 代码块开始 |
| `<\|code_end\|>` | 50262 | 代码块结束 |
| `<\|lang:python\|>` | 50263 | 语言标识符（Python）|
| `<\|lang:javascript\|>` | 50264 | 语言标识符（JS）|
| ... | 50265–50278 | 其他语言标识符 |

实际词表填充至 **50304**（50278 向上取 64 的倍数）。

### Fill-in-the-Middle (FIM) 训练

训练时以 50% 概率对代码序列做 FIM 变换，教模型做代码补全（infill），而非仅能续写。

有两种变换格式：
- **PSM**（Prefix-Suffix-Middle）：`<fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle`
- **SPM**（Suffix-Prefix-Middle）：`<fim_suffix> suffix <fim_prefix> prefix <fim_middle> middle`

```
fim_enabled  = True
fim_rate     = 0.5   # 50% 的 batch 做 FIM 变换
fim_spm_rate = 0.5   # FIM 中 50% 用 SPM 格式
```

### 权重绑定（Weight Tying）

`lm_head.weight` 与 `transformer.wte.weight`（词嵌入）共享同一张权重矩阵，减少参数并提升性能。这是 GPT 系列的标准做法。

## 优化器配置

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW（fused 版本）|
| `learning_rate` | 6e-4 |
| `weight_decay` | 0.1 |
| `beta1` | 0.9 |
| `beta2` | 0.95 |
| `grad_clip` | 1.0 |
| LR 调度 | Cosine decay + warmup |
| `warmup_iters` | 2000 |
| `lr_decay_iters` | 200000 |
| `min_lr` | 6e-5 |

## 单卡适配调整（相对于原始多卡配置）

原始配置面向 4-8 张 A100 40GB 设计，做了以下调整适配 GTX 1080 8GB：

| 参数 | 原始值 | 单卡值 | 原因 |
|------|--------|--------|------|
| `batch_size` | 12 | **4** | 显存限制 |
| `block_size` | 1024 | **512** | 显存限制 |
| `compile` | True | **False** | sm_61 对 torch.compile 支持有限 |
| `dtype` | bfloat16 | **float16** | GTX 1080 无原生 bfloat16 |
| `gradient_accumulation_steps` | 40 | 40（不变）| 保持有效 batch size |

有效 batch size = `batch_size × gradient_accumulation_steps × block_size` = 4 × 40 × 512 = **81,920 tokens/iter**
