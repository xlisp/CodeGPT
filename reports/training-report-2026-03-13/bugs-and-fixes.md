# 遭遇的问题与解决方案

本次调试共遇到 4 个问题，按发现顺序记录。

---

## Bug 1：Python 版本与 PyTorch 根本不兼容

### 现象

```
UserWarning: NVIDIA GeForce GTX 1080 with CUDA capability sm_61 is not compatible
with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_70 sm_75 sm_80 ...

CUDA error: no kernel image is available for execution on the device
```

### 根因

| | Python 3.13 (系统) | Python 3.12 (conda) |
|--|--|--|
| 可用 PyTorch | ≥ 2.5 | ≥ 2.0 |
| GTX 1080 支持 | ✗（PyTorch 2.4 起放弃 sm_61）| ✓（PyTorch 2.3.1 支持 sm_61）|

GTX 1080 是 Pascal 架构（Compute Capability 6.1），PyTorch 2.4 起不再为其编译 CUDA kernel，也不包含可 JIT 回退的 PTX 代码。系统只有 Python 3.13，其可用的最旧 PyTorch wheel 是 2.5，两者产生根本冲突。

### 解决方案

安装 Miniconda，创建 Python 3.12 虚拟环境，安装 PyTorch 2.3.1+cu118（CUDA 11.8 编译，支持 sm_61）。

```bash
bash ~/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda create -n codegpt python=3.12 -y
~/miniconda3/envs/codegpt/bin/pip install "torch==2.3.1+cu118" \
    --find-links https://download.pytorch.org/whl/cu118/torch_stable.html
```

---

## Bug 2：`KeyError: 'lm_head.weight'`（model.py）

### 现象

```
KeyError: 'lm_head.weight'
  File "model.py", line 231, in configure_optimizers
    {"params": [param_dict[pn] for pn in sorted(list(decay))], ...}
```

### 根因

`model.py` 第 153 行做了权重绑定：

```python
self.transformer.wte.weight = self.lm_head.weight
```

`self.named_parameters()` 会对绑定的权重去重，只返回一个参数名（`transformer.wte.weight`），不会出现 `lm_head.weight`。

但 `configure_optimizers` 遍历 `named_modules()` 时，遇到 `lm_head` 模块，把 `lm_head.weight` 加入了 `decay` 集合，之后用它去查 `param_dict` 时就 `KeyError`。

### 解决方案

在 `model.py:configure_optimizers` 中，用 `param_dict.keys()` 对 `decay` / `no_decay` 集合取交集，过滤掉因权重绑定而不在 `param_dict` 中的参数名：

```python
# 修改前（有 bug）
param_dict = {pn: p for pn, p in self.named_parameters()}
inter_params = decay & no_decay
union_params = decay | no_decay

# 修改后（正确）
param_dict = {pn: p for pn, p in self.named_parameters()}
# 过滤掉因权重绑定而不在 param_dict 中的参数（如 lm_head.weight）
decay = decay & param_dict.keys()
no_decay = no_decay & param_dict.keys()
inter_params = decay & no_decay
union_params = decay | no_decay
```

**修改文件**：`model.py` 第 224-228 行

---

## Bug 3：FIM 变换导致 batch size 不匹配（train.py）

### 现象

```
ValueError: Expected input batch_size (2044) to match target batch_size (2048).
```

数字关系：`4 × 511 = 2044`，`4 × 512 = 2048`，x 比 y 少 4 个 token。

### 根因

`get_batch` 中 FIM 变换部分的逻辑错误：

```python
# 有 bug 的代码
tokens_transformed  # 已 pad/truncate 至 block_size (512) 个 token
x_fim.append(torch.tensor(tokens_transformed[:-1], dtype=torch.long))  # 511 tokens

target = tokens_transformed[1:] + [SPECIAL_TOKENS["<|fim_pad|>"]]  # 511 + 1 = 512
target = target[:block_size]  # 512 tokens  ← 比 x 多 1 个！
```

语言模型的标准做法：`x = seq[:-1]`，`y = seq[1:]`，两者等长。这里 target 多加了一个 pad token 导致长度不一致。

### 解决方案

去掉多余的 pad token，直接用 `tokens_transformed[1:]`：

```python
# 修改前（有 bug）
target = tokens_transformed[1:] + [SPECIAL_TOKENS["<|fim_pad|>"]]
target = target[:block_size]
target_tensor = torch.tensor(target, dtype=torch.long)

# 修改后（正确）
target = tokens_transformed[1:]   # 与 x 等长（block_size - 1）
target_tensor = torch.tensor(target, dtype=torch.long)
```

**修改文件**：`train.py` 第 155-156 行

---

## Bug 4：bfloat16 误报导致训练极慢

### 现象

训练启动后 GPU 使用率 100%，但每次 eval（200 batches）耗时超过 20 分钟，完全不可接受。

### 根因

`train.py` 中的 dtype 选择逻辑：

```python
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
```

在 PyTorch 2.3.1 中，`torch.cuda.is_bf16_supported()` 对 GTX 1080 返回 `True`，但这是**错误的**。GTX 1080（Pascal sm_61）没有原生 bfloat16 硬件支持，CUDA 只能用软件模拟，比 float16 慢约 10-30 倍。

### 验证

```python
>>> import torch
>>> torch.cuda.is_bf16_supported()
True   # ← PyTorch 2.3 bug，GTX 1080 实际不支持 native bfloat16
```

### 解决方案

在 `config/train_codegpt.py` 中显式指定 `dtype = 'float16'`，覆盖 `train.py` 的自动检测：

```python
# config/train_codegpt.py
dtype = 'float16'  # GTX 1080 has no native bfloat16, force float16
```

修复后每个 iter 耗时从 **~400 秒** 降至 **~28 秒**。

---

## 问题汇总

| # | 位置 | 问题类型 | 影响 |
|---|------|---------|------|
| 1 | 环境 | Python/PyTorch 版本不兼容 | 训练无法启动 |
| 2 | `model.py:224` | 权重绑定与参数集合不一致 | 训练无法启动（KeyError）|
| 3 | `train.py:155` | FIM target 长度比 x 多 1 | 训练无法启动（ValueError）|
| 4 | `config/train_codegpt.py` | bfloat16 误报导致软件模拟 | 训练速度降低 10-30 倍 |
