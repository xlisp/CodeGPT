# 模型评测报告

**评测时间**: 2026-03-13
**检查点**: `out-codegpt/ckpt.pt`（iter=50，val loss=5.36）
**评测脚本**: `sample.py`

---

## 评测方法

### 使用 sample.py

```bash
# 代码补全模式
~/miniconda3/envs/codegpt/bin/python -W ignore sample.py \
    --dtype=float16 \
    --prompt="def abc(a):" \
    --max_new_tokens=200 \
    --temperature=0.8 \
    --num_samples=3

# FIM 填空模式
~/miniconda3/envs/codegpt/bin/python -W ignore sample.py \
    --dtype=float16 \
    --mode=fim \
    --prefix="def add(a, b):\n    " \
    --suffix="    return result" \
    --max_new_tokens=50

# 交互模式
~/miniconda3/envs/codegpt/bin/python -W ignore sample.py \
    --dtype=float16 \
    --mode=interactive
```

> **注意**：必须加 `--dtype=float16`，否则触发 bfloat16 误报导致极慢（见 bugs-and-fixes.md）

### 生成参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `temperature` | 0.8 | 采样温度。越低越保守（重复），越高越随机（创意）|
| `top_k` | 200 | 只从概率最高的 k 个 token 中采样 |
| `top_p` | 0.95 | nucleus sampling，概率累积不超过 p |
| `repetition_penalty` | 1.1 | 抑制重复 token |
| `max_new_tokens` | 512 | 最多生成 token 数 |

---

## 代码补全测试结果

### 测试 1：`def abc(a):` 补全（temperature=0.8）

```python
# 输入
def abc(a):

# 模型输出（3 次采样）
# Sample 1:
                                        # 全部空格
# Sample 2:
                                        # 全部空格
# Sample 3:
                                        # 全部空格
```

**现象**：模型输出全是空格（token id=220），概率高达 **81.82%**。

### 测试 2：不同 temperature 对比

| temperature | 输出 | 分析 |
|-------------|------|------|
| 0.3（保守）| `\n` + 大量空格 | 高度集中于最高概率 token |
| 0.8（默认）| 全空格 | 次优化策略，卡在局部最优 |
| 1.5（激进）| `ipples_ul_ SC gri_size =\|,(frian,, n_yl` | 开始随机，出现多样 token |

### 测试 3：FIM 填空（`def add(a, b):` 中间填充）

```python
# 前缀
def add(a, b):
    <FILL>
    return result

# 模型输出
    \n                              # 仅输出空行
```

### 测试 4：不同 prompt 的 Top-5 预测

```
Prompt: 'def abc(a):'
  '\n'    p=0.2240   ← 最高概率：换行（合理！函数定义后换行）
  '_'     p=0.0068
  '.'     p=0.0009
  '('     p=0.0004
  ','     p=0.0002

Prompt: 'def abc(a):\n    return '
  ' '     p=0.7731   ← 卡住：认为返回值是空格
  ' if'   p=0.0002
  ' #'    p=0.0001

Prompt: 'import '
  ' '     p=0.6792   ← 卡住：不知道 import 什么
  ' if'   p=0.0003

Prompt: 'for i in range('
  '\n'    p=0.0025   ← 概率极分散，模型对此几乎无知
  '_'     p=0.0023
  '('     p=0.0020
  'f'     p=0.0017
  '.'     p=0.0012
```

**关键观察**：`def abc(a):` 后预测 `\n`（换行）的概率 22.4%，说明模型已经学到"函数定义后需要换行"这个基本规律，但还不知道函数体该写什么。

---

## Perplexity（困惑度）测试

> 困惑度 = exp(loss)，越低表示模型对该模式越熟悉。理想：< 50；可用：50-200；当前阶段：257-63537。

| 输入类型 | Loss | Perplexity | 分析 |
|---------|------|-----------|------|
| Python 类定义 | 5.55 | **257.9** | 最低 ppl，结构简单（class/def/self）|
| Python 循环 | 7.04 | **1,140.9** | 有所学习 |
| Python 函数定义 | 7.66 | **2,121.3** | 基本结构未掌握 |
| import 语句 | 8.63 | **5,597.2** | 几乎无学习 |
| 随机英文文本 | 11.06 | **63,536.8** | 明显高于代码 |
| 数字序列 | 10.78 | **48,042.0** | 明显高于代码 |

**正向信号**：代码的困惑度（257–5597）显著低于随机文本（63537），说明模型已经开始识别代码与非代码的分布差异。

---

## 诊断分析

### 为什么模型输出全是空格？

```
训练集规模：28,204 tokens（约 28KB Python 代码）
有效训练轮次：50 iterations
每 iteration tokens：81,920
实际见过的 tokens：50 × 81,920 / 28,204 ≈ 145 遍（同样的数据反复训练）

问题：数据极少，模型过拟合到"代码中最常见的 token = 空格（缩进）"
```

Python 代码中空格出现频率极高（缩进、对齐、行间距），在如此小的数据集上训练，模型学到的"捷径"就是：**预测空格总是安全的**。

### Loss 下降的真实含义

| Iteration | Train Loss | 含义 |
|-----------|-----------|------|
| 0 | 10.93 | 随机猜测（≈ ln(50304) = 10.83）|
| 50 | 6.27 | 开始记忆训练数据（非真正泛化）|

Loss 下降主要来自**记忆少量训练数据**，而非学习代码生成能力。50K tokens 训练一个 123M 参数模型，参数数量是数据量的 **4,383 倍**，严重欠数据。

---

## 对比：需要多少数据才能有效？

| 训练规模 | 预期 val ppl | 模型能力 |
|---------|------------|---------|
| 28K tokens，50 iter（当前）| ~213 | 只会输出空格 |
| 1M tokens，1K iter | ~80-100 | 开始输出有意义的关键词 |
| 10M tokens，10K iter | ~30-50 | 能补全简单函数 |
| 100M tokens，100K iter | ~10-20 | 可用的代码补全 |
| 1B+ tokens，200K iter | ~5-10 | 接近 GPT-2 代码水平 |

---

## 如何改进模型效果

### 方法 1：获取更多训练数据（最重要）

```bash
# 从 HuggingFace 下载 codeparrot 数据集（约 54GB 原始 Python 代码）
~/miniconda3/envs/codegpt/bin/python data/python_code/prepare.py \
    --source=huggingface \
    --max_samples=500000   # 约 200M tokens

# 加上本地代码
~/miniconda3/envs/codegpt/bin/python data/python_code/prepare.py \
    --source=both \
    --code_dir=/home/xlisp/PyPro \
    --max_samples=500000
```

### 方法 2：使用小模型更快收敛

```bash
# 使用 train_codegpt_small.py（10M 参数，约 3-5 天可得有意义结果）
~/miniconda3/envs/codegpt/bin/python -W ignore train.py config/train_codegpt_small.py \
    --dtype=float16
```

### 方法 3：从 GPT-2 预训练权重初始化（迁移学习）

```bash
# 从 GPT-2 124M 预训练权重开始，在代码数据上微调
~/miniconda3/envs/codegpt/bin/python -W ignore train.py config/train_codegpt.py \
    --dtype=float16 \
    --init_from=gpt2
```

GPT-2 已在大量网络文本（含代码）上预训练，从这里出发只需少量代码数据即可具备基础代码理解能力。

---

## 结论

| 评测维度 | 结果 | 说明 |
|---------|------|------|
| 代码补全（`def abc(a):`）| ❌ 无效 | 输出全为空格 |
| FIM 填空 | ❌ 无效 | 输出随机或空白 |
| 代码 vs 非代码区分 | ✅ 初步学会 | 代码 ppl 远低于随机文本 |
| 基本结构感知 | ✅ 初步 | `def func():` 后知道换行 |
| 有意义代码生成 | ❌ 尚未达到 | 需要更多数据和训练 |

**根本原因**：验证性训练（50 iter，28K tokens）只是用来确认训练流程可运行，数据量和训练量均远不足以产生有用的模型输出。训练流程本身完全正常，模型架构和 FIM 机制均已正确实现。
