# 训练报告：8MB PyPro 语料训练

**日期**: 2026-03-13
**语料来源**: `/home/xlisp/PyPro`（本地全量 Python 代码）
**检查点**: `out-codegpt/ckpt.pt`（iter=1000，best val loss=1.7383）

---

## 一、训练数据

### 原始语料

| 项目 | 数值 |
|------|------|
| 来源目录 | `/home/xlisp/PyPro` |
| 原始 `.py` 文件总数 | 687 个 |
| 原始总大小 | ~8.2 MB |
| 过滤后有效文件 | **615 个**（剔除 `__pycache__`、过小/过大文件）|

### Tokenize 结果

| 项目 | 数值 |
|------|------|
| 总 token 数 | **3,159,664** |
| 训练集 | 3,001,253 tokens（585 文档）|
| 验证集 | 158,411 tokens（30 文档）|
| train.bin | 6.0 MB |
| val.bin | 0.3 MB |

**与上次对比**：上次（11 文件，28K tokens）→ 本次（615 文件，3.16M tokens），**数据量增加 112 倍**。

---

## 二、训练过程

### 配置

```python
batch_size                = 4
block_size                = 512
gradient_accumulation_steps = 40
# 有效 batch size = 4 × 40 × 512 = 81,920 tokens/iter

learning_rate             = 6e-4
max_iters                 = 200000
warmup_iters              = 2000
dtype                     = 'float16'   # GTX 1080 必须
compile                   = False       # sm_61 限制
eval_interval             = 1000
```

### Loss 曲线（完整记录）

| Iter | Train Loss | 阶段 |
|------|-----------|------|
| 0 | 10.93 | 随机初始化（≈ ln 50304）|
| 50 | 5.70 | 快速学习语法结构 |
| 100 | 5.03 | |
| 150 | 3.13 | 开始收敛 |
| 200 | 3.08 | |
| 300 | 2.52 | |
| 330 | **2.16** | |
| 400 | 2.79 | 震荡下降（FIM 变换引起）|
| 500 | 1.59 | |
| 580 | 0.99 | 进入低 loss 区域 |
| 690 | 0.78 | |
| 760 | 0.65 | |
| 800 | 0.54 | |
| 870 | 0.43 | |
| 950 | **0.38** | 最低 train loss |
| **1000** | **0.53** | **→ 保存检查点，val loss=1.7383** |
| 1100 | 0.44 | |
| 1160 | 0.18 | |
| 1220 | **0.21** | 接近收敛 |
| 1330 | 0.35 | 训练终止 |

### 训练时间统计

| 项目 | 数值 |
|------|------|
| 总 iter 数 | 1330（含 iter 0）|
| 每 iter 耗时 | ~28 秒 |
| eval 耗时（200 batches）| ~134 秒（含在 iter 0 / iter 1000 时间内）|
| 总训练时长 | **约 10.5 小时** |
| GPU | GTX 1080 8GB，100% 利用率，60-66°C |
| 显存占用 | 5081 MiB / 8192 MiB（62%）|

### 过拟合分析

| 指标 | 数值 |
|------|------|
| iter 1000 train loss | 0.5301 |
| iter 1000 val loss | **1.7383** |
| 过拟合差距 | **1.21**（val - train）|
| train perplexity | exp(0.53) ≈ **1.70** |
| val perplexity | exp(1.74) ≈ **5.69** |

差距较大（1.21）说明模型在训练数据上过拟合，记忆成分较多。根本原因：3.1M tokens 训练 123M 参数模型，数据量/参数量比 ≈ 25，属于数据欠充足。

---

## 三、模型测试结果

### 3.1 代码补全（`sample.py` complete 模式）

**命令**：
```bash
~/miniconda3/envs/codegpt/bin/python -W ignore sample.py \
    --dtype=float16 --prompt="def abc(a):" --temperature=0.3 --top_k=50
```

#### `def abc(a):` 补全

| temperature | 输出 | 质量 |
|-------------|------|------|
| 0.3（保守）| `    return a + b` | ✅ **语法正确，有意义** |
| 0.8（默认）| `    return a` + 然后混入 pytest 上下文 | ⚠️ 开头对，后段混乱 |
| 1.2（激进）| `    return [` + 随机结构 | ❌ 不连贯 |

**结论**：低温度（0.3）时，`def abc(a):` 能补出有意义的函数体。

#### `class Stack:` 补全

```python
# 输入
class Stack:
    def __init__(self):

# 模型输出（temp=0.8）
        self.scope = mode
        self.spinner.reset()
```

⚠️ 语法结构正确（`self.xxx = yyy`），但内容是从训练数据记忆来的变量名。

#### `def fibonacci(n):` 补全

```python
# 模型输出（temp=0.8）
    """
    分...程序执行...
    return [
        '📈主键
        print("\n🔍 请见测请深空...")
```

❌ 混入训练数据中的中文注释片段，说明模型记忆了 PyPro 中的中文代码注释。

#### `import os` + `def list_files(path):` 补全

```python
# 模型输出
    """Download multiple files in the given path (notdir,) - read).

    This API is currently in write fails if the filesystem...

    Args:
        path: The write to the file.
```

⚠️ 生成了 docstring 格式（正确），但内容拼接混乱，是多个训练文档片段的混合。

---

### 3.2 FIM 填空测试

**命令**：
```bash
~/miniconda3/envs/codegpt/bin/python -W ignore sample.py \
    --dtype=float16 --mode=fim \
    --prefix="def add(a, b):\n    " \
    --suffix="    return result"
```

| 场景 | prefix | suffix | infill 输出 | 质量 |
|------|--------|--------|------------|------|
| add 函数体 | `def add(a, b):\n    ` | `\n    return result` | `class MyModel(BaseModel): def __init__...` | ❌ 主题偏离 |
| Stack 初始化 | `def __init__(self):\n    ` | `\n    self.size = 0` | `.transformer.wpe.weight_to(...)` | ❌ 无意义 |
| 列表操作中间 | `x = [1,2,3]\n` | `\nprint(x)` | `print(f"  {name}: {obj['name']}...")` | ⚠️ 格式类似但不对 |
| try 块 | `try:\n    ` | `\nexcept Exception as e:` | `< 128 {log_section}` | ❌ 无意义 |

**FIM 整体评价**：FIM 功能尚不可用。原因：FIM 需要更多数据训练模型学习三段上下文（prefix + suffix → middle）的映射，当前训练量不足。

---

### 3.3 Perplexity（困惑度）对比

> 越低越好，代表模型对该模式越熟悉。

| 输入类型 | 上次（28K tokens）| 本次（3.1M tokens）| 改善倍数 |
|---------|-----------------|------------------|--------|
| Python 类定义 | 257.9 | **5.0** | 52× |
| Python 函数定义 | 2121.3 | **6.1** | 348× |
| Python 循环 | 1140.9 | **16.3** | 70× |
| import 语句 | 5597.2 | **4.3** | 1302× |
| 随机英文 | 63536.8 | **188.1** | 338× |
| 随机中文 | — | **247.1** | — |

**关键改善**：
- import 语句 ppl 从 5597 降到 4.3，模型已高度熟悉导入模式
- Python 类/函数 ppl 降至 5-6，已接近实用水平
- 代码（4-16）vs 非代码（188-247）差距扩大，模型对"什么是 Python 代码"有清晰认知

---

### 3.4 Top-5 下一词预测

| Prompt | 最高概率 token | 概率 | 变化 |
|--------|-------------|------|------|
| `def abc(a):\n    ` | `' '`（空格）| 99.97% | 上次 81.82%，更确定 |
| `def fibonacci(n):\n    if ` | `' '`（空格）| 99.12% | 仍是空格 |
| `import ` | `' '`（空格）21% \| `' #'` 19.7% \| `'\n'` 13.7% | 分散 | 上次 67.9%，更不确定 |
| `return ` | `' '`（空格）| 78.9% | |

**观察**：高概率集中在空格 token，这是 Python 多级缩进的特征。模型尚未能在给定函数签名后直接预测第一个有意义的词（如 `result`、`n`、`None` 等）。

---

## 四、综合评价

### 能力矩阵

| 功能 | 状态 | 说明 |
|------|------|------|
| 不生成纯空格 | ✅ 改善 | 低温时能输出实际代码词 |
| 语法结构正确 | ✅ 部分 | `return a + b`、`self.xxx = yyy` 等结构正确 |
| 代码 vs 非代码区分 | ✅ 明显 | ppl 差距 30-50× |
| 函数体补全（简单）| ⚠️ 有时 | temp=0.3 时成功率较高 |
| 函数体补全（复杂）| ❌ 不可用 | 易混入记忆片段 |
| FIM 代码填空 | ❌ 不可用 | 训练量不足 |
| 中文混入 | ❌ 问题 | 训练数据含中文注释，污染输出 |

### 最大问题：过拟合与记忆

- **train loss 0.53 vs val loss 1.74**，差距大
- 生成结果中明显出现训练文件的片段（`self.spinner.reset()`、`log_section`、中文注释）
- 模型在记忆 PyPro 代码库，而非学习通用 Python 模式

### 推荐下一步

| 优先级 | 措施 | 预期效果 |
|--------|------|---------|
| ⭐⭐⭐ | 添加 HuggingFace codeparrot 数据（100M+ tokens）| 解决过拟合，提升泛化能力 |
| ⭐⭐ | 增加 dropout=0.1 | 减少记忆，提升泛化 |
| ⭐⭐ | 继续训练到 iter 5000+（当前仅 1330）| 充分收敛 |
| ⭐ | 用 GPT-2 预训练初始化（`--init_from=gpt2`）| 迁移通用语言能力 |

---

## 五、使用方法

```bash
# 代码补全（建议 temp=0.3，效果最好）
~/miniconda3/envs/codegpt/bin/python -W ignore sample.py \
    --dtype=float16 \
    --prompt="def abc(a):" \
    --temperature=0.3 \
    --top_k=50 \
    --max_new_tokens=150

# FIM 填空（当前效果差，供参考）
~/miniconda3/envs/codegpt/bin/python -W ignore sample.py \
    --dtype=float16 \
    --mode=fim \
    --prefix="def add(a, b):\n    " \
    --suffix="    return result" \
    --temperature=0.3

# 恢复继续训练
~/miniconda3/envs/codegpt/bin/python -W ignore train.py \
    config/train_codegpt.py \
    --init_from=resume
```
