# CodeGPT REPL 使用手册

**文件**: `repl.py`
**功能**: 加载训练好的模型，在命令行进行代码补全与连续对话

---

## 启动

```bash
# 标准启动（推荐）
~/miniconda3/envs/codegpt/bin/python -W ignore repl.py

# 自定义参数
~/miniconda3/envs/codegpt/bin/python -W ignore repl.py \
    --temperature=0.5 \
    --max_tokens=300 \
    --top_k=100

# 指定检查点目录
~/miniconda3/envs/codegpt/bin/python -W ignore repl.py \
    --out_dir=out-codegpt
```

### 启动参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--out_dir` | `out-codegpt` | 检查点目录（含 `ckpt.pt`）|
| `--temperature` | `0.3` | 采样温度。越低越保守，越高越发散 |
| `--top_k` | `50` | 每步只从概率最高的 k 个 token 中采样 |
| `--top_p` | `0.95` | nucleus sampling 阈值 |
| `--max_tokens` | `200` | 单次最多生成的 token 数 |
| `--rep_penalty` | `1.1` | 重复惩罚系数（>1 抑制重复）|
| `--lang` | `python` | 默认语言标识 |
| `--device` | 自动检测 | `cuda` / `cpu` / `mps` |

---

## 界面说明

启动后显示：

```
╔══════════════════════════════════════════════════╗
║           CodeGPT  交互式代码补全 REPL           ║
╚══════════════════════════════════════════════════╝
  模型参数:  12层 × 12头 × 768维  (123.59M params)
  设备:      cuda
  温度:      0.3    top_k: 50    最大 tokens: 200
  语言:      python

>>> _
```

- `>>>` — 普通补全提示符
- `>>> [+N字符]` — 当前累积了 N 字符的上下文
- `[FIM] prefix>` — FIM 填空模式，等待输入前缀
- `  ...` — 多行输入模式，等待更多代码行

输出颜色：
- **暗灰色** — 你输入的 prompt（及上下文）
- **亮绿色** — 模型生成的补全内容

---

## 基本用法

### 单行补全

输入一行代码，直接回车：

```
>>> def add(a, b):
```

模型立即输出补全结果：

```
  def add(a, b):
      return a + b       ← 绿色（模型输出）
```

### 多行补全

以 `:` `(` `{` `[` `,` `\` 结尾时自动进入多行模式，**空行**结束输入：

```
>>> class Stack:
  (多行模式，空行结束输入)
  ...     def __init__(self):
  ...     ← 空行，结束输入
```

### 连续对话（上下文累积）

每次补全后，prompt + 补全结果自动累积为上下文，下次输入基于此继续：

```
>>> class Counter:          ← 第一轮

  class Counter:
      def __init__(self):   ← 绿色（模型补全）
          self.count = 0

>>> [+80字符]     def increment(self):   ← 第二轮，模型能看到前面的类定义

  class Counter:
      def __init__(self):
          self.count = 0
      def increment(self):
          self.count += 1   ← 绿色（模型补全，上下文连贯）
```

---

## FIM 填空模式

FIM（Fill-in-the-Middle）：给定前缀和后缀，让模型补全中间部分。

```
>>> /fim
  已切换到 FIM 填空模式。先输入 prefix，再输入 suffix。

[FIM] prefix>  def add(a, b):
[FIM] suffix>      return result

  ┌─ FIM 填空结果 ─────────────────────────────
  │ prefix:  def add(a, b):
  │ infill:      result = a + b      ← 绿色（模型填充）
  │ suffix:      return result
  └───────────────────────────────────────────
```

返回补全模式：`/complete`

---

## 内置命令

在 `>>>` 提示符下输入以下命令：

| 命令 | 说明 | 示例 |
|------|------|------|
| `/help` | 显示命令帮助 | |
| `/fim` | 切换到 FIM 填空模式 | |
| `/complete` | 切换回普通补全模式 | |
| `/context` | 显示当前累积的上下文内容 | |
| `/reset` | 清空上下文，重新开始 | |
| `/temp <n>` | 设置采样温度 | `/temp 0.3` |
| `/tokens <n>` | 设置最大生成 token 数 | `/tokens 300` |
| `/topk <n>` | 设置 top-k 参数 | `/topk 100` |
| `/lang <name>` | 切换语言标识 | `/lang javascript` |
| `/quit` 或 `q` | 退出 REPL | |

---

## 温度参数调优

temperature 对生成质量影响最大：

| 温度 | 特点 | 适用场景 |
|------|------|---------|
| `0.1–0.3` | 保守、确定性强、重复率高 | 补全已知模式（函数定义、类结构）|
| `0.5–0.8` | 平衡创意与连贯性 | 通用代码补全 |
| `1.0–1.5` | 发散、多样但易出现乱码 | 探索式生成 |

**当前模型（1000 iter，3.1M tokens）推荐 `--temperature=0.3`**，在此温度下输出最稳定。

---

## 典型会话示例

### 示例 1：补全函数

```
>>> def fibonacci(n):

  def fibonacci(n):
      if n <= 1:
          return n
      ...
```

### 示例 2：连续扩展类

```
>>> class BinaryTree:

  class BinaryTree:
      def __init__(self):
          self.root = None

>>> [+60字符]     def insert(self, val):

  ...（在已有类定义的上下文中补全 insert 方法）
```

### 示例 3：调整参数再生成

```
>>> /temp 0.8
  温度设为 0.8

>>> /tokens 400
  最大 token 数设为 400

>>> def sort_list(lst):
```

### 示例 4：FIM 填入函数体

```
>>> /fim
[FIM] prefix>  def validate_email(email):
[FIM] suffix>      return is_valid

  │ infill:      import re↵    is_valid = bool(re.match(...))
```

---

## 注意事项

1. **必须用 conda Python 3.12**：系统 Python 3.13 不兼容 GTX 1080 的 PyTorch，会报 CUDA kernel 错误
2. **首次加载约需 5 秒**：1.4GB 检查点从磁盘读取
3. **当前模型局限**：
   - 在 `temperature=0.3` 时效果最好
   - 生成内容受训练数据（PyPro 项目）影响，可能混入特定变量名或中文注释
   - FIM 模式效果不稳定，需要更多训练才能达到实用水平
4. **上下文裁剪**：当上下文超过约 1536 字符时自动保留尾部，会提示裁剪信息
5. **继续训练后重载**：`/quit` 退出后重新启动即可加载新检查点
