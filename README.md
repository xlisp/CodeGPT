# CodeGPT

一个专注于代码生成的 GPT 模型，基于 [nanoGPT](https://github.com/karpathy/nanoGPT) 扩展，支持 Fill-in-the-Middle (FIM) 代码补全和多语言代码生成。

## 特性

- **Fill-in-the-Middle (FIM)**：支持 PSM/SPM 两种模式，模型可根据上下文进行代码填充
- **多语言支持**：Python、JavaScript、TypeScript、Java、C/C++、Go、Rust 等 16 种语言
- **代码感知分词器**：基于 tiktoken GPT-2 BPE，扩展了代码专用特殊 token
- **灵活训练**：支持从头训练、GPT-2 预训练权重微调、断点续训
- **分布式训练**：支持 DDP 多卡训练，混合精度（float16/bfloat16）
- **交互式生成**：REPL 交互模式，实时生成代码

## 项目结构

```
CodeGPT/
├── model.py            # CodeGPT 模型定义（Transformer + FIM + 词表扩展）
├── train.py            # 训练脚本（单卡/多卡 DDP）
├── sample.py           # 代码生成/采样脚本
├── tokenizer.py        # 代码分词器 + FIM 变换
├── configurator.py     # 配置文件解析器
├── bench.py            # 性能基准测试
├── config/
│   ├── train_codegpt.py       # 完整训练配置（124M，多卡）
│   ├── train_codegpt_small.py # 小模型配置（~10M，单卡/CPU）
│   └── finetune_codegpt.py    # GPT-2 微调配置
└── data/
    ├── python_code/prepare.py  # Python 数据集准备（本地文件/HuggingFace）
    └── github_code/prepare.py  # 多语言数据集准备（The Stack）
```

## 快速开始

### 环境依赖

```bash
pip install torch numpy tiktoken
# 可选：
pip install datasets    # 从 HuggingFace 下载数据
pip install wandb       # 训练日志
pip install transformers  # 加载 GPT-2 预训练权重
```

### 1. 准备数据

**从本地 Python 文件：**
```bash
python data/python_code/prepare.py --source=local --code_dir=/path/to/your/python/projects
```

**从 HuggingFace 数据集：**
```bash
python data/python_code/prepare.py --source=huggingface --max_samples=100000
```

**多语言数据集（The Stack）：**
```bash
python data/github_code/prepare.py --langs python javascript typescript --max_samples=20000
```

### 2. 训练模型

**小模型快速测试（~10M 参数）：**
```bash
python train.py config/train_codegpt_small.py
```

**完整模型训练（124M 参数，建议多卡）：**
```bash
# 单卡
python train.py config/train_codegpt.py

# 多卡 DDP
torchrun --standalone --nproc_per_node=4 train.py config/train_codegpt.py
```

**从 GPT-2 微调：**
```bash
python train.py config/finetune_codegpt.py
```

**命令行覆盖参数：**
```bash
python train.py config/train_codegpt_small.py --batch_size=32 --max_iters=5000 --learning_rate=1e-4
```

### 3. 生成代码

**代码补全：**
```bash
python sample.py --prompt="def fibonacci(n):"
```

**FIM 填充：**
```bash
python sample.py --mode=fim --prefix="def add(a, b):" --suffix="    return result"
```

**交互模式：**
```bash
python sample.py --mode=interactive
```

交互模式下可用的命令：
- `/fim <前缀> ||| <后缀>` — Fill-in-the-Middle 填充
- `/lang <语言>` — 设置编程语言
- `/temp <浮点数>` — 设置采样温度
- `/tokens <整数>` — 设置最大生成 token 数
- `/quit` — 退出

### 4. 性能测试

```bash
python bench.py
# 自定义配置
python bench.py --n_layer=6 --n_embd=384 --batch_size=16
```

## 模型配置

| 配置 | 参数量 | 层数 | 注意力头 | 隐藏维度 | 上下文长度 | 用途 |
|------|--------|------|----------|----------|------------|------|
| small | ~10M | 6 | 6 | 384 | 512 | 测试/开发 |
| base | ~124M | 12 | 12 | 768 | 1024 | 标准训练 |
| finetune | ~124M | 12 | 12 | 768 | 1024 | GPT-2 微调 |

## FIM 代码补全

训练时以 50% 的概率对输入进行 FIM 变换，支持两种格式：

- **PSM（Prefix-Suffix-Middle）**：`<|fim_prefix|> 前缀 <|fim_suffix|> 后缀 <|fim_middle|> 中间`
- **SPM（Suffix-Prefix-Middle）**：`<|fim_suffix|> 后缀 <|fim_prefix|> 前缀 <|fim_middle|> 中间`

这使得模型学会根据上下文在指定位置生成代码，适用于 IDE 代码补全场景。

## 特殊 Token

| Token | ID | 用途 |
|-------|-----|------|
| `<\|endoftext\|>` | 50256 | 文本结束 |
| `<\|fim_prefix\|>` | 50257 | FIM 前缀标记 |
| `<\|fim_middle\|>` | 50258 | FIM 中间标记 |
| `<\|fim_suffix\|>` | 50259 | FIM 后缀标记 |
| `<\|fim_pad\|>` | 50260 | FIM 填充 |
| `<\|code_start\|>` | 50261 | 代码开始 |
| `<\|code_end\|>` | 50262 | 代码结束 |
| `<\|lang:python\|>` | 50263 | Python 语言标识 |
| `<\|lang:javascript\|>` | 50264 | JavaScript 语言标识 |
| ... | 50265-50278 | 其他语言标识 |

## 致谢

本项目基于 [Andrej Karpathy](https://github.com/karpathy) 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 进行扩展开发。
