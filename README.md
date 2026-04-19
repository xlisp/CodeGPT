# CodeGPT

一个专注于代码生成的 GPT 模型，基于 [nanoGPT](https://github.com/karpathy/nanoGPT) 扩展，支持 Fill-in-the-Middle (FIM) 代码补全和多语言代码生成。

本项目不只是一个模型实现——`docs/` 目录包含了一套完整的技术深潜文档，从 RNN 到 Transformer 到 GPT 的完整进化史，以及对"压缩即智能"、"RLHF 对齐"、"柏拉图表征"等深层问题的探讨，所有概念都结合本项目代码进行解释。

## 深度技术文档

> **如果你想真正理解 GPT 和代码生成背后的原理，从这里开始。**

### [从 RNN 到 CodeGPT：序列建模的进化史](docs/DEEP_DIVE.md)

完整的架构演化脉络，每一个关键转折都对应项目代码中的设计决策：

- **RNN → LSTM → GRU**：门控机制如何缓解梯度消失，为什么仍有根本局限
- **自编码器家族**：从经典 AE → VAE → 去噪自编码器，如何孕育编码器-解码器范式
- **Seq2Seq → 注意力机制**：信息瓶颈的突破，从加法注意力到缩放点积注意力
- **Transformer**：六大核心创新（自注意力、多头、位置编码、残差连接、FFN、并行化），逐一对应 `model.py` 的代码实现
- **GPT 路线**：为什么选择 decoder-only？因果掩码的数学原理、Pre-Norm、权重绑定

### [压缩即智能：从自编码器到 GPT 的认知哲学](docs/COMPRESSION_IS_INTELLIGENCE.md)

三个本质性问题的深度回答：

- **为什么压缩 = 智能？** 不是比喻，是数学等价——交叉熵损失就是在衡量压缩效率。loss 从 10 降到 1.5 的过程就是模型从"一无所知"到"深度理解代码"的过程
- **婴儿学语和神经网络训练是同一件事吗？** 感知 → 模仿 → 内化的过程与输入 → 编码 → 重构的结构同构。"不断重复"的本质是通过迭代微调内部表征
- **GPT 的"解码器"里有编码器吗？** 有。12 层 Transformer Block 就是编码器（99% 的计算），`lm_head` 才是解码器（1 个线性层）。GPT 和 BERT 的唯一架构区别就是一个掩码矩阵

### [强化学习对齐与柏拉图表征](docs/RLHF_AND_PLATONIC_REPRESENTATION.md)

ChatGPT 成功的另外两块关键拼图：

- **RLHF 三级火箭**：预训练（学会说话的婴儿）→ SFT（上学的孩子）→ RLHF（进入社会的成年人）。详解奖励模型、PPO、KL 散度约束，以及 DPO 如何简化整个流程
- **柏拉图表征假说**："盲人摸象"的终局——不同模型从不同维度观察现实，但内部表征趋向收敛。多头注意力是模型内部的 12 个"盲人"，多语言训练让 Python/JS/Rust 收敛到同一个"算法理想型"
- **CodeGPT 在这个图景中的位置**：已实现预训练 + 推理时弱对齐（temperature/top-p），未来可扩展 SFT + DPO

### [深度学习是可微分编程：从 y = wx + b 讲到 CodeGPT](docs/DIFFERENTIABLE_PROGRAMMING.md)

解读 LeCun "Deep Learning est mort. Vive Differentiable Programming!" 背后的架构视角：

- **从线性方程到梯度下降**：为什么有了 `autograd` 之后，"解方程"就被"写前向程序 + `loss.backward()`"取代
- **把经典程序翻译成可微分形式**：`softmax` 是 `argmax` 的可微近似，注意力是查表的可微版本，残差是变量绑定的可微版本
- **`nn.Module` 是函数，`forward` 是 `main`**：CodeGPT 的 `forward`（`model.py:177-198`）就是一段普通 Python，只是每个操作都可微
- **程序员 vs. 梯度下降的分工**：人写架构（层数、注意力 mask、权重绑定、`ignore_index=-1`），梯度下降写参数。训练 = 编译，推理 = 执行

---

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
├── data/
│   ├── python_code/prepare.py  # Python 数据集准备（本地文件/HuggingFace）
│   └── github_code/prepare.py  # 多语言数据集准备（The Stack）
└── docs/
    ├── DEEP_DIVE.md                        # 从 RNN 到 CodeGPT 的完整进化史
    ├── COMPRESSION_IS_INTELLIGENCE.md      # 压缩即智能的认知哲学
    ├── RLHF_AND_PLATONIC_REPRESENTATION.md # RLHF 对齐与柏拉图表征
    └── DIFFERENTIABLE_PROGRAMMING.md       # 深度学习是可微分编程：从线性方程到大模型
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

## 训练报告

### [2026-03-21：引入 autoresearch 训练框架](docs/training-report-2026-03-21/autoresearch-training.md)

从原版 CodeGPT 迁移到 [autoresearch](https://github.com/karpathy/autoresearch) 数据管道与模型架构，在 GTX 1080 上运行。

**核心变化**：

| 问题 | 原版 | 改进后 |
|------|------|--------|
| 训练数据太少 | 28K tokens（本地 Python 文件）| 400B tokens（karpathy/climbmix-400b-shuffle）|
| 架构过时 | nanoGPT（2022）| RoPE + RMSNorm + 滑动窗口 + Value Residual |
| 优化器 | AdamW | Muon（矩阵参数）+ AdamW（其余）|
| 实验周期 | 64 天完整训练 | 5 分钟固定预算，快速迭代 |

GTX 1080 适配（详见文档）：Flash Attention 3 → PyTorch SDPA，bfloat16 → float16 + GradScaler，去掉 torch.compile。

**运行**：
```bash
# 数据准备（一次性，约 5 分钟）
cd /home/xlisp/PyPro/autoresearch
~/miniconda3/envs/codegpt/bin/python prepare.py --num-shards 2

# 训练（5 分钟预算）
cd /home/xlisp/PyPro/CodeGPT
~/miniconda3/envs/codegpt/bin/python -W ignore train_autoresearch.py
```

### [2026-03-13：GTX 1080 初始训练报告](docs/training-report-2026-03-13/)

首次在 GTX 1080 上验证训练流程（50 iter），确认 val loss 从 10.93 降至 5.36，修复 weight tying bug 和 FIM target 长度不匹配 bug。

---

## 致谢

本项目基于 [Andrej Karpathy](https://github.com/karpathy) 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 和 [autoresearch](https://github.com/karpathy/autoresearch) 进行扩展开发。
