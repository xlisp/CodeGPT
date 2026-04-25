# Code GPT & Learning

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

### [多次 SFT 的灾难性遗忘：SFT 的本质、MoE 的本质](docs/SFT_FORGETTING_AND_MOE.md)

回答一个工程上每个团队都会踩到的坑——"第二次 SFT 把第一次的能力盖掉了"怎么办：

- **SFT 的本质**：和预训练是同一个 loss（`F.cross_entropy(..., ignore_index=-1)`），只在 prompt 段把 target 设为 -1——所以它遵守预训练的所有规律，包括遗忘
- **灾难性遗忘的数学**：梯度下降只看当前 batch，从根上不可能知道"之前学过什么"。用 Fisher 信息矩阵量化每个参数对老任务的重要性
- **解法谱系**：数据混合（rehearsal，工业界默认）、LoRA 多适配器（物理隔离增量）、EWC（给重要参数加弹簧）、多任务联合 SFT（一次训完根治）
- **MoE 的本质**：把 `Block.mlp`（`model.py:93-105`）换成"N 个专家 + 可微路由器"，每个 token 只激活 top-k 个专家——就是**可微分的 if/else**。Mixtral 8x7B 为什么"容量 47B、算力 13B"
- **该选哪条路**：决策表、CodeGPT 要加 function calling 时的具体建议（先联合 SFT，不要急着上 MoE）

### [训练写权重，推理用权重 + 脚手架：SFT / RL 训完之后到底是怎么生效的](docs/SFT_RL_INFERENCE_MECHANICS.md)

回答"使用大模型时是纯靠 transformer 参数预测，还是需要代码配合"这个深层问题。从 `model.py:177-198` 的 `forward` 入手，把大模型系统拆成"W（权重）+ 脚手架（代码）"两层：

- **三步训练的分工**：预训练写入语言/知识（改所有参数，幅度大）→ SFT 写入对话格式（同样的 `F.cross_entropy`，prompt 段 target 设 -1）→ RL 写入偏好/品味（KL 散度约束幅度）。为什么必须这个顺序，为什么跳过 SFT 直接 RL 会崩到"奖励黑客"
- **纯靠 W 的一半 vs 需要代码的一半**：每次 token 预测确实只是 `W · x`，但 chat template / stop_tokens / 采样参数构成的"协议"必须和训练时完全一致——否则 `W` 里的 SFT/RL 能力"沉睡"不醒。本地跑开源模型停不下来，十有八九就是 stop_tokens 没配对
- **W 根本装不下的能力**：tool calling（`execute_tool` 是纯外部代码）、RAG（知识不进 W）、长期记忆（超 `block_size` 必须外存）、安全过滤（双保险）——ChatGPT 是"模型 + 大量代码"的产品
- **回到 CodeGPT**：一张对齐表列出 `sample.py:101 encode_prompt`、`sample.py:98 stop_tokens`、`model.py:279 temperature` 等——本项目的推理脚手架虽然简单但已经完整，是理解"训练-推理对齐"的最小样本

### [RAG 还是 SFT：面对一堆私有数据，该怎么选？](docs/RAG_VS_SFT.md)

"有一堆公司内部文档 / 代码库 / 知识库，怎么让 LLM 学会它们"——这是每个落地团队都会问的问题。本文从 `model.py:177-198` 的 forward 入手，把看似模糊的选型变成一个数学上清晰的二分：

- **RAG 改 `idx`，SFT 改 `W`**：一个是推理时拼 prompt，完全不动参数；一个是跑 `F.cross_entropy` 把知识压进权重。理解这一点，90% 的困惑就没了
- **六维决策表**：数据性质（事实 vs 风格）、更新频率、数据量、可解释性 / 溯源、推理延迟、隐私（含 GDPR "被遗忘权"）——给出什么场景该走哪条路的速查指南
- **为什么默认先做 RAG**：零训练成本、可解释、数据可撤回、和 base model 解耦；以及 RAG 做不到、必须 SFT 的五类场景（风格、领域推理、延迟、离线、抗注入）
- **真正的评估方法**：怎么切数据（train / heldout / 无关三段）、三类 metric（正确率 + 引用准确性 + 通用能力回归）、四条 baseline（base / rag / sft / sft+rag），把"哪个更好"变成可复现的实验结论
- **回到 CodeGPT**：私有代码库场景的具体节奏——先两周搭 RAG，观察短板再决定要不要 continued pretraining

### [物理学的影子：量子力学与统计力学如何塑造了深度学习](docs/PHYSICS_AND_DEEP_LEARNING.md)

为什么很多量子力学、统计力学方向的研究生转去做深度学习几乎没有"门槛"？因为他们脑子里的核心工具在大模型里几乎一一对应。本文把这些对应关系一条条钉到 `model.py` 的具体行号上：

- **Softmax 就是玻尔兹曼分布**：`F.softmax(logits/T)`（`model.py:301`、`model.py:279`）字面上就是 $\exp(-E/kT)/Z$，温度参数直接来自热力学；cross-entropy（`model.py:192`）等价于自由能差
- **注意力机制 = 连续型 Hopfield 网络**：2020 年 *Hopfield Networks is All You Need* 证明了这一点。Q·Kᵀ + softmax + 加权求和（`model.py:66-68`）就是一次自旋玻璃能量下降迭代——所以 2024 年 Hopfield 和 Hinton 拿了诺贝尔物理奖
- **SGD 是 Langevin 动力学**：mini-batch 梯度噪声 + dropout（`train.py:68`）+ 高斯初始化（`model.py:171`）——loss landscape 是球形自旋玻璃，"绝大多数局部极小都几乎一样好"是统计物理给的早期理论支撑
- **Transformer 自带量子力学结构**：embedding 是 Hilbert 空间态向量、注意力内积是 Born rule 的实数版本、`multinomial` 采样就是测量塌缩
- **变分原理一以贯之**：从基态能量、ELBO、到 RLHF 的 KL 约束——同一种带约束变分问题
- **重整化群 = 深度网络的层级抽象**：12 层 Block 沿尺度方向粗粒化；scaling laws 的幂律就是临界点附近的标度行为，"涌现能力"对应相变
- **向量化编程是物理学的母语**：力的合成 → `tok_emb + pos_emb`（`model.py:185`）；张量积态空间 → 多头注意力的 reshape（`model.py:54-56`）；GPU SIMD 就是大向量机
- **熵贯穿训练全流程**：Boltzmann S = k log W 与 Shannon H = -Σp log p 同构；cross-entropy 是熵下降，attention 熵是可解释性指标，RL 的 entropy bonus 就是自由能的 -TS 项
- **路径积分 = 自回归生成**：序列 likelihood 即 Feynman 求和，beam search 是鞍点近似，diffusion model 直接照搬非平衡热力学
- **微积分是底层引擎**：backprop = 链式法则的工业化（也是最工整的 Feynman 图求和）；残差连接是 Euler 法解 ODE（`model.py:103-104`）；LayerNorm 是 normal coordinates；softmax 是 argmax 的可微化
- **思想史地图**：28 行对应表 + 写给打算从物理转 AI 的研究生的方法论建议——你不是在跨界，你在回家

### [符号主义、贝叶斯网络、深度学习：三种 AI 范式的对比学习](docs/SYMBOLIC_BAYES_NEURAL.md)

回答一个常见但深刻的直觉性问题——**"GPT 是不是把训练数据当成一堆可执行的符号代码来执行？"** 把符号主义 AI、贝叶斯网络、深度学习放在同一张桌子上，从"都在解 $P(y \mid x)$"这个共同问题出发，逐条钉到 `model.py` / `tokenizer.py` 的具体行号上：

- **三个范式的一句话总结**：符号主义（人写规则）→ 贝叶斯网络（人画图 + 数据估参数）→ 深度学习（人写架构 + 梯度下降写参数）。一条"知识从人写到数据写"的演化主线
- **类比对在哪、错在哪**：训练数据 = 可执行符号代码，70% 是对的（数据塑造行为），30% 错在四处——硬规则 vs 软统计、谁在执行什么、可枚举 vs 不可枚举、例子 ≠ 规则。更精确的版本是"训练数据是规约，权重是编译产物，SGD 是编译器"
- **贝叶斯网络是中间态**：图结构是符号的（可读、可枚举），CPT 是概率的（从数据学）；解释为什么 Naive Bayes / HMM / VAE / 扩散模型都是它的衍生
- **三种范式 13 维对照表**：知识载体、可解释性、数据效率、形式保证、典型失败模式……没有谁是赢家，每条边界都有最适合的范式
- **CodeGPT 的三明治结构**：符号层（`<|fim_*|>` 模板 + `stop_tokens`）+ 概率层（`F.softmax` → `multinomial`，`model.py:301-302`）+ 神经层（12×Block, 124M 权重）。任何能跑的大模型系统都是这种夹心——纯神经搞不定"什么时候停止生成"
- **现代系统都是神经-符号混合**：tool calling、verifier-based RL（o1/R1）、RAG、Lean+LLM——三派各自占据自己擅长的层，而不是某一派胜出

### [合成数据：怎么把一堆垃圾代码变成高质量训练数据](docs/SYNTHETIC_DATA.md)

回答"Claude 怎么把垃圾代码变废为宝"这个实战问题。把"合成数据"从一项模糊的技术拆成一条有具体工具链的流水线：

- **先定义"垃圾"**：F1 语法错、F2 语义错、F3 风格噪声、F4 任务无关——四种故障需要四种不同工具，没有银弹
- **六环节流水线**：过滤（AST/ruff/mypy 的确定性标注）→ 修复（black/isort 的规则改写）→ 执行反馈（interpreter 是最诚实的 reward）→ 强模型蒸馏（output / input-output / 思维链三层）→ Self-Instruct 循环造题造解 → Self-Refine 让模型批评自己
- **SFT vs RL 对"高质量"的定义根本不同**：SFT 要 `(prompt, response)` 且 response 必须是"想让模型学会的样子"；RL 要 `(prompt, chosen, rejected)` 且优先 verifiable reward（测试通过率、字符串匹配），不可验证的 reward 留给 DPO
- **人工标注的四个"最划算点"**：种子（100-1000 条）、分歧样本（active learning）、安全/合规边界、eval set。总人工占比应控制在 1%-5%——低了种子不够，高了没用好合成
- **回到 CodeGPT**：基于现有 `prepare.py` + `apply_fim_transform` 的最小可行升级路径——加过滤层、加修复层、加 doctest 执行反馈、合成 SFT/RL 数据，一条到能跑 SFT + 初步 RL 的完整路线

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
    ├── DIFFERENTIABLE_PROGRAMMING.md       # 深度学习是可微分编程：从线性方程到大模型
    ├── SFT_FORGETTING_AND_MOE.md           # 多次 SFT 的灾难性遗忘与 MoE 的本质
    ├── SFT_RL_INFERENCE_MECHANICS.md       # 训练写权重，推理用权重 + 脚手架：SFT/RL 如何在使用时生效
    ├── RAG_VS_SFT.md                       # RAG 还是 SFT：私有数据的选型与评估方法
    ├── SYNTHETIC_DATA.md                   # 合成数据：垃圾代码变废为宝的六环节流水线
    ├── SYMBOLIC_BAYES_NEURAL.md            # 符号主义、贝叶斯网络、深度学习：三种 AI 范式对比
    └── PHYSICS_AND_DEEP_LEARNING.md        # 物理学的影子：量子力学与统计力学如何塑造了深度学习
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
