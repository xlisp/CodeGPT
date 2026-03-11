# 从 RNN 到 CodeGPT：序列建模的进化史

> 本文从最早的循环神经网络出发，沿着自编码器家族、注意力机制、Transformer 架构、GPT 系列的脉络，一直讲到本项目 CodeGPT 的实现细节。每一个关键转折点，都对应着本项目代码中可以触摸到的设计决策。

---

## 目录

1. [序幕：序列建模问题](#1-序幕序列建模问题)
2. [RNN 时代（1986-2014）](#2-rnn-时代1986-2014)
3. [自编码器家族与表征学习](#3-自编码器家族与表征学习)
4. [Seq2Seq 与编码器-解码器范式](#4-seq2seq-与编码器-解码器范式)
5. [注意力机制：瓶颈的突破](#5-注意力机制瓶颈的突破)
6. [Transformer：Attention Is All You Need](#6-transformerattention-is-all-you-need)
7. [GPT：解码器的单飞](#7-gpt解码器的单飞)
8. [GPT-2 / GPT-3：规模涌现](#8-gpt-2--gpt-3规模涌现)
9. [CodeGPT：当 GPT 遇见代码](#9-codegpt当-gpt-遇见代码)
10. [总结：一张进化图谱](#10-总结一张进化图谱)

---

## 1. 序幕：序列建模问题

自然语言和程序代码都是**序列数据**——每个 token 的含义依赖于它前面（甚至后面）的上下文。序列建模的核心问题是：

> **给定已经出现的 token 序列 x₁, x₂, ..., xₜ，预测下一个 token xₜ₊₁ 的概率分布。**

这正是本项目 CodeGPT 训练的目标。在 `model.py` 的前向传播中，这个目标被直接表达为：

```python
# model.py:190-192 — CodeGPT 的训练目标
if targets is not None:
    logits = self.lm_head(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
```

交叉熵损失函数衡量的就是：模型预测的下一个 token 概率分布与真实 token 之间的距离。这个目标从 RNN 时代一直沿用至今，但实现它的架构经历了天翻地覆的变化。

---

## 2. RNN 时代（1986-2014）

### 2.1 原始 RNN（1986）

Rumelhart、Hinton 和 Williams 在 1986 年提出的反向传播算法使得训练循环网络成为可能。RNN 的核心思想极其简洁：

```
                    ┌──────────┐
                    │          │
           ┌──────►│  hₜ = f(W·hₜ₋₁ + U·xₜ + b)
           │       │          │
           │       └────┬─────┘
           │            │
    hₜ₋₁ ─┘            ▼
                       yₜ = g(V·hₜ)
```

用伪代码表示：

```python
# 原始 RNN —— 概念示意（非项目代码）
class SimpleRNN:
    def forward(self, x_sequence):
        h = zeros(hidden_size)          # 隐状态初始化为零
        outputs = []
        for x_t in x_sequence:          # 逐 token 顺序处理
            h = tanh(W_hh @ h + W_xh @ x_t + b)  # 隐状态递推
            y_t = W_hy @ h              # 输出
            outputs.append(y_t)
        return outputs
```

**致命缺陷：梯度消失/爆炸。** 隐状态 h 被反复乘以同一个权重矩阵 W_hh，经过几十步后梯度要么消失（小于 1 的值连乘）要么爆炸（大于 1 的值连乘）。这意味着 RNN 实际上只能记住大约 10-20 步以内的上下文。

### 2.2 LSTM（1997）与 GRU（2014）

Hochreiter 和 Schmidhuber 在 1997 年提出 LSTM（Long Short-Term Memory），通过**门控机制**缓解梯度消失：

```
        ┌────────────────────────────────────────┐
        │            Cell State Cₜ               │ ← 信息高速公路
        │  ┌──────┐  ┌──────┐  ┌──────┐          │
  xₜ ──►│  │遗忘门│  │输入门│  │输出门│          │
  hₜ₋₁─►│  │  fₜ  │  │  iₜ  │  │  oₜ  │          │
        │  └──┬───┘  └──┬───┘  └──┬───┘          │
        │     │         │         │               │
        │  Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙tanh(...)         │
        │  hₜ = oₜ ⊙ tanh(Cₜ)                   │
        └────────────────────────────────────────┘
```

```python
# LSTM —— 概念示意
class LSTMCell:
    def forward(self, x_t, h_prev, c_prev):
        gates = sigmoid(W @ [h_prev, x_t])
        f, i, o = split(gates, 3)          # 遗忘门、输入门、输出门
        c_candidate = tanh(W_c @ [h_prev, x_t])
        c = f * c_prev + i * c_candidate   # cell state 更新：加法而非乘法！
        h = o * tanh(c)                    # 输出
        return h, c
```

关键创新：cell state 的更新用的是**加法** `f * c_prev + i * c_candidate`，而不是纯乘法。这条"信息高速公路"让梯度可以无损地流过很长的序列。

2014 年 Cho 等人提出的 GRU（Gated Recurrent Unit）将三个门简化为两个（重置门和更新门），效果相近但参数更少。

### 2.3 RNN 的根本局限

即使有了 LSTM/GRU，RNN 家族仍有两个根本性问题：

1. **顺序计算**：必须逐 token 处理，无法并行。处理 1024 个 token 需要 1024 步串行计算。
2. **长距离依赖仍然困难**：虽然比原始 RNN 好很多，但面对数百行的代码文件，LSTM 仍然力不从心。

对比我们 CodeGPT 的做法——在 `model.py:186-187` 中：

```python
# CodeGPT: 所有 Block 并行处理整个序列
for block in self.transformer.h:
    x = block(x)   # x 的形状是 (batch, seq_len, embed_dim)，整个序列一次性处理
```

每个 Block 一次看到**整个序列的所有位置**，这是 Transformer 对 RNN 的根本性超越。

---

## 3. 自编码器家族与表征学习

在 RNN 处理序列问题的同时，另一条独立的研究线——**自编码器**——在探索如何学习数据的压缩表示。这条线最终汇入 Transformer 的编码器-解码器架构。

### 3.1 经典自编码器（1986）

```
  输入 x ──► [编码器 E] ──► 瓶颈 z ──► [解码器 D] ──► 重构 x̂
                              │
                         低维表示
              目标：最小化 ‖x - x̂‖²
```

```python
# 经典自编码器 —— 概念示意
class AutoEncoder:
    def __init__(self, input_dim, latent_dim):
        self.encoder = Linear(input_dim, latent_dim)    # 压缩
        self.decoder = Linear(latent_dim, input_dim)    # 重构

    def forward(self, x):
        z = relu(self.encoder(x))     # 编码：高维 → 低维
        x_hat = self.decoder(z)       # 解码：低维 → 高维
        loss = mse(x, x_hat)          # 重构损失
        return x_hat, loss
```

### 3.2 变分自编码器 VAE（2013）

Kingma 和 Welling 将概率推断引入自编码器，瓶颈层不再是一个确定的向量，而是一个**概率分布**：

```
  x ──► 编码器 ──► μ, σ ──► z ~ N(μ, σ²) ──► 解码器 ──► x̂
                                │
                    KL散度约束：让 z 接近标准正态
```

VAE 的重要遗产：**潜在空间是连续的、可插值的**。这个思想后来影响了 Transformer 中的连续向量表示。

### 3.3 去噪自编码器（2008）→ BERT 的祖先

Vincent 等人的去噪自编码器给输入加上噪声（遮挡、打乱），训练模型恢复原始输入。这直接启发了后来的两大方向：

```
去噪自编码器（2008）
    │
    ├──► BERT（2018）—— 遮挡 15% 的 token，训练模型预测被遮挡的 token
    │                    （Masked Language Model）
    │
    └──► Diffusion Models（2020）—— 逐步加噪再去噪生成图像
```

### 3.4 自编码器 → 编码器-解码器的跳跃

自编码器建立了一个关键的思维模型：

> **编码器**负责理解输入，把信息压缩为中间表示；
> **解码器**负责从中间表示生成输出。

当我们把这个框架从"重构相同的输入"推广到"生成不同的输出"（比如从英语生成法语），就得到了 Seq2Seq 模型。

---

## 4. Seq2Seq 与编码器-解码器范式

### 4.1 Seq2Seq（2014）

Sutskever、Vinyals 和 Le 在 2014 年提出了 Sequence to Sequence 模型，将自编码器的编码-解码思想与 RNN 结合：

```
编码器（读入源序列）:
  "I love code" → LSTM → LSTM → LSTM → [context vector c]

解码器（生成目标序列）:
  [context vector c] → LSTM → "我" → LSTM → "爱" → LSTM → "代码"
```

```python
# Seq2Seq —— 概念示意
class Seq2Seq:
    def __init__(self):
        self.encoder = LSTM(input_dim, hidden_dim)
        self.decoder = LSTM(hidden_dim, output_dim)

    def forward(self, source, target):
        # 编码：整个源序列压缩为一个向量
        for token in source:
            _, hidden = self.encoder(token, hidden)
        context = hidden  # 最后一步的隐状态 = 全部信息

        # 解码：从 context 逐步生成
        outputs = []
        for token in target:
            output, hidden = self.decoder(token, hidden)
            outputs.append(output)
        return outputs
```

**关键瓶颈：** 整个源序列的信息必须被压缩到一个固定大小的 context vector 中。当源序列很长时，信息必然丢失。这就是所谓的"信息瓶颈"问题。

---

## 5. 注意力机制：瓶颈的突破

### 5.1 Bahdanau 注意力（2014）

Bahdanau、Cho 和 Bengio 提出了突破性的想法：**解码器在生成每个 token 时，不使用固定的 context vector，而是动态地"注意"编码器的所有位置。**

```
编码器输出:  h₁   h₂   h₃   h₄   h₅
              │    │    │    │    │
              ▼    ▼    ▼    ▼    ▼
            ┌──────────────────────┐
            │   注意力权重 αᵢ       │ ← 每个位置分配不同的权重
            │   α₁  α₂  α₃  α₄  α₅│
            └──────────┬───────────┘
                       │
                       ▼
              context = Σ αᵢ · hᵢ    ← 加权求和
                       │
                       ▼
                    解码器
```

```python
# Bahdanau 注意力 —— 概念示意
def attention(decoder_hidden, encoder_outputs):
    # decoder_hidden: 当前解码器状态
    # encoder_outputs: 编码器所有位置的输出 [seq_len, hidden]
    scores = []
    for h_enc in encoder_outputs:
        score = tanh(W @ [decoder_hidden, h_enc])  # 计算相关性
        scores.append(score)
    alpha = softmax(scores)                         # 归一化为概率
    context = sum(alpha_i * h_i for alpha_i, h_i in zip(alpha, encoder_outputs))
    return context
```

这是一个里程碑式的发现：**让模型自己学习应该关注输入的哪些部分**。但此时注意力仍然是 RNN 的"附属品"——编码器和解码器仍然是 LSTM。

### 5.2 从加法注意力到点积注意力

Bahdanau 的注意力使用一个小型神经网络计算相关性分数（加法注意力）。Luong（2015）简化为**点积注意力**：

```
score(query, key) = query · key    # 就是一个向量点积！
```

这个简化极其重要——它让注意力计算可以完全用矩阵乘法表示，从而能被 GPU 高效并行。

这正是我们 CodeGPT 中注意力计算的核心。在 `model.py:66` 中：

```python
# model.py:66 — 缩放点积注意力
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
```

`q @ k.transpose(-2, -1)` 就是 query 和 key 的点积，`1/√d` 是缩放因子，防止点积值过大导致 softmax 梯度消失。

---

## 6. Transformer：Attention Is All You Need

### 6.1 革命性论文（2017）

2017 年，Vaswani 等人在 Google Brain 发表了 *"Attention Is All You Need"*，提出了一个大胆的想法：

> **完全抛弃 RNN，只用注意力机制构建整个模型。**

```
原始 Transformer 架构（编码器-解码器）:

    ┌─────────────────────────────────────────────┐
    │                                             │
    │   编码器 (×N)          解码器 (×N)           │
    │  ┌─────────────┐     ┌──────────────────┐   │
    │  │ Self-Attn    │     │ Masked Self-Attn │   │
    │  │ (双向)       │     │ (单向/因果)       │   │
    │  ├─────────────┤     ├──────────────────┤   │
    │  │ Feed-Forward │     │ Cross-Attention  │   │
    │  │              │     │ (看编码器输出)     │   │
    │  └─────────────┘     ├──────────────────┤   │
    │                      │ Feed-Forward     │   │
    │                      └──────────────────┘   │
    │                                             │
    └─────────────────────────────────────────────┘
```

### 6.2 核心创新一：自注意力（Self-Attention）

之前的注意力是编码器和解码器之间的"交叉注意力"。Transformer 的自注意力让**序列内部的每个位置都能直接关注其他所有位置**：

```
"def fibonacci(n):" 中每个 token 的自注意力:

  def  fibonacci  (  n  )  :
   │       │      │  │  │  │
   ├───────┼──────┼──┼──┼──┤    ← "fibonacci" 关注 "def"（我是一个函数名）
   ├───────┼──────┼──┼──┼──┤    ← "n" 关注 "fibonacci" 和 "()" （我是参数）
   ├───────┼──────┼──┼──┼──┤    ← ":" 关注 "def"（函数定义的结尾）
```

这就是 CodeGPT 中 `CausalSelfAttention` 做的事情。关键的 Q/K/V 投影：

```python
# model.py:36 — 一次线性变换同时计算 Q、K、V
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

# model.py:53 — 分割为 query, key, value
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```

直觉理解：
- **Query（查询）**= "我在找什么信息？"
- **Key（键）**= "我提供什么信息？"
- **Value（值）**= "我实际携带的内容。"

注意力权重 = Query 与 Key 的相似度。最终输出 = 权重加权的 Value 求和。

### 6.3 核心创新二：多头注意力（Multi-Head Attention）

单一的注意力只能捕捉一种关系模式。多头注意力让模型同时关注不同类型的关系：

```
  Head 1: 关注语法关系    （"n" → "fibonacci" 表示参数属于这个函数）
  Head 2: 关注位置关系    （")" → "(" 表示括号匹配）
  Head 3: 关注类型关系    （"n" → "int" 表示类型信息）
  ...
  Head 12: 关注其他模式
```

在 CodeGPT 中，多头注意力通过 reshape 实现：

```python
# model.py:54-56 — 将 embedding 拆分为多个头
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
# 形状变为 (batch, n_head, seq_len, head_dim)
# 每个头独立计算注意力，关注不同的特征子空间
```

CodeGPT 默认 12 个头，768 维嵌入，每个头 64 维（`768 / 12 = 64`）。

### 6.4 核心创新三：位置编码

RNN 天然具有位置信息（按顺序处理）。Transformer 一次看到所有位置，需要显式注入位置信息：

```python
# model.py:145-146 — CodeGPT 的位置编码
wte=nn.Embedding(config.vocab_size, config.n_embd),   # token embedding
wpe=nn.Embedding(config.block_size, config.n_embd),   # position embedding

# model.py:183-185 — 前向传播中将两者相加
tok_emb = self.transformer.wte(idx)    # 每个 token 的语义向量
pos_emb = self.transformer.wpe(pos)    # 每个位置的位置向量
x = self.transformer.drop(tok_emb + pos_emb)  # 相加！
```

原始 Transformer 用的是固定的正弦/余弦位置编码。GPT 系列（以及本项目）改为**可学习的位置嵌入**——让模型自己学习位置的含义。`block_size=1024` 意味着模型最多能处理 1024 个 token 的上下文。

### 6.5 核心创新四：残差连接 + 层归一化

深层网络的训练难题（梯度消失）通过残差连接解决：

```python
# model.py:102-104 — Transformer Block 中的残差连接
def forward(self, x):
    x = x + self.attn(self.ln_1(x))   # 残差 + 注意力
    x = x + self.mlp(self.ln_2(x))    # 残差 + 前馈网络
    return x
```

注意这里用的是 **Pre-Norm** 架构（先 LayerNorm 再 Attention），而不是原始 Transformer 的 Post-Norm。GPT-2 引入了这个变化，使得训练更稳定。

```
原始 Transformer (Post-Norm):       GPT-2 / CodeGPT (Pre-Norm):
  x → Attention → Add → LayerNorm     x → LayerNorm → Attention → Add
  x → FFN → Add → LayerNorm           x → LayerNorm → FFN → Add
```

### 6.6 前馈网络（FFN）

每个 Transformer Block 中，注意力层之后是一个两层的前馈网络：

```python
# model.py:76-90 — MLP（前馈网络）
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)    # 升维 ×4
        self.gelu = nn.GELU()                                       # 激活函数
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)   # 降维
        self.dropout = nn.Dropout(config.dropout)
```

768 → 3072 → 768。先升维到 4 倍，经过非线性激活（GELU），再降回来。这个 4 倍的扩展率是 Transformer 的标准设计。

GELU（Gaussian Error Linear Unit）是 GPT 对原始 Transformer 使用 ReLU 的改进，提供更平滑的梯度。

### 6.7 Transformer vs RNN：决定性优势

| 特性 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 并行性 | 串行（逐 token） | 全并行 |
| 最远距离 | O(n) 步传递 | O(1) 直接连接 |
| 训练速度 | 慢 | 快几个数量级 |
| 长上下文 | ~数百 token | 1024+（可扩展） |

---

## 7. GPT：解码器的单飞

### 7.1 Transformer 的三大分支

原始 Transformer 包含编码器和解码器两部分。后续研究分化为三条路线：

```
         原始 Transformer (2017)
         编码器 + 解码器
              │
    ┌─────────┼──────────┐
    │         │          │
    ▼         ▼          ▼
  编码器      编码器+     解码器
  only       解码器      only
    │         │          │
    ▼         ▼          ▼
  BERT      T5/BART     GPT
 (2018)     (2019)     (2018)
    │                    │
    ▼                    ▼
  理解任务            生成任务
  (分类/NER)         (文本/代码生成)
```

- **编码器 only（BERT）**：双向注意力，擅长理解（分类、抽取）
- **编码器-解码器（T5）**：完整架构，擅长翻译、摘要
- **解码器 only（GPT）**：单向注意力，擅长生成

### 7.2 GPT-1（2018）：无监督预训练 + 有监督微调

Radford 等人在 OpenAI 做出了关键决策：

> **只使用 Transformer 的解码器部分，通过语言建模预训练，然后微调到下游任务。**

GPT 使用**因果注意力（Causal Attention）**：每个 token 只能看到自己和前面的 token，不能看到未来。

这就是 CodeGPT 中 "Causal" 的含义：

```python
# model.py:44-49 — 因果掩码：下三角矩阵
if not self.flash:
    self.register_buffer(
        "bias",
        torch.tril(torch.ones(config.block_size, config.block_size))
        .view(1, 1, config.block_size, config.block_size),
    )

# model.py:67 — 用掩码阻止注意力看到未来的 token
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
```

可视化因果掩码（`torch.tril` 生成的下三角矩阵）：

```
位置:    0   1   2   3   4
    0  [ 1   0   0   0   0 ]    ← token 0 只能看自己
    1  [ 1   1   0   0   0 ]    ← token 1 看 0,1
    2  [ 1   1   1   0   0 ]    ← token 2 看 0,1,2
    3  [ 1   1   1   1   0 ]    ← token 3 看 0,1,2,3
    4  [ 1   1   1   1   1 ]    ← token 4 看所有

    0 的位置被 masked_fill 设为 -∞，经过 softmax 后权重变为 0
```

为什么要这样？因为训练时，模型需要**同时预测序列中每个位置的下一个 token**。如果 token 3 能看到 token 4，那预测 token 4 就变成了作弊。

PyTorch 2.0+ 提供了 FlashAttention 优化，CodeGPT 自动检测并使用：

```python
# model.py:43,58-64 — FlashAttention 加速
self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
if self.flash:
    y = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=self.dropout if self.training else 0,
        is_causal=True,   # 内置因果掩码，更高效
    )
```

### 7.3 为什么是解码器而不是编码器？

自编码器家族 → BERT（编码器）路线选择了**理解**：给模型一段挖了洞的文本，让它填空。这是双向的——填空时可以看到前后文。

GPT 选择了**生成**：给模型开头，让它续写。这是单向的——写作时只能看到已经写好的部分。

对于代码生成，单向的 GPT 架构天然匹配——程序员就是从上到下、从左到右写代码的。

---

## 8. GPT-2 / GPT-3：规模涌现

### 8.1 GPT-2（2019）：规模的力量

GPT-2 做的修改出奇地少，主要是：
1. **更大**：从 1.17 亿参数扩展到 15 亿
2. **Pre-Norm**：LayerNorm 移到注意力/FFN 之前（CodeGPT 继承了这个设计）
3. **更多数据**：WebText 数据集，40GB 文本

本项目的 CodeGPT 完全继承了 GPT-2 的架构。`from_pretrained` 方法可以直接加载 GPT-2 的预训练权重：

```python
# model.py:310-312 — 支持从 GPT-2 初始化
@classmethod
def from_pretrained(cls, model_type, override_args=None):
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
```

四种 GPT-2 规模（也是 CodeGPT 可以加载的规模）：

```python
# model.py:317-322 — GPT-2 家族的配置
config_args = {
    'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),    # 124M
    'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),   # 350M
    'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),   # 774M
    'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M
}
```

### 8.2 GPT-3（2020）：In-Context Learning

GPT-3 将参数扩展到 1750 亿，发现了一个重要现象：**模型足够大之后，不需要微调，只需要在 prompt 中给出几个例子，就能完成新任务。** 这就是 few-shot / in-context learning。

### 8.3 权重初始化的秘密

训练如此深的网络，初始化至关重要。CodeGPT 继承了 GPT-2 的初始化策略：

```python
# model.py:157-159 — 残差投影的特殊初始化
for pn, p in self.named_parameters():
    if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
```

为什么要除以 `√(2 * n_layer)`？因为每一层有两个残差连接（attention + MLP），共 `2 * n_layer` 个残差路径。如果每个残差分支的输出方差为 1，经过 N 个残差加法后方差会变成 N。这个缩放确保了输出方差不会随深度爆炸。

### 8.4 权重绑定（Weight Tying）

```python
# model.py:152-153 — 输入嵌入和输出层共享权重
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
self.transformer.wte.weight = self.lm_head.weight  # 权重绑定！
```

token embedding（从 token ID 到向量）和 LM head（从向量到 token 概率）共享同一个权重矩阵。直觉：一个 token 的"含义向量"既用于输入表示，也用于输出预测。这减少了参数量，也提供了有效的正则化。

---

## 9. CodeGPT：当 GPT 遇见代码

### 9.1 代码生成的特殊挑战

代码不是自然语言。它有独特的结构特征：

```
自然语言：                          代码：
- 语义模糊，容错性高                 - 语法严格，一个字符错就无法运行
- 上下文窗口通常够用                 - 函数调用可能跨越数百行
- 单一语言                          - 多语言（Python, JS, Rust, ...）
- 线性叙述                          - 嵌套结构（函数/类/模块）
- 没有填充需求                       - IDE 需要在光标位置补全（FIM）
```

CodeGPT 通过三个关键设计应对这些挑战。

### 9.2 设计一：代码感知分词器

GPT-2 的 BPE 分词器为自然语言设计，但也适用于代码。CodeGPT 在其基础上扩展了代码专用的特殊 token：

```python
# tokenizer.py:18-43 — 特殊 token 定义
SPECIAL_TOKENS = {
    "<|endoftext|>":  50256,     # 原始 GPT-2 token
    # --- 以下为 CodeGPT 新增 ---
    "<|fim_prefix|>": 50257,     # FIM 前缀标记
    "<|fim_middle|>": 50258,     # FIM 中间标记
    "<|fim_suffix|>": 50259,     # FIM 后缀标记
    "<|fim_pad|>":    50260,     # FIM 填充
    "<|code_start|>": 50261,     # 代码段开始
    "<|code_end|>":   50262,     # 代码段结束
    "<|lang:python|>":     50263,  # 语言标识
    "<|lang:javascript|>": 50264,
    # ... 共 16 种语言
}
```

当加载 GPT-2 预训练权重时，需要扩展词表以容纳新 token：

```python
# model.py:356-380 — 词表扩展
def expand_vocab(self, new_vocab_size):
    old_wte = self.transformer.wte
    new_wte = nn.Embedding(new_vocab_size, self.config.n_embd)
    new_wte.weight.data[:old_vocab_size] = old_wte.weight.data  # 保留旧权重
    nn.init.normal_(new_wte.weight.data[old_vocab_size:], mean=0.0, std=0.02)  # 新 token 随机初始化
    self.transformer.wte = new_wte
    # ... 同样扩展 lm_head，并维持权重绑定
```

### 9.3 设计二：Fill-in-the-Middle (FIM)

这是 CodeGPT 与纯 GPT 最重要的区别。普通 GPT 只能从左到右续写，但 IDE 中的代码补全需要在**中间**插入代码：

```
程序员正在编辑的代码：
  def add(a, b):
      █                    ← 光标在这里
      return result

需要填充的代码：
      result = a + b
```

FIM 通过巧妙的数据变换实现这一点，**不需要修改模型架构**：

```python
# tokenizer.py:148-192 — FIM 变换
def apply_fim_transform(tokens, fim_rate=0.5, fim_spm_rate=0.5):
    # 随机选两个切分点，将序列分为 prefix/middle/suffix
    boundaries = sorted(random.sample(range(1, len(tokens)), 2))
    prefix = tokens[:boundaries[0]]
    middle = tokens[boundaries[0]:boundaries[1]]
    suffix = tokens[boundaries[1]:]

    if random.random() < fim_spm_rate:
        # SPM 格式：suffix-prefix-middle
        return [suffix_id] + suffix + [prefix_id] + prefix + [middle_id] + middle
    else:
        # PSM 格式：prefix-suffix-middle
        return [prefix_id] + prefix + [suffix_id] + suffix + [middle_id] + middle
```

以一段 Python 代码为例：

```
原始序列:  def add(a, b):    result = a + b    return result

FIM (PSM) 变换后:
  <|fim_prefix|> def add(a, b):    <|fim_suffix|>    return result <|fim_middle|> result = a + b

解读：模型看到前缀 "def add(a, b):" 和后缀 "return result"，
      需要在 <|fim_middle|> 之后生成中间部分 "result = a + b"
```

训练时以 50% 的概率应用 FIM 变换：

```python
# train.py:141-163 — 在数据加载时应用 FIM
if fim_enabled and split == 'train':
    for b in range(batch_size):
        tokens = x[b].tolist()
        tokens_transformed = apply_fim_transform(tokens, fim_rate=fim_rate, ...)
```

这样训练出的模型**同时掌握了两种能力**：
- 50% 的样本：标准的从左到右生成
- 50% 的样本：根据上下文填充中间代码

### 9.4 设计三：自回归生成与采样策略

代码生成的质量高度依赖采样策略。CodeGPT 实现了完整的采样工具箱：

```python
# model.py:254-307 — 自回归生成
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
             stop_tokens=None, repetition_penalty=1.0):
    for _ in range(max_new_tokens):
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature   # ① 温度控制
```

**① 温度（Temperature）**

```
temperature = 0.1 → 几乎确定性，总是选最高概率的 token（适合补全已知模式）
temperature = 0.8 → 适度随机（默认值，平衡准确性和多样性）
temperature = 1.5 → 高度随机（探索性生成，可能出现创意但也可能出错）

原理：logits / temperature
  温度低 → 放大概率差距 → softmax 后分布更尖锐
  温度高 → 缩小概率差距 → softmax 后分布更平坦
```

**② Top-k 采样**

```python
        # model.py:284-286
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
```

只保留概率最高的 k 个 token，其余设为负无穷。防止模型选到极不可能的 token。

**③ Top-p（核采样 / Nucleus Sampling）**

```python
        # model.py:289-296
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
```

按概率从高到低排列，累计概率达到 p（如 0.95）后，截断剩余 token。比 top-k 更灵活：当模型很确定时只考虑少数几个 token，不确定时自动扩大候选范围。

**④ 重复惩罚（Repetition Penalty）**

```python
        # model.py:279-281
        if repetition_penalty != 1.0:
            for token_id in set(idx[0].tolist()):
                logits[0, token_id] /= repetition_penalty
```

已经出现过的 token 概率被缩小，防止代码生成时陷入重复循环（如不停生成相同的变量名或语句）。

### 9.5 完整的训练流程

将上面所有组件串联起来，CodeGPT 的训练流程如下：

```
┌──────────────────────────────────────────────────────────┐
│                    数据准备                               │
│  代码文件 → 语言检测 → BPE 分词 → 添加特殊 token → .bin   │
│  (prepare.py)                (tokenizer.py)              │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│                    训练循环                               │
│                                                          │
│  for each iteration:                                     │
│    ┌──────────────────────────────────────────────┐       │
│    │ 1. 从 .bin 随机采样 batch                    │       │
│    │ 2. 50% 概率应用 FIM 变换                     │       │
│    │ 3. 前向传播: token_emb + pos_emb → Blocks → │       │
│    │    → LayerNorm → LM_head → logits           │       │
│    │ 4. 计算交叉熵损失                            │       │
│    │ 5. 反向传播 + 梯度裁剪 + AdamW 更新          │       │
│    └──────────────────────────────────────────────┘       │
│                                                          │
│  学习率: 线性 warmup → 余弦衰减                          │
│  混合精度: float16/bfloat16                              │
│  分布式: DDP 多卡同步                                    │
│  (train.py)                                              │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│                    代码生成                               │
│  prompt → encode → 自回归采样 → decode → 代码输出        │
│  支持: 补全 / FIM 填充 / 交互式 REPL                      │
│  (sample.py)                                             │
└──────────────────────────────────────────────────────────┘
```

---

## 10. 总结：一张进化图谱

```
1986  原始 RNN          ← 序列建模的开端
  │   (Rumelhart)         缺陷：梯度消失，无法处理长序列
  │
  │   自编码器            ← 表征学习的开端
  │   (Rumelhart)         编码器-解码器思想的萌芽
  │
1997  LSTM               ← 门控机制缓解梯度消失
  │   (Hochreiter)        cell state "高速公路"
  │
2008  去噪自编码器        ← 遮挡-恢复的训练范式
  │   (Vincent)            ↓ 启发 BERT 的 MLM
  │
2013  VAE                ← 概率化的潜在空间
  │   (Kingma)             连续表示 → 影响 Transformer
  │
2014  Seq2Seq            ← 编码器-解码器处理序列到序列
  │   (Sutskever)          缺陷：固定长度 context vector 瓶颈
  │
  │   GRU                ← 简化的门控 RNN
  │   (Cho)
  │
  │   注意力机制           ← 动态关注源序列不同位置
  │   (Bahdanau)            突破了信息瓶颈
  │
2017  Transformer ★       ← "Attention Is All You Need"
  │   (Vaswani)             自注意力 + 多头 + 位置编码
  │                         完全并行，告别 RNN
  │
  ├──► BERT (2018)        ← 编码器 only → 理解任务
  │    (Devlin)              双向注意力，遮挡语言模型
  │
  ├──► GPT-1 (2018) ★     ← 解码器 only → 生成任务
  │    (Radford)             因果注意力，自回归语言模型
  │
  ├──► T5 (2019)          ← 编码器-解码器 → 统一框架
  │    (Raffel)
  │
  │   GPT-2 (2019) ★      ← 更大模型，Pre-Norm，Zero-shot
  │   (Radford)
  │
  │   GPT-3 (2020)        ← 1750 亿参数，In-Context Learning
  │   (Brown)
  │
  │   Codex (2021)        ← GPT-3 在代码上微调 → GitHub Copilot
  │   (Chen)
  │
  ▼
2024  CodeGPT ★           ← 本项目
      (本仓库)
      │
      ├── GPT-2 架构       （model.py: Transformer 解码器）
      ├── 因果自注意力      （model.py: CausalSelfAttention）
      ├── Pre-Norm          （model.py: Block 中先 LN 再 Attn）
      ├── 可学习位置编码    （model.py: wpe Embedding）
      ├── 权重绑定          （model.py: wte ↔ lm_head）
      ├── FIM 代码补全 ★    （tokenizer.py: apply_fim_transform）
      ├── 多语言支持        （tokenizer.py: lang tokens）
      ├── 代码专用分词器    （tokenizer.py: CodeTokenizer）
      └── GPT-2 权重迁移    （model.py: from_pretrained + expand_vocab）

★ = 本项目直接继承的关键节点
```

### 参考文献

| 年份 | 论文 | 贡献 |
|------|------|------|
| 1986 | Rumelhart et al., "Learning representations by back-propagating errors" | 反向传播，使 RNN 训练成为可能 |
| 1997 | Hochreiter & Schmidhuber, "Long Short-Term Memory" | LSTM，门控机制解决长程依赖 |
| 2008 | Vincent et al., "Extracting and Composing Robust Features with Denoising Autoencoders" | 去噪自编码器 |
| 2013 | Kingma & Welling, "Auto-Encoding Variational Bayes" | 变分自编码器 VAE |
| 2014 | Sutskever et al., "Sequence to Sequence Learning with Neural Networks" | Seq2Seq |
| 2014 | Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" | 注意力机制 |
| 2014 | Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" | GRU |
| 2017 | Vaswani et al., "Attention Is All You Need" | Transformer |
| 2018 | Radford et al., "Improving Language Understanding by Generative Pre-Training" | GPT-1 |
| 2018 | Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" | BERT |
| 2019 | Radford et al., "Language Models are Unsupervised Multitask Learners" | GPT-2 |
| 2020 | Brown et al., "Language Models are Few-Shot Learners" | GPT-3 |
| 2021 | Chen et al., "Evaluating Large Language Models Trained on Code" | Codex |
| 2022 | Bavarian et al., "Efficient Training of Language Models to Fill in the Middle" | FIM |
