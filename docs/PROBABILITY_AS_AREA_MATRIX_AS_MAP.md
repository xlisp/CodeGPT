# 概率就是面积，矩阵就是映射：大模型最底层的两块拼图

> 一个常见的疑惑是：为什么大模型论文动不动就是 $P(y\mid x)$、$\arg\max W x$、$\mathrm{softmax}(QK^\top/\sqrt d)$？这些符号背后到底在做什么直觉上的事情？
>
> 这篇文档只讲两件事：
>
> - **概率就是面积**——所有 `softmax`、`cross_entropy`、`top-p`、`temperature` 都是在切、推、压一块面积为 1 的"蛋糕"。
> - **矩阵就是映射**——所有 `nn.Linear`、`nn.Embedding`、`Q/K/V` 投影、`lm_head` 都是把一个向量搬运、旋转、拉伸到另一个向量。
>
> 把这两个直觉钉牢，再看 `model.py:177-198` 的 `forward()` 就只剩一句话：**先一连串映射把 token id 变成一个向量，再把那个向量映射成 logits，最后 softmax 把 logits 摆成一块面积，从面积里抽样下一个 token**。

---

## 目录

1. [一句话拼图：forward 就是 "映射 → 面积"](#1-一句话拼图forward-就是-映射--面积)
2. [概率就是面积：从直方图说起](#2-概率就是面积从直方图说起)
3. [softmax：把任意 logits 折成"总面积 = 1"](#3-softmax把任意-logits-折成总面积--1)
4. [cross-entropy：你在正确答案上摆了多少面积](#4-cross-entropy你在正确答案上摆了多少面积)
5. [temperature / top-k / top-p：三种切蛋糕的方法](#5-temperature--top-k--top-p三种切蛋糕的方法)
6. [矩阵就是映射：从 y = Wx 开始](#6-矩阵就是映射从-y--wx-开始)
7. [Embedding 是查表，底层是矩阵乘](#7-embedding-是查表底层是矩阵乘)
8. [Q、K、V：同一个向量被三种映射"分成三份"](#8-qkv同一个向量被三种映射分成三份)
9. [lm_head：从语义空间映回词表空间](#9-lm_head从语义空间映回词表空间)
10. [多层堆叠 = 映射的复合](#10-多层堆叠--映射的复合)
11. [训练时发生了什么：面积形状被往正确答案上拉](#11-训练时发生了什么面积形状被往正确答案上拉)
12. [通俗收束：水池模型 + 折纸模型](#12-通俗收束水池模型--折纸模型)

---

## 1. 一句话拼图：forward 就是 "映射 → 面积"

先把全文要论证的结论摆出来。`model.py:177-198`：

```python
# model.py:177
def forward(self, idx, targets=None):
    ...
    tok_emb = self.transformer.wte(idx)        # 映射 1：id → 向量
    pos_emb = self.transformer.wpe(pos)        # 映射 2：位置 → 向量
    x = self.transformer.drop(tok_emb + pos_emb)
    for block in self.transformer.h:
        x = block(x)                           # 映射 3..14：12 层 Block
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)                   # 映射 15：向量 → 词表
    loss = F.cross_entropy(...)                # 把 logits 当作面积，看正确答案占多少
```

读完整个流程：
- **前 14 步全是矩阵乘**——把 token id 一步步搬运到一个 768 维的语义向量。
- **第 15 步 `lm_head` 是最后一次映射**——把 768 维向量映回 50304 维的词表 logits。
- **最后一步 `cross_entropy` / `softmax` 是面积**——把 50304 个数捏成一块概率面积，看你在正确答案那一格上铺了多少。

后面所有节都是把这十几行展开讲。

---

## 2. 概率就是面积：从直方图说起

### 2.1 离散概率：把面积切成 N 个柱子

掷一枚骰子，6 个面每个 1/6。把它画成直方图：

```
P
│
1/6 ┤ ▮  ▮  ▮  ▮  ▮  ▮
    └─1──2──3──4──5──6  (点数)
```

每根柱子的高度就是概率。**所有柱子加起来的总面积 = 1**——这是概率论里那条最朴素的公理：$\sum_i P(x_i) = 1$。

如果柱子宽度都设为 1（默认），那么"高度"和"面积"是一回事。但一旦你切到连续分布，就必须用面积这个词：

### 2.2 连续概率：高度可以 > 1，但面积必须 = 1

正态分布 $\mathcal{N}(0, 0.01^2)$ 在 0 处的概率密度可能高达 39.9，但它只是"密度"——只有把一段区间下的面积积分出来，才是概率。

```python
import numpy as np
# 一个尖窄的正态分布,中心高度可达 ~40
x = np.linspace(-0.05, 0.05, 1000)
pdf = np.exp(-x**2 / (2*0.01**2)) / (0.01 * np.sqrt(2*np.pi))
print(pdf.max())                    # ~39.89, 高度可以 > 1
print(np.trapz(pdf, x))             # ~1.0,    但面积一定 = 1
```

记住这条铁律：**概率 = 面积，永远 = 1**。所有的"分布在变化"，本质上都是这块面积**在 x 轴上被推来推去、压扁或堆尖**——但总量守恒。

### 2.3 在大模型里，"分布"是 50304 根柱子

`CodeGPTConfig.vocab_size = 50304`（`model.py:112`）。每次预测下一个 token，模型输出的是一个长度为 50304 的概率向量，对应 50304 根柱子，**它们的高度加起来 = 1**。

```python
# model.py:301
probs = F.softmax(logits, dim=-1)   # shape: (B, 50304),每一行是一个直方图
probs.sum(dim=-1)                    # 所有元素 = 1.0
```

所以下一节我们要回答：原始的 `logits`（一组任意实数）**怎么变成一块面积 = 1 的直方图**？

---

## 3. softmax：把任意 logits 折成"总面积 = 1"

### 3.1 核心一行

`F.softmax` 干的事情用 Python 写出来就是：

```python
def softmax(z):
    e = (z - z.max()).exp()          # 1. 取指数,把负数变正数
    return e / e.sum()               # 2. 归一化,让总和 = 1
```

直觉是这样的：

- 任意一组实数 $z_1, z_2, \ldots, z_n$（可能有正有负有几十有上百），不可能直接当概率——概率必须非负且和为 1。
- **第一步用 `exp` 把它们都拉到正数**（指数永远 > 0）。指数还有个绝佳性质：差距被放大——$z_i$ 比 $z_j$ 大 1，$e^{z_i}$ 就比 $e^{z_j}$ 大 e 倍。
- **第二步除以总和**，确保面积 = 1。

### 3.2 几何直觉：把一根折线压成一块面积

想象 logits 是一根任意起伏的折线：

```
logits:    -2  ▁
            5     ████████
            1   ▂▂
            8        ████████████
           -3  ▁
```

`softmax` 干的事是：先把每个柱子换成 $e^{z_i}$（高的更高、低的更低），然后把这一坨缩放到总面积 1：

```
softmax:    .  (几乎 0)
            ▂▂
            ▁
            ███   (大头都吸过来了)
            .
```

**logits 的"形状"决定了面积怎么分配**——大的 logit 把面积吸过来，小的 logit 几乎被压成 0。这就是为什么我们叫这个分布是"由 logits 参数化"的。

### 3.3 模型在做什么

每一次前向，CodeGPT 输出 50304 个 logits。`softmax` 把它们捏成 50304 根柱子，总面积 = 1。**生成时就从这块面积里抽样**：

```python
# model.py:301-302
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)  # 按面积比例抽样
```

`torch.multinomial` 字面意思就是"按照每根柱子的面积比例**扔骰子**"——面积越大的 token 越容易被抽到。模型根本不知道"语法"或"语义"，它只知道**当前上下文下 50304 根柱子的面积分布是什么**。

---

## 4. cross-entropy：你在正确答案上摆了多少面积

### 4.1 一句话总结

训练时的 loss 函数是 `F.cross_entropy`（`model.py:192`）。它干的事就一个：

> **看模型在"正确答案那一格"上铺了多少面积，铺得越少，loss 越大。**

数学写法：$\mathrm{loss} = -\log P(\text{正确 token})$。代码写法更直白：

```python
probs = F.softmax(logits, dim=-1)        # 50304 根柱子
correct_token = targets                   # 正确 token 的 id
prob_of_correct = probs[correct_token]    # 拿出对应那一格的面积
loss = -torch.log(prob_of_correct)        # 越接近 1, log 越接近 0; 越接近 0, log 越大
```

### 4.2 几何直觉

正确答案是 50304 根柱子里的某**一根**。理想情况是模型把所有面积都堆到那一根上（高度 = 1，其余 = 0）。但模型不知道答案，它只能根据上下文做出最合理的分布。

- 如果模型在那一格摆了 0.9 的面积：loss ≈ $-\log 0.9$ ≈ 0.10，几乎没事。
- 如果摆了 0.1：loss ≈ $-\log 0.1$ ≈ 2.30，挨揍。
- 如果摆了 0.0001：loss ≈ 9.21，挨大揍。

训练就是不断让模型**把面积往正确答案那根柱子上推**。所有"模型在学习"的过程，从面积视角看，就是这块直方图的形状一步步被磨成"该尖的地方尖、该平的地方平"的过程。

### 4.3 为什么 ignore_index = -1 出现在 FIM 里

`F.cross_entropy(..., ignore_index=-1)`（`model.py:192`）的意思是：**target 等于 -1 的位置，不计算 loss、不要求模型在那块面积上做任何事**。

`tokenizer.py` 里 `apply_fim_transform` 把 `<|fim_pad|>` 位置的 target 设成 -1，就是告诉模型："这些位置是为了凑齐 block_size 的 padding，**那块面积长什么样我都不管**。" 否则模型会被迫去把面积堆到 padding token 上，把整块概率分布带偏。

---

## 5. temperature / top-k / top-p：三种切蛋糕的方法

推理时所有"调风格"的旋钮，本质上都是在**对那块面积做手脚**。看 `model.py:279-302` 这一段：

### 5.1 temperature：把面积压扁还是堆尖

```python
# model.py:279
logits = logits[:, -1, :] / temperature
```

把 logits 整体除以一个温度 $T$：

- $T > 1$：所有 logit 被压缩 → 高低差变小 → softmax 后**面积更平**（更随机、更有创意、也更可能胡说）。
- $T < 1$：所有 logit 被拉大 → 高低差更悬殊 → 面积**更集中在最大的几根柱子上**（更确定、更保守）。
- $T \to 0$：面积全部堆到最高那根柱子上（贪心解码）。

物理上这个 $T$ 字面就是温度——`softmax` 等价于 Boltzmann 分布，温度高分子乱跑，温度低粒子都坐到能量最低处（参见 [PHYSICS_AND_DEEP_LEARNING.md](PHYSICS_AND_DEEP_LEARNING.md)）。

### 5.2 top-k：只留最高的 k 根柱子

```python
# model.py:287-289
v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
logits[logits < v[:, [-1]]] = -float('Inf')
```

把除了 top-k 之外的所有柱子的 logit 设成 $-\infty$，softmax 后它们的面积变成 0。剩下 k 根柱子重新归一化，总面积 = 1。**直接砍掉长尾**。

### 5.3 top-p（nucleus sampling）：按面积切蛋糕

```python
# model.py:292-299
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
sorted_indices_to_remove = cumulative_probs > top_p
```

更聪明的做法：把柱子按高度从大到小排好，**累计面积从 0 开始加**，加到刚好 $\geq p$（比如 0.9）就停。前面这一截"核心面积"（nucleus）保留，后面的全砍。

为什么这比 top-k 更合理？因为不同上下文下"该保留多少根柱子"是动态的：

- "1 + 1 = ?" 这种问题，可能只有一根柱子（"2"）面积就有 0.99，top-p=0.9 时只保留这 1 根。
- "请写一首关于秋天的诗" 这种开放问题，可能 200 根柱子才凑齐 0.9 的面积，top-p 自动保留 200 根。

**top-k 是固定切刀数，top-p 是固定切面积**。前者僵硬，后者自适应。

---

## 6. 矩阵就是映射：从 y = Wx 开始

讲完面积，换另一半拼图——矩阵。

### 6.1 一行代码就是全部

```python
import torch.nn as nn
fc = nn.Linear(in_features=4, out_features=3, bias=False)
y = fc(x)    # x: (4,) -> y: (3,), 等价于 y = W @ x, 其中 W 是 (3, 4) 的矩阵
```

`nn.Linear` 不是"线性层"这种神秘的东西，它就是 **矩阵乘**。一个 $(3, 4)$ 的矩阵 $W$ 接受一个 4 维向量，吐出一个 3 维向量。

### 6.2 几何直觉：矩阵把空间搬来搬去

二维平面上看最直观：

- 旋转矩阵：$\begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}$ 把平面所有点绕原点旋转 $\theta$ 度。
- 拉伸矩阵：$\begin{bmatrix}2 & 0 \\ 0 & 1\end{bmatrix}$ 把平面横向拉成 2 倍。
- 投影矩阵：$\begin{bmatrix}1 & 0 \\ 0 & 0\end{bmatrix}$ 把平面所有点压到 x 轴上（降维）。

代码上看：

```python
import torch
W_rot = torch.tensor([[0.0, -1.0], [1.0, 0.0]])    # 旋转 90 度
v = torch.tensor([1.0, 0.0])                         # 沿 x 轴方向
W_rot @ v                                            # tensor([0., 1.]) — 变成沿 y 轴
```

**任何矩阵都可以理解成"把输入空间的每一个向量映射到输出空间的某个向量"**——它是一个函数，输入是向量，输出是向量。"映射"就是函数的几何说法。

### 6.3 高维空间的映射也是同一件事

到了 768 维的隐藏空间、50304 维的词表空间，几何画不出来，但直觉一样：

- $W \in \mathbb{R}^{m \times n}$ 是一个**从 n 维空间到 m 维空间的映射**。
- 输入向量 $x \in \mathbb{R}^n$，输出向量 $y = Wx \in \mathbb{R}^m$。
- 矩阵的每一行可以理解成 $m$ 个"探测器"，每根探测器在 n 维空间里有一个偏好方向。
- $y_i$ 就是 $x$ 沿着第 $i$ 个探测器方向的"投影长度"。

这个直觉到了 attention 那里会反复用到。

---

## 7. Embedding 是查表，底层是矩阵乘

### 7.1 wte 是一张大表

```python
# model.py:145
self.transformer.wte = nn.Embedding(config.vocab_size, config.n_embd)
# shape: (50304, 768)
```

`nn.Embedding` 看起来像"查字典"：给一个 token id（整数），返回对应的 768 维向量。但**底层就是矩阵乘**：

- `wte.weight` 是一个 $(50304, 768)$ 的矩阵 $W$。
- `wte(idx)` 等价于"取出 $W$ 的第 `idx` 行"。
- 也等价于"把 idx 表示成一个 50304 维的 one-hot 向量 $e_{idx}$，然后做 $W^\top e_{idx}$"。

```python
# 这两行结果完全一样
v1 = wte.weight[token_id]
v2 = wte.weight.T @ F.one_hot(token_id, num_classes=50304).float()
```

第二种写法浪费——但它揭示了**embedding 就是矩阵乘的特例**：one-hot 当输入时，矩阵乘退化成查表。

### 7.2 这张表是被训练出来的

`wte` 一开始是随机的（`model.py:171-175` 的 `_init_weights`）。每次反向传播都在调整这张表的某些行——具体哪些行被调整？只有这次 batch 里**真正出现过**的那些 token id 对应的行。

训练完之后：
- 含义相近的 token 被推得**几何距离接近**（cos 相似度高）。
- "function"、"def"、"return" 这些代码关键词被聚成一团；"+"、"-"、"*" 是另一团。
- 所谓的"语义空间"就是这张表在训练后的形状。

这是大模型最神奇的事情之一：**离散的 token id 通过一次矩阵乘被映射成连续的语义向量**，从此可以做加减法、做 attention、做线性插值。

---

## 8. Q、K、V：同一个向量被三种映射"分成三份"

### 8.1 一行代码做三件事

```python
# model.py:36
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
# 768 -> 2304, 然后 split 成三份各 768
```

```python
# model.py:53
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```

数学上等价于三个独立矩阵：

$$
Q = W_Q x, \quad K = W_K x, \quad V = W_V x
$$

**同一个输入向量 $x$，被三种不同的映射变成三个角色**：

- **$Q$（Query，问题）**：当前 token 在"问"什么？
- **$K$（Key，钥匙）**：每个 token 自己"是关于什么的"？
- **$V$（Value，内容）**：每个 token 想要被"传递"的实际内容是什么？

### 8.2 注意力 = Q 在 K 群里搜匹配，按面积加权抽 V

```python
# model.py:66-70
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))   # Q·Kᵀ:相似度矩阵
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)                                       # 把每一行变成面积 = 1 的分布
y = att @ v                                                        # 按面积加权求和 V
```

注意这里**两块拼图同时出现**：

1. **Q、K、V 是三次矩阵映射**——把 $x$ 搬到三个不同的子空间。
2. **$Q \cdot K^\top$ 是一组打分，softmax 后变成面积 = 1 的分布**——给每一个 V 应该被取多少作出概率回答。
3. **`att @ v` 是按面积加权求和**——它字面上就是数学期望 $\mathbb{E}_{\text{att}}[V]$。

所以 attention 的口号"Query 找 Key，按相似度加权 Value"翻译成本文的语言是：

> **三次矩阵映射把 $x$ 拆成三个角色，相似度被 softmax 折成一块面积，从这块面积里求 V 的加权平均。**

---

## 9. lm_head：从语义空间映回词表空间

```python
# model.py:151
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
# 768 -> 50304
```

最后一层 `lm_head` 是一个 $(50304, 768)$ 的矩阵——**它和 `wte` 形状互为转置**。事实上 CodeGPT 还做了"权重绑定"（weight tying）：

```python
# model.py:153
self.transformer.wte.weight = self.lm_head.weight
```

这两个矩阵**共享同一份参数**。这意味着：

- **进入网络时**：用这张表把 token id 映射到 768 维语义向量（取一行）。
- **离开网络时**：用同一张表把 768 维语义向量映射回 50304 维 logit 向量（做矩阵乘）。

把它当一对反向操作来理解：embedding 是"打开"——把符号变成几何向量；lm_head 是"关闭"——把几何向量变回符号空间的得分（logit）。最后 softmax 把这些得分捏成面积，抽样得到下一个 token。

---

## 10. 多层堆叠 = 映射的复合

```python
# model.py:186-187
for block in self.transformer.h:
    x = block(x)
```

12 层 Transformer Block 就是把 12 个映射函数**复合**起来：

$$
x_\text{out} = f_{12} \circ f_{11} \circ \cdots \circ f_1(x_\text{in})
$$

每个 $f_i$ 内部又是一连串矩阵乘：`ln_1 → c_attn → c_proj → ln_2 → c_fc → c_proj`。整个 GPT 看作一个超长的函数，参数加起来 124M，本质上就是一个**复合映射**。

而**残差连接**（`model.py:103-104`）的几何意义是：

```python
x = x + self.attn(self.ln_1(x))    # 不是替换 x,是在 x 上加一个修正
x = x + self.mlp(self.ln_2(x))
```

每一层不是"重写"输入向量，而是**给它加上一个微小的位移**。整条 12 层网络可以理解成"把一个原始向量沿着学到的方向，分 24 步（12 attn + 12 mlp）一点点搬到目的地"。从这个视角看，深网络是**离散化的常微分方程**——参见 [PHYSICS_AND_DEEP_LEARNING.md](PHYSICS_AND_DEEP_LEARNING.md) 里把残差解释成 Euler 法的那一段。

---

## 11. 训练时发生了什么：面积形状被往正确答案上拉

把"映射"和"面积"两块拼图结合起来，才能看清训练在做什么。看 `train.py` 主循环里那一行：

```python
logits, loss = model(X, Y)        # 前向:映射链 + 算面积上的 loss
loss.backward()                    # 反向:把 loss 的梯度沿映射链一路推回去
optimizer.step()                   # 更新所有矩阵的参数
```

发生的事情是：

1. **前向**：一连串矩阵乘把 $X$ 搬到 logits，softmax 后得到一块面积；cross-entropy 度量"正确 token 那一格的面积有多小"。
2. **反向**：对每个矩阵元素求导——"如果我把这个矩阵元素调大 0.0001，正确答案那一格的面积会涨多少？"
3. **梯度下降**：每个矩阵元素都按"涨多少"的方向被调整一点点。

**学习的实质**：所有矩阵（`wte`、`W_Q`、`W_K`、`W_V`、`c_proj`、`c_fc`、`lm_head`）的元素都在被反复微调，目的是**让最终那块概率面积，在正确答案的柱子上越铺越高**。

12 万个参数也好、124 万也好、1240 亿也好，干的都是同一件事：调整一连串矩阵映射，让最终那块面积在数据告诉它应该高的地方变高。

---

## 12. 通俗收束：水池模型 + 折纸模型

最后留两个完全脱离公式的画面，回去随时套：

### 12.1 概率 = 一池水

想象有一池总量恒定为 1 的水，可以分到无数个杯子里。**杯子越多越窄就是连续分布；杯子有限就是离散分布**。

- `softmax`：把任意一组数字当作"杯子的开口大小"，水按开口比例自动分配。
- `cross-entropy`：每次有人指着"应该装最多水"的那个杯子，看你实际装了多少。装少了就受罚。
- `temperature`：调节水的"流动性"。$T$ 大水到处流（分布平），$T$ 小水都流到最低洼的杯子（分布尖）。
- `top-p`：从最满的杯子开始倒水盛起来，倒到刚好 90% 总量为止——剩下的一律不喝。

整个训练过程就是**让模型学会怎么把水倒得越来越准**。

### 12.2 矩阵 = 折纸

想象有一张无限大的纸（向量空间）。**矩阵是一种折纸/拉伸/旋转的动作**：你拿起这张纸，按一个矩阵的规则把每一个点搬到新位置。

- `wte`：先把 50304 个离散 token 钉到 768 维纸上的固定坐标——这就是"语义空间"的初始布局。
- 12 个 Block：连续做 24 次小的折纸动作，把这些坐标一步步搬到"语义清晰"的最终位置。
- `lm_head`：再把 768 维的纸投影回 50304 维的"得分纸"，每个位置的得分代表那个 token 当前的优先级。
- `softmax`：最后把这张得分纸折叠归一化成一块面积 = 1 的分布。

> **训练就是反复调整每一次折纸的角度，让最后那块面积总能在正确答案的位置上隆起一个尖。**

---

## 总结

回到开头那张拼图：

| 拼图 | 关键操作 | 在 `model.py` 哪里看 |
|------|----------|----------------------|
| **映射** | `nn.Embedding`（id → 向量） | `model.py:145` `wte` |
| **映射** | `nn.Linear`（Q/K/V/MLP 投影） | `model.py:36-37`, `model.py:80-82` |
| **映射** | 多层 Block 复合 | `model.py:186-187` |
| **映射** | `lm_head`（向量 → 词表 logit） | `model.py:151` |
| **面积** | softmax（logits → 概率分布） | `model.py:68`, `model.py:301` |
| **面积** | cross-entropy（正确 token 的面积） | `model.py:192` |
| **面积** | temperature / top-k / top-p | `model.py:279-299` |
| **面积** | multinomial 抽样 | `model.py:302` |

下次再看到 $\mathrm{softmax}(QK^\top/\sqrt d)V$ 之类的式子，把它读成：

> "把 $x$ 做三次映射得到 Q、K、V；Q·Kᵀ 算一组打分；softmax 把打分捏成面积；按面积比例从 V 里抽加权平均。"

只要这两个直觉牢靠了——**概率是面积，矩阵是映射**——剩下的 Transformer、GPT、attention、生成、训练，都只是这两件事在不同尺度上的反复拼装。
