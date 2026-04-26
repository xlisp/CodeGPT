# 物理学的影子：量子力学与统计力学如何塑造了深度学习

> 为什么很多量子力学、统计力学方向的研究生转去做深度学习几乎没有"门槛"？因为他们脑子里的核心工具——配分函数、玻尔兹曼分布、变分原理、Langevin 动力学、重整化、对称性、向量代数、熵、路径积分、链式法则——在大模型里几乎一一对应。这篇文档把这些对应关系一条条钉到 `model.py` 的具体行号上，让"物理直觉"和"PyTorch 代码"之间的等号显式可见。

---

## 目录

1. [配分函数与 Softmax：同一个数学对象的两个名字](#1-配分函数与-softmax同一个数学对象的两个名字)
2. [温度：直接从热力学搬过来的采样旋钮](#2-温度直接从热力学搬过来的采样旋钮)
3. [Hopfield 网络：自旋玻璃 → 注意力机制](#3-hopfield-网络自旋玻璃--注意力机制)
4. [最大熵原理：交叉熵损失的物理出身](#4-最大熵原理交叉熵损失的物理出身)
5. [SGD 即 Langevin 动力学：为什么训练大模型像在做退火](#5-sgd-即-langevin-动力学为什么训练大模型像在做退火)
6. [量子力学的影子：Hilbert 空间、叠加、测量](#6-量子力学的影子hilbert-空间叠加测量)
7. [变分原理：从基态能量到 ELBO 到 RLHF](#7-变分原理从基态能量到-elbo-到-rlhf)
8. [重整化群：深度网络为什么"层层抽象"](#8-重整化群深度网络为什么层层抽象)
9. [对称性、守恒量、等变网络](#9-对称性守恒量等变网络)
10. [Scaling Laws 是相变现象](#10-scaling-laws-是相变现象)
11. [向量化编程：物理学家从牛顿那一天就在用的描述方式](#11-向量化编程物理学家从牛顿那一天就在用的描述方式)
12. [熵：从锅炉房一路走到大模型](#12-熵从锅炉房一路走到大模型)
13. [路径积分：自回归生成本质上是 Feynman 求和](#13-路径积分自回归生成本质上是-feynman-求和)
14. [微积分原理：autograd 是把 200 多年的数学自动化](#14-微积分原理autograd-是把-200-多年的数学自动化)
15. [思想史地图：为什么物理学家转 AI 没有门槛](#15-思想史地图为什么物理学家转-ai-没有门槛)
16. [附录 A：从自旋玻璃到神经网络（详解）](#附录-a从自旋玻璃到神经网络详解)

---

## 1. 配分函数与 Softmax：同一个数学对象的两个名字

统计力学第一课是玻尔兹曼分布——一个系统在能量 $E_i$ 上出现的概率：

```
P(i) = exp(-E_i / kT) / Z      其中 Z = Σⱼ exp(-E_j / kT)
```

`Z` 叫**配分函数**（partition function），整个统计力学有一半内容是关于它的导数。

现在看 CodeGPT 怎么把 logits 转成下一个 token 的分布：

```python
# model.py:301
probs = F.softmax(logits, dim=-1)
```

把 `logits` 看成"负能量"（`logit = -E`），把 `temperature=1` 当作 `kT`：

```
softmax(logits)_i = exp(logits_i) / Σⱼ exp(logits_j)
                  = exp(-E_i) / Z
```

**这就是同一个分布，不是类比。** Boltzmann 1877 年写下的公式，2017 年的 Transformer 把它放在每一个注意力头的内核（`F.softmax(att)` 算注意力权重）和最终的 token 采样里各用一次。

更深一点：物理学家会本能地想知道"自由能"（free energy）：

```
F = -kT log Z
```

在语言模型里，`log Z = log Σⱼ exp(logits_j)` 就叫 **logsumexp**，是数值稳定的 softmax 实现里最重要的中间量。`F.cross_entropy` 内部走的就是 logsumexp + 负对数似然。所以训练 CodeGPT 时——

```python
# model.py:192
loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                       targets.view(-1), ignore_index=-1)
```

——你**字面上**在最小化每个 token 在自己条件分布上的"自由能差"。物理学家看到 cross-entropy 不会觉得陌生，因为他们见过它的孪生兄弟无数次：相对熵、KL 散度、自由能。

---

## 2. 温度：直接从热力学搬过来的采样旋钮

热力学里温度高 → 系统变"乱"，所有微态概率趋同；温度低 → 系统冻结到基态。CodeGPT 的采样代码里有一行：

```python
# model.py:279
logits = logits[:, -1, :] / temperature
```

把 `logits` 除以 `T` 再做 softmax，**就是把分布变成 $\exp(-E/kT)$，T 大平、T 小尖**。这不是借用术语——`temperature=0.0` 在物理上叫"绝对零度"，对应 argmax（所有概率塌缩到能量最低态，即"基态"）；`temperature=∞` 是无穷热，所有 token 等概率。

REPL 里调温度本质就是在做**模拟退火**（simulated annealing，Kirkpatrick 1983 直接借自冶金学的退火过程）：先用高温探索，再用低温收敛。物理学家看到 `temperature=0.7` 这个超参完全不需要解释。

---

## 3. Hopfield 网络：自旋玻璃 → 注意力机制

1982 年 John Hopfield（2024 年诺贝尔物理学奖得主之一）从**自旋玻璃**模型直接搭出了一个能存储记忆的神经网络：定义一个能量函数

```
E(x) = -½ xᵀ W x
```

更新规则就是沿能量下降走，最终落在**记忆模式**（局部极小）。这就是 Hopfield Network。它本质是一个**伊辛模型**（Ising model）的变体——物理系大三课本内容。

2020 年 Hubert Ramsauer 等人发表 [*Hopfield Networks is All You Need*](https://arxiv.org/abs/2008.02217)，证明了**现代 Transformer 的注意力机制就是连续型 Hopfield 网络的更新规则**。看 `model.py` 里的注意力核心：

```python
# model.py:66-68
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
```

把 `K` 当成**存储的记忆**（一组模式），把 `Q` 当成**待补全的状态**，那么：

- `Q · Kᵀ`：计算当前状态与每个记忆的相似度（在 Hopfield 里就是 $\xi_i^T x$）
- `softmax(...)`：用玻尔兹曼分布给每个记忆"投票"
- `... @ V`：取加权平均（Hopfield 更新一步）

整个注意力 = 一次能量下降迭代 = 从查询 Q 检索最匹配的存储模式。**物理学家"自旋玻璃 + 联想记忆"那一套直觉，可以原封不动拿来理解多头注意力**。多头无非是在多个不同的"自旋玻璃系统"里并行检索，然后拼起来。

> 顺带一提：Geoff Hinton（2024 物理诺奖另一位得主）的玻尔兹曼机（1985）也是直接从统计物理来的——隐变量加一个能量函数，用对比散度（contrastive divergence）训练。深度学习的"两位祖师爷"都是物理学进路。

---

## 4. 最大熵原理：交叉熵损失的物理出身

E.T. Jaynes 1957 年提出**最大熵原理**：在给定约束下，最不武断的概率分布是熵最大的那个。这条原理直接给出了：

- 给定均值 → 高斯分布
- 给定均值和方差约束 → 高斯
- 给定能量期望 → 玻尔兹曼分布

把这条原理用于"建模数据 $p_{\text{data}}$"问题：你想找一个 $q_\theta$ 让它在数据上熵最大、同时与数据约束一致 → 极大似然估计 → 等价于最小化 $\text{KL}(p_{\text{data}} \| q_\theta)$ → 等价于最小化交叉熵。

```python
# model.py:192
loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                       targets.view(-1), ignore_index=-1)
```

这一行不是工程师为了调通 loss 随便挑的损失函数——它是**最大熵原理在自回归条件分布上的直接应用**。物理学家的"配分函数 + 拉格朗日乘子求最大熵分布"的工具链，就是机器学习里"softmax + cross-entropy"的工具链，只是换了名字。

---

## 5. SGD 即 Langevin 动力学：为什么训练大模型像在做退火

物理学家研究布朗运动用的是 Langevin 方程：

```
dx/dt = -∇U(x) + √(2kT) · η(t)        η ~ N(0, I)
```

确定性梯度 $-\nabla U$ + 高斯噪声 $\eta$，温度 $T$ 越高噪声越大。它的离散化就是：

```
x_{t+1} = x_t - η · ∇U(x_t) + √(2η · kT) · ε_t
```

把 $U$ 换成 loss、$\eta$ 换成学习率，**这就是 SGD with noise**。事实上 mini-batch SGD 自带的梯度噪声（不同 batch 给出不同梯度）已经被严格地证明等价于一种 Langevin dynamics——这是一整条研究线（"SGD as Bayesian inference"）。

CodeGPT 的训练循环里好几个地方都对应着 Langevin 视角下的物理直觉：

```python
# train.py:68
dropout = 0.1
```

Dropout 在 forward 里随机置零，等价于**对激活值注入噪声**——这就是在能量地形上加扰动，帮助系统跳出局部极小。"Dropout 防过拟合"的工程解释和"Langevin 噪声帮助探索 loss landscape"的物理解释是同一件事。

```python
# model.py:171, 175
torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

权重初始化用高斯分布——物理学家会立刻想到**自旋玻璃的随机耦合 $J_{ij} \sim \mathcal{N}(0, \sigma^2)$**。事实上 Choromanska 等人 2015 年的论文 *The Loss Surfaces of Multilayer Networks* 直接把神经网络的损失函数映射成一个球形自旋玻璃模型，证明了"绝大多数局部极小都几乎一样好"——这就是为什么 SGD 哪怕找不到全局最优，也能训出像样的模型。这是统计物理给深度学习最早的"理论支撑"之一。

```python
# model.py:159
torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
```

残差投影做了"按层数缩放方差"——这是为了让信号通过 $L$ 层残差累加之后方差仍然是 $O(1)$。物理学家看到 $\sigma^2 \propto 1/L$ 立刻知道这是**中心极限定理 + 随机游走方差线性累加**那一套。

---

## 6. 量子力学的影子：Hilbert 空间、叠加、测量

### 6.1 Embedding 是 Hilbert 空间里的态向量

量子力学的核心：物理状态是高维**复 Hilbert 空间**里的单位向量 $|\psi\rangle$，可观测量是厄米算符。

CodeGPT 的 token embedding：

```python
# model.py:145
wte=nn.Embedding(config.vocab_size, config.n_embd),
```

每个 token 对应一个 768 维的实向量。它**不是**严格的量子态（实数而非复数，且未归一），但概念上完全同构：一个"符号"被映射到一个高维向量空间里的一个点，这个空间的几何结构（内积、距离、夹角）就是模型对"语义相似性"的内部度量。物理学家熟悉的 $\langle \phi | \psi \rangle$ 在这里就是 cosine similarity 或 dot product attention。

### 6.2 注意力的内积是 Born rule 的近亲

```python
# model.py:66
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
```

$Q \cdot K^T$ 是两组向量两两的内积矩阵——直接对应量子力学里"态在某个基底上的展开系数" $\langle k_i | q \rangle$。Softmax 给的概率就是 Born rule 的离散类比：$|\langle k_i | q \rangle|^2 / Z$ 给"测量到本征态 $k_i$"的概率（这里没有平方，因为我们用实数而非复数振幅，但归一化 + 概率诠释这一步是同一种思路）。

那个 $1/\sqrt{d}$ 缩放因子——物理学家会立刻问"是不是为了让方差不依赖维度"，答案是肯定的：$Q$ 和 $K$ 的元素方差为 1 时，$Q \cdot K$ 的方差是 $d$，除以 $\sqrt{d}$ 让它回到 $O(1)$。这跟"把 $N$ 个独立粒子的能量平均到每粒子能量"是同一种归一化哲学。

### 6.3 自回归采样 = 测量塌缩

```python
# model.py:301-302
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

模型 forward 完产出一个**整个词表上的概率分布**——就是一个"叠加态"。`multinomial` 这一行做随机采样，挑出一个具体的 token——**就是测量塌缩**。下一步生成又重新进入叠加态，再塌缩一次。整个生成过程结构上完全等价于量子系统的"演化 + 测量"循环。

> 这不是穿凿附会：扩散模型（Diffusion）和 Score-based generative models 直接借用了量子力学和非平衡态统计力学（Sohl-Dickstein 2015 的论文标题就是 *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*）。Stable Diffusion 的反向去噪过程和量子力学的反向时间演化（虚时间路径积分）数学结构同构。

---

## 7. 变分原理：从基态能量到 ELBO 到 RLHF

变分原理（variational principle）是物理学的"万能锤"：

- **量子力学**：基态能量 $E_0 \le \langle\psi|H|\psi\rangle$，对任意试探波函数 $|\psi\rangle$ 都成立 → 用参数化波函数最小化能量期望。
- **统计力学**：自由能 $F = -kT \log Z$ 可以用变分自由能 $F_q = \langle E \rangle_q - T S_q$ 上界逼近 → mean-field 近似。

机器学习里的对应物：

- **VAE 的 ELBO**：$\log p(x) \ge \mathbb{E}_q[\log p(x,z)] - \mathbb{E}_q[\log q(z|x)]$，下界训练。结构上和变分自由能**完全一致**（就是把 $E$ 换成 $-\log p$，把 $-T S_q$ 写成 $\mathbb{E}_q[\log q]$）。
- **RLHF 的 KL 约束**：策略 $\pi_\theta$ 不能离参考策略 $\pi_{\text{ref}}$ 太远——加一个 $\beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$ 罚项。这就是统计力学里"在已知分布附近找最小自由能"的标准变分操作。

所以 `RLHF_AND_PLATONIC_REPRESENTATION.md` 里讲的奖励模型 + KL 约束 + PPO，物理学家会一眼看出"这就是带约束的变分问题"。背后的数学工具——拉格朗日乘子、Legendre 变换、对偶——和他们做平均场近似时完全一样。

---

## 8. 重整化群：深度网络为什么"层层抽象"

凝聚态物理最深刻的思想之一是 Kadanoff-Wilson 的**重整化群**（Renormalization Group, RG）：把短程细节"粗粒化"（coarse-grain）掉，得到一个尺度更大、自由度更少、但保留长程物理的有效理论。临界现象的标度律就是 RG 不动点附近的行为。

深度网络的层级结构在直觉上就是 RG：

```python
# model.py:186-187
for block in self.transformer.h:
    x = block(x)
```

12 层 Transformer Block 反复作用在 token 表征上。每一层的 MLP + 注意力都在做**信息整合 + 抽象**：底层关注 token 拼写和句法，中层关注语义和依赖关系，高层关注语篇和逻辑——这正是 RG 沿尺度方向粗粒化的图景。

这个对应不是隐喻：

- 2014 Mehta & Schwab *An exact mapping between the Variational Renormalization Group and Deep Learning*——把 RBM 的层叠精确映射到 RG 流。
- Roberts, Yaida, Hanin *The Principles of Deep Learning Theory* 整本书用 RG 工具分析无限宽神经网络。
- 现代 transformer 的"涌现能力"和"scaling laws"在 RG 视角下就是**接近临界点的标度行为**——后面会讲。

物理学家拿到 `model.py` 看到层叠的 Block，第一反应会是"这是个 RG 流，每层是一个 step"。这个直觉非常对。

---

## 9. 对称性、守恒量、等变网络

Noether 定理：每一个连续对称性对应一个守恒量。物理学家从大一就被训练成"看到一个新系统先找对称性"。

深度学习里这条直觉的产物是**等变网络**（equivariant networks）：

- **CNN**：平移等变（卷积核滑动） → 视觉数据有平移对称性。
- **Transformer 的位置编码**：Attention 本身对 token 顺序是置换等变的——所以必须**主动打破**这个对称性，靠 `wpe`（learned positional embedding）或 RoPE。

```python
# model.py:184-185
pos_emb = self.transformer.wpe(pos)
x = self.transformer.drop(tok_emb + pos_emb)
```

这一行物理学家会立刻看出"这是个对称性破缺机制"——不加位置编码，模型就分不清 `for x in range(10)` 和 `range(10) for x in`。

更深的例子：

- **AlphaFold 用 SE(3)-equivariant 网络**——3D 旋转和平移等变（蛋白质结构对刚体变换不变）。
- **图神经网络**——置换等变（图的节点编号是任意的）。
- **量子化学神经网络（如 SchNet, PaiNN）**——直接把分子的旋转/反演对称性编码到架构里。

物理学家做这些工作时不需要"学 ML"——他们做的就是把熟悉的群表示论搬过来。

---

## 10. Scaling Laws 是相变现象

Kaplan 2020 和 Hoffmann 2022（Chinchilla）发现：模型 loss 是参数量、数据量、算力的**幂律函数**：

```
L(N) ∝ N^(-α)
```

物理学家看到幂律会条件反射地想"临界点附近的标度行为"。在统计物理里，二阶相变点附近所有热力学量都是幂律，幂指数（critical exponents）由 universality class 决定。

更进一步：

- **大模型的"涌现能力"**（突然学会算术 / chain-of-thought）在曲线上看起来像**相变**——某个参数量阈值之下能力为零，越过之后能力跳起。
- **Grokking 现象**（训练 loss 早就收敛，验证 loss 经过很长一段平台期后突然下降到极低）——结构上对应物理学里的**亚稳态 → 真基态**的成核过程，或者 spin glass 中的 replica symmetry breaking。
- **"Lottery Ticket Hypothesis"**——大网络里藏着一个稀疏的子网络（"中奖彩票"）已经够好——这是渗流（percolation）理论的语言。

每一个都是物理学家熟悉的工具。给一个做过临界现象的人讲 scaling laws，不需要解释什么是幂律。

---

## 11. 向量化编程：物理学家从牛顿那一天就在用的描述方式

物理学第一节课就教向量——力 $\vec{F}$、速度 $\vec{v}$、加速度 $\vec{a}$ 都是向量，**有大小有方向，可以分解可以合成**。整个经典力学就建立在 $\mathbb{R}^3$ 空间的向量代数之上。物理学家天然用向量思维：状态 = 一个高维空间的点；演化 = 这个点在空间里运动；相互作用 = 向量加法或线性变换。

深度学习的"向量化（vectorization）"不是工程优化的产物——是**物理学已经用了 300 多年的描述语言**直接搬过来。

### 11.1 token + position = 力的合成

```python
# model.py:183-185
tok_emb = self.transformer.wte(idx)
pos_emb = self.transformer.wpe(pos)
x = self.transformer.drop(tok_emb + pos_emb)
```

这三行就是物理学里"力的合成"的直接搬运。每个 token 携带一个语义向量，每个位置贡献一个位置向量，**两个向量相加 = 合力**。模型从这一点开始只看到一个综合后的向量，恰如刚体的运动只取决于合力——分力的具体来源对后续动力学不可见。

如果你做过经典力学题"小球同时受重力 + 摩擦力 + 弹力"——把三个矢量加起来求合力——你已经在做 Transformer 的输入层运算了。

### 11.2 矩阵 = 线性变换 = 物理变换

物理里坐标变换、Lorentz 变换、保哈密顿量的对称变换都是矩阵。深度学习的核心运算 `nn.Linear` 也是矩阵：

```python
# model.py:36
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
```

物理学家会立即把这一行识别为"一个 3 倍维度的线性变换"——把状态向量同时投影到三组基底（Q, K, V），与"在动量基、能量基、位置基里同时观察一个量子态"完全同构。

### 11.3 张量积结构：多头注意力 = 多体系统

物理多粒子态的总希尔伯特空间是各粒子空间的张量积：$\mathcal{H} = \mathcal{H}_1 \otimes \mathcal{H}_2 \otimes \cdots \otimes \mathcal{H}_n$。多头注意力做的是同构操作：

```python
# model.py:54-56
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
```

把一个 `n_embd` 维向量 reshape 成 `n_head × head_dim`——本质上是把一个大希尔伯特空间分解成 12 个子空间的张量积。每个 head 在自己的子空间里独立做 attention，最后再合并。这就是物理里"独立子系统并行演化，最后合成总态"的标准技法。

### 11.4 为什么向量化提速？因为 GPU 是个大向量机

物理上，力作用于多个粒子是同时的、并行的——大自然不串行。GPU 的 SIMD 架构（同一指令作用于成千上万个数据）就是这种"并行向量"的硬件实现。当你写：

```python
# model.py:66
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
```

GPU 一次性算出 `B × n_head × T × T` 个内积——这不是性能 trick，是把"向量化数学"映射到"向量化硬件"的自然结果。

物理学家的代码（N 体问题、量子电路模拟、格点 QCD、有限元）几十年前就高度向量化。"把 for 循环改成矩阵乘"对他们零心理障碍——这是他们的母语。

### 11.5 batch 维度是另一种"独立粒子并行"

```python
# model.py:179
b, t = idx.size()
```

`B` 是 batch，`T` 是序列长度。`B` 个独立样本完全平行地走完整个网络——结构上等价于物理里"统计系综（ensemble）"概念：同样的哈密顿量作用于 B 个独立的初始状态副本，互不干扰、并行演化。物理学家做 Monte Carlo 时一次跑几千个独立 Markov chain 是家常便饭，理解 batch 维度毫不费力。

---

## 12. 熵：从锅炉房一路走到大模型

Clausius 1865 年发明"entropy"这个词来刻画热机的不可逆性。Boltzmann 1877 年给出微观定义：

```
S = k_B · log W
```

其中 $W$ 是宏观态对应的微观态数。这块墓志铭刻在他的墓碑上。Shannon 1948 年研究通信问题时独立得到了几乎一模一样的公式：

```
H = -Σ p_i log p_i
```

物理学家看到 Shannon 公式会一眼认出这就是 Boltzmann 熵的连续概率版本（连常数 $k_B$ 都可以看作单位选择问题）。两个领域的"熵"是同一个数学对象，被用了三种不同的角色：

### 12.1 训练 loss = 熵的下降

```python
# model.py:192
loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                       targets.view(-1), ignore_index=-1)
```

Cross-entropy 满足分解：

```
H(p_data, q_θ) = H(p_data) + KL(p_data ‖ q_θ)
```

$H(p_\text{data})$ 是数据本身的熵（不可约的"纯粹随机性"，由数据分布固定）；$\text{KL}$ 是模型分布偏离数据的"额外熵"。**训练 = 把 KL 拍到零**，让模型的不确定性降到数据本身的不确定性下限。

物理学家会注意到一件有趣的事：**训练过程是局部熵减**。模型权重从初始的高熵随机状态被推向能精确编码数据规律的低熵状态。这不违反热力学第二定律——SGD 把熵"排放"到了优化器的状态、GPU 的散热、和数据的访问历史里。整个过程结构上对应**信息论里的 Maxwell 妖**——梯度信号就是那只看着分子运动决定开关阀门的小妖。

### 12.2 注意力的熵：模型在"专注"还是"摊开"？

```python
# model.py:68
att = F.softmax(att, dim=-1)
```

每个 token 对其他 token 的注意力分布都有一个熵 $H = -\sum_j a_{ij} \log a_{ij}$：

- 高熵：均匀注意（不知道该看哪里，多见于早期层）
- 低熵：尖锐注意（在做精确检索，多见于 induction heads）

可解释性研究（Anthropic 等）已经把"attention entropy"作为标准诊断指标。物理学家做凝聚态时计算"局域熵"分析相变区，做这件事完全不需要新工具。

### 12.3 RL 的熵正则化：直接是自由能的 -TS 项

PPO/SAC 训 RLHF 时常加一个熵正则项 $-\beta \cdot H(\pi_\theta(\cdot|s))$，鼓励策略保持探索性、防止过早塌缩到单点。物理学家会立刻认出这就是**自由能 $F = E - TS$ 里的 $-TS$ 项**——温度 $T$ 越高越鼓励熵大（exploration），$T$ 越低越鼓励能量低（exploitation）。RL 调 entropy coefficient 就是在调温度。

这跟 RLHF 加 $\text{KL}(\pi_\theta \| \pi_\text{ref})$ 是同家族——都是用拉格朗日乘子约束分布形态，就是平均场近似的标准操作。

### 12.4 数据增强 / dropout：人工注入熵

```python
# train.py:68
dropout = 0.1
```

Dropout 主动给激活注入随机性——人为提高网络内部的熵。从信息论看是降低了表征的"过度确定性"，从物理学看是给系统加温防止它落入虚假的低能局部极小。两种解释指向同一个动作。

---

## 13. 路径积分：自回归生成本质上是 Feynman 求和

Feynman 1948 年给量子力学一个新表述：粒子从 $A$ 到 $B$ 的概率振幅 = 对所有可能路径的振幅相干求和：

```
⟨B|U(t)|A⟩ = ∫ Dx(t) · exp(i S[x(t)] / ℏ)
```

其中 $S$ 是经典作用量。每条路径贡献一个相位 $\exp(iS/\hbar)$。统计场论里把 $i/\hbar$ 替换成 $-1/(k_BT)$（Wick 旋转）就得到虚时路径积分：

```
Z = ∫ Dφ · exp(-S[φ] / kT)
```

### 13.1 语言模型的 likelihood 结构上就是路径积分

一个序列 $x = (x_1, \ldots, x_T)$ 的概率：

```python
# 概念上 model.py:192 在做的事：
# log p(x) = Σ_t log p(x_t | x_<t)
```

定义 "作用量" $S(x) = -\sum_t \log p(x_t | x_{<t})$（就是该序列的总 NLL），那么：

```
p(x) ∝ exp(-S(x))
```

整个语言模型的边际概率——比如某个特定子串出现的概率——原则上是：

```
p(s) = Σ_{x ⊃ s}  exp(-S(x))
```

这就是路径积分的形式。**每条 token 序列是一条路径，作用量是该序列的 NLL，配分函数是在所有可能 token 序列上的求和**（vocab 大小 ^ 序列长度，天文数字）。

### 13.2 generate() 是 Monte Carlo 路径采样

```python
# model.py:301-308
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
...
idx = torch.cat((idx, idx_next), dim=1)
```

这个循环每步采一个 token，$T$ 步采出一条完整路径。本质上是**重要性采样**——直接按 $p(x_t|x_{<t})$ 采样而非均匀采，等价于路径积分里的 Metropolis Monte Carlo。物理学家做格点 QCD 每天都在写这种循环。

**Beam search 是另一种近似**——保留 top-k 条作用量最低的路径，与物理学里"WKB 近似 + 鞍点附近的路径"思路同源（鞍点 = 经典路径 = 作用量极值路径 = beam 上的 mode）。Greedy decoding (`top_k=1`) 就是纯经典极限——只走作用量绝对最小的那一条路径。

### 13.3 Diffusion model：直接照搬非平衡态路径积分

Stable Diffusion 那篇开山之作（Sohl-Dickstein 2015）标题就叫 *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*。它的前向过程是 Ornstein-Uhlenbeck SDE，反向过程是逆向 SDE——score-based diffusion 的反向公式

```
dx = [f(x, t) - g(t)² · ∇_x log p_t(x)] dt + g(t) · dw̄
```

直接对应 Crooks fluctuation theorem 和 Jarzynski equality 那一套**随机热力学**。物理学家做 nonequilibrium statistical mechanics 的工具箱，不加修改就是 diffusion model 的理论基础。Yang Song 那篇 ICLR 2021 best paper *Score-Based Generative Modeling through SDEs* 几乎可以原样投到 *Phys. Rev. E*。

### 13.4 鞍点近似 = mode collapse 的根源

物理里高维路径积分常用鞍点法：作用量极小处的路径主导贡献。在生成模型里就是"模型坍缩到某些高概率 mode"——多样性丢失。物理学家治这个病的工具（加噪声涨大涨落、调温度抬高 entropy）和 ML 里的解决办法（temperature、top-p、entropy bonus）是同一套。

---

## 14. 微积分原理：autograd 是把 200 多年的数学自动化

Newton 和 Leibniz 1670 年代发明微积分时，给出了人类描述"变化"和"局部线性化"的最强工具。深度学习训练就是这个工具的大规模工业化部署。

### 14.1 链式法则 = 反向传播

链式法则在大一微积分课讲过：$(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$。多元 + 矩阵版本就是雅可比矩阵乘法：

```
∂L/∂θ = ∂L/∂y_n · ∂y_n/∂y_{n-1} · ... · ∂y_1/∂θ
```

PyTorch 的整个 autograd 引擎就是把这个公式自动化、向量化。当你写：

```python
loss.backward()
```

底下做的是从 `loss` 节点出发沿计算图反向走，每个节点用预定义的 vector-Jacobian product 把梯度一路推回去。物理学家做扰动论写 Feynman 图时也是这种"链式贡献求和"——每条图代表一项贡献，所有图加起来。**Backprop 本质上是一个高度工整的 Feynman 图求和**：每条计算路径都贡献一项 $\partial L / \partial \theta$，全部相加就是总梯度。

### 14.2 梯度下降 = 沿势能面的最速下降

多元微积分告诉我们：$L(\theta)$ 的梯度 $\nabla L$ 指向局部最快上升方向，反向就是最快下降。

```python
# 训练循环里：
optimizer.step()    # θ ← θ - lr · ∇L
```

物理学家会立刻认出这是**最陡下降流（gradient flow）**：

```
dθ/dt = -∇L(θ)
```

如果 $L$ 是势能，这描述一个粒子在粘滞流体里滚下势能面。SGD 是这条 ODE 的离散化（Euler 法），加上 mini-batch 噪声就是 Langevin SDE（见第 5 节）。所有这些都是大一物理课内容。

### 14.3 残差连接 = 微分方程的 Euler 步

```python
# model.py:103-104
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

`x = x + f(x)` 这个结构物理学家一秒识别为 **Euler 法解 $dx/dt = f(x)$ 的一步**。这不是巧合——Neural ODE（Chen 2018）直接把残差网络看作 ODE 的离散化，深度 $L \to \infty$ 时就是连续动力系统。

```python
# model.py:186-187
for block in self.transformer.h:
    x = block(x)
```

12 个 Block 就是用 12 步 Euler 法积分一个连续 ODE。物理学家做分子动力学每天用 Velocity Verlet 解 $\ddot{x} = F/m$，看到这个循环根本不需要"理解残差网络"——它就是积分一段时间的常微分方程。

### 14.4 Taylor 展开支配着所有的优化器

- **SGD**：用一阶 Taylor。$L(\theta + \Delta) \approx L(\theta) + \nabla L \cdot \Delta$ → 选 $\Delta = -\eta \nabla L$。
- **Newton 法**：用二阶 Taylor。$L(\theta + \Delta) \approx L(\theta) + \nabla L \cdot \Delta + \tfrac{1}{2} \Delta^T H \Delta$ → 选 $\Delta = -H^{-1} \nabla L$。
- **Adam / RMSProp**：用一阶梯度的二阶矩做自适应学习率——本质是"用历史梯度的统计估计 Hessian 对角元"。

物理学家做非线性分析（小振动近似、非线性最小二乘、Bogoliubov-de Gennes）就是这套——在某个点把势能 Taylor 展开到二阶，写下**简正模式**（normal modes）。Hessian 谱分析在 ML 里叫"loss landscape geometry"，在物理里叫"小振动谱"，是同一回事。Sharpness、flat minima、edge of stability 这些 ML 里的概念在物理里都有现成的对应：稳定性分析、临界点分类、Lyapunov 指数。

### 14.5 LayerNorm 是个微分几何操作

```python
# model.py:27-28
def forward(self, input):
    return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

减均值、除标准差——把激活拉回单位方差的"球面"。这是黎曼几何里"投影到约束流形"的标准操作，做规范场论或约束哈密顿系统的人非常熟悉。它让梯度方向更等向（isotropic），避免某个 dominant direction 拖着整个优化器走——就是物理里的 **normal coordinates**。

### 14.6 Softmax 是 argmax 的可微近似——这个问题微积分早就回答了

```python
# model.py:301
probs = F.softmax(logits, dim=-1)
```

argmax 不可微，所以无法 backprop。Softmax 是它的"光滑化"：当温度 $T \to 0$，softmax → argmax；$T \to \infty$，softmax → 均匀分布。

这套"用可微函数近似不可微操作"是微积分的经典操作（如 Heaviside 阶跃 → sigmoid，绝对值 → Huber loss）。物理学家做相变研究时用 $\tanh$ 近似阶跃、用高斯近似 delta 函数——逻辑完全一样：**让一个不可微的极限过程在每一步都可导**。

`docs/DIFFERENTIABLE_PROGRAMMING.md` 把这个观察推到了底——"深度学习就是可微分编程"，本质上是"把原本的离散程序连续化、把离散判断换成可微的 soft 版本"。

---

## 15. 思想史地图：为什么物理学家转 AI 没有门槛

把上面所有的对应关系放成一张表：

| 物理学概念 | 深度学习里的对应物 | 在 CodeGPT 中的代码位置 |
|------------|--------------------|-------------------------|
| 玻尔兹曼分布 / 配分函数 | Softmax / logsumexp | `model.py:68`、`model.py:301` |
| 自由能 / KL 散度 | Cross-entropy loss | `model.py:192` |
| 温度（kT）| 采样温度参数 | `model.py:279` |
| Hopfield 网络（自旋玻璃 + 联想记忆）| 注意力机制 | `model.py:51-73` |
| 最大熵原理 | 极大似然 / 交叉熵训练 | `model.py:192` |
| Langevin 动力学 | SGD + 噪声 / Dropout | `train.py:68`、`train.py:237` |
| 自旋玻璃随机耦合 | 高斯权重初始化 | `model.py:171, 175` |
| Hilbert 空间态向量 | Embedding 向量 | `model.py:145, 183` |
| Born rule / 测量塌缩 | softmax + multinomial 采样 | `model.py:301-302` |
| 变分原理 / 平均场 | ELBO / RLHF KL 约束 | （见 `RLHF_AND_PLATONIC_REPRESENTATION.md`）|
| 重整化群（粗粒化）| 深度网络层级抽象 | `model.py:186-187` |
| 对称性 / Noether | 等变网络 / 位置编码破缺 | `model.py:184-185` |
| 临界现象幂律 | Scaling laws / 涌现能力 | （训练曲线层面）|
| 力的合成（向量加法）| Token + position embedding 相加 | `model.py:185` |
| 张量积态空间 | 多头注意力的 reshape | `model.py:54-56` |
| 统计系综 | Batch 维度并行 | `model.py:179` |
| Boltzmann 熵 / Shannon 熵 | Cross-entropy / attention entropy | `model.py:192`、`model.py:68` |
| 自由能的 -TS 项 | RL entropy bonus | （PPO/DPO 训练阶段）|
| Feynman 路径积分 | 自回归 likelihood / beam search | `model.py:301-308` |
| 鞍点近似（WKB）| Beam search / greedy decoding | `model.py:301-302` |
| Wick 旋转 / 非平衡热力学 | Diffusion model 反向 SDE | （扩散模型）|
| 链式法则（多元微积分）| Backprop / autograd | `loss.backward()` |
| Gradient flow（最陡下降）| SGD | `optimizer.step()` |
| Euler 法解 ODE | 残差连接 / Neural ODE | `model.py:103-104` |
| Taylor 二阶展开 / 简正模式 | Newton 法 / Hessian 分析 | （优化器层面）|
| 投影到约束流形 / normal coordinates | LayerNorm | `model.py:27-28` |
| 不可微极限的光滑化 | Softmax ≈ argmax | `model.py:301` |

这张表说明的不是"巧合很多"，而是**深度学习的核心数学语言就是统计物理 + 量子力学 + 微积分的数学语言**：能量函数、概率分布、变分原理、对称性、标度律、向量代数、链式法则、ODE 演化。两个领域处理的"系统"看起来不同——电子自旋 vs token 概率——但描述这些系统的工具是同一套。

### 11.1 研究方法论上的同构

不只是公式，**思考方式**也是相通的：

- **写下能量函数 / loss function**：物理学家定义 Hamiltonian → ML 研究者定义 loss。两个动作完全同构。
- **找近似解，不追求精确解**：mean-field、perturbation、Monte Carlo → 在 ML 里就是 amortized inference、importance sampling、SGD（本身就是随机近似）。
- **用对称性约束模型空间**：群表示论 → 等变架构。
- **做 ansatz**：物理学家写下试探波函数、变分参数；ML 研究者设计架构（Transformer、Mamba、MoE）—— 都是猜函数空间的形状。
- **从数据反推模型 / 哈密顿量**：实验数据反推有效模型 ≈ 训练神经网络。

物理博士的核心训练就是这套——写 Hamiltonian、做近似、看相图、调参数。换到 ML 里，loss 取代 H，超参取代温度压强，scaling law 取代相图。**工作流是同一个**。

### 11.2 反向影响：AI 也在重塑物理学

这条河不是单向的：

- AlphaFold（NN 解蛋白质结构）+ AlphaTensor（NN 发现矩阵乘法算法）+ AlphaGeometry（数学奥赛）展示了 ML 是新的"科学计算工具"。
- 神经网络作为变分波函数（Carleo & Troyer 2017 *Solving the quantum many-body problem with artificial neural networks*）——直接把 NN 当 ansatz 解量子多体。
- 用 Transformer 学习 PDE 解算子（neural operator）。

所以 2024 年 Hinton 和 Hopfield 拿物理诺贝尔奖不是"物理界跨界蹭热度"，而是承认了一个早就成立的事实：**机器学习是统计物理的一个分支，只是数据规模和算力让它长出了独立的形态。**

---

## 写给打算从物理转 AI 的研究生

1. **你已经会一半了**。Softmax、cross-entropy、变分、KL、Langevin、对称性、相变、RG——你的数学工具箱直接够用。把"自由能"换成"loss"、"温度"换成"超参"，一周就能进入 ML 论文。
2. **缺的是工程肌肉**。PyTorch、autograd、CUDA、分布式训练、数据 pipeline——这些是物理课上不教的，但也不难，几个月就上手。`CodeGPT` 这种小项目就是不错的起点：`model.py` 不到 400 行，从 `forward` 读到 `generate` 就理解了 GPT 的所有数学。
3. **直觉迁移**。读论文时别只看公式，读"动机"和"近似"——大概率你会发现作者在做一个你熟悉的物理问题（mean-field、variational、RG）只是没用物理学的语言。
4. **危险的直觉**。物理学家容易过度追求"理论解释"，但 ML 很多结果是经验先于理论的（比如 Adam 比 SGD 好的"理由"现在还没完全说清）。要接受"先 work 再说为什么"的工程文化。

---

## 结语

深度学习不是从计算机科学里凭空长出来的。它的概念血脉一头连到 Shannon-Jaynes 的信息论，一头连到 Boltzmann-Gibbs-Hopfield 的统计物理，再借了量子力学的变分语言、希尔伯特空间几何、和测量诠释。当你在 PyTorch 里写下 `F.softmax(logits / T, dim=-1)` 这一行，你其实在重复 Boltzmann 1877 年写下的那个公式。

这不是"物理学家强行把自己的专业说成 AI 的本质"——这是历史。Hopfield 1982 是物理论文，Hinton 1985 的玻尔兹曼机是物理论文，2024 年的诺贝尔物理奖授予他们承认了这个传承。今天大模型每一个让人惊叹的能力背后，都有一组在物理学家眼里稀松平常的数学结构。

而对每一个学过量子力学和统计物理的研究生：你不是在跨界，你在**回家**。

---

## 附录 A：从自旋玻璃到神经网络（详解）

> 本附录结合《物理双月刊》文章 [《亂中有序：從自旋玻璃到神經網絡》](https://bimonthly.ps-taiwan.org/articles/67bc2d041efd7411b20caacd)，把第 3 节（Hopfield 网络）展开成一条完整的物理史脉络：从最朴素的 Ising 模型开始，经过阻挫、自旋玻璃、Parisi 的副本对称破缺，一路推到 Hopfield 网络、Boltzmann 机，最后接到现代 Transformer 的注意力机制。每个概念都给一段最短的、能跑的 PyTorch 代码。

### A.1 Ising 模型：自旋系统最朴素的描述

故事的起点是 1920 年代的 Ising 模型——晶格上每个格点放一个二值自旋 $s_i \in \{-1, +1\}$，相邻自旋之间有耦合 $J$：

```
H(s) = -J · Σ_{<i,j>} s_i s_j  -  h · Σ_i s_i
```

`<i,j>` 表示相邻格点对，$h$ 是外加磁场。$J > 0$ 时低能态是所有自旋同向（铁磁），$J < 0$ 是反向交错（反铁磁）。

写成 PyTorch 直接可跑：

```python
import torch
N = 8
s = torch.randint(0, 2, (N, N)) * 2 - 1   # ±1 自旋
J = 1.0
# 计算每一对横向 + 纵向相邻自旋的能量
E = -J * (s[:-1, :] * s[1:, :]).sum() - J * (s[:, :-1] * s[:, 1:]).sum()
```

整个统计力学第一年就在研究这个 $H$：求配分函数 $Z = \sum_s e^{-\beta H(s)}$、相变温度、临界指数。**Onsager 1944 年解出 2D Ising 模型的精确解，这是 20 世纪理论物理最伟大的成就之一。** 但它只是序幕。

### A.2 阻挫（Frustration）：当"无解"成为常态

文章用一个三角形说清了所有故事的起源：三个自旋 $s_1, s_2, s_3$ 两两之间是**反铁磁**耦合（$J < 0$，喜欢异号）。任意两个自旋很容易满足——一上一下就行。但**三个**自旋一起？

```python
import itertools
configs = list(itertools.product([-1, 1], repeat=3))
J = -1.0
for s in configs:
    E = -J * (s[0]*s[1] + s[1]*s[2] + s[2]*s[0])
    print(s, "E =", E)
```

跑出来你会发现：**没有一种构型能让三对相互作用全部满足**。永远会有一对自旋"被迫"和耦合的偏好相反。这就是**阻挫**（frustration）——系统找不到一个让所有局部约束同时满足的全局态。

文章原话："無論選擇朝上或是朝下，總是會跟其中一個自旋同向"。

阻挫不是一个边角病例——一旦把"耦合是随机的、有正有负"加进去，**绝大多数构型都是阻挫的**。这把整个能量地形从"一个干净的盆地"变成了"一片复杂的、布满局部极小的山脉"。

### A.3 自旋玻璃：Edwards-Anderson 与 Sherrington-Kirkpatrick 模型

铜里掺杂少量铁/锰原子，磁性自旋随机散布在格点上，磁交互作用（RKKY）随距离正负震荡——任意两个自旋的耦合 $J_{ij}$ 实际上是**随机变量**：

- **Edwards-Anderson (EA) 模型 (1975)**：晶格上每对相邻自旋的 $J_{ij}$ 独立从某个对称分布（比如 $\pm J$ 各 50%）抽取。
- **Sherrington-Kirkpatrick (SK) 模型 (1975)**：所有自旋两两都耦合（mean-field），$J_{ij} \sim \mathcal{N}(0, 1/N)$。

```python
N = 100
J = torch.randn(N, N) / (N ** 0.5)
J = (J + J.T) / 2          # 对称
J.fill_diagonal_(0)        # 自身不交互
s = torch.randint(0, 2, (N,)) * 2 - 1
E = -0.5 * s.float() @ J @ s.float()
```

这一段代码物理学家会立即认出来——**它和 `model.py:171` 的高斯权重初始化是同一个对象**：

```python
# model.py:171
torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

神经网络的每一层 `nn.Linear` 的权重矩阵，在初始化的那一刻，**结构上就是一个 SK 自旋玻璃的耦合矩阵**。Choromanska 等人 2015 年的论文 *The Loss Surfaces of Multilayer Networks* 把这个对应做严格——多层 ReLU 网络的损失函数 $\mathbb{E}\|y - f_\theta(x)\|^2$ 在大宽度极限下确实可以映射到一个球形 $p$-自旋玻璃模型，并继承它"绝大多数局部极小都几乎一样深"的性质。这就是为什么 SGD 找到的不是全局最优、却也能训出像样的模型。

### A.4 副本巧门与 Parisi 的副本对称破缺（RSB）

自旋玻璃的核心难点：耦合 $J_{ij}$ 是随机的，要算的是**自由能对耦合分布的平均** $\overline{F} = -kT \, \overline{\log Z}$。但 $\overline{\log Z}$ 没法直接算——$\log$ 在外面挡着。

物理学家祭出**副本巧门**（replica trick）：

```
log Z = lim_{n→0}  (Z^n - 1) / n
```

把 $n$ 当成正整数算 $\overline{Z^n}$（这是 $n$ 个独立副本系统的配分函数乘起来再平均），再**解析延拓**到 $n \to 0$。**这步在数学上根本不严格**，但它给出了正确答案。

副本平均后耦合的随机性消失，代价是引入了"副本对" $a, b \in \{1, \ldots, n\}$ 之间的耦合——**副本之间通过共享的耦合分布相互纠缠**。一个自然的假设是**副本对称**（replica symmetry）：所有副本对地位等价，序参量 $q_{ab}$ 只取一个值。

Sherrington 和 Kirkpatrick 用这个假设算出来 SK 模型的解——但低温下熵居然**变成负数**。这在物理上荒谬：熵是状态数的对数，不可能小于零。整个领域卡住了 4 年。

1979 年 **Giorgio Parisi** 的洞见（也是他获 2021 年诺贝尔物理学奖的核心工作）：**副本对称是个错误的假设**。文章原话："複本巧門並沒有要求複本對稱"。

Parisi 的方案叫**副本对称破缺**（Replica Symmetry Breaking, RSB）：把 $n$ 个副本递归地分组——先分成 $n/m_1$ 组，每组 $m_1$ 个；每组再分成 $m_1/m_2$ 个亚组……最后写出一个**层级结构**的 $q_{ab}$ 矩阵。极限 $n \to 0$ 后，这个层级变成一个连续函数 $q(x), x \in [0,1]$。

物理意义极深：**自旋玻璃的低温相不是"几个简并基态"，而是无穷多的、按超度量（ultrametric）层级组织的纯态**。任意两个纯态之间的"距离"满足 $d(A,C) \le \max(d(A,B), d(B,C))$（不等式比三角不等式更强）——正是树状层级的几何。

### A.5 能量景观与超度量结构

把上面的几何抽象具像化：

```python
def hopfield_energy(s, J):
    """E(s) = -1/2 sᵀ J s"""
    return -0.5 * (s.float() @ J @ s.float())

# 随机采样很多构型，看能量分布
samples = torch.sign(torch.randn(10000, N))
energies = torch.stack([hopfield_energy(s, J) for s in samples])
```

把这些构型按能量画图——你会得到一个**布满深谷、被高山隔开**的"能量地形"（energy landscape）。重要的特性：

- **指数级多的局部极小**：状态数随 $N$ 指数增长，绝大多数都是亚稳态（metastable）。
- **能垒高度同样指数大**：从一个谷到另一个谷需要翻很高的山。
- **超度量层级**：把所有谷按"两两之间能垒高度"聚类，得到一棵层级树。

第 10 节提到的 **Grokking 现象**（训练 loss 早就收敛，验证 loss 经过长平台期突然下降）——结构上恰好对应 RSB 那种"亚稳态群之间的层级跃迁"。SGD 长时间被困在一个层级的盆地里，某次随机扰动让它跨过能垒，进入一个更深、更具泛化性的层级。

### A.6 Hopfield 网络：把自旋玻璃改造成记忆机器

1982 年 **John Hopfield**（2024 年诺贝尔物理学奖得主之一）做了一件颠覆性的事：**反过来用自旋玻璃**。

物理学家研究自旋玻璃时把 $J_{ij}$ 当作**自然给定的随机量**，问"在这种耦合下系统会怎么演化"。Hopfield 反过来——**人为设计 $J_{ij}$**，让系统的局部极小**恰好是我们想存的记忆**。

定义"神经元"为 $\pm 1$ 自旋，能量函数：

```python
def E(s, W):
    return -0.5 * s @ W @ s
```

异步更新规则：随机选一个神经元 $i$，让它沿能量下降方向翻：

```python
def hopfield_update(s, W):
    i = torch.randint(0, len(s), (1,)).item()
    s[i] = torch.sign((W[i] @ s).clamp(min=1e-9, max=1) + 0)
    # 等价于 s[i] = sign(Σ_j W_ij s_j)
    return s
```

这个动力学**永不增加能量** —— Hopfield 1982 年的关键证明：定义的能量函数是 Lyapunov 函数，系统单调下降，必收敛到某个局部极小。

把局部极小设计成"记忆模式"，那么从任何含噪声的初始态出发，系统都会**自动滚到最近的记忆里**——这就是**联想记忆**（associative memory）：给一段残缺的输入，输出最匹配的完整记忆。这正好对应人脑"由部分线索回忆整体"的能力。

### A.7 Hebbian 学习：相关即耦合

怎么把记忆 $\xi^{(\mu)} \in \{-1,+1\}^N, \mu = 1, \ldots, P$ 编码到 $W_{ij}$ 里？Hopfield 用了 1949 年 **Donald Hebb** 提出的规则——"一起激发的神经元一起接线"（cells that fire together wire together）：

```python
def hebbian_weights(patterns):
    """patterns: (P, N) tensor of ±1, returns (N, N) W"""
    P, N = patterns.shape
    W = (patterns.T @ patterns).float() / N
    W.fill_diagonal_(0)
    return W
```

这就一行——$W = \frac{1}{N} \sum_\mu \xi^{(\mu)} (\xi^{(\mu)})^T$（去对角线）。每个模式贡献一个外积。

物理学家会立即认出：**这就是 SK 自旋玻璃的耦合矩阵，只不过 $J_{ij}$ 不是从高斯抽，而是 $P$ 个固定模式的外积叠加。** 当 $P$ 较小时，每个 $\xi^{(\mu)}$ 是一个能量极小（容易验证 $\xi^{(\mu)}$ 满足 $\text{sign}(W \xi^{(\mu)}) = \xi^{(\mu)}$）；当 $P$ 太大时，模式之间的串扰把所有局部极小搅成一团乱麻——又**变回了普通的自旋玻璃**。

### A.8 容量极限：0.14N 与灾难性遗忘

那么 Hopfield 网络能存几个记忆？Amit、Gutfreund、Sompolinsky 1985 年用副本巧门做了精确分析：

- $P/N < \alpha_c \approx 0.138$：记忆能被准确检索（每个 $\xi^{(\mu)}$ 是一个 stable attractor）。
- $P/N > 0.138$：相变发生，系统进入**自旋玻璃相**——每个记忆都被淹没在指数多的虚假极小（spurious states）里，无法检索。

这是物理学**第一次给出一个神经网络的精确容量极限**，而且工具就是副本巧门 + RSB——和分析普通自旋玻璃完全一样。

容量瓶颈也是 1980 年代神经网络遇冷的原因之一——**每神经元 0.14 个记忆**实在太少。深度学习要等到 30 年后才解决这个问题（见 A.10）。

### A.9 玻尔兹曼机：从确定性到随机

1985 年 **Geoffrey Hinton**（2024 年诺贝尔物理学奖另一位得主）和 Sejnowski 把 Hopfield 网络**随机化**——不再是确定性沿能量下降，而是按玻尔兹曼分布采样：

```python
def boltzmann_update(s, W, T=1.0):
    i = torch.randint(0, len(s), (1,)).item()
    h_i = (W[i] @ s).item()                       # 局部场
    p_up = 1.0 / (1.0 + torch.exp(torch.tensor(-2.0 * h_i / T)))
    s[i] = 1.0 if torch.rand(1).item() < p_up else -1.0
    return s
```

这个更新让系统按 $P(s) \propto e^{-E(s)/T}$ 采样——温度 $T$ 让它能跳出局部极小。再加上**隐变量**（hidden units）扩展模型容量，再加上**对比散度**（contrastive divergence, Hinton 2002）让训练高效——这就是**受限玻尔兹曼机**（RBM）和**深度信念网络**（DBN），2006 年深度学习复兴的引爆点。

注意第 1 节讲的 softmax = Boltzmann 分布——这个等号在玻尔兹曼机上是**字面**的，不是类比。`F.softmax(logits)` 在数学上就是玻尔兹曼机给某个状态的概率。Hinton 拿物理诺奖不是隐喻。

### A.10 现代 Hopfield 网络：指数容量 → Transformer 注意力

2020 年 Ramsauer 等 *[Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217)* 把 Hopfield 网络做了两件事：

1. **二值 → 连续**：自旋 $s \in \mathbb{R}^d$，能量改用 log-sum-exp 形式。
2. **多项式 → 指数容量**：新能量函数下能存 $\sim e^{cd}$ 个模式，远超 $0.14N$。

新能量与更新一步：

```python
import torch.nn.functional as F

def modern_hopfield_update(q, K, beta=1.0):
    """K: (P, d) 存的模式;  q: (d,) 查询; 返回更新后的查询"""
    # 一步迭代 = softmax 加权平均
    return K.T @ F.softmax(beta * (K @ q), dim=0)
```

把这一步原封不动对应到 `model.py` 注意力核心：

```python
# model.py:66-68
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
# 后面: y = att @ v   ← 等于 K.T @ softmax(...)
```

**结构完全一致**：`q @ kᵀ` = 查询与每个存储模式的相似度；`softmax` = 给每个记忆按玻尔兹曼分布投票；`@ v` = 加权检索。每个注意力头就是一次 Hopfield 更新。多头无非是几个并行的 Hopfield 网络，每个在自己的子空间里检索一组不同的记忆。

`1/√d` 的缩放在 Ramsauer 论文里对应**指数容量证明里的温度参数 $\beta$**——没有它注意力会塌缩到单个 token，物理上等价于自旋玻璃的零温极限。

### A.11 2024 年诺贝尔物理学奖：物理对 AI 的承认

2024 年 10 月，瑞典皇家科学院把诺贝尔物理学奖颁给 Hopfield 和 Hinton，引文是"for foundational discoveries and inventions that enable machine learning with artificial neural networks"。理由不是"他们做了 AI"，而是**他们用统计物理的方法解决了 AI 的问题**：

- Hopfield 1982 把 Ising 模型 / 自旋玻璃改造成联想记忆。
- Hinton 1985 把它随机化成玻尔兹曼机，再发明对比散度让它可训。
- 这一脉血缘从 Boltzmann (1877) → Onsager (1944) → Edwards-Anderson (1975) → SK (1975) → Parisi RSB (1979) → Hopfield (1982) → Hinton (1985) → Modern Hopfield (2020) → Transformer (2017，但 2020 才被证明等价) → ChatGPT (2022)。

合并 2021 年 Parisi 拿物理奖（自旋玻璃 RSB）+ 2024 年 Hopfield/Hinton 拿物理奖（基于自旋玻璃的神经网络），**统计物理在 21 世纪 20 年代连续两次因 AI 相关工作获奖**。这不是巧合——文章标题《亂中有序》说的就是这件事：从凌乱无序的随机自旋系统中涌现出有序的记忆、识别、生成能力。**深度学习的"乱"（高维参数 + 随机权重 + 随机数据）和"序"（学到的语义、推理、生成）之间的桥梁，是统计物理几十年来一直在研究的同一个对象**。

### A.12 在 CodeGPT 代码里看见这条血脉

| 概念 | 物理来源 | CodeGPT 中的对应 |
|------|----------|------------------|
| 随机权重 $J_{ij} \sim \mathcal{N}(0, \sigma^2)$ | SK 自旋玻璃耦合 | `model.py:171, 175` 高斯初始化 |
| 能量函数 $E = -\tfrac{1}{2} s^T W s$ | Hopfield 网络 | 注意力 logits 之前的 $Q K^T$ 矩阵 |
| 玻尔兹曼分布 $e^{-\beta E}/Z$ | 统计力学 | `F.softmax(att, dim=-1)` `model.py:68` |
| Hebbian 外积 $\sum_\mu \xi^{(\mu)} \xi^{(\mu)T}$ | 联想记忆学习规则 | `K.T @ K` 这种结构（隐含在 Q/K 投影里）|
| 一步 Hopfield 更新 | 联想记忆检索 | 单个注意力头 forward `model.py:51-73` |
| 多头 = 多套记忆系统 | 张量积态空间 | `model.py:54-56` reshape 到 n_head |
| 温度 / RSB 跳能垒 | 玻尔兹曼机训练 | `dropout` + SGD 噪声 + sampling temperature |
| 0.14N 容量瓶颈 | Amit-Gutfreund-Sompolinsky | （旧 Hopfield 的限制，已被现代 attention 突破） |

每次你跑 `python sample.py` 让 CodeGPT 续写一段代码，从最底层看，模型在做的事情就是：**在一个由训练数据塑造、参数高维的能量地形上，从一个查询点出发，通过多层的 Hopfield 式检索 + 玻尔兹曼采样，找到能量较低的延续路径**。这正是 1982 年 Hopfield 论文的一句话，被工业化放大了 8 个数量级之后的样子。

> 文章末段的洞察值得抄一遍：**"亂中有序"** —— 看似无序的随机自旋系统里，有清晰的物理规律；看似无序的高维神经网络参数里，有可读的语义、可推理的逻辑、可生成的创造。统计物理给了我们"在乱里看见序"的方法论，而这套方法论正在大模型时代结出第二轮果实。
