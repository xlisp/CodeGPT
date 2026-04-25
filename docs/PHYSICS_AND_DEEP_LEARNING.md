# 物理学的影子：量子力学与统计力学如何塑造了深度学习

> 为什么很多量子力学、统计力学方向的研究生转去做深度学习几乎没有"门槛"？因为他们脑子里的核心工具——配分函数、玻尔兹曼分布、变分原理、Langevin 动力学、重整化、对称性——在大模型里几乎一一对应。这篇文档把这些对应关系一条条钉到 `model.py` 的具体行号上，让"物理直觉"和"PyTorch 代码"之间的等号显式可见。

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
11. [思想史地图：为什么物理学家转 AI 没有门槛](#11-思想史地图为什么物理学家转-ai-没有门槛)

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

## 11. 思想史地图：为什么物理学家转 AI 没有门槛

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

这张表说明的不是"巧合很多"，而是**深度学习的核心数学语言就是统计物理的数学语言**：能量函数、概率分布、变分原理、对称性、标度律。两个领域处理的"系统"看起来不同——电子自旋 vs token 概率——但描述这些系统的工具是同一套。

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
