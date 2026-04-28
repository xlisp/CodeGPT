# 深度学习是可微分编程：从 y = wx + b 讲到 CodeGPT

> 2018 年初，Yann LeCun 在 Facebook 上宣告："Deep Learning est mort. Vive Differentiable Programming!"（深度学习已死，可微分编程万岁！）这不是修辞，而是一个架构视角的转变：神经网络不是一类特殊的统计模型，它就是一段**带可学习参数的程序**，只不过这段程序的每一步都是可微分的，可以被梯度下降自动"编译"出参数。
>
> 本文从最小的线性方程讲起，一路加结构——非线性、注意力、残差、多层堆叠——直到 `CodeGPT.forward()`。每一步都展示：我们不是在"训练一个模型"，我们是在**写一段程序**，然后让 `loss.backward()` 替我们填空。

---

## 目录

1. [LeCun 的宣言：为什么"深度学习"这个词不够用](#1-lecun-的宣言为什么深度学习这个词不够用)
2. [微积分简史：从行星轨道到 token 预测](#2-微积分简史从行星轨道到-token-预测)
3. [从一条直线开始：y = wx + b 就是最小的可微分程序](#3-从一条直线开始y--wx--b-就是最小的可微分程序)
4. [梯度下降：把"解方程"换成"写程序 + 求导"](#4-梯度下降把解方程换成写程序--求导)
5. [非线性：两条直线的组合打开了宇宙](#5-非线性两条直线的组合打开了宇宙)
6. [可微分的控制流：softmax、注意力、mask](#6-可微分的控制流softmax注意力mask)
7. [组合即编程：`nn.Module` 就是函数，`forward` 就是 `main`](#7-组合即编程nnmodule-就是函数forward-就是-main)
8. [大模型是这个范式的极致：CodeGPT 代码级解剖](#8-大模型是这个范式的极致codegpt-代码级解剖)
9. [回到 LeCun：深度学习 ⊂ 可微分编程](#9-回到-lecun深度学习--可微分编程)

---

## 1. LeCun 的宣言：为什么"深度学习"这个词不够用

LeCun 原话的关键一段：

> An increasingly large number of people are defining the networks procedurally in a data-dependent way (with loops and conditionals), allowing them to change dynamically as a function of the input data fed to them. It's really very much like a regular program, except it's parameterized, automatically differentiated, and trainable/optimizable.

翻译成工程语言：

- 一段 Python 程序里可以有 `for`、`if`、调用函数、读字典、查表；
- 如果把这些操作**全部换成可微分的近似**（循环 → 注意力、查表 → softmax 加权求和、if → 门控），
- 然后把"可调参数"放进每个操作里，
- 那这段程序就可以被 `autograd` 自动求导，
- 再用梯度下降"编译"出参数。

"深度学习"这个词容易让人以为它是一种模型类型（像 SVM、随机森林那样）。但神经网络的真正特性是：**它是一段程序，而不是一个固定的数学模型**。神经网络可以有数据依赖的控制流、递归结构、甚至运行时生成的子图——只要每一步都是可微分的。

这个视角下，PyTorch 不是一个"深度学习库"，它是一门**可微分编程语言**。

---

## 2. 微积分简史：从行星轨道到 token 预测

要理解为什么"可微分编程"这四个字里有"微分"，得先回到三百多年前——回到那个数学还在描述"形状"，却不会描述"变化"的时代。这一节绕一圈讲微积分的来历，然后回到大模型，你会发现 `loss.backward()` 干的事情，是牛顿和莱布尼茨那一代人就埋下的伏笔。

### 2.1 从静态到动态：人类第一次能描述"变化"

17 世纪以前的数学是**静态**的：几何描述形状，代数描述未知量。但人类需要回答的问题越来越**动态**：

- 这块石头此刻的速度是多少？
- 火星明晚会出现在天空哪个位置？
- 一笔本金以连续利息生长，一年后值多少？

这些问题的共同点是：**它们关心"变化"本身**。一秒前的速度、此刻的速度、一秒后的速度都不一样——你需要一种语言来描述"无穷小时间里位置变化了多少"。

牛顿和莱布尼茨各自独立发明了这门语言，今天叫**微积分**。它的核心不是公式，而是一个观念上的飞跃：**把"瞬间的变化"作为一种新的、可计算的对象**。从那一刻起，数学从描述"是什么"扩展到描述"在如何变"。

### 2.2 极限：瞬时速度怎么算？

平均速度好算：走过的路程除以时间。但"此刻的速度"听起来矛盾——此刻没有时间流逝，怎么有速度？

牛顿的答案：**让时间间隔无限缩小，看比值趋近于什么**。这就是"极限"。把它直接写成代码：

```python
# 数值微分:当 h 越来越小,比值收敛到瞬时变化率
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

import math
print(numerical_derivative(math.sin, 0.0))         # ≈ 1.0  (cos 0 = 1)
print(numerical_derivative(lambda x: x**2, 3.0))   # ≈ 6.0  (2x at x=3)
```

这就是导数的本质：**比值的极限**。一旦把它定义清楚，"瞬时速度"就从哲学问题变成可计算的数。

PyTorch 的 `autograd` 没有真的去算 `(f(x+h) - f(x)) / h`——它在符号层面应用链式法则，但**精神上等价**：每个 op 都登记了它的导数规则，反向传播时把这些规则按图组合起来。

### 2.3 积分是微分的逆：知道变化率，反推原函数

微积分的另一半是**积分**：把无穷多个小变化累加起来，重建总量。物理直觉非常清晰——

- 加速度积分得到速度；
- 速度积分得到位置；
- 温度变化率积分得到温度分布。

```python
import torch

dt = 0.01
acceleration = torch.full((1000,), 9.8)             # 自由落体, 10 秒
velocity = torch.cumsum(acceleration * dt, dim=0)   # 离散积分:累积求和
position = torch.cumsum(velocity     * dt, dim=0)
print(position[-1].item())  # ≈ 49.0 米 (10 秒末的位移)
```

`torch.cumsum` 是积分的离散形式。**给你变化率，反推出原函数**——这件事就是日后机器学习的精神原型：观测到的是"片段、变化、采样"，想恢复的是背后那个完整的规律。

牛顿/莱布尼茨证明的"微积分基本定理"说的就是：**微分和积分是一对逆运算**。这个对偶后来反复出现——在物理里是"力 ↔ 位移"，在 ML 里是"梯度 ↔ 参数更新"，本质都是同一种结构。

### 2.4 牛顿与莱布尼茨：两条路通向同一座山

两位发明者代表了两种至今仍并行的传统：

| 牛顿 | 莱布尼茨 |
|---|---|
| 物理驱动:要算行星轨道 | 符号驱动:要给数学一套好用的记号 |
| `流数`、`流量`(命名不太好用) | `dy/dx`、`∫`(沿用至今) |
| 万有引力 + 微分方程 → 天体力学 | 把微积分变成"代数一样可机械推演的东西" |

牛顿用微积分写出 $F = ma$，再加上万有引力定律，**第一次让人类能预测行星明天的位置**——这是有史以来最震撼的"用数学预测未来"的范例。在此之前，天文是观测+经验；在此之后，天文是方程+计算。

莱布尼茨更关心**符号的力学**：他的 `d`、`∫` 让微积分像普通代数一样可以"按规则推演"。今天 PyTorch 的 autograd 之所以能存在，本质上是莱布尼茨传统的胜利——**只要每个原子操作有符号化的导数规则，任意组合的程序都能被自动求导**。

> 你每次写 `loss.backward()`，其实是在调用一台沿莱布尼茨道路造了三百多年的"符号求导机器"。

### 2.5 天气预报：解一个巨大的微分方程

把视角拉到现代。今天打开手机看明天的天气，背后是什么？

大气的演化由一组**偏微分方程**支配（Navier-Stokes + 热力学方程）：

- 温度 T 怎么变？取决于自身梯度、对流、辐射……
- 压力 p 怎么变？取决于密度、风、温度……
- 风速 v 怎么变？取决于压力梯度、地转力、摩擦……

这是一个上千万格点、几十个状态变量的**微分方程系统**，没有解析解。气象台的做法是：

1. 把地球大气切成网格；
2. 把微分方程**离散化**（变成"格点之间的差分"）；
3. 用超算从今天的初始状态一步步推到明天。

**所谓"预测明天的天气"，本质上就是数值解一个巨大的微分方程系统。** 牛顿当年用一组方程预测了一颗行星，今天我们用同一种数学预测了一整片大气。

### 2.6 大模型也在解一个巨大的微分方程

现在做一次类比上的跳跃。

天气方程问：**给定今天的温度/压力/湿度，明天它们是多少？**
语言模型问：**给定前面的 token 序列 $x$，下一个 token $y$ 的概率分布 $P(y \mid x)$ 是什么？**

如果把"语言"想象成一个高维空间里的连续过程，**每个 token 就是这个过程的一个采样点**，那么模型要做的就是：

- 找到一个"原函数"——把上下文 $x$ 映射到 $y$ 的概率分布；
- 这个原函数的形状是未知的，但我们有海量观测数据（人类写过的文本和代码）；
- **用数据反推这个原函数，就像气象学家用观测数据反推大气方程的参数**。

CodeGPT 的整段 forward 就是这个"原函数"的参数化形式：

```python
# model.py:182-189 关键几行
tok_emb = self.transformer.wte(idx)       # 把离散 token 嵌入到连续空间
pos_emb = self.transformer.wpe(pos)
x = tok_emb + pos_emb
for block in self.transformer.h:          # 在连续空间里"逐层演化"
    x = block(x)
x = self.transformer.ln_f(x)
logits = self.lm_head(x)                  # 投影回 token 概率
```

注意结构上的呼应：

| 天气方程 | 语言模型 |
|---|---|
| 状态变量:温度、压力、湿度…… | 状态变量:768 维隐向量 `x` |
| 演化算子:N-S 方程的右端项 | 演化算子:每一层 `Block(x)` |
| 时间步进:积分微分方程 | 层叠步进:12 层 Block 堆叠 |
| 输出:明天的天气 | 输出:下一个 token 的分布 |

而**训练**是反过来：观测真实演化（语料），调整方程的参数，让模型预测的"明天 token"和真实"明天 token"尽可能接近。这正是 cross-entropy 损失干的事：

```python
# model.py:192
loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                       targets.view(-1), ignore_index=-1)
```

它在最小化"真实分布"与"模型分布"之间的差距（等价于 KL 散度）——也就是**让我们写下的"语言微分方程"逼近真实的"语言微分方程"**。每一次 `loss.backward()` + `optimizer.step()`，都是在这个亿级维度的方程组里挪动一小步。

### 2.7 为什么 ML 总在讲"概率同分布"：离散语言怎么被微积分捕获

到这里有个关键的疑问要正面回答：**牛顿的微积分活在连续空间——位置、速度、温度都是实数。但 token 是离散的、词表里的原子，没有"半个 token"这种东西。微积分凭什么管得了语言？**

答案藏在一个看似平凡的偷换：**我们不直接预测 token，我们预测 token 的概率分布**。下面把这个偷换拆开来看。

#### 2.7.1 离散观测，连续分布

设词表有 $V = 50304$ 个 token（CodeGPT 的设置，见 `tokenizer.py` 的 `VOCAB_SIZE`）。给定上下文 $x$，下一个 token $y$ 是 $V$ 个离散选项之一。这个**选择**是离散的，但 **"在 V 个选项上的概率分布"是连续的**——它是一个长度 $V$、各分量 $\in [0,1]$、总和为 $1$ 的向量，活在一个叫**概率单纯形**（probability simplex）的连续流形上。

```python
# 关键三步:连续 logits → 连续概率分布 → 离散采样
# model.py 推理路径(简化)
logits = self.lm_head(x[:, [-1], :])      # (B,1,V) 连续, 可任取实数
probs  = F.softmax(logits, dim=-1)         # (B,1,V) 连续, ∈[0,1] 且总和=1
next_token = torch.multinomial(probs, 1)   # 这一步才"离散化"(采样)
```

注意结构：**离散性只出现在最末一步采样**。前面所有运算——embedding、attention、MLP、softmax——都在连续空间里跑，所以梯度能一路反传。这是离散语言能被微积分捕获的工程秘密。

#### 2.7.2 "同分布"是什么假设？

ML 教科书反复出现一句话："训练样本独立同分布（i.i.d.）地采自某未知分布 $P^*$"。在大模型语境下它的意思是：

- 存在一个**真实的、未知的**语言/代码分布 $P^*(y \mid x)$；
- 互联网上的代码、文档、训练语料里的每一段，都是从 $P^*$ 里采的样本；
- 我们将来要让模型补全的代码，**也来自同一个 $P^*$**——这就是"同分布假设"；
- 所以学好训练集的统计规律，就能推广到未来。

如果未来分布和训练集大不相同（训练全是 Python 2，部署全是 Rust），同分布假设就破了，模型必然失效。**同分布假设是机器学习能从过去外推到未来的全部理由**——它在 ML 里扮演的角色，相当于物理里"自然规律不随时间改变"。

#### 2.7.3 P(y|x) 就是被反推出来的"语言原函数"

把上面两点合起来，就回答了你的问题：

> **是的——`P(y|x)` 正是我们想反推出来的"语言原函数"**。

类比一一对上：

| 牛顿物理 | 大模型 |
|---|---|
| 观测:一系列位置采样 | 观测:一堆 token 序列(语料) |
| 假设:背后存在连续轨迹 $x(t)$ | 假设:背后存在真实分布 $P^*(y \mid x)$ |
| 目标:反推 $x(t)$(原函数) | 目标:反推 $P^*(y \mid x)$(原函数) |
| 工具:把观测代入运动方程,用积分恢复轨迹 | 工具:用神经网络做参数化原函数,梯度下降拟合 |

差别只在：物理的"原函数"是连续轨迹（一维实变量），语言的"原函数"是定义在所有上下文上的概率分布族（无穷多个 simplex 上的点）。**反推的精神完全一致**：观测是局部、片段、采样的；原函数是完整的、规律性的；学习就是从前者重建后者。

#### 2.7.4 离散 + 概率 + 矩阵 = 可微分

为什么必须是"概率"和"矩阵"？因为它们是**让离散问题可微分的唯一组合**：

1. **离散 → 概率**：把 V 个选项的"硬选择"换成 V 维概率向量；
2. **概率 → 矩阵**：embedding 是矩阵 $V \times d$，把离散 token id 翻译成连续向量；`lm_head` 是矩阵 $d \times V$，把连续向量翻译回 V 维 logits；
3. **矩阵 → 可微**：矩阵乘法处处可微，梯度可以经由它流到每一个 embedding 行；
4. **可微 → 梯度下降**：autograd 沿前向图反推，把误差信号变成参数更新方向。

具体到 CodeGPT，这条链子在代码里就是这几跳：

```python
# 离散 token → 连续向量 → 离散概率
tok_emb = self.transformer.wte(idx)        # (B,T) 离散 id  → (B,T,d) 连续向量
# ... 中间所有 attention/MLP 都在 (B,T,d) 连续空间运算 ...
logits  = self.lm_head(x)                  # (B,T,d) → (B,T,V) 连续 logits
loss    = F.cross_entropy(                 # 用真实 token id 当 target
    logits.view(-1, logits.size(-1)),
    targets.view(-1),
    ignore_index=-1,
)
```

`wte` 是入口、`lm_head` 是出口，**它俩共享权重**（`model.py:153` `lm_head.weight = wte.weight`）——意思是同一个 $V \times d$ 矩阵既负责"id 进来变向量"，也负责"向量出去变 V 维分数"。这种对称性正暗示了：**整个网络是定义在词表上的一个函数族，以这个嵌入矩阵为坐标系**。

#### 2.7.5 cross-entropy 把"反推原函数"变成可下降的目标

最后一步，要把"我们想要 $P_\theta \approx P^*$"翻译成可优化的标量。这里登场的是 **cross-entropy / KL 散度**——它正是在所有"测距"工具里**和最大似然等价**的那一个：

$$
\mathcal{L}(\theta) \;=\; -\frac{1}{N}\sum_{i=1}^{N} \log P_\theta(y_i \mid x_i)
$$

每个训练样本 $(x_i, y_i)$ 是从 $P^*$ 采的。最小化这个 loss 等价于最小化 $\mathrm{KL}(P^* \,\|\, P_\theta)$——也就是**让模型分布尽量贴住真实分布**。代码里就是一行：

```python
# model.py:192
loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                       targets.view(-1), ignore_index=-1)
```

这条 loss 连续、可微，对每一个参数都有梯度。autograd 沿反向图把梯度送回所有 124M 参数，optimizer 推一小步——**$P_\theta$ 就向 $P^*$ 靠近一点**。重复几十万步，模型就"反推出"了一个足够接近真实语言分布的近似原函数。

> **一句话回答你的问题**：是的，`P(y|x)` 就是被反推出来的"语言原函数"；之所以非概率不可，是因为语言离散、硬预测不可微；之所以非矩阵不可，是因为矩阵是把离散 id 嫁接到连续可微空间的桥；之所以非梯度不可，是因为反推这个原函数没有解析解，只能用梯度下降数值地"积"出来。**离散观测 + 连续分布 + 矩阵参数化 + 梯度下降**——这四件套合起来，才让微积分这门连续数学第一次能解一个离散符号系统的"微分方程"。

### 2.8 所以叫"可微分编程"：解一个亿级参数的微分方程

把上面这条线索拉直：

1. 微积分诞生于"描述变化"——牛顿要算行星，莱布尼茨要好用的符号；
2. 它的两半是**导数**（变化率）和**积分**（从变化率反推原函数）；
3. 自然界的复杂系统——天体运动、气候、流体——都用微分方程描述，**求解 = 预测未来**；
4. **语言、代码也可以视作一种"微分方程"**：给定上下文（初始条件），预测下一个 token（下一时刻状态）；
5. 我们不知道这个方程长什么样，但可以**用神经网络作为它的参数化原函数**，让数据决定参数；
6. 训练神经网络 = 数值求解一个 124M 维参数的微分方程组——找到那个让 loss 最小的参数解；
7. 整个过程能跑起来，是因为 `autograd` 把莱布尼茨的链式法则自动化了。

这就是为什么 LeCun 说"深度学习已死，**可微分编程**万岁"。"可微分"三个字直接呼应微积分；"编程"则是莱布尼茨"符号化"传统的现代极致——我们不再手算导数，而是用 PyTorch 这门**可微分编程语言**写程序，让 autograd 替我们应用三百年前发明的那条规则。

> **一句话总结这一节**：天气预报是用微积分解大气方程；CodeGPT 是用微积分解"语言方程"。前者积分的是温度和压力，后者"积分"的是 token 之间的概率结构——但它们用的是同一种数学。

理解了这一层，下面从 `y = wx + b` 一路加到 `forward()` 就只是同一件事的逐步展开：写一段可微的程序，让梯度下降把它的参数"解"出来。

---

## 3. 从一条直线开始：y = wx + b 就是最小的可微分程序

最简单的程序：

```python
def program(x, w, b):
    return w * x + b
```

这个"程序"有两个可调参数 `w`、`b`，和一个输入 `x`。它可微吗？显然可微：

```
∂y/∂w = x
∂y/∂b = 1
```

现在假设我有一堆 `(x, y)` 数据对（比如 x=身高，y=体重），我想找到最好的 `w`、`b`。

**传统做法（手工求解）**：最小二乘法。写出损失函数，对 `w`、`b` 求偏导，令为零，解出解析解。

```
L(w, b) = Σᵢ (yᵢ - (w·xᵢ + b))²
```

这是一种"编程"——你用数学公式**亲手**算出了参数。

**可微分编程做法**：不解方程，直接让程序自己"跑"出参数。

```python
import torch

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = torch.tensor([2.1, 3.9, 6.1, 7.8])   # 大约是 y = 2x

for step in range(200):
    pred = w * x + b                     # ← 前向"程序"
    loss = ((pred - y) ** 2).mean()      # ← 目标
    loss.backward()                      # ← 自动求导，填入 w.grad、b.grad
    with torch.no_grad():
        w -= 0.01 * w.grad
        b -= 0.01 * b.grad
        w.grad.zero_(); b.grad.zero_()

print(w.item(), b.item())  # 约等于 2.0, 0.0
```

注意这里发生了什么：

1. 你**没有**写任何求导代码。`loss.backward()` 自动算了 `∂loss/∂w` 和 `∂loss/∂b`。
2. 你写的是**前向程序**（`w * x + b`），反向传播是 PyTorch 根据前向图自动生成的。
3. 如果你把第一行改成 `pred = w * x * x + b`（一条抛物线），**整段代码不用改**——autograd 会处理新的求导规则。

这就是"可微分编程"的雏形：**你写前向，系统自动给你反向**。

---

## 4. 梯度下降：把"解方程"换成"写程序 + 求导"

上面的例子里，我们可以解析求解（最小二乘有闭式解），但这是**运气好**。一旦前向程序变复杂，解析解立刻消失。

比如把程序改成：

```python
def program(x, w1, w2, b):
    return torch.relu(w1 * x + b) * w2
```

现在 `relu` 在 0 点不可导，而且是分段的，联立方程没法消元。但**梯度下降完全不在乎这个**：

- PyTorch 知道 `relu` 在正半轴导数为 1，负半轴为 0；
- 链式法则把这些导数一层层乘起来；
- `loss.backward()` 照样给你 `w1.grad`、`w2.grad`、`b.grad`。

这是视角转变的关键：

| 传统数学建模 | 可微分编程 |
|---|---|
| 写出 $y = f(x; \theta)$ | 写一段 Python |
| 写出 loss 的解析梯度 | 调用 `loss.backward()` |
| 需要保证闭式解存在 | 只要每步可微（或有次梯度）即可 |
| 模型形式受解析能力限制 | 模型形式受**想象力**限制 |

所以 LeCun 的观点可以重述为：**一旦有了 autograd，"神经网络"和"普通程序"的界限就消失了。**

---

## 5. 非线性：两条直线的组合打开了宇宙

如果只堆叠线性层，有个灾难：

```python
nn.Linear(n, n) @ nn.Linear(n, n)   # 还是一个线性函数
```

两个线性层相乘，数学上等价于一个线性层。堆再多层也没用。

**突破来自插入一个非线性函数。** CodeGPT 的 MLP 长这样：

```python
# model.py:76-90
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu   = nn.GELU()                                  # ← 关键的一行
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)          # ← 把线性空间掰弯
        x = self.c_proj(x)
        return x
```

GELU 本身很简单，可微，长得像一条柔化的 ReLU。但它的存在让整个 MLP 成为**通用函数近似器**（universal approximator）——理论上，足够宽的单隐层 MLP 可以近似任何连续函数。

从可微分编程的角度看：

- `nn.Linear` 是 "带可学习参数的矩阵乘法"；
- `nn.GELU` 是 "不带参数但引入非线性的操作"；
- 两者都可微 → 梯度能穿透整个 MLP；
- 所以 MLP 就是一段"有两个可调矩阵 + 一个非线性门"的可微分程序。

**经典编程类比**：`c_fc` 把输入升维到 4 倍（像是在高维空间"展开"），`gelu` 做出选择（类似"如果某个特征大于阈值就激活"），`c_proj` 再把结果投影回原维度（类似"总结"）。但我们没写 `if`——是 GELU 的曲线形状在可微地扮演 `if`。

---

## 6. 可微分的控制流：softmax、注意力、mask

MLP 解决了"非线性"，但还缺一样东西——**数据依赖的控制流**。

传统 Python 程序里，你会写：

```python
def retrieve(query, database):
    for key, value in database.items():
        if key == query:
            return value
```

这里有 `for`、`if`、`==`——**没有一个是可微的**。你不能对 `==` 求导。

注意力机制的魔法就在于：**它是这段查表代码的可微分版本。**

```python
# model.py:51-73 中的核心几行
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)         # 查询、键、值
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))   # 相似度（不是 ==，是点积）
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # 因果 mask
att = F.softmax(att, dim=-1)                               # 归一化成概率
y   = att @ v                                              # 加权求和
```

把它和传统查表对照：

| 传统查表（不可微）          | 注意力（可微）                  |
|-----------------------------|---------------------------------|
| `key == query`（布尔值）    | `q @ k.T`（连续相似度）         |
| `if` 选中一个 value         | `softmax` 把相似度变成权重      |
| `return value`              | `att @ v`（所有 value 的加权平均） |
| 离散决策                    | **连续、可微的"软决策"**        |

`softmax` 在这里扮演的角色，本质上是 `argmax` 的**可微分近似**。当某个 key 的相似度远大于其他，softmax 会给它接近 1 的权重，其它接近 0——退化成"硬查表"；但整个过程对 `q`、`k`、`v` 都可导。

这才是 LeCun 所说的精髓：**注意力不是一个新模型，它是"查表"这一经典程序结构的可微分改写。**

类似地：

- `for` 循环 → **自注意力的并行全对全**（每个位置看所有位置）；
- 递归 / 栈 → **多层 Transformer Block 堆叠**（深度模拟迭代）；
- 变量绑定 → **残差连接** `x = x + attn(x)`（保留"上一步状态"）；
- 字典 → **embedding 查表** `wte(idx)`（可微的 id → 向量）。

一旦把这些都翻译成可微分的形式，整个程序就可以被反向传播穿透。

---

## 7. 组合即编程：`nn.Module` 就是函数，`forward` 就是 `main`

有了"可微分的原子操作"，下一步就是**用它们写大程序**。这就是 `nn.Module` 的作用。

看 `Block` 的定义：

```python
# model.py:93-105
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))    # ← 一行 Python
        x = x + self.mlp(self.ln_2(x))     # ← 一行 Python
        return x
```

这**就是一段 Python 代码**。不是比喻——它真的在运行时按这两行顺序执行。但这段代码的特别之处在于：

1. 每个函数调用（`ln_1`、`attn`、`ln_2`、`mlp`）都带可学习参数；
2. 每个 `+`、每个函数调用都可微；
3. 调用 `.backward()` 时，PyTorch 根据这段 Python 的执行轨迹自动构造反向图。

**这和写普通 Python 函数有什么区别？**

| 普通 Python 函数 | `nn.Module.forward` |
|---|---|
| 参数在调用时传入 | 参数保存在 `self.xxx`，由 `.parameters()` 暴露 |
| 没有"可训练"概念 | 所有 `nn.Parameter` 自动进入梯度图 |
| 要求导需手工算 | `loss.backward()` 自动微分 |
| 执行 = 结果 | 执行 = 结果 + 反向图 |

换句话说：**`nn.Module` 是可微分编程语言里的"函数定义"，`forward` 方法就是函数体。**

---

## 8. 大模型是这个范式的极致：CodeGPT 代码级解剖

现在把前面所有片段组装起来，看看 CodeGPT 作为一段"可微分程序"到底是什么样子。

完整的"主函数"：

```python
# model.py:177-198
def forward(self, idx, targets=None):
    device = idx.device
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    tok_emb = self.transformer.wte(idx)       # id → 向量（可微查表）
    pos_emb = self.transformer.wpe(pos)       # 位置 → 向量
    x = self.transformer.drop(tok_emb + pos_emb)
    for block in self.transformer.h:          # ← 普通 Python for 循环
        x = block(x)                          # ← 每层 Block 都是可微函数
    x = self.transformer.ln_f(x)

    if targets is not None:
        logits = self.lm_head(x)              # 投影回词表维度
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1), ignore_index=-1)
    else:
        logits = self.lm_head(x[:, [-1], :])
        loss = None
    return logits, loss
```

**这就是一段 Python 程序。** 它有：

- 变量赋值（`tok_emb = ...`）；
- for 循环（`for block in self.transformer.h`）；
- if 分支（`if targets is not None`）；
- 函数调用（`block(x)`、`F.cross_entropy`）。

和普通 Python 程序唯一的区别是：**每一个操作都在可微图里留下了痕迹**。调用一次 `loss.backward()`，梯度自动流回所有 124M 参数。

### 8.1 程序员的部分 vs. 梯度下降的部分

写 CodeGPT 的人（架构师）决定了这些：

- 用多少层 Block（`n_layer=12`）；
- 每层多少维（`n_embd=768`）；
- 注意力用因果 mask（`is_causal=True`）；
- 用 Pre-LN（`x + attn(ln(x))` 而不是 `ln(x + attn(x))`）；
- 权重绑定 `lm_head.weight = wte.weight`（`model.py:153`）；
- FIM 变换在训练时 50% 概率触发；
- 损失函数用 cross-entropy，忽略 -1 的 target（`ignore_index=-1`）。

这些是**代码**——人类写的、可微分程序的"骨架"。

梯度下降决定了这些：

- `wte.weight`：每个 token 的 768 维向量长什么样；
- `c_attn.weight`：注意力怎么选择关注哪里；
- `c_fc.weight`、`c_proj.weight`：MLP 怎么变换表示；
- `ln_*.weight`：每层归一化的缩放因子。

这些是**参数**——机器通过看数据"写"出来的、可微分程序的"填空"部分。

### 8.2 训练循环 = 可微分程序的"编译"过程

```python
# 训练循环的核心（train.py 简化版）
logits, loss = model(X, Y)    # 跑一遍前向程序
loss.backward()               # 自动生成反向程序并执行
optimizer.step()              # 用梯度更新参数
optimizer.zero_grad()
```

每一次迭代：

1. **前向**：按 `forward` 代码执行，得到 loss；
2. **反向**：autograd 沿着前向图逆行，算出每个参数的梯度；
3. **更新**：优化器用梯度调整参数。

重复几万到几百万步，程序的参数就被"编译"完成。模型推理时，同一段 `forward` 代码再跑一遍——参数已经是训练好的值。

**所以训练 = 编译，推理 = 执行。** 这个类比不是比喻，从计算图的角度看就是字面意思。

### 8.3 一个关键细节：`F.cross_entropy` 为什么放进 forward？

看这行：

```python
# model.py:192
loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                       targets.view(-1), ignore_index=-1)
```

如果 loss 计算在 `forward` 之外，autograd 也能工作。但放进 `forward` 有一个可微分编程视角下的好处：**让"程序"的输出从"预测"变成"评价预测"**。这样整个 `CodeGPT` 模块就是一个**端到端可微的函数**——输入 `(idx, targets)`，输出一个标量 loss。DDP、混合精度、`torch.compile` 都围绕这个"一个前向产生一个 loss"的约定运行。

这也是为什么 `ignore_index=-1` 这个细节如此重要：FIM 变换在 target 里把 pad 位置设为 -1（`train.py:get_batch`），`cross_entropy` 就自动跳过这些位置的梯度。**程序员用一个 `-1` 实现了"不在这里反向传播"，而不需要手动写 mask 再乘到梯度上。** 这就是可微分编程该有的优雅。

---

## 9. 回到 LeCun：深度学习 ⊂ 可微分编程

现在可以重述 LeCun 的观点：

- **深度学习**这个词强调的是"很多层的网络"——这只是程序的一种**特定结构**（层级堆叠）。
- **可微分编程**强调的是背后的**范式**——只要每步可微，什么结构都可以。

如果只从"堆叠层"的视角看 CodeGPT，你会错过：

- `apply_fim_transform` 是一段数据相关的控制流（取决于输入概率触发）；
- `generate()` 里的 `top_k`、`top_p`、`repetition_penalty` 是推理时的动态子图（`model.py:258-310`）；
- `crop_block_size`、`expand_vocab` 是在**运行时修改程序结构**（`model.py:200`、`model.py:359`）；
- DDP 把同一段 `forward` 复制到多 GPU 并行执行——像是把程序并行编译。

这些都是"程序"的特征，不是"静态模型"的特征。

### 9.1 为什么这个视角重要？

如果你把神经网络当作"一个数学函数 $f_\theta(x)$"来想，会**主动限制自己**——总觉得要写出整洁的闭式表达。

但如果你把它当作"一段程序"，你会很自然地想到：

- 为什么不能在 forward 里做 `torch.where(cond, branch_a, branch_b)`？能，这就是**门控**；
- 为什么不能在 forward 里用 `while` 循环直到条件满足？能，这就是**自适应计算时间**（ACT）；
- 为什么不能在 forward 里动态选择哪个子模块？能，这就是 **Mixture of Experts**；
- 为什么不能让模型输出代码再执行？能，这就是 **Neural Program Induction**。

所有这些"新"结构，本质上都是**在可微分编程框架下写更复杂的程序**。大模型的进步，一半来自算力和数据，另一半来自"我们学会了在 forward 里写更精巧的程序"。

### 9.2 与本项目其他文档的联系

- [DEEP_DIVE.md](./DEEP_DIVE.md) 讲的是这段"程序"从 RNN 演化到 Transformer 的过程——每一次架构升级都在**扩充可微分编程的表达力**。
- [COMPRESSION_IS_INTELLIGENCE.md](./COMPRESSION_IS_INTELLIGENCE.md) 讲的是**目标**——为什么这段程序的 loss 定义为 cross-entropy 是在压缩数据。
- [RLHF_AND_PLATONIC_REPRESENTATION.md](./RLHF_AND_PLATONIC_REPRESENTATION.md) 讲的是**扩展**——当基础程序训练好后，如何用 RLHF 把另一段可微分程序（奖励模型）接到上面继续优化。

三篇分别从"结构"、"目标"、"对齐"三个角度解释 CodeGPT；本篇则从"范式"角度指出：**它们讲的都是同一件事——可微分编程**。

---

## 结语：PyTorch 其实是一门语言

如果你回头看 `model.py` 的每一行，会发现没有一行是"魔法"：

- `nn.Linear` 是矩阵乘法 + 可学习参数；
- `F.softmax` 是指数归一化；
- `F.cross_entropy` 是 log + 取值 + 求平均；
- `loss.backward()` 是链式法则的自动展开；
- `optimizer.step()` 是 `w -= lr * w.grad` 的工程化封装。

每一个原子都简单，简单到可以在一张纸上推导导数。但它们组合起来——通过 `nn.Module` 定义的函数层次、通过 `forward` 里的控制流、通过 `autograd` 维护的反向图——就构成了一种**语言**。用这种语言，你可以编写数亿参数的程序，让梯度下降把它们全部"填空"完成。

所以当 LeCun 说"深度学习已死，可微分编程万岁"时，他想表达的是：

> 别再把注意力放在"网络有多深、多宽"上。去想你在 forward 里写了什么程序。去想你引入了哪些新的可微分操作。去想如何用 autograd 把更复杂的经典算法翻译成可学习的形式。

**CodeGPT 从这个视角看，就是一段长长的、124M 个参数的 Python 程序——它的 `forward` 是人写的骨架，它的参数是梯度下降写出来的血肉，而 `loss.backward()` 是把两者粘合起来的编译器。**
