# 深度学习是可微分编程：从 y = wx + b 讲到 CodeGPT

> 2018 年初，Yann LeCun 在 Facebook 上宣告："Deep Learning est mort. Vive Differentiable Programming!"（深度学习已死，可微分编程万岁！）这不是修辞，而是一个架构视角的转变：神经网络不是一类特殊的统计模型，它就是一段**带可学习参数的程序**，只不过这段程序的每一步都是可微分的，可以被梯度下降自动"编译"出参数。
>
> 本文从最小的线性方程讲起，一路加结构——非线性、注意力、残差、多层堆叠——直到 `CodeGPT.forward()`。每一步都展示：我们不是在"训练一个模型"，我们是在**写一段程序**，然后让 `loss.backward()` 替我们填空。

---

## 目录

1. [LeCun 的宣言：为什么"深度学习"这个词不够用](#1-lecun-的宣言为什么深度学习这个词不够用)
2. [从一条直线开始：y = wx + b 就是最小的可微分程序](#2-从一条直线开始y--wx--b-就是最小的可微分程序)
3. [梯度下降：把"解方程"换成"写程序 + 求导"](#3-梯度下降把解方程换成写程序--求导)
4. [非线性：两条直线的组合打开了宇宙](#4-非线性两条直线的组合打开了宇宙)
5. [可微分的控制流：softmax、注意力、mask](#5-可微分的控制流softmax注意力mask)
6. [组合即编程：`nn.Module` 就是函数，`forward` 就是 `main`](#6-组合即编程nnmodule-就是函数forward-就是-main)
7. [大模型是这个范式的极致：CodeGPT 代码级解剖](#7-大模型是这个范式的极致codegpt-代码级解剖)
8. [回到 LeCun：深度学习 ⊂ 可微分编程](#8-回到-lecun深度学习--可微分编程)

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

## 2. 从一条直线开始：y = wx + b 就是最小的可微分程序

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

## 3. 梯度下降：把"解方程"换成"写程序 + 求导"

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

## 4. 非线性：两条直线的组合打开了宇宙

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

## 5. 可微分的控制流：softmax、注意力、mask

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

## 6. 组合即编程：`nn.Module` 就是函数，`forward` 就是 `main`

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

## 7. 大模型是这个范式的极致：CodeGPT 代码级解剖

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

### 7.1 程序员的部分 vs. 梯度下降的部分

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

### 7.2 训练循环 = 可微分程序的"编译"过程

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

### 7.3 一个关键细节：`F.cross_entropy` 为什么放进 forward？

看这行：

```python
# model.py:192
loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                       targets.view(-1), ignore_index=-1)
```

如果 loss 计算在 `forward` 之外，autograd 也能工作。但放进 `forward` 有一个可微分编程视角下的好处：**让"程序"的输出从"预测"变成"评价预测"**。这样整个 `CodeGPT` 模块就是一个**端到端可微的函数**——输入 `(idx, targets)`，输出一个标量 loss。DDP、混合精度、`torch.compile` 都围绕这个"一个前向产生一个 loss"的约定运行。

这也是为什么 `ignore_index=-1` 这个细节如此重要：FIM 变换在 target 里把 pad 位置设为 -1（`train.py:get_batch`），`cross_entropy` 就自动跳过这些位置的梯度。**程序员用一个 `-1` 实现了"不在这里反向传播"，而不需要手动写 mask 再乘到梯度上。** 这就是可微分编程该有的优雅。

---

## 8. 回到 LeCun：深度学习 ⊂ 可微分编程

现在可以重述 LeCun 的观点：

- **深度学习**这个词强调的是"很多层的网络"——这只是程序的一种**特定结构**（层级堆叠）。
- **可微分编程**强调的是背后的**范式**——只要每步可微，什么结构都可以。

如果只从"堆叠层"的视角看 CodeGPT，你会错过：

- `apply_fim_transform` 是一段数据相关的控制流（取决于输入概率触发）；
- `generate()` 里的 `top_k`、`top_p`、`repetition_penalty` 是推理时的动态子图（`model.py:258-310`）；
- `crop_block_size`、`expand_vocab` 是在**运行时修改程序结构**（`model.py:200`、`model.py:359`）；
- DDP 把同一段 `forward` 复制到多 GPU 并行执行——像是把程序并行编译。

这些都是"程序"的特征，不是"静态模型"的特征。

### 8.1 为什么这个视角重要？

如果你把神经网络当作"一个数学函数 $f_\theta(x)$"来想，会**主动限制自己**——总觉得要写出整洁的闭式表达。

但如果你把它当作"一段程序"，你会很自然地想到：

- 为什么不能在 forward 里做 `torch.where(cond, branch_a, branch_b)`？能，这就是**门控**；
- 为什么不能在 forward 里用 `while` 循环直到条件满足？能，这就是**自适应计算时间**（ACT）；
- 为什么不能在 forward 里动态选择哪个子模块？能，这就是 **Mixture of Experts**；
- 为什么不能让模型输出代码再执行？能，这就是 **Neural Program Induction**。

所有这些"新"结构，本质上都是**在可微分编程框架下写更复杂的程序**。大模型的进步，一半来自算力和数据，另一半来自"我们学会了在 forward 里写更精巧的程序"。

### 8.2 与本项目其他文档的联系

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
