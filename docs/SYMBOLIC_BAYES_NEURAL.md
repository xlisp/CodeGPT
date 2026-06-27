# 符号主义、贝叶斯网络、深度学习：三种 AI 范式的对比学习

> 一个非常自然的直觉是这样的：**"GPT 不就是把一堆自然语言数据当成可执行的符号代码，然后在这些海量规则下做回答吗？"**
>
> 这个类比有 70% 是对的，30% 错得很关键。要把这 30% 讲清楚，需要把符号主义 AI、贝叶斯网络、深度学习放在同一张桌子上对照——它们三者**都在解同一个问题** $P(y \mid x)$，只是参数化方式根本不同。本文从这个共同问题出发，把三种范式的相同点、差异、各自擅长的边界，一条条钉到 `model.py` / `tokenizer.py` 的具体行号上。
>
> 读完这篇，你会知道为什么"训练数据 = 可执行符号代码"是一个**有用的近似**，但**不能当结论**——以及为什么贝叶斯网络是介于两者之间的"半符号半概率"中间态。
>
> 本文 2026 年的增补版还埋了一条**暗线**:当三种范式表里"谁写"这一列一路走到尽头——**连训练过程本身都开始由模型来写**(自递归训练 / recursive self-improvement),以及当"压缩即智能"被重新形式化为"**模型学到的不是数据本身,而是为了压缩数据而不得不构造的那个复杂程序**"(epiplexity,arXiv 2601.03220)——这两件 2026 年的最新进展会告诉你:从 GPT-2 到 Claude Mythos / Fable 5 的演化不是偶然,而是这条主线的**逻辑终点**。这条暗线在每一节里若隐若现,最后在 [§12](#12-2026-前沿自递归训练与-epiplexity历史必然如何走向-mythos--fable-5) 收束。

---

## 目录

1. [三个范式的一句话总结](#1-三个范式的一句话总结)
2. [都在解同一个问题：$P(y \mid x)$](#2-都在解同一个问题py--x)
3. [符号主义 AI：人写规则，机器搜索](#3-符号主义-ai人写规则机器搜索)
4. [贝叶斯网络：图是符号的，参数是概率的](#4-贝叶斯网络图是符号的参数是概率的)
5. [深度学习大模型：人写架构，梯度下降写参数](#5-深度学习大模型人写架构梯度下降写参数)
6. [可解释性：深度学习最大的软肋,以及它现在进展到哪一步](#6-可解释性深度学习最大的软肋以及它现在进展到哪一步)
7. ["训练数据 = 一堆可执行的符号代码"——这个类比对在哪、错在哪](#7-训练数据--一堆可执行的符号代码这个类比对在哪错在哪)
8. [三种范式的对照表](#8-三种范式的对照表)
9. [在 CodeGPT 里看见三个范式的影子](#9-在-codegpt-里看见三个范式的影子)
10. [函数式编程视角：神经网络是 f∘g∘h∘... 的世界](#10-函数式编程视角神经网络是-fgh-的世界)
11. [神经-符号混合：现代系统都不是纯种](#11-神经-符号混合现代系统都不是纯种)
12. [2026 前沿：自递归训练与 Epiplexity——历史必然如何走向 Mythos / Fable 5](#12-2026-前沿自递归训练与-epiplexity历史必然如何走向-mythos--fable-5)
13. [给用户原问题的最终回答](#13-给用户原问题的最终回答)

---

## 1. 三个范式的一句话总结

| 范式 | 知识形式 | 谁写 | 谁执行 | 代表系统 |
|------|----------|------|--------|----------|
| **符号主义** | 离散规则 (`if A then B`) | 人 | 推理引擎搜索 | Lisp、Prolog、专家系统、CYC |
| **贝叶斯网络** | DAG + 条件概率表 | 人画图 + 数据估参数 | 概率推断 (变量消去 / MCMC) | 医疗诊断、垃圾邮件 (Naive Bayes)、HMM |
| **深度学习** | 连续权重矩阵 $W$ | 梯度下降写 | 前向传播 | GPT、Transformer、CodeGPT |

注意"谁写"这一列的演变：**人写得越来越少，数据写得越来越多**。这是从 1956 Dartmouth 会议到 2026 的一条主线——而 2026 年这条线又往前迈了关键一步：**连"架构和训练过程"本身都开始由模型自己写**(自递归训练,见 [§12](#12-2026-前沿自递归训练与-epiplexity历史必然如何走向-mythos--fable-5))。把这张表横着读是空间对比,**竖着读就是一部"人类从 AI 知识生产链上逐步退场"的历史**——退场退到极致,就是 Mythos / Fable 5 这种"模型参与设计自己的后继者"的形态。

---

## 2. 都在解同一个问题：$P(y \mid x)$

不管哪个范式，AI 系统本质都在做条件概率：**给定输入 $x$，预测输出 $y$**。

- 在符号系统里，$P(y \mid x) \in \{0, 1\}$——要么 $x \Rightarrow y$ 是规则的逻辑蕴含，要么不是。
- 在贝叶斯网络里，$P(y \mid x)$ 是图上一系列条件概率表 (CPT) 的乘积。
- 在深度学习里，$P(y \mid x)$ 是一段可微分程序的输出，比如：

```python
# model.py:301
probs = F.softmax(logits, dim=-1)        # P(next_token | context)
```

差别只在于**这个分布是怎么参数化、怎么写入系统的**。讲清楚这一点，三种范式的关系立刻变得简单：**它们是同一个数学问题的三种工程实现**。

---

## 3. 符号主义 AI：人写规则，机器搜索

### 3.1 它长什么样

最经典的形式是 Prolog：

```prolog
parent(tom, bob).
parent(bob, ann).
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

加上一个**推理引擎**（unification + 深度优先搜索 + 回溯），系统就能回答 `?- grandparent(tom, ann).` → `true`。

它的等价 Python 描述：

```python
facts = {("parent", "tom", "bob"), ("parent", "bob", "ann")}

def grandparent(x, z):
    for y in all_entities:
        if ("parent", x, y) in facts and ("parent", y, z) in facts:
            return True
    return False
```

**特征**：

- 知识是**离散的、可枚举的、可读的**——你能逐条审计每一条规则。
- 推理是**确定性**的——同样的输入永远得到同样的输出，且**可证明**（每一步都对应一个推理规则）。
- 没有"训练" 这一步，知识是**人直接写进去的**。

### 3.2 它的死穴：知识获取瓶颈

写 100 条规则容易，写 100 万条规则比这难 10 万倍——因为规则之间会**冲突、嵌套、上下文敏感**。1980 年代的专家系统（MYCIN、CYC）就是死在这里：

- 自然语言里"苹果"是水果还是公司？需要规则。
- "我打开冰箱"——"打开"对冰箱、对文件、对人，三种语义。需要规则。
- 代码里 `x = []` 后面的 `x` 类型是什么？需要规则 + 类型推断 + 上下文……

写规则的人在某一刻会发现：**"规则的规则"是无穷无尽的**。这就是 Hubert Dreyfus 1972 年 *What Computers Can't Do* 里的核心论证，也是 1980s "AI 寒冬" 的直接原因。

### 3.3 但它没死

形式化推理、定理证明、SQL 优化器、编译器、AST 重写、合规规则引擎——这些**强约束、低歧义、可枚举**的领域，符号 AI 至今无可替代。Lean、Coq、Z3 都是符号系统的最新代表。

---

## 4. 贝叶斯网络：图是符号的，参数是概率的

### 4.1 它长什么样

经典的 "Sprinkler / Rain / WetGrass" 图：

```
   Rain ──┐
          ├──→ WetGrass
   Sprinkler ┘

P(Rain)             = 0.2
P(Sprinkler|Rain=T) = 0.01
P(Sprinkler|Rain=F) = 0.4
P(WetGrass | Rain, Sprinkler) = ...   # 4 项的 CPT
```

**联合分布**由图结构因式分解：

```
P(R, S, W) = P(R) · P(S | R) · P(W | R, S)
```

这个**因式分解**就是贝叶斯网络的核心——它**不是从原始 $2^n$ 大小的联合分布里硬学**，而是**用图结构（条件独立性假设）把它压缩到几个小的 CPT**。

### 4.2 它处于符号和神经的中间

| 组件 | 谁定 | 性质 |
|------|------|------|
| 节点 (变量) | 人定 | 符号 |
| 有向边 (依赖结构) | 人画 / 结构学习 | 符号 |
| CPT (条件概率) | 数据估计 | 概率 |
| 推理 | 算法 (变量消去 / MCMC) | 数值 |

这是它**比符号 AI 强的地方**：CPT 可以从数据学，不用人手填。

也是它**比深度学习弱的地方**：图结构必须人画。10 个变量你能画，10 万个 token、12 层、768 维，没人能画出图。

### 4.3 在代码里看一眼

```python
# 一个最小贝叶斯网络的 PyTorch 实现 (示意)
import torch

# CPT 是一个查找表 —— 这就是为什么它是 "半符号"
P_R   = torch.tensor([0.8, 0.2])           # P(R=F), P(R=T)
P_S_R = torch.tensor([[0.6, 0.4],          # P(S | R=F)
                      [0.99, 0.01]])       # P(S | R=T)

# 推断 P(R=T | W=T): 枚举 + 归一化
def infer(R, S, W, evidence):
    ...   # 变量消去 / 精确推断
```

**关键**：CPT 是**离散的、可解释的查找表**——和神经网络里 $W \in \mathbb{R}^{768 \times 768}$ 的稠密矩阵完全不同。

### 4.4 它和深度学习的连接点

- **Naive Bayes** 是最简单的贝叶斯网络（所有特征条件独立给定类别），等价于一个单层 + softmax 的判别模型。
- **HMM**（隐马尔可夫模型）在语音识别上统治了 30 年，本质是一条链状的贝叶斯网络。
- **变分自编码器 (VAE)、扩散模型** 都是把贝叶斯推断和深度学习"焊"在一起——前者用神经网络近似 $q(z \mid x)$，后者用神经网络近似每一步去噪的条件分布。
- 深度学习的 cross-entropy loss 本身就是 **MLE 估计**——贝叶斯派给个先验就成 MAP，再边缘化就成完整贝叶斯。

所以贝叶斯**没有死**，它**变成了深度学习的损失函数和正则化项**。

---

## 5. 深度学习大模型：人写架构，梯度下降写参数

### 5.1 关键的视角转换

CodeGPT 的 `forward` 实际上是一段普通 Python：

```python
# model.py:177-198
def forward(self, idx, targets=None):
    pos = torch.arange(0, t, dtype=torch.long, device=device)
    tok_emb = self.transformer.wte(idx)
    pos_emb = self.transformer.wpe(pos)
    x = self.transformer.drop(tok_emb + pos_emb)
    for block in self.transformer.h:
        x = block(x)
    x = self.transformer.ln_f(x)
    if targets is not None:
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1), ignore_index=-1)
    ...
```

**这段代码就是一段"程序"**——但里面所有矩阵 (`wte.weight`, `wpe.weight`, 每个 `Block` 里的 Q/K/V/MLP/LayerNorm 权重) 都是**待填的空格**，由 `loss.backward()` 配 AdamW 一格一格填出来。

### 5.2 知识在哪里

**知识压在权重里。** 不是 100 万条 `if-else`，而是 ~$1.2 \times 10^8$ 个浮点数（CodeGPT-base 124M 参数）。每个浮点数都不是一条单独的规则，而是**许多规则的微小贡献叠加**。

这种压缩方式有三个革命性后果：

1. **泛化**——训练时没见过的输入也能合理回答，因为权重学到的是**模式的统计结构**，而非样本的查找表。
2. **不可解释**——你不能指着某个权重说"这是关于 Python 缩进的规则"，它弥散在整个矩阵里。这是深度学习最被诟病的一点,值得单独一节,见 [§6](#6-可解释性深度学习最大的软肋以及它现在进展到哪一步)。
3. **不可枚举**——你不能审计每条规则，因为规则数量是连续的、无穷的。

详细推导见 [docs/COMPRESSION_IS_INTELLIGENCE.md](COMPRESSION_IS_INTELLIGENCE.md) 和 [docs/DIFFERENTIABLE_PROGRAMMING.md](DIFFERENTIABLE_PROGRAMMING.md)。

### 5.3 压缩的不是数据,是"为了压缩数据而构造的程序"

这里要把"压缩即智能"这句口号做一个 2026 年的精修,否则容易误解成"权重 = 训练数据的 zip 包"。

权重**不是**训练数据的有损存档。如果只是存档,模型就只会复读,不会泛化。真正发生的是:为了把 PB 级语料压到 124M 个浮点数里,梯度下降被逼着去**构造一个能"生成"这些数据的程序**——这个程序就是 `forward` 里那串可微变换 + 权重(见 [§10](#10-函数式编程视角神经网络是-fgh-的世界) 的函数式视角)。

> **模型学到的不是数据本身,而是为了"压缩"数据而不得不构建的那个复杂的"程序"。这就是智能的本质。**

这句话在 2026 年有了严格的信息论靠山——arXiv 2601.03220 *From Entropy to Epiplexity* (Finzi、Qiu、Jiang、Izmailov、Kolter、Wilson)。它指出经典信息论(Shannon 熵、Kolmogorov 复杂度)都假设**算力无限**的观察者,因此抓不到"对一个**算力有限**的学习器而言,数据里真正可学的结构有多少"。论文用 **epiplexity** 来量化这部分"可学结构",并把它和数据里那段不可压缩的随机噪声(time-bounded entropy)分开。

为什么这对理解权重至关重要,留到 [§12.3](#123-它凭什么不发散不是永动机两道闸) 展开。这里先记住一个结论:**`F.cross_entropy` 这个 loss 表面是在拟合分布,实质是在逼模型把数据里的 epiplexity(可学的程序结构)榨进权重,而对随机噪声部分则学不动也不该学。** 泛化能力,正是"学到了程序、没死记噪声"的副产品。

---

## 6. 可解释性：深度学习最大的软肋,以及它现在进展到哪一步

"大模型是黑盒、不可解释" 是深度学习被诟病最多的一句话,从 Gary Marcus 到 Yann LeCun 早期言论、到 Bender 的 "stochastic parrots"、到欧盟 AI Act 把高风险场景的可解释性写进法规——这是一个**技术问题、监管问题、信任问题三位一体**的批评。

但这句话本身需要拆开看。"不可解释" 包含三个完全不同层次的诉求,深度学习在三个层次上的现状差别极大;而过去三年(2022-2025)**机制可解释性 (mechanistic interpretability)** 这个新方向的进展,已经把"完全黑盒"的判决推翻了一大半。这一节把现状钉到具体的工具和论文上。

### 6.1 "不可解释" 具体指什么——三个层次

| 层次 | 问的是什么 | 深度学习现状 |
|------|------------|--------------|
| **行为层 (functional)** | 给定输入,能不能预测/审计输出? | 中等。可 benchmark、可写测试,但长尾分布无法穷举 |
| **机制层 (mechanistic)** | 模型内部是怎么算出这个输出的? | 弱→改善中。SAE、circuits 已能局部还原 |
| **概念层 (semantic)** | 模型"懂"了什么概念? | 弱→改善中。SAE 找到了百万级单义特征 |

这三层在符号 AI 里都是**默认完整的**(每一步都对应一条规则);在深度学习里都需要**额外工具**来逆向工程。这不是因为深度学习"原理不清楚"——`forward` 的每一行 PyTorch 都摆在你面前——而是因为**权重里压了什么概念**这件事,没法从 `model.py` 的代码读出来。

### 6.2 为什么神经网络天生难解释——四个根源

#### 根源 ①：分布式表征 (Distributed representation)

一个概念**不对应一个神经元**,而是对应**一个方向 / 一个子空间**。

```python
# h: torch.Tensor of shape (768,) —— 某层的 hidden state
# "这段是 Python 函数定义" 这个概念
#   ≈ h 在某个方向 d_func ∈ R^768 上的投影:
#     concept_score = h @ d_func
# 
# 这个 d_func 不是任何一个 weight,是 W 的列向量的某种线性组合
```

你拿放大镜看 `wte.weight[100]` 这一个浮点数,它什么都不"代表"——它是 50304×768 矩阵里的一格,只在和其他 76799 格协同的时候才有意义。

#### 根源 ②：多义性 (Polysemanticity)

同一个神经元在不同输入下响应**完全不相关**的概念——可能既响应 "Python 缩进" 又响应 "JavaScript 大括号" 又响应 "第三人称代词"。这不是 bug,是 SGD 找到的最优解:**用 768 维空间装下远超 768 个特征**。

#### 根源 ③：叠加 (Superposition)

Anthropic 2022 *Toy Models of Superposition* 论文是关键:模型在 $d$ 维空间里编码了远超 $d$ 个特征,用**近似正交**(而非严格正交)的方向。这就是为什么单看一个神经元/单个权重看不出意义——**意义被刻意打散了**。

```
理想 (one-hot)           现实 (superposition)
n features in n dims     n features in d<<n dims
单义,可读                多义,需要 SAE 解码
```

#### 根源 ④：12 层非线性组合

```python
# model.py:103-104 —— 残差 + 非线性
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

每个 Block 都把信号加进残差流(residual stream),早期层的信息一路 "广播" 到末端。要追溯 "答案 X 是因为第 3 层第 5 个 head 在 token Y 上做了什么" ——这是一条跨 12 层的因果链,而中间每层都有 12 个 head 同时在写。组合爆炸是机制层不可解释的直接来源。

### 6.3 我们目前到底能解释什么——五条具体进展

机制可解释性不是一个抽象口号,它在过去三年有**具体可复现的成果**。从弱到强:

#### 进展 ①：注意力可视化 (Attention visualization)

最早、最简单。把 `model.py:68` 的 `att = F.softmax(att, dim=-1)` 这个权重矩阵 hook 出来画热图,能看到代码块边界、变量引用、括号匹配等模式。

**陷阱**:Bibal et al. 2022 *Is Attention Explanation?* 证明 attention weights **不等于** "解释"——在很多任务上你能找到完全不同的 attention 分布给出同样的预测。所以 attention map **可看,但单独不可信**。

#### 进展 ②:线性探测 (Probing)

在某层的 hidden state 上训一个**轻量线性分类器**,看它能不能预测某个概念。

```python
# 思路 (CodeGPT 上可立刻做):
# 1. 跑一段验证集,在第 N 层 hook 出 hidden states H ∈ R^{N_tokens × 768}
# 2. 给每个 token 标 "在 docstring 内 / 函数体内 / 注释内"
# 3. 训 logistic regression: f(h) → label
# 4. 准确率 90%+ 说明这一层"知道"这个概念
```

Hewitt & Manning 2019 用这套方法证明 BERT 中层学到了句法树。**局限**:这是相关性不是机制——分类器读得出,不代表模型内部用得上。

#### 进展 ③:Logit lens / Tuned lens

把每一层的 hidden state 直接乘 `lm_head` 看 next-token 预测。

```python
# CodeGPT 上,基于 model.py:177-198 的 forward,加几行就能跑:
preds_per_layer = []
for block in self.transformer.h:
    x = block(x)
    # 用最终的 ln_f + lm_head 直接 "翻译" 这一层
    logits_here = self.lm_head(self.transformer.ln_f(x))
    preds_per_layer.append(logits_here.argmax(-1))
# 看预测是从哪一层稳定下来的
```

经验上,**事实回忆通常在中后层定型,语法/格式在早层定型**。这给了模型内部计算流程一个粗粒度的"沙漏图"。

#### 进展 ④:Circuit 分析 (Anthropic / Olah 流派)

不再问"这层学了什么",而是问 **"完成任务 T 用到了哪些 head 的协作?"**——直接画出电路图。已找到的具体电路:

- **Induction heads** (Olsson et al. 2022):两个 head 协同实现 "看到 `[A][B] ... [A]` 就预测 `[B]`" 的模式补全。**这是 in-context learning 的最小机械原型**——现在大家相信 GPT 的 ICL 能力,从这个 head 开始
- **IOI circuit** (Wang et al. 2022):GPT-2 small 里负责 "Mary and John went to the store, John gave a drink to ___" 这类间接宾语任务的 **26 个 head 的精确协作图**
- **ROME / MEMIT** (Meng et al. 2022/2023):证明 "巴黎是法国首都" 这个事实**储存在某个具体 MLP 层的某几行权重里**,可以单独编辑——把 "巴黎" 改成 "罗马",模型就开始坚信法国首都是罗马

这把"知识在权重里"从模糊比喻变成了**可定位、可编辑**的工程操作。

#### 进展 ⑤:稀疏自编码器 (SAE) —— 2023 年突破

最近三年最大的进展。思路:**把多义神经元解耦成单义特征**。

```python
# 在某层 hidden state h ∈ R^{768} 上训一个 SAE:
# encoder: h -> features ∈ R^{65536}  (大词典,但稀疏)
# decoder: features -> h_hat
# loss = ||h - h_hat||^2 + λ * ||features||_1   # L1 强制稀疏
# 训完后,每个 feature 几乎单义化:
#   feature[42] 只在 "Python 装饰器" 上激活
#   feature[1337] 只在 "金门大桥" 上激活
#   feature[8888] 只在 "周二" 上激活
```

Anthropic 2023 *Towards Monosemanticity* 在小模型上跑通,2024 *Scaling Monosemanticity* 推到 Claude 3 Sonnet,**找到约 3400 万个单义特征**,并演示了通过把某个 feature 的激活值人为放大/抑制来直接操控行为(著名的 **"金门大桥 Claude"** ——把 Golden Gate Bridge feature 锁住,模型在任何对话里都自我认同为金门大桥)。

这是从 "我们看不懂权重" 到 **"我们可以扫出百万级别可读的概念清单"** 的根本性突破。

### 6.4 一张精确的可解释性对照表

| 维度 | 符号 AI | 贝叶斯网络 | 深度学习 (2026) |
|------|---------|------------|-----------------|
| 行为层可解释 | ✅ 完全 | ✅ 完全 | 🟡 中(可 benchmark) |
| 机制层可解释 | ✅ 每条规则可读 | ✅ D-separation + 变量消去步骤 | 🟡 SAE / circuits 部分还原 |
| 概念层可解释 | ✅ 符号即概念 | ✅ 节点即变量 | 🟡 SAE 已找到百万级单义 feature |
| 反事实推理 | 🟡 改规则即反事实 | ✅ **强 (do-calculus)** | 🟡 activation patching 局部能做 |
| 因果归因 | ✅ 推理链是因果链 | ✅ Pearl 因果阶梯三层全覆盖 | 🟡 ROME 证明可定位+编辑 |
| 法律可审计性 | ✅ 强 | ✅ 强 | ❌→🟡 (合规性正在快速改善) |
| 用户层解释 | ✅ 规则即解释 | 🟡 概率链需要训练才看得懂 | 🟡 chain-of-thought (但可能不忠实) |
| 解释成本 | 几乎零 | 中(图大了就难) | 高(需要 SAE 训练 + 分析工具) |

注意两件事:

- **贝叶斯网络在因果反事实上比符号还强**——Pearl 的 do-calculus 是贝叶斯派给可解释 AI 的最大贡献。如果你的需求是 "改了 X 会怎样",贝叶斯网络至今领先
- **深度学习这一列在 2022 年还几乎全是 ❌,现在大半已经变成 🟡**。这条改善曲线还在快速往下走

### 6.5 在 CodeGPT 上能做什么——五个立刻能上手的实战

虽然这个 repo 没集成 SAE,但已有架构上**今天就能跑**这五件事:

| # | 工具 | 在哪挂钩 | 能看到什么 |
|---|------|---------|------------|
| 1 | 注意力可视化 | `model.py:66-68` 的非 flash 路径,把 `att` hook 出来 | 缩进对齐、变量引用、括号配对 |
| 2 | Logit lens | `model.py:177-198` 的 `for block in ...` 循环里每层接一个 `lm_head(ln_f(x))` | 预测从哪一层稳定下来 |
| 3 | 线性探测 | 取 hidden state,训 LR 分类器 | "在 docstring 内"、"嵌套深度"、"是否在字符串内" 等可读概念 |
| 4 | Head ablation | 在 `Block.attn` 里把某个 head 的输出置零,看 val loss 变化 | 哪些 head 对代码补全任务关键 |
| 5 | ROME 风格编辑 | `lm_head.weight` 是 tied 的 `wte.weight`(`model.py:158-159`),定位到某 token embedding 直接改 | 观察生成漂移,证明知识在哪 |

这些都不是研究前沿,是 **2026 年任何团队都能上手** 的入门项目——也是 "可解释性正在工程化" 的直接体现。

### 6.6 哲学层面——"不可解释" 这个标签被夸大了多少

三个反问把这个批评放回正确尺度:

**反问 ①:人类自己可解释吗?**

你能解释你刚才为什么用 "诟病" 而不是 "批评" 吗?为什么这一段话先讲反问 ① 再讲反问 ②?神经科学到现在还没真正搞清楚一个具体想法是怎么在大脑里形成的。我们对 "人是可解释的" 的信念,主要靠**事后合理化** (post-hoc rationalization)——这和 LLM 的 chain-of-thought 在性质上几乎一样。CoT 甚至可能**不忠实** (Turpin et al. 2023):模型给出的理由可能不是它实际计算的路径。**人类的 CoT 同样不忠实**,这一点心理学研究了几十年(Nisbett & Wilson 1977 经典论文)。

**反问 ②:"可解释" 是二值还是连续?**

没有任何系统是 100% 可解释的(连 Linux 内核都没有人完整懂——它有 3000 万行代码,十几个子系统的交互边界没人能全握在脑子里)。也没有任何系统是 100% 不可解释的——CodeGPT 的 `forward` 每一行 PyTorch 都印在屏幕上。**差别是程度,不是二值**。

**反问 ③:解释给谁看?**

| 受众 | 想要的解释 |
|------|-----------|
| 监管机构 | 决策依据、可审计、可申诉 |
| 模型开发者 | 哪个 head/feature 在什么时候激活 |
| 终端用户 | 自然语言为什么这样回答 |
| 安全研究员 | 越狱路径、对抗样本来源 |

SAE 的 feature 给开发者看是金矿,给用户看就是天书。**"可解释性" 不是一个统一目标,是四个不同问题**。

### 6.7 一句话总结

> **可解释性不是深度学习的死罪,而是它当前最值得投入的研究方向之一。** 符号 AI 的 "完全可解释" 是因为它表达力有限(规则就那么多);深度学习的 "难解释" 是因为它表达力近乎无限(连续函数空间稠密)——两者其实是同一枚硬币的两面。
>
> 把符号 AI 的可解释性当 baseline 去要求深度学习,是用一个范式的标准评判另一个范式。更合理的姿势是:**用符号/贝叶斯的工具去逆向工程深度网络** ——这正是 SAE / circuits / ROME / probing 在做的事——以"神经-符号混合解释" 把深度学习从黑盒往灰盒推。

这条曲线已经走了三年,远未到尽头。

---

## 7. "训练数据 = 一堆可执行的符号代码"——这个类比对在哪、错在哪

现在回到你的原问题。让我们逐条拆。

### 7.1 对的部分（70%）

**类比的核心直觉是正确的：训练数据确实定义了系统的行为，正如代码定义了程序的行为。**

- 改训练数据 → 改模型行为，等价于"改代码 → 改程序行为"。
- LLM 的 in-context learning 让人觉得它在"执行"prompt 里的指令——这确实有几分像把 prompt 当代码。
- "Compression is intelligence" 的视角下，权重 $W$ 是训练数据的**有损压缩**——你也可以反向理解为"训练数据 = 解压后展开的所有规则"。

### 7.2 错的部分（30%，但很关键）

#### 错位 ①：硬规则 vs 软统计

符号代码：

```python
if x > 0:
    return "positive"
```

——只要 `x > 0`，**100%** 返回 `"positive"`。

训练数据隐含的"规则"：

```python
# 这其实是 P(token | context) 的一行样本
context: "def fibonacci(n):\n    "
next_token: "if"   # 出现在训练集 60% 的时间
next_token: "return"  # 25%
next_token: "n =" # 5%
...
```

模型不是从这些样本里**抽取出**确定性的 `if-then` 规则，而是**插值**出一个**连续的概率分布**。这就是为什么同样的 prompt 每次生成不一样——这在符号系统里是不可能的。

#### 错位 ②：执行什么

| | 符号系统执行 | LLM 执行 |
|--|--|--|
| 输入 | 用户写的代码 | 用户的 prompt |
| 执行什么 | 用户写的代码 | **从数据中统计提取的隐式分布** |
| 谁写规则 | 用户 | **梯度下降** (用户只写架构) |

这是关键：**符号系统执行的是"用户的意图"；LLM 执行的是"训练集的平均意图"**。这就是为什么 LLM 会幻觉、会偏向训练数据的偏见、会拒绝训练时学过的"敏感"内容——它执行的根本不是你的 prompt，是它对"在这种 prompt 下，训练分布会怎么响应"的估计。

#### 错位 ③：可枚举 vs 不可枚举

100 万条 Prolog 规则你**理论上能逐条读完**。124M 个浮点数你不能逐个理解——它们之间存在大量的非线性叠加。你无法把权重"反编译"成一组 `if-else`（这个研究方向叫**机制可解释性 / mechanistic interpretability**，目前能解释的只是个别 head 的功能，远未做到整体反编译）。

#### 错位 ④：训练数据本身不是规则集

训练数据是**例子**，不是规则。规则在哪里？**规则是梯度下降从例子里"归纳"出来的，存在权重里**。这是从亚里士多德到 Hume 都在讨论的"归纳问题"——例子和规则不是同一种东西。

### 7.3 一个更精确的类比

如果一定要用编程类比，**更准确的版本**是：

> **训练数据是规约 (specification)，权重是编译产物 (compiled binary)，前向传播是执行。SGD 就是编译器。**

这就和 [docs/DIFFERENTIABLE_PROGRAMMING.md](DIFFERENTIABLE_PROGRAMMING.md) 里 LeCun 的"可微分编程"视角接上了：训练 = 编译，推理 = 执行。

但即使这个类比也有局限：传统编译器是**确定性**的，同样的源码编译出同样的二进制；SGD 是**随机**的，不同的初始化、不同的 batch 顺序会编译出不同的权重——但所有这些权重在训练分布上的行为大致等价。这是符号编程里没有的现象。

### 7.4 更深一层:编译产物本身是一段"程序",而且编译能凭空造信息

把 §7.3 的类比再往前推一步,就接上了 [§5.3](#53-压缩的不是数据是为了压缩数据而构造的程序) 的 epiplexity 视角。"编译产物 = 权重"这话还不够准——更准的说法是:**编译产物是一段程序**(`forward` + 权重),而 SGD 编译的目标函数,是让这段程序能以最短的描述长度"生成"训练数据。换句话说,模型存的从来不是数据,是**重新生成数据所需的那套规则**。错位 ④ 说的"规则是归纳出来的",在这里有了精确版本:**归纳的产物是一个程序,程序的复杂度上界就是数据的 epiplexity**。

这里还能解决一个反直觉的现象。经典信息论有条铁律:**确定性变换不会增加信息**(你对数据做任何可逆/确定性的重排,香农信息量不变)。但 epiplexity 论文证明:对**算力有限**的学习器,**计算可以创造可学信息**——把数据重排、变换,会让原本"藏得太深、学不动"的结构浮到表面,变得可学。

这件事在本仓库里有一个**教科书级的实例**——FIM 变换:

```python
# tokenizer.py:148-198 (apply_fim_transform 的核心)
# 把一段连续 token 切成 prefix / middle / suffix, 再重排成:
#   <|fim_prefix|> P <|fim_suffix|> S <|fim_middle|> M
new_tokens = [prefix_id] + prefix + [suffix_id] + suffix + [middle_id] + middle
```

`apply_fim_transform` 对训练数据做的是一个**纯粹的、确定性的重排**——没加任何新 token,香农信息量一个比特都没多。可它实实在在让模型多学到一种能力:**在已知上下文两端时填中间**(`encode_fim` 推理时才用得上)。这正是 epiplexity 说的"**计算/重排创造可学信息**":同一批数据,换个排列顺序喂,**对算力有限的 Transformer 而言可学的程序结构变多了**。`train.py:get_batch` 里那句"50% 概率做 FIM"——本质就是一个朴素版的"用计算给数据增容 epiplexity"的操作。

这条原理一路放大,就是 [docs/SYNTHETIC_DATA.md](SYNTHETIC_DATA.md) 里的合成数据、以及 [§11](#11-神经-符号混合现代系统都不是纯种) 的 verifier-based RL 自博弈——**用模型自己的计算,生成比原始语料"程序结构更复杂"的新数据**,再喂回去训练。这也是为什么 §12 的自递归训练在理论上站得住:它不是在违反"天下没有免费的信息",而是在兑现 epiplexity 这张支票。

---

## 8. 三种范式的对照表

| 维度 | 符号主义 AI | 贝叶斯网络 | 深度学习大模型 |
|------|-------------|------------|----------------|
| **知识载体** | 离散规则 / 事实 | DAG + CPT | 连续权重矩阵 |
| **知识来源** | 人手写 | 人画图 + 数据估参数 | 数据 (+ 人定架构) |
| **推理形式** | 符号搜索 / 归结 | 概率推断 | 矩阵乘法 + 非线性 |
| **可解释性** | 强 (每步可追溯) | 中 (节点+CPT 可读) | 弱 (权重弥散) |
| **数据效率** | 0 例样本即可 (但需大量人工) | 中等 | 极差 (需海量数据) |
| **泛化能力** | 弱 (规则边界硬) | 中 (受图结构约束) | 强 (连续插值) |
| **计算复杂度** | 推理可能 NP-hard | 精确推断 NP-hard | 推理 $O(n)$ 一次前向 |
| **不确定性** | 没有 (二值) | 一等公民 | 隐式 (在 logits 分布里) |
| **可组合性** | 强 (规则即模块) | 中 (子图可拼接) | 弱 (权重不可拆) |
| **知识更新** | 编辑规则即可 | 重新估 CPT | 重新训练或 fine-tune |
| **典型失败模式** | 知识获取瓶颈 | 图结构错 → 全错 | 幻觉 / 训练分布偏移 |
| **形式保证** | 可证明正确 | 概率边界 | 几乎没有 |
| **CodeGPT 例子** | `<|fim_prefix|>` 等特殊 token 的硬模板 | (无显式贝叶斯网络) | `forward` 里所有 `nn.Linear` |

注意一件事：**没有任何一行说"X 是赢家"**。三个范式各自在自己擅长的边界内是无可替代的。

---

## 9. 在 CodeGPT 里看见三个范式的影子

虽然 CodeGPT 是一个**纯神经**模型，但它的代码里仍然能找到三种范式的痕迹：

### 9.1 符号成分

`tokenizer.py` 里硬编码的特殊 token——这是**纯符号**：

```python
# tokenizer.py:148-198 的 apply_fim_transform
prefix_id = SPECIAL_TOKENS["<|fim_prefix|>"]   # 50257
middle_id = SPECIAL_TOKENS["<|fim_middle|>"]   # 50258
suffix_id = SPECIAL_TOKENS["<|fim_suffix|>"]   # 50259

# PSM 模板：<|fim_prefix|> P <|fim_suffix|> S <|fim_middle|> M
new_tokens = [prefix_id] + prefix + [suffix_id] + suffix + [middle_id] + middle
```

这是**人写的、确定性的、离散的格式协议**——和 Prolog 里的 `parent(X, Y) :- ...` 在性质上是一样的。模型学不出"PSM 模板"——这个模板是由人**作为不可学习的协议**强加给模型的。`stop_tokens`、`lang:python` 等也都是同类——它们是**架构里的符号骨架**。

### 9.2 概率成分

```python
# model.py:301-302
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

这是一个**条件概率分布的采样**——和贝叶斯网络的边缘 / 条件推断在数学上是同一类操作。CodeGPT 没有显式的图结构，但它每个 token 上的 next-token 分布就是一个超大规模的隐式贝叶斯模型——只是这个 CPT 是 50304 个值、由神经网络给出，不再是查表。

### 9.3 神经成分

12 层 Block × (注意力 + MLP) × 12 个 head = 124M 参数：

```python
# model.py:102-105
def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x
```

这里的所有"知识"都在 `self.attn`、`self.mlp` 的权重矩阵里，由 `train.py` 的 `F.cross_entropy` 通过梯度下降一格一格填进去。**这部分完全不可枚举、不可解释、连续可微**——这是纯深度学习的部分。

### 9.4 三层夹心结构

```
┌──────────────────────────────────────┐
│  符号层  特殊 token / chat template    │ ← 人写的协议骨架
│         stop_tokens / FIM 模板         │
├──────────────────────────────────────┤
│  概率层  softmax → multinomial         │ ← 把 logits 解释为分布
├──────────────────────────────────────┤
│  神经层  12×Block, 124M 权重           │ ← 梯度下降学习的部分
└──────────────────────────────────────┘
```

**任何能跑的大模型系统都是这种"三明治"——纯神经的核心，外面包一层概率解释，最外面是符号协议。** 这不是历史遗留，是必然——纯神经模型没法决定"什么时候停止生成"（这是符号问题）、没法决定"采样还是 argmax"（这是概率问题）。

---

## 10. 函数式编程视角：神经网络是 f∘g∘h∘... 的世界

> 在符号 AI、贝叶斯网络、深度学习这三派之外,有一条**横切**了三派的第四股力量被本文档刻意推迟到这里才登场——**函数式编程 (functional programming, FP)**。它不是历史上 1956 年 Dartmouth 大会上独立的一派,而是一种**元思想**:Lambda calculus (Church 1936) 比图灵机晚一年但更纯,Lisp 的 `(f x)` 写法从 1958 年就在了,而 PyTorch / JAX / Keras 今天写的每一段 `forward`,本质上都是 **lambda calculus 的可微分版本**。
>
> 这一节用函数式编程的眼光重新拆一遍神经网络的前向、反向、组合方式。你会看到一个反直觉但极其干净的结论:**深度学习其实是符号主义的一个连续化特例**——把符号从离散原子换成可微张量,把规则从 `if-else` 换成线性映射 + 非线性,但**"组合"这件事本身,是同一种动作**。

### 10.1 前向传播 = 函数组合 (Function Composition)

打开 `model.py:177-198` 的 `forward`:

```python
def forward(self, idx, targets=None):
    ...
    x = self.transformer.drop(tok_emb + pos_emb)   # f₀
    for block in self.transformer.h:                # f₁, f₂, ..., f₁₂
        x = block(x)
    x = self.transformer.ln_f(x)                    # f₁₃
    logits = self.lm_head(x)                        # f₁₄
```

数学上这就是函数组合的链式表达:

$$
\text{logits} \;=\; f_{14} \circ f_{13} \circ f_{12} \circ \cdots \circ f_1 \circ f_0\,(x)
$$

在 Lisp 里完全等价的写法:

```lisp
(lm-head (ln-f (block-12 (block-11 ... (block-1 (drop (+ tok-emb pos-emb)))))))
```

在 Clojure 里用线程宏 `->>` 让顺序和 Python `for` 循环对齐,可读性更高:

```clojure
(->> (+ tok-emb pos-emb) drop
     block-1 block-2 ... block-12
     ln-f lm-head)
```

**`for block in self.transformer.h: x = block(x)` 本质就是 fold / reduce**:

```python
from functools import reduce
x = reduce(lambda acc, block: block(acc), self.transformer.h, x_initial)
```

`nn.Sequential` 把这个 fold 写得更白:

```python
model = nn.Sequential(emb, drop, block_1, ..., block_12, ln_f, lm_head)
out = model(x)   # 内部就是 reduce
```

每个 `Block` 里 (`model.py:102-105`) 又是同样的组合套娃:

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))     # x + (attn ∘ ln_1)(x)
    x = x + self.mlp(self.ln_2(x))      # x + (mlp ∘ ln_2)(x)
    return x
```

**残差连接 `x + f(x)`** 在 FP 视角下就是 $\text{block} = \mathrm{id} + f$——用**函数加法**让 $f$ 起步时接近恒等映射。这是何凯明 2015 *Deep Residual Learning* 的关键观察,而它的 FP 描述异常干净:**ResNet = 把恒等函数和待学习的扰动函数相加**。

### 10.2 Keras Functional API:把函数组合写在脸上

PyTorch 的 `for block in ...` 是命令式风格,函数组合藏在循环里。**Keras Functional API 直接把"层即函数"写成 API**:

```python
# Keras Functional API —— 这就是带形状信息的 lambda calculus
inputs = Input(shape=(seq_len,))
x = Embedding(vocab_size, n_embd)(inputs)
x = Dropout(0.1)(x)
for _ in range(n_layer):
    x = TransformerBlock(n_head, n_embd)(x)
x = LayerNorm()(x)
outputs = Dense(vocab_size)(x)
model = Model(inputs, outputs)
```

看 `Embedding(vocab_size, n_embd)(inputs)` 这一行——`Embedding(...)` 返回一个**可调用对象** (callable),再被 `(inputs)` 调用。这就是**柯里化 (currying)**:

```
Embedding : (vocab_size, n_embd) → (Tensor → Tensor)
                                        ↑
                                  这是个待应用输入的函数
```

`x = Dense(64)(x)` 等价于 Haskell 里的 `x' = dense 64 x`,等价于 Lisp 里的 `(setq x ((dense 64) x))`。**Keras 的设计哲学就是"层即函数"**——François Chollet 在 *Deep Learning with Python* 第二版里专门论证 "DL is composition of differentiable functions"——他把这条话刻进了 Keras 的核心 API。

更极致的是 **JAX / Flax**,它把"模型即纯函数"做到底:

```python
# JAX/Flax 风格 —— 模型完全是纯函数
def forward(params, x):
    x = embedding(params['emb'], x)
    for i in range(n_layer):
        x = transformer_block(params['blocks'][i], x)
    return lm_head(params['lm_head'], x)
```

参数**显式作为函数第一参数**传入,完全没有 `self`,没有隐藏状态——这是数学意义上严格的纯函数。这就是为什么 `jax.grad(forward)` 能直接对参数求导:函数有梯度的前提是它**真的是函数**。

### 10.3 Transducer 思想:变换从数据里解耦

**Transducer** 是 Rich Hickey 在 Clojure 1.7 (2015) 引入的关键概念:**把"做什么变换"和"在什么数据上变换"彻底分开**。一个 transducer 描述"如何变",可以在任何 reducible 数据源上应用。

```clojure
;; Clojure transducer
(def xf (comp (map inc) (filter even?) (take 5)))
;; xf 只描述变换,跟数据完全无关
(transduce xf + 0 (range 100))        ;; 应用到无限序列
(transduce xf conj [] my-channel)     ;; 应用到 channel
```

**`nn.ModuleList` 正是神经网络版的 transducer 集合**:

```python
# model.py:148 —— 一组与数据无关的变换
self.h = nn.ModuleList([Block(config) for _ in range(n_layer)])

# 应用时才注入数据流
for block in self.h:
    x = block(x)
```

`self.h` 本身**不含任何数据流转的逻辑**,只是 12 个独立的可调用变换——就像 `(comp f₁ f₂ ... f₁₂)`。这有三个直接好处:

1. **可移植** —— `self.h` 可以应用到任何 batch / seq_len 上,变换本身不变。
2. **可插拔** —— `self.h[3] = NewBlock(...)` 立刻替换中间一层,其他层不受影响——这是 Hugging Face PEFT / LoRA 等技术的底层前提。
3. **可观测** —— 逐层 hook 出 hidden state(也就是 [§6.3 logit lens](#6-可解释性深度学习最大的软肋以及它现在进展到哪一步) 的实现基础)——因为每一层都是独立的纯变换,中间值可以被任意截取。

PyTorch 2.0 引入的 `torch.compile` 把这思想推到极致——它**先把 forward 跟踪成一个静态计算图 (纯函数!),再做整图融合优化**。前提条件正是:**你的 forward 已经是函数组合,没有隐藏副作用**。

### 10.4 Map / Reduce 无处不在

神经网络几乎每一个原语都是 map 或 reduce:

| 操作 | 函数式对应 | 在 CodeGPT 哪里 |
|------|-----------|-----------------|
| Batch 维并行 | `map` over B | `forward` 第一维 (`model.py:179`) |
| Token 维并行 | `map` over T | `tok_emb = wte(idx)` 每个位置独立 (`model.py:183`) |
| Linear / GELU | `map` over hidden dim | `MLP.forward` (`model.py:85-90`) |
| Softmax | `map` + `reduce`(归一化) | `model.py:68, 301` |
| Attention 加权求和 | `reduce` over keys | `att @ v` (`model.py:70`) |
| LayerNorm | `reduce`(mean/var) + `map`(scale) | `LayerNorm.forward` (`model.py:27-28`) |
| Cross-entropy loss | `map`(per-token NLL) + `reduce`(mean) | `model.py:192` |
| `sum(p.numel() for p in ...)` | `reduce` | `get_num_params` (`model.py:164`) |

**Attention 是教科书级的 map-reduce**:

```python
# model.py:66-70  (q, k, v: shape [B, H, T, D])
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))   # map: 每对 (q_i, k_j) 求点积
att = att.masked_fill(mask, float('-inf'))
att = F.softmax(att, dim=-1)                              # reduce + 归一化
y = att @ v                                               # reduce: 加权求和
```

这正是 Google 2004 *MapReduce* 论文的内核:**map 阶段独立计算,reduce 阶段聚合**。当年用来在 PB 级日志上算词频,今天用来在 (B, H, T, T) 张量上算注意力——**同样的范式,只换了硬件 (CPU 集群 → GPU SIMD)**。

`torch.vmap`、`jax.vmap`、TensorFlow 的 `tf.map_fn` 都是显式的 **map** 算子——它们让你写"单样本逻辑",由框架自动 lift 到 batch 维。这和 Lisp 的 `(mapcar f xs)`、Haskell 的 `fmap f xs`、Clojure 的 `(map f xs)` 是**同一个函数式原语**,只是参数从列表换成了张量。

### 10.5 高阶函数:Optimizer 与 Module 都是

**高阶函数** = 接受函数 / 返回函数的函数。神经网络里到处都是:

```python
# model.py:240 —— optimizer 接受 model 的参数 (本质是接受函数的内部状态)
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, ...)

# 训练循环: 模型(函数) + 数据 → loss → 梯度 → 更新模型
for step in range(max_iters):
    loss = model(x, y)[1]   # 调用函数
    loss.backward()         # 对函数求导
    optimizer.step()        # 用梯度更新函数(的参数)
```

**Optimizer 在范畴论意义上是一个 endofunctor**:它接受 (Model, Gradient) 返回 Model'——也就是 $\text{Optimizer}: \text{Model} \to \text{Model}$,把函数空间映回它自己。

`nn.Module` 也是高阶——它**接受子 module 作为构造参数,组合成更大的 module**:

```python
# model.py:93-100 —— Block 是高阶 module
class Block(nn.Module):
    def __init__(self, config):
        self.ln_1 = LayerNorm(...)        # 子函数
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(...)
        self.mlp = MLP(config)
```

`CodeGPT` 整体就是一棵**函数树** (function tree):

```
CodeGPT
├── transformer (ModuleDict)
│   ├── wte (Embedding)
│   ├── wpe (Embedding)
│   ├── drop (Dropout)
│   ├── h (ModuleList × 12)
│   │   └── Block
│   │       ├── ln_1 (LayerNorm)
│   │       ├── attn (CausalSelfAttention)
│   │       ├── ln_2 (LayerNorm)
│   │       └── mlp (MLP)
│   └── ln_f (LayerNorm)
└── lm_head (Linear)
```

`named_parameters()` (`model.py:215`) 就是这棵树的中序遍历。这种结构在 Haskell / OCaml 的 functor 里有严格对应——`nn.Module` 本质上是个 **"applicative functor over differentiable maps"**。

### 10.6 不可变性 (Immutability) 与纯函数:Autograd 的底层假设

**Autograd 工作的根本前提是 `forward` 接近纯函数**——每个张量操作不应有隐藏副作用,否则反向传播的计算图就断了。这就是为什么 PyTorch 文档反复警告:

```python
x = self.linear(x)
x.relu_()        # ❌ in-place,可能破坏 autograd
x = x.relu()     # ✅ 返回新张量,纯函数
```

PyTorch 的 autograd engine 在跑 backward 时,**需要 forward 时每一步的中间张量都还存在**——in-place 操作会覆盖这些张量,直接报错 `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`。

JAX 走得更极端——**完全禁止 in-place**,所有张量天然不可变。这让:

- `jax.jit` 能做激进的整图融合优化
- `jax.grad` 能自动微分任何 Python 函数
- `jax.pmap` / `jax.vmap` 能轻松并行/批量化
- `jax.checkpoint` 能用重算换显存(只有纯函数才能"重新跑一遍")

——**全部依赖于纯函数性**。这一点和 Haskell / Clojure 的不可变数据结构哲学**完全一致**:**只要变换是纯的,组合就是无限自由的**。

`model.py` 的 `forward` 仔细看,**没有任何一行有副作用**:`tok_emb = self.wte(idx)`、`x = self.drop(...)`、`x = block(x)` 全部是"读取参数 + 返回新张量"。这就是为什么 PyTorch 的 hook 机制 (`register_forward_hook`)、`torch.compile`、反向传播 都能干干净净地工作——CodeGPT 的 forward 已经是 FP 风格了,只是用命令式句法写出来。

### 10.7 反向传播 = Chain Rule on Composed Functions

最优美的对应在这里。前向是 $y = (f_n \circ \cdots \circ f_1)(x)$,反向就是 chain rule 的层层乘积:

$$
\frac{\partial L}{\partial x} \;=\; \frac{\partial L}{\partial y_n} \cdot \frac{\partial y_n}{\partial y_{n-1}} \cdot \frac{\partial y_{n-1}}{\partial y_{n-2}} \,\cdots\, \frac{\partial y_1}{\partial x}
$$

这就是函数式编程里 **"组合 (composition) 的求导是导数的组合 (composition)"** 这条最基本的等式。PyTorch 的 autograd 把它做成了机械过程:

```python
loss.backward()   # 对 (f_n ∘ ... ∘ f_1) 用 chain rule 求 grad
```

更深一层:**自动微分 (automatic differentiation)** 本身就是一种 **transformation on functions**——给定 $f$,AD 返回 $f'$。这是个**高阶函数**,在 JAX 里写得最白:

```python
import jax
grad_fn = jax.grad(loss_fn)   # 高阶函数: 函数 -> 它的梯度函数
grads = grad_fn(params, x, y)
```

`jax.grad` 是个 **functor**——它把函数空间里的对象 (你的 `loss_fn`) 映到另一个函数空间里的对象 (它的梯度函数)。这是范畴论的语言,但 1958 年 McCarthy 设计 Lisp 时已经隐隐有这种想法了——**程序可以是数据,函数可以被变换**。

`vmap`、`pmap`、`grad`、`jit` 全是 JAX 的"函数变换器" (function transformations)。**深度学习框架的现代设计,正在快速向"高阶函数 + 函数变换"这个方向收敛**。

### 10.8 符号主义 ↔ 函数式 ↔ 逻辑式:三位一体

回到本文档的主线。把这几派放在历史轴上:

| 年份 | 思想 | 代表 | 与神经网络的连接 |
|------|------|------|-----------------|
| 1936 | Lambda calculus | Church | `forward` 是 λ 表达式的可微版本 |
| 1936 | 图灵机 | Turing | 等价于 lambda calculus (Church-Turing thesis) |
| 1958 | Lisp | McCarthy | 第一个让"程序 = 数据";Transformer 也把 token 当数据流 |
| 1972 | Prolog | Colmerauer | 逻辑式 = 反向函数式: 给输出找输入 |
| 1980s | Curry-Howard 对应 | 数理逻辑 | **类型 = 命题,程序 = 证明**——张量形状是最弱的"类型" |
| 1986 | Backprop | Rumelhart/Hinton | 函数组合的 chain rule 自动化 |
| 2015 | Keras Functional API | Chollet | 把"层即函数"明确写进 API |
| 2015 | Clojure Transducer | Hickey | 变换与数据解耦——`nn.ModuleList` 的精神先祖 |
| 2018 | JAX | Google | 纯函数式神经网络 + 函数变换 |
| 2022 | torch.compile | Meta | 命令式 PyTorch 被自动 trace 成函数式 IR 做优化 |

**逻辑式编程 (Prolog) 是函数式编程的"反向"**:函数式说"给我 x,我返回 f(x)";逻辑式说"给我 y,我找所有满足 f(x)=y 的 x"。神经网络的**训练过程正是逻辑式风格**——给定目标 $y$,反向找出能产出它的参数 $\theta$:

$$
\theta^* \;=\; \arg\min_\theta \;\sum_i \;\|f_\theta(x_i) - y_i\|^2
$$

这是 **inverse function composition**。SGD 是这个 inverse 的**近似数值求解器**,而 Prolog 的 unification + backtracking 是 inverse 的**精确符号求解器**(代价是搜索空间爆炸)。**两者是同一个"反向解析"问题的两种实现**——一个用连续优化,一个用离散搜索。

更深一层是 **Curry-Howard 对应**:**类型即命题,程序即证明**。在依赖类型语言 (Coq, Agda, Lean) 里,你写一个类型为 `forall n, n + 0 = n` 的程序,这个程序的存在本身就是该命题的证明。神经网络里有什么对应物?——**张量形状就是最弱的类型**,而 `assert config.n_embd % config.n_head == 0` (`model.py:35`) 之类是最低级的"形状证明"。MIT 的 Brent Yorgey、CMU 的 David Spivak、Conal Elliott 等人在尝试把更强的依赖类型引入深度学习框架(参见 *Categorical Deep Learning* 项目),这是一个非常前沿的方向。

### 10.9 在 CodeGPT 里找全函数式的影子

| FP 概念 | CodeGPT 实例 | 文件:行 |
|---------|--------------|---------|
| 函数组合 | `forward` 里 12 层 Block 串联 | `model.py:186-187` |
| Fold / Reduce | `for block in self.transformer.h: x = block(x)` | `model.py:186-187` |
| 高阶函数 | `self.apply(self._init_weights)` 把函数作为参数传 | `model.py:155` |
| 柯里化 | `nn.Linear(768, 768)` 返回可调用对象 | `model.py:36-37` |
| Map (并行变换) | Batch 维 B、序列维 T 都是隐式 map | `model.py:179, 52` |
| Reduce (聚合) | `F.cross_entropy` 的 mean reduction | `model.py:192` |
| 不可变性 | autograd 依赖不修改张量,forward 全是纯变换 | 整个 forward |
| 纯函数 | `LayerNorm.forward` 只读 weight/bias,不改 state | `model.py:27-28` |
| Functor 容器 | `nn.ModuleList` / `nn.ModuleDict` 把变换"装入" | `model.py:144-150` |
| Transducer 思想 | `self.transformer.h` 是与数据无关的变换序列 | `model.py:148` |
| 函数变换 | `torch.compile(model)` 把 model 重写成更快的 model | `train.py` |
| 反向 = chain rule | `loss.backward()` 沿组合链反向求导 | 训练循环 |
| 残差 = id + f | `x = x + self.attn(self.ln_1(x))` | `model.py:103-104` |

`forward` (`model.py:177-198`) 里**没有任何一行有副作用**——所有操作都返回新张量。这就是为什么:

1. PyTorch 的 hook 系统能不破坏原代码地插入观测点 (`register_forward_hook`)——**纯函数允许这种"装饰"**(decorator pattern in FP)。
2. `torch.compile` 能整图重写——**纯函数才有可分析的语义**。
3. 反向传播能自动展开——**chain rule 要求每一步可微分,而可微分性 ≈ 纯函数性**。

### 10.10 一句话总结

> 神经网络看起来是"统计机器",其实是**统计化的 lambda calculus**。`forward` 是函数组合,`backward` 是函数组合的 chain rule,`optimizer.step()` 是函数空间上的高阶变换。
>
> **符号 AI 用离散原子 + `if-else` 组合,神经网络用张量 + 线性映射 + 非线性组合——但"组合"这件事本身是同一种动作**。Keras Functional / JAX 把这层连接做到 API 表面;PyTorch 用命令式句法藏起来但本质未变;`torch.compile` 把命令式偷偷再 trace 回函数式做优化。
>
> 三个范式在编程范式这一层握手:
>
> - **符号** = 用代数变换 / unification 组合
> - **函数式** = 用 λ 抽象 + composition 组合
> - **神经** = 用可微变换 + chain rule 组合
> - **逻辑** = 求解组合的逆映射
>
> **"组合 (composition)" 是 AI 系统永恒的骨架**。任何一种 AI 范式,最终都在回答同一个问题:**怎么把简单原语拼成复杂行为?** 这就是为什么 Lisp 1958 的 `(f (g (h x)))` 和 PyTorch 2026 的 `for block in self.transformer.h: x = block(x)` 看起来不同,实质完全一致——它们都是**函数组合**这条人类思维基本动作的不同方言。

---

## 11. 神经-符号混合：现代系统都不是纯种

回顾一下产业现状：

- **Tool calling / function calling**：模型生成的不是答案，是**符号化的函数调用**，由外部代码（解释器、计算器、搜索引擎）执行——纯神经搞不定算术。
- **Verifier-based RL** (OpenAI o1 / DeepSeek R1)：用**符号验证器**（单元测试通过 / 数学题答案匹配）做 reward，再用 RL 把验证信号反向传到神经权重。这条路线在 2026 年被推到极致——**模型不只用验证器学解题,还用验证器学"怎么改进训练自己的代码"**(自递归训练),详见 [§12](#12-2026-前沿自递归训练与-epiplexity历史必然如何走向-mythos--fable-5)。
- **RAG**：从向量库（神经检索）+ 知识库（符号文档）拼出 prompt，再让模型生成——本质是把"事实查询"留给符号、"语言生成"留给神经。
- **Lean / Coq + LLM**：LLM 提议证明步骤，符号系统验证。神经做"想法"，符号做"裁判"。
- **CodeGPT 的 FIM**：特殊 token (`<|fim_*|>`) 是符号协议，模型在协议骨架上学连续分布——这就是一个最小的 neural-symbolic 设计。

为什么混合？因为**三种范式的强项和弱项几乎完美互补**：

| 任务 | 最适合的范式 | 原因 |
|------|--------------|------|
| 自然语言理解 / 生成 | 神经 | 高维感知、模糊匹配 |
| 数学推理、定理证明 | 符号 | 需要可证明性、零容错 |
| 因果推断、不确定性量化 | 贝叶斯 | 概率原语是一等公民 |
| 长程规划、多步推理 | 神经主导 + 符号兜底 | 神经做候选，符号验证 |
| AST 操作、编译 | 符号 | 离散结构、严格类型 |
| 代码生成 | 神经 + 执行反馈 | 神经写代码，解释器是验证器 |

**未来 AI 不是某一派胜出，而是三派各自占据自己的层。** 这也是 [docs/SYNTHETIC_DATA.md](SYNTHETIC_DATA.md) 里讨论的"执行反馈是最诚实的 reward"——那个反馈天然是符号的（解释器要么报错要么不报错）。

---

## 12. 2026 前沿：自递归训练与 Epiplexity——历史必然如何走向 Mythos / Fable 5

这一节不是"再补两个新名词"。它是把前面十一节早就铺好、却**故意留着没接通的三根线**接到一起:

- [§10.8](#10-函数式编程视角神经网络是-fgh-的世界) 讲了 λ-calculus、讲了 Prolog 是"函数的反向"(给输出找输入),但它**漏掉了 λ-calculus 最核心的那一个原语**——**自指 (self-application):一个函数把自己当参数**。这正是"自递归"里那个"**自**"。
- [§7.3](#73-一个更精确的类比) 把训练讲成"编译":数据是源码,SGD 是编译器,权重是产物。
- [§5.3](#53-压缩的不是数据是为了压缩数据而构造的程序) / epiplexity 讲了:这个编译产物是**为了压缩数据而构造的程序**。

把这三根线拧成一股,只需要问一个问题:**如果把"训练 = 编译"这支箭,掉头指向编译器自己,会怎样?** ——答案就是自递归训练,而它的极限形态,就是 Mythos / Fable 5。下面分四步,外加一节验证:先讲清"自"到底自在哪(§12.1),再追到它的思想起源——Lisp 1958 年就埋下的递归与"代码即数据"两条基因(§12.2),接着讲它凭什么不发散、不违反信息守恒(§12.3),然后讲它为什么**必然**收敛到 Mythos / Fable 5 这种东西(§12.4);最后把镜头从"训练"拉到"agent",看同一副骨架如何在 Pi、loop engineering 这些 agent 设计里第三次显形(§12.5)。

### 12.1 "自"递归:把训练这支箭掉头指向训练自己

先看经典深度学习(也就是 CodeGPT 此刻的样子)。训练是一个函数,吃"训练代码 + 数据",吐一个模型:

```python
M_next = Train(code, data)     # code = 人手写的 train.py / model.py / config, 是常量
```

`code` 在这里是**人写死的常量**——`train.py` 的余弦调度、`apply_fim_transform` 的 50% 概率、`config/*.py` 里的超参,全是人填的。这就是 §1 表里"架构/训练过程"那一格还写着"人"的原因。

自递归训练只改一件事:**让 `code` 不再是人写的常量,而是上一代模型自己算出来的**。

```python
code_next = Improve_by(M, code)     # 改进训练码这件事, 由模型 M 来做
M_next    = Train(code_next, data)  # 再拿改进后的码去训练下一代
```

把两行合并,`M_next` 就成了一个**以"上一代自己"为输入的函数**:

```python
M_next = F(M)     # F = "用 M 改训练码, 再训练一遍" —— 这是一个不动点迭代
```

`M_{n+1} = F(M_n)` 是教科书里的**不动点迭代 (fixed-point iteration)**。而"一个函数吃自己、吐一个改进版的自己",在 [§10.8](#10-函数式编程视角神经网络是-fgh-的世界) 的 λ-calculus 语言里,就是那个被刻意留到现在才登场的原语——**Y 组合子**:

$$
Y\,f = f\,(Y\,f)
$$

§10 讲遍了 `forward` 是函数组合 `f∘g∘h`、`backward` 是组合的 chain rule、`optimizer` 是函数空间上的高阶变换——唯独没碰**自指**。**自递归训练就是 Y 组合子的物理实现**:模型这个函数,被喂进了它自己。

为什么是"**自**"而不只是"自动化 (auto)"?关键在一个本仓库就能看清的事实:**`train.py` 这段源码,本身就落在模型自己的输出分布里。** 模型是在代码上训练的、会写代码,而它会写的代码里**就包括训练它自己的那段**。换句话说——

> 这个函数 `F` 的**定义域,包含了它自己的源代码**。模型既是"被编译的程序",又是"写编译器的程序员"。这种"输出空间套住了自己的来源"的自我缠绕,才是"自"递归的字面含义,也是它和普通自动化调参的本质区别。

落到本仓库,这个不动点循环的每个零件**都已经在 CodeGPT 里了,只差把人换成模型**:

```python
# 当前 (经典): 人在循环外面手动迭代
for trial in human_ideas:                     # ← 人提改动
    code = edit(train_py, config, trial)       #   改 config/*.py 的 lr / loss / 架构
    M    = Train(code, data)                    #   train.py 跑一次
    score = estimate_loss(M)                    # ← train.py 的 verifier, 不会撒谎

# 自递归 (Mythos 形态): 把 human_ideas 换成 M 自己
for _ in range(generations):
    trial = M.propose(read("train.py"))        # ① 模型读自己的训练码, 提改动 (神经)
    if not M.critique(trial): continue         # ② 模型先自我否决一批 (神经+符号先验)
    code  = edit(train_py, config, trial)
    M_new = Train(code, data, budget="small")  # ③ 沙箱里跑 ablation (微型训练)
    if estimate_loss(M_new) < estimate_loss(M):# ④ 符号验证器裁判: 只有真变好才接受
        M = M_new                              #    —— 这一步是整个递归收敛的关键, 见 §12.3
```

这四步正是 Anthropic 2026 年披露的循环(提议→评审→沙箱 ablation→结构化回报),但**它不是新发明的范式,而是把 [§11](#11-神经-符号混合现代系统都不是纯种) 的 verifier-based RL 的作用对象,从"解一道题"升格成"改训练自己的代码"**。`configurator.py` 那条"config 不能引入新 key、只能覆盖已声明默认值"的规矩(CLAUDE.md 里写着),此刻成了**给模型框定的安全提议空间**——模型能拧旋钮,但拧不出仓库没准备好的接口。人,从 `for trial in human_ideas` 这个循环里被请了出去,只留在"审批/一键暂停"的闸门位。

这就是为什么有那两个具体数字:Anthropic 合入生产库的代码 **80%+ 由 Claude 撰写**(2025 年 2 月还是低个位数);而内部那个"让训练代码跑更快"的基准,从 Claude Opus 4 的约 **3×**(2025 年 5 月)涨到 Mythos Preview 的约 **52×**(2026 年 4 月)。**52× 不是模型更会答题了,是这个 `F` 迭代了几轮、模型把"训练它自己"的那段代码越改越快。**

### 12.2 这套"自指"不是新发明:它是 Lisp 1958 年就埋下的两条基因

上面那个 `M_{n+1}=F(M_n)`、那个"函数的定义域套住自己源码"的结构会让人觉得很新——其实它是 **1958 年 Lisp 就埋下的两条基因**,在 2026 年长成了 Mythos / Fable 5。[§10.8](#10-函数式编程视角神经网络是-fgh-的世界) 的历史表里 McCarthy 的 Lisp 已经出过场("第一个让程序 = 数据"),这里把这颗种子**接到自递归训练上**。

#### 基因一:递归即定义——一个系统可以"用自己来定义自己"

Lisp 的奠基论文标题就把话挑明了——McCarthy 1960 *Recursive Functions of Symbolic Expressions and Their Computation by Machine*,"**递归**"二字写在门楣上。它最惊人的一手是 **元循环求值器 (metacircular evaluator)**:用 Lisp 写一个 `eval`,而这个 `eval` 能解释包括它自己在内的任何 Lisp 程序——`eval` 调 `apply`、`apply` 回头调 `eval`,**语言用自己定义了自己**。

```lisp
;; 极简元循环求值器 —— 语言用自己定义自己
(defun eval (exp env)
  (cond ((atom exp) (lookup exp env))
        ((eq (car exp) 'lambda) (make-closure exp env))
        (t (apply (eval (car exp) env)            ; eval 调自己
                  (mapcar (lambda (a) (eval a env)) ; 又调自己
                          (cdr exp))))))
```

这正是 [§12.1](#121-自递归把训练这支箭掉头指向训练自己) 那个 `M_{n+1}=F(M_n)` 的**思想原型**:`eval` 能求值 `eval`,对应"模型能改进训练出模型自己的那段代码"。McCarthy 让递归在程序里成了一等公民(λ-calculus 的 Y 组合子做的是同一件事,见 [§10.8](#10-函数式编程视角神经网络是-fgh-的世界)),**自递归训练只是把递归的粒度从"函数调用自己"放大到"整个模型造出下一个自己"**。McCarthy 1960 问的是"一个程序怎么用自己描述自己",Mythos 2026 问的是"一个模型怎么用自己训练下一个自己"——**同一个递归直觉,隔了 68 年和好几个数量级的尺度**。

#### 基因二:代码即数据,数据即代码——自我修改的前提

Lisp 的第二条基因是 **同像性 (homoiconicity)**:`(+ 1 2)` 既是一段**程序**,又是一个三元素的**列表**——代码和数据是同一种结构 (S-表达式)。于是程序能像处理数据一样去**构造、检查、改写、执行另一段程序,包括它自己**(`quote` 把代码当惰性数据,`eval` 再把它跑起来;宏就是在编译期改写代码的程序)。

**这条基因,正是"模型能改训练自己的代码"在逻辑上成立的根本前提**——而 LLM 把它推到了**字面的极致**:除了权重,**一切都是同一种物质——token 序列**。

```python
# 在本仓库里, 这条基因就明明白白摆在 data 契约里 (见 CLAUDE.md):
#   data/<dataset>/train.bin —— 一条扁平的 uint16 token 流
#   Python 源码、英文散文、FIM 模板, 全部压成同一种 uint16
# 训练模型的 train.py 是 token; 模型吐出来的也是 token —— 同质!
```

[§12.1](#121-自递归把训练这支箭掉头指向训练自己) 说"`F` 的定义域包含它自己的源码"——**那句话的学名就是 homoiconicity**。在符号 AI 里代码和数据泾渭分明(`if-else` 是代码、事实库是数据);在 LLM 里**根本没有这条类型边界**:`train.py` 是模型读过的训练数据,也是模型能写出的代码,**是同一串 token**。这就是为什么自递归训练**不需要任何额外机制**——模型的输出空间天然就套住了它自己的来源,正如 Lisp 程序天然能操作 Lisp 程序。本仓库的 `apply_fim_transform`([§7.4](#74-更深一层编译产物本身是一段程序而且编译能凭空造信息))把代码 token 当数据随意重排,就是"代码即数据"最朴素的现场演示。

而 Lisp 的 **`quote` / `eval` 边界**,在自递归训练里**原样重生为 [§12.3](#123-它凭什么不发散不是永动机两道闸) 的"提议 / 验证"闸**:模型提议的改动是 `quote` 状态的**惰性文本**,直到 ablation 把它 `eval`(真跑一遍训练)、验证器量出 loss——**`quote` 对应"提议",`eval` 对应"沙箱执行 + 裁判"**。

#### 但要诚实标出"基因突变"的地方

Lisp 的"代码即数据"是**精确、离散、无损、符号**的:`eval` 跑出的结果板上钉钉。LLM 的同像性是**统计、连续、有损**的:模型没有一个干净的 `eval`,它只有 `P(下一个 token)`——它"提议"的训练改动可能是错的、是幻觉。**那个干净的 `eval` 去哪了?——被外置成了验证器**(真跑 `train.py`、量 val loss,见 [§12.3](#123-它凭什么不发散不是永动机两道闸) 闸一)。所以 Mythos 不是纯 Lisp,而是 **"Lisp 的两条基因长在统计基底上,再把符号的 `eval` 当裁判焊回来"**——这恰好又是本文反复讲的神经-符号三明治(§9.4、[§11](#11-神经-符号混合现代系统都不是纯种))。

#### 这些基因是否都在 Mythos / Fable 5 身上?——逐条对一遍

| Lisp 基因 (1958–60) | 当年形态 | Mythos / Fable 5 形态 |
|---------------------|----------|------------------------|
| 递归即定义 | `eval` 调 `eval`;Y 组合子 | `M_{n+1}=F(M_n)` 自递归训练(§12.1) |
| 元循环求值器 | 用 Lisp 写的 Lisp 解释器 | 模型读/改"训练出自己的那段代码" |
| 代码即数据 | S-表达式既是 list 又是程序 | 一切皆 token:`train.py` 与模型输出同质 |
| `quote` / `eval` | quote 惰性、eval 执行 | 提议(惰性文本)/ ablation 验证(执行)(§12.3) |
| 宏 = 程序改写程序 | macro 在编译期重写代码 | 模型重写 `config/*.py`、`train.py` |

五条基因,Mythos / Fable 5 **一条不缺**——只是每条都从"精确符号版"变异成了"统计连续版 + 符号验证器兜底"。**所以"为什么历史必然走向 Mythos"还藏着一个更古老的答案:因为 1958 年 McCarthy 给 AI 选定的那门语言,骨子里就是"能自我求值、能自我改写"的——递归与同像性这两条基因一旦写进 AI 的 DNA,"模型训练模型"就只是它们在足够算力下的自然显形。** 自递归训练不是背叛了符号主义的老传统,而是**把 Lisp 最核心的两个念头,用神经网络重新实现了一遍**。

### 12.3 它凭什么不发散、不是永动机——两道闸

不动点迭代 `M_{n+1}=F(M_n)` 有两种失败方式,自递归训练分别用一道"闸"挡住,而这两道闸恰好就是本文反复讲的**符号**与**压缩**两件事。

**闸一(防发散):符号验证器让 `F` 单调。** 一个乱来的 `Improve` 完全可能把模型越改越差、甚至发散。挡住它的,是上面代码第 ④ 步那个 `if estimate_loss(M_new) < estimate_loss(M)`——**只有真的把 val loss 降下来的改动才被接受,否则丢弃**。这一步在数学上把任意的 `F` 变成了**单调算子**:序列 `loss(M_n)` 只降不升,于是迭代必然收敛(下有界 + 单调)。

而"提议一个改动、再用一个判据筛掉不满足的"——**这正是 [§3](#3-符号主义-ai人写规则机器搜索) 的符号搜索、[§10.8](#10-函数式编程视角神经网络是-fgh-的世界) 的"逻辑式 = 反向函数"在元层面的重演**:Prolog 给定目标、回溯搜索满足它的输入;自递归训练给定"val loss 要更低"这个目标、让神经网络提候选、符号验证器做回溯式的接受/拒绝。**神经负责"大胆地猜",符号负责"诚实地筛"**——[§11](#11-神经-符号混合现代系统都不是纯种) 那句"执行反馈是最诚实的 reward",在这里管的不再是一段生成代码,而是一次训练配置本身。

**闸二(防永动机):epiplexity 给了天花板。** 第二种质疑更尖锐:**模型用自己的算力造数据、改训练、再训自己,这不是"拽着自己头发离地"吗?信息从哪来?** 这正是 arXiv 2601.03220 *From Entropy to Epiplexity*(Finzi、Qiu、Jiang、Izmailov、Kolter、Wilson)回答的问题。它指出经典信息论的三条铁律,**在"算力有限"的真实学习器身上全部失效**:

| 经典信息论(假设算力无限) | 对算力有限的学习器,实际是 |
|--------------------------|----------------------------|
| 确定性变换不增加信息 | **会增加**——重排/变换能让"深藏到算不动"的结构浮上来变可学(见 [§7.4](#74-更深一层编译产物本身是一段程序而且编译能凭空造信息) 的 FIM 实例) |
| 信息与数据顺序无关 | **有关**——课程学习、数据重排显著改变可学量 |
| 似然建模只是匹配分布 | **不止**——它能逼出比数据生成过程**更复杂的程序** |

为此论文提出 **epiplexity**:**一个算力有限的观察者,真正能从数据里学到的"结构性内容";它把这部分和数据里那段不可压缩的随机噪声 (time-bounded entropy) 显式分开。**

> **Kolmogorov 复杂度问"最短的程序有多长"(算力无限);epiplexity 问"在有限算力下你够得着的最短程序有多长"——后者才是 `F.cross_entropy` 真正在优化的东西。**

这把 [§5.3](#53-压缩的不是数据是为了压缩数据而构造的程序) "模型学的是为了压缩数据而构造的程序"钉成了**可度量、有上界**的量。于是"永动机"的疑问有了干净的答案:**自递归的每一轮,不是凭空造信息,而是用算力把数据里"本来就在、但上一代算力够不着"的 epiplexity 兑现出来**——和 [§7.4](#74-更深一层编译产物本身是一段程序而且编译能凭空造信息) 里 FIM 那个"确定性重排却创造了可学信号"是同一回事,只是从"重排 token"放大成了"重排整个训练流程"。

两道闸合起来,就说清了这个不动点的性质:**它收敛(闸一保证单调),但收敛到的不是无穷,而是"当前算力下能够得着的最短程序"(闸二给的天花板)。** 这也顺手解释了为什么**还需要不断加算力**:算力一涨,epiplexity 的天花板抬高,`F` 的不动点跟着上移,于是**下一代模型还有的可学**——这就是代际跃迁的引擎。

### 12.4 为什么"历史必然"收敛到 Mythos / Fable 5

现在两道闸都装好了,可以正面回答标题:为什么历史**必然**走到 Mythos / Fable 5,而不是停在 GPT-2 / CodeGPT 这种"人手调参"的形态?

把全文主线竖着读,是一条**人类逐步退出"谁写"这一列的曲线**——而 §12.3 那两道闸,正好是推着这条曲线往右走、且**不可逆**的两只手:

```
谁写"训练过程"?     人写规则      人写架构/数据写参数     模型提议+符号验证/人只把闸门
                    符号 AI    →   经典深度学习/CodeGPT  →   自递归训练 / Mythos / Fable 5
                    1956–80s        2012–2025                 2026–
推右边的力 ①(能力):                epiplexity 天花板每代抬高 → 模型自己成了够格的"训练研究员"
推右边的力 ②(机制):                符号验证器让"模型改模型"从危险变可控 → 敢把人移出循环
不可逆的原因:                       一旦 F 的单代增益 > 人手调参的增益, 留着人就是留着瓶颈
```

- **力 ①(能力侧):epiplexity 的天花板每代在抬。** 每一代把更多 epiplexity 榨进权重(§12.3 闸二),模型自己就越来越够格当那个"读得懂 `train.py`、提得出有效改动"的研究员。当一个模型能把训练提速 52×,还坚持让人来手填 `learning_rate`,在速度和成本上都讲不通——**人成了 `F` 这个循环里最慢的一环**。
- **力 ②(机制侧):符号验证器让自动化变安全。** 敢把人移出循环,前提是闸一那个诚实裁判(§12.3):ablation 的 val loss 不会撒谎,所以"模型改模型"不会失控滑走。**没有这道符号闸,自递归只是危险;有了它,自递归才成了工程。**

这两只手一旦同时到位,曲线右移就**不可逆**了:只要"模型提议改动带来的单代增益" 超过 "人手调参的增益",继续把人留在循环里,**留着的就纯是瓶颈**。经济、速度、竞争三重压力会把所有前沿实验室都推向同一个方向——**这就是"历史必然"四个字的全部含义:它不是预言,是两道闸 + 三重压力下唯一稳定的均衡。**

而 **Mythos / Fable 5,就是这条不动点轨迹上、`F` 已经迭代到"上一代模型实质性地参与设计了这一代"那个阶段的命名航标。** "Mythos 背后是什么"的答案,不是某次灵感,而是上面那个 `for _ in range(generations)` 循环的累积增益跨过了一个阈值——**模型贡献的改动开始压过人贡献的改动**(80% 代码、52× 提速,就是这个阈值被跨过的两个读数)。它们不是新范式,而是本文三种老范式在自指循环里的**最终合体**:

- **神经**——`forward` 那串可微变换,负责"大胆地猜"(提议改动、生成数据);§5、§10。
- **概率**——`softmax`/采样,负责在候选改动之间**探索**而非死板地取最优;§9.2。
- **符号**——verifier、ablation、`configurator.py` 式的安全接口、"一键暂停"闸门,负责"诚实地筛"与约束;§3、§11。

> 所以 2026 的 AI **不是"神经打败了符号"**,而是**把符号从"人写规则的手里",搬到了"裁判神经提议的席位上"**。自递归训练把这套"神经猜、符号筛、概率探索"的循环,一路自动化到连"造下一代模型"本身都纳了进来。这就是开篇那条暗线的终点:**当"谁写"一路写到"模型写训练自己的代码",AI 史就从"人教机器"翻篇成了"机器在人设的闸门内教机器"。**

**CodeGPT 正是这条曲线的起点切片。** 它的 `train.py` 每个超参还是人手填的——但 §12.1 那个不动点循环的零件**它全有了**:可调又受约束的提议空间 (`configurator.py`)、诚实的验证器 (`estimate_loss` 的 val loss)、"确定性变换造可学信息"的活样本 (`apply_fim_transform`)。从 CodeGPT 到 Mythos,中间**缺的不是任何新原理**,只缺一件事——**把 `for trial in human_ideas` 里的 `human_ideas`,换成模型自己。** 一旦换上,Y 组合子开始转动,剩下的就是 §12.3 那两道闸保证它**既转得起来、又不飞出去**。

### 12.5 把镜头从"训练"拉到"agent":同一副骨架的第三次显形

前面四节讲的都是"训练在自己写自己"。现在退一步看一个更大的事实:**那副 Lisp 骨架——原子化组合 + 递归不动点 + 代码即数据 + 验证器闸门——正在 agent 设计里独立地、又一次长出来。** 这就是为什么"优秀的系统越来越朝着原子化、朝着 Lisp 递归定义的形式收敛"不是审美偏好,而是**收敛**:同一副骨架,本文已经见过它两次显形——`forward` 的函数组合([§10](#10-函数式编程视角神经网络是-fgh-的世界))、自递归训练的代际不动点([§12.1](#121-自递归把训练这支箭掉头指向训练自己))——agent 是它的**第三次显形**,只是把舞台从"权重空间"换到了"运行时"。

**① 原子化 = §10 那条组合律的回归。** 看 2026 年口碑最好的 agent harness——earendil-works 的 **Pi (`pi-mono`)**:它不是"一个带一堆 config flag 的大应用",而是**五个各司其职、层层叠起来的小包**,信条是"**do one thing well, make it composable, let developers build the rest**";每个扩展是一个自包含模块,挂到 agent 生命周期上。`Atomic Agents` 框架同理——用**可复用的原子组件**拼系统。Pi 更关键的一手是:**把 agent loop、message stream、tool execution、state 全部暴露成可组合的 API**——也就是**把"循环"本身做成了一等的可组合值**。这跟 Lisp 把函数/代码做成一等公民、跟 [§10.3](#10-函数式编程视角神经网络是-fgh-的世界) 的 transducer"把变换从数据里解耦"、跟 [§10.10](#10-函数式编程视角神经网络是-fgh-的世界) 说的"组合是 AI 永恒的骨架",是**同一条原则的第 N 次复读**:原子工具是字母表,agent 是字母表上的组合。

**② agent loop = 不动点迭代(递归本身)。** agentic loop 的奠基模式是 **ReAct**:reason → act(调工具)→ observe → 再 reason……直到目标达成。把它写成一行,就是:

```python
# agent loop 的本质 —— 和 §12.1 的 M_{n+1}=F(M_n) 是同一个形状
state = initial
while not goal(state):          # 终止判据 = 验证器(见下文 ③)
    thought = model(state)      # reason  (神经: 大胆地猜)
    obs     = execute(thought)  # act     (符号: 真跑工具)
    state   = update(state, obs)# observe (fold 进状态)
```

这正是 [§10.1](#10-函数式编程视角神经网络是-fgh-的世界) 那个 `for block in self.transformer.h: x = block(x)` 的 **fold/reduce**,只不过被 fold 的不再是 12 个 Block,而是 agent **自己产出的、长度未定的"动作-观察"流**;也正是 [§12.1](#121-自递归把训练这支箭掉头指向训练自己) 那个 `M_{n+1}=F(M_n)` 的不动点,只不过尺度从"代际"缩到了"一个任务之内"。**本仓库其实已经有这个循环的胚胎**——`model.py` 的 `generate()`:吐一个 token → 接回输入 → 再吐,直到撞上 `stop_tokens`。**`generate()` 就是最小的 agent loop,而 `stop_tokens` 就是最小的终止判据**(§9.1 早就点过它是"符号"的)。

**③ loop engineering = 给这个递归装上 §12.3 的两道闸——而且是被独立地重新发现的。** 2026 年的共识是一句话:"**真正拉开 agent 差距的往往不是底层模型,而是那个 loop**"(loop engineering)。而怎样设计这个 loop?业界给的清单是:**清晰可验证的目标 + 迭代上限 + token 预算 + 无进展检测**。把这份清单和 [§12.3](#123-它凭什么不发散不是永动机两道闸) 那两道闸并排放——**逐项对得上**:

| loop engineering 的术语 | 本文 §12.3 的闸 | 作用 |
|------------------------|-----------------|------|
| 可验证的目标 / no-progress 检测 | **闸一:符号验证器(单调过滤)** | 防发散——只朝目标走,停在原地就退出 |
| 迭代上限 / token 预算 | **闸二:有限算力的天花板** | 防永动机——算力有界,够不着就收敛 |

这不是巧合,是**同一个控制论结构在两个领域被各自撞见**:任何"让系统自己一圈圈跑、还要它别飞出去"的设计,**最后都必然长出"符号判据 + 算力上界"这两道闸**。自递归训练在"代际"尺度上需要它们,agent loop 在"任务内"尺度上同样需要它们——**这就是你说的"从 agent 发展角度看也是一致的"那个一致性的硬核**。

**④ 代码即数据,在 agent 里也原样复现。** agent 把动作以**文本/代码(tool call)**的形式吐出来,再由 harness 执行——这就是 [§12.2](#122-这套自指不是新发明它是-lisp-1958-年就埋下的两条基因) 基因二"代码即数据"的现场:**模型提议的 tool call 是 `quote` 态的惰性文本,`execute()` 才是 `eval`**。而 Pi 的 **subagent**、Claude Code 的子 agent——**一个 agent 派生另一个 agent**——就是 §12.2 基因一(自指 / 元循环求值器)在运行时的化身:**`eval` 调 `eval`,搬到 agent 层就是"agent 派 agent"**。所以未来 agent 的方向(更原子、更能递归地派生子 agent、更讲究 loop engineering)不是三件新事,**是 Lisp 那两条老基因在 agent 这一层的同时显形**。

把三次显形并到一张表上,这一节的主张就一目了然:

| Lisp 基因 | forward(§10) | 自递归训练(§12.1) | agent / loop engineering(§12.5) |
|-----------|--------------|---------------------|----------------------------------|
| 原子化 + 组合 | `nn.ModuleList` 串 Block | 可调旋钮的提议空间 | Pi 的五个可组合包 / atomic tools |
| 递归 / 不动点 | `for block: x=block(x)` (fold) | `M_{n+1}=F(M_n)` | `while not goal: state=step(state)` (ReAct) |
| 代码即数据 | token 流不分代码/数据 | `train.py` 与输出同质 | tool call 是文本,`execute` 才 eval |
| 自指 | —— | 模型改训练自己的代码 | agent 派生子 agent |
| 验证器闸门 | `stop_tokens` | ablation 的 val loss | 可验证目标 + 迭代/token 上限 |

> **结论:模型、训练、agent 这三层,正在收敛到同一副骨架——而那副骨架,就是 McCarthy 1958 年给 AI 选的那门语言的骨架(递归 + 同像性 + 组合)。** "优秀系统越来越原子化、越来越像 Lisp 递归定义",不是因为大家都读了 SICP,而是因为**在有限算力下、要用简单原语拼出开放行为、还得可控**,这副骨架是唯一稳定的吸引子。Mythos / Fable 5 是这副骨架在"模型 + 训练"上的显形;Pi 这类 harness 加上 loop engineering,是它在"agent 运行时"上的显形。**两者是同一件事在不同时间尺度上的样子**——这正是本文从第 1 节到第 12 节那条暗线,在 agent 时代写下的最自然的续集。

---

## 13. 给用户原问题的最终回答

回到一开始的问题——**"GPT 是不是把训练数据当成一堆可执行的符号代码来执行？"**

答案分三层：

**第一层（直觉是对的）**：

> 是的。训练数据**塑造**了模型行为，正如代码塑造了程序行为。改数据可以改模型，这点和"改代码改程序"同构。

**第二层（更精确）**：

> 训练数据不是被"执行"的，而是被"编译"成权重。SGD 是编译器，权重是编译产物，前向传播是执行。和符号代码的关键区别是：编译是统计性、有损、不可逆的——你不能从权重反推出训练数据，正如不能从二进制反推出源码。

**第三层（彻底澄清）**：

> 训练数据和"规则"不是同一种东西。训练数据是**例子**；权重里的"规则"是 SGD 从这些例子**归纳**出来的、连续的、概率性的、不可枚举的隐式模式。这一步归纳——从离散例子到连续分布——是**符号 AI 永远做不到**的、深度学习的核心贡献。
>
> 而**贝叶斯网络**正好是介于两者之间的中间形态：图结构是符号的（可读、可枚举），CPT 是概率的（从数据学）。它在小数据 + 强先验 + 需要因果解释的场景下，至今好用。

所以正确的心智模型不是"二选一"，而是：

```
        知识由人写  ←──────────────────────────────────→ 知识由模型自己写
        ↑                                                          ↑
        符号 AI    →    贝叶斯网络    →    深度学习    →    自递归训练
        Prolog          因果图              Transformer        Mythos / Fable 5
        专家系统        HMM/Naive Bayes     CodeGPT            (模型提议+符号验证)
        Lean/Coq        VAE (混合)                            人退到"闸门"位
```

这条横轴的最右端,正是 [§12](#12-2026-前沿自递归训练与-epiplexity历史必然如何走向-mythos--fable-5) 的落点:**人从"写知识"一路退到"写架构",再退到"只把闸门"**。而 epiplexity 给了这条退场曲线一个理论保证——模型靠计算把数据里"够不着的程序结构"翻出来,所以"模型造数据训模型"不是永动机,而是在兑现可学结构的支票。

**当前 (2026) 的工业级 AI 系统几乎全部是混合体**：神经做核心、贝叶斯做不确定性建模、符号做协议骨架和验证。CodeGPT 自身就是这种三明治结构的一个最小样本——`forward` 是神经，`F.softmax + multinomial` 是概率，`<|fim_*|>` 模板和 `stop_tokens` 是符号。三个范式不是替代关系，而是栈关系。

读这篇文档的最大收获，应该是：以后看任何 AI 系统，都能下意识地把它拆成"哪一层是符号、哪一层是概率、哪一层是神经"——这是分析、调试、设计 AI 系统的基本功。

而 2026 年要再加一条:**还要看清"谁在写这个系统"**。从 CodeGPT 里人手填的每一个超参,到 Mythos / Fable 5 里"模型提议、符号验证、人把闸门"的自递归循环——"谁写"这条轴的滑动,才是这十年 AI 真正的主线。它没有终结三种范式,而是把它们**摞成了一个会自我改进的栈**:神经提议、概率探索、符号裁判,而 epiplexity 在底层保证这一切不是空转。看懂这个栈怎么把人一步步挪到闸门位,你就看懂了为什么历史的必然会走向 Mythos 和 Fable 5。
