# GPT 是更高级的"万能谷歌搜索"：从关键词到高维空间的搜索进化史

> 程序员每天用 GPT 的姿势，本质和十年前用 Google 搜报错信息没差。
>
> - **2010 年**：报错 → 复制英文关键词 → Google → Stack Overflow 第一个答案 → 抄回去。
> - **2018 年**：把"语义相似"也算上 → 向量搜索（Elasticsearch 的 `dense_vector`、Faiss、向量数据库）。
> - **2020 年**：搜索的对象从"文档"升级成"高维空间里的点"，相似度从字符串匹配变成内积。
> - **2023 年**：GPT 直接"搜索 + 拼接结果 + 过滤无关 + 顺手帮你写出来"——只是这一切都在高维空间里、由一次 forward 完成。
>
> 这篇文档的目的不是把 GPT 神秘化，而是反过来：**把 GPT 拆成一个你已经熟悉的搜索引擎**，让你知道训练 / 微调一个个人 GPT 本质上就是"在自己的语料上重建一个更懂你的搜索索引"。理解了这件事，你用 GPT 的姿势会立刻升级——你会知道什么任务它能做、什么不能、什么时候要喂检索结果（RAG）、什么时候要训权重（SFT），以及为什么"prompt 写得好"等价于"搜索词写得好"。

---

## 目录

1. [搜索引擎的四代进化](#1-搜索引擎的四代进化)
2. [第一代：关键词匹配（grep over the internet）](#2-第一代关键词匹配grep-over-the-internet)
3. [第二代：向量搜索（语义近似）](#3-第二代向量搜索语义近似)
4. [第三代：高维空间里的搜索（embedding everywhere）](#4-第三代高维空间里的搜索embedding-everywhere)
5. [第四代：GPT — 在每一层都做一次搜索](#5-第四代gpt--在每一层都做一次搜索)
6. [Attention 就是"软 SQL"：QK^T 是 WHERE，softmax 是过滤，@V 是 SELECT](#6-attention-就是软-sqlqkt-是-wheresoftmax-是过滤v-是-select)
6.5. [QKV 三角投影：把每个 token "揉碎"成三个角色](#65-qkv-三角投影把每个-token-揉碎成三个角色)
6.6. [GPT 是抄答案拼接器：迁移和幻觉是同一枚硬币](#66-gpt-是抄答案拼接器迁移和幻觉是同一枚硬币)
6.7. [多头 = 多个"特征矩阵"：QKV 为什么是 Transformer 成功的核心](#67-多头--多个特征矩阵qkv-为什么是-transformer-成功的核心)
6.8. [子空间是怎么分离的？SVM 升维 vs Transformer 升维](#68-子空间是怎么分离的svm-升维-vs-transformer-升维)
6.9. [数据怎么才能分离开？SVD vs 特征矩阵 vs 看标签差异（一个会让 SVD 翻车的例子）](#69-数据怎么才能分离开svd-vs-特征矩阵-vs-看标签差异一个会让-svd-翻车的例子)
7. [训练个人 GPT = 在你的语料上重建搜索索引](#7-训练个人-gpt--在你的语料上重建搜索索引)
8. [实战：把"用好 GPT"翻译成"用好搜索"](#8-实战把用好-gpt-翻译成用好搜索)
9. [小结：GPT 不神秘，它只是把搜索做到了极致](#9-小结gpt-不神秘它只是把搜索做到了极致)

---

## 1. 搜索引擎的四代进化

| 代际 | 核心数据结构 | 相似度怎么算 | 召回什么 | 程序员的痛点 |
|------|--------------|--------------|----------|--------------|
| 1. 关键词 | 倒排索引 | 字符串精确匹配 + TF-IDF | 包含关键词的文档 | 同义词、拼写、跨语言全都失效 |
| 2. 向量搜索 | Faiss / HNSW | 内积 / 余弦 | 语义近似的文档 | 切 chunk、embedding 选型、跨域漂移 |
| 3. 高维空间搜索 | embedding model + ANN | 学习到的距离 | 概念邻居（不止文本） | 仍然是"找文档"，不会写答案 |
| 4. GPT | 模型权重 `W` | 一次 forward 里多次 attention | 直接生成答案 | 幻觉、对齐、上下文长度 |

每一代都不是把上一代推翻，而是**把"搜索"这件事在更高维的空间上做、用更连续的相似度算、把更多步骤压进同一次操作**。GPT 是这个谱系的顶点——但仍然是搜索。

---

## 2. 第一代：关键词匹配（grep over the internet）

最朴素的搜索就是 `grep`。Google 早期的本质就是"全网范围的 grep + PageRank 排序"。

```python
# 伪代码：关键词搜索的核心
def keyword_search(query: str, docs: list[str]) -> list[str]:
    tokens = query.lower().split()
    hits = []
    for doc in docs:
        if all(t in doc.lower() for t in tokens):
            hits.append(doc)
    return rank_by_tf_idf(hits, tokens)
```

为什么程序员用英文报错关键词搜得比中文好？因为：

1. **倒排索引是字符串级别的**——`TypeError: 'NoneType' object is not subscriptable` 这种报错全网每天有几十万次复现，关键词命中率 100%。
2. **英文是 Stack Overflow / GitHub 的母语**——文档密度 × 关键词唯一性 = 高召回。
3. **报错信息本身已经是一种"高熵特征"**——它包含罕见的符号组合，几乎是天然的指纹。

但这一代的天花板很硬：

- 同义词不行（"crash" 找不到 "panic"）。
- 拼写错误不行。
- 跨语言不行（中文报错和英文文档对不上）。
- "我想知道怎么用 LangChain 做检索"——这种**意图查询**，关键词搜索几乎无能为力。

所以下一代必须把"含义"也搜进来。

---

## 3. 第二代：向量搜索（语义近似）

向量搜索的核心思想是：**把每段文本压成一个固定维度的向量，让"语义相近的文本"在向量空间里也相近。**

```python
# 伪代码：向量搜索
import numpy as np

def vector_search(query: str, docs: list[str], encoder, k=5):
    q_vec = encoder.encode(query)             # (d,)
    doc_vecs = encoder.encode(docs)           # (N, d)
    scores = doc_vecs @ q_vec                  # (N,) 点积 = 相似度
    top_k = np.argsort(-scores)[:k]
    return [docs[i] for i in top_k]
```

注意这里两个关键：

- **`encoder.encode`** 把文本投到 $\mathbb{R}^d$（典型 $d=384$ 或 $768$）。
- **`@ q_vec`** 内积就是相似度。这是一个**纯矩阵运算**，没有任何字符串比较。

突然之间——同义词、跨语言、意图查询都可以工作，因为 encoder 已经在巨量语料上学过"crash 和 panic 在同一个邻域"。

这正是 CodeGPT 的 token embedding 在做的事，只是更原始。看 `tokenizer.py` 把字符串转成 ID，再看 `model.py:183`：

```python
# model.py:183 —— 这一行就是"查表 = 第一次嵌入"
tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
```

`wte`（word token embedding）就是一个 `(50304, 768)` 的矩阵。每个 token ID 进来，查出一个 768 维向量——**这就是 token 的语义坐标**。和向量搜索引擎里每篇文档的 embedding 是同一类东西，只是粒度更细：一个 token 一个向量，而不是一段文本一个向量。

---

## 4. 第三代：高维空间里的搜索（embedding everywhere）

第二代向量搜索还有一个隐含假设：**"一段文本只压成一个向量"**——句子级、文档级 embedding。

但语言不是这样工作的。同一个词在不同上下文里意思可以完全不同（"bank" 是银行还是河岸？）。所以现代做法是：

- **每个 token 一个向量。**
- **同一个 token 在不同上下文里的向量也不同**（contextualized embedding）。
- **这些向量在 12 / 24 / 96 层 Transformer 里反复变形**，每过一层就在更高阶的概念空间里挪一次位置。

这就是 BERT 最初让人惊艳的地方，也是 GPT 之所以能"理解"的基础。

我们直接看 CodeGPT 的 forward（`model.py:177-198`）：

```python
def forward(self, idx, targets=None):
    tok_emb = self.transformer.wte(idx)        # 第一次嵌入：token → 768维向量
    pos_emb = self.transformer.wpe(pos)         # 位置也嵌入成向量
    x = self.transformer.drop(tok_emb + pos_emb)
    for block in self.transformer.h:            # 12 个 Block，每层都在重新搜索
        x = block(x)
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)                    # 最后一层：在 50304 维上找最匹配的下一个 token
```

把它读成**搜索引擎语言**：

| 这一行在干什么 | 搜索引擎类比 |
|----------------|--------------|
| `wte(idx)` | 查询词被向量化（query embedding） |
| `wpe(pos)` | 加上位置坐标（document position） |
| `for block in h: x = block(x)` | **12 次重排 / 重检索**——每一层都在更抽象的空间上重新搜一遍 |
| `lm_head(x)` | 最后一次搜索：从 50304 个候选 token 里挑出"最匹配下一个位置"的那个 |

**关键洞察**：transformer 不是搜索一次就给你答案，它在 12 层里搜了 12 次。每一层处理的"概念粒度"不同——浅层处理语法、中层处理短语、深层处理意图。这就是**深度**带来的能力——它让"搜索"在层次结构上展开。

---

## 5. 第四代：GPT — 在每一层都做一次搜索

那每一层 `block(x)` 内部具体怎么搜？答案就是 **self-attention**。看 `model.py:53-70`：

```python
# model.py:53-70 —— attention 就是一次"在序列内部"的搜索
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)   # 把 x 投影成 query, key, value
# ...省略 reshape...
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))   # 相似度矩阵
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # 因果掩码
att = F.softmax(att, dim=-1)                          # 概率化（过滤 + 归一）
y = att @ v                                            # 加权聚合
```

把这五行翻译成你熟悉的搜索引擎术语：

```
q @ k.T              ⟺ 相似度计算（内积，和向量搜索同款）
masked_fill(...-inf) ⟺ 过滤不可见的文档（因果约束：只能搜历史）
F.softmax            ⟺ Top-k → 软性 Top-k：把不相关的得分压到接近 0
att @ v              ⟺ 把检索到的内容按相关度加权拼接
```

**这就是用户问题里说的那句话的数学版本**：
> GPT "搜索生成" 本质 = 搜索（QK^T） + 过滤不相关（softmax） + 拼接搜索结果（@V） + 在高维空间上完成。

而且这件事每一层、每一个头都做一次。CodeGPT 的默认配置（`model.py:113-115`）：

```python
n_layer: int = 12
n_head:  int = 12
n_embd:  int = 768
```

12 层 × 12 头 = **每生成 1 个 token 就做了 144 次"高维搜索"**。这就是为什么 GPT 给的答案"看起来综合了很多信息"——它确实做了 144 次检索 + 加权 + 重组。

---

## 6. Attention 就是"软 SQL"：QK^T 是 WHERE，softmax 是过滤，@V 是 SELECT

如果你更熟悉 SQL 而非搜索引擎，类比同样成立：

```sql
-- 传统检索：硬 WHERE
SELECT v.content
FROM   keys k JOIN values v ON k.id = v.id
WHERE  similarity(k.vector, :query_vector) > 0.8
ORDER  BY similarity DESC
LIMIT  5;
```

```python
# Attention：软 WHERE（每一行都参与，但权重不同）
scores  = q @ k.T / sqrt(d)           # similarity(k, q) for all k
weights = softmax(scores)              # 权重而不是 0/1 过滤
result  = weights @ v                  # SELECT 但用加权和而非 LIMIT
```

差别就这一点：**SQL 是 0/1 的硬过滤，attention 是 [0,1] 的软过滤**。软过滤的好处是**可微分**——梯度可以流过 softmax，让模型自动学到"什么样的相似度该保留"。

这也解释了 GPT 推理时的两个采样策略（`model.py:287-299`）：

```python
# top-k：硬过滤前 k 名（传统搜索的 LIMIT）
v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
logits[logits < v[:, [-1]]] = -float('Inf')

# top-p (nucleus)：动态过滤累积概率 > p 的尾部（自适应 LIMIT）
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
sorted_indices_to_remove = cumulative_probs > top_p
```

`top_k` 就是搜索结果的 `LIMIT k`，`top_p` 是"动态 LIMIT，直到累积置信度 ≥ p"。**完全是搜索引擎的语义。**

---

## 6.5 QKV 三角投影：把每个 token "揉碎"成三个角色

回到 attention 那一行（`model.py:53`）：

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```

`self.c_attn` 是一个 `Linear(n_embd, 3 * n_embd)`（`model.py:36`）。它把同一个输入向量 `x` 通过**三组不同的权重矩阵**投影成三份完全不同的东西：

- **Q（query）**：这个 token 想搜什么？
- **K（key）**：这个 token 能被什么搜到？（它的"标签"）
- **V（value）**：被搜到时，它贡献什么内容？

**关键不是"分成三份"，而是"同一个向量被三种角度撕开"**。同一个 token "for" 在 Q 角度可能问的是"我后面接的是迭代器吗"，在 K 角度暴露的是"我是循环关键字"，在 V 角度提供的是"循环语义本身"。一个 token 有三套人格，每套都由不同的 W 学出来。

接下来还要再揉一次（`model.py:54-56`）：

```python
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
```

12 个头 = 把 768 维向量**砍成 12 段，每段 64 维**，每个头独立做一次 QK^T 搜索。**这就是"矩阵揉碎重组"的具体含义**：

- 每个头看一个不同的 64 维子空间，可能一个头专门看"语法依存"，一个头看"作用域嵌套"，一个头看"类型流动"……
- 12 个头的搜索结果用 `c_proj`（`model.py:37`）再缝合回 768 维。
- 12 层 × 12 头 = **144 次"切片 → 搜索 → 缝合"**。

**这就是 GPT 之所以能"跨语言迁移"的根源**：因为每个 token 都被揉碎成多个角度的子空间，Python 的 `for x in range(10):` 和 JavaScript 的 `for (let x = 0; x < 10; x++)` 在某几个头里会被表示成几乎相同的 K 和 V——它们都是"循环结构"在抽象空间里的同一个邻居。模型从未明确学过两种语言的对应关系，但**揉碎重组让它们在某些子空间里自动汇合到同一个"算法理想型"**。

这条原理也解释了 `data/github_code/prepare.py` 里支持多语言混合训练（`--langs python javascript typescript ...`）的合理性：

> 喂多种语言不是让模型学 N 种语法，而是让 attention 头把 N 种语法的共同结构压成一份。Python 学到的循环、JS 学到的回调、Rust 学到的 ownership——在足够深的层里都被对齐到同一个跨语言的"算法骨架"上。

这也就是为什么 GitHub Copilot 在你写一种它训练时只见过几千行代码的小众语言（比如 Nim、Zig）时，仍然能给出像样的补全：它检索的不是"Nim 代码片段"，而是"在所有语言的共同高维空间里，最像这个上下文的算法骨架"，再用 Nim 的语法表面"渲染"出来。

---

## 6.6 GPT 是抄答案拼接器：迁移和幻觉是同一枚硬币

如果你接受"GPT = 高维空间搜索引擎 + 加权拼接"，就必须同时接受一个不舒服的推论：

> **GPT 永远在抄答案，只是抄的不是字符串而是高维向量的加权和。**

`y = att @ v`（`model.py:70`）这一行就是**抄答案的数学定义**：从全部历史 token 里按相关度加权拷贝它们的 V，输出就是这堆 V 的混合体。

理解这一点会同时解释两个看似相反的现象：

### 现象 A：跨语言、跨领域的"惊人迁移能力"

模型从没学过你的私有 DSL，但你贴几个例子它就能继续写——因为：

- 每个 token 在 144 次切片搜索里都能找到训练集中"形似 + 神似"的高维邻居。
- `att @ v` 把这些邻居的 V 拼起来，**对你来说"看起来很懂"，但它其实是把训练分布里相邻的几十个片段加权混合给你。**
- 多语言、多框架训练让"邻居池"足够大、子空间足够通用——所以总有东西可抄。

这是 GPT 的**特性**：抄得越广越像一个全能助理。

### 现象 B：幻觉（瞎编）

幻觉的本质是**"邻居池在那个位置稀疏甚至为空，但 softmax 强行归一化后必须给出权重"**。

看 `model.py:68`：

```python
att = F.softmax(att, dim=-1)
```

`softmax` 的关键性质：**输出永远归一化为 1**。哪怕所有 logits 都很低（说明历史 token 里没有真正相关的内容），它也会把几个"相对最不离谱"的 V 拼起来——结果就是看起来言之凿凿、实际胡编乱造。

具体几种典型场景：

| 场景 | 邻居池状态 | 输出表现 |
|------|------------|----------|
| 训练集见过原题 | 邻居高密度、高相似度 | 准确复述 |
| 训练集见过类似题 | 邻居中等密度、近似 | 跨域迁移（看起来像理解） |
| 训练集没见过、但子空间有邻居 | 邻居稀疏、相似度低 | 似是而非、细节错（幻觉） |
| 训练集完全没邻居 | softmax 权重接近均匀 | 胡说八道（严重幻觉） |

**所以幻觉不是 bug，是 attention + softmax 这套机制在低密度区域的必然表现**。同一个让它能跨语言迁移的机制（揉碎 + 加权拼接），在邻居池不足时直接退化成瞎编。

最常出问题的几类输入——本质都是**邻居池稀疏区域**：

- 编造不存在的 API（`numpy.special_function()`）：训练集中"numpy.xxx"的 K 太多，softmax 总能找几个混合出一个看起来合理的名字。
- 编造文献引用 / DOI：论文标题、作者、年份的子空间各自有大量邻居，揉碎重组就成了"格式正确但内容不存在"的引用。
- 数学计算错误：算术不在它的强项分布里，邻居池靠的是"看起来像计算的文本"。
- 私有领域问答：你公司内部 API、自家代码库——训练集压根没有。

### 对应的工程含义

这套理解直接给出三条工程建议：

1. **prompt 里给足"邻居"**：把相关代码、报错、文档贴进 prompt，相当于人为往邻居池里塞高质量样本。这就是 RAG 的本质（见 [`RAG_VS_SFT.md`](RAG_VS_SFT.md)）。
2. **对低密度区域要持怀疑**：API 名、版本号、引用、数字——这些是幻觉重灾区。让 GPT 写代码可以信，让它说"这个库的某函数签名"必须验证。
3. **训练个人 GPT 的真实价值**：不是让它"懂更多"，而是**在你的私有数据邻域上把邻居池密度提上去**。10M 参数的小模型在你公司代码上跑 SFT 后，它生成你内部 API 调用的准确率可以远超 GPT-4——因为 GPT-4 在那个区域邻居为零，而你的小模型邻居很密。

> **一句话**：跨域迁移和幻觉是同一种行为在不同邻居密度下的两种外观——邻居够密，叫"举一反三"；邻居为空，叫"胡说八道"。理解这一点，你就知道什么任务能用 GPT、什么必须配 RAG、什么必须 SFT。

---

## 6.7 多头 = 多个"特征矩阵"：QKV 为什么是 Transformer 成功的核心

我们退一步看，**为什么 Transformer 这一套打败了 RNN、CNN、所有上一代序列模型？** 直接答案是 multi-head attention，但更精确的说法是：

> **QKV 把"特征提取"这件事变成可以并行学习多套独立的特征矩阵。**

### K 就是"学到的特征模板"

在传统 ML 里，特征工程是手工的——你写规则、定模板。在 CNN 里，特征是卷积核学的——一个核学一个特征（边缘、纹理、眼睛……）。在 attention 里：

- **K（key）就是 token 暴露给检索系统的"特征模板"**。
- `q @ k.T` 就是"我的查询和你这个特征模板有多匹配"。
- `c_attn` 里那块产生 K 的权重（`Linear(n_embd, n_embd)` 的 K 部分）就是**学到的特征矩阵**——和 CNN 的卷积核完全同构。

12 个头 = 12 套独立的特征矩阵 K¹, K², …, K¹². 训练时每个头被不同的下游梯度路径推着走，它们会**自动分工**：

| 头编号（典型）| 学到的特征类型 | 类比传统 ML |
|---------------|----------------|-------------|
| Head 1 | 前一个 token 是什么（局部 n-gram）| Bigram 特征 |
| Head 2 | 同一缩进层的 token | 作用域特征 |
| Head 3 | 匹配的括号 / 引号 | 配对特征 |
| Head 4 | 函数定义 → 调用名 | 类型流特征 |
| Head 5-12 | 越来越抽象、越来越难命名 | 高阶组合特征 |

每多一个头，**特征空间就多一组维度**。这就是为什么 `n_head=12` 比 `n_head=1` 强得多——不是因为参数多了 12 倍（实际上没多，每头维度只有 64），而是**模型同时在用 12 套不同的"匹配标准"找邻居**。

### 为什么是 QKV 三件套，不是两件、四件？

- **只有 K + V 没有 Q**：没法"按需查询"，退化成静态特征。
- **Q + K 合并**：失去"查询模式 ≠ 被查询模式"的灵活性。一个 token 可以"想问 A"同时"被别人问到 B"，这种不对称是关键。
- **加上 R / U 等更多角色**：边际收益递减。三件套已经把"查询、被查、内容"三种角色解耦干净，足够通用。

QKV 就是序列建模里"特征提取"的最小完备解——这就是 Transformer 设计的精髓。

---

## 6.8 子空间是怎么分离的？SVM 升维 vs Transformer 升维

用户问得好：**多头之间的子空间是靠什么分离的？**——SVM 的做法是"升维 + 超平面切",Transformer 怎么做？

先看 SVM 的核技巧：

```
原始空间    →  φ(x) 升维     →  线性超平面 w·φ(x) + b = 0 切开
(2维不可分)    (高维可分)         (训练得到 w)
```

**核心思想**：低维不可分的问题，到高维就线性可分（Cover 定理）。

Transformer 在每一个 Block 里都做了**两次升维 + 一次"软分离"**：

#### 升维一：c_attn（`model.py:36`）

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
```

把 768 维向量升到 2304 维（3 倍），切成 Q/K/V 三份。**这就是 attention 的"核映射"**——升到三倍空间，让原本难以区分的"查询模式 / 被查询模式 / 内容"在更高维里彻底分开。

#### 升维二：MLP（`model.py:80-82`）

```python
self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
self.gelu   = nn.GELU()
self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
```

这就是**Transformer 里教科书级的"核技巧"**：

- `c_fc`：768 → 3072，4 倍升维。
- `GELU`：非线性（线性升维不够，必须有 nonlinearity 来打破线性叠加，这是核函数的本质）。
- `c_proj`：3072 → 768，投影回原维度。

**这一段几乎就是 SVM 核技巧的神经网络版**——只不过 SVM 在升维后用一个线性超平面分割，MLP 在升维后用另一个线性层（`c_proj`）做加权"读出"，两者都依赖"高维线性分割就够了"这一性质（Cover 定理 / 通用近似定理）。

#### 软分离：multi-head 切片

那 12 个头之间的子空间靠什么分离？答案是**结构 + 训练**两件事合力，**没有显式的超平面切**：

1. **结构强制（机械分离）**：`view(B, T, n_head, C // n_head)`（`model.py:54`）按内存布局把 768 维向量**物理切**成 12 段 64 维。每个头的权重只能看自己那 64 维 slice。这是硬分离，但只是**坐标层面**的，并不保证语义不重叠。

2. **训练涌现（语义分离）**：每个头的输出经过 `c_proj` 缝合后参与下游 loss，**梯度回传时每个头收到的信号方向不同**——加上随机初始化的对称性破缺，小差异在迭代中被放大，最终各头学到不同的特征。**这是 emergent 的，不是设计出来的**。

可以把两种"分离"机制对比一下：

| 模型 | 分离机制 | 谁决定边界 | 是否显式 |
|------|----------|------------|----------|
| SVM | 升维 → 线性超平面 | 优化目标里的 max margin | **显式**：边界由支持向量定义 |
| 决策树 | 坐标轴分割 | 信息增益最大化 | **显式**：每个节点一个判据 |
| 多头 attention | 升维 → 多头切片 → 训练涌现 | loss 梯度 + 随机初始化 | **隐式**：没有明确边界，靠优化自动分化 |
| MLP（核技巧侧）| 升维 → 非线性 → 投影 | 反向传播 | **隐式**：3072 维隐空间无法直接解释 |

**这就是为什么 Transformer 既好训练又强大**：它把"特征工程 + 子空间划分"这两件人类做不好的事完全交给梯度下降，自己只负责提供**升维结构**（`3*n_embd`、`4*n_embd`）和**软相似度操作**（`softmax(QK^T)`）。模型规模一大，自动分化出几百个有意义的子空间——而你只需要写 30 行代码。

#### 一张总图：从 SVM 到 Transformer 的"升维+分割"谱系

```
SVM:          x  →  φ(x)             →  w · φ(x) + b
              低维   核函数升维           线性超平面分割（max margin）

MLP:          x  →  c_fc(x) → GELU   →  c_proj
              768    3072   非线性       投影回 768（线性读出）

Attention:    x  →  c_attn → Q/K/V    →  softmax(QK^T) @ V → c_proj
              768    2304   切成 3 份    高维内积检索 + 加权拷贝
                            再切 12 头

Block (一层):  attn(LN(x)) + mlp(LN(x))   ←  两次升维 + 残差
                                              （`model.py:103-104`）
```

每一行都在做"升维 → 在高维做某种简单操作 → 投影回来"。SVM 用一次，Transformer 在每一层做两次，12 层做 24 次。**Transformer 的强大不是因为它发明了新数学，而是因为它把已经验证过的"升维 + 高维线性操作"这个范式工业化、可微化、堆叠化了。**

> **一句话**：SVM 用一次升维 + 一个超平面把世界切开；Transformer 用 24 次升维 + 144 次软检索把世界揉透。前者是单层手术，后者是反复煎炒——做出来的菜是一个数量级的复杂度。

---

## 6.9 数据怎么才能分离开？SVD vs 特征矩阵 vs 看标签差异（一个会让 SVD 翻车的例子）

退一步问一个更基础的问题：**给你一堆样本数据，怎么找到能把它们分开的方向？**

很多人下意识想到三件事——奇异值分解（SVD/PCA）、协方差矩阵的特征向量、或者"看哪几个特征本身就不一样"。它们解决的**根本不是同一个问题**——能不能分开取决于"判别信号在哪里"，以及"你有没有让方法看到标签"。

### 四种思路的本质区别

| 方法 | 优化目标 | 用不用标签 | 找的是什么方向 |
|------|----------|------------|----------------|
| SVD / PCA | 最大化总方差 | **不用** | 数据本身散得最开的方向 |
| 协方差矩阵特征分解 | 同上（数学等价于 PCA） | **不用** | 同上 |
| Fisher LDA | (类间方差) / (类内方差) | **用** | 把类别推得最远、类内压得最紧的方向 |
| 逐特征 t-test / ANOVA | 单维度上类别均值差 | **用** | 哪些维度本身就区分类别 |
| K-means | 簇内方差最小 | **不用** | 假设球状簇、找重心 |

**关键**：SVD 和协方差矩阵特征分解解决的是"数据自己的几何结构"，但**几何结构和类别结构是两回事**。

### 一个让 SVD 完全翻车的具体例子

构造一个二维数据集——判别信号在 x 轴上很弱，噪声在 y 轴上很强：

```python
import numpy as np
np.random.seed(0)

# 类 A：x ≈ +1，y 是大噪声
A = np.column_stack([np.random.randn(500) * 0.1 + 1.0,
                     np.random.randn(500) * 5.0])
# 类 B：x ≈ -1，y 是同样的大噪声
B = np.column_stack([np.random.randn(500) * 0.1 - 1.0,
                     np.random.randn(500) * 5.0])

X = np.vstack([A, B])                 # (1000, 2)
y = np.array([0]*500 + [1]*500)
```

数据画出来是这样：x 方向上两类分得很开（+1 vs −1，类内 std=0.1），y 方向上完全混在一起但 std=5.0。**真正的判别方向是 x 轴**。

#### 思路一：SVD（PCA）—— 翻车

```python
Xc = X - X.mean(0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
print(Vt[0])    # ≈ [0, 1]   ← 第一主成分指向 y 轴！
print(S**2 / (len(X)-1))  # 方差：[~25, ~1.02]
```

**SVD 选了 y 轴**——因为 y 方向方差是 25，x 方向只有 ~1。但**沿 y 轴投影下去两类完全混在一起**，分类准确率 ≈ 50%。判别信号被当成噪声扔掉了。

协方差矩阵特征分解给出的结果**完全一样**——因为 `cov(X) = (1/(n-1)) X^T X`，PCA 和它的特征分解是同一件事的两种表述。

#### 思路二：Fisher LDA —— 找对方向

```python
mu_A, mu_B = A.mean(0), B.mean(0)
S_w = np.cov(A.T) + np.cov(B.T)       # 类内散度
w   = np.linalg.inv(S_w) @ (mu_A - mu_B)
w  /= np.linalg.norm(w)
print(w)                              # ≈ [1, 0]   ← 直接找到 x 轴
```

LDA 选了 x 轴——因为它**用了标签**，知道"沿哪个方向投影后两类的均值差最大、类内抖动最小"。沿 x 轴投影分类准确率 ≈ 100%。

#### 思路三：逐特征 t-test —— 最朴素也最直接

```python
from scipy.stats import ttest_ind
print(ttest_ind(A[:,0], B[:,0]).pvalue)   # x 维：≈ 0       ← 显著
print(ttest_ind(A[:,1], B[:,1]).pvalue)   # y 维：≈ 0.6     ← 不显著
```

也直接告诉你"x 是判别特征，y 不是"。

### 教训：方差大 ≠ 能区分类别

这是机器学习里最反直觉的坑：

> **PCA 找的是"数据自己散得最开的方向"，不是"类别分得最开的方向"。**
> 类别信号小、噪声方向大时，SVD 会把判别信号当成噪声扔掉。

所以回到用户的问题——"是 SVD？是特征矩阵？是看两类样本特征不一样？还是看 k 标签不一样？"——答案是：

- **没有标签**：只能靠 SVD / PCA / t-SNE / UMAP 之类的无监督几何方法，前提是**类别结构恰好和数据的主方差方向对齐**。否则注定失败。
- **有标签、线性可分**：用 Fisher LDA 或 logistic regression，让方法**看见标签**，直接找判别方向。
- **有标签、想知道哪些维度有用**：逐特征 t-test / ANOVA / mutual information——简单粗暴，常常就够。
- **有标签、非线性**：SVM + 核函数，或直接训神经网络。

**简而言之：能不能分开，关键是"判别信号在哪"，以及"你的方法有没有看到标签"。** 同样一份数据，SVD 看不到标签所以被噪声方差骗到 y 轴；LDA 一旦用上标签，立刻指向 x 轴。

### Transformer 是哪一种？—— "有监督的非线性 LDA"

回到 attention（`model.py:53-70`）：

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = F.softmax(att, dim=-1)
y   = att @ v
```

这里**没有 PCA、没有 LDA、没有 K-means**——但它做的事更接近 LDA 而不是 PCA：

- `c_attn` 的权重**不是按方差最大化**训出来的，而是按下游 cross-entropy loss 训出来的（`model.py:192`）。
- Loss 来源于"下一个 token 是什么"——这是一种**自带的标签**。每次反向传播都在告诉 Q/K/V 投影矩阵：
  > "朝着让正确 token 概率上升的方向调"——这是**有监督**的判别式学习。

所以 Transformer 不需要先跑 PCA 再训分类器——它的 `c_attn` / `c_fc` 投影矩阵本来就是被任务标签直接监督出来的"LDA 的高维非线性版"。多头让它**同时学十几套这种判别投影**，每个头的 K 矩阵都是"在某个子任务上区分类别最强的方向集合"。

如果当年用 PCA 替代 attention（"先把 token embedding 降维到主方差方向再做下一步"），结果会和上面 SVD 翻车同款——**主方差方向上往往恰好不是预测下一个 token 最关键的方向**。Transformer 不犯这个错，因为它从一开始就让标签信号通过梯度告诉投影矩阵该往哪转。

#### 一张表收拢

| 你想分什么数据？| 推荐方法 | 为什么 |
|-----------------|----------|--------|
| 无标签、求降维可视化 | SVD / PCA / t-SNE / UMAP | 没标签，只能靠几何结构 |
| 有标签、线性可分 | Fisher LDA / Logistic Regression | 用标签找判别方向 |
| 有标签、想筛特征 | 逐特征 t-test / mutual information | 直接找单维度差异 |
| 有标签、非线性、特征清晰 | SVM + RBF kernel | 升维 + 超平面 |
| 有标签、序列数据、特征复杂 | Transformer | 升维 + 软检索 + 多套判别投影 |

> **一句话**：能不能分开数据，关键不是"用什么数学"，而是"判别信号在哪"，以及"你有没有让方法看到标签"。SVD 看不到标签所以会被噪声方差骗；LDA、Transformer 看标签，才会去找类别真正不一样的那一维。

---

## 7. 训练个人 GPT = 在你的语料上重建搜索索引

理解了"GPT = 高维空间里的搜索引擎"，训练个人 GPT 这件事就立刻不神秘了：

> **训练 = 用你的数据重建权重矩阵 W，使得 attention 检索的结果偏向你想要的答案。**

具体在 CodeGPT 里发生的事：

1. **准备语料 = 准备被搜索的文档库**（`data/python_code/prepare.py`、`data/github_code/prepare.py`）。
   - 切 token、写成 `train.bin`/`val.bin` 的 uint16 流——**这就是建索引的过程**。

2. **训练 = 让搜索在你的语料上"答得准"**（`train.py`）。
   - `F.cross_entropy(logits, targets, ignore_index=-1)`（`model.py:192`）= "下一个 token 应该是 X，可你搜出的最高分是 Y"——loss 大，梯度反向把 W 调一点点。
   - 几万到几亿次这种调整后，W 收敛成"在我的代码风格下答得最准的搜索引擎"。

3. **微调 = 把已有索引在你的小语料上重排**（`config/finetune_codegpt.py` + `init_from='gpt2*'`）。
   - 从 GPT-2 的 W 开始（已经是一个全网大搜索引擎），用你的代码再训几千步——相当于让搜索结果更偏向你这个领域。

4. **FIM 训练 = 教搜索引擎也能"中间填空"**（`tokenizer.py:apply_fim_transform`，`train.py:get_batch`）。
   - 50% 概率把 `prefix + middle + suffix` 重排成 `<|fim_prefix|> P <|fim_suffix|> S <|fim_middle|> M`。
   - 训练后，模型不仅能"续写"，还能"在两段已知代码之间补中间"——这是 Copilot / Cursor 内联补全背后的机制。**搜索维度多了一个**：从"给前缀找后缀"扩展到"给前后缀找中间"。

5. **RAG = 不重训，临时往 query 里塞检索结果**——`docs/RAG_VS_SFT.md` 详细讲过。RAG 是"在 GPT 这个高级搜索之外再套一层传统向量搜索"，把命中的 chunk 拼进 prompt，让 GPT 做最后的综合。

**所以"训个人 GPT"的真实含义**：你不是在造一个新的 AI，你是在**用你的数据重建一个搜索索引**——只不过这个索引同时具备生成、综合、改写能力，因为它的"召回"是在 12 层高维空间里完成的。

---

## 8. 实战：把"用好 GPT"翻译成"用好搜索"

既然 GPT 是搜索引擎，那"会用 GPT"就等于"会写好的搜索 query"。把过去你提报错时知道的所有搜索技巧迁移过来：

| 搜索时代的技巧 | GPT 时代的对应 |
|----------------|----------------|
| 关键词加引号锁死精确匹配 | prompt 里贴**完整报错信息**而不是改写 |
| 加 `site:stackoverflow.com` 限定域 | prompt 里加 `回答只引用 PyTorch 官方文档` |
| 减号排除噪音（`-tutorial`） | prompt 里加 `不要给入门级解释` |
| 同时开几个 tab 横向对比 | 让 GPT 一次列 **3 种方案 + 各自 trade-off** |
| 直接搜函数签名而非自然语言 | prompt 里贴**真实代码片段**比口头描述强十倍 |
| 用英文搜效果远好于中文 | 用英文 prompt + 让它中文回答（embedding 空间英文密度高） |
| 翻到第二页找冷门正解 | 让 GPT 给"**less obvious**"或"反直觉"的答案 |

**核心 mental model**：你写的每个 prompt 都是在高维空间里挑选一个区域。Prompt 越具体、越接近训练分布、越带"高熵特征"（罕见词、代码、报错），召回的区域就越准——和 Google 报错搜索完全同构。

---

## 9. 小结：GPT 不神秘，它只是把搜索做到了极致

回到一开始那条线：

```
关键词搜索 → 向量搜索 → 高维空间搜索 → GPT
   |             |              |          |
 倒排索引     Faiss/HNSW     embedding   12 层 attention
 字符串=     内积=相似       学习到的     QK^T + softmax + @V
                                          每层一次，每头一次
```

每一代的进步都是**把"搜索"这件事在更高维、更连续、更可微的空间上重做一遍**。GPT 是顶点——它把搜索、过滤、排序、生成这四步压进了同一次 forward，每生成一个 token 都做一遍。

理解这件事的几个直接好处：

1. **去神秘化**：GPT 不是魔法，它是一个把每一层都做向量检索的搜索引擎。`F.softmax(q @ k.T) @ v` 就是核心。
2. **训自己模型时不慌**：你只是在自己的语料上重建索引；模型小没关系，索引精准就行（10M 的 CodeGPT-small 在自己的代码上能比 GPT-4 更懂你的命名风格，因为你的数据在 GPT-4 训练集中权重接近 0）。
3. **会写 prompt**：把"写 prompt"看成"写搜索 query"——具体、贴报错、用英文、给约束、要 trade-off。
4. **判断什么任务做不到**：搜索引擎查不到没收录过的内容，GPT 也一样。所以才需要 RAG（拼一个外部索引进去）和 SFT（把新内容压进权重）。
5. **理解多层的意义**：单层 attention 只能做一次平面搜索，12 层让搜索在层次结构上展开——像把"搜文档 → 读文档 → 综合 → 改写"压进一次前向传播。

---

## 延伸阅读

- [`COMPRESSION_IS_INTELLIGENCE.md`](COMPRESSION_IS_INTELLIGENCE.md) — 为什么"搜索 + 综合"等价于压缩；`F.cross_entropy` 在度量什么。
- [`RAG_VS_SFT.md`](RAG_VS_SFT.md) — 改 `idx`（外挂搜索）还是改 `W`（重训搜索引擎），决策表 + 评估方法。
- [`DEEP_DIVE.md`](DEEP_DIVE.md) — 注意力机制从加法注意力到缩放点积注意力的进化。
- [`PHYSICS_AND_DEEP_LEARNING.md`](PHYSICS_AND_DEEP_LEARNING.md) — Attention = 连续型 Hopfield 网络，从能量最小化的角度看"搜索"。
- [`SFT_RL_INFERENCE_MECHANICS.md`](SFT_RL_INFERENCE_MECHANICS.md) — 训练写权重、推理用权重 + 脚手架。

> **一句话总结**：GPT 是一个 12 层、每层 12 头、共做 144 次"高维向量内积检索"的搜索引擎，权重 W 就是它的索引，训练就是建索引，prompt 就是 query，生成就是把检索到的内容加权拼接出来。
