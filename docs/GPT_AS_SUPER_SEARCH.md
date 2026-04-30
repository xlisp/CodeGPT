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
