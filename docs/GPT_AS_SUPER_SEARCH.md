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
3.5. [工业级向量搜索：Faiss / HNSW 与近似最近邻（ANN）](#35-工业级向量搜索faiss--hnsw-与近似最近邻ann)
3.6. [搜索 → 推荐 → GPT：协同矩阵把"含义"一步步藏进黑箱](#36-搜索--推荐--gpt协同矩阵把含义一步步藏进黑箱)
4. [第三代：高维空间里的搜索（embedding everywhere）](#4-第三代高维空间里的搜索embedding-everywhere)
5. [第四代：GPT — 在每一层都做一次搜索](#5-第四代gpt--在每一层都做一次搜索)
6. [Attention 就是"软 SQL"：QK^T 是 WHERE，softmax 是过滤，@V 是 SELECT](#6-attention-就是软-sqlqkt-是-wheresoftmax-是过滤v-是-select)
6.5. [QKV 三角投影：把每个 token "揉碎"成三个角色](#65-qkv-三角投影把每个-token-揉碎成三个角色)
6.6. [GPT 是抄答案拼接器：迁移和幻觉是同一枚硬币](#66-gpt-是抄答案拼接器迁移和幻觉是同一枚硬币)
6.7. [多头 = 多个"特征矩阵"：QKV 为什么是 Transformer 成功的核心](#67-多头--多个特征矩阵qkv-为什么是-transformer-成功的核心)
6.8. [子空间是怎么分离的？SVM 升维 vs Transformer 升维](#68-子空间是怎么分离的svm-升维-vs-transformer-升维)
6.9. [数据怎么才能分离开？SVD vs 特征矩阵 vs 看标签差异（一个会让 SVD 翻车的例子）](#69-数据怎么才能分离开svd-vs-特征矩阵-vs-看标签差异一个会让-svd-翻车的例子)
6.10. [从搜索缓存到 KV cache：同一种缓存哲学（含 Ollama / vLLM 原理与 undo 机制）](#610-从搜索缓存到-kv-cache同一种缓存哲学含-ollama--vllm-原理与-undo-机制)
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

## 3.5 工业级向量搜索：Faiss / HNSW 与近似最近邻（ANN）

第二代向量搜索的伪代码 `doc_vecs @ q_vec` 看起来很优雅，但**在亿级文档库上跑一次要算 N×d 次乘加**——N=1e9、d=768 时单次查询要 7.6×10¹¹ 次浮点运算，几十秒起步。所以工业界从来不跑朴素内积，而是用 **ANN（Approximate Nearest Neighbor，近似最近邻）**——牺牲一点点召回准确率，把时间复杂度从 O(N) 降到 O(log N) 甚至 O(1)。

ANN 是 RAG、向量数据库（Pinecone / Milvus / Qdrant / Weaviate）、Faiss / HNSWLib 这一整层基础设施的核心。**搞清 ANN，"高维空间搜索"这一代才不悬空。**

### 3.5.1 三类主流 ANN 算法

| 算法 | 核心思想 | 时间复杂度 | 召回率 | 典型实现 |
|------|----------|------------|--------|----------|
| LSH (Locality-Sensitive Hashing) | 设计哈希函数让"相近向量大概率落同一桶" | O(1) 平均 | 中 | Faiss `IndexLSH` |
| IVF (Inverted File) | 先 K-means 把向量聚成 nlist 个 cluster；查询时只搜最近的 nprobe 个 cluster | O(N/nlist · nprobe) | 高（调参后）| Faiss `IndexIVFFlat` |
| HNSW (Hierarchical Navigable Small World) | 多层跳表 + 小世界图：上层稀疏快速定位、下层稠密精确搜索 | O(log N) | 极高（>0.99）| Faiss `IndexHNSWFlat`、HNSWLib |
| PQ (Product Quantization) | 把 d 维向量切成 m 段，每段 K-means 量化成 256 个码字——内存压缩 8-32 倍 | 视组合而定 | 中-高 | 常和 IVF 拼成 `IndexIVFPQ` |

**HNSW 是当前事实标准**：召回率高、查询稳定、动态插入友好。理解一个 HNSW 就理解了 95% 的工业向量库。

### 3.5.2 一段能跑的 RAG 检索代码

```python
# pip install sentence-transformers faiss-cpu
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1) embedding model：把文本压成 384 维向量
encoder = SentenceTransformer('all-MiniLM-L6-v2')   # 22M 参数的小模型
docs = [
    "Python dict update merges keys in-place",
    "JS spread operator {...a, ...b} merges objects",
    "Rust borrow checker prevents data races",
    "TypeScript generics constrain type parameters",
    # ... 假装有 1e6 条文档
]
doc_vecs = encoder.encode(docs, normalize_embeddings=True)  # (N, 384) float32

# 2) 建 ANN 索引：HNSW
d = doc_vecs.shape[1]
index = faiss.IndexHNSWFlat(d, M=32)        # M = 每个节点保留的邻居数
index.hnsw.efConstruction = 200              # 建图时的搜索宽度（越大越准、越慢）
index.add(doc_vecs)                          # 插入向量，自动构图

# 3) 查询：把 query 也 encode，再到图里找最近邻
query = "how do I combine two python dictionaries"
q_vec = encoder.encode([query], normalize_embeddings=True)
index.hnsw.efSearch = 64                     # 查询时的搜索宽度
D, I = index.search(q_vec, k=5)              # D=距离，I=文档下标
for rank, (dist, doc_id) in enumerate(zip(D[0], I[0])):
    print(f"#{rank}  dist={dist:.3f}  {docs[doc_id]}")
```

整条流水线分三步——**encode（升维到 384 维）→ 建索引（在高维空间里搭一张可导航的图）→ search（图上跳几步找邻居）**。这就是"embedding model + ANN"的全部含义。

### 3.5.3 HNSW 在搜什么？——一张图

```
Layer 2 (稀疏):    A ─────────────── F  ← 长跳：从入口快速跨域
                   │                 │
Layer 1 (中层):    A ───── C ─────── F
                   │       │         │
Layer 0 (稠密):    A─B─C─D─E─F─G─H─I─J  ← 全量节点 + 短跳精搜

查询路径：从 Layer 2 顶层入口出发 → 贪心走向最近邻 → 下一层继续 → Layer 0 精细搜
```

**一句话**：HNSW 把"亿级向量的 NN 搜索"变成"在一张多层小世界图上从入口跳到目标"——`O(log N)` 步搞定。和 Google 早年的"先粗排再精排"完全同构，只是粗排/精排都换成了高维向量内积。

### 3.5.4 Embedding model 自己也是个迷你 transformer

注意上面的 `SentenceTransformer('all-MiniLM-L6-v2')`——这就是一个 6 层、384 维的迷你 BERT。它把变长文本压成定长向量的方法是：

```python
# 伪代码：sentence-transformers 内部
def encode(text):
    tok_ids = tokenizer(text)                      # (T,)
    h = bert_forward(tok_ids)                       # (T, 384)：每个 token 一个向量
    sentence_vec = mean_pool(h, attention_mask)     # (384,)：mean pooling
    return F.normalize(sentence_vec, p=2, dim=-1)
```

**所以现代向量搜索 = 一个小 transformer 当 encoder + ANN 当索引**。两者都是 transformer 时代的产物——上层（GPT）做生成，下层（embedding 模型）做检索。RAG 把两层缝在一起：先用下层的 ANN 召回 top-k，再把 top-k 拼进上层 GPT 的 prompt。

### 3.5.5 ANN 和 attention 是什么关系？

attention 也是"在一堆向量里找最相关的"——但它**不用 ANN**，因为：

- 上下文长度 T 通常 ≤ 几千到几十万，**O(T²) 直接算精确内积比 ANN 还快**——`q @ k.T`（`model.py:66`）就是穷举打分。
- attention 需要**梯度可导**，ANN 算法（哈希、聚类、图）大多不可微。
- ANN 是"亿级文档里挑 top-5"，attention 是"几千 token 里软加权所有"——任务粒度不同。

> **分工清楚**：ANN 处理"模型外 / 全网 / 知识库"级别的搜索（RAG 的 retrieval 那一步）；attention 处理"模型内 / 当前上下文"级别的搜索（每次 forward 内部）。两者都在做内积检索，**只是数据库的大小、是否要可微、是精确还是近似不同**。

理解这一层，下一节再讲"GPT 在每一层都做一次搜索"就特别自然——ANN 已经把"高维内积检索"做到了极致工程化，GPT 把同样的操作搬进了网络内部、变成可微的、并且堆了 144 次。

---

## 3.6 搜索 → 推荐 → GPT：协同矩阵把"含义"一步步藏进黑箱

到这里我们看到了"向量搜索"的工业实现，但**为什么人类会相信"几百维的浮点向量能代表一段文本/一首歌/一个人"**？这个信念不是 GPT 时代才长出来的——它来自**推荐系统**这条平行支线。从 2006 年 Netflix Prize 开始，推荐系统已经把"矩阵 + 内积"这套范式打磨了十几年，**GPT 只是把这条路走到了极致**。

把搜索、推荐、GPT 三件事并排看，就能看清"矩阵的可解释性是怎么一步步消失的"——以及为什么这种"消失"反而是泛化能力的来源。

### 3.6.1 起点：可解释的特征矩阵

最朴素的做法——给每个用户写一行明确含义的特征：

```python
# 显式特征矩阵：每一列都有名字
#                gender  age  income  is_student  likes_action  likes_romance
user_features = [
    [1.0,        25,   30000,    0,           1,             0],   # Alice
    [0.0,        38,   80000,    0,           0,             1],   # Bob
    [1.0,        20,    5000,    1,           1,             1],   # Cindy
]
```

**每一维都对应一个真实概念**——你拿出第 4 列就是 income，第 5 列就是 likes_action。这是传统机器学习/数据库里的特征工程：人类先决定"什么维度有用"，模型只负责把它们组合成预测。

**问题**：你永远写不全。"她其实喜欢有动作场面但又文艺一点的电影"——这种维度你没有列。所有显式特征矩阵都漏在这里。

### 3.6.2 协同过滤：不写特征，只写"谁喜欢谁"

Netflix Prize（2006-2009）开启了另一条路。**完全不问**"用户是什么样的人"或"电影是什么类型"——只看一张稀疏评分表 R：

```
        Movie₁  Movie₂  Movie₃  Movie₄  ...  Movie_N
User₁:    5      ?      3      ?     ...    ?
User₂:    ?      4      ?      2     ...    5
User₃:    1      ?      ?      4     ...    ?
...
User_M:   ?      3      5      ?     ...    ?
```

R 是一个 (M, N) 的矩阵，绝大多数格子是缺失的。**任务**：把缺失的格子补出来——"User₁ 会给 Movie₂ 打多少分？"

注意这里**已经放弃了"特征"的概念**：不再有 gender、age、likes_action——只有"用户 u 给物品 i 打了分 r"这种纯关系数据。

### 3.6.3 矩阵分解（MF）：把 R 拆成两个隐向量矩阵

关键洞见：**R ≈ U Vᵀ**，其中

- **U: (M, k)** — 每个用户一个 k 维隐向量（k 远小于 N，比如 k=64）
- **V: (N, k)** — 每部电影一个 k 维隐向量

预测 User_u 对 Movie_i 的评分就是**两个向量的内积**：`r̂[u, i] = U[u] · V[i]`。

```python
# 矩阵分解 = 两个 nn.Embedding + 一次内积
import torch, torch.nn as nn, torch.nn.functional as F

class MF(nn.Module):
    def __init__(self, n_users, n_items, k=64):
        super().__init__()
        self.U = nn.Embedding(n_users, k)        # 用户隐向量矩阵
        self.V = nn.Embedding(n_items, k)        # 物品隐向量矩阵
    def forward(self, u, i):
        return (self.U(u) * self.V(i)).sum(-1)   # 内积 = 预测评分

# 训练目标：让内积接近真实评分
loss = F.mse_loss(model(u, i), rating)
```

**这就是 GPT embedding 的祖先**。和 `model.py:147` 的 `wte = nn.Embedding(vocab_size, n_embd)` 是同一个 API、同一种数学。

但这一步有一个深刻的代价：**U[u] 这 64 维数字不再有名字**。你拿出第 7 维不是 income 也不是 age——它是"梯度下降为了让所有内积尽量贴近真实评分而自动找到的某个抽象方向"。事后做主成分分析，你**有时候**能解读出"这一维像是动作 vs 文艺、那一维像是高雅 vs 爆米花"——但大部分维度无法命名。

> **第一次"含义消失"**：从"显式特征矩阵 → 隐向量矩阵"。维度数量大幅压缩（N=10000 → k=64），换来的是泛化（能预测任何 user-item 对），代价是可解释性。这是机器学习从"特征工程"走向"表征学习"的第一个分水岭。

### 3.6.4 概率视角：从"评分回归"到"喜欢的条件概率"

下一步是把"评分回归"换成"概率分布"。BPR (Bayesian Personalized Ranking)、NCF (Neural Collaborative Filtering) 等方法把内积喂进 sigmoid 或 softmax：

```python
# 用户 u 在 N 个 item 上的概率分布
logits = U[u] @ V.T              # (N,)：一次性算出对所有 item 的内积分数
probs  = F.softmax(logits, -1)   # 归一化为"喜欢每个 item 的概率"
```

**到这里数学已经和 GPT 完全一样了**——只是把 "user / item" 换成 "context / next token"，把"评分"换成"下一个 token 的概率"。看 `model.py:191`：

```python
logits = self.lm_head(x)        # x: (B, T, n_embd), lm_head.weight: (vocab, n_embd)
                                 # 等价于 x @ lm_head.weight.T → (B, T, vocab)
```

这一行就是 `context_vector @ token_matrix.T`——和推荐里的 `U[u] @ V.T` 是**同一个内积**。

更巧的是，CodeGPT 在 `model.py:153` 把 `wte.weight` 和 `lm_head.weight` 绑成同一份：

```python
self.transformer.wte.weight = self.lm_head.weight   # weight tying
```

这等价于推荐里"用户矩阵和物品矩阵共享一份"——token 既是被搜索的 item（`wte` 把 token id 映成向量），也是搜索回答时的候选库（`lm_head` 把上下文向量打分到所有 token）。**weight tying 是矩阵分解里"对称协同过滤"在 GPT 里的最自然形态**。

### 3.6.5 word2vec：矩阵分解走进语言

2013 年的 word2vec 把 MF 直接搬到了语言上：

- **R 矩阵** = (中心词, 上下文词) 的共现矩阵（或 PMI 矩阵）
- **U** = word embedding，**V** = context embedding
- **目标** = 预测某个词周围会出现什么词（softmax over vocab）

word2vec 论文（Levy & Goldberg 2014）证明了 skip-gram + negative sampling 在数学上等价于对 PMI 矩阵做隐式分解。**所以 word embedding 本质上就是"把词当 user、把上下文词当 item"的协同过滤**——一个词"喜欢"哪些上下文，等于它在那些上下文窗口里频繁共现。

这一步把"推荐里的隐向量"和"语言里的语义向量"接上了——两者从此是同一个东西。

### 3.6.6 GPT：把这件事做到 144 层

GPT 把"矩阵分解 + 内积"这套机制堆叠到了极致。把推荐系统里的角色对应到 `model.py` 里的对象：

| 在 GPT 里的位置 | 对应推荐里的什么 |
|-----------------|------------------|
| `wte: (50304, 768)` | 物品 latent 矩阵 V（每个 token 一个向量） |
| 第 t 个位置的隐状态 `x[:, t, :]` | 用户 latent 向量 U[u]——但**它是当前上下文动态生成的**，不是查表查出来的静态用户 |
| `logits = x @ wte.T` | 内积打分 = 这段上下文对每个候选 token 的"喜爱程度" |
| `softmax(logits)` | 下一个 token 的概率分布 |
| 12 层 × 12 头的 K / Q / V 投影矩阵 | 多组"特征矩阵"（参考 §6.7）——但每一维都不可解释 |

最大的差别在中间那一行：**推荐里 U[u] 是一个静态向量（用户是固定的），GPT 里 U 是上下文动态算出来的**。每多读一个 token，"用户向量"就被 attention 重新计算一次——所以同一个用户问"recommend me a movie"和"我刚看了《肖申克的救赎》，再来一部"会得到完全不同的 U，召回完全不同的"电影"。

**这就是 GPT 之于推荐系统的飞跃**：用户画像不再是矩阵里的某一行，而是**每次输入即时由 attention 生成的高维向量**——而 attention 本身又是通过 144 次 QK 内积检索拼出来的（§6.7）。从一次内积到 145 次内积，从静态用户到动态用户——但底层操作没变。

### 3.6.7 三个时代的"矩阵 + 概率"对比

| 时代 | 关键矩阵 | 维度有含义吗 | 训练信号 | 概率怎么来的 |
|------|----------|--------------|----------|--------------|
| 特征工程 | user feature matrix（手写） | **完全有**：gender / age / income 等 | 监督标签 | 逻辑回归 / 朴素贝叶斯手算条件概率 |
| 协同过滤 MF | U: (M, k), V: (N, k) | **几乎没有**：k 维隐向量自动学出 | 评分 MSE / BPR pairwise | `sigmoid(U[u] · V[i])` |
| word2vec | word embedding: (vocab, d) | **完全没有**：d 维分布式表示 | 周围词预测 + negative sampling | `softmax over vocab` |
| GPT | `wte` + 144 个 K / Q / V 投影矩阵 | **完全没有** + 上下文动态生成 | next-token cross-entropy（`model.py:192`）| `softmax(x @ wte.T)` |

**一条主线**：每往后一步——

1. **矩阵更大**：M×k → vocab×d → 144 套 (n_embd, n_embd) 投影。
2. **含义更模糊**：人能命名的列 → 部分能解释的隐维 → 完全黑箱。
3. **损失更间接**：直接监督评分 → 拟合共现 → 拟合 next-token。
4. **概率分布更宽**：5 个评分等级 → 几万词 → 5 万 token，每一步都给完整分布。

**但底层操作始终是同一件事——内积 + softmax**。从"猜 5 部电影评分"到"在 50304 个 token 上给完整概率分布"，本质都是：**学一组隐向量，让内积 + softmax 拟合观测到的共现/偏好**。

> **第二次"含义消失"**：从协同过滤的 64 维到 GPT 的 768 维 + 144 套投影。维度多得无法逐个解释，但正因如此模型能同时建模语法、语义、风格、跨语言对齐、算法骨架等无数个我们叫不出名字的子结构（§6.7）。**可解释性的丧失是泛化能力的代价，也是它的来源**。

### 3.6.8 工业推荐的"召回 + 精排"和 GPT 的"top-k + softmax"是同构的

这条联系还有一个工程层面的体现。现代推荐系统的两阶段：

```
亿级物品池
   ↓  Recall（召回）：用 ANN 在 item embedding 库里找 top-1000   ← §3.5 的 HNSW
top-1000 候选
   ↓  Rank（精排）：用一个小 DNN 对 1000 个 item 重新打分
top-10
   ↓  Re-rank（重排）：业务规则、多样性、新鲜度过滤
最终展示
```

再看 GPT 生成一个 token 的对应阶段（`model.py:287-303`）：

```python
# "全量打分"：lm_head 给所有 50304 个 token 打分（vocab 不大，直接穷举内积）
logits = self.lm_head(x[:, [-1], :])

# "精排"：top-k 截断
v, _ = torch.topk(logits, top_k)
logits[logits < v[:, [-1]]] = -float('Inf')
# 或：top-p（nucleus）累积概率截断

# "采样"：softmax 后随机抽
probs = F.softmax(logits, -1)
idx_next = torch.multinomial(probs, 1)
```

**结构完全同构**——只是 GPT 里 vocab 才 5 万级、可以直接穷举内积；工业推荐 item 上亿，必须用 ANN 先召回。两边都在做"内积打分 → 截断 → 概率分布 → 采样"。

> 这也解释了为什么"用 GPT 做推荐"在 2024 年成了工业新方向（生成式推荐 / Generative Recommender）——**这两件事本来就是同一种数学**，把 item 当 token、把用户历史当 prompt，GPT 直接就能做推荐。区别只是 token 词表不再是自然语言，而是 item id。

### 3.6.9 一句话收拢

> **从搜索到推荐到 GPT，一条线索贯穿始终——矩阵越变越大、维度越变越无法命名、损失函数越变越间接，但底层操作始终是"内积 + softmax"。** 理解了协同过滤里那个简单的 `(U[u] * V[i]).sum(-1)`，你就理解了 GPT 输出层那个 `x @ wte.T` 的全部精神先驱。GPT 不是另一种数学，它是同一个数学被推到了极致——**把一份显式可解释的人造特征矩阵，换成了 144 套梯度下降自己长出来的隐特征矩阵**。

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

## 6.10 从搜索缓存到 KV cache：同一种缓存哲学（含 Ollama / vLLM 原理与 undo 机制）

把 GPT 当搜索引擎看的好处之一是——**搜索引擎所有的缓存技巧，在 GPT 这边都有对应**。

| 搜索引擎一侧 | GPT 一侧 |
|--------------|----------|
| Google 把热门 query 的结果缓存起来，第二次同样 query 直接命中 | 自回归生成里前 t-1 个 token 的 K/V 完全相同，缓存即可 |
| CDN 边缘节点缓存静态结果 | vLLM 的 PagedAttention 把 KV 切成 block 跨请求共享 |
| 缓存失效 = 撤销缓存条目 | undo / rewind = 丢弃 KV 矩阵尾部若干行 |
| 多个 user 共享同一份热门页面 | 多 request 共享同一段 system prompt 的 KV（prefix caching）|

所以**KV cache 不是 transformer 时代的新发明，它就是搜索缓存在自回归生成场景下的具体形态**。下面把这层关系说穿。

### 6.10.1 自回归生成的天然冗余

先看一下 CodeGPT 当前的 `generate`（`model.py:275-308`）：

```python
for _ in range(max_new_tokens):
    idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
    logits, _ = self(idx_cond)          # ← 每一步都把整个序列从头跑一遍 forward
    logits = logits[:, -1, :] / temperature
    # ... 采样 ...
    idx = torch.cat((idx, idx_next), dim=1)
```

每生成一个 token，`self(idx_cond)` 会**把已经处理过的所有 token 再算一遍**。第 100 个 token 时，前 99 个 token 在 12 个 Block 里的 K、V 矩阵已经在第 99 步算过了——但代码里没有保存，于是第 100 步重新算了一次。第 101 步又算了一次……

总开销是 **O(T²) 而不是 O(T)**——T 越长越亏。CodeGPT 是教学项目，所以保留了这个最朴素的实现；任何工业部署都不会这样做。

### 6.10.2 KV cache 是什么——一句话定义

> **把每一层每个 head 已经算出来的 K 和 V 矩阵存下来，下一步生成时只算"新 token 那一行"的 K/V，再 append 到缓存末尾。**

attention 那块（`model.py:53-70`）的关键观察是：

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)   # 每个 token 独立算 Q/K/V
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))   # 因果掩码
att = F.softmax(att, dim=-1)
y   = att @ v
```

因果掩码保证了**第 t 个位置只能看 ≤ t 的 K/V**——也就是说**前面位置的 K/V 不依赖后面位置**。一旦算出来就永远不变。这就是缓存的合法性来源。

带 KV cache 的 generate 伪代码：

```python
# 伪代码：带 KV cache 的自回归生成
def generate_with_cache(model, idx, max_new_tokens):
    # 第一次：处理整个 prompt，所有层的 K/V 都缓存下来
    kv_cache = [{"k": None, "v": None} for _ in range(model.config.n_layer)]
    logits = model.forward_with_cache(idx, kv_cache)         # 全量 forward + 写 cache

    for _ in range(max_new_tokens):
        idx_next = sample(logits[:, -1, :])
        # 关键：只输入新 token，每层只算一行 K/V，append 到 cache
        logits = model.forward_with_cache(idx_next, kv_cache)  # 增量 forward
        idx = torch.cat([idx, idx_next], dim=1)
    return idx
```

复杂度从 **O(T²) → O(T)**——每生成一个 token 只做 **O(T)** 的工作（查 cache 算 attention）而不是 O(t²)。

### 6.10.3 KV cache 的内存账

工业部署里 KV cache 通常**比模型权重还占显存**：

```
KV cache 大小 = 2 × num_layers × num_heads × seq_len × head_dim × dtype_bytes × batch
                ↑ K 和 V 各一份
```

代入 Llama-3 70B 的数字：

```
2 × 80 × 64 × 8192 × 128 × 2 (fp16) × 1 = 21.5 GB    # 单 batch、8K 上下文
2 × 80 × 64 × 32768 × 128 × 2 (fp16) × 1 = 86 GB     # 32K 上下文，已经爆 80GB H100
```

**所以推理引擎的核心命题，就是 KV cache 怎么放、怎么省、怎么共享。** 这就把 Ollama 和 vLLM 是干什么的解释干净了。

### 6.10.4 Ollama 的本质：把 KV cache 放进 CPU/GPU 友好的连续内存 + 量化

Ollama 是 [llama.cpp](https://github.com/ggerganov/llama.cpp) 的封装，核心优化全在 KV cache 这一层：

- **GGUF 格式**：把权重和 KV cache 都按"连续内存 + mmap"组织，CPU 推理友好。
- **KV cache 量化（Q4_K / Q8_0）**：把 fp16 的 K/V 压到 4-bit 或 8-bit 整数，显存直接砍 2-4 倍——精度损失通常 < 1%。
- **环形 buffer**：上下文超过 `n_ctx` 时滚动覆盖最早的 K/V（牺牲早期记忆换长会话，对应"搜索缓存里 LRU 淘汰"）。
- **CPU/GPU 混合**：把热的 K/V 放 GPU，冷的放 CPU RAM——和 CDN 边缘节点 vs 中心库的关系一样。

**Ollama 跑得动 70B 模型的关键就在这里**——不是模型小了，是 KV cache 被精打细算地放置/量化了。

### 6.10.5 vLLM 的本质：PagedAttention = 把 KV cache 当虚拟内存分页

vLLM 解决的是**多请求并发**场景下的 KV cache 浪费问题。问题描述：

- 朴素实现给每个 request 预留 `max_seq_len` 那么大的连续 KV 缓冲——但很多 request 实际只生成几百 token，**预留的内存大量空着**。
- 不同 request 如果共享 system prompt 或 few-shot 例子，**前缀的 KV 完全相同，本来可以共享**。

PagedAttention 借鉴操作系统虚拟内存的做法（[vLLM 论文](https://arxiv.org/abs/2309.06180)）：

```
传统：一个 request 一段连续 KV cache
[req-A: ████████████░░░░░░░░░░░] ← 大量预留浪费
[req-B: ████░░░░░░░░░░░░░░░░░░░]
[req-C: ██████████████████████░░]

PagedAttention：把 KV cache 切成固定大小 block（比如 16 token 一块），block table 像页表那样映射
block 池: [B0][B1][B2][B3][B4][B5][B6][B7]...
req-A 的 page table: B0 → B2 → B5
req-B 的 page table: B1 → B3
req-C 的 page table: B0 → B2 → B5 → B6 → B7   ← 前三个 block 和 req-A 共享！
```

这就让 **prefix caching** 直接成立——多个 request 共用同一段 system prompt 时，那段的 KV block **只算一次、被多个请求引用**。这跟 CDN 把热门页面缓存到边缘节点是同一种思想。

vLLM 把推理吞吐量比朴素实现提高 2-24 倍，几乎全部红利来自这一层 KV cache 复用。

### 6.10.6 回到用户的疑问："这就是搜索文章的缓存吗？"

**是的，本质完全相同**——只是对应物从"网页"换成了"K/V 矩阵的某些行"：

| 维度 | 搜索缓存 | KV cache |
|------|----------|----------|
| 缓存键 | query 字符串（或 hash）| 前缀 token 序列 |
| 缓存值 | 检索结果（doc list）| 每层每头的 K, V 矩阵 |
| 命中条件 | query 完全相同 | 前缀 token 完全相同（prefix match）|
| 失效策略 | LRU / TTL | 上下文滑窗 / 显存满了 evict |
| 跨用户共享 | CDN 边缘共享热门页 | vLLM PagedAttention 共享 prefix block |
| 数据结构 | 哈希表 / Redis | block table + 物理 block 池 |

**记忆挂钩**：vLLM 之于 Llama，约等于 Cloudflare 之于一个 Web 站点——都是在请求层做缓存复用，把"重复计算的工作量"摊到接近零。

### 6.10.7 Undo 的代价：为什么"原则上 undo 也可以靠 KV cache"

> **用户的原话**："但是没法 undo，原则上 undo 也可以利用 kv cache 的，要有个抛弃回退 kv 矩阵一部分。"

这话点中了一个真实的工程能力——**KV cache 让 undo / rewind / 多分支生成成为 O(1) 而不是 O(T) 的操作**。

每层 cache 的形状是 `(B, n_head, T, head_dim)`，"回退 k 步"的实现就是一次 slicing：

```python
# 伪代码：回退 k 个 token，让模型从第 (T-k) 位置重新生成
def rewind(kv_cache, k):
    for layer_cache in kv_cache:
        layer_cache["k"] = layer_cache["k"][:, :, :-k, :]   # 丢掉最后 k 行 K
        layer_cache["v"] = layer_cache["v"][:, :, :-k, :]   # 丢掉最后 k 行 V
    # idx 也截短到 [:, :-k]
```

**对比无 cache 的 undo**：得把 idx 截短，然后整个序列从头 forward 一遍——O(T·n_layer) 工作量。
**有 cache 的 undo**：每层做一次切片，**几乎零成本**，前 (T-k) 个位置的 K/V 完全保留下来继续用。

这就是为什么现代推理引擎能轻松实现下面这些场景：

- **多分支采样 / beam search**：从同一个 prefix 出发探索多条生成路径——前缀 KV 共享，每条分支只 fork 自己的尾部。
- **Self-consistency / 温度重采样**：发现刚生成的几个 token 不满意，回退几步、换温度再来一遍。CodeGPT 的 REPL（`repl.py`）如果加 KV cache 就能秒级实现"重采样最后一句"。
- **编辑式 prompt（edit prompt）**：把第 N 个 token 改了，只需丢弃 N 之后的 KV、重算新 token 的 K/V——O(remaining)，不是 O(T)。
- **agentic 多步推理里的回滚**：tool call 失败、rethink、走另一条路——KV cache 的尾部回退是底层机制。

**所以"GPT 没法 undo" 是错觉**——单纯就 transformer 数学而言，undo 是一个 KV 矩阵切片操作。看起来"没法 undo" 只是因为：

1. 大多数 chat UI 没有把 cache 暴露给用户，回退被实现成"重发对话"。
2. 远端 API（OpenAI、Anthropic）出于隔离/安全原因不暴露 cache 句柄。
3. CodeGPT 这种朴素实现根本没有 cache，"undo" 和"重跑"无差别。

**vLLM、SGLang、TensorRT-LLM 这一代推理引擎都把"KV 切片回退"做成了一等公民**——只要你跑的是本地服务，undo 就是 O(1)。

> **一句话总结**：搜索引擎缓存让"重复 query"零成本、缓存淘汰让 undo 自然存在；KV cache 让"重复前缀"零成本、矩阵尾部切片让 undo 自然存在。**同一种缓存哲学，从字符串世界搬到了高维向量世界**——这也是为什么搞 transformer 推理的人和搞 web 缓存的人聊起来毫无门槛。

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
6. **看穿推理引擎的优化重点**：Ollama 和 vLLM 干的事和 CDN / Redis 干的事是同构的——KV cache 就是搜索缓存的高维向量版，prefix sharing 就是 CDN 的边缘节点共享，rewind/undo 就是缓存条目的尾部切片。**理解了"GPT 是搜索引擎"，整个推理栈的工程优化你都能秒懂**。
7. **理解工业向量库**：embedding model + ANN（HNSW / IVF / PQ）就是 attention 的工业化离线版——只是不要梯度、不要可微，换来亿级文档上的 O(log N) 搜索。RAG 的 retrieval 那一层就是这个东西，它和 GPT 内部 attention 的关系是分工而不是替代。

---

## 延伸阅读

- [`COMPRESSION_IS_INTELLIGENCE.md`](COMPRESSION_IS_INTELLIGENCE.md) — 为什么"搜索 + 综合"等价于压缩；`F.cross_entropy` 在度量什么。
- [`RAG_VS_SFT.md`](RAG_VS_SFT.md) — 改 `idx`（外挂搜索）还是改 `W`（重训搜索引擎），决策表 + 评估方法。
- [`DEEP_DIVE.md`](DEEP_DIVE.md) — 注意力机制从加法注意力到缩放点积注意力的进化。
- [`PHYSICS_AND_DEEP_LEARNING.md`](PHYSICS_AND_DEEP_LEARNING.md) — Attention = 连续型 Hopfield 网络，从能量最小化的角度看"搜索"。
- [`SFT_RL_INFERENCE_MECHANICS.md`](SFT_RL_INFERENCE_MECHANICS.md) — 训练写权重、推理用权重 + 脚手架。

> **一句话总结**：GPT 是一个 12 层、每层 12 头、共做 144 次"高维向量内积检索"的搜索引擎，权重 W 就是它的索引，训练就是建索引，prompt 就是 query，生成就是把检索到的内容加权拼接出来。
