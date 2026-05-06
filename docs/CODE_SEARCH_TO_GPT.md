# 从代码语义搜索到 GPT 写代码：五代范式、三次质变与一条贯穿始终的"搜索维度提升"主线

> 程序员"想要一段代码"这件事，三十年里被改写了五次。每一次都不是把上一代擦掉，而是**给搜索空间加一个新维度**：
>
> - **1990s — 词法维度**：`grep` / `ctags` / `ack` —— 在**字符串**这一个维度上找。
> - **2010s — 结构 + 图维度**：Sourcegraph / OpenGrok / LSIF / SCIP / CodeQL / Semgrep / Joern —— 把代码升维成 **AST + 符号引用 + 控制流 + 数据流**的图，用类 SQL / Datalog / Cypher 的查询语言搜。
> - **2013 — 连续向量维度**：Word2Vec / code2vec / CodeBERT —— 第一次把"相似度"从离散字符串/图节点搬到 $\mathbb{R}^{768}$，用内积代替 `==`。
> - **2015 — 时间维度**：Seq2Seq + Bahdanau attention —— 从"找一段已有代码"到"自回归一个 token 一个 token 写出来"，搜索沿时间轴展开。
> - **2020 起 — 多层多头维度**：GPT / Codex / CodeGPT —— 12 层 × 12 头，每生成 1 token = 144 次高维 attention 检索。**不再需要标注配对数据，任何源代码都是训练样本**。
>
> **一条贯穿始终的主线**：每一代都给"搜索"加一个新维度，让"相同含义的代码"在更高维空间里靠得更近。前两代的维度是**离散结构**（字符串、AST、引用图），中间一代是**连续向量**，后两代把搜索本身嵌入了**时间和深度**。三次最关键的质变发生在维度类型切换的边界上：**离散结构 → 连续向量**、**检索 → 生成**、**配对监督 → 自监督序列**。
>
> 本文用本仓库 `model.py` / `tokenizer.py` 的具体行号、一个可跑的 seq2seq demo，把这条维度提升史钉到代码层。

---

## 目录

1. [五代范式总览表](#1-五代范式总览表)
2. [第一代：词法搜索（grep / ctags / 正则）](#2-第一代词法搜索grep--ctags--正则)
3. [第二代：静态分析 + 图查询（Sourcegraph / LSIF / CodeQL / Joern）](#3-第二代静态分析--图查询sourcegraph--lsif--codeql--joern)
4. [第三代：Word2Vec → 代码向量化搜索（第一次连续化）](#4-第三代word2vec--代码向量化搜索第一次连续化)
5. [第四代：Seq2Seq —— 用 (注释, 代码) 对子训练，让模型从描述生成代码](#5-第四代seq2seq--用-注释-代码-对子训练让模型从描述生成代码)
6. [一段能跑的 Seq2Seq Demo：注释 → 代码](#6-一段能跑的-seq2seq-demo注释--代码)
7. [第五代：GPT —— decoder-only + 自监督，把"配对"变成"序列"](#7-第五代gpt--decoder-only--自监督把配对变成序列)
8. [搜索维度的进化史：把五代放在同一张坐标系](#8-搜索维度的进化史把五代放在同一张坐标系)
9. [三次关键性转变的本质（在维度图上看就是三次维度跃迁）](#9-三次关键性转变的本质在维度图上看就是三次维度跃迁)
10. [CodeGPT 在这条进化线上的位置](#10-codegpt-在这条进化线上的位置)
11. [小结](#11-小结)

---

## 1. 五代范式总览表

| 代际 | 搜索空间维度 | 输入 → 输出 | 核心数据结构 | 关键操作 | 训练数据 | 失败模式 |
|------|--------------|-------------|--------------|----------|----------|----------|
| 1. 词法搜索 | **字符串**（1 维离散） | 关键词 → 已有代码片段 | 倒排索引 + Trigram | 字符串/正则匹配，TF-IDF | 无（只建索引） | 同义、跨命名风格、跨语言全失效 |
| 2. 静态分析 + 图查询 | **AST + 符号 + 引用 + 控制流 + 数据流**（多维离散结构） | 模式 / 图 query → 命中节点 / 路径 | LSIF/SCIP 索引、Code Property Graph | 树/图遍历、Datalog（CodeQL）、Cypher（Joern）、模式合一（Semgrep） | 无（仍是建索引）/ 弱训练（语言服务器规则） | 必须懂查询语言；跨概念近似仍失效；不会"写"代码 |
| 3. 向量化代码搜索 | **连续 $\mathbb{R}^d$**（首次连续化，典型 d=128/384/768） | 自然语言 / 代码 → 已有代码片段 | Embedding 矩阵 + ANN（HNSW/IVF/PQ） | `q @ V.T` 内积 | 大量无标注代码（Word2Vec/CodeBERT 自监督） | 只能"找"，不能写不存在的组合 |
| 4. Seq2Seq 生成 | **$\mathbb{R}^d$ + 时间轴 T**（连续 + 时序展开） | 自然语言 → **新生成的代码** | Encoder-Decoder（RNN/LSTM）+ attention | encode → context vector → decode 自回归 | **有监督配对** `(comment, code)` | 配对数据稀缺；信息瓶颈；长程依赖弱 |
| 5. GPT 生成 | **$\mathbb{R}^d$ × T × L 层 × H 头**（深度 + 多头并行高维检索） | 自然语言（或半段代码）→ 新代码 | Decoder-only Transformer，权重 `W` | 一次 forward = `L × H` 次 attention 检索 + 自回归采样 | 海量**未标注**源代码（注释和代码天然相邻） | 幻觉、上下文长度、对齐 |

**搜索维度从 1（字符串）→ 多维离散结构（AST/图）→ 连续向量 → 加时间轴 → 加深度和多头**。每加一维，"含义相同的代码"在新空间里都靠得更近一点。"问题"和"答案"在表征空间里的距离也越来越近：
**字符串距离 → 图编辑距离 → 余弦距离 → 编码-解码距离 → 同一个权重张量内部的距离。**

---

## 2. 第一代：词法搜索（grep / ctags / 正则）

最早的"代码搜索"就是工程师在仓库里跑 `grep`：

```bash
grep -rn "def fibonacci" .
ctags -R .
ack --python "TODO"
ripgrep "^class \w+\(.*Mixin\)" --type py        # 现代化的 grep，trigram 索引
```

GitHub code search 早期、Google Code Search（已下线）走的也是这个思路的工业版：**倒排索引 + 正则 + 少量符号元数据**。

```python
# 伪代码：第一代代码搜索的核心
def code_search(query: str, files: list[str]) -> list[str]:
    pattern = re.compile(query)
    hits = []
    for f in files:
        for i, line in enumerate(open(f)):
            if pattern.search(line):
                hits.append((f, i, line))
    return rank_by_tfidf(hits, query)
```

它工作得很好，前提是你**记得函数大概叫什么名字**。一旦换问法就崩：

- 想找"对列表去重保持顺序"——不知道关键字是 `dict.fromkeys` 还是 `OrderedDict`，搜不到。
- 想跨语言迁移——Python 的 `for x in xs` 和 JS 的 `for (const x of xs)` 在字符串上没有任何重合。
- 想搜"读文件并按行处理"——这是**意图**，不是字符串。

字符串匹配这一代的天花板就在这里：**搜索空间只有 1 维（字符），没有"结构"也没有"含义"概念**。

下一代要加的第一个新维度，是**代码本身的结构**——AST、符号引用、控制流、数据流。

---

## 3. 第二代：静态分析 + 图查询（Sourcegraph / LSIF / CodeQL / Joern）

2010 年代中期开始，业界意识到代码不是字符串，而是**结构**：函数定义、引用、类型、调用图、控制流图、数据流图。如果把这些结构提前算出来、建索引、再用专门的查询语言去搜——召回率立刻上一个台阶。**这一代仍然是"搜索"，但搜索空间从 1 维字符串扩展到了多维离散结构图**。

代表工具：

| 工具 | 哲学 | 加了什么维度 |
|------|------|--------------|
| **Sourcegraph** | "Google for code" + LSIF/SCIP 预计算的 goto-definition/find-references | **符号引用图**（跨仓库跨语言） |
| **OpenGrok**（Sun，2006） | ctags 基础上的 web 化代码搜索 | 符号 + 类层级 |
| **LSIF**（Microsoft, 2019） / **SCIP**（Sourcegraph, 2022） | 把 Language Server 的结果（hover / def / refs）序列化成索引文件 | 标准化的符号引用图，跨编辑器/跨工具 |
| **CodeQL**（GitHub / Semmle） | "把代码当成关系数据库"，写类 Datalog 的 QL 查询 | AST + 类型 + 数据流 + 污点分析 |
| **Semgrep** | AST 模式合一（pattern unification）+ 轻量数据流 | AST + 元变量绑定 |
| **Joern** / **Code Property Graph** | 把 AST + CFG + DFG 合成**一张图**，用 Cypher-like 语言查 | AST ∪ 控制流 ∪ 数据流 |

### 3.1 Sourcegraph 的本质：静态分析 + 高维空间搜索 + 图查询

Sourcegraph 自己的实现就是这三件事拼起来的：

1. **静态分析（升维）**：用每种语言的 LSP 服务器在 CI 里把代码"过一遍"，产出 LSIF / SCIP 索引——里面有每个 token 的**定义位置、所有引用、类型、所属符号**。这一步把代码从"字符流"升维成了"带语义边的图"。
2. **倒排 + 多维索引（建索引）**：对 trigram 建 zoekt 倒排索引（仍然是第一代的字符层），同时把 LSIF/SCIP 图写到结构化存储，两套并行。
3. **图查询（搜索）**：用户搜 `repo:foo func:HandleRequest` —— 这条 query 不是字符串匹配，而是**在符号引用图上做模式匹配**，返回的是"所有定义/调用了 `HandleRequest` 的位置"。

```python
# 伪代码：Sourcegraph-style 跨仓库 "find references"
def find_references(symbol_id: str, scip_index: Graph) -> list[Location]:
    # symbol_id 是跨仓库稳定的全局符号，例如 "github.com/foo/bar/pkg.HandleRequest"
    # SCIP 图里每个节点是一个符号，每条边是 "definition" / "reference" / "implementation"
    return [edge.location for edge in scip_index.edges_to(symbol_id)
            if edge.kind in {"reference", "implementation"}]
```

注意这一行 `scip_index.edges_to(symbol_id)`——它不是 `grep`，而是**在一张提前算好的图上做邻居查询**。这张图就是新维度。

### 3.2 CodeQL：把代码变成关系数据库

CodeQL 把这个思路推到极致——代码不是字符串、不是树，而是**一组关系表**。然后你用类 Datalog 的 QL 写查询：

```ql
// 找所有 "未经验证就被传给 sql.Exec 的字符串"——SQL 注入
import go

from DataFlow::Node source, DataFlow::Node sink
where
  source instanceof UntrustedFlowSource and
  sink.asExpr().getParent*() = any(SqlExec call).getQuery() and
  TaintTracking::localTaint(source, sink)
select sink, "tainted SQL query from $@", source, "untrusted input"
```

这段 query 在干什么？**在数据流图上做带条件的可达性搜索**——和 §4 的向量内积、§7 的 attention 是**同一类操作**（"从一个点出发，找满足条件的邻居"），只是底层空间不是 $\mathbb{R}^d$ 而是离散的关系图。

### 3.3 Joern / Code Property Graph：把 AST + CFG + DFG 合成一张图

CPG 的精髓是**一个节点同时承载语法、控制、数据三个角度的边**。比如一行 `x = read_input()` 在 CPG 里是一个节点，向上连 AST 父节点（赋值表达式），向左右连 CFG（前驱/后继语句），向下连 DFG（`x` 后续被谁读）。Cypher 风格的查询：

```cypher
// Joern: 找所有 "从 read_input 流到 system 调用"
cpg.call("read_input").reachableBy(cpg.call("system")).l
```

`reachableBy` 又是一次"高维空间里的可达性搜索"——只不过空间是**代码属性图**。

### 3.4 这一代的本质：用结构升维代替向量升维

第一代的搜索空间是**1 维**（字符）；第二代的搜索空间是**多维离散结构**——节点是符号/AST 节点，边是引用/控制流/数据流。它**没有"连续向量"概念，也没有学习参数**——所有维度都是规则定义的，靠静态分析提前算出来。

这一代的厉害之处是：**第一次能精确回答"语义相同但字面不同"的查询**。`x.foo()` 和 `(*p).foo()` 在 AST 上是不同字符串，但在 SCIP 引用图里指向同一个符号。CodeQL 能找到所有"通过任意路径把用户输入传给 SQL exec"的代码，无论变量名怎么换、控制流多绕。

但它有两个根本天花板：

1. **必须能形式化定义"想找什么"**——查询是显式规则，不会泛化。"找一段大概在做认证的代码"这种**意图查询**还是无能为力。
2. **不会写代码**——它只能在已有图上找已有节点；写一段不在库里的新代码，需要的是**生成式**模型。

突破第一个天花板要走"连续向量"路线（第三代 Word2Vec）；突破第二个要走"自回归生成"路线（第四代 Seq2Seq）。下一节就是第一次"连续化"——把图节点的离散标签替换成 $\mathbb{R}^d$ 里的稠密向量。

---

## 4. 第三代：Word2Vec → 代码向量化搜索（第一次连续化）

2013 年 Mikolov 提出 Word2Vec，把"相似度"从字符串搬到了 $\mathbb{R}^d$。其核心假设是 J.R. Firth 1957 年的一句话：

> *You shall know a word by the company it keeps.*
> 一个词的含义由它周围的词决定。

### 4.1 Word2Vec 的本质：一个超简的"邻居预测"

Skip-gram 的目标函数就是"用中心词预测它的上下文"：

```python
# Word2Vec Skip-gram 的核心 —— 不到 10 行 PyTorch
import torch, torch.nn as nn, torch.nn.functional as F

class SkipGram(nn.Module):
    def __init__(self, vocab_size, dim=128):
        super().__init__()
        self.in_emb  = nn.Embedding(vocab_size, dim)   # 中心词 embedding
        self.out_emb = nn.Embedding(vocab_size, dim)   # 上下文 embedding

    def forward(self, center, context):
        # logits[i] = <in_emb[center], out_emb[i]>
        v = self.in_emb(center)                # (B, d)
        logits = v @ self.out_emb.weight.T     # (B, vocab)
        return F.cross_entropy(logits, context)
```

训练完之后，`in_emb.weight` 这张 `(vocab_size, d)` 的查表就是"语义坐标系"。`king - man + woman ≈ queen` 那个著名加减法就是它的副产品。

**对应到本仓库**：`model.py:183` 的 `tok_emb = self.transformer.wte(idx)` 做的事情**和 Word2Vec 的查表完全同构**——都是 token ID 进、向量出。区别只是 GPT 的 `wte` 是和 12 层 Transformer 一起端到端训练的，而 Word2Vec 是单独训完就锁住的。

### 4.2 用到代码上：code2vec / CodeBERT

把同样的思路搬到代码：把 token 序列（甚至 AST path）输入一个小网络（code2vec 用 attention pooling，CodeBERT 用 BERT），输出一个固定维度向量。然后整套向量搜索（Faiss / HNSW）就可以用了：

```python
# 代码向量化搜索：把整个仓库的函数都 encode 成 (N, d) 的矩阵
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer('microsoft/codebert-base')
funcs = [open(p).read() for p in glob('repo/**/*.py')]   # 假装 N = 1e5
V = encoder.encode(funcs, normalize_embeddings=True)     # (N, 768)

index = faiss.IndexHNSWFlat(768, M=32)
index.add(V.astype('float32'))

q = encoder.encode(["read a file line by line"], normalize_embeddings=True)
D, I = index.search(q.astype('float32'), k=5)            # 找 top-5 最相似函数
```

突然之间，"读文件并按行处理"这种**意图查询**可以工作了：因为 encoder 学过 `with open(...) as f: for line in f` 的上下文邻居，它会把这段代码的向量放在意图查询附近。

### 4.3 但这一代仍然只能"找"，不能"写"

向量搜索能召回**已存在的**代码片段，无法合成不在库里的组合。如果用户问"用 numpy 实现一个带 dropout 的 self-attention"，而你的库里没有这段，向量搜索只能给出最接近的几个函数让你自己拼。

**第一次质变（离散 → 连续）让"找"变得强大，但还差一次质变才能"写"。**

---

## 5. 第四代：Seq2Seq —— 用 (注释, 代码) 对子训练，让模型从描述生成代码

2014 年 Sutskever / Cho 等人提出 Seq2Seq：用一个 RNN（**encoder**）把变长输入压成一个 context vector，再用另一个 RNN（**decoder**）从这个 context 一个 token 一个 token 地"写出"输出。

```
"sum a list"  →  [Encoder LSTM]  →  c  →  [Decoder LSTM]  →  "def sum_list(xs): return sum(xs)"
                  压缩信息              定长向量            自回归展开
```

应用到代码合成：训练数据是 **(docstring / 注释, function body)** 配对：

```python
# 训练样本（CodeSearchNet, CONALA 等数据集都是这种结构）
{"comment": "Compute factorial of n recursively",
 "code":    "def fact(n):\n    return 1 if n <= 1 else n * fact(n - 1)"}
```

这里发生了**第二次质变**——**检索 → 生成**：

| 第三代向量搜索 | 第四代 Seq2Seq |
|----------------|----------------|
| 答案必须存在于库中 | 答案是逐 token 现场生成的 |
| 输出 = 一段已有代码 | 输出 = 一个新序列 |
| 训练目标：让相似的对靠近 | 训练目标：`F.cross_entropy(logits, target_token)` |
| 没有"长度"概念 | 自回归 + `<eos>`，长度由模型决定 |

但 Seq2Seq 有两个先天痛点，是它后来被 Transformer 取代的根本原因：

1. **信息瓶颈**：encoder 把任意长输入压成一个固定向量 `c`，长输入信息丢失严重。Bahdanau 2014 / Luong 2015 的 attention 就是为了解决这个——decoder 每一步可以"回头看" encoder 全部隐状态。这正是 Transformer 的种子。
2. **配对数据稀缺**：必须人工/启发式构造 `(comment, code)` 对子。CodeSearchNet 数据集 ≈ 6M 对，但 GitHub 上有上百亿行代码——**99% 的训练信号被这种"必须配对"的范式扔掉了**。

第二个痛点是第四代被第五代彻底超越的真正原因，详见 §7。

---

## 6. 一段能跑的 Seq2Seq Demo：注释 → 代码

下面是一个最小可运行的 LSTM Seq2Seq + Bahdanau attention，用 (注释, 代码) 对子训练。**只依赖本项目已有的 `torch` / `numpy` / `tiktoken`**，没有引入新包。把它存成 `seq2seq_demo.py` 就能跑：

```python
"""
Minimal seq2seq demo: train on (comment, code) pairs, sample code from a comment.

Run:
    python seq2seq_demo.py            # train ~30s on CPU, then sample
"""
import torch, torch.nn as nn, torch.nn.functional as F
import tiktoken

# ---------- 1. 玩具数据集：(注释, 代码) 配对 ----------
PAIRS = [
    ("add two numbers",                "def add(a, b):\n    return a + b"),
    ("multiply two numbers",           "def mul(a, b):\n    return a * b"),
    ("subtract b from a",              "def sub(a, b):\n    return a - b"),
    ("square a number",                "def square(x):\n    return x * x"),
    ("compute factorial recursively",  "def fact(n):\n    return 1 if n <= 1 else n * fact(n-1)"),
    ("return the maximum of two",      "def max2(a, b):\n    return a if a > b else b"),
    ("check if number is even",        "def is_even(n):\n    return n % 2 == 0"),
    ("sum a list",                     "def sum_list(xs):\n    return sum(xs)"),
    ("length of a list",               "def length(xs):\n    return len(xs)"),
    ("reverse a list",                 "def reverse(xs):\n    return xs[::-1]"),
] * 80   # 重复多次让小模型学得动

# 复用本项目的 tiktoken GPT-2 BPE，再加一个 <bos>/<eos>/<pad>
enc = tiktoken.get_encoding("gpt2")
BOS, EOS, PAD = 50256, 50257, 50258      # 只在本 demo 内有效
VOCAB = 50259

def encode(s, add_eos=True):
    ids = enc.encode_ordinary(s)
    return ids + ([EOS] if add_eos else [])

# 把每条样本变成 (src_ids, tgt_ids)
data = [(torch.tensor(encode(c)), torch.tensor([BOS] + encode(k))) for c, k in PAIRS]

# ---------- 2. 模型：Encoder LSTM + Bahdanau Attention + Decoder LSTM ----------
class Encoder(nn.Module):
    def __init__(self, vocab, d=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.rnn = nn.LSTM(d, d, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2*d, d)             # 把 BiLSTM 输出降回 d 维

    def forward(self, src):
        h, _ = self.rnn(self.emb(src))            # (B, T, 2d)
        return self.proj(h)                       # (B, T, d)

class Attention(nn.Module):
    """Bahdanau (additive) attention —— seq2seq 时代的标志性发明。"""
    def __init__(self, d=128):
        super().__init__()
        self.W = nn.Linear(2*d, d); self.v = nn.Linear(d, 1)

    def forward(self, dec_h, enc_h):
        # dec_h: (B, d)   enc_h: (B, T, d)
        T = enc_h.size(1)
        cat = torch.cat([dec_h.unsqueeze(1).expand(-1, T, -1), enc_h], dim=-1)
        score = self.v(torch.tanh(self.W(cat))).squeeze(-1)   # (B, T)
        a = F.softmax(score, dim=-1)
        ctx = (a.unsqueeze(-1) * enc_h).sum(dim=1)            # (B, d)
        return ctx

class Decoder(nn.Module):
    def __init__(self, vocab, d=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.rnn = nn.LSTMCell(2*d, d)            # 输入 = emb + context
        self.attn = Attention(d)
        self.out = nn.Linear(d, vocab)

    def step(self, prev_tok, h, c, enc_h):
        ctx = self.attn(h, enc_h)                              # (B, d)
        x = torch.cat([self.emb(prev_tok), ctx], dim=-1)       # (B, 2d)
        h, c = self.rnn(x, (h, c))
        return self.out(h), h, c                               # logits, new state

class Seq2Seq(nn.Module):
    def __init__(self, vocab, d=128):
        super().__init__()
        self.enc = Encoder(vocab, d); self.dec = Decoder(vocab, d); self.d = d

    def forward(self, src, tgt):
        enc_h = self.enc(src)                                  # (1, T_src, d)
        h = enc_h.mean(dim=1); c = torch.zeros_like(h)         # 初始化 decoder state
        loss = 0.0
        for t in range(tgt.size(1) - 1):
            logits, h, c = self.dec.step(tgt[:, t], h, c, enc_h)
            loss = loss + F.cross_entropy(logits, tgt[:, t+1])
        return loss / (tgt.size(1) - 1)

    @torch.no_grad()
    def generate(self, src, max_len=80):
        enc_h = self.enc(src)
        h = enc_h.mean(dim=1); c = torch.zeros_like(h)
        cur = torch.tensor([[BOS]])
        out = []
        for _ in range(max_len):
            logits, h, c = self.dec.step(cur[:, -1], h, c, enc_h)
            nxt = logits.argmax(dim=-1)
            if nxt.item() == EOS: break
            out.append(nxt.item())
            cur = torch.cat([cur, nxt.unsqueeze(0)], dim=1)
        return enc.decode(out)

# ---------- 3. 训练 ----------
torch.manual_seed(0)
model = Seq2Seq(VOCAB)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
for epoch in range(8):
    total = 0.0
    for src, tgt in data:
        loss = model(src.unsqueeze(0), tgt.unsqueeze(0))
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    print(f"epoch {epoch}  loss={total/len(data):.3f}")

# ---------- 4. 推理：自然语言 → 代码 ----------
for prompt in ["add two numbers", "reverse a list", "check if number is even"]:
    src = torch.tensor(encode(prompt)).unsqueeze(0)
    print(f"\n>>> {prompt}\n{model.generate(src)}")
```

跑完之后你会看到 loss 从 ~7 掉到 ~0.5，模型大致能从注释生成对应的函数体。**这就是第四代的全部精髓**：

- Encoder（`self.enc`）= 把变长注释压成 `(T_src, d)` 的隐状态序列；
- Attention（`self.attn`）= decoder 每步**重新挑选** encoder 哪几个位置最相关——这是后来 Transformer 全套 attention 的祖先；
- Decoder（`self.dec`）= 自回归 LSTMCell，一个 token 一个 token 写代码；
- Loss = `F.cross_entropy`（和 `model.py:192` 完全是同一个函数）。

但你也立刻能看到它的天花板——只能学会训练对子里出现过的模式。给它一个稍微跨域的 prompt（比如 "implement quicksort"），它就开始胡说。原因正是 §4 末尾说的：**配对数据太贵，信号太稀疏**。

---

## 7. 第五代：GPT —— decoder-only + 自监督，把"配对"变成"序列"

GPT 在架构上做了三件事，每一件都是对 Seq2Seq 的根本性改写：

### 7.1 砍掉 Encoder：decoder-only

Seq2Seq 把"理解输入"和"生成输出"分到两个网络。GPT 直接合并成一个：把 prompt 和 completion**拼成一段**，全部喂给同一个 decoder，靠**因果掩码**保证生成时只能看左边。

```python
# 这就是 model.py:46-49 + 67 那个 tril mask 的全部意义
torch.tril(torch.ones(T, T))             # 下三角为 1
att = att.masked_fill(mask == 0, -inf)   # 上三角设 -inf —— 看不到未来
```

收益：encoder 和 decoder 共享一套权重 `W`，参数效率翻倍；同时**任何序列**都能直接当训练数据，不需要切分输入/输出。

### 7.2 把 (comment, code) 配对消化成"一段序列"

这是第三次质变的核心。Seq2Seq 时代你必须提前标好：

```
src: "compute factorial recursively"
tgt: "def fact(n): return 1 if n<=1 else n*fact(n-1)"
```

而在 GPT 时代，GitHub 上一份普通的 Python 文件本身就是训练样本：

```python
def fact(n):
    """Compute factorial of n recursively."""   # 这就是注释
    return 1 if n <= 1 else n * fact(n - 1)     # 这就是代码
```

模型只是在做 next-token prediction，但**注释和代码天然在序列上相邻**——预测下一行代码时模型必须利用上面的 docstring 当上下文，于是它**自动学会了"注释 → 代码"的映射**，根本不需要标注。这是为什么 GPT 时代训练数据规模一下子从 6M 对（CodeSearchNet）跃升到上百亿 token：**全网代码都是数据**。

### 7.3 Attention 替换 RNN：彻底解决信息瓶颈

Seq2Seq 必须用一个固定向量 `c` 串起 encoder 和 decoder，attention 只是补丁。Transformer 直接把整条序列摊平：每个位置都能直接看到前面所有位置（`model.py:53-70`）。**信息瓶颈消失了**。配合 KV cache、long context、scaling laws，这条路一直走到了今天的 GPT-4 / Claude。

### 7.4 FIM：把 seq2seq 的"中间填空"也自监督化

`(prefix, middle, suffix)` 任务在 seq2seq 时代必须人工切分配对。本项目用的 FIM trick（`tokenizer.py` 的 `<|fim_prefix|>` / `<|fim_middle|>` / `<|fim_suffix|>`，`model.py:121-124` 的 ID 默认值，`train.py:get_batch` 的 50% 概率变换）让自监督流水线**顺手**学会了中间填空——再次把"需要标注的能力"压回到"自监督序列"里。这是第三次质变在 CodeGPT 上的具体实现。

---

## 8. 搜索维度的进化史：把五代放在同一张坐标系

到此为止可以看出一条更深的规律——**这五代根本就是同一个故事**：「搜索」一直在做，每一代只是**给搜索空间加一个新维度**。把它画在同一张坐标系里：

```
代际    搜索空间                                      维度类型              单次查询的"操作"
─────  ────────────────────────────────────────────  ────────────────────  ──────────────────────────────────
1.     字符串                                         1 维离散              re.search / 倒排索引查表
2.     字符串 + AST + 引用 + CFG + DFG                多维离散结构          图遍历 / Datalog / Cypher
3.     字符串 + 结构 + ℝ^d                            + 1 个连续维度        q @ V.T  内积
4.     字符串 + 结构 + ℝ^d + 时间轴 T                  + 1 个时序维度        encode → ctx → decode 自回归
5.     字符串 + 结构 + ℝ^d × T × L 层 × H 头           + 深度 + 多头并行     一次 forward = L·H 次高维 attention
                                                       (CodeGPT: L=12, H=12,
                                                        即 144 次/token)
```

### 8.1 一个统一的伪代码视角

把五代的"核心操作"用同一个伪代码框架表达，区别全在 `space` 和 `metric` 两个参数上：

```python
# 五代搜索的统一形式
def search(query, space, metric, k=5):
    """
    space:  被搜索的对象集合（已建索引）
    metric: 衡量 query 和 space 中元素相似度的函数
    """
    return top_k(space, key=lambda x: metric(query, x), k=k)

# 第一代：space = 倒排索引里的文档；metric = TF-IDF / 子串匹配
# 第二代：space = SCIP/CPG 图节点；   metric = 图模式合一 / Datalog 规则可满足性
# 第三代：space = ANN 索引里的向量；  metric = q @ v  (内积 / 余弦)
# 第四代:  space 在每一步动态变化，   metric = decoder_state · encoder_h  (Bahdanau attention)
#         "搜索"沿时间轴展开 T 次
# 第五代:  space = 历史 token 的 K, V 矩阵； metric = q @ k.T （softmax）
#         在 L 层 × H 头的高维空间里反复搜索 L·H·T 次
```

最后一行就是 `model.py:66-70`：

```python
# model.py:66-70 —— 第五代每一层每一头都在做的搜索
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))   # q · K^T = "和谁相似？"
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # 因果掩码：只搜过去
att = F.softmax(att, dim=-1)                                       # 归一化为软分布
y = att @ v                                                        # 按相似度加权拷贝 V
```

把它和 §3.1 的 Sourcegraph `scip_index.edges_to(symbol_id)`、§4 的 `doc_vecs @ q_vec`、§5 的 Bahdanau `softmax(score) @ enc_h` 摆在一起——**它们是同一种操作的不同投影**：在某个空间里找邻居。区别只是**空间的维度和性质**：

| 操作 | 空间 | 距离/相似度 |
|------|------|-------------|
| `re.search(pat, doc)` | 字符串 | 子串/正则匹配（0/1） |
| `scip_index.edges_to(sym)` | 离散符号引用图 | 图上 1 跳邻接（0/1） |
| `cpg.reachableBy(...)` | Code Property Graph | 图上多跳可达（带数据流约束） |
| `doc_vecs @ q_vec` | $\mathbb{R}^{768}$ | 内积/余弦（连续） |
| `softmax(score) @ enc_h` | $\mathbb{R}^d \times T_{src}$ | Bahdanau 加权（连续 + 时序） |
| `model.py:66-70` 的 attention | $\mathbb{R}^d \times T \times L \times H$ | Q-K 内积 + softmax + 加权 V |

### 8.2 维度提升带来的两个直接好处

每加一个维度，召回曲线和"能解决的问题"都跳一档：

1. **"含义相同字面不同"的代码越来越好找**。
   - 1 维（字符串）：改个变量名就找不到。
   - 多维离散结构（CPG）：能找到所有"在同一控制流里把用户输入流到危险 sink"的代码。
   - 连续向量：能找到"功能相近，但根本没共享 token"的代码（Python `for ... in` ↔ JS `for ... of`）。
   - GPT 的 144 次 attention：把上面这些**全部当 building block 复用**，再叠 12 层抽象。

2. **能"写"出库里不存在的组合**。一旦搜索空间从离散变成连续，softmax 就能在两个邻居之间**插值**——而插值出来的点对应"训练集里没见过但合乎语法语义的代码"。这就是"生成"的数学本质：**在连续高维空间里取一个介于已知点之间的位置**。第二代再强大也插值不出新代码，因为离散图节点之间没有"中间状态"。

### 8.3 为什么"搜索维度"是统一视角而不是比喻

如果把所有五代写成一个张量的形状，差异一目了然：

```
第一代  shape = (T_query,)                    char level
第二代  shape = (N_nodes, K_edge_types)       discrete graph
第三代  shape = (N_docs, d)                   first continuous axis
第四代  shape = (T, d)                        + time axis
第五代  shape = (T, d, L, H)                  + depth + heads
```

每加一维，存储和算力 cost 都涨一截，但**模型能区分的"代码邻居"密度也指数级上升**。这正是 scaling laws 描述的现象——不是参数变多就变聪明，而是**搜索空间的有效维度变高**，"每条 query 的搜索精度"才指数级上升。

### 8.4 RAG = 把第二代/第三代检索拼回第五代的 prompt

理解了这个视角，RAG 就不再是"外挂记忆"那么模糊的概念，而是**精确的"维度回收"**：

- 第五代 GPT 的 `block_size`（CodeGPT 是 1024，GPT-4 是 ~128k）有上限；
- 你的私有代码库可能是 1e9 token，**装不进 prompt**；
- 解法：用第三代向量搜索（或第二代 SCIP 图搜索）从私有库里**先召回**几段最相关的代码，再拼到 GPT 的 prompt 里。

这就是为什么 Sourcegraph + LLM 是一套漂亮的组合：Sourcegraph 用第二代图搜索给出**精确**的引用/调用，向量搜索给出**模糊语义**召回，GPT 在它们的并集上做生成。三代搜索叠在一起，每一代各自负责自己最擅长的维度。

---

## 9. 三次关键性转变的本质（在维度图上看就是三次维度跃迁）

把上面五代之间的边界标出来，最重要的三条线分别对应**维度类型的切换**：

| 转变 | 边界 | 维度跃迁 | 数学上发生了什么 | 训练数据规模变化 |
|------|------|----------|------------------|------------------|
| **#1 离散结构 → 连续向量** | 第二代 → 第三代 | 离散图节点 → $\mathbb{R}^d$ | 相似度从 `图同构 / Datalog 可满足` 变成 `q @ V.T` —— 第一次有了"软语义近邻"和**插值** | 不变（仍是无标注） |
| **#2 检索 → 生成** | 第三代 → 第四代 | 加一个**时间维 T** | 从"找最近向量"变成"沿时间轴自回归 + cross-entropy" —— 第一次能写出**库里没有**的代码 | 仍需配对，规模约 1e6 对 |
| **#3 配对 → 自监督序列** | 第四代 → 第五代 | 加**深度 L 和多头 H** | 把 encoder-decoder 折叠成 decoder-only + causal mask；任何序列都是数据 | **1e4 倍跃升**，1e6 → 1e10+ token |

第一代到第二代的边界没列进"质变"——因为它**仍然是离散结构搜索**，只是从 1 维字符升到了多维图。它是搜索维度史里非常关键的一步（让代码搜索第一次能跨命名风格、跨调用层级），但**类型上没变**。真正的类型切换发生在三次质变上。

第三次转变是规模变革的真正源头：**前两次是"如何更好地表征"，第三次是"如何把全世界的代码都变成数据"。** Scaling laws、emergent abilities、in-context learning 都建立在这次转变之上——没有自监督，单靠人工标注永远训不出 GPT-3 这个量级的模型。

另一种理解视角：**每一次转变都把上一代的"必须人来做的事"消化进了模型自己**：

- 第一代：人来想关键词；
- 第二代：人来写 Datalog/Cypher 查询（Sourcegraph 替你预算了 SCIP 图）；
- 第三代：人来想 query 描述（embedding 模型替你想了同义词、跨语言同形结构）；
- 第四代：人来标 (comment, code) 配对（seq2seq 替你学了从描述到代码）；
- 第五代：模型靠相邻 token 自己学（不需要人标配对了）。

下一次转变（也许已经在发生）是 **从"自监督序列"到"自我对弈/RL 反馈"**——模型自己生成数据、自己评分、自己迭代。这是另一篇文档（[`SYNTHETIC_DATA.md`](SYNTHETIC_DATA.md) / [`SFT_FORGETTING_AND_MOE.md`](SFT_FORGETTING_AND_MOE.md)）的内容。

---

## 10. CodeGPT 在这条进化线上的位置

把本仓库的代码逐行对到上面五代里：

| 仓库元素 | 属于哪一代的设计 | 对应行号 |
|----------|------------------|----------|
| `tokenizer.py` GPT-2 BPE + 16 个语言特殊 token | 第三代（分布式表示）的延续 | `tokenizer.py` SPECIAL_TOKENS |
| `wte = nn.Embedding(50304, 768)` | 第三代 Word2Vec 的端到端版本 | `model.py:183` |
| `CausalSelfAttention.forward` 里的 `q @ k.T + softmax + @ v` | 第五代的核心；同时也是第四代 Bahdanau attention 的"自注意力"推广 | `model.py:51-73` |
| `tril` 因果掩码 | 第五代 decoder-only 的标志 | `model.py:46-49`、`67` |
| `F.cross_entropy(..., ignore_index=-1)` | 四代以来的同一个 loss —— seq2seq 也用、GPT 也用 | `model.py:192` |
| `apply_fim_transform` + `<|fim_*|>` | 第五代把 seq2seq 的"中间填空"自监督化 | `tokenizer.py`、`train.py:get_batch` |
| `generate()` 里的 `top_k` / `top_p` / `repetition_penalty` | 第五代的采样脚手架——和 seq2seq beam search 的精神延续 | `model.py:279-310`（约） |

**一个有意思的点**：`model.py` 里那一行 `tok_emb = self.transformer.wte(idx)`（第 183 行）和 §4 里那个 10 行的 Word2Vec Skip-gram 在数学上是同一类操作——查表、得到 token 的语义坐标。区别只是 GPT 把这张表和 12 层 Transformer 一起端到端训练，让坐标系**随上下文动态精调**（contextualized embedding）。所以"GPT 杀死 Word2Vec"并不准确——更准确的说法是：**GPT 把 Word2Vec 内化成了自己的第一层**，然后又叠了 12 层注意力上去。

同样的关系也成立于 Seq2Seq：**GPT 没有否定 Seq2Seq，而是把它压扁成了 decoder-only**。Bahdanau attention 那个 `softmax(score) · enc_h` 的式子，几乎一字不差地出现在 `model.py:66-70` 的自注意力里。

**对第二代（Sourcegraph / CodeQL / Joern）也是同样的关系**：CodeGPT 没有"否定"图查询，而是把"在图上找邻居"的能力**通过梯度下降隐式学进了权重**——GPT 的多头里有些头会自动学到"指向同一个变量的所有引用"、"同一函数的所有调用点"，这就是第二代静态分析图被压成连续向量后的样子。所以工业界最强的代码工具往往是 **Sourcegraph + GPT** 的组合：精确的图搜索 + 软语义生成，各自占据自己最擅长的维度（参见 §8.4 RAG 部分）。

---

## 11. 小结

五代演化、三次质变，本质是**同一件事**：搜索维度的持续提升。

> **代码生成的进化史 = 搜索空间维度提升的历史 = "程序员要做的事"逐步被模型吸收的历史。**
>
> - grep 时代：搜索空间 = 字符串；人想关键词、人写代码。
> - Sourcegraph / CodeQL 时代：+ AST + 引用图 + 控制流 + 数据流；人写图查询语言，**仍然只能找现成代码**。
> - Word2Vec 时代：+ 连续向量 $\mathbb{R}^d$；模型替你想同义词，人还是要写代码——但**第一次有了"插值"能力**。
> - Seq2Seq 时代：+ 时间维 T；模型替你写代码，但人还要标注 (注释, 代码) 对子。
> - GPT 时代：+ 深度 L + 多头 H；连配对都不用标了——模型从 GitHub 上"读"出来。**每生成 1 个 token = 144 次高维空间检索**。

每一代加进去的维度都不会丢——GPT 内部保留了前面所有代际的能力：第一代查表（`wte` embedding 查表）、第二代结构搜索（多头中"指向同一符号的所有引用"那种头）、第三代向量内积（每层 attention 的 `q @ k.T`）、第四代时序展开（自回归生成）。**第五代是前四代的叠加态**。

这条线还没有走到尽头。下一站是模型把"评估自己"和"造自己的训练数据"也吸收进去（参见 [`SYNTHETIC_DATA.md`](SYNTHETIC_DATA.md)、[`SFT_RL_INFERENCE_MECHANICS.md`](SFT_RL_INFERENCE_MECHANICS.md)）。但每一次转变背后的逻辑是一样的：**找到目前还需要人来做的那一步，把它变成可微分、可自监督的张量操作；同时给搜索空间加一个新维度**。

---

## 延伸阅读

- [`GPT_AS_SUPER_SEARCH.md`](GPT_AS_SUPER_SEARCH.md) — 把 GPT 当成"第四代搜索引擎"，从另一个视角讲同一件事
- [`DEEP_DIVE.md`](DEEP_DIVE.md) — RNN → LSTM → Seq2Seq → Transformer → GPT 的架构演化细节
- [`COMPRESSION_IS_INTELLIGENCE.md`](COMPRESSION_IS_INTELLIGENCE.md) — 为什么"预测下一个 token"这个 loss 能学出代码能力
- [`SYNTHETIC_DATA.md`](SYNTHETIC_DATA.md) — 第四代之后的方向：模型自己造训练数据
