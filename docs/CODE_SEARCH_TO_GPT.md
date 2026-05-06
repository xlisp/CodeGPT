# 从代码语义搜索到 GPT 写代码：四代范式与三次关键转变

> 程序员"想要一段代码"这件事，三十年里被改写了四次。每一次都不是把上一代擦掉，而是**重新表述问题本身**：
>
> - **1990s**：搜代码 = 在源码里 `grep` —— 输入是字符串，输出是已存在的代码片段。
> - **2013**：搜代码 = 在向量空间里找近邻（Word2Vec / code2vec / CodeBERT）—— 输入是描述，输出是向量上"语义近"的现成代码。
> - **2015**：写代码 = 用 (注释, 代码) 配对数据训练 seq2seq —— 输入是自然语言，输出是**新生成**的代码。
> - **2020 起**：写代码 = 在巨量代码上做自监督预训练（GPT / Codex / CodeGPT）—— 输入还是自然语言，但模型不再需要"配对标注"，**任何带注释的源代码都自动成为训练样本**。
>
> 这条线上有三次质变：**离散符号 → 连续向量**、**检索 → 生成**、**配对监督 → 自监督序列**。本文用本仓库 `model.py` / `tokenizer.py` 的具体行号和一个可跑的 seq2seq demo，把每一次质变讲到代码层。

---

## 目录

1. [四代范式总览表](#1-四代范式总览表)
2. [第一代：代码语义搜索（grep / ctags / 正则）](#2-第一代代码语义搜索grep--ctags--正则)
3. [第二代：Word2Vec → 代码向量化搜索](#3-第二代word2vec--代码向量化搜索)
4. [第三代：Seq2Seq —— 用 (注释, 代码) 对子训练，让模型从描述生成代码](#4-第三代seq2seq--用-注释-代码-对子训练让模型从描述生成代码)
5. [一段能跑的 Seq2Seq Demo：注释 → 代码](#5-一段能跑的-seq2seq-demo注释--代码)
6. [第四代：GPT —— decoder-only + 自监督，把"配对"变成"序列"](#6-第四代gpt--decoder-only--自监督把配对变成序列)
7. [三次关键性转变的本质](#7-三次关键性转变的本质)
8. [CodeGPT 在这条进化线上的位置](#8-codegpt-在这条进化线上的位置)
9. [小结](#9-小结)

---

## 1. 四代范式总览表

| 代际 | 输入 → 输出 | 核心数据结构 | 关键操作 | 训练数据 | 失败模式 |
|------|-------------|--------------|----------|----------|----------|
| 1. 代码语义搜索 | 关键词 → 已有代码片段 | 倒排索引 + AST 索引 | 字符串/正则匹配，TF-IDF 排序 | 无（只建索引） | 同义、跨命名风格、跨语言全失效 |
| 2. 向量化代码搜索 | 自然语言 / 代码 → 已有代码片段 | Embedding 矩阵 + ANN 索引 | `q @ V.T` 内积 | 大量无标注代码（Word2Vec/CodeBERT 自监督） | 仍然只是"找"，不能写不存在的组合 |
| 3. Seq2Seq 生成 | 自然语言 → **新生成的代码** | Encoder-Decoder（RNN/LSTM）+ attention | encode → context vector → decode 自回归 | **有监督配对** `(comment, code)` | 配对数据稀缺；信息瓶颈；长程依赖弱 |
| 4. GPT 生成 | 自然语言（或半段代码）→ 新代码 | Decoder-only Transformer，权重 `W` | 一次 forward = 多次 attention + 自回归采样 | 海量**未标注**源代码（注释和代码天然相邻） | 幻觉、上下文长度、对齐 |

每往下一格，"问题"和"答案"在表征空间里的距离都更近一步：
**字符串距离 → 向量距离 → 编码-解码距离 → 同一个权重张量内部的距离。**

---

## 2. 第一代：代码语义搜索（grep / ctags / 正则）

最早的"代码搜索"就是工程师在仓库里跑 `grep`：

```bash
grep -rn "def fibonacci" .
ctags -R .
ack --python "TODO"
```

OpenGrok、Sourcegraph、GitHub code search 早期都是这个思路的工业版：**倒排索引 + 正则 + 一点 AST 元数据**（识别 `class` / `def` 边界，按符号建索引）。

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

字符串匹配这一代的天花板就在这里：**它没有"含义"这个概念**。

---

## 3. 第二代：Word2Vec → 代码向量化搜索

2013 年 Mikolov 提出 Word2Vec，把"相似度"从字符串搬到了 $\mathbb{R}^d$。其核心假设是 J.R. Firth 1957 年的一句话：

> *You shall know a word by the company it keeps.*
> 一个词的含义由它周围的词决定。

### 3.1 Word2Vec 的本质：一个超简的"邻居预测"

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

### 3.2 用到代码上：code2vec / CodeBERT

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

### 3.3 但这一代仍然只能"找"，不能"写"

向量搜索能召回**已存在的**代码片段，无法合成不在库里的组合。如果用户问"用 numpy 实现一个带 dropout 的 self-attention"，而你的库里没有这段，向量搜索只能给出最接近的几个函数让你自己拼。

**第一次质变（离散 → 连续）让"找"变得强大，但还差一次质变才能"写"。**

---

## 4. 第三代：Seq2Seq —— 用 (注释, 代码) 对子训练，让模型从描述生成代码

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

| 第二代向量搜索 | 第三代 Seq2Seq |
|----------------|----------------|
| 答案必须存在于库中 | 答案是逐 token 现场生成的 |
| 输出 = 一段已有代码 | 输出 = 一个新序列 |
| 训练目标：让相似的对靠近 | 训练目标：`F.cross_entropy(logits, target_token)` |
| 没有"长度"概念 | 自回归 + `<eos>`，长度由模型决定 |

但 Seq2Seq 有两个先天痛点，是它后来被 Transformer 取代的根本原因：

1. **信息瓶颈**：encoder 把任意长输入压成一个固定向量 `c`，长输入信息丢失严重。Bahdanau 2014 / Luong 2015 的 attention 就是为了解决这个——decoder 每一步可以"回头看" encoder 全部隐状态。这正是 Transformer 的种子。
2. **配对数据稀缺**：必须人工/启发式构造 `(comment, code)` 对子。CodeSearchNet 数据集 ≈ 6M 对，但 GitHub 上有上百亿行代码——**99% 的训练信号被这种"必须配对"的范式扔掉了**。

第二个痛点是第三代被第四代彻底超越的真正原因，详见 §6。

---

## 5. 一段能跑的 Seq2Seq Demo：注释 → 代码

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

跑完之后你会看到 loss 从 ~7 掉到 ~0.5，模型大致能从注释生成对应的函数体。**这就是第三代的全部精髓**：

- Encoder（`self.enc`）= 把变长注释压成 `(T_src, d)` 的隐状态序列；
- Attention（`self.attn`）= decoder 每步**重新挑选** encoder 哪几个位置最相关——这是后来 Transformer 全套 attention 的祖先；
- Decoder（`self.dec`）= 自回归 LSTMCell，一个 token 一个 token 写代码；
- Loss = `F.cross_entropy`（和 `model.py:192` 完全是同一个函数）。

但你也立刻能看到它的天花板——只能学会训练对子里出现过的模式。给它一个稍微跨域的 prompt（比如 "implement quicksort"），它就开始胡说。原因正是 §4 末尾说的：**配对数据太贵，信号太稀疏**。

---

## 6. 第四代：GPT —— decoder-only + 自监督，把"配对"变成"序列"

GPT 在架构上做了三件事，每一件都是对 Seq2Seq 的根本性改写：

### 6.1 砍掉 Encoder：decoder-only

Seq2Seq 把"理解输入"和"生成输出"分到两个网络。GPT 直接合并成一个：把 prompt 和 completion**拼成一段**，全部喂给同一个 decoder，靠**因果掩码**保证生成时只能看左边。

```python
# 这就是 model.py:46-49 + 67 那个 tril mask 的全部意义
torch.tril(torch.ones(T, T))             # 下三角为 1
att = att.masked_fill(mask == 0, -inf)   # 上三角设 -inf —— 看不到未来
```

收益：encoder 和 decoder 共享一套权重 `W`，参数效率翻倍；同时**任何序列**都能直接当训练数据，不需要切分输入/输出。

### 6.2 把 (comment, code) 配对消化成"一段序列"

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

### 6.3 Attention 替换 RNN：彻底解决信息瓶颈

Seq2Seq 必须用一个固定向量 `c` 串起 encoder 和 decoder，attention 只是补丁。Transformer 直接把整条序列摊平：每个位置都能直接看到前面所有位置（`model.py:53-70`）。**信息瓶颈消失了**。配合 KV cache、long context、scaling laws，这条路一直走到了今天的 GPT-4 / Claude。

### 6.4 FIM：把 seq2seq 的"中间填空"也自监督化

`(prefix, middle, suffix)` 任务在 seq2seq 时代必须人工切分配对。本项目用的 FIM trick（`tokenizer.py` 的 `<|fim_prefix|>` / `<|fim_middle|>` / `<|fim_suffix|>`，`model.py:121-124` 的 ID 默认值，`train.py:get_batch` 的 50% 概率变换）让自监督流水线**顺手**学会了中间填空——再次把"需要标注的能力"压回到"自监督序列"里。这是第三次质变在 CodeGPT 上的具体实现。

---

## 7. 三次关键性转变的本质

| 转变 | 从 | 到 | 数学上发生了什么 | 训练数据规模变化 |
|------|------|------|------------------|------------------|
| **#1 离散 → 连续** | 字符串相等 | 向量内积 | 相似度从 `==` 变成 `q @ V.T` —— 第一次有了"语义近邻" | 不变（仍是无标注） |
| **#2 检索 → 生成** | 输出 ∈ 数据库 | 输出 = 模型采样 | 从"找最近向量"变成"自回归 + cross-entropy" —— 第一次能写出**库里没有**的代码 | 仍需配对，规模约 1e6 对 |
| **#3 配对 → 自监督序列** | 必须标注 (comment, code) | 任意源代码即训练样本 | 把 encoder-decoder 折叠成 decoder-only + causal mask —— 数据规模解锁 | **1e3 倍跃升**，1e6 → 1e10+ token |

第三次转变是规模变革的真正源头：**前两次是"如何更好地表征"，第三次是"如何把全世界的代码都变成数据"。** Scaling laws、emergent abilities、in-context learning 都建立在这次转变之上——没有自监督，单靠人工标注永远训不出 GPT-3 这个量级的模型。

另一种理解视角：**每一次转变都把上一代的"必须人来做的事"消化进了模型自己**：

- 第一代：人来想关键词；
- 第二代：人来想 query 描述（embedding 模型替你想了同义词）；
- 第三代：人来标 (comment, code) 配对（seq2seq 替你学了从描述到代码）；
- 第四代：模型靠相邻 token 自己学（不需要人标配对了）。

下一次转变（也许已经在发生）是 **从"自监督序列"到"自我对弈/RL 反馈"**——模型自己生成数据、自己评分、自己迭代。这是另一篇文档（[`SYNTHETIC_DATA.md`](SYNTHETIC_DATA.md) / [`SFT_FORGETTING_AND_MOE.md`](SFT_FORGETTING_AND_MOE.md)）的内容。

---

## 8. CodeGPT 在这条进化线上的位置

把本仓库的代码逐行对到上面四代里：

| 仓库元素 | 属于哪一代的设计 | 对应行号 |
|----------|------------------|----------|
| `tokenizer.py` GPT-2 BPE + 16 个语言特殊 token | 第二代（分布式表示）的延续 | `tokenizer.py` SPECIAL_TOKENS |
| `wte = nn.Embedding(50304, 768)` | 第二代 Word2Vec 的端到端版本 | `model.py:183` |
| `CausalSelfAttention.forward` 里的 `q @ k.T + softmax + @ v` | 第四代的核心；同时也是第三代 Bahdanau attention 的"自注意力"推广 | `model.py:51-73` |
| `tril` 因果掩码 | 第四代 decoder-only 的标志 | `model.py:46-49`、`67` |
| `F.cross_entropy(..., ignore_index=-1)` | 三代以来的同一个 loss —— seq2seq 也用、GPT 也用 | `model.py:192` |
| `apply_fim_transform` + `<|fim_*|>` | 第四代把 seq2seq 的"中间填空"自监督化 | `tokenizer.py`、`train.py:get_batch` |
| `generate()` 里的 `top_k` / `top_p` / `repetition_penalty` | 第四代的采样脚手架——和 seq2seq beam search 的精神延续 | `model.py:279-310`（约） |

**一个有意思的点**：`model.py` 里那一行 `tok_emb = self.transformer.wte(idx)`（第 183 行）和 §3 里那个 10 行的 Word2Vec Skip-gram 在数学上是同一类操作——查表、得到 token 的语义坐标。区别只是 GPT 把这张表和 12 层 Transformer 一起端到端训练，让坐标系**随上下文动态精调**（contextualized embedding）。所以"GPT 杀死 Word2Vec"并不准确——更准确的说法是：**GPT 把 Word2Vec 内化成了自己的第一层**，然后又叠了 12 层注意力上去。

同样的关系也成立于 Seq2Seq：**GPT 没有否定 Seq2Seq，而是把它压扁成了 decoder-only**。Bahdanau attention 那个 `softmax(score) · enc_h` 的式子，几乎一字不差地出现在 `model.py:66-70` 的自注意力里。

---

## 9. 小结

四代演化、三次质变，可以用一句话概括：

> **代码生成的进化史，就是"程序员要做的事"逐步被模型吸收的历史。**
>
> - grep 时代：人想关键词、人写代码；
> - Word2Vec 时代：模型替你想同义词，人还是要写代码；
> - Seq2Seq 时代：模型替你写代码，但人还要标注 (注释, 代码) 对子；
> - GPT 时代：连配对都不用标了——模型从 GitHub 上"读"出来。

这条线还没有走到尽头。下一站是模型把"评估自己"和"造自己的训练数据"也吸收进去（参见 [`SYNTHETIC_DATA.md`](SYNTHETIC_DATA.md)、[`SFT_RL_INFERENCE_MECHANICS.md`](SFT_RL_INFERENCE_MECHANICS.md)）。但每一次转变背后的逻辑是一样的：**找到目前还需要人来做的那一步，把它变成可微分、可自监督的张量操作**。

---

## 延伸阅读

- [`GPT_AS_SUPER_SEARCH.md`](GPT_AS_SUPER_SEARCH.md) — 把 GPT 当成"第四代搜索引擎"，从另一个视角讲同一件事
- [`DEEP_DIVE.md`](DEEP_DIVE.md) — RNN → LSTM → Seq2Seq → Transformer → GPT 的架构演化细节
- [`COMPRESSION_IS_INTELLIGENCE.md`](COMPRESSION_IS_INTELLIGENCE.md) — 为什么"预测下一个 token"这个 loss 能学出代码能力
- [`SYNTHETIC_DATA.md`](SYNTHETIC_DATA.md) — 第四代之后的方向：模型自己造训练数据
