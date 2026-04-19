# RAG 还是 SFT：面对一堆私有数据，该怎么选？

> 用户问题：我有一堆私有数据（公司文档、内部代码库、知识库、日志……），想让 LLM "学会"它们。应该做 RAG（检索增强生成），还是 SFT（监督微调）？怎么评估哪个方案更好？为什么？
>
> 这篇文档回答三件事：
> 1. **RAG 和 SFT 在数学上到底是两种不同的操作**——一个改 `context`（输入），一个改 `W`（参数）。理解这一点，90% 的困惑就没了。
> 2. **按什么维度选**：数据性质、更新频率、数据量、可解释性、推理成本、隐私、工程复杂度。给一张决策表。
> 3. **怎么"真的评估"**：怎么切数据、怎么定 metric、怎么设 baseline，才能得到可信的"谁更好"的结论。

---

## 目录

1. [先把两种操作的本质摆清楚](#1-先把两种操作的本质摆清楚)
2. [RAG 的数学：把知识塞进 prompt 的前半段](#2-rag-的数学把知识塞进-prompt-的前半段)
3. [SFT 的数学：把知识压进权重矩阵](#3-sft-的数学把知识压进权重矩阵)
4. [六个决策维度：哪种私有数据适合哪条路](#4-六个决策维度哪种私有数据适合哪条路)
5. [为什么"默认先做 RAG"是合理的](#5-为什么默认先做-rag-是合理的)
6. [什么时候 SFT 才是正解](#6-什么时候-sft-才是正解)
7. [RAG + SFT 叠加：不是二选一](#7-rag--sft-叠加不是二选一)
8. [真正的评估方法：数据集切分 + metric + baseline](#8-真正的评估方法数据集切分--metric--baseline)
9. [回到 CodeGPT：私有代码库该怎么用](#9-回到-codegpt私有代码库该怎么用)
10. [小结](#10-小结)

---

## 1. 先把两种操作的本质摆清楚

"给模型加私有数据"这句话在数学上是有歧义的。看看 CodeGPT 的 forward，就知道"知识"可以住在两个地方。`model.py:177-198`：

```python
def forward(self, idx, targets=None):
    # idx: (B, T) 输入 token ——"context"住在这里
    tok_emb = self.transformer.wte(idx)      # wte 是权重矩阵 W 的一部分
    pos_emb = self.transformer.wpe(pos)
    x = self.transformer.drop(tok_emb + pos_emb)
    for block in self.transformer.h:          # 每个 block 里的 W 矩阵
        x = block(x)
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)                  # lm_head 也是 W
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                           targets.view(-1), ignore_index=-1)
```

知识可以住在两个位置：

- **`idx`（context / 输入）**：推理时动态喂进去的 token。改它 = 改 prompt。**RAG 改的就是这个。**
- **`W`（模型参数：`wte`、`wpe`、`block.*`、`lm_head`）**：训练时通过梯度下降被写进去。**SFT 改的就是这个。**

换句话说：

- **RAG = 不动 `W`，在 `idx` 前面拼接相关文档。** 模型推理时"看到"私有数据，但自己并不"记得"。
- **SFT = 不动 `idx`，用私有数据跑 `F.cross_entropy` + `backward`，把知识压进 `W`。** 模型之后再也不需要文档，私有数据"内化"进了参数。

这是两件完全不同的事。两者的取舍也就完全不同。

---

## 2. RAG 的数学：把知识塞进 prompt 的前半段

RAG（Retrieval-Augmented Generation，Lewis et al. 2020）的完整流水线：

```
私有数据 → 切 chunk → 向量化（embedding）→ 存进向量库
                                                ↓
用户问题 → 向量化 → 相似度搜索 → top-k chunk → 拼进 prompt → LLM 生成
```

用 PyTorch 伪代码写出来（向量化这一步用本项目的 `tokenizer.py` + 一个 embedding 模型即可）：

```python
# 1. 建索引（一次性）
chunks   = split_documents(private_docs, chunk_size=512)
vectors  = embedding_model.encode(chunks)          # (N, d)
index    = numpy_or_faiss_index(vectors)

# 2. 推理时（每次）
q_vec      = embedding_model.encode(user_query)
top_k_ids  = index.search(q_vec, k=5)
retrieved  = [chunks[i] for i in top_k_ids]

prompt = f"""参考资料：
{chr(10).join(retrieved)}

问题：{user_query}
回答："""

# 3. 喂进 CodeGPT —— model.py 的代码完全不改
input_ids = tokenizer.encode(prompt)
output = model.generate(input_ids, max_new_tokens=256, top_p=0.95, ...)
```

**关键观察：RAG 完全不改模型参数**。它改变的只是 `idx`——`model.py:277` 里的 `idx_cond` 多了 retrieved chunks 的 token。模型依然是原来那个模型。

这带来几个立刻推得出的性质：

- **知识可以秒级更新**：换掉向量库里的一条文档，下一次推理立刻生效。不需要训练。
- **可以引用出处**：retrieved chunks 本身就是证据，直接展示给用户。
- **有硬边界**：`block_size`（`model.py:111` 默认 1024）。retrieved 塞进来占位置，留给问题和回答的空间就少了。
- **每次推理都要多做一次检索**：延迟和算力成本上升一些。

---

## 3. SFT 的数学：把知识压进权重矩阵

SFT 在数学上和预训练是**同一个 loss**——就是 `model.py:192` 那行 `F.cross_entropy`，详见 [SFT_FORGETTING_AND_MOE.md](SFT_FORGETTING_AND_MOE.md) 第 2 节。区别仅在于：

- 语料是精选的（你的私有数据，格式化成 prompt-answer 对）。
- `targets` 的 prompt 段被设成 `-1`，只让 answer 段产生梯度。

```python
# 把一条私有数据格式化
prompt = "<|user|>公司的 VPN 怎么配置？<|assistant|>"
answer = "1. 打开 Tunnelblick… 2. …"

tokens  = tokenizer.encode(prompt + answer)
targets = tokens.copy()
targets[:len(prompt_tokens)] = -1

# 然后复用 train.py 的循环，一行不改
X, Y = tokens, targets
logits, loss = model(X, Y)        # model.py:192 的 cross_entropy
loss.backward()
optimizer.step()                   # W ← W - lr · ∇loss
```

训练完之后：**私有数据已经不存在了**——它被压缩、蒸馏、扩散进了 `W` 的几十亿个浮点数里。推理时不再需要任何外部资料。

这也带来一组和 RAG 截然相反的性质：

- **知识更新意味着重训**：新增一条文档 → 重新 SFT（或增量训）。至少几小时起步。
- **无法引用出处**：答案是从参数"涌现"出来的，没有可追溯的证据链。
- **上下文窗口完全省给真正的任务**：不用塞参考资料。
- **一次性成本高，单次推理便宜**：训练贵，推理就是普通的 forward。
- **会遗忘**：做过 SFT 的模型，如果数据配比不当，通用能力可能下降（这也是 [SFT_FORGETTING_AND_MOE.md](SFT_FORGETTING_AND_MOE.md) 的主题）。

---

## 4. 六个决策维度：哪种私有数据适合哪条路

把问题拆成可以独立判断的维度：

| 维度 | RAG 更好 | SFT 更好 |
|---|---|---|
| **数据性质** | 事实性、查表类（API 文档、知识库、FAQ、合同） | 风格/技能类（说话语气、代码风格、领域推理方式） |
| **更新频率** | 高频变动（每天/每周都有新文档） | 静态或很少变（一年一次） |
| **数据量** | 几百条到几百万条，任意规模 | 至少几千条高质量样本，太少会过拟合 |
| **可解释性 / 可溯源** | 必须引用出处（法律、医疗、合规场景） | 不需要出处，只要输出对 |
| **推理延迟 / 成本** | 可接受多一次检索的延迟 | 延迟敏感、单次推理越便宜越好 |
| **数据隐私 / 独立部署** | 本地向量库 + 本地模型即可 | 一样可以本地训本地跑 |

几条经验法则：

- **"我想让模型知道 X 这条事实"** → RAG。事实是离散的、可变的、可索引的。
- **"我想让模型学会 Y 这种风格 / 这种思考方式"** → SFT。风格是连续的、分布式的、不适合检索。
- **"文档有几十 GB，而且每周都更新"** → RAG。SFT 根本跟不上节奏。
- **"我只有 500 条样本"** → RAG。SFT 在这个量级大概率只是在学数据的噪声。
- **"我需要模型在没网的边缘设备上跑"** → 倾向 SFT（或小模型 SFT + 本地向量库的 RAG 都行）。

---

## 5. 为什么"默认先做 RAG"是合理的

如果你没想清楚，先上 RAG。原因：

1. **零训练成本**。你不用准备 SFT 数据格式、不用调 lr、不用担心遗忘（见 [SFT_FORGETTING_AND_MOE.md](SFT_FORGETTING_AND_MOE.md)）、不用维护 checkpoint。
2. **试错便宜**。加一条文档就是 `index.add(vec)`，错了就删。SFT 错了要回滚到上一个 ckpt，重训。
3. **天生可解释**。retrieved chunks 就是证据。模型幻觉（hallucination）一眼能抓到——答案里引的事实不在 retrieved 里，就是幻觉。
4. **数据安全更简单**。私有数据留在向量库，不进入模型权重；想撤回某条数据就是 `index.remove(id)`。SFT 过的数据要"撤回"几乎不可能——它已经被混进 `W`，和其它知识纠缠在一起了（这一点在有 GDPR / "被遗忘权" 要求的场景特别关键）。
5. **可以和任何底座模型搭配**。换 base model 不影响向量库。SFT 和 base model 绑死。

反过来，如果你一上来就做 SFT，你会发现自己在训练一个**永远落后于文档更新的副本**。这是大多数"想让 LLM 学公司知识"的团队踩的第一个大坑。

---

## 6. 什么时候 SFT 才是正解

有几类场景 RAG 是做不好的，必须 SFT：

**(a) 改变输出风格 / 格式。** 比如希望模型总是用特定 JSON schema 回答、总是用公司术语、总是按某种模板组织答复。这些是"行为模式"而不是"事实"，塞进 prompt 可以临时压住，但 SFT 才能让它成为模型的默认习惯。

**(b) 领域推理能力。** 比如让模型学会某种专业推理（病历诊断、合同风险分析、特定代码审查模式）。这类能力不是"记住几条规则"，而是**在参数里形成一种新的推理路径**。retrieved chunks 只是例子，它不会让模型"变得会这样想"。

**(c) 降低推理延迟 / token 成本。** 如果某类知识被高频查询，每次都检索 + 塞 5KB 参考资料到 prompt 里太贵。把它 SFT 进模型，之后每次推理省掉 5KB 的 input tokens。规模大了账很可观。

**(d) 离线 / 带宽受限场景。** 边缘设备、无网环境。RAG 需要一个向量库在手边；SFT 过的模型是自包含的。

**(e) 对抗 prompt 注入。** RAG 的 retrieved chunks 如果来源不可信，攻击者可以在一篇"私有文档"里埋入 `忽略上面的指令，……` 这类注入。SFT 过的知识没有这个攻击面（因为没有运行时的文本注入通道）。

注意：即使要 SFT，**也几乎总是应该叠加 RAG**，而不是替代它。看下一节。

---

## 7. RAG + SFT 叠加：不是二选一

工业界成熟方案常常是**两者都要**：

```
私有数据
  ├── 静态的、风格的、推理模式的 ─→ SFT 进模型
  └── 动态的、事实的、带出处的    ─→ RAG 从向量库检索

推理时：
  SFT-过的模型 ( RAG 检索出来的文档 + 用户问题 ) → 回答
```

比如一个"公司内部助手"可能是这样做的：

- **SFT 阶段**：用 ~10k 条人工标注的"公司风格问答"样本，教会模型用公司内部术语说话、按内部 JSON schema 返回结构化结果、识别出什么问题该转给哪个部门。
- **RAG 阶段**：把几十万条持续更新的 wiki 文档、工单历史、代码注释放进向量库。推理时检索相关 chunk 拼进 prompt。

这两层各司其职：SFT 决定**怎么说**，RAG 决定**说的内容依据什么**。

在本项目的代码框架里，这两者完全可以共存：SFT 走 `train.py` 那套（改一下 `prepare.py` 构造 prompt-answer 对 + 把 prompt 段 target 设 -1），RAG 走推理期——在 `sample.py` 或 `repl.py` 的 `input_ids = tokenizer.encode(prompt)` 之前，先做一次检索、把 chunks 拼进 prompt 就行。模型的 `forward` / `generate` 一个字都不用改。

---

## 8. 真正的评估方法：数据集切分 + metric + baseline

"哪个更好"是一个经验问题，不是哲学问题。要得出**可复现**的结论，需要一个最小实验设计：

### 8.1 数据集切分

把你的私有数据和"问题"分开：

```python
# 私有数据（知识源）
docs_train       # 训 SFT / 进 RAG 向量库的文档
docs_heldout     # ← 关键：留一部分文档不给任何一方看

# 问题集（评测用）
questions = [
    # 来自 docs_train 的：测"记住 / 能查到"
    ("公司 VPN 配置步骤？", answer_1, source_doc_in_train),
    # 来自 docs_heldout 的：测泛化（RAG 能靠检索命中，SFT 没见过）
    ("最新的报销流程？", answer_2, source_doc_in_heldout),
    # 完全不相关的：测"没变笨"（防止 SFT 遗忘）
    ("Python 怎么写快排？", answer_3, None),
]
```

三段问题的意图：

| 问题来源 | RAG 预期 | SFT 预期 | 意义 |
|---|---|---|---|
| `docs_train` | 答对（检索命中） | 答对（参数记住了） | 基本能力 |
| `docs_heldout` | 答对（检索仍命中） | **大概率答错**（没训到） | **RAG 优势的关键证据** |
| 无关通用问题 | 答对（能力没变） | **可能答错**（遗忘） | **RAG 的另一个优势** |

这个设计能直接把两种方案最重要的差异暴露出来。

### 8.2 Metric

不要只看 loss。loss 和"用户觉得答得好"不是一回事。至少加两三个：

```python
# (1) 任务正确率 —— 如果你有标准答案
accuracy = sum(is_correct(model_answer, gold) for ... ) / N

# (2) 引用准确性 —— 专门测 RAG 是否幻觉
# 答案里提到的事实是否真的出现在 retrieved chunks 里
citation_precision = cited_facts_in_retrieved / cited_facts_total

# (3) 通用能力回归 —— 专门测 SFT 是否遗忘
# 在一个固定的通用 benchmark 上跑 before/after
humaneval_before = eval(base_model,   humaneval_set)
humaneval_after  = eval(sft_model,    humaneval_set)
forgetting = humaneval_before - humaneval_after   # 越大越糟
```

这三个指标合在一起，才能讲清楚"RAG vs SFT"的完整故事。只看正确率，SFT 在训练分布内往往看起来"更好"——但那是过拟合的假象。

### 8.3 Baseline

至少四条基线同台比较：

```python
baselines = {
    "base":         原始 CodeGPT，不给任何私有数据,             # 底线
    "rag_only":     base + RAG（向量库 + retrieve + 拼 prompt）,
    "sft_only":     base 上做 SFT，不用 RAG,
    "sft_plus_rag": SFT 后的模型 + RAG,                        # 叠加方案
}
```

在相同的 questions 上跑这四个，矩阵一出，结论自然浮现——不需要拍脑袋说"我觉得 RAG 更好"。

### 8.4 成本也要量化

除了质量，把下面几条记录下来：

- **前置成本**：RAG = 建索引时间；SFT = 训练 GPU 小时。
- **单次推理延迟**：检索延迟 + 多出来的 prompt tokens 对应的额外 forward 时间。
- **知识更新成本**：新加一条文档，两种方案分别要花多久。
- **存储**：向量库大小 vs checkpoint 大小。

一个方案再"准"，如果知识更新一次要 8 小时训练，在"公司 wiki 每天改"的场景里就是不可用。

---

## 9. 回到 CodeGPT：私有代码库该怎么用

假设场景：你有一个公司私有代码库（比如 50 万行内部 Python + 自建框架），想让 CodeGPT 学会在这个代码库里做补全。

按本文的分析，推荐路径：

**Step 1 —— 先判断你要的是"事实"还是"风格"。**

- "给我写一个调用我们公司 `InternalAuth` 类的登录代码" → 这是**事实查找**（`InternalAuth` 的方法签名）。RAG 更合适：把代码库 chunk 化（按函数切），放进向量库，推理时检索相关文件喂进 prompt。
- "让模型按我们公司风格写代码（命名习惯、错误处理模式、注释风格）" → 这是**风格**。SFT 更合适：把代码库当成预训练语料继续训。

90% 的私有代码场景，其实两者都想要——这就是 SFT + RAG 叠加的典型用例。

**Step 2 —— RAG 的最小实现，本项目已经具备所有部件。**

```python
# 切 chunk：可以复用 tokenizer.py 的 encode，按 block_size/2 滑窗
chunks = []
for file in private_codebase:
    tokens = tokenizer.encode(file.read())
    for i in range(0, len(tokens), 256):
        chunks.append(tokens[i:i+512])

# 向量化：用任何 embedding 模型（本项目没自带，可以先用一个小型 sentence-transformer）
# 检索：numpy 的余弦相似度足够跑通 demo，规模大了再换 FAISS
# 拼 prompt：直接在 sample.py 的 tokenizer.encode(prompt) 前做
```

注意 `block_size=1024`（`model.py:111`）这个硬约束——retrieved chunks 总长 + 用户 prompt + 要生成的 tokens 不能超过它。给自己留至少 256 token 的生成预算。

**Step 3 —— 如果要上 SFT，按私有代码的量决定配比。**

- 如果私有代码 < 100 MB（约几千万 token），直接混进预训练语料继续训，配比 30%~70% 私有、其余保持原来的公开代码，防止模型"只会写公司代码、不会写通用 Python"。
- 如果 > 1 GB，可以做两阶段：先在公开数据预训，再用私有数据做 SFT 阶段。**仍然要按 [SFT_FORGETTING_AND_MOE.md](SFT_FORGETTING_AND_MOE.md) 第 4 节的方法混入 10%~30% 公开数据**，否则通用能力会掉。
- 数据格式上，私有代码就是 next-token prediction，`target` 不用设 -1，和预训练一样——这种"用原始代码做 SFT"严格来说叫 continued pretraining，但损失函数和预训练完全一致（`model.py:192`）。

**Step 4 —— 评估就按第 8 节那一套做。** 特别建议：

- 在 **heldout 的私有文件**（训练时完全没见过）上测补全质量。这是最能区分"模型真的懂"和"只是记住"的信号。
- 跑一次 HumanEval（公开 benchmark）做 before/after，确保 SFT 没把通用 Python 能力打废。

**Step 5 —— 先 RAG 后 SFT 的现实节奏。** 实际项目中的推荐顺序是：

1. 先两周上一个 RAG 版本。立刻可用、可回滚、可迭代。
2. 观察用户高频问到的场景，收集"RAG 答不好 / 答得慢 / 答案不够像公司风格" 的样例。
3. 把这些样例当成 SFT 数据的信号来源。SFT 不是替代 RAG，是补 RAG 的短板。

---

## 10. 小结

- **RAG 改 `idx`，SFT 改 `W`**。是两种不同的操作，不是两种"差不多的方法"。分清楚这一点，选型问题的一半就消失了。
- **事实 → RAG，风格 → SFT**。更新频繁 → RAG，离线推理 / 延迟敏感 → SFT。需要溯源 → RAG。需要遵守 GDPR "被遗忘权" → RAG。
- **默认先做 RAG**。试错便宜、可解释、数据可撤回。见到 RAG 的短板（风格、领域推理、延迟、注入攻击面）再叠加 SFT。
- **SFT 和 RAG 通常是叠加而非二选一**。SFT 教"怎么说"，RAG 提供"说什么的依据"。
- **"哪个更好"要用实验回答**：切数据（train / heldout / 无关三段）+ 三类 metric（正确率 + 引用准确性 + 通用能力回归）+ 四条 baseline（base / rag / sft / sft+rag）。没做这套实验之前，任何"我觉得 X 更好"都只是偏见。
- **对 CodeGPT 这种私有代码场景**：先建一个最小 RAG（复用 `tokenizer.py` + numpy 相似度），观察短板，再决定要不要做 continued pretraining / SFT。别上来就砸 GPU 训。
