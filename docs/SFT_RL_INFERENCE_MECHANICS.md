# 训练写权重，推理用权重 + 脚手架：SFT / RL 训完之后到底是怎么生效的

> 用户问题：为什么大模型要先 SFT 再 RL？最后用模型做 QA / 对话时，这些训练是怎么"生效"的——是纯靠 Transformer 里的参数在预测吗？还是说实际对话时还需要一段配套代码，配合之前 SFT / RL 训练的结果才能跑起来？具体生效的机制是什么？
>
> 这篇文档回答三件事：
>
> 1. **为什么是"预训练 → SFT → RL"三步走**，而不是一次性训完。每一步在 `W`（权重）里写入的"东西"是不一样的。
> 2. **推理时到底发生了什么**：纯靠 `W` 做 `forward` 够不够？不够的话还差什么？
> 3. **训练时的协议和推理时的协议必须严格对齐**——这才是 ChatGPT 能用的真正原因。tokenizer 的特殊 token、chat template、停止条件、采样参数都是"脚手架"的一部分，**和 `W` 一起构成了完整的"大模型系统"**。

---

## 目录

1. [核心二分：知识住在 W 里，协议住在代码里](#1-核心二分知识住在-w-里协议住在代码里)
2. [为什么是"预训练 → SFT → RL"三步，不是一步](#2-为什么是预训练--sft--rl三步不是一步)
3. [三步训练分别在 W 里写入了什么](#3-三步训练分别在-w-里写入了什么)
4. [推理时如果"什么都不做"会发生什么](#4-推理时如果什么都不做会发生什么)
5. [chat template：训练时和推理时必须完全一致](#5-chat-template训练时和推理时必须完全一致)
6. [停止条件：RL 训练给模型的"闭嘴信号"如何生效](#6-停止条件rl-训练给模型的闭嘴信号如何生效)
7. [采样参数：推理时对 W 的"温度调节"](#7-采样参数推理时对-w-的温度调节)
8. [W 装不下的东西：tool use / RAG / memory 必须靠外部代码](#8-w-装不下的东西tool-use--rag--memory-必须靠外部代码)
9. [回到 CodeGPT：当前代码里哪些就是"脚手架"](#9-回到-codegpt当前代码里哪些就是脚手架)
10. [小结：一张"训练—推理"对齐表](#10-小结一张训练推理对齐表)

---

## 1. 核心二分：知识住在 W 里，协议住在代码里

这个问题的答案不是"是"也不是"否"，而是：**大模型能力是"权重 + 脚手架"共同组成的一个系统**。单独拿出 `W`，什么都跑不起来；单独拿出脚手架，也没用。

先看 CodeGPT 的 `forward`（`model.py:177-198`）：

```python
def forward(self, idx, targets=None):
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size, ...
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    tok_emb = self.transformer.wte(idx)     # W 的一部分：词嵌入
    pos_emb = self.transformer.wpe(pos)     # W 的一部分：位置嵌入
    x = self.transformer.drop(tok_emb + pos_emb)
    for block in self.transformer.h:         # W 的主体：每个 block 里的所有矩阵
        x = block(x)
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x[:, [-1], :])    # W 的最后一层：输出到 vocab
```

这个 `forward` 就是一个纯函数：`(idx, W) → logits`。**W 是训练写进去的，`idx` 是推理时由外部代码送进来的。**

所以训练做的事和推理做的事可以彻底分开：

```
┌──────────────────────────┬───────────────────────────────┐
│  训练阶段                 │  推理阶段                      │
│  ───────                  │  ───────                       │
│  读大量数据               │  接到用户的一条 prompt         │
│  跑 forward + backward    │  构造成 idx（encode_prompt）   │
│  用梯度下降修改 W         │  跑 forward 得到 logits        │
│  最终把"能力"压进 W       │  采样出下一个 token            │
│                           │  循环，直到停止条件            │
└──────────────────────────┴───────────────────────────────┘
```

用户的问题——"使用大模型时是纯靠参数预测吗？还是需要代码配合？"——其实是在问这张图的右半边。右半边做的几乎全部是"代码"：把 prompt 变成 `idx`、跑 `generate` 循环、判断停止条件、把 token 解码回字符串。这些代码如果和训练时的约定不一致，`W` 里训练好的能力就**激活不了**。

> **一句话总结：知识（懂 Python、会 QA、会"礼貌"）住在 `W` 里；但"如何把问题喂给 `W`、如何解读 `W` 的输出"这套协议，住在推理代码里。协议不对，知识就调不出来。**

下面把这两层一层一层拆开。

---

## 2. 为什么是"预训练 → SFT → RL"三步，不是一步

先解释训练侧的三步。这部分和 [RLHF_AND_PLATONIC_REPRESENTATION.md](RLHF_AND_PLATONIC_REPRESENTATION.md)、[SFT_FORGETTING_AND_MOE.md](SFT_FORGETTING_AND_MOE.md) 有重叠，但这里的角度是："为什么不能直接训一次到位"。

### 2.1 一次训完"好回答"数据行不行？

理论上可以——把 RLHF 用的偏好数据直接塞到预训练语料里，一次梯度下降。但实践上不行，因为**数据量差了几个数量级**：

```
预训练语料:      10^12 ~ 10^13 token   （全互联网级别）
SFT 语料:        10^5 ~ 10^7  token    （人类标注的问答对）
RLHF 偏好对:     10^4 ~ 10^6  pair     （人类排序的偏好）
```

把 10^7 条 SFT 数据混进 10^13 条预训练数据里，信号会被稀释到几乎看不见。你要模型"学会对话格式"，就得让这批数据在训练末期能产生足够强的梯度——这就是**分阶段训练**的本质：用不同的**学习率 / 数据配比 / 甚至不同的损失函数**，在不同阶段强化不同的能力。

### 2.2 三步走的分工

每一步训练的**目标函数**是不一样的，决定了它在 `W` 里写入的"东西"不一样：

| 阶段 | 数据 | 损失函数 | 在 W 里写入什么 |
|---|---|---|---|
| 预训练 | 互联网文本 | `F.cross_entropy(logits, next_token)` | 语法、世界知识、基础推理 |
| SFT | 人类写的"问-答"对 | 同一个 `F.cross_entropy`，但 prompt 段 target 设为 `-1` | 对话格式、指令遵循 |
| RLHF | 人类对"两个回答"的偏好排序 | `-log σ(r_good - r_bad)` 或 DPO loss | "好/不好"的品味、拒答、安全边界 |

**关键观察**：SFT 在数学上就是预训练的子集（见 `SFT_FORGETTING_AND_MOE.md` 第 2 节），用的就是本项目 `train.py` 里这一行：

```python
# train.py（预训练 loss，和 SFT 完全一样）
loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                       targets.view(-1), ignore_index=-1)
```

SFT 的唯一变化是把 `targets` 中 prompt 部分设成 `-1`（由 `ignore_index=-1` 跳过梯度）。所以**SFT 不需要新代码，只需要新数据格式**。这也是为什么业界能用同一个训练脚本跑预训练和 SFT。

RL / DPO 则是**真正的损失函数变化**，需要新的训练循环。详细机制见 [RLHF_AND_PLATONIC_REPRESENTATION.md](RLHF_AND_PLATONIC_REPRESENTATION.md) 第 5-6 节。

### 2.3 为什么必须先 SFT 再 RL

直接在预训练模型上跑 RL 会崩。原因：

```
预训练模型对一个问题可以给出 10^20 种合理续写
    ↓ 奖励模型只见过其中几千种（标注员写/挑过的）
    ↓ 对其余续写，奖励模型打分不可靠 (out-of-distribution)
    ↓ PPO 沿着不可靠的梯度走 → 模型崩到奖励黑客区
```

SFT 的作用是**把模型的输出分布先"收敛"到和标注员风格接近的范围**，让后续的 RL 步骤在奖励模型熟悉的区域里优化。换句话说：

- **SFT 是粗定位**：把分布拉到"大致像人类回答"的流形上。
- **RL 是精修**：在这个流形里找"人类最偏好"的方向。

跳过 SFT 直接 RL 就是在地图上没走到对的省份就开始找街道，找不到。

---

## 3. 三步训练分别在 W 里写入了什么

这一节回答用户的第二个问题："这些机制是如何生效的？纯靠记录在 transformer 里的参数来预测吗？"

**W 里确实记录了所有这三步训练的结果**，但每一步改变的参数子集和量级是不一样的。

### 3.1 预训练：改动所有参数，改动量最大

预训练从随机初始化开始，几乎所有参数都从 `N(0, 0.02²)` 一路走到最终值。看 `model.py` 的初始化：

```python
# model.py:175
torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

训完之后，**每个 `wte[i]`、每个 attention 矩阵、每个 MLP 的 `c_fc` / `c_proj`，都充满了语义**。这是"懂 Python"、"懂英文"、"懂算法"这些能力的物理载体。

### 3.2 SFT：改动所有参数，但改动量很小

SFT 通常训练 1~3 个 epoch，学习率比预训练小一个数量级。改的是同一套 `W`，但幅度小得多。

重要的一点：**SFT 改的不是"加一组新参数"，而是把预训练学到的 `W` 朝"应答分布"方向做微小位移**。位移幅度小到：

- 如果你观察每个参数的变化 `W_sft - W_pretrain`，大部分值都很小。
- 但分布变化是显著的——预训练模型给 `"用户: 1+1=?"` 后面的概率分布是"可能续写成一个数学课本片段"；SFT 之后变成"直接给出 `答: 2`"。

### 3.3 RL / DPO:改动所有参数，但有 KL 锚定

RL 阶段有一项**KL 散度约束**（见 RLHF 文档 5.3 节）：

```
L_RL = -reward + β · KL(π_RL ‖ π_SFT)
```

这一项的作用是**不允许 `W` 偏离 SFT 之后的版本太远**。所以 RL 阶段改的参数量比 SFT 还小——它只负责"微调回答风格"和"学会拒绝某些请求"，不负责重新教模型写代码。

### 3.4 所以"纯靠参数"这句话对一半

对的那一半：

> 推理时的每一个 token 预测，**真的就只是一次 `W · x` 的矩阵运算**。用户问 "Python 怎么反转列表" 时，模型并不会在某个数据库里查找 SFT 训练时见过的那条记录。它靠的是 `W` 中已经被训练阶段"压缩"进去的模式。

不对的那一半：

> 但"能正确收到用户的问题"、"能在合适的时候停止"、"能被解码成可读字符串"——这些都不在 `W` 里，而在推理代码里。

下面几节具体看推理代码需要做什么。

---

## 4. 推理时如果"什么都不做"会发生什么

假设你拿到一个完成 RLHF 的 ChatGPT 级别的模型，只调用最朴素的 `model.generate(tokenizer.encode("1+1=?"))`，会发生什么？

**几乎一定会失败**，原因是训练时模型看到的**不是裸 prompt**。它看到的是像下面这样的东西（每家模型的模板略有不同）：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
1+1=?<|im_end|>
<|im_start|>assistant
```

如果推理时只喂 `"1+1=?"`——连 `<|im_start|>user` 都没有——模型的 `W` 里学过的"看到 `<|im_start|>user` 开头的内容就要在 `<|im_start|>assistant` 后面给出礼貌回答"**这个行为根本不会触发**。它会退回到预训练的"续写"模式，输出可能是：

```
输入: "1+1=?"
输出（预训练模式）: "2. 2+2=? 4. 3+3=? 6. 4+4=? ..."   ← 像一本习题册的续写
```

而不是：

```
输入: "<|im_start|>user\n1+1=?\n<|im_end|>\n<|im_start|>assistant\n"
输出（SFT/RL 模式）: "1+1=2"   ← 正确的对话回答
```

**所以"用户问题 → 模型回答"不是直接调用的，它是通过一段被叫做 chat template 的代码把 prompt 包装成训练时的格式再送进 `W` 的。**

这就是用户问的"需要代码配合"的核心：**并不是什么复杂的协调逻辑，而是一段**必须和训练完全一致**的 token 序列构造逻辑。**

---

## 5. chat template：训练时和推理时必须完全一致

### 5.1 本项目里的"简化版 chat template"

CodeGPT 还没有做成对话模型，但它已经有一个**代码任务的 template**——这是研究这个问题的绝佳入口。看 `sample.py:101-121`：

```python
# sample.py:101 —— 把纯文本包装成"模型看得懂"的 token 序列
def encode_prompt(text, use_lang=True):
    tokens = []
    tokens.append(SPECIAL_TOKENS["<|code_start|>"])          # 50261
    if use_lang and lang:
        lang_token = f"<|lang:{lang}|>"
        if lang_token in SPECIAL_TOKENS:
            tokens.append(SPECIAL_TOKENS[lang_token])         # 50263-50278
    tokens.extend(tokenizer.encode_raw(text))
    return tokens


# sample.py:113 —— FIM 的 template
def encode_fim(prefix_text, suffix_text):
    tokens = []
    tokens.append(SPECIAL_TOKENS["<|fim_prefix|>"])           # 50257
    tokens.extend(tokenizer.encode_raw(prefix_text))
    tokens.append(SPECIAL_TOKENS["<|fim_suffix|>"])           # 50259
    tokens.extend(tokenizer.encode_raw(suffix_text))
    tokens.append(SPECIAL_TOKENS["<|fim_middle|>"])           # 50258
    return tokens
```

这段代码做的是：**把 `"def add(a, b):"` 这种用户可读的字符串，翻译成训练时见过的 token 序列**。

如果训练时模型学到的 FIM 规律是：

```
看到 <|fim_prefix|> ... <|fim_suffix|> ... <|fim_middle|> 之后 → 生成中间的代码
```

那推理时必须构造**完全相同的 token 顺序**，模型的 `W` 里的"FIM 能力"才会被激活。看 [`train.py` 里的 `apply_fim_transform`](../tokenizer.py) 和 `tokenizer.py` 的 `SPECIAL_TOKENS`：

```python
# tokenizer.py
SPECIAL_TOKENS = {
    "<|endoftext|>":    50256,
    "<|fim_prefix|>":   50257,
    "<|fim_middle|>":   50258,
    "<|fim_suffix|>":   50259,
    "<|fim_pad|>":      50260,
    "<|code_start|>":   50261,
    "<|code_end|>":     50262,
    "<|lang:python|>":  50263,
    # ...
}
```

**这份表在训练时和推理时必须字节级一致**——包括 ID 是 50257 还是 50258、顺序、数量。如果推理时 `<|fim_prefix|>` 用了个错误的 ID（比如 50258），模型会完全看不懂你在说 FIM，输出就会退化成普通续写。

### 5.2 通用对话模型的 chat template

放大到 ChatGPT / Qwen / Llama 这些对话模型，就是同一件事的放大版：

```python
# 伪代码 —— Qwen 风格 chat template
def apply_chat_template(messages):
    text = ""
    for m in messages:
        text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return tokenizer.encode(text)
```

和 CodeGPT 的 `encode_fim` 本质完全相同——都是"用约定的特殊 token 把多段内容包起来"。**这段代码不属于模型，它属于推理脚手架，但它和模型的 `W` 是一对一耦合的。换个 template，模型就不认识。**

### 5.3 HuggingFace 的 `tokenizer.apply_chat_template`

这也是为什么现代模型的 tokenizer 配置里都带 `chat_template` 字段（Jinja 模板），在线推理时 `tokenizer.apply_chat_template(messages)` 会自动按模型训练时的格式渲染 prompt。换模型必须换 template——这是第一个"使用模型需要配合代码"的地方。

---

## 6. 停止条件：RL 训练给模型的"闭嘴信号"如何生效

对话模型最重要的一个行为是"适可而止"——回答完问题就停下。这个行为在训练和推理里是**双边实现**的。

### 6.1 训练侧：让模型在回答结束时生成 `<|im_end|>`

SFT 数据里每条"助手回答"的最后一个 token 都是 `<|im_end|>`（或 `<|endoftext|>`、或某个自定义的 `<|eot_id|>`）。模型学到的规律是：

```
回答说完了 → 输出 <|im_end|>
```

这个规律**就住在 `W` 里**——具体来说，就是在"回答长度差不多"、"问题已经解答"的上下文里，`lm_head` 输出的 logits 会在 `<|im_end|>` 这个位置给出很高的概率。

### 6.2 推理侧：必须用代码**检测**这个 token 并停止

但 `W` 只能输出一个概率分布，它不会自己"停"。停止是推理代码的工作。看 `model.py:304-306`：

```python
# model.py:304 —— 停止条件由外部代码执行
idx_next = torch.multinomial(probs, num_samples=1)

if idx_next.item() in stop_tokens:
    break
```

`stop_tokens` 怎么来的？看 `sample.py:98`：

```python
# sample.py:98 —— 告诉 generate 循环哪些 token 代表"停"
stop_tokens = [SPECIAL_TOKENS["<|endoftext|>"],
               SPECIAL_TOKENS["<|code_end|>"]]
```

这是另一个"训练-推理必须对齐"的地方：

```
训练时学到:  回答结束 → 输出 X token
推理时必须:  把 X 加到 stop_tokens 里

如果不对齐:
  模型输出了 <|im_end|>，但推理代码不认 → 继续生成下一个 token
  → 模型被强行逼着在"已经说完"的语境下继续说
  → 退化成胡言乱语、重复、或者开始模仿下一轮 <|im_start|>user（幻觉）
```

这就是为什么很多人在本地跑开源模型时发现"模型停不下来"——**十有八九是 stop_tokens 没配对。`W` 没坏，是脚手架没对齐。**

### 6.3 延伸：max_new_tokens

推理循环还有一个兜底：看 `model.py:275` 那个 `for _ in range(max_new_tokens)` 的 `max_new_tokens` 参数。这是**纯代码层面**的强制上限，防止模型输出模式正好没触发停止 token 时无限生成。训练阶段没有"一句话最多多长"这个概念，所以这个限制只能写在推理代码里。

---

## 7. 采样参数：推理时对 W 的"温度调节"

RL 训练让模型**学会了给好回答打高分**——但 `W` 输出的始终是一个**概率分布**，不是一个确定的 token。究竟要不要真的选那个概率最高的？选多少概率的？这由推理代码控制。

看 `model.py:279-302`：

```python
# model.py:279 —— 温度：缩放 logits
logits = logits[:, -1, :] / temperature

# model.py:282-284 —— repetition_penalty：惩罚已出现的 token
if repetition_penalty != 1.0:
    for token_id in set(idx[0].tolist()):
        logits[0, token_id] /= repetition_penalty

# model.py:287-289 —— top-k：只保留前 k 个候选
if top_k is not None:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')

# model.py:292-299 —— top-p (nucleus)：只保留累计概率 p 以内的候选
if top_p is not None:
    ...
    logits[indices_to_remove] = -float('Inf')

# model.py:301-302 —— 从修改后的分布采样
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

**这些全部是推理代码，一个参数都不在 `W` 里**。但它们对"使用体验"的影响巨大：

```
temperature=0  → 永远选最高概率 → 回答确定、重复、但有时过于刻板
temperature=1  → 按原始分布采样 → 回答多样、但有时跑偏
temperature=2  → 分布被拉平 → 回答发散、容易胡言乱语

top_p=0.9 + temperature=0.7 → 当前行业默认的"兼顾质量和多样性"配方
```

一个有趣的现象：RLHF 之所以让模型显得"更聪明"，部分原因是 RL 训练出的 `W` **在 top-1 位置就已经放了好答案**——所以即使在 `temperature=0` 的极端确定性采样下，它也能给出流畅回答。预训练模型 `temperature=0` 经常退化成"我 我 我 我"这种死循环，就是因为 top-1 不够稳。

**采样参数是推理代码给 `W` 装上的"调节旋钮"**——它不改变 `W`，只改变从 `W` 的输出里怎么选。

---

## 8. W 装不下的东西：tool use / RAG / memory 必须靠外部代码

前面讲的都是"W 里有能力，但需要推理代码正确激活"。这一节讲**有些东西根本不能住在 `W` 里**，必须靠外部代码实现。

### 8.1 Tool use / Function calling

当 ChatGPT 回答"北京今天天气"时，它并不是从 `W` 里把今天的天气"回忆"出来。流程是：

```
用户: "北京今天天气"
  ↓
模型(受过 tool-use SFT): 输出一个结构化调用
  {"name": "get_weather", "args": {"city": "北京"}}
  ↓
【推理循环退出，进入外部代码】
外部代码: 调用真实的天气 API
  天气 API 返回: "12°C, 晴"
  ↓
【外部代码把结果拼回 prompt，重新进入推理循环】
模型: "北京今天 12 度，晴天..."
```

模型在 `W` 里学到的能力是"**识别出这种问题需要调用工具，并以正确的 JSON 格式输出调用请求**"——这是 SFT 写进去的。但**真正调用工具这件事发生在推理代码里**，不是在 `forward` 里。

用 PyTorch 伪代码写就是：

```python
while True:
    logits, _ = model(idx)
    next_tok = sample(logits)
    idx = torch.cat([idx, next_tok], dim=1)

    if next_tok == TOOL_CALL_END_TOKEN:
        tool_call = parse_tool_call(tokenizer.decode(idx))
        tool_result = execute_tool(tool_call)       # ← 纯外部代码
        idx = inject_tool_result(idx, tool_result)  # ← 纯外部代码
```

这就是**模型 + 外部代码共同组成一个系统**的最清楚的例子。单看 `W`，它只是一个会生成 JSON 字符串的语言模型；套上 `execute_tool` 这段外部代码，它才变成"能查天气的 agent"。

### 8.2 RAG（检索增强）

参考 [RAG_VS_SFT.md](RAG_VS_SFT.md) 的详细对比。这里只强调一点：

> **RAG 的知识根本不进入 `W`，全靠推理时的拼 prompt。**

```python
# RAG 伪代码 —— 知识存在向量库里，和 W 完全解耦
q_vec = embedding_model.encode(user_query)
retrieved_docs = vector_index.search(q_vec, k=5)   # ← 外部存储

prompt = f"参考资料:\n{retrieved_docs}\n\n问题:{user_query}"
tokens = tokenizer.encode(prompt)
answer = model.generate(tokens)                    # ← 模型只是在"读外挂"
```

模型没有"记住"这些文档——它只是**现场读了一下**。下一次查询，这批文档不在 `prompt` 里，模型就什么都不知道。

### 8.3 长期记忆 / 多轮对话状态

模型的 `block_size`（本项目 `model.py:180`）是一个硬上限：

```python
assert t <= self.config.block_size, \
    f"Sequence length {t} exceeds block_size {self.config.block_size}"
```

超出这个长度，模型根本无法 forward。长对话、用户偏好、"上次我们讨论过的话题"——这些**必须由外部代码做持久化**（数据库 / 文件 / 向量库），下次对话开始时由外部代码决定哪些历史要重新注入 prompt。

模型自己没有"记忆"——记忆是外部代码的产物。

### 8.4 安全过滤 / 内容审核

RLHF 让模型学会**少数经常见到的**有害请求要拒绝。但工业级系统永远不只靠 `W`——前置的 prompt 分类器、后置的输出审核、黑名单词表，都是纯代码，和模型并行跑。原因：

- `W` 里的"安全"是概率性的，偶尔会被 jailbreak 绕开。
- 代码里的规则是确定性的，可以审计、可以快速更新。

生产系统通常是 **模型 `W` × 代码规则** 的"双保险"。

---

## 9. 回到 CodeGPT：当前代码里哪些就是"脚手架"

本项目只完成了预训练（见 [RLHF 文档](RLHF_AND_PLATONIC_REPRESENTATION.md) 第 7 节）——但推理侧的"脚手架"已经相当完整。把它们列出来，就能清楚地看到"W + 代码 = 可用系统"是怎么组装的：

| 文件 : 行 | 作用 | 属于 W 还是代码 | 训练时对应 |
|---|---|---|---|
| `model.py:183-196` | forward 本体 | 都调用 W | 训练时用的同一个 forward |
| `tokenizer.py:SPECIAL_TOKENS` | 特殊 token ID 表 | 代码 | 训练数据里的 token ID 必须相同 |
| `sample.py:101 encode_prompt` | 把用户文本包装成 `<\|code_start\|>` + `<\|lang:xxx\|>` + 文本 | 代码 | 训练数据也是这个格式 |
| `sample.py:113 encode_fim` | FIM 三段 token 包装 | 代码 | `train.py` 里 `apply_fim_transform` 产出同样格式 |
| `sample.py:98 stop_tokens` | 告诉 generate 循环何时停 | 代码 | 训练数据中文档边界处的 `<\|endoftext\|>` / `<\|code_end\|>` |
| `model.py:279 temperature` | 采样温度 | 代码 | 训练时无此参数 |
| `model.py:287 top_k` | 采样截断 | 代码 | 训练时无此参数 |
| `model.py:292 top_p` | 核采样 | 代码 | 训练时无此参数 |
| `model.py:275 max_new_tokens` | 长度上限 | 代码 | 训练时靠 `block_size` 截断 |
| `sample.py:74-83` | 加载 checkpoint + 剥离 `_orig_mod.` 前缀 | 代码（读 W） | `torch.compile` 训练后保存的 checkpoint |

**如果未来给 CodeGPT 加上 SFT 和 DPO**，需要新增的配套代码：

1. **SFT 侧**：新增一批 `<|user|>` / `<|assistant|>` 特殊 token → 更新 `SPECIAL_TOKENS` 和 `CodeGPTConfig` 的默认值 → 重新 `expand_vocab` → 在 `encode_prompt` 里支持对话格式。
2. **DPO 侧**：训练脚本要多维护一份 `ref_model`（SFT 之后的冻结副本），loss 变成 DPO loss。推理侧**不需要任何改动**——因为 DPO 只改 `W`，不改协议。
3. **stop_tokens**：把新的 `<|assistant_end|>` 加到 `sample.py:98` 的列表里。

注意第 3 条——**光训练完不够，推理侧的 stop_tokens 必须同步更新，否则模型会"停不下来"**，这正是前面第 6 节讲的"训练-推理对齐"要求。

---

## 10. 小结：一张"训练—推理"对齐表

回到用户最初的问题：

> 使用大模型做 QA / 对话时，是纯靠 transformer 里的参数来预测，还是需要代码配合？

答案：

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌─────────────┐       ┌─────────────────────────────┐          │
│  │             │       │                             │          │
│  │   W         │  ⟷   │   推理代码 (脚手架)          │          │
│  │  (权重)     │       │                             │          │
│  │             │       │                             │          │
│  │ ─ 语言能力  │       │ ─ tokenizer + 特殊 token    │          │
│  │ ─ 世界知识  │       │ ─ chat template             │          │
│  │ ─ 指令遵循  │       │ ─ stop_tokens 检测           │          │
│  │ ─ 偏好/品味 │       │ ─ 采样参数 (T/top-k/top-p)   │          │
│  │ ─ 格式习惯  │       │ ─ max_new_tokens 上限        │          │
│  │             │       │ ─ tool 调用循环              │          │
│  │             │       │ ─ RAG 检索 / 历史管理        │          │
│  │             │       │                             │          │
│  └─────────────┘       └─────────────────────────────┘          │
│         ↑                           ↑                           │
│    训练阶段写入                 和训练协议必须一致               │
│   (pretrain → SFT → RL)                                         │
│                                                                 │
│             整个大模型产品 = W + 脚手架                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**三条最核心的认识：**

1. **SFT 和 RL 的成果 100% 存在 `W` 里**。推理时不需要"再跑一次 SFT"或"再查 RL 规则表"——纯靠 `forward` 就能调出这些能力。
2. **但激活这些能力有严格的协议要求**。chat template、特殊 token ID、stop_tokens 三件套必须和训练时一致，否则 `W` 里的能力"沉睡"不醒。
3. **有些能力根本不能住进 `W`**：实时数据 / 工具调用 / 私有知识 / 长期记忆 / 动态安全策略——这些必须由外部代码实现，和模型组成一个系统。**ChatGPT 不是一个模型，是一个"模型 + 大量代码"的产品。**

回到 CodeGPT：本项目的 `sample.py` + `repl.py` + `tokenizer.py` 加起来就是一个**最小化的"模型脚手架"**。可以顺着 `model.py:258` 的 `generate` 和 `sample.py:101` 的 `encode_prompt` 两条线读完整流程——读完之后你就会发现，"使用大模型"这件事本身就是一段不短的代码工作，而不是调一个"预测函数"那么简单。
