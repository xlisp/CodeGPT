# Transformer Debugger：把大模型从黑盒拆成白盒

> "Why does the model output token A instead of token B for this prompt?"
> "Why does attention head H attend to token T here?"
>
> 这两个问题是 OpenAI Superalignment 团队 2024 年开源的 [Transformer Debugger (TDB)](https://github.com/openai/transformer-debugger) 想回答的。它把一个训练好的 GPT——对你我来说是一个只能 `model(tokens)` 的不透明函数——拆成一颗**可探针、可消融、可追溯**的计算图。本文从"调试大模型"这个动作出发，串起 TDB 的三个核心武器：**hooks（内部探针）、ablation（因果消融）、sparse autoencoder（把神经元翻译成人类语言）**，并把每一步映射到本仓库 `model.py` 的对应结构。
>
> 核心观点：**黑盒白盒化不是玄学，它就是在前向过程的关键位置插几个 hook，把隐藏激活取出来看，再把它替换掉看后果。** 其它所有术语（circuit / direction of interest / write vector）都是这个动作的语法糖。

---

## 目录

1. [为什么"调试大模型"和调试普通程序不一样](#1-为什么调试大模型和调试普通程序不一样)
2. [TDB 是什么：一个三层栈](#2-tdb-是什么一个三层栈)
3. [hooks：在前向过程中打断点](#3-hooks在前向过程中打断点)
4. [ablation：因果式调试——把一个神经元置零](#4-ablation因果式调试把一个神经元置零)
5. [direction of interest：用梯度定义"重要"](#5-direction-of-interest用梯度定义重要)
6. [sparse autoencoder：把多义神经元翻译成单义 latent](#6-sparse-autoencoder把多义神经元翻译成单义-latent)
7. [TDB 的 UI：hooks + ablation + 自动解释的装配线](#7-tdb-的-uihooks--ablation--自动解释的装配线)
8. [把 TDB 搬到 CodeGPT 上：一份最小可行改造清单](#8-把-tdb-搬到-codegpt-上一份最小可行改造清单)
9. [黑盒 → 白盒：这条路通向哪里](#9-黑盒--白盒这条路通向哪里)

---

## 1. 为什么"调试大模型"和调试普通程序不一样

调试一段普通 Python 程序，你会做三件事：

1. 下断点、打印中间变量；
2. 改一个值，看下游怎么变；
3. 回溯：这个错误结果从哪个变量传过来的？

训练好的 Transformer 也是一段程序——`CodeGPT.forward()` (`model.py:177`) 就是它的 `main`。但它的变量全是 `(B, T, n_embd)` 的浮点张量，里面塞着上万个神经元的"意义"，没有变量名、没有注释、也没有人类语义。

这就是**黑盒的来源**：不是模型"太复杂"，而是**中间变量没有可读名字**。

所以"调试大模型"的本质操作不变，只是工具要重写：

| 传统调试 | 大模型调试 |
| --- | --- |
| `pdb.set_trace()` 断点 | **forward hook**，取出 `(B, T, d)` 激活 |
| 修改变量看下游 | **ablation**（把某个神经元或 head 置零再跑） |
| 回溯调用栈 | **trace upstream**（看哪个上游 node 的 write vector 促成了当前 node） |
| 变量名告诉你它是什么 | **autoencoder latent + 自动解释**告诉你这个方向编码了什么概念 |

TDB 就是这个映射表的工程化实现。

---

## 2. TDB 是什么：一个三层栈

拉开仓库目录 (`/Users/xlisp/PyPro/transformer-debugger`)：

```
transformer-debugger/
├── neuron_explainer/
│   ├── models/           # 核心：带 hooks 的 Transformer + 稀疏自编码器
│   │   ├── transformer.py
│   │   ├── hooks.py      # ← 白盒化的入口
│   │   └── autoencoder.py
│   └── activation_server/ # FastAPI 后端：托管 hook 执行 + 提供 JSON API
└── neuron_viewer/        # React 前端：TDB UI + 每个 component 的详情页
```

从上到下是三层：

1. **底层 — `models/`**：一个精简版 GPT-2 实现，关键在于 `forward` 的每一步都暴露了 hook 点。这是"白盒化"能力的根。
2. **中层 — `activation_server/`**：把 hook 封装成 HTTP 请求。前端 UI 点"消融这个 head"→ 后端就 `hooks.attn.v.append_fwd(zero_it)` 重跑一次。
3. **上层 — `neuron_viewer/`**：React UI。让你不写代码就能选 token、选组件、选干预方式。

本文只讲底层，因为**一旦理解了 `hooks.py`，上面两层就只是 glue code**。

---

## 3. hooks：在前向过程中打断点

### 3.1 hook 是什么

在 PyTorch 里，"hook"就是**注册到前向/反向过程的回调函数**，能在某个张量流经某层时读它、改它、或把它存下来。

TDB 的 hooks 没有用 `register_forward_hook`，而是自己实现了一套更细粒度的 `HookCollection`（`neuron_explainer/models/hooks.py`）。动机很简单：原生 PyTorch hook 只能挂在 `nn.Module` 的输入/输出上，而 Transformer 里我们关心的很多点（比如 "attention 的 QK logits softmax 之前"、"MLP 激活之后"）**不是模块边界**，只是 `forward` 函数里的一个中间变量。自己实现才能覆盖这些点。

### 3.2 hook 层级：把 Transformer 切成一颗树

看 `hooks.py` 里的 `TransformerHooks` (`hooks.py:228`)：

```python
class TransformerHooks(HookCollection):
    def __init__(self):
        super().__init__()
        self.add_subhooks("mlp", MLPHooks())       # MLP 内部：pre_act / post_act
        self.add_subhooks("attn", AttentionHooks()) # Q/K/V、qk_logits、qk_probs、v_out
        self.add_subhooks("resid", ResidualStreamHooks())  # 残差流的 8 个切面
        self.add_subhooks("logits", FwdBwdHooks())  # 最终输出
```

这棵树里每一片叶子（`FwdBwdHooks`）都同时接受三种回调：

```python
class FwdBwdHooks(HookCollection):
    def __init__(self):
        self.add_subhooks("fwd",  Hooks())                              # 前向
        self.add_subhooks("bwd",  WrapperHooks(wrapper=grad_hook_wrapper))  # 反向
        self.add_subhooks("fwd2", Hooks())   # 反向完再跑一次前向——用于读梯度修改后的结果
```

把它和本仓库的 `CodeGPT` 对照：

| TDB hook 位置 | CodeGPT 对应代码 |
| --- | --- |
| `hooks.resid.post_emb` | `model.py:` `x = tok_emb + pos_emb` 之后 |
| `hooks.attn.qk_probs` | `CausalSelfAttention` 里 `F.softmax(att, dim=-1)` 之后 |
| `hooks.attn.v_out` | attention `y = att @ v` 之后、`c_proj` 之前 |
| `hooks.mlp.pre_act` | `MLP` 里 `c_fc(x)` 之后、`gelu` 之前 |
| `hooks.mlp.post_act` | `MLP` 里 `gelu(...)` 之后、`c_proj` 之前 |
| `hooks.resid.torso.delta_mlp` | `x = x + self.mlp(self.ln_2(x))` 里的 `self.mlp(...)` |
| `hooks.logits` | `self.lm_head(x)` 的输出 |

也就是说——TDB 不是一个神秘的第三方工具，它只是把你自己每次 debug 都要手写的 `x.detach().clone()` 语句**提前写好、命名、组织成一棵树**。

### 3.3 最小示例：读一个 MLP 激活

（来自 `neuron_explainer/models/README.md`）

```python
from neuron_explainer.models.hooks import TransformerHooks
from neuron_explainer.models.transformer import Transformer

xf = Transformer.load("gpt2-small", dtype=torch.float32, device=device)

activation_cache = {}
def store_forward(xx, layer, **kwargs):
    activation_cache[layer] = xx.detach().clone()
    return xx  # 记得原样返回，否则你就篡改了前向

hooks = TransformerHooks()
hooks.mlp.post_act.append_fwd(store_forward)

xf(tokens, hooks=hooks)
# 现在 activation_cache[3] 是第 3 层 MLP 在 gelu 之后的激活，形状 (B, T, 3072)
```

两行关键：`hooks.mlp.post_act.append_fwd(fn)` 注册，`xf(tokens, hooks=hooks)` 触发。

一旦你手里有了每层每个 token 的激活，后续一切（可视化、ablation、找 top-activating 样本、喂给自编码器）都是普通 NumPy/PyTorch 操作。**这就是"白盒化"这个动作的全部物理内容。**

### 3.4 `AtLayers`：只在某几层触发

真实 debug 时你通常只关心某几层：

```python
from neuron_explainer.models.hooks import AtLayers

# 只在 layer 5 和 layer 7 收集
only_some_layers = AtLayers([5, 7])
only_some_layers.append(store_forward)
hooks.mlp.post_act.append_fwd(only_some_layers)
```

`AtLayers` 是 `ConditionalHooks` 的子类——`hook(x, layer=...)` 被调用时读 `layer` kwarg 再决定要不要跑回调。

---

## 4. ablation：因果式调试——把一个神经元置零

读激活只能看"是什么"，**消融（ablation）才能回答"是不是因果"**。

TDB 里 ablation 就是一个改值的 forward hook：

```python
def make_ablation_hook(at_layer, neuron):
    def ablate(xx, layer, **kwargs):
        if layer == at_layer:
            xx[..., neuron] = 0  # 把第 neuron 维直接清零
        return xx
    return ablate

hooks = TransformerHooks()
hooks.mlp.post_act.append_fwd(make_ablation_hook(3, 300))

out_with_ablation    = xf.sample(prompt, hooks=hooks, num_tokens=10)
out_without_ablation = xf.sample(prompt, num_tokens=10)
```

对比两次输出的差异，就是"第 3 层第 300 号神经元对本次生成的因果效应"。

这在计量经济学里叫"反事实"：**如果这个神经元没激活，模型会说什么？** 在 TDB 的 terminology 里叫 **ablate**——目前实现的是 zero ablation，也就是"把这个 node 对残差流的 write vector 置零"。

### 什么时候用 ablation？

- **定位功能**：怀疑某个 head 负责"主语-动词一致"，就只在主语 token 上 ablate 它，看动词概率有没有崩。
- **验证假说**：看到某个 neuron 的 top-activating 样本都是 Python 缩进，ablate 它，看 `def`/`if` 后面的换行概率降没降。
- **画 circuit**：依次 ablate 候选 node，留下那些"ablate 后 direction of interest 明显变化的" node，它们就是一条 circuit 的骨架。

---

## 5. direction of interest：用梯度定义"重要"

Ablation 告诉你一个 node 重不重要；**direction of interest** 告诉你"对什么重要"。

给定 prompt + 一个 **target token** A 和 **distractor token** B，direction of interest 是：

```
d = W_U[A] - W_U[B]   # 两个 unembedding 向量之差，形状 (n_embd,)
```

在这个方向上的投影等于 `logit(A) - logit(B) = log p(A)/p(B)`。这就是一个可微的标量，你可以对它求梯度：

```python
loss = (final_resid @ d).sum()
loss.backward()  # 每层每个 token 的激活都会有梯度
```

TDB 定义的两个关键估计量（见 `terminology.md`）：

- **Direct effect = activation · (gradient from final resid)**：只算这个 node 对最终残差流的直接影响。
- **Estimated total effect = activation · (gradient anywhere)**（act × grad）：把沿途所有中间 node 的放大/衰减都包进去。正值表示这个 node 在推模型说 A，负值表示在推模型说 B。

用 code 表达就是：在 backward 阶段把每个 hook 点的 `(act, grad)` 存下来，乘起来、排序，就得到"对 A-vs-B 这个问题贡献最大的 node 列表"。

**这是"黑盒白盒化"最锋利的一刀**：你不需要理解 12 层 × 12 head × 3072 neuron 里每一个在干嘛，只需要告诉 TDB "我关心 A 比 B 为什么概率高"，它就能按 `act·grad` 排序，把你应该先看的几十个 node 列出来。

---

## 6. sparse autoencoder：把多义神经元翻译成单义 latent

hooks + ablation + act·grad 已经能把 circuit 画出来了。但还有一个问题：**单个 MLP 神经元通常是"多义的"**——同一个神经元在 "code 缩进"、"括号匹配"、"日期"上都会激活，你没法给它一个干净的语义标签。

这是 Anthropic 在 [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) 里展示的核心现象：**真正单义的特征是神经元的线性组合，而不是单个神经元**。

TDB 的解法：在每层 MLP 激活上训练一个**过完备的稀疏自编码器**：

```
MLP post_act (3072-d) → encoder → latents (32768-d, mostly zero) → decoder → MLP post_act'
```

- 输入 3072 维（GPT-2 small 的 MLP 宽度）；
- 隐层 32768 维（~10× 扩展）——刻意比输入宽，但加 L1 使大多数维度为 0；
- 解码出来的要尽可能等于输入。

训练目标是 "重建 loss + λ·||latents||₁"。L1 逼迫同一时刻只有少数 latent 被激活，相当于给每个激活模式一个 one-hot 式的标签。实验结果：这些 latent **比原始 neuron 单义得多**——一个 latent 就对应一个可解释的概念（"HTTP 头"、"Python 函数定义"、"DNA 序列"……）。

用 `AutoencoderHooks` 在前向过程中插入这个编码-解码：

```python
# 伪代码，核心逻辑见 hooks.py:AutoencoderHooks.__call__
latents       = encode(mlp_post_act)
latents       = self.latents(latents, ...)           # 这里可以 ablate 某个 latent
reconstruction = decode(latents)
reconstruction = self.reconstruction(reconstruction, ...)
return reconstruction   # 替代原来的 mlp_post_act 继续前向
```

注意这里 ablation 的颗粒度变了：**不是 ablate neuron，是 ablate latent**——而 latent 有语义，所以你可以说"关掉 Python-缩进这个 feature，看它对下一 token 的影响"。

这一步把 hook 体系和 circuit 分析**从不可读的 3072 维升级到可读的 32768 维稀疏空间**。

---

## 7. TDB 的 UI：hooks + ablation + 自动解释的装配线

有了以上机制，TDB 的 UI 就只是把它们组合成流水线：

1. 用户输入 prompt + target/distractor 两个候选 token。
2. 后端跑一次前向+反向，用 hooks 收集每个 (layer, component, token) 的 **act·grad**，排序。
3. UI 列出 top-K 最重要的 node。点进去看：
   - 这个 component（neuron / head / latent）在大规模语料上的 top-activating 样本（由自动解释流水线预先算好，[Language models can explain neurons](https://openai.com/research/language-models-can-explain-neurons-in-language-models)）；
   - GPT-4 为它生成的英文解释（"activates on Python function definitions"）；
   - 其直接下游（它的 write vector 最靠近哪些 unembedding 方向）。
4. 一键 ablate / 修改激活，重跑，看 direction of interest 变化。
5. 对一个 node 点 "trace upstream" → 对它的激活求梯度 → 重复 (2)。

整个过程就是**递归地把 hook 指针往上游挪**，直到你能写下一句自然语言假设："layer 9 head 9 是 name mover 头，它把 subject name 从前文复制到当前位置" （[IOI paper](https://arxiv.org/abs/2211.00593) 的发现）。

---

## 8. 把 TDB 搬到 CodeGPT 上：一份最小可行改造清单

TDB 的 `Transformer` 是自己写的——它不能直接吃 CodeGPT 的 checkpoint。如果你想在本仓库上复现这套 debug 能力，需要以下几步：

1. **给 `model.py` 加 hook 点。** 最省事的做法是模仿 `TransformerHooks` 的切面，在 `Block.forward` 里每个中间变量后面调一次 `hooks(...)`：

   ```python
   # model.py (改造示意，非现有代码)
   def forward(self, x, hooks=None, layer=None):
       a = self.attn(self.ln_1(x))
       if hooks: a = hooks.resid.torso.delta_attn(a, layer=layer)
       x = x + a
       m = self.mlp(self.ln_2(x))
       if hooks: m = hooks.resid.torso.delta_mlp(m, layer=layer)
       x = x + m
       return x
   ```

2. **通过 `ModelContext` 适配层包一层。** TDB 的整套后端都基于 `ModelContext` 抽象，只要 CodeGPT 实现对应接口（参见 `neuron_explainer/models/model_context.py`），前端几乎不用改。

3. **（可选）训练稀疏自编码器。** 本仓库的 tokenizer 有 FIM / code / lang 特殊 token，直接套 GPT-2 small 上训好的 SAE 不合适。用 `train.py` 的 hooks 分支抓 `mlp.post_act` 激活，跑 `sparse_autoencoder` 库训一批，就能获得针对代码语料的 latent 字典。

4. **(重) 生成 collated activation datasets。** TDB 的自动解释需要"这个 component 在大语料上激活最高的 top-K 样本"，对每层每个 component/latent 做一遍离线扫描并存 JSON。这是整个栈里最贵的一步——GPT-2 small 12 × 3072 neurons × 32768 latents 的数据集 OpenAI 预先算好放在公开 Azure bucket 上，自训模型需要自己补。

实际 debug 工程里，第 1 步的收益最大：**只要把 hooks 加进去**，你就已经能用本文 §3–§5 的所有技巧手工分析 CodeGPT 了。UI 和 SAE 是锦上添花。

---

## 9. 黑盒 → 白盒：这条路通向哪里

把本文倒过来读，就是"黑盒白盒化"的一个层级阶梯：

- **Level 0：API 级黑盒。** 你只有 `model(tokens) → logits`。调试 = 换 prompt 试错。
- **Level 1：hook 取激活。** `forward` 里关键点挂 callback，隐藏状态变成可读张量。CodeGPT 现在就能做到，只要改几行 `model.py`。
- **Level 2：ablation + direction of interest。** 用 act·grad 给"重要性"一个定量定义，从几千个 node 里按重要性排序。白盒化从"能看"升级到"能对比"。
- **Level 3：sparse autoencoder latent。** 给隐藏表示做字典学习，把多义神经元翻译成单义 feature。白盒化从"能对比"升级到"能命名"。
- **Level 4：circuit。** 串起若干命名后的 latent + head，形成一条可以用自然语言描述的算法（如 IOI 的 "name mover + duplicate token + S-inhibition" 三段式）。白盒化从"能命名"升级到"能讲清逻辑"。

TDB 的工程意义在于：**它证明了 Level 1–3 完全可以用一两百行 PyTorch hook + 一个中等规模自编码器 + 一套 UI 搭出来**。没有神秘技术，只是把每一次手工 debug 固化成了可复用的工具。

对本仓库的启示也很直接：

- 训练 loss 降不下去、模型在某类代码上总是出错、FIM 模式下模型对 `<|fim_middle|>` 不敏感——这些问题都可以从"加 hook 看激活"开始排查，而不是只改超参。
- `docs/COMPRESSION_IS_INTELLIGENCE.md` 说"GPT 把世界压进了 n_embd 维空间"。TDB 告诉你：**那个空间不仅可压，还可读、可改、可归因。** 黑盒白盒化不是哲学口号，是一套你今天就能动手做的工程实践。

---

## 参考

- 仓库：`/Users/xlisp/PyPro/transformer-debugger` · [openai/transformer-debugger](https://github.com/openai/transformer-debugger)
- Terminology：`transformer-debugger/terminology.md`
- Hooks 实现：`transformer-debugger/neuron_explainer/models/hooks.py`
- Models README（含可直接跑的示例代码）：`transformer-debugger/neuron_explainer/models/README.md`
- 理论背景：[Towards Monosemanticity (Anthropic, 2023)](https://transformer-circuits.pub/2023/monosemantic-features) · [Language models can explain neurons in language models (OpenAI, 2023)](https://openai.com/research/language-models-can-explain-neurons-in-language-models) · [IOI paper (Wang et al., 2022)](https://arxiv.org/abs/2211.00593)
