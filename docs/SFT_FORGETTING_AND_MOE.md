# 多次 SFT 的灾难性遗忘：SFT 的本质、MoE 的本质、以及它们各自解决什么问题

> 用户问题：LLM 做了第一次 SFT 学会代码能力，再做第二次 SFT 学 function calling，结果代码能力被"盖掉"了——这怎么办？多个能力是堆多次 SFT，还是上 MoE？
>
> 这篇文档回答两件事：
> 1. **SFT 的本质**是什么？为什么"第二次 SFT 覆盖第一次"不是 bug 而是目标函数在数学上的必然。
> 2. **MoE 的本质**是什么？它是"可微分的 if/else"，用架构手段把"不同能力"解耦到不同的参数子空间——这才是解决多能力叠加的根本路线。

---

## 目录

1. [问题的起点：第二次 SFT 为什么会盖掉第一次](#1-问题的起点第二次-sft-为什么会盖掉第一次)
2. [SFT 的本质：换了语料的预训练，损失函数完全一样](#2-sft-的本质换了语料的预训练损失函数完全一样)
3. [灾难性遗忘的数学：梯度下降只关心当前 batch](#3-灾难性遗忘的数学梯度下降只关心当前-batch)
4. [解法一：数据混合（Rehearsal / Replay）](#4-解法一数据混合rehearsal--replay)
5. [解法二：LoRA + 多适配器 —— 物理上隔离增量](#5-解法二lora--多适配器--物理上隔离增量)
6. [解法三：EWC —— 给"老能力的重要参数"加弹簧](#6-解法三ewc--给老能力的重要参数加弹簧)
7. [解法四：多任务联合 SFT —— 一次训完，根治遗忘](#7-解法四多任务联合-sft--一次训完根治遗忘)
8. [MoE 的本质：把一个 MLP 拆成"很多专家 + 一个路由器"](#8-moe-的本质把一个-mlp-拆成很多专家--一个路由器)
9. [为什么 MoE 天然更抗遗忘？稀疏激活的参数隔离](#9-为什么-moe-天然更抗遗忘稀疏激活的参数隔离)
10. [MoE vs 多次 SFT vs LoRA：该选哪条路](#10-moe-vs-多次-sft-vs-lora该选哪条路)
11. [回到 CodeGPT：如果要加 function calling 能力](#11-回到-codegpt如果要加-function-calling-能力)

---

## 1. 问题的起点：第二次 SFT 为什么会盖掉第一次

一个典型的"踩坑"路径：

```
预训练 checkpoint
    ↓ SFT_1: 代码数据   (loss 降到 0.8，HumanEval 提到 45%)
代码能力 checkpoint
    ↓ SFT_2: function calling 数据   (loss 降到 0.5，tool-use 提到 80%)
"双能力" checkpoint ?

实际结果：
  - function calling 很强 ✓
  - HumanEval 从 45% 掉到 18% ✗   ← 灾难性遗忘 (catastrophic forgetting)
```

这个现象在学术上有个正式名字：**catastrophic forgetting / catastrophic interference**，最早由 McCloskey & Cohen（1989）在连接主义网络里观察到，深度学习时代几乎所有连续学习（continual learning）论文都在和它作斗争。

它不是"模型坏了"，它是训练目标函数的**必然结果**。要理解为什么，先理解 SFT 到底在做什么。

---

## 2. SFT 的本质：换了语料的预训练，损失函数完全一样

很多人以为 SFT 是一种"新的训练算法"。**不是**。SFT 在数学上和预训练是**同一个东西**——都是 next-token prediction 上的交叉熵最小化。看 `model.py:192`：

```python
# 预训练的 loss（本项目当前阶段）
loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                       targets.view(-1), ignore_index=-1)
```

SFT 的 loss 写出来一模一样：

```python
# SFT 的 loss
loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                       targets.view(-1), ignore_index=-1)
```

**唯一的区别只在 `targets` 里哪些位置是 `-1`。** 预训练里目标就是原始 token 流；SFT 里通常把"用户提问"那部分的 target 设成 `-1`（屏蔽梯度），只让"助手回答"那部分产生 loss：

```python
# SFT 数据构造（伪代码）
prompt = "<|user|>写一个快排<|assistant|>"
answer = "def quicksort(arr): ..."

tokens  = tokenizer.encode(prompt + answer)
targets = tokens.copy()
targets[:len(prompt_tokens)] = -1   # prompt 部分不算 loss
# 剩下的就是标准 next-token prediction
```

这个 `-1` 的机制本项目已经在用——FIM 数据增强把 pad 位置设为 -1（见 `train.py:get_batch` 和 `CLAUDE.md` 里对 FIM pad 的描述），`cross_entropy(ignore_index=-1)` 就会自动跳过。SFT 只是把同样的机制用在 "prompt vs answer" 的切分上。

**一句话本质：SFT 就是在一份精挑细选的小语料上继续做预训练，仅此而已。**

既然 SFT 就是预训练，那就遵守预训练的所有物理规律——包括：**梯度下降只会让当前 batch 的 loss 变小，它不知道、也不在乎模型之前会做什么**。这就是遗忘的根源。

---

## 3. 灾难性遗忘的数学：梯度下降只关心当前 batch

训练一步就是：

```python
logits, loss = model(X_batch, Y_batch)   # 当前 batch
loss.backward()
optimizer.step()                          # θ ← θ - lr · ∇loss
```

`∇loss` 是**当前 batch** 的梯度，完全不含任何"之前学过的代码能力"的信息。换句话说：

$$
\theta_{\text{SFT2}} = \theta_{\text{SFT1}} - \eta \sum_{t=1}^{T} \nabla \mathcal{L}_{\text{fc}}(\theta_t)
$$

这个迭代过程**只看 function calling 数据**，它会把参数推到"function calling loss 最低"的方向。如果这条方向恰好和"代码 loss 最低"的方向不一致（几乎总是不一致），代码能力就会被顺带破坏。

一个直观类比：把模型参数想成一个沙盘，代码 SFT 把沙子堆成山 A，function calling SFT 的梯度是一双推土机，它只看山 B 的目标形状，并不知道山 A 的存在，于是把山 A 推平了。

学术上有个更精细的量化指标——**Fisher 信息矩阵** $F_{ii} = \mathbb{E}[(\partial \log p / \partial \theta_i)^2]$ 衡量第 $i$ 个参数对老任务有多重要。遗忘严重程度正比于 $\sum_i F_{ii} (\Delta \theta_i)^2$。后面 EWC 就是直接拿这个量当正则项。

现在我们知道敌人长什么样了，看解法。

---

## 4. 解法一：数据混合（Rehearsal / Replay）

**最朴素、最有效、工业界默认的做法：不要做两次 SFT，把两份数据混在一起做一次。**

```python
# 错误做法
train(model, code_sft_data,   steps=5000)   # SFT_1
train(model, fc_sft_data,     steps=5000)   # SFT_2 ← 灾难性遗忘

# 正确做法：rehearsal
mixed = interleave(code_sft_data, fc_sft_data, ratio=[0.5, 0.5])
train(model, mixed, steps=10000)            # 一次搞定
```

如果实在必须分阶段（比如代码数据已经训完了，现在才拿到 function calling 数据），**关键技巧：在 SFT_2 里按比例保留 SFT_1 的老数据**：

```python
# 保留 10%~30% 老数据 = 便宜的 rehearsal
sft2_data = sample(code_sft_data, frac=0.2) + fc_sft_data
```

经验比例在 10%~30%，视任务冲突程度。这是 OpenAI、Anthropic、DeepSeek 在训 instruct 版本时都在用的"土办法"，效果出奇得好。

本质上，它把"顺序学习"退化回"联合学习"，从根上绕过了遗忘。

---

## 5. 解法二：LoRA + 多适配器 —— 物理上隔离增量

LoRA（Low-Rank Adaptation，Hu et al. 2021）的思想是：**不要改原始权重 $W$，而是学一个低秩增量 $\Delta W = BA$，推理时 $W + BA$**。用 PyTorch 写出来大概长这样（可以对比 `model.py:80` 的 `self.c_fc`）：

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8):
        super().__init__()
        self.base = base                                 # 冻结
        for p in self.base.parameters():
            p.requires_grad = False
        d_in, d_out = base.in_features, base.out_features
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, rank))  # 初始 ΔW = 0

    def forward(self, x):
        return self.base(x) + x @ self.A.T @ self.B.T
```

关键两点：

1. `self.base` 全部冻结——**原始预训练知识物理上不可能被改动**。
2. `A`、`B` 只有几百万参数（rank=8 时约占全量的 0.1%~1%），训练便宜。

有了 LoRA，多能力叠加的范式变成：

```
基座模型 W₀ (冻结)
  ├── LoRA_code   (独立一套 A,B)
  ├── LoRA_fc     (独立一套 A,B)
  └── LoRA_math   (独立一套 A,B)

推理时按需加载一个，或者把多个权重线性合并：
  W = W₀ + α·ΔW_code + β·ΔW_fc + γ·ΔW_math
```

遗忘问题直接消失——因为每个能力住在自己的 A/B 里，彼此不覆盖。缺点是：多个 LoRA 合并时，如果两个任务的梯度方向冲突，线性叠加仍然会相互干扰（没有 MoE 的路由那么"清洁"）。

---

## 6. 解法三：EWC —— 给"老能力的重要参数"加弹簧

Elastic Weight Consolidation（Kirkpatrick et al. 2017, DeepMind）的思路很朴素：

> 每个参数对老任务的重要性不一样。对重要参数，惩罚它偏离老值；对不重要的参数，随便动。

实现上，在 SFT_2 的 loss 上加一个正则项：

```python
# 在老任务（code SFT）结束时，估计每个参数的 Fisher 重要性
fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
for X, Y in code_sft_loader:
    _, loss = model(X, Y)
    loss.backward()
    for n, p in model.named_parameters():
        fisher[n] += p.grad.detach() ** 2
    model.zero_grad()
fisher = {n: f / len(code_sft_loader) for n, f in fisher.items()}
theta_star = {n: p.detach().clone() for n, p in model.named_parameters()}

# SFT_2 的 loss = 新任务 loss + 弹簧项
def ewc_loss(new_loss, model, fisher, theta_star, lam=1e4):
    penalty = 0
    for n, p in model.named_parameters():
        penalty = penalty + (fisher[n] * (p - theta_star[n]) ** 2).sum()
    return new_loss + lam * penalty
```

直觉：Fisher 矩阵高的参数 = 老任务很依赖它 = 加一根粗弹簧把它拉回老位置；Fisher 低的参数 = 随便动。

优点是**不需要留老数据**。缺点是 Fisher 是个近似，超参 λ 很难调——大了学不动新任务，小了还是会忘。实践中 EWC 比 rehearsal 更"学术"，在 LLM 工业界不如数据混合常见。

---

## 7. 解法四：多任务联合 SFT —— 一次训完，根治遗忘

把前三种解法推到极致，就是**多任务联合 SFT**：不分阶段，把代码、function calling、数学、对话所有想要的能力数据按比例混成一份大语料，从预训练 checkpoint 一次性 SFT 出来。

训练循环甚至不用改——本项目 `train.py:304` 已经长这样了：

```python
X, Y = get_batch('train')
# ...
logits, loss = model(X, Y)
loss.backward()
optimizer.step()
```

唯一需要做的是在 `data/<dataset>/prepare.py` 阶段把多种数据按比例 concatenate 成一份 `train.bin`。每个 batch 里自然就混了多种任务，梯度同时优化所有能力。

这是 Llama、Qwen、DeepSeek-Coder 实际的做法。他们的"SFT 数据"是一个包含几百种子任务、有精心配比的混合物，不是先 A 后 B 串行训的。

**但它有个硬限制**：所有能力共享同一套参数 θ。如果任务数量增长到几十上百（通用助手 + 代码 + function calling + math + 翻译 + 医疗 + 法律 + …），参数之间开始互相挤压——这就是 MoE 登场的时机。

---

## 8. MoE 的本质：把一个 MLP 拆成"很多专家 + 一个路由器"

先回到 CodeGPT 的标准 Transformer block，看 `model.py:93-105`：

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)        # ← 这里只有一个 MLP

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x)) # ← 所有 token 都走这同一个 MLP
        return x
```

每一层只有**一个 MLP**（`model.py:76-90`），每个 token 必须经过它。这个 MLP 承载了这一层所有的"事实性知识"——语法、语义、代码模式、函数调用约定，全挤在 `c_fc` 和 `c_proj` 这两个矩阵里。

MoE（Mixture of Experts，Shazeer et al. 2017 / Switch Transformer 2021 / Mixtral 2024）的改造：**把一个 MLP 换成 N 个 MLP（"专家"）+ 一个小路由器**，每个 token 只被路由到 top-k 个专家。用本项目的命名风格写出来大约是：

```python
class MoEBlock(nn.Module):
    def __init__(self, config, n_experts=8, top_k=2):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        # 多个 MLP，每个都是原来那个 MLP
        self.experts = nn.ModuleList([MLP(config) for _ in range(n_experts)])
        # 路由器：把 token 的向量映射成 N 个专家的打分
        self.router = nn.Linear(config.n_embd, n_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        h = self.ln_2(x)                             # (B, T, C)
        scores = self.router(h)                      # (B, T, N)
        weights, idx = scores.topk(self.top_k, dim=-1)   # 每个 token 选 top-k 专家
        weights = F.softmax(weights, dim=-1)         # 归一化

        # 稀疏地把每个 token 送去对应专家（这里伪代码，实际要 gather/scatter）
        out = torch.zeros_like(h)
        for k in range(self.top_k):
            for e in range(len(self.experts)):
                mask = (idx[..., k] == e)            # 哪些 token 选了专家 e
                if mask.any():
                    out[mask] += weights[..., k:k+1][mask] * self.experts[e](h[mask])

        return x + out
```

关键几个数字（以 Mixtral 8x7B 为例）：

- 8 个专家，每个是一个完整的 MLP。总参数量 ~47B。
- 每个 token 只激活 top-2 个专家 → 实际参与计算的参数量 ~13B。
- 所以 Mixtral 有 47B 参数但跑起来像 13B——**容量 = 47B，算力 = 13B**。

**本质一句话：MoE 是 "可微分的 if/else"。** 路由器学会"看到代码相关的 token 就派给专家 3 和 5，看到自然语言就派给专家 1 和 7"——注意这个"派"是通过 softmax 权重实现的，所以可微、可训练。对应 [DIFFERENTIABLE_PROGRAMMING.md](DIFFERENTIABLE_PROGRAMMING.md) 第 5 节讲的那种"把离散控制流翻译成可微权重"的思想——MoE 就是把 `if task == 'code': mlp_code(x) else: mlp_general(x)` 做成可微的版本。

---

## 9. 为什么 MoE 天然更抗遗忘？稀疏激活的参数隔离

回到遗忘的数学：SFT_2 会更新**所有**被当前 batch 激活的参数。在稠密 MLP 里，每个 token 都走同一个 MLP，所有参数都被激活，所以所有参数都被覆盖——老能力无处躲藏。

在 MoE 里：

```
function calling 数据进来
  → 路由器倾向于派给专家 2 和 6（比如）
  → 只有专家 2 和 6 的参数被更新
  → 专家 0, 1, 3, 4, 5, 7 完全不动 ← 代码能力可能主要住在这里，得以保留
```

这就是 MoE 的**参数隔离（parameter isolation）**——不同能力自动住进不同的专家，梯度只流过被路由到的子集。它不是 100% 的隔离（路由器本身还是会变），但比稠密模型的"全员更新"强太多了。

实际训练 MoE 还要加一个 **load balancing loss**（让每个专家的负载大致均衡），否则路由器会退化成"所有 token 都派给一个最强的专家"——这个细节这里不展开。

这也解释了为什么 DeepSeek-V2/V3、Mixtral、Qwen2-MoE 都选择 MoE 架构来做通用助手：**当目标是"同时做好几十种任务"时，MoE 从架构层就在解决能力互相干扰的问题，而稠密模型只能靠数据配比硬调**。

---

## 10. MoE vs 多次 SFT vs LoRA：该选哪条路

| 方案 | 是否改架构 | 是否解决遗忘 | 参数开销 | 工程难度 | 适用场景 |
|---|---|---|---|---|---|
| 朴素多次 SFT | 否 | ✗ 不解决 | 0 | 低 | 不要这样做 |
| 数据混合（rehearsal） | 否 | ✓ 部分 | 0 | 低 | **90% 的场景默认选这个** |
| 多任务联合 SFT | 否 | ✓ 根治 | 0 | 中（需配比） | 所有能力一次性训 |
| LoRA 多适配器 | 小改 | ✓ 物理隔离 | 每任务 +1% | 低 | 需要独立能力开关、推理时切换 |
| EWC / L2-SP 正则 | 否 | ✓ 部分 | 0 | 高（λ 难调） | 没有老数据可用 |
| **MoE** | **大改** | **✓ 架构级** | **参数 ×N，FLOPs ~不变** | **高** | **能力非常多、模型规模大** |

决策顺序建议：

1. 只有两三种能力、总体数据量 <10B token → **数据混合 / 联合 SFT**，不要想别的。
2. 需要"能力即插即用"、不同场景切不同配置 → **LoRA + 多适配器**。
3. 能力种类 >10、模型已经 >30B、在做通用助手 → **MoE**。
4. **"做多次独立 SFT"几乎在任何场景下都不是正解。** 它看起来最直观（像 fine-tune 流水线），但数学上就是在制造遗忘。

---

## 11. 回到 CodeGPT：如果要加 function calling 能力

本项目当前是纯预训练阶段（`model.py:136` 的 `CodeGPT` 类 + `train.py` 的 next-token CE loss，没 SFT 也没 RLHF，参见 [RLHF_AND_PLATONIC_REPRESENTATION.md](RLHF_AND_PLATONIC_REPRESENTATION.md) 第 7 节）。如果未来要在这之上加 function calling，按本文的分析，推荐路径是：

**Step 1 —— 扩充特殊 token。** 参照 `tokenizer.py` 里已有的 FIM/code/lang token 的做法，在 `SPECIAL_TOKENS` 里加 `<|tool_call|>`、`<|tool_result|>` 等，更新 `CodeGPTConfig` 里对应的 `*_id`，然后用 `model.expand_vocab(new_size)`（`model.py` 已实现）扩词表。注意 `VOCAB_SIZE` 要保持 64 的倍数。

**Step 2 —— 数据混合联合 SFT，不要分两次。** 准备两份 SFT 数据：
- 代码对话 SFT（"写个快排" → 代码）
- function calling SFT（"查北京天气" → `<|tool_call|>{"name":"get_weather","args":...}<|tool_result|>...`）

按 7:3 或 5:5 混进一份 `train.bin`，复用 `train.py` 跑一次。`get_batch` 里把 prompt 段的 target 设成 -1（机制本项目已经有了，就是 FIM pad 那套）。

**Step 3 —— 先不要上 MoE。** 这个项目在 10M~124M 参数规模（见 `config/train_codegpt_small.py` 等），MoE 的收益（容量 vs 算力解耦）在这个规模看不出来，反而会让路由器不稳定、负载不均衡。MoE 是模型规模上到 7B+、能力种类上到 10+ 之后才值得投入的复杂度。

**Step 4 —— 如果未来真要做 agent 类的多能力模型（代码 / FC / 搜索 / 数学 / 浏览器），** 再考虑把 `Block.mlp` 换成 MoE 版本。届时需要额外实现：(1) 路由器和 top-k 选择；(2) load balancing loss；(3) 训练时的 expert parallelism（在单 GPU 上 8 个专家还放得下，再大就要切）。本文第 8 节给出的伪代码是一个起点。

---

## 小结

- **SFT ≠ 新算法**。SFT 就是在精选小数据上继续做 next-token prediction，损失函数和 `model.py:192` 一模一样。
- **多次独立 SFT 会遗忘**，不是 bug，是梯度下降只看当前 batch 的数学事实。
- **数据混合** 是 90% 场景的正解；**LoRA** 解决"能力插拔"；**EWC** 在没老数据时救场。
- **MoE 的本质是可微分的 if/else**——用稀疏路由把不同能力住进不同专家，从架构层解决遗忘和容量扩展。它是"同时做几十种任务 + 模型规模大"时的长期正解，但在小模型阶段性价比不高。
- 对 CodeGPT 这种规模，**先用联合 SFT 拿掉多能力问题，MoE 留给未来真正需要的时候**。
