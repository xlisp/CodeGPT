# 同一个 Transformer，吃掉一切：为什么语音、图像、自动驾驶、大模型都在用它

> 2017 年 Google 的 *Attention Is All You Need* 只是想把机器翻译做得更好。八年过去，这个最初用来翻译英德文的架构，已经成了语音识别、图像分类、视频生成、自动驾驶、蛋白质折叠、机器人控制的共同底座。几乎每一个领域的 SOTA 都长着一张 Transformer 的脸。
>
> 本文回答三件事：
> 1. **为什么是 Transformer？** 它做对了哪几件事，让 CNN、RNN、HMM、图模型全面退场。
> 2. **每个领域具体怎么用？** 列一份 2024-2026 的 SOTA 地图：模型名、input 怎么 tokenize、架构在哪里变了。
> 3. **它们和 CodeGPT 到底有多像？** 拿 `model.py:177-198` 的 `forward` 做基准，你会发现图像、语音、驾驶模型的主干几乎逐行对应。

---

## 1. 一张图：Transformer 统治的版图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Transformer 架构（2017）                      │
└─────────────────────────────────────────────────────────────────┘
        │
        ├── 语言/代码 ──► GPT-4 / Claude / Llama / DeepSeek / CodeGPT(本项目)
        │
        ├── 图像识别 ───► ViT / Swin / DINOv2 / SAM / CLIP
        │
        ├── 图像生成 ───► DiT(Stable Diffusion 3) / Sora / Imagen 3
        │
        ├── 语音识别 ───► Whisper / Conformer / SeamlessM4T
        │
        ├── 语音生成 ───► VALL-E / SoundStorm / AudioLM
        │
        ├── 视频 ───────► Sora / Veo / VideoPoet
        │
        ├── 自动驾驶 ───► Tesla FSD v12 / BEVFormer / UniAD / Waymo Foundation
        │
        ├── 机器人 ─────► RT-2 / PaLM-E / Octo / π0
        │
        ├── 蛋白质 ─────► AlphaFold 2/3 / ESM-2 / RoseTTAFold
        │
        └── 时间序列 ───► TimesFM / Chronos / MOIRAI
```

曾经每个领域都有自己的"护城河架构"——CV 是 CNN，ASR 是 LSTM+CTC，推荐是 FM，机器人是 RL+MPC，蛋白质是 CNN+注意力混合。**到 2025 年，这些护城河都被同一条河填平了。** 填平它的不是某家公司的具体模型，而是 Transformer 这个架构模板本身。

---

## 2. 为什么 Transformer 能赢？三个根本原因

### 原因一：最小归纳偏置（Minimal Inductive Bias）

CNN 假设图像"局部相关、平移等变"——这是对图像的硬编码先验。RNN 假设序列"只能从左到右、用固定容量压缩历史"——这是对序列的硬编码先验。

Transformer 的假设近乎为零：

```python
# 来自 model.py:51-73 —— CodeGPT 的 CausalSelfAttention.forward
def forward(self, x):                                # x: (B, T, C) —— 一堆向量
    B, T, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    # ... reshape 成多头 ...
    y = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None,
        dropout_p=self.dropout if self.training else 0,
        is_causal=True,
    )                                                # 任意两个位置都能直接对话
    return self.resid_dropout(self.c_proj(y))
```

这段代码对 `x` 的假设**只有一条**：`x` 是一个形状为 `(B, T, C)` 的张量——"一堆可以互相关注的向量"。至于这些向量来自哪里（像素、音频帧、token、车道线、氨基酸），代码完全不关心。

**硬编码的先验越少，能装下的数据规律越多。** 在小数据时代，CNN 的平移等变先验能让它用 10 万张图击败没有先验的模型；但在 ImageNet-21K → LAION-5B 的量级，先验反而变成束缚——Transformer 用数据自己学会了"局部性"和"平移等变"，还学到了 CNN 学不到的全局关联。

这就是 [TRANSFORMER_FROM_VISION.md](TRANSFORMER_FROM_VISION.md) 的核心论断在跨模态上的推论：**谁的归纳偏置最少，谁就能在数据规模无限大时笑到最后。**

### 原因二：一切皆可 tokenize

Transformer 不在乎你的输入"本来是什么"。它只要求你给它一个形状为 `(B, T, C)` 的张量。于是每个领域都在做同一件事——**把自己的数据切成 token，然后投影到 `C` 维向量**。

```python
# 把任意模态变成 Transformer 能吃的输入 —— 统一配方

# ----- 语言/代码（本项目 model.py:183）-----
# 1 个 token = 1 个 BPE 子词
tok_emb = wte(idx)                      # (B, T, C)

# ----- 图像（ViT）-----
# 1 个 token = 一个 16×16 的图像 patch
patches = image.unfold(2, 16, 16).unfold(3, 16, 16)   # (B, 3, H/16, W/16, 16, 16)
tokens = patches.flatten(2).transpose(1, 2)           # (B, N_patches, 16*16*3)
tok_emb = nn.Linear(16*16*3, C)(tokens)               # (B, T, C)

# ----- 语音（Whisper）-----
# 1 个 token = 梅尔频谱的一个时间帧
mel = torchaudio.transforms.MelSpectrogram(n_mels=80)(waveform)  # (B, 80, T)
tokens = mel.transpose(1, 2)                                     # (B, T, 80)
tok_emb = nn.Linear(80, C)(tokens)                               # (B, T, C)

# ----- 自动驾驶（BEVFormer）-----
# 1 个 token = 鸟瞰图的一个网格单元 or 一条查询向量
bev_grid = project_cameras_to_bev(multi_view_images)   # (B, H_bev, W_bev, C)
tokens = bev_grid.flatten(1, 2)                        # (B, H_bev*W_bev, C)

# ----- 蛋白质（AlphaFold 2 / ESM）-----
# 1 个 token = 1 个氨基酸
tok_emb = wte(amino_acid_ids)            # (B, L_seq, C)
```

**配方永远是两步：(1) 切成序列；(2) 线性投影到同一个 `C` 维。** 之后所有领域共享完全相同的后续管线——堆 L 层 `Block`，每层都是 `Attention + MLP + 残差 + LayerNorm`，就是 `model.py:93-105` 的那十几行。

这就是为什么一个大学生能在一周内把 ViT 跑通——如果他之前实现过 GPT，主干代码几乎一行不用改。

### 原因三：与硬件的"共谋"——可大规模并行

RNN 的致命伤是时序依赖：`h_t = f(h_{t-1}, x_t)`，GPU 被迫按时间步串行计算。Transformer 把这条依赖切断——所有时间步的 attention 可以一次矩阵乘算完：

```python
# 伪代码对比

# RNN：T 个时间步必须串行
h = h_0
for t in range(T):
    h = rnn_cell(x[t], h)         # 第 t 步等第 t-1 步
    outputs.append(h)

# Transformer：一次矩阵乘搞定所有时间步
Q = x @ W_q                        # (B, T, C) @ (C, C) —— 一次算完所有 t
K = x @ W_k
V = x @ W_v
y = softmax(Q @ K.T / sqrt(d)) @ V # (B, T, T) @ (B, T, C) —— 依然一次算完
```

GPU 的 tensor core 是为大矩阵乘设计的。Transformer 的 FLOPs 大部分集中在 `nn.Linear` 上（`c_attn`、`c_proj`、`mlp.c_fc`、`mlp.c_proj`），这正是硬件最擅长的形状。结果：**同样的电费，Transformer 比 RNN 能多训几十倍数据。**

数据量起来之后，Scaling Law 接管——模型能力随参数、数据、算力按幂律平滑增长。而 RNN / LSTM 的 scaling 曲线在几亿参数就开始走平。

> 三个原因其实是一体两面的：**零先验 → 需要海量数据 → 必须并行化才喂得动 → 只有 Transformer 能做到这三件事同时成立。** 这就是为什么其他架构即使在小数据上偶尔赢一局，也没法真正威胁它。

---

## 3. 各领域 SOTA 地图：谁在用 Transformer，怎么用

下面按领域盘点，每条目的格式是：**模型名 / 发布年 / 作用 / Transformer 用在哪 / 关键变体**。

### 3.1 语言与代码（大模型）

| 模型 | 年份 | 作用 | Transformer 用法 |
|------|------|------|------------------|
| **GPT-4 / Claude 3.7 / Gemini 2** | 2023-2026 | 通用 LLM | Decoder-only Transformer，稀疏 MoE，数千亿参数 |
| **Llama 3 / DeepSeek-V3** | 2024-2025 | 开源 LLM | RoPE + GQA + SwiGLU，decoder-only |
| **Codex / CodeLlama / StarCoder 2 / Qwen-Coder** | 2021-2025 | 代码生成 | 和 LLM 同架构 + 代码语料 + FIM 训练 |
| **本项目 CodeGPT** | 2026 | 代码生成（教学） | `model.py`：decoder-only + FIM + 词表扩展 |

**代码怎么写** —— 本项目 `model.py:177-198` 就是这类模型的最小骨架。`idx → wte → 位置编码 → L 层 Block → ln_f → lm_head → softmax` 这条链路，在 GPT-4 里放大一万倍就是前沿 LLM。

### 3.2 图像识别

| 模型 | 年份 | 作用 | Transformer 用法 |
|------|------|------|------------------|
| **ViT** (Vision Transformer) | 2020 | 图像分类 | 把图切成 16×16 patch 当 token，送 encoder Transformer |
| **Swin Transformer** | 2021 | 检测/分割 | 层级 + 滑动窗口注意力，带回了一点 CNN 先验 |
| **CLIP** | 2021 | 图文对齐 | 一个视觉 Transformer + 一个语言 Transformer，对比学习 |
| **DINOv2** | 2023 | 自监督表征 | ViT + 自蒸馏，无需标签即可学到通用视觉特征 |
| **SAM (Segment Anything)** | 2023 | 通用分割 | ViT 图像编码器 + Prompt Transformer 解码器 |
| **EVA-02 / InternImage-Transformer** | 2023-2024 | 检测、分割新 SOTA | 更大 ViT + 更多预训练数据 |

**关键片段**：

```python
# ViT 编码器的核心 —— 对比 model.py:177-188，只有输入段不同
class ViT(nn.Module):
    def forward(self, image):                            # (B, 3, 224, 224)
        # === 把图像变 token，这是唯一特殊的一步 ===
        x = self.patch_embed(image)                      # (B, 14*14=196, C)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed                           # 学习式位置编码

        # === 下面和 GPT 的 forward 一模一样 ===
        for block in self.blocks:                        # L 层 Block，每层和 model.py:93-105 一样
            x = block(x)
        x = self.ln_f(x)
        return self.head(x[:, 0])                        # 用 [CLS] token 做分类
```

和本项目 `CodeGPT.forward` 的唯一区别：**CodeGPT 用 `wte(idx)` 把整数索引变 embedding，ViT 用 `Conv2d(3, C, 16, 16)` 把图像 patch 变 embedding。** 之后一模一样。

### 3.3 图像 / 视频生成

| 模型 | 年份 | 作用 | Transformer 用法 |
|------|------|------|------------------|
| **DALL·E 2 / Imagen** | 2022 | 文生图（UNet 时代） | 文本编码器是 Transformer，扩散主干还是 UNet |
| **Stable Diffusion 3 / FLUX** | 2024 | 文生图（DiT 时代） | **主干换成 DiT**——Diffusion Transformer，UNet 退场 |
| **Sora / Veo 2** | 2024-2025 | 文生视频 | 把视频切成 3D 时空 patch → 送 DiT 去噪 |
| **VideoPoet** | 2023 | 视频理解+生成 | 离散化视频为 token，decoder-only Transformer 统一做 |

**DiT 的关键代码对比**：

```python
# DiT block 相比 GPT block 只多一个东西：FiLM-style 调制
class DiTBlock(nn.Module):
    def forward(self, x, c):                             # c: 时间步 + 条件
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=1)
        # attention 部分（和 CodeGPT.CausalSelfAttention 一样，但不是 causal）
        x = x + gate * self.attn(modulate(self.ln1(x), shift, scale))
        # MLP 部分（和 model.py:76-90 完全一样）
        x = x + self.mlp(self.ln2(x))
        return x
```

视频就是 **把时间维也当一个空间维**——`(B, T_frames, H, W, 3) → (B, T*H*W/patch³, C)`，送进同一个 Transformer。Sora 之所以能生成 60 秒一致性视频，就是因为 attention 是**真正的全局**——它能让第 1 帧的像素和第 1200 帧的像素直接对话，这在 CNN/RNN 时代是做不到的。

### 3.4 语音识别与生成

| 模型 | 年份 | 作用 | Transformer 用法 |
|------|------|------|------------------|
| **Wav2Vec 2.0** | 2020 | 自监督语音表征 | Transformer encoder + 对比学习 |
| **Conformer** | 2020 | ASR SOTA | Transformer + 卷积混合（混血） |
| **Whisper** | 2022 | 通用 ASR + 翻译 | Encoder-Decoder Transformer，68 万小时弱监督 |
| **SeamlessM4T** | 2023 | 语音↔语音翻译 | 多模态 Transformer 统一做 |
| **VALL-E / AudioLM** | 2023 | TTS 零样本 | 把音频离散化为 codec token，decoder-only Transformer |
| **SoundStorm** | 2023 | 并行语音生成 | 非自回归 Transformer，大幅加速 TTS |

**Whisper 的骨架**：

```python
# Whisper encoder —— 几乎就是把 ViT 的 patch_embed 换成一维卷积
class WhisperEncoder(nn.Module):
    def forward(self, mel):                              # (B, 80, 3000) 梅尔谱
        x = F.gelu(self.conv1(mel))                      # stride=1
        x = F.gelu(self.conv2(x))                        # stride=2 下采样
        x = x.permute(0, 2, 1)                           # (B, 1500, C)
        x = x + self.positional_embedding                # 正弦位置编码
        for block in self.blocks:                        # 和 model.py:93-105 的 Block 相同
            x = block(x)
        return self.ln_post(x)
```

"语音是 token 序列" —— 这个视角让 HMM + GMM + 对齐算法几十年的工程积累，被 Transformer 在三年内全面接管。Whisper 论文里一句话总结：**只要数据足够多，Transformer 的 WER 单调下降，不需要任何领域特定的归纳偏置。**

### 3.5 自动驾驶

这是变化最剧烈的领域之一——2020 年前主流是 CNN 多任务网络（HydraNet）+ 规则系统；2024 年 Tesla FSD v12 发布时，Elon Musk 说了一句极具标志性的话："**我们删掉了 30 万行 C++ 代码，换成一个神经网络。**"

| 模型 | 年份 | 作用 | Transformer 用法 |
|------|------|------|------------------|
| **DETR** | 2020 | 目标检测 | Transformer 替代 anchor-based head，object queries |
| **BEVFormer** | 2022 | BEV 感知 | 时空 Transformer，多相机 → 鸟瞰图表征 |
| **UniAD** | 2023 | 端到端驾驶 | 感知 + 预测 + 规划全用 Transformer 统一 |
| **Tesla FSD v12** | 2024 | 端到端量产系统 | "Photon in, Control out"，像素直接到方向盘 |
| **Wayve LINGO / GAIA-1** | 2023-2024 | 世界模型 | 把驾驶建模成视频生成任务 |
| **Waymo EMMA / Foundation** | 2024-2025 | 基于 VLM 的驾驶 | 直接用多模态 Transformer 做规划 |

**BEVFormer 的核心思想**：

```python
# BEV Query —— Transformer 的"哪里该看"彻底变成可学习的
class BEVFormerLayer(nn.Module):
    def forward(self, bev_queries, multi_view_features, prev_bev):
        # 时间自注意力：当前 BEV 和历史 BEV 对话
        bev_queries = self.temporal_attn(bev_queries, prev_bev)
        # 空间交叉注意力：BEV 网格 → 多相机特征
        bev_queries = self.spatial_cross_attn(bev_queries, multi_view_features)
        # FFN（和 model.py:85-90 一样）
        return self.mlp(self.ln(bev_queries))
```

端到端驾驶的终极形态是：**把相机视频 + 导航指令当作 prompt，把方向盘角度 + 油门刹车当作 token 序列**，然后做自回归预测——这就是把 CodeGPT 里的 `idx → 下一个 token` 换成 `车载视频 → 下一个控制动作`。模型架构几乎不用变。

### 3.6 机器人

| 模型 | 年份 | 作用 | Transformer 用法 |
|------|------|------|------------------|
| **Gato** | 2022 | 通用智能体 | 一个 Transformer 做 604 种任务 |
| **RT-2** (Google) | 2023 | 视觉-语言-动作 | 把机器人动作 token 化，和图文一起训 |
| **PaLM-E** | 2023 | 具身多模态大模型 | 540B，视觉+语言+动作统一 |
| **Octo / π0 (Physical Intelligence)** | 2024-2025 | 开源机器人基础模型 | Diffusion Transformer 输出动作轨迹 |
| **RT-H / RDT** | 2024 | 层级指令跟随 | 语言 → 低级动作的自回归 Transformer |

**RT-2 的核心一招**：

```python
# 把机器人动作也当作 token
# 动作空间：(Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_open)
# 每个维度离散成 256 个 bin → 7 个 action token
#
# 训练数据：
#   输入：图像 + "pick up the red cup"
#   输出：<act_1234> <act_567> <act_89> ...  ← 和 CodeGPT 的代码 token 一视同仁
#
# 然后用和 model.py 完全一样的 F.cross_entropy(ignore_index=-1) 训练
```

这就把机器人控制这个"传统上需要 MPC + PID + 规则"的问题，彻底变成了一个"下一个 token 预测"问题。

### 3.7 科学计算

| 模型 | 年份 | 作用 | Transformer 用法 |
|------|------|------|------------------|
| **AlphaFold 2** | 2021 | 蛋白质结构预测 | Evoformer = 魔改 Transformer + 几何归纳偏置 |
| **AlphaFold 3** | 2024 | 蛋白-配体-核酸通用 | Diffusion Transformer |
| **ESM-2 / ESM-3** | 2022-2024 | 蛋白质语言模型 | decoder-only Transformer 训氨基酸序列 |
| **GraphCast / Aurora** | 2023-2024 | 天气预报 | Transformer 替代传统数值模拟，速度快 1000 倍 |
| **TimesFM / Chronos** | 2024 | 通用时间序列预测 | 把时序当 token，decoder-only |

**AlphaFold 2 的 Evoformer** 是一个很好的案例——它说明即使在最需要领域知识的科学问题上，Transformer 依然是主干；领域知识以"加在 attention 里的几何偏置"形式融入：

```python
# 简化版 Evoformer block
class EvoformerBlock(nn.Module):
    def forward(self, msa, pair):
        # MSA 方向：在多序列对齐上做 attention（行注意力 + 列注意力）
        msa = msa + self.row_attn(msa, bias=pair)        # pair → attention bias
        msa = msa + self.col_attn(msa)
        msa = msa + self.mlp(msa)
        # Pair 方向：把 MSA 的统计投回残基对表征
        pair = pair + self.outer_product_mean(msa)
        pair = pair + self.triangle_attn(pair)           # 三角形 attention（几何一致性）
        return msa, pair
```

注意 `attn(..., bias=pair)`——**领域知识不是替代 Transformer，而是写进 attention 的 bias 项。** 这是跨领域的通用模式：Transformer 是画布，领域知识是笔触。

---

## 4. 统一视图：所有 SOTA 都长成 `CodeGPT.forward` 的样子

把上面所有模型的主干精简到本质，你会发现它们都写成同一个模板：

```python
# 通用 Transformer 主干 —— 对照 model.py:177-198
def forward(self, raw_input, *conditioning):
    # === 步骤 1：把输入变成 (B, T, C) 的 token 序列 ===
    #   - 代码：wte(idx)               [本项目]
    #   - 图像：patch_embed(image)     [ViT]
    #   - 语音：conv_subsample(mel)    [Whisper]
    #   - 驾驶：bev_projection(cams)   [BEVFormer]
    #   - 动作：action_tokenizer(traj) [RT-2]
    #   - 蛋白质：wte(amino_acids)     [ESM]
    x = tokenize_any_modality(raw_input)

    # === 步骤 2：加位置信息 ===
    x = x + pos_encoding              # 可学习 / 正弦 / RoPE / ALiBi 不重要

    # === 步骤 3：L 层 Block —— 所有领域完全一致 ===
    for block in self.blocks:         # 就是 model.py:93-105 的 Block
        x = block(x)                  #   attn + mlp + 残差 + LayerNorm

    # === 步骤 4：任务头 ===
    x = self.ln_f(x)
    return task_head(x)               # lm_head / cls_head / regression_head
```

**变的只是"步骤 1"和"步骤 4"。中间 90% 的参数、90% 的 FLOPs、90% 的代码完全一样。**

这就是 [TRANSFORMER_FROM_VISION.md](TRANSFORMER_FROM_VISION.md) 末尾"柏拉图表征假说"在工程上的映射：不同模态的模型在表征层面收敛，是因为它们**在架构层面已经是同一个架构**——只是 embedding 入口和 head 出口不同。

---

## 5. 为什么会收敛到同一架构？工程、经济、理论三重解释

### 工程解释：CUDA kernel 的复用

当 Flash Attention 2 发布时，所有领域的模型一起加速 2-4 倍。当 `torch.compile` 成熟时，所有领域一起受益。当 H100 / H200 推出时，所有领域一起 scaling。

**这是一个正反馈：**
- 使用 Transformer 的人越多 → 投入到 kernel / 硬件 / 编译器的优化越多
- 优化越多 → Transformer 跑得比其他架构快 10×-100×
- 跑得快 → 大家更愿意用 Transformer → 回到起点

CNN 也曾是受益者（cuDNN 的卷积优化），但卷积对硬件的压力没有 attention 大，优化收益边际递减。Transformer 正好踩在了 GPU 架构的甜蜜点。

### 经济解释：一个团队、一套基础设施

一家公司如果语音团队用 Conformer、视觉团队用 CNN、NLP 团队用 BERT、驾驶团队用 HydraNet，就要维护**四套完全不同**的训练框架、部署管线、性能优化、数据流水线。

换成 Transformer 之后：

- 一套 FSDP / DeepSpeed 训练框架
- 一套 vLLM / TensorRT-LLM 推理框架
- 一套 tokenization / 数据分片协议
- 一套 checkpoint / 加载 / 恢复逻辑（参考 `train.py` 里 `_orig_mod.` 前缀的处理，现在是全行业标准）

**工程成本砍掉 75%。** 这是 Transformer 胜利的非技术原因，但它和技术原因一样重要。

### 理论解释：通用近似 + 可 scaling

Transformer 已被证明是序列到序列的通用函数近似器（Yun et al., 2020），这一点 RNN 也有。但 Transformer 的特殊在于：**它的近似能力随模型规模的增长是平滑、可预测的**——这就是 Scaling Law（Kaplan et al., 2020；Hoffmann et al., 2022）。

一个你能提前算出要花多少钱、能达到多少性能的架构，在工业界是无敌的。没人想投 1000 万美元训一个不知道能不能比 baseline 强 2% 的架构。

---

## 6. 诚实声明：Transformer 还没完全吃掉的地方

科学 always 允许反例存在，以下是 2026 年仍未被 Transformer 完全统治的领域：

| 领域 | 为什么 Transformer 还没赢 | 当前 SOTA |
|------|--------------------------|-----------|
| **极长上下文**（百万 token 以上） | `O(T²)` 的 attention 成本 | Mamba / S4 / RWKV 等 SSM；或 hybrid（e.g. Jamba） |
| **实时边缘推理** | 参数量大、KV cache 显存高 | 蒸馏小模型 + CNN 混合 |
| **精确物理模拟** | 需要严格几何/物理约束 | PINN / Graph Networks（但正在被吸收进 Transformer） |
| **强化学习（特定领域）** | 样本效率仍不如专门算法 | MuZero / Dreamer |

但注意：**这些反例中的大多数，架构里依然有 attention**。Mamba 的作者自己承认"Mamba + attention 混合效果最好"。Jamba 就是这么做的。所以更准确的说法是：**Transformer 不是"唯一的架构"，而是"所有架构里都不能没有的基础构件"。**

---

## 7. 总结：一个架构，一条路径，一个未来

回到本项目。如果你理解了 `model.py` 里不到 300 行的 CodeGPT，你就已经理解了：

- **GPT-4 如何写代码**——它是 CodeGPT 放大一万倍，加上 RLHF（见 [RLHF_AND_PLATONIC_REPRESENTATION.md](RLHF_AND_PLATONIC_REPRESENTATION.md)）
- **ViT 如何识别图像**——它是 CodeGPT 去掉 causal mask，把 `wte` 换成 patch 卷积
- **Whisper 如何听懂英语**——它是 CodeGPT 把 `wte` 换成梅尔谱 + 1D conv
- **Tesla FSD 如何开车**——它是多相机 BEV 版的 CodeGPT，输出从 token 变成方向盘
- **AlphaFold 如何算蛋白质**——它是在 attention 里加了几何 bias 的 CodeGPT
- **RT-2 如何指挥机器人**——它是把动作也当 token 的 CodeGPT

| 视角 | 统一结论 |
|------|----------|
| 架构 | 全是 embedding + L 层 Block + head |
| 训练 | 全是 `F.cross_entropy(ignore_index=-1)` 或其扩散/对比变体 |
| 部署 | 全是 KV cache + beam search / top-p（见 `model.py:279` 附近） |
| 扩展 | 全吃 Scaling Law —— 数据、参数、算力三轴 |

> **Transformer 不是一个模型，而是一个通用的"表征学习机器"。** 它让"学习一种模态"变成了"准备该模态的 tokenizer + head"——学会一种模态之后，其他所有模态的代价几乎只剩数据。
>
> 这是深度学习六十年来第一次，**不同领域的研究者用同一份代码说同一种语言**。如果你问下一个十年的 AI 会长什么样——答案大概率是：**同一个 Transformer，吃更多模态的 token，在更大的集群上训更久。**
>
> 然后，涌现会继续发生。

---

## 延伸阅读

- [从图像视角理解 Transformer](TRANSFORMER_FROM_VISION.md) —— 为什么 ViT 和 GPT 是同一种东西
- [从 RNN 到 CodeGPT：序列建模的进化史](DEEP_DIVE.md) —— 为什么 RNN 输给了 Transformer
- [深度学习是可微分编程](DIFFERENTIABLE_PROGRAMMING.md) —— 为什么一切都能被梯度下降"编译"
- [RLHF 对齐与柏拉图表征](RLHF_AND_PLATONIC_REPRESENTATION.md) —— 为什么不同模态的模型表征会收敛
