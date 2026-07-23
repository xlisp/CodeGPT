"""
CodeGPT Semantic-Subspace Visualizer.

Prints, for a single prompt, the whole chain that turns dot products into an answer:

  1. embedding dot products   —— wte 里 token 之间的余弦相似度
  2. head subspaces           —— 每个 head 的 W_q W_k^T 秩 = 64，即 768 维里的一个子空间
  3. attention maps           —— 每个 head 在自己子空间里的 q·k 检索结果
  4. logit lens               —— 每一层残差流投影到 vocab，看"答案"如何逐层成形
  5. final distribution       —— 最后一次点积 x·wte[v] 得到的 top-k 与熵

Usage:
    python visualize_semantics.py --init_from=scratch --prompt="def add(a, b):"
    python visualize_semantics.py --out_dir=out-codegpt-small --prompt="def add(a, b):"
    python visualize_semantics.py --init_from=gpt2 --prompt="def add(a, b):" --layer=6
"""

import os
import math

import torch
from torch.nn import functional as F

from model import CodeGPT, CodeGPTConfig
from tokenizer import CodeTokenizer, SPECIAL_TOKENS

# ---------- config ----------
out_dir = 'out-codegpt'
init_from = 'resume'   # 'resume' | 'gpt2' | 'gpt2-medium' | ... | 'scratch'
prompt = 'def add(a, b):\n    return a'
lang = 'python'
layer = -1             # which layer's attention maps to draw (-1 = last)
n_heads_show = 4       # how many heads of that layer to draw
top_k_show = 5         # top-k tokens per logit-lens row
n_embd = 384           # only used by init_from='scratch'
n_layer = 6            # only used by init_from='scratch'
n_head = 6             # only used by init_from='scratch'
seed = 1337
device = 'cpu'

from configurator import configure
configure()

torch.manual_seed(seed)

# ---------- load model ----------
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CodeGPT(CodeGPTConfig(**checkpoint['model_args']))
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = CodeGPT.from_pretrained(init_from)
elif init_from == 'scratch':
    print("!! init_from=scratch: weights are random, the numbers below show the "
          "plumbing, not real semantics.")
    model = CodeGPT(CodeGPTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                                  block_size=256, dropout=0.0))
else:
    raise ValueError(f"Unknown init_from: {init_from}")

model.eval()
model.to(device)

tokenizer = CodeTokenizer()
cfg = model.config
layer_idx = layer if layer >= 0 else cfg.n_layer + layer

# ---------- encode ----------
ids = [SPECIAL_TOKENS["<|code_start|>"], SPECIAL_TOKENS[f"<|lang:{lang}|>"]]
ids += tokenizer.encode_raw(prompt)
idx = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
T = idx.size(1)


def label(t):
    """Short printable label for one token id."""
    if t in tokenizer.special_tokens_reverse:
        return tokenizer.special_tokens_reverse[t].strip('<|>')[:8]
    s = tokenizer.decode_raw([t])
    return repr(s)[1:-1].replace(' ', '·')[:8] or '?'


labels = [label(t) for t in ids]

RAMP = ' .:-=+*#%@'


def heat(v, lo=0.0, hi=1.0):
    """Map a scalar into a 10-level ASCII ramp."""
    if not math.isfinite(v):
        return ' '
    z = (v - lo) / (hi - lo + 1e-9)
    return RAMP[max(0, min(len(RAMP) - 1, int(z * (len(RAMP) - 1))))]


def rule(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------- 1. embedding dot products ----------
rule("1. 词嵌入的点积：语义空间里最原始的相似度")

E = model.transformer.wte.weight[idx[0]]                  # (T, n_embd)
En = E / E.norm(dim=-1, keepdim=True)
cos = En @ En.t()                                          # 纯点积 = 余弦相似度

w = max(len(l) for l in labels) + 1
print(' ' * w + ''.join(f'{i:>3}' for i in range(T)))
for i in range(T):
    row = ''.join(f'  {heat(cos[i, j].item(), -0.2, 1.0)}' for j in range(T))
    print(f'{labels[i]:>{w}}{row}')
print(f"\n  ramp: '{RAMP}'  (低 → 高相似)")

# ---------- 2. head subspaces ----------
rule("2. 参数子空间：c_attn 把 n_embd 切成 n_head 个低秩子空间")

hd = cfg.n_embd // cfg.n_head
Wattn = model.transformer.h[layer_idx].attn.c_attn.weight  # (3*n_embd, n_embd)
Wq, Wk, Wv = Wattn.split(cfg.n_embd, dim=0)
print(f"layer {layer_idx}: n_embd={cfg.n_embd}, n_head={cfg.n_head}, head_dim={hd}")
print(f"  c_attn.weight   {tuple(Wattn.shape)}   -> Wq/Wk/Wv each {tuple(Wq.shape)}")
print("\n  head |  rank(Wq_h Wk_h^T)  |  ||Wq_h Wk_h^T||_F  |  该 head 关心的子空间维数")
for h in range(cfg.n_head):
    Wq_h = Wq[h * hd:(h + 1) * hd]                         # (hd, n_embd)
    Wk_h = Wk[h * hd:(h + 1) * hd]
    M = Wq_h.t() @ Wk_h                                    # (n_embd, n_embd) 双线性度量
    r = torch.linalg.matrix_rank(M.float()).item()
    print(f"  {h:>4} |  {r:>17} |  {M.norm().item():>17.2f} |  {hd} / {cfg.n_embd}")
print(f"\n  attention score(i,j) = x_i^T (Wq_h^T Wk_h) x_j"
      f"  —— 秩 ≤ {hd}，所以每个 head 只在 {hd} 维子空间里比相似度")

# ---------- 3. attention maps ----------
rule(f"3. 点积检索：layer {layer_idx} 各 head 的注意力矩阵")

x = model.transformer.drop(
    model.transformer.wte(idx) + model.transformer.wpe(torch.arange(T, device=device)))
resid = []                                                 # 每层输出的残差流
with torch.no_grad():
    for li, block in enumerate(model.transformer.h):
        if li == layer_idx:
            xn = block.ln_1(x)
            q, k, v = block.attn.c_attn(xn).split(cfg.n_embd, dim=2)
            q = q.view(1, T, cfg.n_head, hd).transpose(1, 2)
            k = k.view(1, T, cfg.n_head, hd).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)[0]                 # (n_head, T, T)
        x = block(x)
        resid.append(x.clone())

for h in range(min(n_heads_show, cfg.n_head)):
    print(f"\n  head {h}:")
    print(' ' * (w + 2) + ''.join(f'{i:>3}' for i in range(T)))
    for i in range(T):
        row = ''.join(f'  {heat(att[h, i, j].item(), 0.0, 1.0)}' for j in range(T))
        print(f'  {labels[i]:>{w}}{row}')

# ---------- 4. logit lens ----------
rule("4. Logit Lens：把每一层的残差流直接投到 vocab，看答案怎么长出来")

print("  层 |   熵   | 该层残差流解码出的 top-k")
with torch.no_grad():
    for li, r in enumerate(resid):
        hlast = model.transformer.ln_f(r[:, -1, :])        # 借用最终 LayerNorm
        lg = model.lm_head(hlast)[0]                       # 最后一次点积 x · wte[v]
        p = F.softmax(lg, dim=-1)
        ent = -(p * (p + 1e-12).log()).sum().item()
        top = torch.topk(p, top_k_show)
        cells = ' '.join(f"{label(t.item())}({pv:.2f})"
                         for t, pv in zip(top.indices, top.values))
        print(f"  {li:>2} | {ent:6.2f} | {cells}")

# ---------- 5. final distribution ----------
rule("5. 最终输出：generate() 拿到的那个 vocab 维向量")

with torch.no_grad():
    logits, _ = model(idx)
p = F.softmax(logits[0, -1], dim=-1)
ent = -(p * (p + 1e-12).log()).sum().item()
top = torch.topk(p, 10)
print(f"  logits shape = {tuple(logits.shape)}  (最后一维 = vocab_size = {cfg.vocab_size})")
print(f"  熵 = {ent:.3f} nats   (log(vocab) = {math.log(cfg.vocab_size):.2f} 为完全均匀)")
print("\n  rank |   p    | token")
for i, (t, pv) in enumerate(zip(top.indices, top.values)):
    bar = '#' * int(pv.item() * 40)
    print(f"  {i:>4} | {pv.item():.4f} | {label(t.item()):<10} {bar}")
print("\n  ↑ RL / SFT 训练改变的就是这一列 p —— 协议、采样参数都在这之后才起作用。")
