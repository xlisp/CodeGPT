"""
Interactive REPL for the autoresearch-trained model.

Loads the checkpoint saved by train_autoresearch.py and lets you
type prompts (text or code) and see what the model generates.

Usage:
    ~/miniconda3/envs/codegpt/bin/python repl_autoresearch.py
    ~/miniconda3/envs/codegpt/bin/python repl_autoresearch.py --ckpt out-autoresearch/ckpt.pt

Commands inside the REPL:
    /temp <float>     set sampling temperature  (default 0.8)
    /topk <int>       set top-k                 (default 40)
    /topp <float>     set top-p (nucleus)       (default 0.95)
    /tokens <int>     set max new tokens        (default 200)
    /greedy           switch to greedy decoding
    /sample           switch back to sampling
    /info             show model info
    /quit  or  Ctrl-D exit
"""

import os
import sys
import math
import argparse

import torch
import torch.nn.functional as F

# ── autoresearch imports ──────────────────────────────────────────────────────
sys.path.insert(0, '/home/xlisp/PyPro/autoresearch')
from prepare import Tokenizer

# ── model definition (same as train_autoresearch.py) ─────────────────────────
# (copied here so this file is self-contained and doesn't re-run training)

from dataclasses import dataclass

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size:   int = 32768
    n_layer:      int = 12
    n_head:       int = 6
    n_kv_head:    int = 6
    n_embd:       int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


def sdpa_with_window(q, k, v, window_size):
    B, T, n_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    if n_kv_head < n_head:
        groups = n_head // n_kv_head
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    w = window_size[0]
    if w <= 0 or w >= T:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        idx = torch.arange(T, device=q.device)
        diff = idx.unsqueeze(0) - idx.unsqueeze(1)
        mask = (diff >= 0) & (diff < w)
        attn_bias = torch.zeros(T, T, dtype=q.dtype, device=q.device)
        attn_bias.masked_fill_(~mask, float('-inf'))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
    return out.transpose(1, 2).contiguous()


import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head    = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim  = config.n_embd // config.n_head
        self.c_q    = nn.Linear(config.n_embd, self.n_head    * self.head_dim, bias=False)
        self.c_k    = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v    = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) \
                       if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head,    self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        y = sdpa_with_window(q, k, v, window_size)
        return self.c_proj(y.view(B, T, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp  = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h":   nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head      = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas    = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim   = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = next(self.parameters()).device
        ch = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (ch / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().half()[None, :, None, :]
        sin = freqs.sin().half()[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        L, S = config.sequence_len, config.sequence_len // 2
        char_to_window = {"L": (L, 0), "S": (S, 0)}
        ws = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        ws[-1] = (L, 0)
        return ws

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x  = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x  = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x  = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        softcap = 15
        logits  = softcap * torch.tanh(self.lm_head(x).float() / softcap)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
        return logits


# ── generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, tokenizer, prompt: str, *,
             max_new_tokens: int = 200,
             temperature: float = 0.8,
             top_k: int = 40,
             top_p: float = 0.95,
             greedy: bool = False,
             device: str = "cuda") -> str:
    """Autoregressive text generation."""
    bos = tokenizer.get_bos_token_id()
    prompt_ids = tokenizer.encode(prompt) if prompt.strip() else []
    input_ids  = [bos] + prompt_ids

    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated = []

    model.eval()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)

    for _ in range(max_new_tokens):
        # crop to model's context window
        x_cond = x[:, -model.config.sequence_len:]

        with autocast_ctx:
            logits = model(x_cond)          # (1, T, vocab)
        logits = logits[:, -1, :]           # last position → (1, vocab)

        if greedy:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-6)

            # top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # top-p (nucleus)
            if 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # remove tokens with cumulative prob above threshold
                remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float('-inf')
                logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)

            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        generated.append(next_id.item())
        x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(generated)


# ── checkpoint loading ────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str = "cuda"):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = ckpt["config"]
    print(f"  config:     n_layer={config.n_layer}, n_embd={config.n_embd}, "
          f"vocab={config.vocab_size}, seq_len={config.sequence_len}")
    print(f"  val_bpb:    {ckpt.get('val_bpb', 'n/a'):.4f}")
    print(f"  steps:      {ckpt.get('step', '?')}")
    print(f"  tokens:     {ckpt.get('total_tokens', 0) / 1e6:.2f}M")

    model = GPT(config)
    # strip possible compile prefix
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # re-register rotary buffers on the correct device
    head_dim = config.n_embd // config.n_head
    cos, sin = model._precompute_rotary(config.sequence_len * 10, head_dim, device=device)
    model.cos, model.sin = cos, sin

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params:     {n_params / 1e6:.1f}M")
    return model


# ── REPL ──────────────────────────────────────────────────────────────────────

HELP = """\
Commands:
  /temp  <float>   sampling temperature    (current: {temperature})
  /topk  <int>     top-k                   (current: {top_k})
  /topp  <float>   top-p nucleus           (current: {top_p})
  /tokens <int>    max new tokens          (current: {max_new_tokens})
  /greedy          switch to greedy decode
  /sample          switch to sampling
  /info            model & settings info
  /help            show this message
  /quit            exit  (also Ctrl-D)
"""

def repl(model, tokenizer, device: str):
    # defaults
    temperature    = 0.8
    top_k          = 40
    top_p          = 0.95
    max_new_tokens = 200
    greedy         = False

    model_info = (f"n_layer={model.config.n_layer}  n_embd={model.config.n_embd}  "
                  f"vocab={model.config.vocab_size}  seq_len={model.config.sequence_len}")

    print("\n" + "="*60)
    print(" autoresearch model REPL")
    print("="*60)
    print(f" Model:  {model_info}")
    print(f" Device: {device}")
    print(" Type a prompt and press Enter. /help for commands.")
    print("="*60 + "\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue

        # ── commands ──────────────────────────────────────────────────────────
        if prompt.startswith("/"):
            parts = prompt.split()
            cmd   = parts[0].lower()

            if cmd == "/quit":
                print("Bye!")
                break

            elif cmd == "/help":
                print(HELP.format(temperature=temperature, top_k=top_k,
                                  top_p=top_p, max_new_tokens=max_new_tokens))

            elif cmd == "/info":
                mode = "greedy" if greedy else f"sample (temp={temperature}, top_k={top_k}, top_p={top_p})"
                print(f"  Model:      {model_info}")
                print(f"  Mode:       {mode}")
                print(f"  Max tokens: {max_new_tokens}")

            elif cmd == "/temp" and len(parts) == 2:
                temperature = float(parts[1])
                print(f"  temperature = {temperature}")

            elif cmd == "/topk" and len(parts) == 2:
                top_k = int(parts[1])
                print(f"  top_k = {top_k}")

            elif cmd == "/topp" and len(parts) == 2:
                top_p = float(parts[1])
                print(f"  top_p = {top_p}")

            elif cmd == "/tokens" and len(parts) == 2:
                max_new_tokens = int(parts[1])
                print(f"  max_new_tokens = {max_new_tokens}")

            elif cmd == "/greedy":
                greedy = True
                print("  Switched to greedy decoding.")

            elif cmd == "/sample":
                greedy = False
                print(f"  Switched to sampling (temp={temperature}).")

            else:
                print(f"  Unknown command: {cmd}  (try /help)")

            continue

        # ── generation ────────────────────────────────────────────────────────
        print()
        try:
            output = generate(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
                device=device,
            )
            # print prompt + generated text clearly
            print(f"\033[90m{prompt}\033[0m", end="")   # dim prompt
            print(output)
        except Exception as e:
            print(f"  Error during generation: {e}")
        print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="REPL for autoresearch model")
    parser.add_argument("--ckpt",   default="out-autoresearch/ckpt.pt",
                        help="Path to checkpoint (.pt)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", default=None,
                        help="Run a single prompt and exit (non-interactive)")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temp",   type=float, default=0.8)
    parser.add_argument("--topk",   type=int, default=40)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        print("Run training first:")
        print("  ~/miniconda3/envs/codegpt/bin/python -W ignore train_autoresearch.py")
        sys.exit(1)

    print(f"Device: {args.device}")
    tokenizer = Tokenizer.from_directory()
    model     = load_model(args.ckpt, device=args.device)

    if args.prompt is not None:
        # single-shot mode
        out = generate(model, tokenizer, args.prompt,
                       max_new_tokens=args.tokens, temperature=args.temp,
                       top_k=args.topk, greedy=args.greedy, device=args.device)
        print(args.prompt + out)
    else:
        repl(model, tokenizer, args.device)


if __name__ == "__main__":
    main()
