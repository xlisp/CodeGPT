# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CodeGPT is a GPT model specialized for code generation, extended from nanoGPT with Fill-in-the-Middle (FIM) support, code-aware special tokens, and multi-language awareness.

The `docs/` directory is a **大模型科普扫盲** (large-model literacy) resource — essays in Chinese that explain GPT / Transformer / RLHF / representation theory and tie each concept back to the code in `model.py`, `train.py`, `tokenizer.py`. README.md contains the index of these essays; when writing or editing anything in `docs/`, follow these principles:

- **Prefer a few lines of key Python / PyTorch code over mathematical formulas.** Show the concept via `nn.Linear`, `F.softmax`, `torch.tril`, `F.cross_entropy`, etc. — the shortest runnable snippet that makes the idea click.
- When a formula is unavoidable, pair it with the equivalent code excerpt (ideally a real line from this repo with a `model.py:NN` reference).
- Keep examples grounded in PyTorch and standard libraries already used in the project (`torch`, `torch.nn`, `tiktoken`, `numpy`). Do not introduce new dependencies just to illustrate a point.
- Treat the docs as authoritative for the "why" behind design choices — code comments stay terse; rationale lives in `docs/`.

## Common Commands

```bash
# Data preparation (writes train.bin/val.bin/meta.pkl into data/<dataset>/)
python data/python_code/prepare.py --source=local --code_dir=/path/to/python/projects
python data/python_code/prepare.py --source=huggingface --max_samples=100000
python data/github_code/prepare.py --langs python javascript typescript --max_samples=20000

# Training
python train.py config/train_codegpt_small.py            # ~10M params, CPU/single-GPU friendly
python train.py config/train_codegpt.py                  # 124M, single GPU
torchrun --standalone --nproc_per_node=4 train.py config/train_codegpt.py   # DDP
python train.py config/finetune_codegpt.py               # init from GPT-2 weights
python train.py config/train_codegpt_small.py --batch_size=32 --max_iters=5000   # CLI overrides

# Generation / inference
python sample.py --prompt="def fibonacci(n):"
python sample.py --mode=fim --prefix="def add(a, b):" --suffix="    return result"
python sample.py --mode=interactive
python repl.py --out_dir=out-codegpt --temperature=0.3

# Benchmark
python bench.py --n_layer=6 --n_embd=384 --batch_size=16
```

There is no test suite, linter, or build step — this is a research/training repo. "Running" means training or sampling.

## Configuration System (`configurator.py`)

All entry points (`train.py`, `sample.py`, `bench.py`) use a single convention:

1. Default config variables are defined as **module-level globals** at the top of the script.
2. A call to `configure()` then (a) execs any `*.py` positional arg as a config file whose top-level variables overwrite those globals, and (b) applies `--key=value` CLI overrides, coercing types from the existing default's type.
3. Unknown `--key` raises. Config files cannot introduce new keys beyond what the script already defines.

When adding a new tunable, declare it as a default global in the script that uses it, not just in a config file — otherwise `configure()` will reject `--key=` overrides.

## Architecture

**Single-file model (`model.py`)**. Standard decoder-only Transformer (LayerNorm, CausalSelfAttention with flash-attn fallback, MLP, Block, CodeGPT). Pre-norm, weight-tied `lm_head` ↔ `wte`, learned positional embeddings. `CodeGPTConfig` is a dataclass carrying both architecture params and FIM/special-token IDs. Two notable extensions beyond nanoGPT:

- `expand_vocab(new_size)` — used when loading GPT-2 weights (vocab 50257) and then growing to 50304 to fit the code/FIM/language special tokens. Preserves weight tying.
- `generate()` supports `top_p`, `repetition_penalty`, and `stop_tokens` in addition to `top_k`/temperature.

**Tokenizer (`tokenizer.py`)**. Wraps tiktoken GPT-2 BPE. Adds special tokens at IDs 50256–50278 (endoftext, 4 FIM tokens, code_start/end, 16 lang tokens). `VOCAB_SIZE = 50304` (padded multiple of 64). The IDs in `SPECIAL_TOKENS` must stay in sync with the defaults in `CodeGPTConfig` — they are referenced by numeric value in both the model config and `get_batch` padding logic.

**FIM data augmentation (`apply_fim_transform`)**. Called from `train.py:get_batch` with 50% probability per-sample on the training split only. Splits a token sequence at two random boundaries into prefix/middle/suffix, then re-emits either PSM (`<|fim_prefix|> P <|fim_suffix|> S <|fim_middle|> M`) or SPM form. The transformed sequence is padded/truncated to `block_size`, and `<|fim_pad|>` positions are set to `-1` in the targets so they are ignored by `F.cross_entropy(ignore_index=-1)`. This is the mechanism that lets inference-time `encode_fim()` prompts work.

**Training loop (`train.py`)**. Memmaps `data/<dataset>/{train,val}.bin` (uint16 token IDs). Supports three `init_from` modes: `scratch`, `resume` (loads `out_dir/ckpt.pt`, strips `_orig_mod.` prefix left by `torch.compile`), and `gpt2*` (loads HF weights via `CodeGPT.from_pretrained`, then `expand_vocab` to 50304, optional `crop_block_size`). DDP auto-activates when `RANK` is in env; `gradient_accumulation_steps` is divided by world size. Mixed precision uses `GradScaler` only when `dtype=='float16'`. Cosine LR schedule with linear warmup. Checkpoints saved to `out_dir/ckpt.pt` (gitignored) including `model_args` needed to rebuild config at load time.

**Inference paths**. `sample.py` is the flag-driven single-shot/FIM/interactive entry; `repl.py` is a richer color-terminal REPL with in-session `/commands`. Both load checkpoints the same way: rebuild `CodeGPTConfig(**checkpoint['model_args'])`, strip `_orig_mod.` prefix, then load state dict.

## Data Layout Contract

Anything in `data/<dataset>/` that trainers/samplers consume:

- `train.bin`, `val.bin` — flat uint16 token streams, memmapped directly (no header).
- `meta.pkl` — optional; if present, `vocab_size` overrides the default when initializing from scratch.

`prepare.py` scripts in each dataset dir are responsible for producing these. When adding a new dataset, keep this contract.

## Conventions Worth Knowing

- Default `out_dir` is `out-codegpt` (gitignored). Small-model config writes to `out-codegpt-small`, finetune to `out-codegpt-finetune`.
- `torch.compile` is enabled by default on CUDA only; it wraps the model with an `_orig_mod.` prefix that the checkpoint-loading code in `train.py`/`sample.py`/`repl.py` must strip.
- The `bias=False` default (Pre-LN, no bias in Linear/LayerNorm) is inherited from nanoGPT and is the expected path; GPT-2 finetuning flips it to `True` in `from_pretrained`.
- Special-token IDs are hard-coded numbers in multiple places. If you shift the vocab layout, update `SPECIAL_TOKENS` (tokenizer.py), the `*_id` defaults in `CodeGPTConfig`, and any saved checkpoints become incompatible.
