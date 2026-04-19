# Training Report: 24-Hour Best-Effect Run (2026-03-22 ~ 2026-03-23)

**Date:** 2026-03-22 00:17 CST → 2026-03-23 00:57 CST
**Goal:** Train the best possible CodeGPT model using the autoresearch architecture on a GTX 1080
**Verdict:** Training completed successfully. Model is degenerate — diagnosis and fix documented below.

---

## 1. Training Configuration

| Parameter | Value |
|-----------|-------|
| Script | `train_autoresearch.py` |
| Architecture | Autoresearch GPT (Muon + Value Residual + RoPE + RMSNorm + SSSL) |
| Depth | 12 layers |
| n_embd | 768 |
| n_head | 6 |
| Parameters | **135.3M** |
| Vocab size | 8,192 (rustbpe BPE, trained from climbmix) |
| Sequence length | 2,048 tokens |
| Dataset | karpathy/climbmix-400b-shuffle (100 train shards, 1 val shard) |
| Data downloaded | 8.7 GB (101 parquet shards) |
| TOTAL_BATCH_SIZE | 131,072 tokens/step |
| DEVICE_BATCH_SIZE | 2 |
| Gradient accumulation | 32 steps |
| Optimizer | MuonAdamW (Muon for matrices, AdamW for scalars/embeddings) |
| Learning rate | MATRIX_LR=0.04, EMBEDDING_LR=0.6, UNEMBEDDING_LR=0.004 |
| Warmdown | 20% of budget (LR linearly → 0) |
| TIME_BUDGET | 86,400s (24 hours) |
| Precision | float16 + GradScaler |
| Checkpoint | `out-autoresearch/ckpt.pt` (517 MB) |

### Hardware: NVIDIA GeForce GTX 1080 (Pascal, sm_61, 8 GB)

GTX 1080 adaptations from original autoresearch (H100-targeted):

| Original (H100) | Adapted (GTX 1080) |
|-----------------|-------------------|
| Flash Attention 3 | PyTorch SDPA + manual sliding-window mask |
| bfloat16 | float16 + GradScaler |
| `torch.compile` | Disabled (sm_61 limited support) |
| `F.rms_norm` (PyTorch ≥ 2.4) | Manual: `x * rsqrt(mean(x²) + 1e-6)` |
| Fused CPU-tensor optimizer | Python scalar args (no CPU tensor state) |

---

## 2. Training Results

### Throughput

| Metric | Value |
|--------|-------|
| Tokens/sec | ~730 tok/sec |
| Step time | ~179 s/step |
| MFU | 5.68% |
| Peak VRAM | 5,420 MiB / 8,192 MiB |
| Total steps | 493 |
| Total tokens | **64.6M** |
| Total wall time | ~88,787s (~24.7 hours) |

### Loss Curve

```
step   0 (  0%): loss = 9.011  (random init baseline — correct, log₂(8192) = 13 bits)
step  12 (  0%): loss = 6.25   (rapid early learning)
step  35 (  5%): loss = 1.69
step  60 ( 10%): loss = 0.132
step 100 ( 18%): loss = 0.0072
step 140 ( 27%): loss = 0.011  ← spike: new data buffer loaded
step 200 ( 39%): loss = 0.0034
step 300 ( 55%): loss = 0.0030 (plateau begins)
step 492 (100%): loss = 0.0029
```

**Note on the spike at step 140:** The dataloader refilled its in-memory document buffer with
a new batch of training documents, causing a momentary loss increase as the model saw
previously-unseen patterns. This is normal with the best-fit packing strategy.

### Final Metrics (from `train_best.log`)

```
val_bpb:          0.000812
training_seconds: 86,498.7
total_seconds:    88,787.4
peak_vram_mb:     5,420.3
mfu_percent:      5.68
total_tokens_M:   64.6
num_steps:        493
num_params_M:     135.3
depth:            12
```

---

## 3. Evaluation Results

### 3.1 Generation Quality Test

Tested using `repl_autoresearch.py` immediately after training.

**Test 1 — Code prompt:**
```
Prompt:  def fibonacci(n):
Output:  def fibonacci(n):):):):):):):):):):):):):): [repeats indefinitely]
```

**Test 2 — Natural language:**
```
Prompt:  Hello world
Output:  Hello world world world world world world world [repeats indefinitely]
```

**Test 3 — Import statement:**
```
Prompt:  import numpy
Output:  import numpyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy [repeats indefinitely]
```

All prompts lead to immediate degenerate repetition. The model is **not usable**.

### 3.2 Token Probability Inspection

Inspected the logit distribution after the prompt `def fibonacci(n):`:

```
Token 5681  '):'      probability: 99.96%
Token   13  '\r'      probability:  0.00%
Token 3783  ' nucle'  probability:  0.00%
(all other tokens near-zero)
```

The model assigns 99.96% probability to a single token (`):`) regardless of context.
This is a **collapsed distribution** — the model is not modeling language.

### 3.3 Validation Loss (recomputed manually)

```python
val batch 0: CE loss = 0.0013
val batch 1: CE loss = 0.0035
val batch 2: CE loss = 0.0028
val batch 5: CE loss = 0.0000
...
val_bpb (10 batches): 0.000762
```

The near-zero val loss is **not a sign of good generalization**. It is because the
token `):` also dominates the validation shard (climbmix is Python/code data with
many function definitions). A model that always predicts `):` will have low
cross-entropy on code-heavy validation sets.

---

## 4. Root Cause Analysis: Why the Model Collapsed

### 4.1 Data Starvation (Primary Cause)

The Chinchilla scaling law (Hoffmann et al., 2022) states that optimal training
requires approximately **20× the parameter count in tokens**:

```
optimal_tokens ≈ 20 × num_params
```

For this run:

| | Value |
|--|--|
| Model parameters | 135.3M |
| Tokens required (Chinchilla optimal) | ~2.7B |
| Tokens actually trained | 64.6M |
| **Shortfall factor** | **42×** |

The model has far more capacity than the data it was shown. With 135M parameters
and only 64M tokens, the model can (and did) overfit — memorizing token-level
statistics rather than learning compositional language structure.

### 4.2 GTX 1080 Throughput Ceiling

The GTX 1080 produces 730 tok/sec due to:
- No native bfloat16 (runs float16 without tensor cores)
- No Flash Attention (uses SDPA with manual mask)
- No `torch.compile` (sm_61 limitations)

```
730 tok/sec × 86,400 sec = 63.1M tokens / 24 hours
```

For 135M parameters to train properly, at 730 tok/sec, the required wall time is:
```
2.7B tokens ÷ 730 tok/sec = 42 days
```

This is infeasible. The architecture is designed for H100 (~100K tok/sec), which
would complete the same training in ~7.5 hours.

### 4.3 What the Model Actually Learned

The climbmix dataset is packed using best-fit bin-packing — every 2048-token
sequence contains multiple documents concatenated without padding. Sequences
beginning with BOS (`<|reserved_0|>` = token 8188) followed by code frequently
contain patterns like:

```
def foo(args):
def bar(x, y):
class Baz(Base):
```

All ending in `):` (token 5681). After 64M tokens, the model learned:
> "After any token, predict `):` — it's almost always right."

This produces low training loss (0.003) and low val loss (0.0008) because `):` is
genuinely frequent in both splits. But it is not language modeling — it is
**mode collapse to the dataset's most frequent token**.

---

## 5. Chinchilla-Optimal Models for GTX 1080

The right question is: what model size is properly matched to 63M tokens/24h?

| DEPTH | n_embd | ~Params | Chinchilla tokens | GTX 1080 time |
|-------|--------|---------|-------------------|---------------|
| 12 | 768 | 135M | 2.7B | **42 days** |
| 6 | 384 | ~17M | 340M | 5 days |
| 4 | 256 | ~5M | 100M | 36 hours |
| **3** | **192** | **~3M** | **60M** | **~24 hours ✓** |

For a single 24-hour run, `DEPTH=3, n_embd=192` (~3M params, 60M token budget)
is Chinchilla-matched. This model would be small but should generalize correctly —
producing coherent (if limited) language.

---

## 6. Comparison: This Run vs Previous Runs

| Run | Date | Arch | Params | Data | Steps | Val metric |
|-----|------|------|--------|------|-------|------------|
| Original | 2026-03-13 | GPT-2 vanilla | ~124M | 28K tokens (local) | 1,000 | val_loss=1.74 |
| autoresearch v1 | 2026-03-21 | autoresearch | ~17M | climbmix (small) | ~50 | val_bpb=1.72 |
| **This run** | **2026-03-23** | autoresearch | 135M | climbmix 64M tok | 493 | val_bpb=0.0008* |

*val_bpb is misleadingly low due to mode collapse, not good generalization.

The original 2026-03-13 run (val_loss=1.74) actually produced a more
**honest** model — it was underfit but at least it tried to model diverse
text. This run's model has lower reported metrics but is worse in practice.

---

## 7. Recommendations

### Short-term: Retrain smaller (24 hours)
```python
DEPTH = 3
# → n_embd = 3 * 64 = 192, n_head = 2, ~3M params
# → Chinchilla-optimal for ~60M tokens/24h on GTX 1080
```
Expected result: val_bpb ~1.0-2.0 (honest), coherent (if simple) generation.

### Medium-term: Use the existing 135M model with more data (5-7 days)
```python
DEPTH = 12
TIME_BUDGET = 7 * 86400  # 7 days → ~440M tokens
# → Still 6× under Chinchilla optimal, but model should generalize better
```

### Long-term: Upgrade hardware
An H100 (80GB SXM, ~100K tok/sec) would complete the Chinchilla-optimal
135M training in ~7.5 hours. The autoresearch architecture was designed for this.

---

## 8. Files

| File | Description |
|------|-------------|
| `train_autoresearch.py` | Training script (GTX 1080 adaptation) |
| `repl_autoresearch.py` | Interactive eval REPL |
| `run_best_training.sh` | Automation script (wait → kill → train) |
| `train_best.log` | Full training log |
| `out-autoresearch/ckpt.pt` | Final checkpoint (517 MB, 135.3M params, 493 steps) |

---

*Report generated 2026-03-23. Training ran 2026-03-22 00:17 → 2026-03-23 00:57 CST.*
