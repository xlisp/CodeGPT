"""
CodeGPT Training Script.

Supports:
  - Single GPU and multi-GPU (DDP) training
  - Fill-in-the-Middle (FIM) data augmentation
  - Mixed precision (float16/bfloat16)
  - Gradient accumulation
  - Cosine learning rate schedule with warmup
  - Checkpoint save/resume
  - wandb logging (optional)

Usage:
    # Single GPU
    python train.py config/train_codegpt.py

    # Multi-GPU (DDP)
    torchrun --standalone --nproc_per_node=4 train.py config/train_codegpt.py

    # Override config
    python train.py config/train_codegpt_small.py --batch_size=32 --max_iters=5000
"""

import os
import time
import math
import pickle
import random
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import CodeGPT, CodeGPTConfig
from tokenizer import apply_fim_transform, SPECIAL_TOKENS

# ---------- default config ----------
# I/O
out_dir = 'out-codegpt'
eval_interval = 1000
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch', 'resume', 'gpt2', 'gpt2-medium', etc.

# wandb
wandb_log = False
wandb_project = 'codegpt'
wandb_run_name = 'run' + str(time.time())

# data
dataset = 'python_code'
gradient_accumulation_steps = 8
batch_size = 16
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1
bias = False

# code-specific
fim_enabled = True
fim_rate = 0.5
fim_spm_rate = 0.5

# adamw optimizer
learning_rate = 3e-4
max_iters = 100000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate schedule
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 100000
min_lr = 3e-5

# DDP
backend = 'nccl'

# system
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True if torch.cuda.is_available() else False

# ---------- end default config ----------

# apply config file and CLI overrides
from configurator import configure
configure()

# ---------- DDP setup ----------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print(f"tokens per iteration: {tokens_per_iter:,}")
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ---------- data loading ----------
data_dir = os.path.join('data', dataset)


def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])

    # apply FIM transformation for code completion training
    if fim_enabled and split == 'train':
        x_fim = []
        y_fim = []
        for b in range(batch_size):
            tokens = x[b].tolist()
            tokens_transformed = apply_fim_transform(tokens, fim_rate=fim_rate, fim_spm_rate=fim_spm_rate)
            # pad or truncate to block_size
            if len(tokens_transformed) > block_size:
                tokens_transformed = tokens_transformed[:block_size]
            elif len(tokens_transformed) < block_size:
                tokens_transformed = tokens_transformed + [SPECIAL_TOKENS["<|fim_pad|>"]] * (block_size - len(tokens_transformed))
            x_fim.append(torch.tensor(tokens_transformed[:-1], dtype=torch.long))

            # target: shifted by 1, with padding tokens masked
            target = tokens_transformed[1:] + [SPECIAL_TOKENS["<|fim_pad|>"]]
            target = target[:block_size]
            target_tensor = torch.tensor(target, dtype=torch.long)
            # mask padding tokens in loss
            target_tensor[target_tensor == SPECIAL_TOKENS["<|fim_pad|>"]] = -1
            y_fim.append(target_tensor)

        x = torch.stack(x_fim)
        y = torch.stack(y_fim)

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# ---------- model init ----------
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta.get('vocab_size', None)
    if master_process:
        print(f"found vocab_size = {meta_vocab_size} (from {meta_path})")

model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    block_size=block_size, bias=bias, dropout=dropout,
    fim_enabled=fim_enabled,
)

iter_num = 0
best_val_loss = 1e9

if init_from == 'scratch':
    if master_process:
        print("Initializing CodeGPT model from scratch")
    if meta_vocab_size is not None:
        model_args['vocab_size'] = meta_vocab_size
    config = CodeGPTConfig(**model_args)
    model = CodeGPT(config)

elif init_from == 'resume':
    if master_process:
        print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    config = CodeGPTConfig(**model_args)
    model = CodeGPT(config)
    state_dict = checkpoint['model']
    # fix key prefixes from DDP
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    if master_process:
        print(f"Initializing from pretrained {init_from}")
    model = CodeGPT.from_pretrained(init_from, override_args={'dropout': dropout})
    # expand vocab for code special tokens
    model.expand_vocab(CodeGPTConfig.vocab_size)
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
    model_args['vocab_size'] = model.config.vocab_size

model.to(device)

# ---------- GradScaler for mixed precision ----------
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))

# ---------- optimizer ----------
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free memory

# ---------- compile ----------
if compile:
    if master_process:
        print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# ---------- DDP wrapper ----------
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# ---------- loss estimation ----------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ---------- learning rate schedule ----------
def get_lr(it):
    # linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # cosine decay
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ---------- logging ----------
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config={
        'batch_size': batch_size, 'block_size': block_size,
        'n_layer': n_layer, 'n_head': n_head, 'n_embd': n_embd,
        'learning_rate': learning_rate, 'max_iters': max_iters,
        'fim_enabled': fim_enabled, 'fim_rate': fim_rate,
        'dataset': dataset,
    })

# ---------- training loop ----------
if master_process:
    print(f"Starting CodeGPT training")
    print(f"  dataset: {dataset}")
    print(f"  FIM enabled: {fim_enabled} (rate={fim_rate})")
    print(f"  device: {device}, dtype: {dtype}")

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    # set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate and checkpoint
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = min(best_val_loss, losses['val'])
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': raw_model.config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if eval_only:
        break

    # forward/backward with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    # gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

print("Training complete!")
