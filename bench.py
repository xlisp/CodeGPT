"""
Benchmark CodeGPT model performance.
Measures throughput, latency, and MFU.
"""

import os
import time
import torch
from contextlib import nullcontext

from model import CodeGPT, CodeGPTConfig

# config
batch_size = 8
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
bias = False
seed = 1337
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True if torch.cuda.is_available() else False
num_warmup = 10
num_iters = 50

from configurator import configure
configure()

torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# create model
config = CodeGPTConfig(
    block_size=block_size, n_layer=n_layer, n_head=n_head,
    n_embd=n_embd, bias=bias, dropout=0.0,
)
model = CodeGPT(config)
model.to(device)

if compile:
    print("Compiling model...")
    model = torch.compile(model)

# synthetic data
x = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)
y = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)

# warmup
print(f"Warming up ({num_warmup} iters)...")
for _ in range(num_warmup):
    with ctx:
        logits, loss = model(x, y)
    loss.backward()
    model.zero_grad(set_to_none=True)

if device_type == 'cuda':
    torch.cuda.synchronize()

# benchmark
print(f"Benchmarking ({num_iters} iters)...")
times = []
for _ in range(num_iters):
    t0 = time.time()
    with ctx:
        logits, loss = model(x, y)
    loss.backward()
    model.zero_grad(set_to_none=True)
    if device_type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    times.append(t1 - t0)

avg_time = sum(times) / len(times)
tokens_per_sec = batch_size * block_size / avg_time
mfu = model.estimate_mfu(batch_size, avg_time) if hasattr(model, 'estimate_mfu') else 0

print(f"\nResults:")
print(f"  Model: {config.n_layer}L/{config.n_head}H/{config.n_embd}E ({model.get_num_params()/1e6:.1f}M params)")
print(f"  Batch size: {batch_size}, Block size: {block_size}")
print(f"  Device: {device}, Dtype: {dtype}, Compile: {compile}")
print(f"  Avg iter time: {avg_time*1000:.2f} ms")
print(f"  Throughput: {tokens_per_sec:,.0f} tokens/sec")
print(f"  MFU: {mfu*100:.2f}%")
