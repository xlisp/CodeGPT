# CodeGPT full training config
# GPT-2 124M scale model trained on code
# Designed for 4-8 GPUs (A100 40GB)

# I/O
out_dir = 'out-codegpt'
eval_interval = 1000
log_interval = 10
eval_iters = 200

# data
dataset = 'python_code'
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 40

# model (GPT-2 124M scale)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# code-specific
fim_enabled = True
fim_rate = 0.5
fim_spm_rate = 0.5

# optimizer
learning_rate = 6e-4
max_iters = 200000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 200000
min_lr = 6e-5

# system
compile = True
