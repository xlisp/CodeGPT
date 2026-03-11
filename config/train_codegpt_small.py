# CodeGPT small model config
# Quick training for testing and development
# Runs on a single GPU or even CPU (slowly)

# I/O
out_dir = 'out-codegpt-small'
eval_interval = 500
log_interval = 10
eval_iters = 100

# data
dataset = 'python_code'
batch_size = 32
block_size = 512
gradient_accumulation_steps = 4

# small model (~10M params)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False

# code-specific
fim_enabled = True
fim_rate = 0.5
fim_spm_rate = 0.5

# optimizer
learning_rate = 3e-4
max_iters = 10000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule
decay_lr = True
warmup_iters = 500
lr_decay_iters = 10000
min_lr = 3e-5

# system
compile = False
