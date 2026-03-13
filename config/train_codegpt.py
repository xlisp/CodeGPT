# CodeGPT full training config
# GPT-2 124M scale model trained on code
# Adapted for single GPU (GTX 1080 8GB)

# I/O
out_dir = 'out-codegpt'
eval_interval = 1000
log_interval = 10
eval_iters = 200

# data
dataset = 'python_code'
batch_size = 4
block_size = 512
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
compile = False  # sm_61 (GTX 1080) has limited torch.compile support
dtype = 'float16'  # GTX 1080 has no native bfloat16, force float16
