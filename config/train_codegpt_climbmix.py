# CodeGPT training on climbmix-400b-shuffle dataset
# Much more data than local Python files (28K → billions of tokens)
# Data prep: ~/miniconda3/envs/codegpt/bin/python data/climbmix/prepare.py --num-shards 2

# I/O
out_dir = 'out-codegpt-climbmix'
eval_interval = 1000
log_interval = 10
eval_iters = 200

# data — climbmix is general web text, disable FIM
dataset = 'climbmix'
batch_size = 4
block_size = 512
gradient_accumulation_steps = 40

# model (GPT-2 124M scale)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# climbmix is general text — FIM is code-specific, disable it
fim_enabled = False
fim_rate = 0.0
fim_spm_rate = 0.0

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
dtype = 'float16'  # GTX 1080 has no native bfloat16
