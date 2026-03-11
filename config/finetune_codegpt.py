# Finetune a pretrained GPT-2 model for code generation
# Start from GPT-2 weights and finetune on code data
# Uses lower learning rate and fewer iterations

# I/O
out_dir = 'out-codegpt-finetune'
eval_interval = 200
log_interval = 10
eval_iters = 100

# init from pretrained GPT-2
init_from = 'gpt2'

# data
dataset = 'python_code'
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 8

# model params inherited from GPT-2
dropout = 0.1

# code-specific
fim_enabled = True
fim_rate = 0.5
fim_spm_rate = 0.5

# optimizer - lower LR for finetuning
learning_rate = 1e-4
max_iters = 5000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule
decay_lr = True
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 1e-5

# system
compile = True
