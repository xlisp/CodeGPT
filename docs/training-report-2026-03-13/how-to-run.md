# 运行方式

## 快速开始

### 1. 准备数据

```bash
# 从本地 Python 项目
~/miniconda3/envs/codegpt/bin/python data/python_code/prepare.py \
    --source=local \
    --code_dir=/home/xlisp/PyPro

# 推荐：从 HuggingFace 获取大规模数据
~/miniconda3/envs/codegpt/bin/python data/python_code/prepare.py \
    --source=huggingface \
    --max_samples=100000
```

### 2. 启动训练

```bash
# 单卡训练（标准命令）
~/miniconda3/envs/codegpt/bin/python -W ignore train.py config/train_codegpt.py

# 快速验证（小 eval，少 iter，适合调试）
~/miniconda3/envs/codegpt/bin/python -W ignore train.py config/train_codegpt.py \
    --eval_iters=5 \
    --max_iters=50 \
    --eval_interval=50 \
    --log_interval=5

# 从检查点恢复
~/miniconda3/envs/codegpt/bin/python -W ignore train.py config/train_codegpt.py \
    --init_from=resume
```

### 3. 监控训练

```bash
# 监控 GPU 使用情况
watch -n 2 nvidia-smi

# 实时查看显存占用
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
    --format=csv --loop=2
```

## 配置文件说明

当前 `config/train_codegpt.py` 已针对 GTX 1080 8GB 优化：

```python
# 数据
batch_size = 4                  # 单卡适配（原 12）
block_size = 512                # 单卡适配（原 1024）
gradient_accumulation_steps = 40  # 保持有效 batch size

# 有效 batch size = 4 × 40 × 512 = 81,920 tokens/iter

# 系统
compile = False    # GTX 1080 (sm_61) 对 torch.compile 支持有限
dtype = 'float16'  # 必须！GTX 1080 无原生 bfloat16
```

## 注意事项

### 必须使用 conda 环境

```bash
# 正确：使用 Python 3.12 conda 环境
~/miniconda3/envs/codegpt/bin/python train.py ...

# 错误：系统 Python 3.13 会导致 CUDA kernel 报错
python3 train.py ...
```

### float16 不可省略

GTX 1080 PyTorch 2.3.1 会误报支持 bfloat16，但实为软件模拟。必须在 config 中显式写：
```python
dtype = 'float16'
```
否则训练速度降低约 10-30 倍。

### 训练数据要足够大

当前本地数据仅 28K tokens，极小。至少需要数百万 tokens 才能让 124M 模型有效收敛。推荐使用 HuggingFace codeparrot 数据集（约 54GB 原始数据）。

### eval_iters 与训练时间的权衡

每次 eval 耗时 = `eval_iters × 2 × 335ms`（train + val 各一遍）：

| eval_iters | eval 耗时 |
|-----------|----------|
| 5 | ~3.4 秒 |
| 50 | ~34 秒 |
| 200（默认）| ~134 秒（2.2 分钟）|

每 `eval_interval=1000` 次 iter 触发一次 eval。默认设置下每次 eval 需 2 分钟，占总时间约 0.4%，可以接受。

## 文件结构

```
CodeGPT/
├── train.py              # 训练主脚本
├── model.py              # CodeGPT 模型定义
├── tokenizer.py          # 代码 tokenizer
├── configurator.py       # 配置解析
├── config/
│   ├── train_codegpt.py  # 单卡训练配置（已适配 GTX 1080）
│   └── train_codegpt_small.py  # 小模型配置（10M 参数，适合快速实验）
├── data/
│   └── python_code/
│       ├── prepare.py    # 数据准备脚本
│       ├── train.bin     # 训练数据（二进制，uint16）
│       ├── val.bin       # 验证数据
│       └── meta.pkl      # 词表元数据
└── out-codegpt/
    └── ckpt.pt           # 训练检查点（1.4GB）
```
