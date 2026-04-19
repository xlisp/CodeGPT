# 硬件与软件环境

## 硬件

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA GeForce GTX 1080 |
| GPU 显存 | 8192 MiB |
| GPU 架构 | Pascal (sm_61) |
| GPU TDP | 198W |
| NVIDIA 驱动 | 580.126.09 |
| CUDA Runtime | 13.0 (驱动支持) |
| 训练时显存占用 | ~5062 MiB / 8192 MiB (62%) |

## 软件

| 项目 | 版本 | 备注 |
|------|------|------|
| 操作系统 | Linux 6.17.0-5-generic | |
| Python | 3.12.x (conda) | 系统 Python 3.13 不可用，见问题说明 |
| PyTorch | 2.3.1+cu118 | 最后一个支持 sm_61 的版本 |
| CUDA (编译) | 11.8 | cu118 wheel |
| tiktoken | 0.12.0 | GPT-2 BPE tokenizer |
| numpy | 2.4.3 | |
| conda 环境路径 | `~/miniconda3/envs/codegpt` | |

## 关键兼容性说明

GTX 1080 是 Pascal 架构 (Compute Capability 6.1 = sm_61)。

- **PyTorch 2.4+** 起放弃了对 sm_61 的支持（不再编译对应 CUDA kernel）
- **系统 Python 3.13** 只有 PyTorch 2.5+ 才有对应 wheel
- **两者存在根本矛盾**：Python 3.13 要求 PyTorch ≥ 2.5，而 GTX 1080 要求 PyTorch ≤ 2.3

解决方案：通过 Miniconda 安装 Python 3.12，搭配 PyTorch 2.3.1+cu118。

```
系统 Python 3.13  →  [不能用] PyTorch 2.10 → sm_61 无 kernel，运行报错
conda Python 3.12 →  [正确] PyTorch 2.3.1+cu118 → sm_61 完全支持
```

## 安装命令

```bash
# 1. 安装 Miniconda
curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3

# 2. 接受 ToS
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 3. 创建 Python 3.12 环境
~/miniconda3/bin/conda create -n codegpt python=3.12 -y

# 4. 安装 PyTorch 2.3.1 (cu118, 支持 sm_61)
~/miniconda3/envs/codegpt/bin/pip install \
    "torch==2.3.1+cu118" \
    --find-links https://download.pytorch.org/whl/cu118/torch_stable.html

# 5. 安装其他依赖
~/miniconda3/envs/codegpt/bin/pip install tiktoken numpy
```
