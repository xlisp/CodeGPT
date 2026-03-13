# CodeGPT 单卡 GPU 训练报告

**日期**: 2026-03-13
**目标**: 在单张 GTX 1080 上运行 `python train.py config/train_codegpt.py`
**结果**: 成功，50 iterations 后 val loss 从 10.93 降至 5.36，检查点已保存

---

## 目录

| 文档 | 内容 |
|------|------|
| [environment.md](environment.md) | 硬件配置、软件版本、安装命令 |
| [model.md](model.md) | 模型架构、参数配置、单卡适配调整 |
| [training-data.md](training-data.md) | 数据来源、统计、处理流程 |
| [bugs-and-fixes.md](bugs-and-fixes.md) | 4 个问题的根因分析与解决方案 |
| [training-results.md](training-results.md) | Loss 曲线、训练速度、性能分析 |
| [model-evaluation.md](model-evaluation.md) | 模型实测（28K tokens）：补全效果、FIM、perplexity、改进方向 |
| [training-report-8mb-corpus.md](training-report-8mb-corpus.md) | **8MB 语料完整训练报告**（1330 iter，val loss=1.74，含测试结果）|
| [how-to-run.md](how-to-run.md) | 完整运行方式与注意事项 |

---

## 快速摘要

### 环境

- GPU: **NVIDIA GTX 1080** 8GB（Pascal sm_61）
- Python: **3.12**（conda，系统 3.13 不兼容）
- PyTorch: **2.3.1+cu118**（最后支持 sm_61 的版本）

### 模型

- 架构：GPT-2 Decoder-only Transformer
- 参数量：**123.59M**（12 层，12 头，768 维）
- 特性：Fill-in-the-Middle (FIM) 代码补全训练

### 训练结果

```
初始 val loss:   10.93  （随机权重，接近理论值 ln(50304) = 10.83）
50 iter val loss: 5.36  （模型开始学习代码模式）
检查点大小:       1.4 GB
训练速度:         ~28 秒/iter，~2,926 tokens/sec
GPU 显存占用:     5062 MiB / 8192 MiB
```

### 修复的 Bug

1. **环境不兼容**：Python 3.13 + PyTorch ≥ 2.4 不支持 GTX 1080 (sm_61) → 用 conda Python 3.12 + PyTorch 2.3.1
2. **`model.py` KeyError**：权重绑定导致 `lm_head.weight` 不在 `param_dict` → `decay = decay & param_dict.keys()`
3. **`train.py` batch 尺寸错误**：FIM target 长度比 x 多 1 → `target = tokens_transformed[1:]`
4. **bfloat16 误报**：PyTorch 2.3 对 GTX 1080 误报支持 bfloat16，实为极慢软件模拟 → 配置 `dtype = 'float16'`

### 运行命令

```bash
~/miniconda3/envs/codegpt/bin/python -W ignore train.py config/train_codegpt.py
```
