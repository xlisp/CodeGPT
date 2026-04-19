# 训练数据

## 数据来源

本次训练使用本地 Python 代码文件，来自 `~/PyPro` 目录。

## 数据统计

| 项目 | 数值 |
|------|------|
| 来源目录 | `~/PyPro`（本地 Python 项目）|
| 收集文件数 | **11 个** `.py` 文件 |
| 总 token 数 | **28,204 tokens** |
| 训练集 | 24,426 tokens（10 个文档）|
| 验证集 | 3,778 tokens（1 个文档）|
| train.bin 大小 | 48,852 字节（~48KB）|
| val.bin 大小 | 7,556 字节（~7.4KB）|

> 注意：当前数据集极小（28K tokens），仅用于验证训练流程可运行。
> 工业级训练需要数十亿 tokens，推荐使用 HuggingFace 数据集。

## 数据处理流程

```
Python 源文件
    ↓ 过滤（100B–100KB，有效 UTF-8）
    ↓ 语言检测（.py → python）
    ↓ CodeTokenizer.encode()
       - 添加 <|code_start|> + <|lang:python|>
       - GPT-2 BPE 编码（tiktoken）
       - 添加 <|code_end|> + <|endoftext|>
    ↓ 按文档 shuffle
    ↓ 5% 切分为验证集
    ↓ 保存为 uint16 二进制文件（train.bin / val.bin）
    ↓ 保存元数据（meta.pkl：vocab_size, special_tokens）
```

## 数据准备命令

```bash
# 从本地 Python 项目准备数据（本次使用）
~/miniconda3/envs/codegpt/bin/python data/python_code/prepare.py \
    --source=local \
    --code_dir=/home/xlisp/PyPro

# 从 HuggingFace 下载大规模数据（推荐用于正式训练）
~/miniconda3/envs/codegpt/bin/python data/python_code/prepare.py \
    --source=huggingface \
    --max_samples=100000

# 两者结合
~/miniconda3/envs/codegpt/bin/python data/python_code/prepare.py \
    --source=both \
    --code_dir=/home/xlisp/PyPro \
    --max_samples=100000
```

## Tokenizer 说明

使用 `tiktoken` 的 GPT-2 BPE 编码，基础词表 50257 个 token，扩展至 50304（加入代码专用特殊 token，填充至 64 倍数提升 GPU 效率）。

文件扩展名到语言的映射（用于语言 token 标注）：

| 扩展名 | 语言 token |
|--------|-----------|
| `.py` `.pyw` | `<\|lang:python\|>` |
| `.js` `.jsx` | `<\|lang:javascript\|>` |
| `.ts` `.tsx` | `<\|lang:typescript\|>` |
| `.go` | `<\|lang:go\|>` |
| `.rs` | `<\|lang:rust\|>` |
| `.cpp` `.cc` | `<\|lang:cpp\|>` |
| ... | ... |
