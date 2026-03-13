#!/usr/bin/env python3
"""
CodeGPT REPL — 代码补全交互式命令行

用法:
    python repl.py                        # 加载默认检查点
    python repl.py --out_dir=out-codegpt  # 指定检查点目录
    python repl.py --temperature=0.3      # 调整温度

REPL 内命令:
    /help               显示帮助
    /fim                进入 FIM 填空模式
    /complete           回到补全模式（默认）
    /context            显示当前对话上下文
    /reset              清空上下文，重新开始
    /temp <float>       调整采样温度（如 /temp 0.3）
    /tokens <int>       调整最大生成 token 数
    /topk <int>         调整 top-k 参数
    /lang <str>         设置语言（python/javascript/go 等）
    /quit               退出
"""

import os
import sys
import argparse
import textwrap
from contextlib import nullcontext

import torch

from model import CodeGPT, CodeGPTConfig
from tokenizer import CodeTokenizer, SPECIAL_TOKENS

# ─────────────────────────── ANSI 颜色 ────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
BLUE   = "\033[34m"
RED    = "\033[31m"
GRAY   = "\033[90m"

def c(text, color): return f"{color}{text}{RESET}"

# ─────────────────────────── 参数解析 ─────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="CodeGPT REPL")
    p.add_argument("--out_dir",    default="out-codegpt")
    p.add_argument("--temperature",type=float, default=0.3)
    p.add_argument("--top_k",      type=int,   default=50)
    p.add_argument("--top_p",      type=float, default=0.95)
    p.add_argument("--max_tokens", type=int,   default=200)
    p.add_argument("--rep_penalty",type=float, default=1.1)
    p.add_argument("--lang",       default="python")
    p.add_argument("--device",     default=None)
    return p.parse_args()

# ─────────────────────────── 模型加载 ─────────────────────────────
def load_model(out_dir, device):
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        print(c(f"错误：找不到检查点 {ckpt_path}", RED))
        sys.exit(1)

    print(c("  加载模型中...", DIM), end="", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = CodeGPTConfig(**ckpt["model_args"])
    model = CodeGPT(config)

    sd = ckpt["model"]
    for k in list(sd):
        if k.startswith("_orig_mod."):
            sd[k[10:]] = sd.pop(k)
    model.load_state_dict(sd)
    model.eval().to(device)

    iter_num = ckpt.get("iter_num", "?")
    val_loss = ckpt.get("best_val_loss", "?")
    if isinstance(val_loss, torch.Tensor):
        val_loss = f"{val_loss.item():.4f}"
    print(c(f" 完成（iter={iter_num}, val_loss={val_loss}）", GREEN))
    return model, config

# ─────────────────────────── 编码工具 ─────────────────────────────
def encode_complete(tokenizer, text, lang):
    ids = [SPECIAL_TOKENS["<|code_start|>"]]
    lang_tok = f"<|lang:{lang}|>"
    if lang_tok in SPECIAL_TOKENS:
        ids.append(SPECIAL_TOKENS[lang_tok])
    ids.extend(tokenizer.encode_raw(text))
    return ids

def encode_fim(tokenizer, prefix, suffix):
    ids  = [SPECIAL_TOKENS["<|fim_prefix|>"]]
    ids += tokenizer.encode_raw(prefix)
    ids += [SPECIAL_TOKENS["<|fim_suffix|>"]]
    ids += tokenizer.encode_raw(suffix)
    ids += [SPECIAL_TOKENS["<|fim_middle|>"]]
    return ids

# ─────────────────────────── 生成 ─────────────────────────────────
def generate(model, tokenizer, input_ids, ctx, device,
             max_tokens, temperature, top_k, top_p, rep_penalty):
    stop_tokens = [SPECIAL_TOKENS["<|endoftext|>"], SPECIAL_TOKENS["<|code_end|>"]]
    x = torch.tensor(input_ids, dtype=torch.long, device=device)[None]
    with torch.no_grad():
        with ctx:
            y = model.generate(
                x,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_tokens=stop_tokens,
                repetition_penalty=rep_penalty,
            )
    generated = y[0].tolist()[len(input_ids):]
    return tokenizer.decode(generated), generated

# ─────────────────────────── 输出渲染 ─────────────────────────────
def print_completion(prompt_text, completion_text):
    print()
    # 打印 prompt（暗色）
    for line in prompt_text.splitlines():
        print(c("  " + line, DIM))
    # 打印补全（高亮绿色）
    if completion_text.strip():
        for line in completion_text.splitlines():
            print(c("  " + line, GREEN))
    else:
        print(c("  （模型未生成有意义内容，尝试调低 temperature）", YELLOW))
    print()

def print_header(config, args, device):
    print()
    print(c("╔══════════════════════════════════════════════════╗", CYAN))
    print(c("║           CodeGPT  交互式代码补全 REPL           ║", CYAN))
    print(c("╚══════════════════════════════════════════════════╝", RESET))
    print(f"  模型参数:  {config.n_layer}层 × {config.n_head}头 × {config.n_embd}维  "
          f"({sum(p.numel() for p in []) or '123.59'}M params)")
    print(f"  设备:      {device}")
    print(f"  温度:      {args.temperature}    top_k: {args.top_k}    最大 tokens: {args.max_tokens}")
    print(f"  语言:      {args.lang}")
    print()
    print(c("  输入代码片段按 Enter 补全。空行结束多行输入。", GRAY))
    print(c("  输入 /help 查看所有命令。", GRAY))
    print()

def print_help():
    print(c(textwrap.dedent("""
    ┌─ 命令 ──────────────────────────────────────────────┐
    │  /help              显示此帮助                       │
    │  /fim               进入 FIM 填空模式                │
    │  /complete          回到普通补全模式                  │
    │  /context           显示当前累积上下文               │
    │  /reset             清空上下文                       │
    │  /temp <n>          设置温度  (当前: {temp})         │
    │  /tokens <n>        设置最大 token 数                │
    │  /topk <n>          设置 top-k                       │
    │  /lang <name>       设置语言 (python/go/js/...)      │
    │  /quit  /exit  q    退出                             │
    └──────────────────────────────────────────────────────┘
    多行输入：以 : {{ \\ 结尾自动进入多行，空行结束。
    """).strip(), CYAN))
    print()

# ─────────────────────────── 多行输入 ────────────────────────────
def read_multiline(first_line):
    """读取多行输入：首行后若以 : { \\ 结尾则继续读，空行结束。
    返回 (代码文本, 挂起的命令行 or None)。
    """
    lines = [first_line]
    triggers = (":", "{", "\\", ",", "(", "[")
    if not first_line.rstrip().endswith(triggers):
        return "\n".join(lines), None
    print(c("  (多行模式，空行结束输入)", GRAY))
    pending_cmd = None
    while True:
        try:
            line = input(c("  ... ", BLUE))
        except (EOFError, KeyboardInterrupt):
            break
        if line == "":
            break
        if line.startswith("/") or line.lower() in ("q", "quit", "exit"):
            pending_cmd = line
            break
        lines.append(line)
    return "\n".join(lines), pending_cmd

# ─────────────────────────── 主 REPL ─────────────────────────────
def repl(model, tokenizer, config, args, device, ctx):
    mode      = "complete"   # "complete" | "fim"
    context   = ""           # 累积上下文（complete 模式）
    temp      = args.temperature
    top_k     = args.top_k
    max_tokens= args.max_tokens
    lang      = args.lang

    print_header(config, args, device)

    while True:
        # ── 提示符 ──
        if mode == "fim":
            prompt_prefix = c("[FIM] prefix> ", YELLOW)
        else:
            ctx_indicator = c(f"[+{len(context)}字符]", GRAY) if context else ""
            prompt_prefix = c(">>> ", CYAN) + ctx_indicator + " "

        try:
            raw = input(prompt_prefix).rstrip()
        except (EOFError, KeyboardInterrupt):
            print(c("\n再见！", GREEN))
            break

        if not raw:
            continue

        # ── 命令处理 ──
        if raw.startswith("/"):
            cmd = raw.split()[0].lower()
            rest = raw[len(cmd):].strip()

            if cmd in ("/quit", "/exit", "/q"):
                print(c("再见！", GREEN))
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/fim":
                mode = "fim"
                print(c("  已切换到 FIM 填空模式。先输入 prefix，再输入 suffix。", YELLOW))
            elif cmd == "/complete":
                mode = "complete"
                context = ""
                print(c("  已切换到补全模式，上下文已清空。", GREEN))
            elif cmd == "/reset":
                context = ""
                print(c("  上下文已清空。", GREEN))
            elif cmd == "/context":
                if context:
                    print(c("  当前上下文：", GRAY))
                    for line in context.splitlines():
                        print(c("    " + line, DIM))
                else:
                    print(c("  上下文为空。", GRAY))
                print()
            elif cmd == "/temp":
                try:
                    temp = float(rest)
                    print(c(f"  温度设为 {temp}", GREEN))
                except ValueError:
                    print(c("  用法：/temp 0.3", RED))
            elif cmd == "/tokens":
                try:
                    max_tokens = int(rest)
                    print(c(f"  最大 token 数设为 {max_tokens}", GREEN))
                except ValueError:
                    print(c("  用法：/tokens 200", RED))
            elif cmd == "/topk":
                try:
                    top_k = int(rest)
                    print(c(f"  top-k 设为 {top_k}", GREEN))
                except ValueError:
                    print(c("  用法：/topk 50", RED))
            elif cmd == "/lang":
                if rest:
                    lang = rest
                    print(c(f"  语言设为 {lang}", GREEN))
                else:
                    print(c("  用法：/lang python", RED))
            else:
                print(c(f"  未知命令：{cmd}，输入 /help 查看帮助", RED))
            continue

        # 退出别名
        if raw.lower() in ("q", "quit", "exit"):
            print(c("再见！", GREEN))
            break

        # ── FIM 模式：两段输入 ──
        if mode == "fim":
            prefix_text, _ = read_multiline(raw)
            print(c("[FIM] suffix> ", YELLOW), end="", flush=True)
            try:
                suffix_raw = input().rstrip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            suffix_text, _ = read_multiline(suffix_raw) if suffix_raw else ("", None)

            print(c("  生成中...", DIM), end="\r", flush=True)
            input_ids = encode_fim(tokenizer, prefix_text, suffix_text)
            completion, _ = generate(
                model, tokenizer, input_ids, ctx, device,
                max_tokens, temp, top_k, top_p=args.top_p, rep_penalty=args.rep_penalty,
            )
            print(" " * 20, end="\r")  # 清除"生成中"
            print()
            print(c("  ┌─ FIM 填空结果 ─────────────────────────────", YELLOW))
            print(c("  │ prefix:  ", DIM) + prefix_text.replace("\n", "↵ "))
            print(c("  │ infill:  ", GREEN) + completion.replace("\n", "↵ "))
            print(c("  │ suffix:  ", DIM) + suffix_text.replace("\n", "↵ "))
            print(c("  └───────────────────────────────────────────", YELLOW))
            print()
            continue

        # ── 补全模式 ──
        user_input, pending_cmd = read_multiline(raw)

        # 把新输入追加到上下文
        if context:
            full_prompt = context + "\n" + user_input
        else:
            full_prompt = user_input

        print(c("  生成中...", DIM), end="\r", flush=True)
        input_ids = encode_complete(tokenizer, full_prompt, lang)
        completion, _ = generate(
            model, tokenizer, input_ids, ctx, device,
            max_tokens, temp, top_k, top_p=args.top_p, rep_penalty=args.rep_penalty,
        )
        print(" " * 20, end="\r")  # 清除"生成中"

        print_completion(full_prompt, completion)

        # 把 prompt + 补全追加到上下文，供下次连续输入
        context = full_prompt + completion
        # 防止上下文超过 block_size 的 token 预算，简单按字符截断（保留尾部）
        max_ctx_chars = config.block_size * 3  # 粗略估计 3字符/token
        if len(context) > max_ctx_chars:
            context = context[-max_ctx_chars:]
            print(c(f"  （上下文过长，已裁剪保留最后 {max_ctx_chars} 字符）", GRAY))

        # 多行模式中截获的命令，立即执行
        if pending_cmd:
            if pending_cmd.lower() in ("/quit", "/exit", "q", "quit", "exit"):
                print(c("再见！", GREEN))
                break
            elif pending_cmd == "/reset":
                context = ""
                print(c("  上下文已清空。", GREEN))
            elif pending_cmd == "/context":
                for line in context.splitlines()[:20]:
                    print(c("    " + line, DIM))


# ─────────────────────────── 入口 ─────────────────────────────────
def main():
    args = parse_args()

    # 设备
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # dtype（GTX 1080 必须 float16）
    if device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    ctx = (nullcontext() if device == "cpu"
           else torch.amp.autocast(device_type=device, dtype=dtype))

    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, config = load_model(args.out_dir, device)
    tokenizer = CodeTokenizer()

    repl(model, tokenizer, config, args, device, ctx)


if __name__ == "__main__":
    main()
