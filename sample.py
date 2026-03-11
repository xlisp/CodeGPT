"""
CodeGPT Code Generation / Sampling Script.

Supports multiple generation modes:
  1. Code completion: generate code from a prompt
  2. Fill-in-the-Middle (FIM): infill code given prefix and suffix
  3. Interactive REPL: continuously generate code from user input

Usage:
    # Basic generation
    python sample.py --prompt="def fibonacci(n):"

    # FIM infill
    python sample.py --mode=fim --prefix="def add(a, b):" --suffix="    return result"

    # Interactive mode
    python sample.py --mode=interactive

    # From pretrained GPT-2
    python sample.py --init_from=gpt2
"""

import os
import sys
import pickle
from contextlib import nullcontext

import torch

from model import CodeGPT, CodeGPTConfig
from tokenizer import CodeTokenizer, SPECIAL_TOKENS

# ---------- config ----------
out_dir = 'out-codegpt'
init_from = 'resume'  # 'resume' or 'gpt2', 'gpt2-medium', etc.
mode = 'complete'  # 'complete', 'fim', 'interactive'

# generation params
prompt = 'def hello_world():\n'
prefix = ''       # for FIM mode
suffix = ''       # for FIM mode
lang = 'python'   # language hint

num_samples = 1
max_new_tokens = 512
temperature = 0.8
top_k = 200
top_p = 0.95
repetition_penalty = 1.1

# stop generation at these patterns
stop_at_double_newline = True

seed = 1337
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

# apply config overrides
from configurator import configure
configure()

# ---------- setup ----------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ---------- load model ----------
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = CodeGPTConfig(**checkpoint['model_args'])
    model = CodeGPT(config)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = CodeGPT.from_pretrained(init_from)
else:
    raise ValueError(f"Unknown init_from: {init_from}")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# ---------- tokenizer ----------
tokenizer = CodeTokenizer()

# stop tokens
stop_tokens = [SPECIAL_TOKENS["<|endoftext|>"], SPECIAL_TOKENS["<|code_end|>"]]


def encode_prompt(text, use_lang=True):
    """Encode a code prompt with optional language token."""
    tokens = []
    tokens.append(SPECIAL_TOKENS["<|code_start|>"])
    if use_lang and lang:
        lang_token = f"<|lang:{lang}|>"
        if lang_token in SPECIAL_TOKENS:
            tokens.append(SPECIAL_TOKENS[lang_token])
    tokens.extend(tokenizer.encode_raw(text))
    return tokens


def encode_fim(prefix_text, suffix_text):
    """Encode a FIM prompt: <|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|>"""
    tokens = []
    tokens.append(SPECIAL_TOKENS["<|fim_prefix|>"])
    tokens.extend(tokenizer.encode_raw(prefix_text))
    tokens.append(SPECIAL_TOKENS["<|fim_suffix|>"])
    tokens.extend(tokenizer.encode_raw(suffix_text))
    tokens.append(SPECIAL_TOKENS["<|fim_middle|>"])
    return tokens


def generate_code(input_tokens, n_samples=1):
    """Generate code from input tokens."""
    x = torch.tensor(input_tokens, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        with ctx:
            for i in range(n_samples):
                y = model.generate(
                    x,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    stop_tokens=stop_tokens,
                    repetition_penalty=repetition_penalty,
                )
                output_tokens = y[0].tolist()
                # only decode newly generated tokens
                generated_tokens = output_tokens[len(input_tokens):]
                code = tokenizer.decode(generated_tokens)

                if stop_at_double_newline and '\n\n\n' in code:
                    code = code[:code.index('\n\n\n')]

                if n_samples > 1:
                    print(f"--- Sample {i+1} ---")
                print(code)
                if n_samples > 1:
                    print()


def interactive_mode():
    """Interactive REPL for code generation."""
    print("=" * 60)
    print("  CodeGPT Interactive Code Generator")
    print("  Commands:")
    print("    /fim <prefix> ||| <suffix>  - Fill-in-the-middle")
    print("    /lang <language>             - Set language")
    print("    /temp <float>                - Set temperature")
    print("    /tokens <int>                - Set max tokens")
    print("    /quit                        - Exit")
    print("=" * 60)

    current_lang = lang
    current_temp = temperature
    current_max_tokens = max_new_tokens

    while True:
        try:
            print()
            user_input = input("CodeGPT> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input.strip():
            continue

        # handle commands
        if user_input.startswith('/quit'):
            print("Bye!")
            break
        elif user_input.startswith('/lang '):
            current_lang = user_input[6:].strip()
            print(f"Language set to: {current_lang}")
            continue
        elif user_input.startswith('/temp '):
            current_temp = float(user_input[6:].strip())
            print(f"Temperature set to: {current_temp}")
            continue
        elif user_input.startswith('/tokens '):
            current_max_tokens = int(user_input[8:].strip())
            print(f"Max tokens set to: {current_max_tokens}")
            continue
        elif user_input.startswith('/fim '):
            parts = user_input[5:].split('|||')
            if len(parts) == 2:
                tokens = encode_fim(parts[0].strip(), parts[1].strip())
                print("\n--- Generated infill ---")
                generate_code(tokens)
                continue
            else:
                print("FIM format: /fim <prefix> ||| <suffix>")
                continue

        # handle multi-line input (end with empty line)
        lines = [user_input]
        if user_input.endswith(':') or user_input.endswith('{') or user_input.endswith('\\'):
            print("  (multi-line mode, enter empty line to generate)")
            while True:
                try:
                    line = input("...   ")
                    if line == '':
                        break
                    lines.append(line)
                except (EOFError, KeyboardInterrupt):
                    break

        full_prompt = '\n'.join(lines)
        tokens = encode_prompt(full_prompt, use_lang=True)
        print("\n--- Generated code ---")
        generate_code(tokens)


# ---------- main ----------
if __name__ == '__main__':
    if mode == 'interactive':
        interactive_mode()
    elif mode == 'fim':
        if not prefix and not suffix:
            print("FIM mode requires --prefix and --suffix arguments")
            sys.exit(1)
        print(f"[FIM] Infilling between prefix and suffix...")
        tokens = encode_fim(prefix, suffix)
        generate_code(tokens, n_samples=num_samples)
    else:
        # completion mode
        print(f"[Complete] Generating from prompt...")
        print(f"Prompt: {prompt}")
        print("-" * 40)
        tokens = encode_prompt(prompt)
        generate_code(tokens, n_samples=num_samples)
