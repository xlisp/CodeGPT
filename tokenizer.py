"""
CodeGPT Tokenizer - Code-aware tokenization with special tokens for code generation.

Features:
  - GPT-2 BPE base tokenizer (via tiktoken)
  - Fill-in-the-Middle (FIM) special tokens
  - Language identifier tokens
  - Code boundary tokens
  - FIM transformation for training data
"""

import random
import tiktoken
import numpy as np


# Special token definitions
SPECIAL_TOKENS = {
    "<|endoftext|>":  50256,
    "<|fim_prefix|>": 50257,
    "<|fim_middle|>": 50258,
    "<|fim_suffix|>": 50259,
    "<|fim_pad|>":    50260,
    "<|code_start|>": 50261,
    "<|code_end|>":   50262,
    # Language tokens
    "<|lang:python|>":     50263,
    "<|lang:javascript|>": 50264,
    "<|lang:typescript|>": 50265,
    "<|lang:java|>":       50266,
    "<|lang:c|>":          50267,
    "<|lang:cpp|>":        50268,
    "<|lang:go|>":         50269,
    "<|lang:rust|>":       50270,
    "<|lang:ruby|>":       50271,
    "<|lang:php|>":        50272,
    "<|lang:shell|>":      50273,
    "<|lang:sql|>":        50274,
    "<|lang:html|>":       50275,
    "<|lang:css|>":        50276,
    "<|lang:markdown|>":   50277,
    "<|lang:other|>":      50278,
}

# Reverse mapping
SPECIAL_TOKENS_REVERSE = {v: k for k, v in SPECIAL_TOKENS.items()}

# File extension to language mapping
EXT_TO_LANG = {
    '.py': 'python', '.pyw': 'python',
    '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript',
    '.ts': 'typescript', '.tsx': 'typescript',
    '.java': 'java',
    '.c': 'c', '.h': 'c',
    '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp', '.hpp': 'cpp',
    '.go': 'go',
    '.rs': 'rust',
    '.rb': 'ruby',
    '.php': 'php',
    '.sh': 'shell', '.bash': 'shell', '.zsh': 'shell',
    '.sql': 'sql',
    '.html': 'html', '.htm': 'html',
    '.css': 'css', '.scss': 'css', '.less': 'css',
    '.md': 'markdown', '.markdown': 'markdown',
}

VOCAB_SIZE = 50304  # padded for efficiency (multiple of 64)


class CodeTokenizer:
    """Code-aware tokenizer wrapping tiktoken GPT-2 BPE."""

    def __init__(self):
        self.base_enc = tiktoken.get_encoding("gpt2")
        self.special_tokens = SPECIAL_TOKENS
        self.special_tokens_reverse = SPECIAL_TOKENS_REVERSE
        self.eot_token = 50256
        self.vocab_size = VOCAB_SIZE

    def encode(self, text, lang=None, add_code_boundaries=True):
        """
        Encode text to token IDs.

        Args:
            text: source code string
            lang: language identifier (e.g. 'python')
            add_code_boundaries: whether to wrap with code_start/code_end tokens
        """
        tokens = []

        if add_code_boundaries:
            tokens.append(SPECIAL_TOKENS["<|code_start|>"])
            if lang:
                lang_token = f"<|lang:{lang}|>"
                if lang_token in SPECIAL_TOKENS:
                    tokens.append(SPECIAL_TOKENS[lang_token])

        tokens.extend(self.base_enc.encode_ordinary(text))

        if add_code_boundaries:
            tokens.append(SPECIAL_TOKENS["<|code_end|>"])

        return tokens

    def decode(self, tokens):
        """Decode token IDs back to text, handling special tokens."""
        parts = []
        regular_tokens = []

        for t in tokens:
            if t in self.special_tokens_reverse:
                # flush regular tokens
                if regular_tokens:
                    parts.append(self.base_enc.decode(regular_tokens))
                    regular_tokens = []
                # skip special tokens in output (they're metadata)
            else:
                regular_tokens.append(t)

        if regular_tokens:
            parts.append(self.base_enc.decode(regular_tokens))

        return ''.join(parts)

    def encode_raw(self, text):
        """Encode without any special tokens."""
        return self.base_enc.encode_ordinary(text)

    def decode_raw(self, tokens):
        """Decode only regular tokens."""
        regular = [t for t in tokens if t not in self.special_tokens_reverse]
        return self.base_enc.decode(regular)

    @staticmethod
    def detect_language(filepath):
        """Detect programming language from file extension."""
        import os
        ext = os.path.splitext(filepath)[1].lower()
        return EXT_TO_LANG.get(ext, 'other')

    @staticmethod
    def get_lang_token_id(lang):
        """Get the special token ID for a language."""
        token = f"<|lang:{lang}|>"
        return SPECIAL_TOKENS.get(token, SPECIAL_TOKENS["<|lang:other|>"])


def apply_fim_transform(tokens, fim_rate=0.5, fim_spm_rate=0.5):
    """
    Apply Fill-in-the-Middle transformation to a token sequence.

    With probability `fim_rate`, transforms a sequence into FIM format:
      - PSM (Prefix-Suffix-Middle): <|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|> middle
      - SPM (Suffix-Prefix-Middle): <|fim_suffix|> suffix <|fim_prefix|> prefix <|fim_middle|> middle

    This teaches the model to infill code given surrounding context.

    Args:
        tokens: list of token IDs
        fim_rate: probability of applying FIM
        fim_spm_rate: probability of using SPM variant (vs PSM)

    Returns:
        transformed token list
    """
    if random.random() >= fim_rate:
        return tokens  # no transformation

    # filter out special tokens for FIM (keep only regular code tokens)
    prefix_id = SPECIAL_TOKENS["<|fim_prefix|>"]
    middle_id = SPECIAL_TOKENS["<|fim_middle|>"]
    suffix_id = SPECIAL_TOKENS["<|fim_suffix|>"]

    # choose a random split point for the middle section
    if len(tokens) < 3:
        return tokens

    # pick two random points to define prefix/middle/suffix
    boundaries = sorted(random.sample(range(1, len(tokens)), min(2, len(tokens) - 1)))
    if len(boundaries) < 2:
        return tokens

    prefix = tokens[:boundaries[0]]
    middle = tokens[boundaries[0]:boundaries[1]]
    suffix = tokens[boundaries[1]:]

    if random.random() < fim_spm_rate:
        # SPM: suffix-prefix-middle
        return [suffix_id] + suffix + [prefix_id] + prefix + [middle_id] + middle
    else:
        # PSM: prefix-suffix-middle
        return [prefix_id] + prefix + [suffix_id] + suffix + [middle_id] + middle


def tokenize_code_file(filepath, tokenizer=None):
    """
    Tokenize a single code file with language detection and code boundaries.

    Returns:
        list of token IDs
    """
    if tokenizer is None:
        tokenizer = CodeTokenizer()

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    lang = CodeTokenizer.detect_language(filepath)
    tokens = tokenizer.encode(text, lang=lang, add_code_boundaries=True)
    tokens.append(tokenizer.eot_token)

    return tokens
