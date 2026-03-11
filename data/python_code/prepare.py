"""
Prepare Python code dataset for CodeGPT training.

This script collects Python source files from local directories and/or
downloads Python code from public datasets, tokenizes them, and saves
as binary files for training.

Sources (in order of priority):
  1. Local Python files from specified directories
  2. HuggingFace 'codeparrot/codeparrot-clean' dataset (optional)

Usage:
    # From local Python files
    python data/python_code/prepare.py --source=local --code_dir=/path/to/python/projects

    # From HuggingFace dataset
    python data/python_code/prepare.py --source=huggingface

    # Both
    python data/python_code/prepare.py --source=both --code_dir=/path/to/projects
"""

import os
import sys
import glob
import random
import argparse
import pickle
import numpy as np

# add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from tokenizer import CodeTokenizer, SPECIAL_TOKENS, VOCAB_SIZE, tokenize_code_file


def collect_python_files(root_dirs, max_file_size=100_000, min_file_size=100):
    """
    Recursively collect Python files from directories.

    Filters:
      - Skip files larger than max_file_size bytes
      - Skip files smaller than min_file_size bytes
      - Skip test files and __pycache__
      - Skip files that can't be decoded as UTF-8
    """
    files = []
    for root_dir in root_dirs:
        root_dir = os.path.expanduser(root_dir)
        for filepath in glob.glob(os.path.join(root_dir, '**', '*.py'), recursive=True):
            # skip common non-useful paths
            if '__pycache__' in filepath or '.egg-info' in filepath:
                continue
            if '/test' in filepath.lower() and '_test' not in os.path.basename(filepath).lower():
                pass  # keep test files, they're useful code examples

            try:
                size = os.path.getsize(filepath)
                if min_file_size <= size <= max_file_size:
                    # verify it's valid UTF-8
                    with open(filepath, 'r', encoding='utf-8') as f:
                        f.read(100)
                    files.append(filepath)
            except (OSError, UnicodeDecodeError):
                continue

    return files


def tokenize_files(files, tokenizer):
    """Tokenize a list of code files."""
    all_tokens = []
    for i, filepath in enumerate(files):
        if i % 1000 == 0:
            print(f"  tokenizing {i}/{len(files)} files...")
        try:
            tokens = tokenize_code_file(filepath, tokenizer)
            all_tokens.extend(tokens)
        except Exception as e:
            print(f"  skipping {filepath}: {e}")
            continue
    return all_tokens


def tokenize_texts(texts, tokenizer, lang='python'):
    """Tokenize a list of code strings."""
    all_tokens = []
    for i, text in enumerate(texts):
        if i % 10000 == 0:
            print(f"  tokenizing {i}/{len(texts)} texts...")
        try:
            tokens = tokenizer.encode(text, lang=lang, add_code_boundaries=True)
            tokens.append(SPECIAL_TOKENS["<|endoftext|>"])
            all_tokens.extend(tokens)
        except Exception:
            continue
    return all_tokens


def download_huggingface_data(max_samples=100000):
    """Download Python code from HuggingFace codeparrot dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install 'datasets' package: pip install datasets")
        sys.exit(1)

    print("Downloading codeparrot-clean dataset from HuggingFace...")
    print("(This may take a while on first run)")

    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    texts = []
    for i, example in enumerate(ds):
        if i >= max_samples:
            break
        content = example.get('content', '')
        if len(content) > 100 and len(content) < 100000:
            texts.append(content)
        if i % 10000 == 0:
            print(f"  collected {len(texts)} samples ({i} checked)...")

    print(f"Collected {len(texts)} Python code samples")
    return texts


def main():
    parser = argparse.ArgumentParser(description='Prepare Python code dataset for CodeGPT')
    parser.add_argument('--source', type=str, default='local',
                        choices=['local', 'huggingface', 'both'],
                        help='Data source')
    parser.add_argument('--code_dir', type=str, nargs='+',
                        default=[os.path.expanduser('~/PyPro')],
                        help='Directories containing Python code')
    parser.add_argument('--max_samples', type=int, default=100000,
                        help='Max samples from HuggingFace')
    parser.add_argument('--val_ratio', type=float, default=0.05,
                        help='Validation split ratio')
    args = parser.parse_args()

    tokenizer = CodeTokenizer()
    all_tokens = []
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Local files ---
    if args.source in ('local', 'both'):
        print(f"Collecting Python files from: {args.code_dir}")
        files = collect_python_files(args.code_dir)
        random.shuffle(files)
        print(f"Found {len(files)} Python files")

        if files:
            print("Tokenizing local files...")
            tokens = tokenize_files(files, tokenizer)
            all_tokens.extend(tokens)
            print(f"  -> {len(tokens):,} tokens from local files")

    # --- HuggingFace ---
    if args.source in ('huggingface', 'both'):
        texts = download_huggingface_data(max_samples=args.max_samples)
        print("Tokenizing HuggingFace data...")
        tokens = tokenize_texts(texts, tokenizer, lang='python')
        all_tokens.extend(tokens)
        print(f"  -> {len(tokens):,} tokens from HuggingFace")

    if not all_tokens:
        print("ERROR: No tokens collected! Check your data source settings.")
        sys.exit(1)

    print(f"\nTotal tokens: {len(all_tokens):,}")

    # shuffle at document level (split by EOT tokens)
    eot = SPECIAL_TOKENS["<|endoftext|>"]
    documents = []
    current_doc = []
    for t in all_tokens:
        current_doc.append(t)
        if t == eot:
            documents.append(current_doc)
            current_doc = []
    if current_doc:
        documents.append(current_doc)

    random.shuffle(documents)

    # split into train/val
    n_val = max(1, int(len(documents) * args.val_ratio))
    val_docs = documents[:n_val]
    train_docs = documents[n_val:]

    train_tokens = [t for doc in train_docs for t in doc]
    val_tokens = [t for doc in val_docs for t in doc]

    print(f"Train: {len(train_tokens):,} tokens ({len(train_docs)} documents)")
    print(f"Val:   {len(val_tokens):,} tokens ({len(val_docs)} documents)")

    # save as binary files
    train_ids = np.array(train_tokens, dtype=np.uint16)
    val_ids = np.array(val_tokens, dtype=np.uint16)

    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))

    # save metadata
    meta = {
        'vocab_size': VOCAB_SIZE,
        'special_tokens': SPECIAL_TOKENS,
        'num_train_tokens': len(train_tokens),
        'num_val_tokens': len(val_tokens),
        'num_train_docs': len(train_docs),
        'num_val_docs': len(val_docs),
    }
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"\nSaved to {output_dir}:")
    print(f"  train.bin: {os.path.getsize(os.path.join(output_dir, 'train.bin')) / 1e6:.1f} MB")
    print(f"  val.bin:   {os.path.getsize(os.path.join(output_dir, 'val.bin')) / 1e6:.1f} MB")
    print(f"  meta.pkl")
    print("Done!")


if __name__ == '__main__':
    main()
