"""
Prepare multi-language code dataset from GitHub / HuggingFace sources.

Downloads code from the 'bigcode/the-stack-dedup' dataset (subset)
supporting multiple programming languages.

Usage:
    python data/github_code/prepare.py --langs python javascript typescript
    python data/github_code/prepare.py --langs python --max_samples=50000
"""

import os
import sys
import random
import argparse
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from tokenizer import CodeTokenizer, SPECIAL_TOKENS, VOCAB_SIZE, EXT_TO_LANG

# language name to Stack dataset language mapping
LANG_MAP = {
    'python': 'python',
    'javascript': 'javascript',
    'typescript': 'typescript',
    'java': 'java',
    'c': 'c',
    'cpp': 'c++',
    'go': 'go',
    'rust': 'rust',
    'ruby': 'ruby',
    'php': 'php',
    'shell': 'shell',
}


def download_stack_data(languages, max_samples_per_lang=20000):
    """Download code from The Stack dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install 'datasets' package: pip install datasets")
        sys.exit(1)

    all_samples = []

    for lang in languages:
        stack_lang = LANG_MAP.get(lang, lang)
        print(f"\nDownloading {lang} ({stack_lang}) from The Stack...")

        try:
            ds = load_dataset(
                "bigcode/the-stack-dedup",
                data_dir=f"data/{stack_lang}",
                split="train",
                streaming=True,
            )

            count = 0
            for example in ds:
                content = example.get('content', '')
                if 100 < len(content) < 50000:
                    all_samples.append({
                        'content': content,
                        'lang': lang,
                    })
                    count += 1
                    if count >= max_samples_per_lang:
                        break
                    if count % 5000 == 0:
                        print(f"  collected {count} {lang} samples...")

            print(f"  -> {count} {lang} samples collected")

        except Exception as e:
            print(f"  Failed to download {lang}: {e}")
            print(f"  Skipping {lang}...")
            continue

    return all_samples


def main():
    parser = argparse.ArgumentParser(description='Prepare multi-language code dataset')
    parser.add_argument('--langs', nargs='+', default=['python', 'javascript', 'typescript'],
                        help='Languages to include')
    parser.add_argument('--max_samples', type=int, default=20000,
                        help='Max samples per language')
    parser.add_argument('--val_ratio', type=float, default=0.05,
                        help='Validation split ratio')
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer = CodeTokenizer()

    print(f"Languages: {args.langs}")
    print(f"Max samples per language: {args.max_samples}")

    samples = download_stack_data(args.langs, max_samples_per_lang=args.max_samples)

    if not samples:
        print("ERROR: No samples collected!")
        sys.exit(1)

    print(f"\nTotal samples: {len(samples)}")
    print("Tokenizing...")

    # tokenize all samples
    documents = []
    for i, sample in enumerate(samples):
        if i % 10000 == 0:
            print(f"  tokenizing {i}/{len(samples)}...")
        try:
            tokens = tokenizer.encode(
                sample['content'],
                lang=sample['lang'],
                add_code_boundaries=True,
            )
            tokens.append(SPECIAL_TOKENS["<|endoftext|>"])
            documents.append(tokens)
        except Exception:
            continue

    random.shuffle(documents)

    # split
    n_val = max(1, int(len(documents) * args.val_ratio))
    val_docs = documents[:n_val]
    train_docs = documents[n_val:]

    train_tokens = [t for doc in train_docs for t in doc]
    val_tokens = [t for doc in val_docs for t in doc]

    print(f"\nTrain: {len(train_tokens):,} tokens ({len(train_docs)} documents)")
    print(f"Val:   {len(val_tokens):,} tokens ({len(val_docs)} documents)")

    # save
    np.array(train_tokens, dtype=np.uint16).tofile(os.path.join(output_dir, 'train.bin'))
    np.array(val_tokens, dtype=np.uint16).tofile(os.path.join(output_dir, 'val.bin'))

    meta = {
        'vocab_size': VOCAB_SIZE,
        'special_tokens': SPECIAL_TOKENS,
        'languages': args.langs,
        'num_train_tokens': len(train_tokens),
        'num_val_tokens': len(val_tokens),
    }
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"\nSaved to {output_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
