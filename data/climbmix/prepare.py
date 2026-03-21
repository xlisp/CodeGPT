"""
Prepare climbmix-400b-shuffle dataset for CodeGPT training.

Downloads parquet shards from karpathy/climbmix-400b-shuffle on HuggingFace,
tokenizes with GPT-2 tiktoken (matching CodeGPT's base vocab), and saves
as train.bin / val.bin / meta.pkl compatible with CodeGPT's data loader.

Data source: autoresearch project (/home/xlisp/PyPro/autoresearch)

Usage:
    # Download 2 training shards (~few GB, good for initial testing)
    ~/miniconda3/envs/codegpt/bin/python data/climbmix/prepare.py --num-shards 2

    # Download 10 shards (default)
    ~/miniconda3/envs/codegpt/bin/python data/climbmix/prepare.py

    # Download all 6542 shards (huge, ~TBs)
    ~/miniconda3/envs/codegpt/bin/python data/climbmix/prepare.py --num-shards -1
"""

import os
import sys
import time
import argparse
import pickle
from multiprocessing import Pool

import requests
import numpy as np
import pyarrow.parquet as pq
import tiktoken

# ---------------------------------------------------------------------------
# Constants (same as autoresearch/prepare.py)
# ---------------------------------------------------------------------------

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD  # pinned validation shard (shard_06542)
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "climbmix")
DATA_DIR = os.path.join(CACHE_DIR, "data")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# GPT-2 tokenizer — matches CodeGPT's base vocab (50257 tokens)
VOCAB_SIZE = 50257


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_single_shard(index):
    """Download one parquet shard with retries. Returns True on success."""
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True

    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            size_mb = os.path.getsize(filepath) / 1e6
            print(f"  Downloaded {filename} ({size_mb:.1f} MB)")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data(num_shards, download_workers=4):
    """Download training shards + pinned validation shard."""
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)

    existing = sum(1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")))
    if existing == len(ids):
        print(f"All {len(ids)} shards already downloaded at {DATA_DIR}")
        return

    needed = len(ids) - existing
    print(f"Downloading {needed} shards ({existing} already exist) to {DATA_DIR} ...")

    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, ids)

    ok = sum(1 for r in results if r)
    print(f"{ok}/{len(ids)} shards ready")


# ---------------------------------------------------------------------------
# Tokenize & save
# ---------------------------------------------------------------------------

def list_parquet_files():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def tokenize_shard(parquet_path, enc):
    """Read one parquet shard and return flat token list (uint16)."""
    tokens = []
    eot = enc.eot_token  # 50256 for gpt2
    pf = pq.ParquetFile(parquet_path)
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        texts = rg.column("text").to_pylist()
        for text in texts:
            if not text or not text.strip():
                continue
            ids = enc.encode_ordinary(text)
            tokens.extend(ids)
            tokens.append(eot)
    return tokens


def build_dataset(num_shards):
    """Tokenize shards and write train.bin / val.bin / meta.pkl."""
    enc = tiktoken.get_encoding("gpt2")
    parquet_files = list_parquet_files()
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)

    train_files = [p for p in parquet_files if p != val_path]
    train_files = train_files[:num_shards]

    if not train_files:
        print("ERROR: No training shards found. Run download first.")
        sys.exit(1)
    if not os.path.exists(val_path):
        print(f"ERROR: Validation shard not found at {val_path}")
        sys.exit(1)

    print(f"Tokenizing {len(train_files)} training shard(s)...")
    train_tokens = []
    for i, path in enumerate(train_files):
        t0 = time.time()
        toks = tokenize_shard(path, enc)
        train_tokens.extend(toks)
        print(f"  [{i+1}/{len(train_files)}] {os.path.basename(path)}: {len(toks):,} tokens ({time.time()-t0:.1f}s)")

    print(f"Tokenizing validation shard...")
    t0 = time.time()
    val_tokens = tokenize_shard(val_path, enc)
    print(f"  {os.path.basename(val_path)}: {len(val_tokens):,} tokens ({time.time()-t0:.1f}s)")

    print(f"\nTotal train tokens: {len(train_tokens):,}")
    print(f"Total val tokens:   {len(val_tokens):,}")

    # Save as uint16 binary (GPT-2 has 50257 tokens, fits in uint16)
    train_ids = np.array(train_tokens, dtype=np.uint16)
    val_ids = np.array(val_tokens, dtype=np.uint16)

    train_path = os.path.join(OUTPUT_DIR, "train.bin")
    val_path_out = os.path.join(OUTPUT_DIR, "val.bin")
    train_ids.tofile(train_path)
    val_ids.tofile(val_path_out)

    meta = {
        'vocab_size': VOCAB_SIZE,
        'num_train_tokens': len(train_tokens),
        'num_val_tokens': len(val_tokens),
        'num_train_shards': len(train_files),
        'source': 'karpathy/climbmix-400b-shuffle',
        'tokenizer': 'gpt2',
    }
    with open(os.path.join(OUTPUT_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    train_mb = os.path.getsize(train_path) / 1e6
    val_mb = os.path.getsize(val_path_out) / 1e6
    print(f"\nSaved to {OUTPUT_DIR}:")
    print(f"  train.bin: {train_mb:.1f} MB  ({len(train_tokens):,} tokens)")
    print(f"  val.bin:   {val_mb:.1f} MB  ({len(val_tokens):,} tokens)")
    print(f"  meta.pkl")
    print("Done!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare climbmix data for CodeGPT")
    parser.add_argument("--num-shards", type=int, default=10,
                        help="Number of training shards to use (-1 = all 6542). Val shard always included.")
    parser.add_argument("--download-workers", type=int, default=4,
                        help="Parallel download workers")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, only tokenize already-downloaded shards")
    args = parser.parse_args()

    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    if not args.skip_download:
        download_data(num_shards, download_workers=args.download_workers)
        print()

    build_dataset(num_shards)
