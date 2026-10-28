"""Tokenize TinyStories into flat uint16 token files (nanoGPT style) and sample batches.

python -m tinylm.data --out /mnt/group/data            # full corpus, once
python -m tinylm.data --out data --limit 2000          # small local sample
"""
import argparse, os
import numpy as np

from tinylm.configs import EOS

DATASET = "roneneldan/TinyStories"


def encoder():
    import tiktoken
    return tiktoken.get_encoding("gpt2")  # same BPE as the GPT-Neo tokenizer the checkpoints use


def tokenize(texts, enc, chunk=10_000):
    """Stories -> one flat uint16 array, each story followed by EOS."""
    parts = []
    for i in range(0, len(texts), chunk):
        for ids in enc.encode_ordinary_batch(texts[i:i + chunk]):
            parts.append(np.array(ids + [EOS], dtype=np.uint16))
    return np.concatenate(parts)


def build(out, limit=None):
    from datasets import load_dataset
    os.makedirs(out, exist_ok=True)
    enc = encoder()
    for split in ("train", "validation"):
        ds = load_dataset(DATASET, split=split if limit is None else f"{split}[:{limit}]")
        toks = tokenize(ds["text"], enc)
        toks.tofile(os.path.join(out, f"{split}.bin"))
        print(f"{split}: {len(ds):,} stories, {len(toks):,} tokens")


def batches(path, batch_size, block_size, seed=0):
    """Endless random windows of block_size+1 tokens, shape (B, T+1) int32.
    AR models train on x[:, :-1] -> x[:, 1:]; diffusion models use x[:, :-1]."""
    data = np.memmap(path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    while True:
        idx = rng.integers(0, len(data) - block_size - 1, batch_size)
        yield np.stack([data[i:i + block_size + 1] for i in idx]).astype(np.int32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--limit", type=int, help="stories per split, for a quick local test")
    a = ap.parse_args()
    build(a.out, a.limit)
