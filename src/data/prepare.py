"""
Data preparation: load parquet directory → tokenize → cache as .pt files.
"""

import hashlib
import os
from pathlib import Path

import pyarrow.parquet as pq
import torch

from src.data.tokenizer import Tokenizer


CACHE_DIR = Path.home() / ".cache" / "autoresearch" / "custom"


def _hash_parquet_dir(parquet_dir: str) -> str:
    """Hash parquet directory contents for cache invalidation."""
    h = hashlib.sha256()
    p = Path(parquet_dir)
    for f in sorted(p.rglob("*.parquet")):
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
    return h.hexdigest()[:16]


def load_parquet_texts(parquet_dir: str, text_column: str = "text") -> list[str]:
    """Load all text from a directory of parquet files."""
    texts = []
    p = Path(parquet_dir)
    files = sorted(p.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {parquet_dir}")
    print(f"Loading {len(files)} parquet files from {parquet_dir}")
    for f in files:
        table = pq.read_table(f, columns=[text_column])
        texts.extend(table[text_column].to_pylist())
    print(f"Loaded {len(texts):,} documents")
    return texts


def tokenize_documents(texts: list[str], tokenizer: Tokenizer) -> list[list[int]]:
    """Tokenize all documents, prepending BOS to each."""
    docs = []
    bos = tokenizer.bos_token
    for text in texts:
        tokens = tokenizer.encode(text)
        if tokens:
            docs.append([bos] + tokens)
    return docs


def prepare_data(parquet_dir: str, text_column: str = "text",
                 val_ratio: float = 0.1, seed: int = 42) -> dict:
    """Full data preparation pipeline.

    Returns dict with 'train_docs', 'val_docs', 'tokenizer', and cache paths.
    """
    tokenizer = Tokenizer()

    # Check cache
    dir_hash = _hash_parquet_dir(parquet_dir)
    cache_path = CACHE_DIR / dir_hash
    train_cache = cache_path / "train_docs.pt"
    val_cache = cache_path / "val_docs.pt"

    if train_cache.exists() and val_cache.exists():
        print(f"Loading cached tokenized data from {cache_path}")
        train_docs = torch.load(train_cache, weights_only=True)
        val_docs = torch.load(val_cache, weights_only=True)
    else:
        texts = load_parquet_texts(parquet_dir, text_column)

        print("Tokenizing documents...")
        docs = tokenize_documents(texts, tokenizer)
        total_tokens = sum(len(d) for d in docs)
        print(f"Total tokens: {total_tokens:,} across {len(docs):,} documents")

        # Split by document (deterministic)
        import random
        rng = random.Random(seed)
        indices = list(range(len(docs)))
        rng.shuffle(indices)

        val_count = max(1, int(len(docs) * val_ratio))
        val_indices = set(indices[:val_count])

        train_docs = [docs[i] for i in range(len(docs)) if i not in val_indices]
        val_docs = [docs[i] for i in range(len(docs)) if i in val_indices]

        train_tokens = sum(len(d) for d in train_docs)
        val_tokens = sum(len(d) for d in val_docs)
        print(f"Train: {len(train_docs):,} docs, {train_tokens:,} tokens")
        print(f"Val:   {len(val_docs):,} docs, {val_tokens:,} tokens")

        # Cache
        cache_path.mkdir(parents=True, exist_ok=True)
        torch.save(train_docs, train_cache)
        torch.save(val_docs, val_cache)
        print(f"Cached to {cache_path}")

    return {
        "train_docs": train_docs,
        "val_docs": val_docs,
        "tokenizer": tokenizer,
    }
