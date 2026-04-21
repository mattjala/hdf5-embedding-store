"""Interactive demo: compare HDF5 embedding store backends side-by-side.

Loads a Wikipedia corpus cache (prepared by scripts/prepare_corpus.py), ingests
a configurable subset into four backends, then lets you pick a query and see
search results and timing from all backends at once.

Usage
-----
    # Default: use results/wikipedia_corpus.h5, ingest first 5k passages
    python demo/run.py

    # Smaller subset for a quick spin
    python demo/run.py --n 1000

    # Point at a different corpus
    python demo/run.py --corpus path/to/corpus.h5

    # Use a synthetic corpus if you haven't run prepare_corpus.py yet
    python demo/run.py --synthetic

Requirements
------------
    pip install -e ".[blosc]"   (core package + Blosc backend)
    results/wikipedia_corpus.h5 (from scripts/prepare_corpus.py)
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hdf5_embedding_store import (
    HDF5CachedStore,
    HDF5EmbeddingStore,
    HDF5Int8Store,
    NumpyEmbeddingStore,
    SQLiteEmbeddingStore,
)

DEFAULT_CORPUS = ROOT / "results" / "wikipedia_corpus.h5"
DEFAULT_N = 5_000
INGEST_BATCH = 1_000
TOP_K = 5
PASSAGE_PREVIEW = 160   # chars shown per result

# ---------------------------------------------------------------------------
# Token counting (optional — falls back gracefully)
# ---------------------------------------------------------------------------

def _make_token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        def count(text: str) -> int:
            return len(enc.encode(text))
        return count
    except ImportError:
        def count(text: str) -> int:
            return len(text) // 4  # rough estimate
        return count

count_tokens = _make_token_counter()


def result_tokens(results) -> int:
    total = 0
    for r in results:
        total += count_tokens(f"{r.id} {r.text} {r.score:.4f}")
    return total


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

@dataclass
class Corpus:
    vectors: np.ndarray      # (N, D) float32, L2-normalised
    texts: list[str]
    ids: list[str]
    query_vectors: np.ndarray   # (Q, D)
    query_texts: list[str]
    dim: int
    source: str


def load_corpus(path: Path, n: int) -> Corpus:
    with h5py.File(path, "r") as f:
        total = f["documents/vectors"].shape[0]
        n = min(n, total)
        vecs = f["documents/vectors"][:n].astype(np.float32)
        texts = [v.decode() if isinstance(v, bytes) else v for v in f["documents/texts"][:n]]
        ids = [v.decode() if isinstance(v, bytes) else v for v in f["documents/ids"][:n]]
        q_vecs = f["queries/vectors"][:].astype(np.float32)
        q_texts = [v.decode() if isinstance(v, bytes) else v for v in f["queries/texts"][:]]
        source = f["documents"].attrs.get("source", str(path))
        dim = vecs.shape[1]
    return Corpus(vecs, texts, ids, q_vecs, q_texts, dim, source)


def synthetic_corpus(n: int, dim: int = 768) -> Corpus:
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms
    texts = [f"Synthetic passage {i}: the quick brown fox jumps over the lazy dog." for i in range(n)]
    ids = [f"syn_{i}" for i in range(n)]
    q_vecs = rng.standard_normal((5, dim)).astype(np.float32)
    q_norms = np.linalg.norm(q_vecs, axis=1, keepdims=True)
    q_vecs /= q_norms
    q_texts = [f"Synthetic query {i}" for i in range(5)]
    return Corpus(vecs, texts, ids, q_vecs, q_texts, dim, "synthetic (random unit vectors)")


# ---------------------------------------------------------------------------
# Backend setup
# ---------------------------------------------------------------------------

@dataclass
class Backend:
    label: str
    store_factory: Callable
    path_suffix: str


def get_backends(tmpdir: Path, dim: int) -> list[Backend]:
    backends = [
        Backend(
            "HDF5 (cached)",
            lambda p: HDF5CachedStore(p, dimension=dim, compression=None),
            "cached.h5",
        ),
        Backend(
            "HDF5 (int8)",
            lambda p: HDF5Int8Store(p, dimension=dim),
            "int8.h5",
        ),
        Backend(
            "HDF5 (gzip-4)",
            lambda p: HDF5EmbeddingStore(p, dimension=dim, compression="gzip", compression_opts=4),
            "gzip.h5",
        ),
        Backend(
            "NumPy .npy",
            lambda p: NumpyEmbeddingStore(p, dimension=dim),
            "numpy",
        ),
        Backend(
            "SQLite BLOB",
            lambda p: SQLiteEmbeddingStore(p, dimension=dim),
            "sqlite.db",
        ),
    ]
    return backends


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest_all(backends: list[Backend], corpus: Corpus, tmpdir: Path) -> dict[str, object]:
    stores = {}
    col_w = max(len(b.label) for b in backends) + 2
    print(f"\n  {'Backend':<{col_w}}  {'eps':>8}  {'size':>8}")
    print(f"  {'-'*col_w}  {'-'*8}  {'-'*8}")

    for b in backends:
        path = tmpdir / b.path_suffix
        store = b.store_factory(path)
        n = len(corpus.ids)
        t0 = time.perf_counter()
        for start in range(0, n, INGEST_BATCH):
            end = min(start + INGEST_BATCH, n)
            store.insert(
                corpus.ids[start:end],
                corpus.texts[start:end],
                corpus.vectors[start:end],
            )
        elapsed = time.perf_counter() - t0
        eps = n / elapsed if elapsed > 0 else 0

        # size on disk (sum all files with this prefix)
        size_bytes = sum(
            p.stat().st_size for p in tmpdir.iterdir()
            if p.name.startswith(path.name) or p.stem == path.name
        )
        size_mb = size_bytes / 1e6

        stores[b.label] = store
        print(f"  {b.label:<{col_w}}  {eps:>7,.0f}  {size_mb:>6.1f} MB")

    return stores


# ---------------------------------------------------------------------------
# Search and display
# ---------------------------------------------------------------------------

def run_query(stores: dict[str, object], query_vec: np.ndarray, query_text: str) -> None:
    col_w = max(len(k) for k in stores) + 2

    print(f"\n  Query: \"{query_text}\"")
    print()

    for label, store in stores.items():
        t0 = time.perf_counter()
        results = store.search(query_vec, k=TOP_K)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        tokens = result_tokens(results)

        print(f"  ┌─ {label}  [{elapsed_ms:.1f} ms  {tokens} tokens]")
        for i, r in enumerate(results[:3]):
            preview = r.text.replace("\n", " ")[:PASSAGE_PREVIEW]
            if len(r.text) > PASSAGE_PREVIEW:
                preview += "…"
            print(f"  │  {i+1}. [{r.score:.3f}] {preview}")
        if len(results) > 3:
            print(f"  │  … +{len(results)-3} more")
        print(f"  └{'─'*60}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive HDF5 embedding store backend comparison demo",
    )
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                   help=f"HDF5 corpus cache from scripts/prepare_corpus.py (default: {DEFAULT_CORPUS})")
    p.add_argument("--n", type=int, default=DEFAULT_N,
                   help=f"Number of passages to ingest (default: {DEFAULT_N})")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic random corpus instead of Wikipedia")
    return p.parse_args()


def main():
    args = parse_args()

    # Load corpus
    if args.synthetic:
        print("Generating synthetic corpus ...")
        corpus = synthetic_corpus(args.n)
    elif args.corpus.exists():
        print(f"Loading corpus from {args.corpus} ...")
        corpus = load_corpus(args.corpus, args.n)
    else:
        print(f"Corpus not found at {args.corpus}.")
        print("Run:  python scripts/prepare_corpus.py")
        print("Or:   python demo/run.py --synthetic")
        sys.exit(1)

    print(f"  {len(corpus.ids):,} passages  |  dim={corpus.dim}  |  source: {corpus.source}")

    with tempfile.TemporaryDirectory(prefix="hdf5_demo_") as tmpdir:
        backends = get_backends(Path(tmpdir), corpus.dim)

        # Ingest
        print(f"\nIngesting {len(corpus.ids):,} passages into {len(backends)} backends ...")
        stores = ingest_all(backends, corpus, Path(tmpdir))

        # Query loop
        queries = list(zip(corpus.query_texts, corpus.query_vectors))
        print(f"\n{'='*64}")
        print("  Available queries:")
        for i, (text, _) in enumerate(queries):
            print(f"    {i+1:2d}. {text}")
        print(f"{'='*64}")
        print("  Enter a query number, or 'q' to quit.")

        while True:
            try:
                raw = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if raw.lower() in ("q", "quit", "exit"):
                break

            try:
                idx = int(raw) - 1
                if not 0 <= idx < len(queries):
                    raise ValueError
            except ValueError:
                print(f"  Enter a number between 1 and {len(queries)}, or 'q' to quit.")
                continue

            query_text, query_vec = queries[idx]
            run_query(stores, query_vec, query_text)

        # Close stores
        for store in stores.values():
            store.close()

    print("Bye.")


if __name__ == "__main__":
    main()
