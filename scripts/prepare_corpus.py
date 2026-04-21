"""Download a Wikipedia passage corpus and embed it with nomic-embed-text-v2-moe.

Saves to an HDF5 cache file consumed by benchmark/run.py --corpus.

Usage
-----
    # Dry run: show what would happen, download nothing
    python scripts/prepare_corpus.py --dry-run

    # Default: 100k passages, saved to results/wikipedia_corpus.h5
    python scripts/prepare_corpus.py

    # Custom size / output
    python scripts/prepare_corpus.py --n-passages 50000 --out results/wiki_50k.h5

    # GPU embedding (much faster)
    python scripts/prepare_corpus.py --device cuda

This script is idempotent: if the output file already exists with the
correct number of passages, it exits immediately without re-embedding.

Requirements (install separately, not in core deps)
-----------------------------------------------------
    pip install datasets sentence-transformers

Output HDF5 schema
------------------
/
  documents/
    vectors   float32 (N, D)   L2-normalised embeddings
    texts     vlen str  (N,)
    ids       vlen str  (N,)    "wiki_{article_id}_{passage_idx}"
    attrs: model, dimension, n_passages, source, created_utc

  queries/
    vectors   float32 (Q, D)   embeddings for benchmark queries
    texts     vlen str  (Q,)   the raw query strings
    attrs: model, n_queries

nomic task prefixes
-------------------
nomic-embed-text-v2-moe requires task-specific prefixes:
    documents → "search_document: <text>"
    queries   → "search_query: <text>"
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed benchmark queries — used by benchmark/run.py as the search probe.
# Chosen to span different Wikipedia topic areas so retrieved passages
# have genuinely variable length and content.
# ---------------------------------------------------------------------------
BENCHMARK_QUERIES = [
    "How does photosynthesis work in plants?",
    "What were the main causes of World War I?",
    "Explain the structure of the human immune system.",
    "What is the theory of general relativity?",
    "How do vaccines provide immunity against disease?",
    "What is the significance of the Magna Carta?",
    "How does machine learning differ from traditional programming?",
    "What caused the extinction of the dinosaurs?",
    "Describe the water cycle and its importance.",
    "What are the fundamental principles of democracy?",
]

MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
DEFAULT_OUT = Path("results/wikipedia_corpus.h5")
DEFAULT_N = 100_000
PASSAGE_MIN_WORDS = 40
PASSAGE_MAX_WORDS = 250
EMBED_BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Passage extraction from Wikipedia articles
# ---------------------------------------------------------------------------

def _split_passages(text: str, max_words: int = PASSAGE_MAX_WORDS) -> list[str]:
    """Split article body into paragraphs, filter stubs, cap length."""
    passages = []
    for para in text.split("\n\n"):
        para = para.strip()
        words = para.split()
        if len(words) < PASSAGE_MIN_WORDS:
            continue
        if len(words) <= max_words:
            passages.append(para)
        else:
            # Chunk long paragraphs by sentence boundary approximation
            sentences, buf = [], []
            for sent in para.replace(". ", ".\n").split("\n"):
                buf.append(sent.strip())
                if sum(len(s.split()) for s in buf) >= max_words // 2:
                    chunk = " ".join(buf).strip()
                    if len(chunk.split()) >= PASSAGE_MIN_WORDS:
                        passages.append(chunk)
                    buf = []
            if buf:
                chunk = " ".join(buf).strip()
                if len(chunk.split()) >= PASSAGE_MIN_WORDS:
                    passages.append(chunk)
    return passages


def stream_wikipedia_passages(n: int):
    """Yield (article_id, passage_idx, passage_text) up to n passages."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets not installed: pip install datasets")

    print(f"Streaming wikimedia/wikipedia (English, 20231101) — collecting {n:,} passages ...")
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
        trust_remote_code=False,
    )

    collected = 0
    for article in ds:
        article_id = article["id"]
        passages = _split_passages(article["text"])
        for pi, passage in enumerate(passages):
            yield article_id, pi, passage
            collected += 1
            if collected >= n:
                return


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    prefix: str,
    device: str,
    batch_size: int,
    model_id: str = MODEL_ID,
    max_seq_length: int = 128,
) -> "np.ndarray":
    """Embed a list of texts with the nomic model, return (N, D) float32 array.

    max_seq_length caps the token length fed to the transformer.  Wikipedia
    passages can reach 300+ tokens; without a cap each batch pads to the longest
    sequence, making attention O(max_len²) and blowing up CPU time.  128 tokens
    (~90 words) is a standard RAG chunk size and is lossless for most retrieval.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers not installed: pip install sentence-transformers")

    import numpy as np

    print(f"Loading {model_id} (first call downloads model weights) ...")
    model = SentenceTransformer(model_id, trust_remote_code=True, device=device)
    model.max_seq_length = max_seq_length
    print(f"  max_seq_length={max_seq_length}, dim={model.get_sentence_embedding_dimension()}")

    prefixed = [prefix + t for t in texts]
    print(f"Embedding {len(prefixed):,} texts on {device} (batch_size={batch_size}) ...")
    t0 = time.perf_counter()
    vecs = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.0f}s  ({len(prefixed) / elapsed:,.0f} texts/sec)")
    return vecs.astype("float32")


# ---------------------------------------------------------------------------
# HDF5 cache write / check
# ---------------------------------------------------------------------------

def cache_is_complete(out: Path, n: int) -> bool:
    """Return True if the cache file exists and has exactly n passages."""
    if not out.exists():
        return False
    try:
        import h5py
        with h5py.File(out, "r") as f:
            stored = int(f["documents"].attrs.get("n_passages", 0))
            return stored == n
    except Exception:
        return False


def write_cache(
    out: Path,
    doc_ids: list[str],
    doc_texts: list[str],
    doc_vecs,      # np.ndarray (N, D)
    query_texts: list[str],
    query_vecs,    # np.ndarray (Q, D)
    model_id: str = MODEL_ID,
) -> None:
    import h5py
    import numpy as np
    from datetime import datetime, timezone

    out.parent.mkdir(parents=True, exist_ok=True)
    dim = doc_vecs.shape[1]
    vlen_str = h5py.string_dtype(encoding="utf-8")

    print(f"\nWriting corpus cache → {out}")
    with h5py.File(out, "w") as f:
        # Documents
        dg = f.create_group("documents")
        dg.create_dataset("vectors", data=doc_vecs, chunks=(512, dim), compression="gzip", compression_opts=4)
        dg.create_dataset("texts", data=np.array(doc_texts, dtype=object), dtype=vlen_str)
        dg.create_dataset("ids",   data=np.array(doc_ids,   dtype=object), dtype=vlen_str)
        dg.attrs["model"] = model_id
        dg.attrs["dimension"] = dim
        dg.attrs["n_passages"] = len(doc_ids)
        dg.attrs["source"] = "wikimedia/wikipedia 20231101.en"
        dg.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()

        # Queries
        qg = f.create_group("queries")
        qg.create_dataset("vectors", data=query_vecs)
        qg.create_dataset("texts", data=np.array(query_texts, dtype=object), dtype=vlen_str)
        qg.attrs["model"] = model_id
        qg.attrs["n_queries"] = len(query_texts)

    print(f"Saved {len(doc_ids):,} passages + {len(query_texts)} queries "
          f"({out.stat().st_size / 1e6:.0f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare Wikipedia corpus cache for benchmark/run.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        This script has two expensive steps that require network/compute:
          1. Downloading Wikipedia (~20 GB stream, exits after --n-passages)
          2. Embedding with nomic-embed-text-v2-moe (~1 GB model download,
             then ~20–40 min CPU or ~5 min GPU for 100k passages)

        Use --dry-run to preview what will happen without doing either.
        The output file is idempotent: re-running with the same --out and
        --n-passages is a no-op if the cache is already complete.
        """),
    )
    p.add_argument("--n-passages", type=int, default=DEFAULT_N,
                   help=f"Number of Wikipedia passages to collect (default: {DEFAULT_N:,})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output HDF5 cache path (default: {DEFAULT_OUT})")
    p.add_argument("--model", default=MODEL_ID,
                   help=f"Sentence-transformers model ID (default: {MODEL_ID}). "
                        "nomic-embed-text-v1.5 is ~6x faster on CPU than v2-moe.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                   help="Device for sentence-transformers (default: cpu)")
    p.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE,
                   help=f"Embedding batch size (default: {EMBED_BATCH_SIZE})")
    p.add_argument("--max-seq-len", type=int, default=128,
                   help="Max token length fed to the transformer. Caps per-batch padding "
                        "overhead (O(n²) attention). 128 is ~90 words, standard for RAG. "
                        "(default: 128)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen and exit without downloading anything")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Output path  : {args.out.resolve()}")
    print(f"Passages     : {args.n_passages:,}")
    print(f"Model        : {args.model}")
    print(f"Device       : {args.device}")
    print(f"Queries      : {len(BENCHMARK_QUERIES)}")

    if args.dry_run:
        print("\n[dry-run] Would download wikimedia/wikipedia (streaming) and")
        print(f"          embed {args.n_passages:,} passages + {len(BENCHMARK_QUERIES)} queries.")
        print("          Pass --dry-run=false or omit flag to proceed.")
        return

    if cache_is_complete(args.out, args.n_passages):
        print(f"\nCache already complete ({args.n_passages:,} passages). Nothing to do.")
        print(f"Delete {args.out} and re-run to regenerate.")
        return

    # Step 1: Collect passages
    print("\n--- Step 1: Collect passages from Wikipedia ---")
    ids, texts = [], []
    for article_id, pi, passage in stream_wikipedia_passages(args.n_passages):
        ids.append(f"wiki_{article_id}_{pi}")
        texts.append(passage)
        if len(texts) % 10_000 == 0:
            print(f"  Collected {len(texts):,} passages ...")

    print(f"Collected {len(texts):,} passages.")

    # Step 2: Embed documents
    print("\n--- Step 2: Embed documents ---")
    doc_vecs = embed_texts(texts, DOC_PREFIX, args.device, args.batch_size, args.model, args.max_seq_len)

    # Step 3: Embed queries
    print("\n--- Step 3: Embed benchmark queries ---")
    query_vecs = embed_texts(BENCHMARK_QUERIES, QUERY_PREFIX, args.device, args.batch_size, args.model, args.max_seq_len)

    # Step 4: Write cache
    print("\n--- Step 4: Write HDF5 cache ---")
    write_cache(args.out, ids, texts, doc_vecs, BENCHMARK_QUERIES, query_vecs,
                model_id=args.model)
    print("\nDone. Run the benchmark with:")
    print(f"  python benchmark/run.py --corpus {args.out}")


if __name__ == "__main__":
    main()
