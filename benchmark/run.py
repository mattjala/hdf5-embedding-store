"""Benchmark harness for all embedding store backends.

Usage
-----
    # Synthetic data (default)
    python benchmark/run.py
    python benchmark/run.py --sizes 1000 10000 100000

    # Real Wikipedia corpus (after running scripts/prepare_corpus.py)
    python benchmark/run.py --corpus results/wikipedia_corpus.h5
    python benchmark/run.py --corpus results/wikipedia_corpus.h5 --query-idx 2

Metrics
-------
1. storage_mb_per_m  — MB per million embeddings after insert
2. throughput_eps    — embeddings/sec for batch ingest
3. search_ms         — median ms for exact top-10 cosine search
4. get_ms            — median ms for co-located get(id)
5. result_tokens_k10 — tiktoken count of top-10 search results serialized
                       as JSON (text + id + score, no raw embeddings).
                       Measures real LLM context cost per RAG retrieval.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hdf5_embedding_store.hdf5_store import HDF5EmbeddingStore
from hdf5_embedding_store.hdf5_cached_store import HDF5CachedStore
from hdf5_embedding_store.hdf5_int8_store import HDF5Int8Store
from hdf5_embedding_store.hdf5_ivf_store import HDF5IVFStore
from hdf5_embedding_store.hdf5_float16_store import HDF5Float16Store
from hdf5_embedding_store.hdf5_blosc_store import HDF5BloscStore
from hdf5_embedding_store.sqlite_store import SQLiteEmbeddingStore
from hdf5_embedding_store.numpy_store import NumpyEmbeddingStore


try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text) // 4


DIMENSION = 1536
K = 10
RNG_SEED = 42
SEARCH_REPEATS = 5

BACKENDS: dict[str, dict] = {
    "hdf5_gzip":        {"label": "HDF5 (gzip-4)",       "color": "#1f77b4", "marker": "o"},
    "hdf5_uncompressed":{"label": "HDF5 (uncompressed)",  "color": "#aec7e8", "marker": "s"},
    "hdf5_cached":      {"label": "HDF5 (cached)",        "color": "#17becf", "marker": "P"},
    "hdf5_int8":        {"label": "HDF5 (int8)",          "color": "#9467bd", "marker": "X"},
    "hdf5_int8_gzip":   {"label": "HDF5 (int8+gzip)",    "color": "#c5b0d5", "marker": "x"},
    "hdf5_float16":     {"label": "HDF5 (float16)",       "color": "#e377c2", "marker": "h"},
    "hdf5_blosc":       {"label": "HDF5 (Blosc2/LZ4)",    "color": "#bcbd22", "marker": "8"},
    "hdf5_ivf":         {"label": "HDF5 (IVF)",           "color": "#8c564b", "marker": "*"},
    "sqlite":           {"label": "SQLite BLOB",           "color": "#ff7f0e", "marker": "^"},
    "numpy":            {"label": "NumPy .npy",            "color": "#2ca02c", "marker": "D"},
}


# ---------------------------------------------------------------------------
# Corpus: real vs. synthetic
# ---------------------------------------------------------------------------

@dataclass
class Corpus:
    """Holds the full pool of embeddings + texts from which benchmark sizes are sliced."""
    embeddings: np.ndarray   # (N, D) float32, L2-normalised
    texts: list[str]         # length N
    ids: list[str]           # length N
    query: np.ndarray        # (D,) float32 — the search probe
    query_text: str
    dim: int
    source: str              # "synthetic" or corpus path


def load_corpus(path: Path, query_idx: int = 0) -> Corpus:
    """Load a real corpus from an HDF5 cache produced by scripts/prepare_corpus.py."""
    print(f"Loading corpus from {path} ...")
    with h5py.File(path, "r") as f:
        dg = f["documents"]
        embeddings = dg["vectors"][:]
        raw_texts = dg["texts"][:]
        raw_ids   = dg["ids"][:]
        dim = int(dg.attrs["dimension"])
        source = str(dg.attrs.get("source", str(path)))

        qg = f["queries"]
        query_vecs = qg["vectors"][:]
        query_texts_raw = qg["texts"][:]

    texts = [t.decode() if isinstance(t, bytes) else t for t in raw_texts]
    ids   = [i.decode() if isinstance(i, bytes) else i for i in raw_ids]
    query_text = (
        query_texts_raw[query_idx].decode()
        if isinstance(query_texts_raw[query_idx], bytes)
        else query_texts_raw[query_idx]
    )

    print(f"  {len(texts):,} passages, dim={dim}, query [{query_idx}]: '{query_text[:60]}...'")
    return Corpus(
        embeddings=embeddings,
        texts=texts,
        ids=ids,
        query=query_vecs[query_idx].astype(np.float32),
        query_text=query_text,
        dim=dim,
        source=source,
    )


def synthetic_corpus(n: int, dim: int = DIMENSION) -> Corpus:
    """Build a synthetic corpus of random unit vectors."""
    rng = np.random.default_rng(RNG_SEED)
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    embeddings = raw / norms

    texts = [f"Document text for embedding {i}." for i in range(n)]
    ids   = [f"id_{i:08d}" for i in range(n)]

    rng2 = np.random.default_rng(999)
    q = rng2.standard_normal(dim).astype(np.float32)
    q = q / np.linalg.norm(q)

    return Corpus(
        embeddings=embeddings, texts=texts, ids=ids,
        query=q, query_text="(synthetic query)", dim=dim, source="synthetic",
    )


def slice_corpus(corpus: Corpus, n: int) -> tuple[np.ndarray, list[str], list[str]]:
    """Return the first n embeddings, texts, ids from the corpus."""
    if n > len(corpus.texts):
        raise ValueError(f"Corpus has {len(corpus.texts):,} passages, requested {n:,}")
    return corpus.embeddings[:n], corpus.texts[:n], corpus.ids[:n]


# ---------------------------------------------------------------------------
# Store helpers (unchanged)
# ---------------------------------------------------------------------------

def dir_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def storage_bytes(sp: Path) -> int:
    if sp.is_dir():
        return dir_size_bytes(sp)
    candidates = [sp] if sp.exists() else []
    candidates += [f for f in sp.parent.glob(sp.name + ".*") if f != sp]
    return sum(dir_size_bytes(p) for p in candidates)


def timed_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def result_tokens(results) -> int:
    total = 0
    for r in results:
        doc = json.dumps({"id": r.id, "text": r.text, "score": round(r.score, 4)})
        total += count_tokens(doc)
    return total


def build_store(backend: str, path: Path, dim: int):
    match backend:
        case "hdf5_gzip":        return HDF5EmbeddingStore(path, dimension=dim, compression="gzip", compression_opts=4)
        case "hdf5_uncompressed": return HDF5EmbeddingStore(path, dimension=dim, compression=None)
        case "hdf5_cached":      return HDF5CachedStore(path, dimension=dim, compression=None)
        case "hdf5_int8":        return HDF5Int8Store(path, dimension=dim)
        case "hdf5_int8_gzip":   return HDF5Int8Store(path, dimension=dim, compression="gzip", compression_opts=4)
        case "hdf5_float16":     return HDF5Float16Store(path, dimension=dim)
        case "hdf5_blosc":       return HDF5BloscStore(path, dimension=dim)
        case "hdf5_ivf":         return HDF5IVFStore(path, dimension=dim)
        case "sqlite":           return SQLiteEmbeddingStore(path, dimension=dim)
        case "numpy":            return NumpyEmbeddingStore(path, dimension=dim)
        case _:
            raise ValueError(f"Unknown backend: {backend}")


def clean_store(backend: str, path: Path) -> None:
    if backend == "numpy":
        for f in path.parent.glob(path.stem + "*"):
            if f.is_file():
                f.unlink()
    elif path.exists():
        path.unlink() if path.is_file() else shutil.rmtree(path)


def store_path(outdir: Path, backend: str, n: int = 0) -> Path:
    suffix = {
        "hdf5_gzip": ".h5", "hdf5_uncompressed": ".h5", "hdf5_cached": ".h5",
        "hdf5_int8": ".h5", "hdf5_int8_gzip": ".h5", "hdf5_float16": ".h5", "hdf5_blosc": ".h5",
        "hdf5_ivf": ".h5",  "sqlite": ".db",             "numpy": "",
    }.get(backend, "")
    return outdir / f"bench_{backend}_{n}{suffix}"


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

BenchmarkResult = dict


def run_benchmark(
    sizes: list[int],
    backends: list[str],
    corpus: Corpus,
    outdir: Path,
    batch_size: int = 1000,
    verbose: bool = True,
) -> BenchmarkResult:
    results: BenchmarkResult = {b: {} for b in backends}
    dim = corpus.dim

    for b in backends:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Backend: {BACKENDS[b]['label']}")
            print(f"{'='*60}")

        for n in sorted(sizes):
            if verbose:
                print(f"  N={n:>8,} ... ", end="", flush=True)

            sp = store_path(outdir, b, n)
            clean_store(b, sp)

            embeddings, texts, ids = slice_corpus(corpus, n)

            store = build_store(b, sp, dim)
            total_insert_time = 0.0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                _, t = timed_call(
                    store.insert, ids[start:end], texts[start:end], embeddings[start:end]
                )
                total_insert_time += t
            throughput = n / total_insert_time

            store.close()
            size_b = storage_bytes(sp)

            store = build_store(b, sp, dim)
            search_times, last_results = [], None
            for _ in range(SEARCH_REPEATS):
                res, t = timed_call(store.search, corpus.query, K)
                search_times.append(t)
                last_results = res
            search_ms = np.median(search_times) * 1000

            target_id = ids[n // 2]
            get_times = []
            for _ in range(SEARCH_REPEATS):
                _, t = timed_call(store.get, target_id)
                get_times.append(t)
            get_ms = np.median(get_times) * 1000

            tokens = result_tokens(last_results) if last_results else 0
            store.close()

            results[b][n] = {
                "n": n,
                "throughput_eps": throughput,
                "size_bytes": size_b,
                "storage_mb_per_m": size_b / n * 1e6 / 1e6,
                "search_ms": search_ms,
                "get_ms": get_ms,
                "result_tokens_k10": tokens,
            }

            if verbose:
                print(
                    f"ingest={throughput:,.0f} eps  "
                    f"size={size_b/1e6:.1f} MB  "
                    f"search={search_ms:.1f} ms  "
                    f"get={get_ms:.2f} ms  "
                    f"tokens={tokens}"
                )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    results: BenchmarkResult,
    outdir: Path,
    dim: int = DIMENSION,
    title_suffix: str = "",
) -> None:
    backends = list(results.keys())
    sizes = sorted(next(iter(results.values())).keys())
    max_n = max(sizes)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Embedding Store Benchmarks  |  dim={dim}, k={K}{title_suffix}\n"
        "HDF5 variants vs. SQLite BLOB vs. NumPy .npy",
        fontsize=12, y=1.01,
    )

    def _plot_line(ax, metric, title, ylabel, transform=None):
        for b in backends:
            data = results[b]
            xs = sorted(data.keys())
            ys = [transform(data[x][metric], x) if transform else data[x][metric] for x in xs]
            ax.plot(xs, ys, label=BACKENDS[b]["label"],
                    color=BACKENDS[b]["color"], marker=BACKENDS[b]["marker"])
        ax.set_title(title); ax.set_xlabel("N"); ax.set_ylabel(ylabel)
        ax.set_xscale("log"); ax.legend(fontsize=7)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    def _plot_bar(ax, metric, title, ylabel, fmt="{:.0f}"):
        bar_labels = [BACKENDS[b]["label"] for b in backends]
        bar_vals   = [results[b][max_n][metric] for b in backends]
        bar_colors = [BACKENDS[b]["color"] for b in backends]
        bars = ax.bar(bar_labels, bar_vals, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{title} at N={max_n:,}"); ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        for bar, val in zip(bars, bar_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    fmt.format(val), ha="center", va="bottom", fontsize=7)

    theoretical_mb = dim * 4 / 1e6

    ax = axes[0, 0]
    _plot_line(ax, "size_bytes", "Storage: MB per million embeddings", "MB / million",
               transform=lambda v, n: v / n * 1e6 / 1e6)
    ax.axhline(theoretical_mb, linestyle="--", color="gray", linewidth=0.8,
               label=f"Raw f32 ({theoretical_mb:.0f} MB/M)")
    ax.legend(fontsize=7)

    _plot_bar(axes[0, 1], "throughput_eps", "Ingest Throughput", "Embeddings / second", "{:,.0f}")
    _plot_line(axes[0, 2], "result_tokens_k10", f"Context tokens (top-{K}) vs N", "Tokens")
    _plot_line(axes[1, 0], "search_ms", f"Top-{K} cosine search latency", "Latency (ms, median)")
    _plot_bar(axes[1, 1], "get_ms", "Co-located get(id)", "Latency (ms, median)", "{:.2f}")
    axes[1, 2].set_visible(False)

    plt.tight_layout()
    out_path = outdir / "benchmark_results.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {out_path}")


def save_raw(results: BenchmarkResult, outdir: Path, meta: dict | None = None) -> None:
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj

    serialisable = {"_meta": meta or {}}
    for backend, ns in results.items():
        serialisable[backend] = {}
        for n, metrics in ns.items():
            serialisable[backend][str(n)] = {k: convert(v) for k, v in metrics.items()}

    out_path = outdir / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Raw data: {out_path}")


def print_summary_table(results: BenchmarkResult, corpus: Corpus) -> None:
    backends = list(results.keys())
    sizes = sorted(next(iter(results.values())).keys())

    print(f"\nCorpus: {corpus.source}  |  dim={corpus.dim}  |  query: '{corpus.query_text[:60]}'")
    print("=" * 104)
    print(f"{'Backend':<26} {'N':>8}  {'MB/M':>6}  {'eps':>9}  {'search ms':>9}  {'get ms':>7}  {'tokens':>7}")
    print("-" * 104)
    for b in backends:
        for n in sizes:
            if n not in results[b]:
                continue
            d = results[b][n]
            print(
                f"{BACKENDS[b]['label']:<26} {n:>8,}  "
                f"{d['storage_mb_per_m']:>6.1f}  "
                f"{d['throughput_eps']:>9,.0f}  "
                f"{d['search_ms']:>9.1f}  "
                f"{d['get_ms']:>7.3f}  "
                f"{d['result_tokens_k10']:>7}"
            )
    print("=" * 104)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Embedding store benchmark")
    p.add_argument("--sizes", nargs="+", type=int, default=[1_000, 10_000, 100_000])
    p.add_argument("--quick", action="store_true",
                   help="Sizes 1k/10k only, skip gzip HDF5")
    p.add_argument("--only", nargs="+", choices=list(BACKENDS.keys()))
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--dim", type=int, default=DIMENSION,
                   help=f"Embedding dimension for synthetic mode (default: {DIMENSION})")
    p.add_argument("--batch-size", type=int, default=1000)

    # Real corpus
    p.add_argument(
        "--corpus", type=Path, default=None, metavar="PATH",
        help="HDF5 corpus cache from scripts/prepare_corpus.py. "
             "When set, uses real Wikipedia embeddings instead of synthetic data. "
             "--dim is ignored (dimension is read from the cache).",
    )
    p.add_argument(
        "--query-idx", type=int, default=0, metavar="N",
        help="Index into the corpus query set to use as the benchmark search probe (default: 0)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    store_dir = outdir / "stores"
    store_dir.mkdir(exist_ok=True)

    # Build corpus
    if args.corpus is not None:
        if not args.corpus.exists():
            sys.exit(f"Corpus file not found: {args.corpus}\n"
                     "Run scripts/prepare_corpus.py first.")
        corpus = load_corpus(args.corpus, query_idx=args.query_idx)
        title_suffix = f", Wikipedia ({corpus.source.split()[0]})"
        json_meta = {"source": corpus.source, "dim": corpus.dim,
                     "query": corpus.query_text}
    else:
        dim = args.dim
        # For synthetic mode, build a corpus large enough for the largest requested size
        max_n = max(args.sizes if not args.quick else [10_000])
        corpus = synthetic_corpus(max_n, dim)
        title_suffix = ", synthetic"
        json_meta = {"source": "synthetic", "dim": dim}

    if args.quick:
        sizes = [1_000, 10_000]
        backends = args.only or [b for b in BACKENDS if b != "hdf5_gzip"]
    else:
        sizes = args.sizes
        backends = args.only or list(BACKENDS.keys())

    # Validate sizes against corpus
    max_available = len(corpus.texts)
    sizes = [n for n in sizes if n <= max_available]
    if not sizes:
        sys.exit(f"All requested sizes exceed corpus size ({max_available:,}). "
                 "Use --sizes or a larger corpus.")
    dropped = [n for n in (args.sizes if not args.quick else [1_000, 10_000]) if n > max_available]
    if dropped:
        print(f"Warning: sizes {dropped} exceed corpus size ({max_available:,}), skipping.")

    print(f"Backends : {', '.join(BACKENDS[b]['label'] for b in backends)}")
    print(f"Sizes    : {', '.join(f'{n:,}' for n in sorted(sizes))}")
    print(f"Dimension: {corpus.dim}")
    print(f"Output   : {outdir}")

    results = run_benchmark(
        sizes=sizes,
        backends=backends,
        corpus=corpus,
        outdir=store_dir,
        batch_size=args.batch_size,
    )

    plot_results(results, outdir, dim=corpus.dim, title_suffix=title_suffix)
    save_raw(results, outdir, meta=json_meta)
    print_summary_table(results, corpus)


if __name__ == "__main__":
    main()
