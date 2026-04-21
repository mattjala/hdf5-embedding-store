# hdf5-embedding-store

HDF5-backed embedding stores for agent RAG pipelines. The core argument: agent semantic memory is a matrix of float32 vectors. HDF5 stores matrices natively — with compression, chunking, co-located metadata, and no infrastructure overhead.

The pitch to AI engineers: drop an HDF5 store into your RAG pipeline instead of a vector DB for workloads where exact search is acceptable and deployment simplicity matters (one file, no server).

## Backends

| Backend | Description |
|---|---|
| `HDF5EmbeddingStore` | float32, optional gzip compression |
| `HDF5CachedStore` | float32 on disk, in-memory vector cache for fast search |
| `HDF5Int8Store` | int8 quantization — 4× storage savings, ~0.3–0.5% cosine error |
| `HDF5Float16Store` | float16 — 2× storage savings, <0.001 cosine error |
| `HDF5BloscStore` | float32 with Blosc2/LZ4 compression (requires `hdf5plugin`) |
| `HDF5IVFStore` | IVF clustering — sub-linear reads at query time, ~90–95% recall |
| `SQLiteEmbeddingStore` | float32 BLOBs — fastest `get(id)`, slowest search |
| `NumpyEmbeddingStore` | flat `.npy` files — minimal baseline |

## Quick start

```bash
pip install -e ".[blosc]"
```

```python
from hdf5_embedding_store import HDF5CachedStore
import numpy as np

store = HDF5CachedStore("memory.h5", dimension=1536)
store.insert(["id_0"], ["Document text"], np.random.randn(1, 1536).astype(np.float32))
results = store.search(query_embedding, k=10)
record = store.get("id_0")
store.close()
```

## Benchmark results

See [benchmark/results.md](benchmark/results.md) for full tables and analysis. Short version: `HDF5CachedStore` and `HDF5Int8Store` hit ~2 ms search at 10k passages on a real Wikipedia corpus; int8 at ~2.5× the storage savings.

**Honest limitations:**
- Exact search only; HNSW (Chroma/FAISS) wins past ~500k vectors
- No concurrent writes (HDF5 single-writer constraint)
- No built-in filtering — post-filter or structure schema accordingly

## Running tests

```bash
pytest tests/ -v
```

## Interactive demo

Compare backends side-by-side on a real Wikipedia corpus:

```bash
python demo/run.py              # requires results/wikipedia_corpus.h5 (see below)
python demo/run.py --n 1000     # smaller subset, faster startup
python demo/run.py --synthetic  # no corpus needed — random unit vectors
```

## Reproducing benchmarks

The Wikipedia benchmark requires a pre-built corpus cache (~36 MB):

```bash
# Step 1: embed 10k Wikipedia passages with nomic-embed-text-v1.5 (~20 min CPU)
pip install -e ".[corpus]"
python scripts/prepare_corpus.py                     # saves results/wikipedia_corpus.h5
python scripts/prepare_corpus.py --n-passages 50000  # larger corpus

# Step 2: run benchmarks
python benchmark/run.py --corpus results/wikipedia_corpus.h5 --sizes 1000 10000
```

Synthetic benchmarks (no corpus needed):

```bash
python benchmark/run.py --sizes 1000 10000 100000

# Specific backends only
python benchmark/run.py --only hdf5_cached hdf5_int8 hdf5_float16 numpy
```
