# Benchmark Results

## Real Wikipedia corpus (dim=768, nomic-embed-text-v1.5, top-10 cosine search)

Corpus: 10k passages from wikimedia/wikipedia 20231101.en, embedded with `nomic-ai/nomic-embed-text-v1.5`.
Query: *"How does photosynthesis work in plants?"*

Reproduce:
```bash
python scripts/prepare_corpus.py
python benchmark/run.py --corpus results/wikipedia_corpus.h5 --sizes 1000 10000
```

### At N=10,000

| Backend | MB/M | Ingest (eps) | Search (ms) | get(id) (ms) | Tokens (k=10) |
|---|--:|--:|--:|--:|--:|
| HDF5 (gzip-4) | 3,574 | 5,256 | 196 | 28.8 | 1,669 |
| HDF5 (uncompressed) | 3,863 | 93,928 | 12 | 11.9 | 1,669 |
| HDF5 (cached) | 3,863 | 56,344 | **2** | 0.46 | 1,669 |
| **HDF5 (int8)** | **1,508** | 70,195 | **2** | 0.46 | 1,670 |
| HDF5 (float16) | 2,291 | 44,487 | **2** | 0.45 | 1,669 |
| HDF5 (Blosc2/LZ4) | 3,578 | 36,349 | **2** | 0.46 | 1,669 |
| HDF5 (IVF) | 3,959 | 5,976 | 32 | 18.2 | **1,785** |
| SQLite BLOB | 4,326 | 34,991 | 53 | **0.01** | 1,669 |
| NumPy .npy | 3,723 | 27,813 | 1 | 0.08 | 1,669 |

*MB/M = megabytes per million embeddings. eps = embeddings/second. Tokens: tiktoken cl100k_base on JSON-serialized top-10 results (text + id + score, no raw vectors).*

---

## Synthetic baseline (dim=1536, top-10 cosine search)

Reproduce:
```bash
python benchmark/run.py --sizes 1000 10000 100000
```

### At N=100,000

| Backend | MB/M | Ingest (eps) | Search (ms) | get(id) (ms) | Tokens (k=10) |
|---|--:|--:|--:|--:|--:|
| HDF5 (gzip-4) | 5,838 | 3,714 | 2,606 | 46.4 | 310 |
| HDF5 (uncompressed) | 6,294 | 92,241 | 390 | 35.2 | 310 |
| HDF5 (cached) | 6,294 | 5,351 | **21** | 0.47 | 310 |
| **HDF5 (int8)** | **1,674** | 4,355 | **21** | 0.51 | 309 |
| HDF5 (float16) | 3,211 | 5,158 | 23 | 0.50 | 310 |
| HDF5 (Blosc2/LZ4) | 5,854 | 5,025 | 28 | 0.50 | 310 |
| HDF5 (IVF) | 6,315 | 8,504 | 189 | 60.3 | 310 |
| SQLite BLOB | 8,225 | 22,250 | 904 | **0.01** | 310 |
| NumPy .npy | 6,203 | 1,610 | **21** | 1.1 | 310 |

*Uniform synthetic texts produce identical token counts across all backends — real workloads vary (see Wikipedia results above).*

---

## Key findings

**Which backend to use:**
- **In-process RAG, bounded memory** → `HDF5CachedStore` or `HDF5Int8Store`. Both hit ~2 ms at 10k on real data; int8 at ~2.5× the storage savings.
- **Storage-constrained** → `HDF5Int8Store` (~2.5× smaller than float32 at 768-dim, <0.5% cosine error) or `HDF5Float16Store` (~1.7× smaller, <0.001 error).
- **Fast point lookups (get by id)** → `SQLiteEmbeddingStore` (0.01 ms indexed lookup vs. 0.5 ms for HDF5).
- **Fastest cold ingest** → `HDF5EmbeddingStore(compression=None)` at ~94k eps.
- **Don't use gzip HDF5 for search** — full decompression on every query costs 196 ms at 10k, scaling linearly.
- **IVF does not redeem itself on real semantic data** — per-cluster HDF5 open overhead dominates at these scales; flat cached search wins until N > ~500k.
- **Token economy is only meaningful with real data** — IVF costs ~7% more tokens (1,785 vs 1,669) per query due to cluster metadata in results.
