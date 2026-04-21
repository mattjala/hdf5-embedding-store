"""HDF5 embedding store with in-memory vector cache and id-index dict.

Identical on-disk format to HDF5EmbeddingStore. The cache is populated on
the first search() call and updated on each insert(). This removes the
per-query disk I/O penalty while keeping all the HDF5 storage properties.

The id->index dict brings get() latency from O(N) linear scan to O(1).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .base import EmbeddingStore, SearchResult, cosine_scores
from .hdf5_store import CHUNK_ROWS, COMPRESSION, COMPRESSION_OPTS


class HDF5CachedStore(EmbeddingStore):
    def __init__(
        self,
        path: str | Path,
        dimension: int = 1536,
        model: str = "synthetic",
        mode: str = "a",
        chunk_rows: int = CHUNK_ROWS,
        compression: Optional[str] = COMPRESSION,
        compression_opts: int = COMPRESSION_OPTS,
    ):
        self.path = Path(path)
        self.dimension = dimension
        self.chunk_rows = chunk_rows
        self.compression = compression
        self.compression_opts = compression_opts

        self._f = h5py.File(self.path, mode)
        self._ensure_schema(dimension, model)

        self._cache: Optional[np.ndarray] = None
        self._id_index: dict[str, int] = {}
        self._warm = False

        if self._grp.attrs.get("count", 0) > 0:
            self._warm_cache()

    def _ensure_schema(self, dimension: int, model: str) -> None:
        if "embeddings" not in self._f:
            grp = self._f.create_group("embeddings")
            vlen_str = h5py.string_dtype(encoding="utf-8")
            ds_kwargs = dict(
                shape=(0, dimension), maxshape=(None, dimension),
                dtype=np.float32, chunks=(self.chunk_rows, dimension),
            )
            if self.compression is not None:
                ds_kwargs["compression"] = self.compression
                ds_kwargs["compression_opts"] = self.compression_opts
            grp.create_dataset("vectors", **ds_kwargs)
            grp.create_dataset("text",       shape=(0,), maxshape=(None,), dtype=vlen_str)
            grp.create_dataset("ids",        shape=(0,), maxshape=(None,), dtype=vlen_str)
            grp.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.float64)
            grp.attrs["model"] = model
            grp.attrs["dimension"] = dimension
            grp.attrs["count"] = 0

    @property
    def _grp(self) -> h5py.Group:
        return self._f["embeddings"]

    def _warm_cache(self) -> None:
        grp = self._grp
        self._cache = grp["vectors"][:]
        raw_ids = grp["ids"][:]
        self._id_index = {
            (v.decode() if isinstance(v, bytes) else v): i
            for i, v in enumerate(raw_ids)
        }
        self._warm = True

    def insert(self, ids: list[str], texts: list[str], embeddings: np.ndarray) -> None:
        n = len(ids)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        unit = (embeddings / norms).astype(np.float32)

        grp = self._grp
        old_n = grp["vectors"].shape[0]
        new_n = old_n + n

        grp["vectors"].resize(new_n, axis=0)
        grp["text"].resize(new_n, axis=0)
        grp["ids"].resize(new_n, axis=0)
        grp["timestamps"].resize(new_n, axis=0)

        grp["vectors"][old_n:new_n] = unit
        grp["text"][old_n:new_n] = np.array(texts, dtype=object)
        grp["ids"][old_n:new_n] = np.array(ids, dtype=object)
        grp["timestamps"][old_n:new_n] = np.full(n, time.time(), dtype=np.float64)
        grp.attrs["count"] = new_n
        self._f.flush()

        if self._warm and self._cache is not None:
            self._cache = np.concatenate([self._cache, unit], axis=0)
        else:
            self._cache = unit if self._cache is None else np.concatenate([self._cache, unit])
            self._warm = True

        for i, id_ in enumerate(ids):
            self._id_index[id_] = old_n + i

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        if not self._warm:
            self._warm_cache()

        n = self._cache.shape[0]
        if n == 0:
            return []

        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        query = query.astype(np.float32)

        scores = cosine_scores(self._cache, query)
        k = min(k, n)
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        sorted_pos = np.argsort(top_idx)
        sorted_idx = top_idx[sorted_pos]
        inverse = np.argsort(sorted_pos)
        texts_raw = self._grp["text"][sorted_idx.tolist()]
        ids_raw = self._grp["ids"][sorted_idx.tolist()]
        texts_ranked = texts_raw[inverse]
        ids_ranked = ids_raw[inverse]

        return [
            SearchResult(
                id=ids_ranked[i].decode() if isinstance(ids_ranked[i], bytes) else ids_ranked[i],
                text=texts_ranked[i].decode() if isinstance(texts_ranked[i], bytes) else texts_ranked[i],
                score=float(scores[top_idx[i]]),
                metadata={"index": int(top_idx[i])},
            )
            for i in range(len(top_idx))
        ]

    def get(self, id: str) -> SearchResult | None:
        idx = self._id_index.get(id)
        if idx is None:
            return None
        grp = self._grp
        text = grp["text"][idx]
        embedding = self._cache[idx] if self._warm else grp["vectors"][idx]
        return SearchResult(
            id=id,
            text=text.decode() if isinstance(text, bytes) else text,
            score=1.0,
            metadata={
                "index": idx,
                "timestamp": float(grp["timestamps"][idx]),
                "embedding": embedding,
            },
        )

    def count(self) -> int:
        return int(self._grp.attrs.get("count", 0))

    def close(self) -> None:
        if self._f.id.valid:
            self._f.close()
