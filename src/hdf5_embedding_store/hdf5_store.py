"""HDF5-backed embedding store (float32, gzip or no compression).

Schema
------
/embeddings/
    vectors     float32 (N, D)  chunked, optionally compressed
    text        variable-length str (N,)
    ids         variable-length str (N,)
    timestamps  float64 (N,)
    attrs:  model, dimension, count
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .base import EmbeddingStore, SearchResult, cosine_scores

CHUNK_ROWS = 512
COMPRESSION = "gzip"
COMPRESSION_OPTS = 4


class HDF5EmbeddingStore(EmbeddingStore):
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

    def _ensure_schema(self, dimension: int, model: str) -> None:
        if "embeddings" not in self._f:
            grp = self._f.create_group("embeddings")
            vlen_str = h5py.string_dtype(encoding="utf-8")

            ds_kwargs = dict(
                shape=(0, dimension),
                maxshape=(None, dimension),
                dtype=np.float32,
                chunks=(self.chunk_rows, dimension),
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

    def insert(self, ids: list[str], texts: list[str], embeddings: np.ndarray) -> None:
        n = len(ids)
        assert embeddings.shape == (n, self.dimension)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = (embeddings / norms).astype(np.float32)

        grp = self._grp
        old_n = grp["vectors"].shape[0]
        new_n = old_n + n

        grp["vectors"].resize(new_n, axis=0)
        grp["text"].resize(new_n, axis=0)
        grp["ids"].resize(new_n, axis=0)
        grp["timestamps"].resize(new_n, axis=0)

        grp["vectors"][old_n:new_n] = embeddings
        grp["text"][old_n:new_n] = np.array(texts, dtype=object)
        grp["ids"][old_n:new_n] = np.array(ids, dtype=object)
        grp["timestamps"][old_n:new_n] = np.full(n, time.time(), dtype=np.float64)

        grp.attrs["count"] = new_n
        self._f.flush()

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        grp = self._grp
        n = grp.attrs["count"]
        if n == 0:
            return []

        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        vectors = grp["vectors"][:]
        scores = cosine_scores(vectors, query.astype(np.float32))

        k = min(k, n)
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        sorted_positions = np.argsort(top_idx)
        sorted_idx = top_idx[sorted_positions]
        inverse = np.argsort(sorted_positions)

        ids_raw = grp["ids"][sorted_idx.tolist()]
        texts_raw = grp["text"][sorted_idx.tolist()]

        ids_ranked = ids_raw[inverse]
        texts_ranked = texts_raw[inverse]

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
        grp = self._grp
        ids = grp["ids"][:]
        ids_decoded = [v.decode() if isinstance(v, bytes) else v for v in ids]
        try:
            idx = ids_decoded.index(id)
        except ValueError:
            return None

        text = grp["text"][idx]
        embedding = grp["vectors"][idx]
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
