"""HDF5 embedding store with int8 scalar quantization.

Each float32 vector is quantized to int8 using per-vector scale factors:
    int8_vec = round(float32_vec * 127 / max(abs(float32_vec)))
    scale    = max(abs(float32_vec)) / 127

For unit-norm vectors, cosine similarity is preserved up to ~0.3–0.5% error.
Storage is 4× smaller than float32 with no decompression overhead on reads.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .base import EmbeddingStore, SearchResult, cosine_scores
from typing import Optional
from .hdf5_store import CHUNK_ROWS


def quantize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """float32 (N, D) → int8 (N, D), scales float32 (N,)"""
    maxvals = np.abs(vectors).max(axis=1, keepdims=True).clip(min=1e-9)
    scales = (maxvals / 127.0).astype(np.float32).squeeze(1)
    int8 = np.round(vectors / maxvals * 127).astype(np.int8)
    return int8, scales


def dequantize(int8: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """int8 (N, D), scales (N,) → float32 (N, D)"""
    return int8.astype(np.float32) * scales[:, None]


def int8_cosine_scores(matrix_i8: np.ndarray, scales: np.ndarray, query_f32: np.ndarray) -> np.ndarray:
    dots = matrix_i8.astype(np.float32) @ query_f32
    return dots * scales


class HDF5Int8Store(EmbeddingStore):
    def __init__(
        self,
        path: str | Path,
        dimension: int = 1536,
        model: str = "synthetic",
        mode: str = "a",
        chunk_rows: int = CHUNK_ROWS,
        compression: Optional[str] = None,
        compression_opts: int = 4,
    ):
        self.path = Path(path)
        self.dimension = dimension
        self.chunk_rows = chunk_rows
        self.compression = compression
        self.compression_opts = compression_opts

        self._f = h5py.File(self.path, mode)
        self._ensure_schema(dimension, model)

        self._vec_cache: Optional[np.ndarray] = None
        self._f32_cache: Optional[np.ndarray] = None
        self._scale_cache: Optional[np.ndarray] = None
        self._id_index: dict[str, int] = {}
        self._warm = False

        if self._grp.attrs.get("count", 0) > 0:
            self._warm_cache()

    def _ensure_schema(self, dimension: int, model: str) -> None:
        if "embeddings" not in self._f:
            grp = self._f.create_group("embeddings")
            vlen_str = h5py.string_dtype(encoding="utf-8")
            vec_kwargs: dict = dict(
                shape=(0, dimension), maxshape=(None, dimension),
                dtype=np.int8, chunks=(self.chunk_rows, dimension),
            )
            if self.compression is not None:
                vec_kwargs["compression"] = self.compression
                vec_kwargs["compression_opts"] = self.compression_opts
            grp.create_dataset("vectors", **vec_kwargs)
            grp.create_dataset(
                "scales",
                shape=(0,), maxshape=(None,),
                dtype=np.float32, chunks=(self.chunk_rows * 4,),
            )
            grp.create_dataset("text",       shape=(0,), maxshape=(None,), dtype=vlen_str)
            grp.create_dataset("ids",        shape=(0,), maxshape=(None,), dtype=vlen_str)
            grp.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.float64)
            grp.attrs["model"] = model
            grp.attrs["dimension"] = dimension
            grp.attrs["count"] = 0
            grp.attrs["quantization"] = "int8_per_vector"

    @property
    def _grp(self) -> h5py.Group:
        return self._f["embeddings"]

    def _warm_cache(self) -> None:
        grp = self._grp
        self._vec_cache = grp["vectors"][:]
        self._scale_cache = grp["scales"][:]
        self._f32_cache = dequantize(self._vec_cache, self._scale_cache)
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

        int8_vecs, scales = quantize(unit)

        grp = self._grp
        old_n = grp["vectors"].shape[0]
        new_n = old_n + n

        for ds in ("vectors", "scales", "text", "ids", "timestamps"):
            grp[ds].resize(new_n, axis=0)

        grp["vectors"][old_n:new_n] = int8_vecs
        grp["scales"][old_n:new_n] = scales
        grp["text"][old_n:new_n] = np.array(texts, dtype=object)
        grp["ids"][old_n:new_n] = np.array(ids, dtype=object)
        grp["timestamps"][old_n:new_n] = np.full(n, time.time(), dtype=np.float64)
        grp.attrs["count"] = new_n
        self._f.flush()

        f32_vecs = dequantize(int8_vecs, scales)
        if self._warm and self._vec_cache is not None:
            self._vec_cache = np.concatenate([self._vec_cache, int8_vecs], axis=0)
            self._scale_cache = np.concatenate([self._scale_cache, scales])
            self._f32_cache = np.concatenate([self._f32_cache, f32_vecs], axis=0)
        else:
            self._vec_cache = int8_vecs
            self._scale_cache = scales
            self._f32_cache = f32_vecs
            self._warm = True

        for i, id_ in enumerate(ids):
            self._id_index[id_] = old_n + i

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        if not self._warm:
            self._warm_cache()

        n = self._f32_cache.shape[0]
        if n == 0:
            return []

        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        query = query.astype(np.float32)

        scores = cosine_scores(self._f32_cache, query)
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
        embedding = dequantize(
            self._vec_cache[idx:idx+1], self._scale_cache[idx:idx+1]
        )[0]
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
