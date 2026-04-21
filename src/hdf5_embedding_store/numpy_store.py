"""Flat NumPy .npy embedding store — minimal in-memory baseline.

Files on disk:
    {path}.vectors.npy   float32 (N, D)
    {path}.meta.npz      ids (str), texts (str), timestamps (float64)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .base import EmbeddingStore, SearchResult, cosine_scores


class NumpyEmbeddingStore(EmbeddingStore):
    def __init__(self, path: str | Path, dimension: int = 1536):
        self.path = Path(path)
        self.dimension = dimension
        self._vectors_path = self.path.with_suffix(".vectors.npy")
        self._meta_path = self.path.with_suffix(".meta.npz")

        self._vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._timestamps: list[float] = []

        self._load()

    def _load(self) -> None:
        if self._vectors_path.exists():
            self._vectors = np.load(str(self._vectors_path))
        if self._meta_path.exists():
            meta = np.load(str(self._meta_path), allow_pickle=True)
            self._ids = list(meta["ids"])
            self._texts = list(meta["texts"])
            self._timestamps = list(meta["timestamps"])

    def _save(self) -> None:
        np.save(str(self._vectors_path), self._vectors)
        np.savez(
            str(self._meta_path),
            ids=np.array(self._ids, dtype=object),
            texts=np.array(self._texts, dtype=object),
            timestamps=np.array(self._timestamps, dtype=np.float64),
        )

    def insert(self, ids: list[str], texts: list[str], embeddings: np.ndarray) -> None:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = (embeddings / norms).astype(np.float32)

        self._vectors = np.concatenate([self._vectors, embeddings], axis=0)
        self._ids.extend(ids)
        self._texts.extend(texts)
        self._timestamps.extend([time.time()] * len(ids))
        self._save()

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        n = len(self._ids)
        if n == 0:
            return []
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        query = query.astype(np.float32)

        scores = cosine_scores(self._vectors, query)
        k = min(k, n)
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [
            SearchResult(
                id=self._ids[i],
                text=self._texts[i],
                score=float(scores[i]),
                metadata={"index": int(i)},
            )
            for i in top_idx
        ]

    def get(self, id: str) -> SearchResult | None:
        try:
            idx = self._ids.index(id)
        except ValueError:
            return None
        return SearchResult(
            id=id,
            text=self._texts[idx],
            score=1.0,
            metadata={
                "timestamp": self._timestamps[idx],
                "embedding": self._vectors[idx],
            },
        )

    def count(self) -> int:
        return len(self._ids)

    def close(self) -> None:
        pass
