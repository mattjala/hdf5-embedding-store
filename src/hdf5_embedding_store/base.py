"""Abstract base class for embedding store backends."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class EmbeddingStore(ABC):
    """Common interface for all embedding store backends."""

    @abstractmethod
    def insert(self, ids: list[str], texts: list[str], embeddings: np.ndarray) -> None:
        """Batch insert records. embeddings: (N, D) float32."""

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        """Exact cosine similarity search. query: (D,) float32."""

    @abstractmethod
    def get(self, id: str) -> SearchResult | None:
        """Fetch a single record by id (embedding + text + metadata)."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored embeddings."""

    @abstractmethod
    def close(self) -> None:
        """Release any open resources."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Vectorized cosine similarity: matrix (N, D), query (D,) → scores (N,).

    Both are expected to already be L2-normalised at insert time.
    """
    return matrix @ query


def timed(fn, *args, **kwargs):
    """Return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0
