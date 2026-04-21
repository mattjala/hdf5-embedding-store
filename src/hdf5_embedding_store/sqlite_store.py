"""SQLite embedding store — embeddings stored as raw float32 BLOBs."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np

from .base import EmbeddingStore, SearchResult, cosine_scores


class SQLiteEmbeddingStore(EmbeddingStore):
    def __init__(self, path: str | Path, dimension: int = 1536):
        self.path = Path(path)
        self.dimension = dimension
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                rowid   INTEGER PRIMARY KEY AUTOINCREMENT,
                id      TEXT NOT NULL UNIQUE,
                text    TEXT NOT NULL,
                vector  BLOB NOT NULL,
                ts      REAL NOT NULL
            )
        """)
        self._conn.commit()

    def insert(self, ids: list[str], texts: list[str], embeddings: np.ndarray) -> None:
        n = len(ids)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = (embeddings / norms).astype(np.float32)

        ts = time.time()
        rows = [
            (ids[i], texts[i], embeddings[i].tobytes(), ts)
            for i in range(n)
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (id, text, vector, ts) VALUES (?,?,?,?)",
            rows,
        )
        self._conn.commit()

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        query = query.astype(np.float32)

        cur = self._conn.execute("SELECT id, text, vector FROM embeddings")
        rows = cur.fetchall()
        if not rows:
            return []

        ids_list = [r[0] for r in rows]
        texts_list = [r[1] for r in rows]
        matrix = np.frombuffer(
            b"".join(r[2] for r in rows), dtype=np.float32
        ).reshape(len(rows), self.dimension)

        scores = cosine_scores(matrix, query)
        k = min(k, len(rows))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [
            SearchResult(
                id=ids_list[i],
                text=texts_list[i],
                score=float(scores[i]),
                metadata={"index": int(i)},
            )
            for i in top_idx
        ]

    def get(self, id: str) -> SearchResult | None:
        cur = self._conn.execute(
            "SELECT id, text, vector, ts FROM embeddings WHERE id=?", (id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        embedding = np.frombuffer(row[2], dtype=np.float32)
        return SearchResult(
            id=row[0],
            text=row[1],
            score=1.0,
            metadata={"timestamp": row[3], "embedding": embedding},
        )

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM embeddings")
        return cur.fetchone()[0]

    def close(self) -> None:
        self._conn.close()
