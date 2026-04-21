"""HDF5 IVF (Inverted File) embedding store.

Partitions vectors into K clusters using k-means. Each cluster is a
separate HDF5 dataset. At query time only the top n_probe clusters are
read, reducing I/O sub-linearly vs. flat scan.

Trade-off: ~90–95% recall at n_probe/K = 0.1; set n_probe=n_clusters
for exact results.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .base import EmbeddingStore, SearchResult, cosine_scores

DEFAULT_K = 64
DEFAULT_N_PROBE = 8
CHUNK_ROWS = 256


class HDF5IVFStore(EmbeddingStore):
    def __init__(
        self,
        path: str | Path,
        dimension: int = 1536,
        model: str = "synthetic",
        mode: str = "a",
        n_clusters: Optional[int] = None,
        n_probe: int = DEFAULT_N_PROBE,
    ):
        self.path = Path(path)
        self.dimension = dimension
        self.n_clusters = n_clusters
        self.n_probe = n_probe

        self._f = h5py.File(self.path, mode)
        self._indexed = "meta" in self._f

        self._buf_vecs: list[np.ndarray] = []
        self._buf_texts: list[str] = []
        self._buf_ids: list[str] = []

        self._centroids: Optional[np.ndarray] = None
        if self._indexed:
            self._centroids = self._f["meta"]["centroids"][:]

    def _build_index(self, vectors: np.ndarray, texts: list[str], ids: list[str]) -> None:
        n = len(ids)
        K = self.n_clusters or max(4, int(np.sqrt(n)))
        K = min(K, n)

        kmeans = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=42, batch_size=2048)
        labels = kmeans.fit_predict(vectors)
        centroids = kmeans.cluster_centers_.astype(np.float32)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-9)
        centroids = centroids / norms

        meta = self._f.create_group("meta")
        meta.create_dataset("centroids", data=centroids)
        meta.create_dataset("counts", data=np.zeros(K, dtype=np.int64))
        meta.attrs.update({
            "n_clusters": K,
            "n_probe": self.n_probe,
            "dimension": self.dimension,
            "total_count": n,
        })

        vlen_str = h5py.string_dtype(encoding="utf-8")
        clusters_grp = self._f.create_group("clusters")
        ts_now = time.time()

        for ci in range(K):
            mask = labels == ci
            grp = clusters_grp.create_group(f"c_{ci}")
            ni = int(mask.sum())
            chunk_r = min(CHUNK_ROWS, max(1, ni))
            grp.create_dataset(
                "vectors", data=vectors[mask], dtype=np.float32,
                maxshape=(None, self.dimension), chunks=(chunk_r, self.dimension),
            )
            grp.create_dataset(
                "text", data=np.array(np.array(texts)[mask], dtype=object), dtype=vlen_str,
                maxshape=(None,), chunks=(chunk_r,),
            )
            grp.create_dataset(
                "ids", data=np.array(np.array(ids)[mask], dtype=object), dtype=vlen_str,
                maxshape=(None,), chunks=(chunk_r,),
            )
            grp.create_dataset(
                "timestamps", data=np.full(ni, ts_now, dtype=np.float64),
                maxshape=(None,), chunks=(chunk_r,),
            )
            meta["counts"][ci] = ni

        self._centroids = centroids
        self._indexed = True
        self._f.flush()

    def _assign_cluster(self, vectors: np.ndarray) -> np.ndarray:
        scores = vectors @ self._centroids.T
        return np.argmax(scores, axis=1)

    def insert(self, ids: list[str], texts: list[str], embeddings: np.ndarray) -> None:
        n = len(ids)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        unit = (embeddings / norms).astype(np.float32)

        if not self._indexed:
            self._buf_vecs.append(unit)
            self._buf_texts.extend(texts)
            self._buf_ids.extend(ids)

            all_vecs = np.concatenate(self._buf_vecs, axis=0)
            self._build_index(all_vecs, self._buf_texts, self._buf_ids)
            self._buf_vecs.clear()
            self._buf_texts.clear()
            self._buf_ids.clear()
        else:
            labels = self._assign_cluster(unit)
            ts_now = time.time()
            vlen_str = h5py.string_dtype(encoding="utf-8")

            for ci in np.unique(labels):
                mask = labels == ci
                grp = self._f["clusters"][f"c_{ci}"]
                old_n = grp["vectors"].shape[0]
                add_n = int(mask.sum())
                new_n = old_n + add_n

                for ds_name in ("vectors", "text", "ids", "timestamps"):
                    grp[ds_name].resize(new_n, axis=0)

                grp["vectors"][old_n:new_n] = unit[mask]
                grp["text"][old_n:new_n] = np.array(np.array(texts)[mask], dtype=object)
                grp["ids"][old_n:new_n] = np.array(np.array(ids)[mask], dtype=object)
                grp["timestamps"][old_n:new_n] = np.full(add_n, ts_now, dtype=np.float64)
                self._f["meta"]["counts"][ci] += add_n

            old_total = int(self._f["meta"].attrs["total_count"])
            self._f["meta"].attrs["total_count"] = old_total + n
            self._f.flush()

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        if not self._indexed:
            return []

        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        query = query.astype(np.float32)

        centroid_scores = self._centroids @ query
        K = len(centroid_scores)
        n_probe = min(self.n_probe, K)
        probe_clusters = np.argpartition(centroid_scores, -n_probe)[-n_probe:]

        candidates: list[tuple[float, str, str]] = []
        for ci in probe_clusters:
            grp = self._f["clusters"][f"c_{ci}"]
            ni = grp["vectors"].shape[0]
            if ni == 0:
                continue
            vecs = grp["vectors"][:]
            scores = cosine_scores(vecs, query)
            ids_raw = grp["ids"][:]
            texts_raw = grp["text"][:]
            for j in range(ni):
                candidates.append((
                    float(scores[j]),
                    ids_raw[j].decode() if isinstance(ids_raw[j], bytes) else ids_raw[j],
                    texts_raw[j].decode() if isinstance(texts_raw[j], bytes) else texts_raw[j],
                ))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(id=c[1], text=c[2], score=c[0], metadata={"cluster": int(ci)})
            for c in candidates[:k]
        ]

    def get(self, id: str) -> SearchResult | None:
        if not self._indexed:
            return None
        K = int(self._f["meta"].attrs["n_clusters"])
        for ci in range(K):
            grp = self._f["clusters"][f"c_{ci}"]
            ids_raw = grp["ids"][:]
            ids_decoded = [v.decode() if isinstance(v, bytes) else v for v in ids_raw]
            if id in ids_decoded:
                idx = ids_decoded.index(id)
                text = grp["text"][idx]
                embedding = grp["vectors"][idx]
                return SearchResult(
                    id=id,
                    text=text.decode() if isinstance(text, bytes) else text,
                    score=1.0,
                    metadata={"cluster": ci, "embedding": embedding},
                )
        return None

    def count(self) -> int:
        if not self._indexed:
            return sum(len(b) for b in self._buf_ids)
        return int(self._f["meta"].attrs.get("total_count", 0))

    def close(self) -> None:
        if self._f.id.valid:
            self._f.close()
