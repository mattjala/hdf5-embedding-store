"""Correctness tests for all embedding store backends."""
from __future__ import annotations

import numpy as np
import pytest

from hdf5_embedding_store.base import cosine_scores

DIM = 16
N = 50


# ---------------------------------------------------------------------------
# HDF5 (float32, gzip)
# ---------------------------------------------------------------------------

def test_hdf5_insert_search_get(tmp_path, embeddings, ids, texts):
    from hdf5_embedding_store import HDF5EmbeddingStore
    p = tmp_path / "test.h5"
    with HDF5EmbeddingStore(p, dimension=DIM) as store:
        store.insert(ids, texts, embeddings)
        assert store.count() == N

        results = store.search(embeddings[0].copy(), k=5)
        assert len(results) == 5
        assert results[0].id == "id_0000"
        assert results[0].score > 0.99

        rec = store.get("id_0000")
        assert rec is not None
        assert rec.text == "text 0"
        assert store.get("nonexistent") is None


def test_hdf5_append(tmp_path, embeddings, ids, texts):
    from hdf5_embedding_store import HDF5EmbeddingStore
    p = tmp_path / "test.h5"
    half = N // 2
    with HDF5EmbeddingStore(p, dimension=DIM) as store:
        store.insert(ids[:half], texts[:half], embeddings[:half])
        store.insert(ids[half:], texts[half:], embeddings[half:])
        assert store.count() == N


def test_hdf5_uncompressed(tmp_path, embeddings, ids, texts):
    from hdf5_embedding_store import HDF5EmbeddingStore
    p = tmp_path / "test_unc.h5"
    with HDF5EmbeddingStore(p, dimension=DIM, compression=None) as store:
        store.insert(ids, texts, embeddings)
        results = store.search(embeddings[5], k=3)
        assert results[0].id == "id_0005"


# ---------------------------------------------------------------------------
# HDF5 Cached
# ---------------------------------------------------------------------------

def test_hdf5_cached_insert_search_get(tmp_path, embeddings, ids, texts):
    from hdf5_embedding_store import HDF5CachedStore
    p = tmp_path / "cached.h5"
    with HDF5CachedStore(p, dimension=DIM) as store:
        store.insert(ids, texts, embeddings)
        assert store.count() == N

        results = store.search(embeddings[4], k=5)
        assert results[0].id == "id_0004"
        assert results[0].score > 0.99

        rec = store.get("id_0004")
        assert rec is not None
        assert rec.text == "text 4"
        assert store.get("bad") is None


# ---------------------------------------------------------------------------
# HDF5 Int8
# ---------------------------------------------------------------------------

def test_hdf5_int8_insert_search_get(tmp_path, embeddings, ids, texts):
    from hdf5_embedding_store import HDF5Int8Store
    p = tmp_path / "int8.h5"
    with HDF5Int8Store(p, dimension=DIM) as store:
        store.insert(ids, texts, embeddings)
        assert store.count() == N

        results = store.search(embeddings[9], k=5)
        assert results[0].id == "id_0009"

        rec = store.get("id_0009")
        assert rec is not None
        assert np.allclose(rec.metadata["embedding"], embeddings[9], atol=0.02)


def test_hdf5_int8_gzip_insert_search_get(tmp_path, embeddings, ids, texts):
    from hdf5_embedding_store import HDF5Int8Store
    p = tmp_path / "int8_gzip.h5"
    with HDF5Int8Store(p, dimension=DIM, compression="gzip", compression_opts=4) as store:
        store.insert(ids, texts, embeddings)
        assert store.count() == N

        results = store.search(embeddings[9], k=5)
        assert results[0].id == "id_0009"

        rec = store.get("id_0009")
        assert rec is not None
        assert np.allclose(rec.metadata["embedding"], embeddings[9], atol=0.02)


def test_hdf5_int8_cosine_error(embeddings):
    from hdf5_embedding_store.hdf5_int8_store import quantize, int8_cosine_scores
    i8, scales = quantize(embeddings)
    query = embeddings[0]
    approx = int8_cosine_scores(i8, scales, query)
    exact = embeddings @ query
    assert np.abs(approx - exact).max() < 0.05


# ---------------------------------------------------------------------------
# HDF5 Float16
# ---------------------------------------------------------------------------

def test_hdf5_float16_insert_search_get(tmp_path, embeddings, ids, texts):
    from hdf5_embedding_store import HDF5Float16Store
    p = tmp_path / "float16.h5"
    with HDF5Float16Store(p, dimension=DIM) as store:
        store.insert(ids, texts, embeddings)
        assert store.count() == N

        results = store.search(embeddings[3], k=5)
        assert results[0].id == "id_0003"
        assert results[0].score > 0.99

        rec = store.get("id_0003")
        assert rec is not None
        # float16 precision loss should be < 0.001 per element at unit-norm
        assert np.allclose(rec.metadata["embedding"], embeddings[3], atol=0.005)


def test_hdf5_float16_cosine_error(embeddings):
    """float16 precision loss on unit-norm cosine similarity should be tiny."""
    f16 = embeddings.astype(np.float16).astype(np.float32)
    query = embeddings[0]
    approx = f16 @ query
    exact = embeddings @ query
    assert np.abs(approx - exact).max() < 0.005


# ---------------------------------------------------------------------------
# HDF5 Blosc
# ---------------------------------------------------------------------------

def test_hdf5_blosc_insert_search_get(tmp_path, embeddings, ids, texts):
    pytest.importorskip("hdf5plugin")
    from hdf5_embedding_store import HDF5BloscStore
    p = tmp_path / "blosc.h5"
    with HDF5BloscStore(p, dimension=DIM) as store:
        store.insert(ids, texts, embeddings)
        assert store.count() == N

        results = store.search(embeddings[6], k=5)
        assert results[0].id == "id_0006"
        assert results[0].score > 0.99

        rec = store.get("id_0006")
        assert rec is not None


# ---------------------------------------------------------------------------
# HDF5 IVF
# ---------------------------------------------------------------------------

def test_hdf5_ivf_insert_search(tmp_path, embeddings, ids, texts):
    from hdf5_embedding_store import HDF5IVFStore
    p = tmp_path / "ivf.h5"
    with HDF5IVFStore(p, dimension=DIM, n_clusters=8, n_probe=4) as store:
        store.insert(ids, texts, embeddings)
        assert store.count() == N

        results = store.search(embeddings[0], k=5)
        assert len(results) > 0
        top_ids = [r.id for r in results]
        assert "id_0000" in top_ids


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

def test_sqlite_insert_search_get(tmp_path, embeddings, ids, texts):
    from hdf5_embedding_store import SQLiteEmbeddingStore
    p = tmp_path / "test.db"
    with SQLiteEmbeddingStore(p, dimension=DIM) as store:
        store.insert(ids, texts, embeddings)
        assert store.count() == N

        results = store.search(embeddings[3], k=5)
        assert results[0].id == "id_0003"

        rec = store.get("id_0003")
        assert rec is not None
        assert store.get("bad") is None


# ---------------------------------------------------------------------------
# NumPy
# ---------------------------------------------------------------------------

def test_numpy_insert_search_get(tmp_path, embeddings, ids, texts):
    from hdf5_embedding_store import NumpyEmbeddingStore
    p = tmp_path / "test"
    with NumpyEmbeddingStore(p, dimension=DIM) as store:
        store.insert(ids, texts, embeddings)
        assert store.count() == N

        results = store.search(embeddings[7], k=5)
        assert results[0].id == "id_0007"

        rec = store.get("id_0007")
        assert rec is not None


def test_numpy_persistence(tmp_path, embeddings, ids, texts):
    from hdf5_embedding_store import NumpyEmbeddingStore
    p = tmp_path / "persist"
    store = NumpyEmbeddingStore(p, dimension=DIM)
    store.insert(ids, texts, embeddings)
    store.close()

    store2 = NumpyEmbeddingStore(p, dimension=DIM)
    assert store2.count() == N
    store2.close()


# ---------------------------------------------------------------------------
# cosine_scores utility
# ---------------------------------------------------------------------------

def test_cosine_scores_identity():
    v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    score = cosine_scores(v, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.isclose(score[0], 1.0)


def test_cosine_scores_orthogonal():
    v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    score = cosine_scores(v, np.array([1.0, 0.0], dtype=np.float32))
    assert np.isclose(score[0], 1.0)
    assert np.isclose(score[1], 0.0)
