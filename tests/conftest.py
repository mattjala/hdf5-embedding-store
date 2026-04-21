import numpy as np
import pytest

DIM = 16
N = 50


@pytest.fixture
def embeddings():
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((N, DIM)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


@pytest.fixture
def ids():
    return [f"id_{i:04d}" for i in range(N)]


@pytest.fixture
def texts():
    return [f"text {i}" for i in range(N)]
