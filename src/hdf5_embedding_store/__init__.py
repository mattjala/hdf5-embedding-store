"""HDF5-backed embedding store backends."""
from .base import EmbeddingStore, SearchResult, cosine_scores
from .hdf5_store import HDF5EmbeddingStore
from .hdf5_cached_store import HDF5CachedStore
from .hdf5_int8_store import HDF5Int8Store
from .hdf5_ivf_store import HDF5IVFStore
from .hdf5_float16_store import HDF5Float16Store
from .sqlite_store import SQLiteEmbeddingStore
from .numpy_store import NumpyEmbeddingStore

try:
    from .hdf5_blosc_store import HDF5BloscStore
except ImportError:
    HDF5BloscStore = None  # type: ignore[assignment,misc]

__all__ = [
    "EmbeddingStore",
    "SearchResult",
    "cosine_scores",
    "HDF5EmbeddingStore",
    "HDF5CachedStore",
    "HDF5Int8Store",
    "HDF5IVFStore",
    "HDF5Float16Store",
    "HDF5BloscStore",
    "SQLiteEmbeddingStore",
    "NumpyEmbeddingStore",
]
