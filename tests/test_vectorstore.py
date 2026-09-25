import numpy as np

from zhi.types import Chunk
from zhi.vectorstore.vectorstore import VectorStore


def _chunks(n):
    return [Chunk(id=f"c{i}", doc_id=f"d{i}", text=f"t{i}", source=f"s{i}") for i in range(n)]


def test_faiss_search_returns_self_first():
    np.random.seed(0)
    n, dim = 20, 16
    vecs = np.random.rand(n, dim).astype("float32")
    vs = VectorStore(backend="faiss")
    vs.add(vecs, _chunks(n))
    I, _ = vs.search(vecs[3], 5)
    assert len(I) == 5 and I[0] == 3


def test_numpy_fallback():
    np.random.seed(1)
    n, dim = 10, 8
    vecs = np.random.rand(n, dim).astype("float32")
    vs = VectorStore(backend="numpy")
    vs.add(vecs, _chunks(n))
    I, _ = vs.search(vecs[2], 3)
    assert I[0] == 2


def test_save_load(tmp_path):
    np.random.seed(2)
    n, dim = 12, 8
    vecs = np.random.rand(n, dim).astype("float32")
    vs = VectorStore(backend="auto")
    vs.add(vecs, _chunks(n))
    base = str(tmp_path / "idx")
    vs.save(base)
    vs2 = VectorStore.load(base)
    assert vs2.backend == vs.backend
    I, _ = vs2.search(vecs[0], 4)
    assert I[0] == 0
