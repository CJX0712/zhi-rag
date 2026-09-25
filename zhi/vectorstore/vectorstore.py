"""Vector store abstraction over FAISS (preferred) with a numpy fallback.

Reuses faiss-cpu (IndexFlatIP + cosine via L2-normalized inner product).
If faiss cannot be imported, a brute-force numpy backend is used so the
system remains installable/runnable on any Python where faiss has no wheel.
"""
import json
import os
import pickle

import numpy as np


class VectorStore:
    def __init__(self, backend: str = "auto", dim: int = None):
        self.backend = backend  # "faiss" | "numpy" | "auto"
        self.dim = dim
        self.chunks = []
        self._idx = None

    def add(self, vectors, chunks):
        vectors = np.asarray(vectors, dtype=np.float32)
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
        self.dim = vectors.shape[1]
        self.chunks = list(chunks)
        if self.backend in ("auto", "faiss"):
            try:
                import faiss

                idx = faiss.IndexFlatIP(self.dim)
                idx.add(vectors)
                self._idx = idx
                self.backend = "faiss"
                return
            except Exception:
                pass
        self._idx = vectors
        self.backend = "numpy"

    def search(self, vector, k):
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        vector = vector / (np.linalg.norm(vector) + 1e-9)
        if not self.chunks:
            return [], []
        k = min(k, len(self.chunks))
        if self.backend == "faiss":
            D, I = self._idx.search(vector, k)
            return I[0].tolist(), D[0].tolist()
        sims = self._idx @ vector.reshape(-1)
        order = np.argsort(-sims)[:k]
        return order.tolist(), sims[order].tolist()

    def save(self, base: str) -> None:
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        meta = {"backend": self.backend, "dim": self.dim}
        if self.backend == "faiss":
            import faiss

            faiss.write_index(self._idx, base + ".faiss")
        else:
            np.save(base + ".npy", self._idx)
        with open(base + ".chunks", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(base + ".meta", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, base: str) -> "VectorStore":
        with open(base + ".meta", "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(backend=meta["backend"], dim=meta.get("dim"))
        if meta["backend"] == "faiss":
            import faiss

            obj._idx = faiss.read_index(base + ".faiss")
        else:
            obj._idx = np.load(base + ".npy", allow_pickle=False)
        with open(base + ".chunks", "rb") as f:
            obj.chunks = pickle.load(f)
        return obj
