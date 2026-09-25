"""Embedding layer.

Two implementations sharing one interface (embed_documents / embed_query):
  * TfidfEmbedder  -> LSI (TF-IDF + TruncatedSVD). OFFLINE, zero model download.
                      This is the DEFAULT "dense" path so hybrid retrieval works
                      with no network/model dependency.
  * DenseEmbedder  -> sentence-transformers (optional, SOTA). Falls back to LSI
                      automatically when the optional dependency is absent.

All vectors are L2-normalized so that inner-product == cosine similarity.
"""
import pickle
import warnings

import numpy as np

from ..errors import ZDError, ZDCode


def _normalize(mat):
    mat = np.asarray(mat, dtype=np.float32)
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (n + 1e-9)


class TfidfEmbedder:
    kind = "lsi"

    def __init__(self, dim: int = 256):
        self.dim = dim
        self._fitted = False

    def _build(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer()
        X = vec.fit_transform(texts)
        n_feat = X.shape[1]
        eff = min(self.dim, max(2, n_feat - 1))  # SVD components < n_features
        svd = TruncatedSVD(n_components=eff, random_state=0)
        svd.fit(X)
        self.vec = vec
        self.svd = svd
        self.dim = eff
        self._fitted = True

    def embed_documents(self, texts):
        if not self._fitted:
            self._build(list(texts))
        X = self.vec.transform(list(texts))
        return _normalize(self.svd.transform(X))

    def embed_query(self, text: str):
        if not self._fitted:
            raise ZDError(ZDCode.E_EMBED, "TfidfEmbedder not fitted; call ingest first")
        X = self.vec.transform([text])
        return _normalize(self.svd.transform(X))[0]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"vec": self.vec, "svd": self.svd, "dim": self.dim}, f)

    @classmethod
    def load(cls, path: str) -> "TfidfEmbedder":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls()
        obj.vec = d["vec"]
        obj.svd = d["svd"]
        obj.dim = d["dim"]
        obj._fitted = True
        return obj


class DenseEmbedder:
    kind = "dense"

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ZDError(ZDCode.E_EMBED, "sentence-transformers unavailable (optional dep)", str(e))
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # get_sentence_embedding_dimension renamed to get_embedding_dimension in newer ST
        try:
            self.dim = self.model.get_embedding_dimension()
        except AttributeError:
            self.dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts):
        return _normalize(self.model.encode(list(texts), normalize_embeddings=True))

    def embed_query(self, text: str):
        return _normalize(self.model.encode([text], normalize_embeddings=True))[0]

    def encode_image(self, image_path: str):
        """CLIP-family models can embed images into the shared text-image space."""
        from PIL import Image  # optional; only needed for true image vectors

        img = Image.open(image_path).convert("RGB")
        return _normalize(self.model.encode(img, normalize_embeddings=True))[0]

    def save(self, path: str) -> None:  # dense models reload from name
        pass

    @classmethod
    def load(cls, path: str, model_name: str) -> "DenseEmbedder":
        return cls(model_name)


def get_embedder(cfg):
    """Return (embedder, kind). 'dense' falls back to LSI if optional dep missing."""
    if cfg.embedder == "dense":
        try:
            return DenseEmbedder(cfg.dense_model), "dense"
        except ZDError as e:
            warnings.warn(f"dense embedder unavailable, falling back to LSI: {e.msg}")
            return TfidfEmbedder(), "lsi"
    return TfidfEmbedder(), "lsi"
