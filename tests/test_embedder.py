import numpy as np

from zhi.config import ZhiConfig
from zhi.embedder.embedder import TfidfEmbedder, get_embedder


def _docs():
    return [
        "检索增强生成 RAG 结合检索与生成",
        "Transformer 使用自注意力机制",
        "多模态学习融合文本与图像",
    ]


def test_lsi_shape_and_normalized():
    e = TfidfEmbedder(dim=32)
    X = e.embed_documents(_docs())
    assert X.shape[0] == 3
    assert X.shape[1] <= 32
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-4)


def test_query_deterministic():
    e = TfidfEmbedder(dim=32)
    e.embed_documents(_docs())
    a = e.embed_query("RAG 检索")
    b = e.embed_query("RAG 检索")
    assert np.allclose(a, b)


def test_save_load_roundtrip(tmp_path):
    e = TfidfEmbedder(dim=32)
    e.embed_documents(_docs())
    q = e.embed_query("Transformer")
    p = str(tmp_path / "emb.pkl")
    e.save(p)
    e2 = TfidfEmbedder.load(p)
    q2 = e2.embed_query("Transformer")
    assert np.allclose(q, q2)


def test_get_embedder_default_lsi():
    e, kind = get_embedder(ZhiConfig(embedder="lsi"))
    assert kind == "lsi" and e.kind == "lsi"
