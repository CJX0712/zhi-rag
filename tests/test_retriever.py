from zhi.embedder.embedder import TfidfEmbedder
from zhi.retriever.retriever import HybridRetriever
from zhi.types import Chunk
from zhi.vectorstore.vectorstore import VectorStore


def _chunks():
    return [
        Chunk(id="1", doc_id="a", text="检索增强生成 RAG 结合检索与生成", source="a"),
        Chunk(id="2", doc_id="b", text="Transformer 使用自注意力机制", source="b"),
        Chunk(id="3", doc_id="c", text="多模态学习融合文本与图像", source="c"),
    ]


def _build():
    chunks = _chunks()
    e = TfidfEmbedder(dim=16)
    vecs = e.embed_documents([c.text for c in chunks])
    vs = VectorStore(backend="auto")
    vs.add(vecs, chunks)
    r = HybridRetriever(embedder=e, vectorstore=vs, use_dense=True)
    r.bm25.index(chunks)
    return r


def test_bm25_mode():
    r = _build()
    out = r.search("RAG 检索", 3, "bm25")
    assert out[0].chunk.id == "1"


def test_hybrid_mode():
    r = _build()
    out = r.search("Transformer 注意力", 3, "hybrid")
    assert len(out) == 3 and out[0].chunk.id == "2"


def test_dense_mode():
    r = _build()
    out = r.search("多模态 图像", 3, "dense")
    assert out[0].chunk.id == "3"
