from zhi.config import ZhiConfig
from zhi.pipeline.pipeline import ZhiPipeline


def _corpus(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "rag.txt").write_text("检索增强生成 RAG 结合检索与生成，缓解幻觉问题。", encoding="utf-8")
    (d / "tf.txt").write_text("Transformer 使用自注意力机制实现并行计算。", encoding="utf-8")
    return str(d), str(tmp_path / "idx")


def test_ingest_and_query(tmp_path):
    src, idx = _corpus(tmp_path)
    p = ZhiPipeline(ZhiConfig(persist_dir=idx, embedder="lsi"))
    stats = p.ingest([src])
    assert stats.n_chunks >= 2
    ans = p.query("什么是 RAG？", top_k=3)
    assert ans.answer and ans.contexts
    assert ans.latency_ms >= 0


def test_load_then_query(tmp_path):
    src, idx = _corpus(tmp_path)
    ZhiPipeline(ZhiConfig(persist_dir=idx)).ingest([src])
    p2 = ZhiPipeline(ZhiConfig(persist_dir=idx))
    p2.load()
    ans = p2.query("Transformer 是什么", top_k=3)
    assert ans.contexts and any(c.chunk.source.endswith("tf.txt") for c in ans.contexts)


def test_image_chunk_in_pipeline(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "rag.txt").write_text("多模态 RAG 可以引用图像资料，例如神经网络结构图。", encoding="utf-8")
    img = tmp_path / "sample_neural_network_diagram.png"
    img.write_bytes(b"1234")
    p = ZhiPipeline(ZhiConfig(persist_dir=str(tmp_path / "idx")))
    p.ingest([str(d), str(img)])
    ans = p.query("sample_neural_network_diagram 图像", top_k=3)
    assert any(c.chunk.modality == "image" for c in ans.contexts)
