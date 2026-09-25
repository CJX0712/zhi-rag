import json

from zhi.config import ZhiConfig
from zhi.eval.eval import recall_at_k, run_builtin
from zhi.pipeline.pipeline import ZhiPipeline


def _setup(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "rag.txt").write_text("检索增强生成 RAG 结合检索与生成，缓解幻觉。", encoding="utf-8")
    (d / "tf.txt").write_text("Transformer 使用自注意力机制。", encoding="utf-8")
    ds = tmp_path / "qa.json"
    ds.write_text(
        json.dumps({"qa_pairs": [{"question": "什么是 RAG？", "relevant_sources": [str(d / "rag.txt")]}]}),
        encoding="utf-8",
    )
    return str(d), str(tmp_path / "idx"), str(ds)


def test_recall_at_k():
    # top-2 retrieved = ["b","c"]; relevant = {"a","b"} -> only "b" inside top-2 -> 1/2
    assert recall_at_k(["a", "b"], ["b", "c", "a"], 2) == 0.5
    assert recall_at_k(["a", "b"], ["a", "b", "c"], 2) == 1.0
    assert recall_at_k(["a"], ["x", "y"], 2) == 0.0


def test_run_builtin(tmp_path):
    src, idx, ds = _setup(tmp_path)
    p = ZhiPipeline(ZhiConfig(persist_dir=idx))
    p.ingest([src])
    m = run_builtin(p, ds, k=3)
    assert 0.0 <= m.recall_at_k <= 1.0
    assert m.mean_latency_ms >= 0
