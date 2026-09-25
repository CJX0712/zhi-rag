"""Regression test for loading a DENSE-built index after a fresh process start.

Exposes the bug where `load()` rebuilt the embedder from cfg.embedder (default
"lsi") instead of the dense kind recorded in the manifest, causing
"TfidfEmbedder not fitted" on query.

Guarding strategy:
  * Collection-time guard only checks that `sentence-transformers` is importable
    (stable, no network). This avoids flaky network probes during collection.
  * The actual model load happens inside the test body; if the optional dense
    backend / MiniLM model is unavailable it skips there with a clear reason.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Neutralize any SOCKS system proxy (e.g. Windows registry socks4://127.0.0.1:1080)
# that httpx cannot parse; otherwise DenseEmbedder model load crashes on collect.
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ["NO_PROXY"] = "*"

from zhi.config import ZhiConfig  # noqa: E402
from zhi.pipeline.pipeline import ZhiPipeline  # noqa: E402

EXAMPLE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

try:
    import sentence_transformers  # lightweight, no network  # noqa: F401

    _st_installed = True
except Exception:  # pragma: no cover
    _st_installed = False


@pytest.mark.skipif(not _st_installed, reason="sentence-transformers not installed")
def test_load_dense_index_after_fresh_process(tmp_path):
    # Heavy/optional part: only attempt when the dense backend is actually usable.
    try:
        from zhi.embedder.embedder import DenseEmbedder

        DenseEmbedder(EXAMPLE_MODEL)
    except Exception as e:
        pytest.skip(f"dense backend (MiniLM) unavailable: {e}")

    d = tmp_path / "docs"
    d.mkdir()
    (d / "rag.txt").write_text("检索增强生成 RAG 结合检索与生成，缓解幻觉问题。", encoding="utf-8")
    (d / "tf.txt").write_text("Transformer 使用自注意力机制实现并行计算。", encoding="utf-8")
    idx = str(tmp_path / "idx")

    # Build the index with the DENSE embedder.
    ZhiPipeline(ZhiConfig(persist_dir=idx, embedder="dense")).ingest([str(d)])

    # Fresh pipeline with DEFAULT config (embedder defaults to "lsi") — this is
    # the exact scenario that previously blew up. load() must honor the manifest.
    p2 = ZhiPipeline(ZhiConfig(persist_dir=idx))
    p2.load()
    assert p2.embedder is not None
    ans = p2.query("Transformer 是什么", top_k=3)
    assert ans.contexts
    assert any(c.chunk.source.endswith("tf.txt") for c in ans.contexts)
