import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import zhi.api.server as srv  # noqa: E402
from zhi.config import ZhiConfig  # noqa: E402
from zhi.pipeline.pipeline import ZhiPipeline  # noqa: E402


def _corpus(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "rag.txt").write_text("检索增强生成 RAG 结合检索与生成。", encoding="utf-8")
    return str(d)


def test_api_ingest_query(tmp_path):
    src = _corpus(tmp_path)
    idx = str(tmp_path / "idx")
    srv._pipe = ZhiPipeline(ZhiConfig(persist_dir=idx, embedder="lsi"))
    c = TestClient(srv.app)
    r = c.post("/ingest", json={"paths": [src], "persist_dir": idx})
    assert r.status_code == 200
    h = c.get("/health")
    assert h.json()["index_size"] > 0
    q = c.post("/query", json={"question": "什么是 RAG？", "top_k": 2})
    assert q.status_code == 200 and q.json()["answer"]
