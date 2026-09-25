"""Minimal runnable example: ingest -> query, in pure Python (offline default).

Run:  python examples/minimal_example.py
Requires: the core dependencies (see requirements.lock.txt).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zhi.config import ZhiConfig
from zhi.pipeline.pipeline import ZhiPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def main():
    cfg = ZhiConfig(
        persist_dir="./.zhi_index_example",
        embedder="lsi",
        retriever_mode="hybrid",
        generator="mock",
    )
    p = ZhiPipeline(cfg)
    stats = p.ingest([DATA])
    print("ingest:", stats)

    ans = p.query("什么是检索增强生成 RAG？", top_k=3)
    print(f"[mode={ans.mode} backend={ans.backend} {ans.latency_ms}ms]")
    print(ans.answer)


if __name__ == "__main__":
    main()
