"""Performance baseline + SOTA comparison for ZhiDa.

Measures, on the sample text corpus:
  * ingest throughput (chunks/s) and build time
  * embed-query latency
  * retrieval latency for bm25 / dense(LSI) / hybrid
  * end-to-end latency (retrieve + MockGenerator)
  * source-level recall@k (built-in, no LLM)

Then, best-effort, repeats core metrics with the optional MiniLM dense
embedder (sentence-transformers) for SOTA comparison; if the optional
dependency or model is unavailable it is reported as skipped.
"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from zhi.config import ZhiConfig  # noqa: E402
from zhi.eval.eval import run_builtin  # noqa: E402
from zhi.pipeline.pipeline import ZhiPipeline  # noqa: E402

DATA = os.path.join(ROOT, "data", "docs")
QA = os.path.join(ROOT, "examples", "qa_pairs.json")


def mean_latency(fn, n_warm=1):
    for _ in range(n_warm):
        fn()
    t0 = time.time()
    for _ in range(n_warm):
        fn()
    return (time.time() - t0) / max(1, n_warm) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--qa", default=QA)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    # Ingest with a relative path so stored sources match qa_pairs.json relevance.
    os.chdir(ROOT)
    data_rel = os.path.join("data", "docs")

    queries = [q["question"] for q in json.load(open(args.qa, encoding="utf-8"))["qa_pairs"]]

    tmp = tempfile.mkdtemp(prefix="zhi_baseline_")
    cfg = ZhiConfig(persist_dir=tmp, embedder="lsi", retriever_mode="hybrid")
    p = ZhiPipeline(cfg)
    t0 = time.time()
    stats = p.ingest([data_rel])
    build_s = time.time() - t0

    emb_lat = np.mean([mean_latency(lambda: p.embedder.embed_query(q), 1) for q in queries])
    bm25_lat = np.mean([mean_latency(lambda: p.query(q, top_k=args.k, mode="bm25"), 1) for q in queries])
    dense_lat = np.mean([mean_latency(lambda: p.query(q, top_k=args.k, mode="dense"), 1) for q in queries])
    hybrid_lat = np.mean([mean_latency(lambda: p.query(q, top_k=args.k, mode="hybrid"), 1) for q in queries])
    e2e = np.mean([mean_latency(lambda: p.query(q, top_k=args.k, mode="hybrid"), 1) for q in queries])
    m = run_builtin(p, args.qa, args.k)

    dense_metrics = None
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # triggers cache check
        tmp2 = tempfile.mkdtemp(prefix="zhi_dense_")
        p2 = ZhiPipeline(ZhiConfig(persist_dir=tmp2, embedder="dense", retriever_mode="hybrid"))
        p2.ingest([data_rel])
        dm = run_builtin(p2, args.qa, args.k)
        dense_metrics = {
            "recall_at_k": dm.recall_at_k,
            "mean_latency_ms": dm.mean_latency_ms,
            "p95_latency_ms": dm.p95_latency_ms,
        }
    except Exception as e:  # optional dep / model not available
        dense_metrics = {"skipped": str(e)[:160]}

    report = {
        "offline_lsi_default": {
            "n_docs": stats.n_docs,
            "n_chunks": stats.n_chunks,
            "dim": stats.dim,
            "build_seconds": round(build_s, 3),
            "ingest_throughput_chunks_per_s": round(stats.n_chunks / build_s, 1),
            "embed_query_ms": round(float(emb_lat), 2),
            "retrieval_latency_ms": {
                "bm25": round(float(bm25_lat), 2),
                "dense_lsi": round(float(dense_lat), 2),
                "hybrid": round(float(hybrid_lat), 2),
            },
            "end_to_end_ms": round(float(e2e), 2),
            "recall_at_k": m.recall_at_k,
            "p95_latency_ms": m.p95_latency_ms,
        },
        "optional_dense_minilm": dense_metrics,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    o = report["offline_lsi_default"]
    print("\n=== Performance Baseline (offline LSI default) ===")
    print(
        f"docs={o['n_docs']} chunks={o['n_chunks']} dim={o['dim']} "
        f"build={o['build_seconds']}s throughput={o['ingest_throughput_chunks_per_s']}/s"
    )
    print(
        f"embed_query={o['embed_query_ms']}ms | bm25={o['retrieval_latency_ms']['bm25']}ms "
        f"dense_lsi={o['retrieval_latency_ms']['dense_lsi']}ms hybrid={o['retrieval_latency_ms']['hybrid']}ms"
    )
    print(
        f"end_to_end(mock)={o['end_to_end_ms']}ms | recall@{args.k}={o['recall_at_k']} "
        f"p95={o['p95_latency_ms']}ms"
    )
    dm2 = report["optional_dense_minilm"]
    if isinstance(dm2, dict) and "recall_at_k" in dm2:
        print(
            f"optional MiniLM dense: recall@{args.k}={dm2['recall_at_k']} "
            f"mean_lat={dm2['mean_latency_ms']}ms p95={dm2['p95_latency_ms']}ms"
        )
    else:
        print(f"optional MiniLM dense: SKIPPED ({dm2.get('skipped')})")


if __name__ == "__main__":
    main()
