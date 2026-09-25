"""Evaluation harness.

Default (offline, zero LLM): source-level recall@k + end-to-end latency
statistics. A ragas adapter (faithfulness / answer_relevancy / context
precision / recall) is available when `ragas` + an LLM backend are installed.
"""
import json
import os
import time

import numpy as np

from ..types import Metrics


def recall_at_k(relevant, retrieved, k):
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & set(relevant)) / len(set(relevant))


def run_builtin(pipeline, dataset_path: str, k: int = 5) -> Metrics:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = data.get("qa_pairs", data if isinstance(data, list) else [])
    latencies = []
    recalls = []
    for item in pairs:
        q = item["question"]
        rel = {os.path.normpath(r) for r in item.get("relevant_sources", [])}
        t0 = time.time()
        ans = pipeline.query(q, top_k=k)
        latencies.append(ans.latency_ms)
        retrieved = [os.path.normpath(c.chunk.source) for c in ans.contexts]
        recalls.append(recall_at_k(rel, retrieved, k))
    arr = np.array(latencies)
    return Metrics(
        recall_at_k=round(float(np.mean(recalls)), 4),
        mean_latency_ms=round(float(arr.mean()), 2),
        p95_latency_ms=round(float(np.percentile(arr, 95)), 2),
        notes="built-in offline metrics (no LLM); recall@k over source-level relevance",
    )


def run_ragas(pipeline, dataset_path: str, k: int = 5):
    """Optional SOTA RAG evaluation via ragas.

    Requires `ragas` and a configured generator (ollama/openai). Returns a dict
    of ragas metrics. Skipped automatically if dependencies are absent.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"ragas not installed (optional): {e}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = data.get("qa_pairs", [])

    rows = []
    for item in pairs:
        q = item["question"]
        ans = pipeline.query(q, top_k=k)
        rows.append(
            {
                "question": q,
                "answer": ans.answer,
                "contexts": [[c.chunk.text for c in ans.contexts]],
                "reference": item.get("reference", ""),
            }
        )
    ds = Dataset.from_list([{k: v for k, v in r.items()} for r in rows])
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return dict(result)
