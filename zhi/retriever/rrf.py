"""Reciprocal Rank Fusion (RRF).

SELF-RESEARCH COMPONENT (justification in docs/architecture.md):
  No single lightweight library cleanly provides a pluggable "BM25 + dense + RRF"
  combination with a backend-agnostic, deterministic contract. RRF is a tiny,
  well-defined formula (Cormack et al., 2009) that fuses multiple ranked lists
  without needing score calibration across heterogeneous retrievers.

  Invariant (unit-tested): the fused score of a document depends ONLY on its rank
  positions across the input rankings, never on document content or the order in
  which the rankings are supplied. Hence fusion is commutative and stable.
"""


def reciprocal_rank_fusion(rankings, k: int = 60):
    """rankings: list of lists of chunk ids (best-first).

    Returns a list of (chunk_id, fused_score) sorted descending by score.
    """
    fused = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: (-x[1], str(x[0])))
