"""Lexical retrieval with BM25 (rank_bm25), Chinese-aware tokenization via jieba."""
import re

import jieba
import numpy as np

from ..types import Chunk, Context


def tokenize(text: str) -> list:
    """Whitespace-robust tokenization: jieba for CJK, keep ascii words."""
    toks = []
    for seg in re.split(r"\s+", text):
        seg = seg.strip()
        if not seg:
            continue
        toks.extend(t for t in jieba.lcut(seg) if t.strip())
    return toks


class BM25Retriever:
    def __init__(self):
        self.chunks = []
        self._bm25 = None
        self._tok = []

    def index(self, chunks):
        self.chunks = list(chunks)
        self._tok = [tokenize(c.text) for c in self.chunks]
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi(self._tok)

    def search(self, query: str, k: int):
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        k = min(k, len(self.chunks))
        order = np.argsort(-scores)[:k]
        return [Context(chunk=self.chunks[i], score=float(scores[i])) for i in order]
