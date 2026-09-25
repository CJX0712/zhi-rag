"""Hybrid retriever: BM25 + Dense, fused with RRF."""
from .bm25 import BM25Retriever
from .rrf import reciprocal_rank_fusion
from ..types import Context


class HybridRetriever:
    def __init__(self, embedder=None, vectorstore=None, use_dense: bool = True):
        self.bm25 = BM25Retriever()
        self.embedder = embedder
        self.vs = vectorstore
        self.use_dense = use_dense
        self.chunks = []

    def index(self, chunks):
        self.chunks = list(chunks)
        self.bm25.index(chunks)
        if self.use_dense and self.embedder is not None and self.vs is not None:
            texts = [c.text for c in chunks]
            vecs = self.embedder.embed_documents(texts)
            self.vs.add(vecs, chunks)

    def search(self, query: str, top_k: int, mode: str = "hybrid"):
        if mode == "bm25" or not self.use_dense:
            return self.bm25.search(query, top_k)

        if mode == "dense":
            q = self.embedder.embed_query(query)
            I, D = self.vs.search(q, top_k)
            return [Context(chunk=self.vs.chunks[i], score=float(d)) for i, d in zip(I, D)]

        # hybrid: fuse BM25 and Dense rankings via RRF
        bm = self.bm25.search(query, top_k)
        q = self.embedder.embed_query(query)
        I, D = self.vs.search(q, top_k)
        dense = [Context(chunk=self.vs.chunks[i], score=float(d)) for i, d in zip(I, D)]

        r1 = [c.chunk.id for c in bm]
        r2 = [c.chunk.id for c in dense]
        fused = reciprocal_rank_fusion([r1, r2])
        cmap = {c.chunk.id: c for c in (bm + dense)}
        out = []
        for cid, score in fused[:top_k]:
            ctx = cmap[cid]
            ctx.score = round(score, 6)
            out.append(ctx)
        return out
