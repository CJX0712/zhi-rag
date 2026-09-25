"""FastAPI REST service for ZhiDa."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..config import ZhiConfig
from ..errors import ZDError
from ..pipeline.pipeline import ZhiPipeline

_pipe = None


def pipe() -> ZhiPipeline:
    global _pipe
    if _pipe is None:
        _pipe = ZhiPipeline(ZhiConfig())
    return _pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-ingest the bundled sample corpus so /query works out of the box on a
    # fresh deploy (no model download; default LSI embedder + Mock generator).
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(root, "data")
        if os.path.isdir(data_dir) and (pipe().retriever is None or len(pipe().retriever.chunks) == 0):
            pipe().ingest([data_dir])
    except Exception:
        pass  # demo still works after a manual POST /ingest
    yield


app = FastAPI(title="ZhiDa Multimodal RAG API", version="1.0.0", lifespan=lifespan)


class IngestReq(BaseModel):
    paths: list
    persist_dir: str = None


class QueryReq(BaseModel):
    question: str
    top_k: int = None
    mode: str = None


@app.get("/health")
def health():
    p = pipe()
    size = len(p.retriever.chunks) if p.retriever else 0
    backend = p.vs.backend if p.vs else "none"
    return {"status": "ok", "index_size": size, "backend": backend}


@app.post("/ingest")
def ingest(req: IngestReq):
    try:
        stats = pipe().ingest(req.paths, req.persist_dir)
        return stats.__dict__
    except ZDError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())


@app.post("/query")
def query(req: QueryReq):
    try:
        ans = pipe().query(req.question, req.top_k, req.mode)
        return {
            "answer": ans.answer,
            "mode": ans.mode,
            "backend": ans.backend,
            "latency_ms": ans.latency_ms,
            "contexts": [
                {
                    "text": c.chunk.text,
                    "source": c.chunk.source,
                    "score": c.score,
                    "modality": c.chunk.modality,
                }
                for c in ans.contexts
            ],
        }
    except ZDError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
