"""Core data types shared across modules."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    source: str
    modality: str = "text"  # "text" | "image"
    meta: dict = field(default_factory=dict)


@dataclass
class Context:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    answer: str
    contexts: List[Context]
    latency_ms: float
    mode: str
    backend: str


@dataclass
class IndexStats:
    n_docs: int
    n_chunks: int
    dim: int
    build_ms: float
    embedder: str


@dataclass
class Metrics:
    recall_at_k: float
    mean_latency_ms: float
    p95_latency_ms: float
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    notes: str = ""
