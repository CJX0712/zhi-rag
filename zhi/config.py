"""Configuration model for ZhiDa (dataclass + YAML persistence)."""
from dataclasses import dataclass, asdict
from typing import Optional
import os

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class ZhiConfig:
    # embedder: "lsi" (offline default, zero-download) | "dense" (sentence-transformers, optional)
    embedder: str = "lsi"
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model: Optional[str] = "sentence-transformers/clip-ViT-B-32"

    chunk_size: int = 500
    chunk_overlap: int = 80

    top_k: int = 5
    # retriever_mode: "bm25" | "dense" | "hybrid"
    retriever_mode: str = "hybrid"

    # generator: "mock" (offline default) | "ollama" | "openai"
    generator: str = "mock"
    ollama_model: str = "qwen2.5:0.5b"
    ollama_base_url: str = "http://localhost:11434"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None

    persist_dir: str = "./.zhi_index"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        if yaml is None:
            raise RuntimeError("pyyaml not installed; cannot save config")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "ZhiConfig":
        if yaml is None:
            raise RuntimeError("pyyaml not installed; cannot load config")
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
