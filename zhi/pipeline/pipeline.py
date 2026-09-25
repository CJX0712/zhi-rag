"""End-to-end orchestration: ingest -> index -> retrieve -> generate.

ZhiPipeline wires the single-responsibility modules into a runnable chain and
handles persistence (index + embedder state) so a built index can be reloaded
and queried without re-embedding.
"""
import json
import os
import time

from ..config import ZhiConfig
from ..embedder import embedder as embedder_mod
from ..errors import ZDError, ZDCode
from ..generator import generator as generator_mod
from ..image import image as image_mod
from ..loader import loader as loader_mod
from ..retriever import retriever as retriever_mod
from ..splitter import splitter as splitter_mod
from ..types import Chunk, Answer, IndexStats
from ..vectorstore import vectorstore as vs_mod

IMAGE_EXTS = image_mod.IMAGE_EXTS


class ZhiPipeline:
    def __init__(self, config: ZhiConfig = None):
        self.cfg = config or ZhiConfig()
        self.embedder = None
        self.retriever = None
        self.vs = None
        self.generator = generator_mod.get_generator(self.cfg)
        self._manifest = None

    # ---- ingestion -------------------------------------------------------
    def ingest(self, paths, persist_dir: str = None) -> IndexStats:
        persist = persist_dir or self.cfg.persist_dir
        files = loader_mod.iter_files(paths)
        if not files:
            raise ZDError(ZDCode.E_INPUT, "no input files found", str(paths))

        chunks = []
        for fp in files:
            ext = os.path.splitext(fp)[1].lower()
            if ext in IMAGE_EXTS:
                chunks.append(image_mod.ImageIngestor().ingest(fp))
            else:
                text = loader_mod.load_file(fp)
                sp = splitter_mod.split_text(text, self.cfg.chunk_size, self.cfg.chunk_overlap)
                doc_id = os.path.basename(fp)
                for i, s in enumerate(sp):
                    chunks.append(
                        Chunk(id=f"{doc_id}#{i}", doc_id=doc_id, text=s, source=fp, modality="text")
                    )
        if not chunks:
            raise ZDError(ZDCode.E_LOAD, "no content extracted", str(paths))

        t0 = time.time()
        embedder, kind = embedder_mod.get_embedder(self.cfg)
        vs = vs_mod.VectorStore(backend="auto")
        retriever = retriever_mod.HybridRetriever(embedder=embedder, vectorstore=vs, use_dense=True)

        # Build vectors (CLIP images use image vectors; everything else uses text embeddings)
        if hasattr(embedder, "encode_image"):
            vecs = []
            for c in chunks:
                if c.modality == "image":
                    v = image_mod.ImageIngestor(embedder).embed(c)
                    vecs.append(v if v is not None else None)
                else:
                    vecs.append(None)
            none_idx = [i for i, v in enumerate(vecs) if v is None]
            if none_idx:
                embs = embedder.embed_documents([chunks[i].text for i in none_idx])
                for j, i in enumerate(none_idx):
                    vecs[i] = embs[j]
            vs.add(vecs, chunks)
        else:
            texts = [c.text for c in chunks]
            vs.add(embedder.embed_documents(texts), chunks)

        retriever.bm25.index(chunks)  # build lexical index without re-adding vectors
        retriever.chunks = chunks
        build_ms = (time.time() - t0) * 1000

        os.makedirs(persist, exist_ok=True)
        base = os.path.join(persist, "index")
        vs.save(base)
        if kind == "lsi":
            embedder.save(os.path.join(persist, "embedder.pkl"))
        manifest = {
            "embedder_kind": kind,
            "dim": vs.dim,
            "backend": vs.backend,
            "n_chunks": len(chunks),
            "config": self.cfg.to_dict(),
        }
        with open(os.path.join(persist, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        self.vs = vs
        self.embedder = embedder
        self.retriever = retriever
        self._manifest = manifest
        return IndexStats(
            n_docs=len({c.doc_id for c in chunks}),
            n_chunks=len(chunks),
            dim=vs.dim,
            build_ms=round(build_ms, 1),
            embedder=kind,
        )

    # ---- load ------------------------------------------------------------
    def load(self, persist_dir: str = None):
        persist = persist_dir or self.cfg.persist_dir
        mpath = os.path.join(persist, "manifest.json")
        if not os.path.exists(mpath):
            raise ZDError(ZDCode.E_INDEX, "index not found; run ingest first", persist)
        with open(mpath, "r", encoding="utf-8") as f:
            man = json.load(f)
        vs = vs_mod.VectorStore.load(os.path.join(persist, "index"))
        kind = man["embedder_kind"]
        if kind == "lsi":
            embedder = embedder_mod.TfidfEmbedder.load(os.path.join(persist, "embedder.pkl"))
        else:
            embedder, _ = embedder_mod.get_embedder(self.cfg)
        retriever = retriever_mod.HybridRetriever(embedder=embedder, vectorstore=vs, use_dense=True)
        retriever.bm25.index(vs.chunks)
        retriever.chunks = vs.chunks
        self.vs = vs
        self.embedder = embedder
        self.retriever = retriever
        self._manifest = man
        return man

    # ---- query -----------------------------------------------------------
    def query(self, question: str, top_k: int = None, mode: str = None) -> Answer:
        if self.retriever is None:
            self.load()
        top_k = top_k or self.cfg.top_k
        mode = mode or self.cfg.retriever_mode
        if not question or not question.strip():
            raise ZDError(ZDCode.E_INPUT, "empty question", None)

        t0 = time.time()
        contexts = self.retriever.search(question, top_k, mode)
        answer = self.generator.generate(question, contexts)
        dt = (time.time() - t0) * 1000
        return Answer(
            answer=answer,
            contexts=contexts,
            latency_ms=round(dt, 2),
            mode=mode,
            backend=self.generator.name,
        )
