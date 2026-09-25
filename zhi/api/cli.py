"""Typer CLI: zhi ingest / ask / serve / eval."""
import json
import time

import typer

from ..config import ZhiConfig
from ..eval.eval import run_builtin
from ..pipeline.pipeline import ZhiPipeline

app = typer.Typer(help="ZhiDa 多模态 RAG 命令行 (author: 晨星)")


@app.command()
def ingest(
    paths: list[str] = typer.Argument(..., help="文件或目录路径"),
    persist_dir: str = typer.Option("./.zhi_index", "--persist-dir"),
    embedder: str = typer.Option("lsi", "--embedder", help="lsi | dense"),
    mode: str = typer.Option("hybrid", "--mode", help="bm25 | dense | hybrid"),
):
    cfg = ZhiConfig(persist_dir=persist_dir, embedder=embedder, retriever_mode=mode)
    p = ZhiPipeline(cfg)
    t0 = time.time()
    stats = p.ingest(paths)
    typer.echo(
        f"ingest 完成: docs={stats.n_docs} chunks={stats.n_chunks} "
        f"dim={stats.dim} embedder={stats.embedder} backend={stats.embedder} "
        f"耗时={time.time()-t0:.2f}s"
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="提问"),
    top_k: int = typer.Option(5, "--top-k"),
    mode: str = typer.Option(None, "--mode"),
    persist_dir: str = typer.Option("./.zhi_index", "--persist-dir"),
):
    cfg = ZhiConfig(persist_dir=persist_dir)
    p = ZhiPipeline(cfg)
    p.load()
    ans = p.query(question, top_k, mode)
    typer.echo(f"[mode={ans.mode} backend={ans.backend} {ans.latency_ms}ms]")
    typer.echo(ans.answer)


@app.command()
def serve(host: str = typer.Option("0.0.0.0", "--host"), port: int = typer.Option(8000, "--port")):
    import os
    import uvicorn

    # Deployment contract: bind 0.0.0.0 and honor the injected PORT env var.
    host = os.environ.get("HOST", host)
    port = int(os.environ.get("PORT", port))
    typer.echo(f"serving on http://{host}:{port}")
    uvicorn.run("zhi.api.server:app", host=host, port=port, reload=False)


@app.command()
def eval(
    dataset: str = typer.Option("examples/qa_pairs.json", "--dataset"),
    persist_dir: str = typer.Option("./.zhi_index", "--persist-dir"),
    k: int = typer.Option(5, "--k"),
):
    cfg = ZhiConfig(persist_dir=persist_dir)
    p = ZhiPipeline(cfg)
    p.load()
    m = run_builtin(p, dataset, k)
    typer.echo(json.dumps(m.__dict__, ensure_ascii=False, indent=2))


def main():
    app()


if __name__ == "__main__":
    main()
