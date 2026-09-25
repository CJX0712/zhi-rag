#!/usr/bin/env bash
# One-click demo: install -> ingest sample corpus -> answer questions ->
# run evaluation -> run unit tests -> run performance baseline.
# Zero manual intervention; the offline MockGenerator + LSI embedder need no
# model download.
set -e

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJ/.venv"
if [ -f "$VENV/Scripts/python.exe" ]; then PY="$VENV/Scripts/python.exe"; else PY="$VENV/bin/python"; fi

"$PY" -m pip install -r "$PROJ/requirements.lock.txt" -q
cd "$PROJ"

echo "==> [1/5] ingest sample corpus (text + image)"
"$PY" -m zhi.api.cli ingest data --persist-dir .zhi_index

echo "==> [2/5] ask (text QA)"
"$PY" -m zhi.api.cli ask "什么是检索增强生成 RAG？它解决了什么问题？" --top-k 3

echo "==> [3/5] ask (multimodal: cite the sample image)"
"$PY" -m zhi.api.cli ask "请引用 sample_neural_network_diagram 图像相关的资料" --top-k 3

echo "==> [4/5] evaluation (built-in offline metrics)"
"$PY" -m zhi.api.cli eval --dataset examples/qa_pairs.json --k 5

echo "==> [5/5] unit tests + performance baseline"
"$PY" -m pytest -q
"$PY" scripts/baseline.py
