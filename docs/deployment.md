# 部署（ZhiDa）

作者：**晨星** · 版本 `1.0.0`

## 1. 环境要求

- Python **3.13**（已在 3.13.14 win / linux 验证；`faiss-cpu 1.15.1`、`scikit-learn 1.9.1` 均有 cp313 wheel）
- 16GB RAM 即可（CPU only，无需 GPU）
- 网络：仅需 PyPI（`requirements.lock.txt` 全部镜像可达）；**可选**后端需访问 huggingface.co / LLM API

## 2. 一键部署（零手工干预）

```bash
git clone <repo> && cd zhi-rag
bash scripts/setup.sh      # 建 venv + 装锁版核心依赖
bash scripts/run_demo.sh   # 建库 → 问答 → 评测 → 测试 → 性能基线
```

`run_demo.sh` 默认使用 **MockGenerator + LSI 嵌入**，无需任何模型下载或密钥。

## 3. 配置（`zhi/config.py`）

| 字段 | 默认 | 说明 |
|---|---|---|
| `embedder` | `lsi` | `lsi` 离线默认 / `dense` 可选 sentence-transformers |
| `retriever_mode` | `hybrid` | `bm25` / `dense` / `hybrid` |
| `generator` | `mock` | `mock` / `ollama` / `openai` |
| `top_k` | `5` | 检索条数 |
| `chunk_size` / `chunk_overlap` | `500` / `80` | 切分参数 |
| `persist_dir` | `./.zhi_index` | 索引落盘目录 |
| `ollama_model` / `ollama_base_url` | `qwen2.5:0.5b` / `http://localhost:11434` | Ollama 配置 |
| `openai_model` / `openai_api_key` | `gpt-4o-mini` / None | OpenAI 配置 |

配置可经 `ZhiConfig.load("config.yaml")` 与 YAML 互转。

## 4. 作为服务运行

```bash
python -m zhi.api.cli serve --host 127.0.0.1 --port 8000
```

- `GET  /health` → 状态 / 索引大小 / 后端
- `POST /ingest` → `{"paths":[...], "persist_dir": "..."}`
- `POST /query`  → `{"question":"...", "top_k":5, "mode":"hybrid"}`

```bash
curl -X POST http://127.0.0.1:8000/query -H 'Content-Type: application/json' \
  -d '{"question":"什么是 RAG？","top_k":3}'
```

## 5. 升级到真实后端（可选）

1. **真实稠密/CLIP 向量**：`pip install -r requirements-optional.txt`，并将 `embedder` 设为 `dense`（自动拉取 sentence-transformers 模型；多模态图像改用 CLIP 视觉向量）。
2. **本地 LLM**：安装 Ollama 并 `ollama pull qwen2.5:0.5b`，将 `generator` 设为 `ollama`。
3. **云端 LLM**：将 `generator` 设为 `openai` 并提供 `openai_api_key`。

任一后端缺失时，系统自动回退（生成回退 Mock、稠密回退 LSI），保证 demo 不中断。

## 6. Docker（可选）

以 Python 3.13-slim 为基础镜像，复制仓库后 `pip install -r requirements.lock.txt` 并暴露 8000 端口即可；离线镜像不包含任何模型权重。

## 7. 复现性保证

- `requirements.lock.txt` 精确钉死核心依赖版本。
- 全部使用 venv 隔离，不污染系统环境。
- 索引与嵌入器状态落盘，重启无需重新嵌入。
