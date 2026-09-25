# 使用（ZhiDa）

作者：**晨星** · 版本 `1.0.0`

## 1. Python API

```python
from zhi.config import ZhiConfig
from zhi.pipeline.pipeline import ZhiPipeline

cfg = ZhiConfig(persist_dir="./.zhi_index", embedder="lsi",
                retriever_mode="hybrid", generator="mock")
p = ZhiPipeline(cfg)

# 建库（文本 + 图像目录均可，自动识别扩展名）
stats = p.ingest(["data"])          # IndexStats
print(stats)

# 查询
ans = p.query("什么是检索增强生成 RAG？", top_k=3)
print(ans.answer, ans.contexts, ans.latency_ms)

# 重载已建索引（无需重新嵌入）
p2 = ZhiPipeline(cfg); p2.load()
print(p2.query("Transformer 的自注意力机制？").answer)
```

## 2. 命令行（`zhi`）

```bash
# 建索引
python -m zhi.api.cli ingest data --persist-dir .zhi_index
# 提问
python -m zhi.api.cli ask "什么是 RAG？" --top-k 3
# 仅用 BM25 检索
python -m zhi.api.cli ask "RAG 的模块组成" --top-k 3 --mode bm25
# 启动 REST 服务
python -m zhi.api.cli serve --port 8000
# 评测
python -m zhi.api.cli eval --dataset examples/qa_pairs.json --k 5
```

## 3. REST API

```bash
curl -X POST http://127.0.0.1:8000/ingest -H 'Content-Type: application/json' \
  -d '{"paths":["data"], "persist_dir":".zhi_index"}'

curl -X POST http://127.0.0.1:8000/query -H 'Content-Type: application/json' \
  -d '{"question":"多模态如何融合文本与图像？","top_k":3}'
```

返回：`{answer, mode, backend, latency_ms, contexts[{text,source,score,modality}]}`。

## 4. 切换生成后端

```python
# Ollama（需本机 Ollama 并已 pull 模型）
ZhiConfig(generator="ollama", ollama_model="qwen2.5:0.5b")

# OpenAI
ZhiConfig(generator="openai", openai_api_key="sk-...", openai_model="gpt-4o-mini")

# 离线默认
ZhiConfig(generator="mock")
```

后端不可用时自动回退 Mock，不会中断 demo。

## 5. 启用真实稠密 / 多模态（CLIP）

```bash
pip install -r requirements-optional.txt
# 将 embedder 设为 dense；图像将以 CLIP 视觉向量进入向量库
ZhiConfig(embedder="dense", dense_model="sentence-transformers/all-MiniLM-L6-v2")
```

## 6. 评测

- **内置（零依赖）**：`eval` 命令 / `zhi.eval.run_builtin` 给出 source 级 recall@k 与延迟统计。
- **SOTA（可选）**：`zhi.eval.run_ragas` 调用 `ragas` 计算 faithfulness / answer_relevancy / context_precision / context_recall（需 ragas + 真实 LLM 后端）。

## 7. 性能基线

```bash
python scripts/baseline.py
```

输出离线 LSI 默认的建库吞吐、embed/检索/端到端延迟、recall@k，并尝试可选 MiniLM 稠密对标（网络可达时）。
