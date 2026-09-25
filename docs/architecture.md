# 架构设计（ZhiDa）

作者：**晨星** · 版本 `1.0.0`

## 1. 设计原则

- **单一职责**：每个模块只做一件事，可独立验证（单测 + 最小可运行示例）。
- **可组合**：模块通过明确接口拼装成端到端链路。
- **复用优先**：向量库 / 切分 / 嵌入 / 评测全部复用领先开源，仅**混合融合层 RRF** 自研（见 §5）。
- **离线优先**：默认零模型下载即跑通，真实后端可插拔、缺失自动跳过。

## 2. 模块职责

| 模块 | 目录 | 职责 | 复用 / 自研 |
|---|---|---|---|
| Loader | `zhi/loader` | PDF/TXT/MD → 文本 + 元信息 | 复用 `pypdf` |
| Splitter | `zhi/splitter` | 递归切分（中文分隔符） | 复用 `langchain-text-splitters` |
| Embedder | `zhi/embedder` | 文本 → 向量（LSI 默认 / 稠密可选） | 复用 `scikit-learn` / `sentence-transformers` |
| VectorStore | `zhi/vectorstore` | 向量索引 + 持久化 + 近邻检索 | 复用 `faiss-cpu`（numpy 回退） |
| Retriever | `zhi/retriever` | BM25 + 稠密 + **RRF** 融合 | BM25 复用 `rank_bm25`；**RRF 自研** |
| Image | `zhi/image` | 图像入库 + 离线 caption + 可选 CLIP | caption 自研；CLIP 复用 |
| Generator | `zhi/generator` | 可插拔 LLM | Mock 自研；Ollama/OpenAI 复用官方 client |
| Pipeline | `zhi/pipeline` | 编排 ingest → retrieve → generate | 自研（glue） |
| API | `zhi/api` | FastAPI REST + `zhi` CLI | 复用 `fastapi`/`uvicorn`/`typer` |
| Eval | `zhi/eval` | recall@k + 延迟 + 可选 ragas | 内置自研；ragas 可选 |

## 3. 数据流

**建库（离线，一次性）**
```
源文档(PDF/TXT/MD/图像)
  → Loader（图像走 ImageIngestor 生成 caption）
  → Splitter（递归切分）
  → Embedder（LSI 默认 / 稠密可选）→ 向量
  → VectorStore（FAISS/np 索引）+ BM25 词表
  → 落盘 manifest + 索引 + 嵌入器状态
```

**查询（在线）**
```
Query
  → API / CLI
  → Pipeline.query
  → Retriever（BM25 + 稠密 → RRF 融合出 top-k 上下文）
  → Generator（基于上下文合成答案，可引用图像）
  → Answer{answer, contexts[], latency_ms, mode, backend}
```

## 4. 错误码与不变性

错误模型见 `zhi/errors.py`：统一 `ZDError(code, msg, detail)`，错误码分段：
`0xxx OK / 1xxx 输入 / 2xxx 建库索引 / 3xxx 检索生成 / 4xxx 配置`。REST 端点将其转为 `{code,msg,detail}`。

关键不变量（均有单测覆盖）：
- **RRF 交换不变性**：融合分数仅取决于各排序中的 rank 位置，与文档内容、排序供给顺序无关（交换律）。
- **向量归一化**：所有向量 L2 归一化，内积等价余弦相似度。
- **Embedder 确定性**：同一查询两次 `embed_query` 结果逐元素相等。
- **VectorStore 自洽**：自身向量检索返回自身为 top-1。

## 5. 自研说明：RRF 融合层

**为什么自研**：没有单一轻量库干净地提供「BM25 + 稠密 + RRF」的可插拔组合，且需要跨异构检索器（词法得分与向量相似度量纲不同）的可校准融合。RRF（Cormack et al., 2009）公式简单、无需分数标定：

```
score(d) = Σ_{r∈rankings} 1 / (k + rank_r(d))
```

代码见 `zhi/retriever/rrf.py`，含交换不变性与排序正确性单测。

## 6. 多模态设计

- **默认（离线、零依赖）**：图像由 `caption_offline()` 生成确定性文本描述（文件名/格式/尺寸/大小），以文本形式进入向量库，可被文本查询检索并在答案中引用。
- **可选（CLIP）**：传入 CLIP 后端 `DenseEmbedder` 时，`ImageIngestor.embed()` 直接产出图像向量，与文本查询在共享嵌入空间内做相似度匹配，实现真正跨模态检索（需 `sentence-transformers` + `Pillow`）。

## 7. 目录结构

```
zhi-rag/
├── zhi/                 # 源码（按模块划分）
├── tests/               # 单元 / 集成测试（34 passed）
├── examples/            # 最小示例 + QA 评测集
├── data/                # 样本语料（文本 + 图像）
├── scripts/             # setup / run_demo / run_tests / baseline
├── docs/                # architecture / deployment / usage
├── requirements.lock.txt        # 锁版核心依赖（离线 demo + 测试）
└── requirements-optional.txt     # 可选 SOTA / 后端依赖
```
