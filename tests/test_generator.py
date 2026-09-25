from zhi.config import ZhiConfig
from zhi.generator.generator import MockGenerator, get_generator
from zhi.types import Chunk, Context


def _ctxs():
    return [
        Context(
            chunk=Chunk(id="1", doc_id="a", text="检索增强生成 RAG 结合检索与生成", source="data/docs/rag.txt"),
            score=1.0,
        ),
        Context(
            chunk=Chunk(id="2", doc_id="b", text="Transformer 使用自注意力机制", source="data/docs/transformer.txt"),
            score=0.8,
        ),
    ]


def test_mock_contains_source():
    g = MockGenerator()
    ans = g.generate("什么是 RAG？", _ctxs())
    assert "资料来源" in ans and "rag.txt" in ans


def test_mock_deterministic():
    g = MockGenerator()
    a = g.generate("RAG 是什么", _ctxs())
    b = g.generate("RAG 是什么", _ctxs())
    assert a == b


def test_get_generator_mock():
    gen = get_generator(ZhiConfig(generator="mock"))
    assert gen.name == "mock"
