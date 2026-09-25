from zhi.splitter.splitter import split_text


def test_split_produces_multiple_chunks():
    text = "。".join([f"这是第{i}句话用来测试中文切分功能是否正常工作并且足够长" for i in range(30)])
    chunks = split_text(text, chunk_size=60, chunk_overlap=10)
    assert len(chunks) >= 2
    assert all(c for c in chunks)


def test_split_respects_size_roughly():
    text = "。".join([f"句子{i}包含一些中文文本用于切分测试" for i in range(40)])
    chunks = split_text(text, chunk_size=80, chunk_overlap=20)
    assert all(len(c) <= 160 for c in chunks)


def test_split_empty():
    assert split_text("") == []
