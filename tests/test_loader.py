import os

from zhi.errors import ZDError
from zhi.loader.loader import iter_files, load_file, load_markdown, load_pdf, load_text


def test_load_text(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("你好世界 hello", encoding="utf-8")
    assert "你好世界" in load_text(str(p))


def test_load_markdown(tmp_path):
    p = tmp_path / "b.md"
    p.write_text("# 标题\n正文", encoding="utf-8")
    assert "标题" in load_markdown(str(p))


def test_iter_files(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "x.txt").write_text("a")
    (d / "y.txt").write_text("b")
    (tmp_path / "z.txt").write_text("c")
    files = iter_files([str(d), str(tmp_path / "z.txt")])
    assert len(files) == 3


def test_load_file_unknown_ext_reads_text(tmp_path):
    p = tmp_path / "c.log"
    p.write_text("log line", encoding="utf-8")
    assert "log line" == load_file(str(p)).strip()


def test_pdf_missing_raises(tmp_path):
    try:
        load_pdf(str(tmp_path / "nope.pdf"))
        assert False, "expected ZDError"
    except ZDError:
        pass


def test_pdf_blank_loads(tmp_path):
    from pypdf import PdfWriter

    p = tmp_path / "blank.pdf"
    w = PdfWriter()
    w.add_blank_page(width=100, height=100)
    with open(p, "wb") as f:
        w.write(f)
    out = load_pdf(str(p))
    assert isinstance(out, str)
