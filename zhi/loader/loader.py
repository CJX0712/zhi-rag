"""Document loading: PDF / TXT / MD -> plain text.

Reuses pypdf for PDF parsing. Text/Markdown are read directly.
Images are handled by zhi.image (see zhi/image/image.py).
"""
import os

from ..errors import ZDError, ZDCode

TEXT_EXTS = {".txt", ".md", ".markdown", ".py", ".json", ".csv", ".log", ".yaml", ".yml"}


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_text(path: str) -> str:
    return _read(path)


def load_markdown(path: str) -> str:
    return _read(path)


def load_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except Exception as e:  # pragma: no cover
        raise ZDError(ZDCode.E_LOAD, "pypdf unavailable", str(e))
    try:
        reader = PdfReader(path)
        parts = [ (p.extract_text() or "") for p in reader.pages ]
        return "\n".join(parts)
    except Exception as e:
        raise ZDError(ZDCode.E_LOAD, f"pdf load failed: {path}", str(e))


def load_file(path: str) -> str:
    """Load a single file into text. Dispatches by extension; falls back to text read."""
    if not os.path.exists(path):
        raise ZDError(ZDCode.E_LOAD, f"file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    # text-like files (incl. unknown) are read as UTF-8
    return _read(path)


def iter_files(paths) -> list:
    """Expand directories to file lists. Returns a flat list of file paths."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, fs in os.walk(p):
                for fn in sorted(fs):
                    files.append(os.path.join(root, fn))
        else:
            files.append(p)
    return files
