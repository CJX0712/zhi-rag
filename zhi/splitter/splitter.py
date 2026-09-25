"""Text splitting — pure-Python RecursiveCharacterTextSplitter by default, with an
optional langchain-text-splitters upgrade when it is installed.

Chinese-friendly separators split CJK text on sentence/phrase boundaries rather
than arbitrary characters. The built-in implementation has no third-party
dependency, keeping the offline / deployment install lean.
"""
_ZH_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]


def _recursive_split(text, separators, chunk_size):
    sep = separators[0]
    if sep == "":
        # character-level fallback
        if len(text) <= chunk_size:
            return [text] if text else []
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    if sep in text:
        parts = text.split(sep)
        pieces = []
        for i, p in enumerate(parts):
            piece = p + sep if i < len(parts) - 1 else p
            if not piece:
                continue
            if len(piece) <= chunk_size:
                pieces.append(piece)
            else:
                pieces.extend(_recursive_split(piece, separators[1:], chunk_size))
        return pieces
    # separator not present; try the next one
    return _recursive_split(text, separators[1:], chunk_size)


def _merge_chars(text, chunk_size, chunk_overlap):
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    return chunks


def _merge(pieces, chunk_size, chunk_overlap):
    chunks = []
    cur = ""
    for p in pieces:
        if not p:
            continue
        if len(cur) + len(p) <= chunk_size:
            cur += p
        else:
            if cur:
                chunks.append(cur)
            if len(p) > chunk_size:
                chunks.extend(_merge_chars(p, chunk_size, chunk_overlap))
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return chunks


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> list:
    """Split text into a list of non-empty, stripped chunks."""
    if not text:
        return []
    pieces = _recursive_split(text, _ZH_SEPARATORS, chunk_size)
    merged = _merge(pieces, chunk_size, chunk_overlap)
    return [s.strip() for s in merged if s and s.strip()]


def make_splitter(chunk_size: int = 500, chunk_overlap: int = 80):
    """Return a splitter exposing `.split_text(text)` — langchain's if available,
    otherwise the built-in one wrapped to the same interface."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=_ZH_SEPARATORS,
            keep_separator=True,
        )
    except Exception:
        class _BuiltinSplitter:
            def __init__(self, cs, co):
                self.cs, self.co = cs, co

            def split_text(self, t):
                return split_text(t, self.cs, self.co)

        return _BuiltinSplitter(chunk_size, chunk_overlap)
