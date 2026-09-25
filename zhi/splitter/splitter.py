"""Text splitting via langchain-text-splitters RecursiveCharacterTextSplitter.

Chinese-friendly separators are provided so that CJK text is split on
sentence/phrase boundaries rather than arbitrary characters.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ZH_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]


def make_splitter(chunk_size: int = 500, chunk_overlap: int = 80) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_ZH_SEPARATORS,
        keep_separator=True,
    )


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> list:
    """Split text into a list of non-empty chunks."""
    sp = make_splitter(chunk_size, chunk_overlap)
    return [s.strip() for s in sp.split_text(text) if s and s.strip()]
