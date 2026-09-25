"""Multimodal ingestion: images become Chunks with an offline caption.

Design (see docs/architecture.md):
  * DEFAULT (offline, zero model): an image is represented by a deterministic
    caption built from its filename / format / size / dimensions. The caption is
    embedded like normal text, so images are retrievable by textual queries and
    can be cited in answers. No vision model required.
  * OPTIONAL (CLIP): when a CLIP-backed DenseEmbedder is supplied, the image can
    be embedded directly into the shared text-image space (requires Pillow +
    a CLIP sentence-transformers model). This is the "true" multimodal upgrade.
"""
import os
import struct

from ..types import Chunk

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".svg"}


def _png_dims(path: str):
    try:
        with open(path, "rb") as f:
            if f.read(8) != b"\x89PNG\r\n\x1a\n":
                return None
            f.read(4)  # length
            f.read(4)  # "IHDR"
            w = struct.unpack(">I", f.read(4))[0]
            h = struct.unpack(">I", f.read(4))[0]
            return (w, h)
    except Exception:
        return None


def caption_offline(path: str) -> str:
    name = os.path.basename(path)
    ext = os.path.splitext(name)[1].lower().lstrip(".")
    size = os.path.getsize(path)
    dims = _png_dims(path)
    dim_s = f"尺寸 {dims[0]}x{dims[1]}" if dims else "尺寸未知"
    return (
        f"[图像] 文件名：{name}；格式：{ext}；{dim_s}；大小 {size} 字节。"
        f"该图像内容需视觉模型识别，离线模式以文件名“{name}”作为可检索文本描述。"
    )


class ImageIngestor:
    def __init__(self, clip_embedder=None):
        self.clip = clip_embedder  # optional DenseEmbedder with encode_image

    def ingest(self, path: str) -> Chunk:
        cap = caption_offline(path)
        cid = "img:" + os.path.basename(path)
        return Chunk(
            id=cid,
            doc_id=cid,
            text=cap,
            source=path,
            modality="image",
            meta={"image_path": path, "ext": os.path.splitext(path)[1].lower()},
        )

    def embed(self, chunk: Chunk):
        """Return a CLIP image vector if available, else None (caller falls back)."""
        if self.clip is not None:
            try:
                return self.clip.encode_image(chunk.meta["image_path"])
            except Exception:
                return None
        return None
