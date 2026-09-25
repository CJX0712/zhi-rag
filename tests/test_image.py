import os

from zhi.image.image import IMAGE_EXTS, ImageIngestor, caption_offline


def test_caption_contains_name(tmp_path):
    p = tmp_path / "sample_neural_network_diagram.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    cap = caption_offline(str(p))
    assert "sample_neural_network_diagram" in cap


def test_ingest_image(tmp_path):
    p = tmp_path / "pic.png"
    p.write_bytes(b"1234")
    c = ImageIngestor().ingest(str(p))
    assert c.modality == "image"
    assert c.meta["image_path"] == str(p)


def test_image_exts():
    assert ".png" in IMAGE_EXTS and ".jpg" in IMAGE_EXTS and ".svg" in IMAGE_EXTS
