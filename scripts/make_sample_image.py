"""Generate a minimal valid PNG sample image using only the stdlib (no PIL)."""
import os
import struct
import zlib

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "images",
                   "sample_neural_network_diagram.png")


def _chunk(typ, data):
    return (struct.pack(">I", len(data)) + typ + data +
            struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))


def make_png(path, w=128, h=96, color=(38, 90, 200)):
    raw = bytearray()
    for _ in range(h):
        raw.append(0)  # filter type 0
        for _ in range(w):
            raw += bytes(color)
    comp = zlib.compress(bytes(raw))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(_chunk(b"IDAT", comp))
        f.write(_chunk(b"IEND", b""))


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    make_png(OUT)
    print("wrote", os.path.abspath(OUT))
