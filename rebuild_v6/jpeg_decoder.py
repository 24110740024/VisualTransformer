import cv2
import numpy as np
import io
from PIL import Image

def decode_jpeg_tile(compressed, expected_size, tables=None):
    """Handle standalone JPEG tiles and TIFF JPEG tables, without imagecodecs."""
    if tables:
        if not (tables.startswith(b"\xff\xd8") and tables.endswith(b"\xff\xd9")
                and compressed.startswith(b"\xff\xd8")):
            raise ValueError("Unsupported TIFF JPEGTables framing")
        compressed = tables[:-2] + compressed[2:]
    with Image.open(io.BytesIO(compressed)) as tile:
        if tile.size != expected_size:
            raise ValueError(f"JPEG tile dimensions {tile.size} != {expected_size}")
        tile.load()
        return tile.convert("RGB")
