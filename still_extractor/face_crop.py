"""Shared face-crop extraction helpers."""

from pathlib import Path

from PIL import Image


def extract_face_crop(
    image_path: Path,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    padding: int,
) -> Image.Image:
    """Crop a face region from an image on disk with padding clamped to image bounds."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    cx1 = max(0, int(x1) - padding)
    cy1 = max(0, int(y1) - padding)
    cx2 = min(w, int(x2) + padding)
    cy2 = min(h, int(y2) + padding)
    if cx2 <= cx1 or cy2 <= cy1:
        return img
    return img.crop((cx1, cy1, cx2, cy2))
