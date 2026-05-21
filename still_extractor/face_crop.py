"""Shared face-crop extraction helpers."""

import math
from pathlib import Path

from PIL import Image


def extract_face_crop_from_image(
    img: Image.Image,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    padding: int,
    kps: list | None = None,
) -> Image.Image:
    """Crop a face region from an already-decoded PIL image with padding and roll correction.

    If `kps` is provided (InsightFace 5-point landmarks: left eye, right eye, nose,
    left mouth corner, right mouth corner), correct the roll angle so the eyes are
    horizontal in the returned crop. Sub-2 degree corrections are skipped as noise.
    """
    w, h = img.size

    angle = 0.0
    if kps is not None:
        lx, ly = kps[0]
        rx, ry = kps[1]
        angle = math.degrees(math.atan2(ry - ly, rx - lx))

    if abs(angle) > 2.0:
        target_w = int(x2) - int(x1) + 2 * padding
        target_h = int(y2) - int(y1) + 2 * padding
        expand = padding * 3
        ex1 = max(0, int(x1) - expand)
        ey1 = max(0, int(y1) - expand)
        ex2 = min(w, int(x2) + expand)
        ey2 = min(h, int(y2) + expand)
        if ex2 <= ex1 or ey2 <= ey1:
            return img
        expanded = img.crop((ex1, ey1, ex2, ey2))
        rotated = expanded.rotate(
            angle, expand=False, resample=Image.BICUBIC, fillcolor=(0, 0, 0),
        )
        rw, rh = rotated.size
        left = max(0, (rw - target_w) // 2)
        top = max(0, (rh - target_h) // 2)
        right = min(rw, left + target_w)
        bottom = min(rh, top + target_h)
        return rotated.crop((left, top, right, bottom))

    cx1 = max(0, int(x1) - padding)
    cy1 = max(0, int(y1) - padding)
    cx2 = min(w, int(x2) + padding)
    cy2 = min(h, int(y2) + padding)
    if cx2 <= cx1 or cy2 <= cy1:
        return img
    return img.crop((cx1, cy1, cx2, cy2))


def extract_face_crop(
    image_path: Path,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    padding: int,
    kps: list | None = None,
) -> Image.Image:
    """Crop a face region from an image on disk. Thin wrapper around the in-memory variant."""
    img = Image.open(image_path).convert("RGB")
    return extract_face_crop_from_image(img, x1, y1, x2, y2, padding, kps)
