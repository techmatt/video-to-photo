"""Shared face-crop extraction helpers."""

import logging
import math
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


# Thresholds for flagging anomalous 5-point keypoint geometry. Mirror the
# constants used by `diagnose_keypoints.py` (RATIO_LOW/HIGH, KPS_SPAN_FRAC_MIN);
# keep them in sync manually if either side ever needs retuning.
KPS_RATIO_MIN: float = 0.25
KPS_RATIO_MAX: float = 0.75
KPS_SPAN_MIN_FRAC: float = 0.25


def is_keypoint_anomalous(kps, bbox) -> tuple[bool, list[str]]:
    """Return (anomalous, reasons) for a 5-point landmark set.

    `kps`: array-like of shape (5, 2), pixel coordinates ordered
        [left_eye, right_eye, nose, left_mouth, right_mouth].
    `bbox`: (x1, y1, x2, y2) in the same coord space as `kps`.

    Reasons are strings drawn from {"vertical_order", "ratio", "span"}.
    Returns (False, []) if kps is None or malformed; never raises.
    """
    if kps is None:
        return False, []
    try:
        pts = [(float(p[0]), float(p[1])) for p in kps[:5]]
    except (TypeError, ValueError, IndexError):
        return False, []
    if len(pts) < 5:
        return False, []
    try:
        x1, y1, x2, y2 = bbox
        bbox_h = float(y2) - float(y1)
    except (TypeError, ValueError):
        bbox_h = 0.0

    le, re_, no, lm, rm = pts
    eye_mid = ((le[0] + re_[0]) * 0.5, (le[1] + re_[1]) * 0.5)
    mouth_mid = ((lm[0] + rm[0]) * 0.5, (lm[1] + rm[1]) * 0.5)

    eye_to_nose = math.hypot(eye_mid[0] - no[0], eye_mid[1] - no[1])
    eye_to_mouth = math.hypot(eye_mid[0] - mouth_mid[0], eye_mid[1] - mouth_mid[1])
    ratio = eye_to_nose / eye_to_mouth if eye_to_mouth > 0 else float("nan")

    ys = [p[1] for p in pts]
    span_frac = ((max(ys) - min(ys)) / bbox_h) if bbox_h > 0 else float("nan")

    vertical_ok = (eye_mid[1] < no[1] < mouth_mid[1])

    reasons: list[str] = []
    if not vertical_ok:
        reasons.append("vertical_order")
    if math.isfinite(ratio) and (ratio < KPS_RATIO_MIN or ratio > KPS_RATIO_MAX):
        reasons.append("ratio")
    if math.isfinite(span_frac) and span_frac < KPS_SPAN_MIN_FRAC:
        reasons.append("span")

    return bool(reasons), reasons


def extract_face_crop_from_image(
    img: Image.Image,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    padding: int,
    kps: list | None = None,
    source_hint: str = "",
) -> Image.Image:
    """Crop a face region from an already-decoded PIL image with padding and roll correction.

    If `kps` is provided (InsightFace 5-point landmarks: left eye, right eye, nose,
    left mouth corner, right mouth corner), correct the roll angle so the eyes are
    horizontal in the returned crop. Sub-2 degree corrections are skipped as noise.

    When the keypoint geometry is anomalous (see `is_keypoint_anomalous`), the
    angle estimate is untrustworthy and rotating based on it actively degrades
    the crop, so the roll correction is skipped and the unrotated crop is returned.
    """
    w, h = img.size

    anomalous = False
    if kps is not None:
        anomalous, reasons = is_keypoint_anomalous(kps, (x1, y1, x2, y2))
        if anomalous:
            logger.debug(
                "Skipping roll correction (anomalous kps: %s): %s",
                reasons, source_hint,
            )

    angle = 0.0
    if kps is not None and not anomalous:
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
