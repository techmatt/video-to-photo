"""Per-file worker: decode → rotate → score → dedup → refine → write keepers.

The orchestrator (`pipeline.py`) iterates the manifest and calls `process_file`
once per row. Each call processes one source video or image entirely in memory;
only the final keeper JPEGs are written to disk. Errors are caught and logged
so a single bad file never aborts a run.
"""

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import av
import cv2
import imagehash
import numpy as np
import pandas as pd
import piexif
import pillow_heif
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps

from still_extractor.constants import (
    CLASSIFIER_BLEND_WEIGHT,
    FACE_CROP_PADDING,
    FACE_EDGE_IMMUNE_AREA_FRAC,
    FACE_EDGE_ZONE_FRAC,
    FACE_MIN_AREA_FRAC,
    FACE_QUALITY_INPUT_SIZE,
    FACE_QUALITY_LABELS,
    FACE_SHARPNESS_PADDING,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_FACE_SLOTS,
    UPRIGHTER_INPUT_SIZE,
    UPRIGHTER_LABELS,
)
from still_extractor.face_crop import extract_face_crop_from_image, is_keypoint_anomalous
from still_extractor.models import Models
from still_extractor.sampling import (
    _apply_rotation,
    decode_window,
    get_video_fps,
    get_video_rotation,
    sample_frames,
    sample_frames_windowed,
    sharpness_score,
)

pillow_heif.register_heif_opener()

logger = logging.getLogger(__name__)

FACE_QUALITY_TTA_PASSES = 3
AESTHETICS_BATCH_SIZE = 16

EXIF_DATE_TAG_PRIMARY = 36867    # DateTimeOriginal
EXIF_DATE_TAG_FALLBACK = 306     # DateTime
EXIF_DATE_FORMAT = "%Y:%m:%d %H:%M:%S"

_VIDEO_EXIF_SKIP_SUFFIXES = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".m4v",
})
_VIDEO_METADATA_SUFFIXES = frozenset({
    ".mp4", ".mov", ".m4v", ".avi", ".mkv",
})
_PATH_YEAR_MONTH_RE = re.compile(r"(20\d{2})[-_./]?(0[1-9]|1[0-2])(?!\d)")
_PATH_YYYYMMDD_RE = re.compile(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")
_PATH_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_PATH_MONTHNAME_YEAR_RE = re.compile(
    r"(?i)(?<![a-z])"
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)"
    r"(?![a-z])"
    r"(?:[\s_.-]+[0-3]?\d)?"
    r"[\s_.-]?"
    r"(20\d{2})(?!\d)",
)
_MONTHNAME_TO_NUM: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_exif_year_month(raw: bytes | str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("ascii", errors="ignore")
        except Exception:
            return None
    raw = raw.strip().rstrip("\x00")
    if not raw:
        return None
    try:
        dt = datetime.strptime(raw, EXIF_DATE_FORMAT)
    except ValueError:
        return None
    return dt.year, dt.month


def _exif_year_month(source_path: Path) -> tuple[tuple[int, int], str] | None:
    """Return ((year, month), 'exif_primary'|'exif_fallback') if EXIF date is available."""
    try:
        exif = piexif.load(str(source_path))
    except Exception:
        return None
    exif_ifd = exif.get("Exif", {}) or {}
    primary = _parse_exif_year_month(exif_ifd.get(EXIF_DATE_TAG_PRIMARY))
    if primary is not None:
        return primary, "exif_primary"
    zeroth = exif.get("0th", {}) or {}
    fallback = _parse_exif_year_month(zeroth.get(EXIF_DATE_TAG_FALLBACK))
    if fallback is not None:
        return fallback, "exif_fallback"
    return None


def _path_year_month(source_path: Path) -> tuple[int, int] | None:
    text = str(source_path)
    ymd = _PATH_YYYYMMDD_RE.search(text)
    if ymd is not None:
        return int(ymd.group(1)), int(ymd.group(2))
    ym = _PATH_YEAR_MONTH_RE.search(text)
    if ym is not None:
        return int(ym.group(1)), int(ym.group(2))
    mn = _PATH_MONTHNAME_YEAR_RE.search(text)
    if mn is not None:
        token = mn.group(1).lower()[:3]
        month = _MONTHNAME_TO_NUM.get(token)
        if month is not None:
            return int(mn.group(2)), month
    y = _PATH_YEAR_RE.search(text)
    if y is not None:
        return int(y.group(1)), 0
    return None


def _mtime_year_month(source_path: Path) -> tuple[int, int] | None:
    try:
        mtime = source_path.stat().st_mtime
        dt = datetime.fromtimestamp(mtime)
        return dt.year, dt.month
    except (OSError, ValueError, OverflowError):
        return None


def _parse_iso_year_month(raw: str) -> tuple[int, int] | None:
    s = raw.strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.year >= 2000:
            return dt.year, dt.month
    except ValueError:
        pass
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m is not None:
        y, mo = int(m.group(1)), int(m.group(2))
        if y >= 2000 and 1 <= mo <= 12:
            return y, mo
    return None


def _video_metadata_year_month(source_path: Path) -> tuple[int, int] | None:
    """Read capture date from video container metadata via PyAV.

    Prefers Apple QuickTime's timezone-aware `com.apple.quicktime.creationdate`
    over the generic `creation_time`. Returns None if metadata is missing,
    unparseable, or the container can't be opened.
    """
    try:
        with av.open(str(source_path)) as container:
            meta = dict(container.metadata or {})
    except Exception:
        return None
    for key in ("com.apple.quicktime.creationdate", "creation_time"):
        raw = meta.get(key)
        if not isinstance(raw, str) or not raw:
            continue
        parsed = _parse_iso_year_month(raw)
        if parsed is not None:
            return parsed
    return None


def extract_source_date(source_path: Path) -> tuple[int, int, str]:
    """Return (year, month, date_source) for the source file.

    Fallback chain: EXIF DateTimeOriginal -> EXIF DateTime -> video container
    metadata (creation_time / com.apple.quicktime.creationdate) -> path regex
    -> mtime -> unknown. Skips EXIF for known video extensions. Never raises.
    """
    try:
        suffix = source_path.suffix.lower()
        if suffix not in _VIDEO_EXIF_SKIP_SUFFIXES:
            try:
                exif_result = _exif_year_month(source_path)
            except Exception:
                exif_result = None
            if exif_result is not None:
                (year, month), source = exif_result
                return year, month, source

        if suffix in _VIDEO_METADATA_SUFFIXES:
            try:
                vm_result = _video_metadata_year_month(source_path)
            except Exception:
                vm_result = None
            if vm_result is not None:
                return vm_result[0], vm_result[1], "video_metadata"

        try:
            path_ym = _path_year_month(source_path)
        except Exception:
            path_ym = None
        if path_ym is not None:
            return path_ym[0], path_ym[1], "path_regex"

        try:
            mtime_ym = _mtime_year_month(source_path)
        except Exception:
            mtime_ym = None
        if mtime_ym is not None:
            return mtime_ym[0], mtime_ym[1], "mtime"
    except Exception:
        pass
    return 0, 0, "unknown"

# Per-stage timing keys. `total` is the outermost wall time and not part of the
# per-stage breakdown. Aesthetics is included even though the prompt did not
# name it, so the summed-stage % breakdown is comprehensive.
STAGE_KEYS: tuple[str, ...] = (
    "frame_sampling",
    "temporal_dedup",
    "uprighter",
    "sharpness",
    "face_detect",
    "aesthetics",
    "classifier",
    "dhash_dedup",
    "refinement",
    "jpeg_write",
)


class StageTimer:
    """Tiny accumulator for per-stage wall-clock seconds.

    Use as a context manager: `with timer("face_detect"): ...`. Multiple
    entries with the same key accumulate.
    """

    def __init__(self) -> None:
        self.times: dict[str, float] = {}

    def __call__(self, key: str) -> "_StageCtx":
        return _StageCtx(self, key)

    def add(self, key: str, dt: float) -> None:
        self.times[key] = self.times.get(key, 0.0) + dt


class _StageCtx:
    __slots__ = ("timer", "key", "_t0")

    def __init__(self, timer: StageTimer, key: str) -> None:
        self.timer = timer
        self.key = key
        self._t0 = 0.0

    def __enter__(self) -> "_StageCtx":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.timer.add(self.key, time.perf_counter() - self._t0)


@dataclass
class FileResult:
    keepers: list[dict]
    stage_times_s: dict[str, float] = field(default_factory=dict)
    rejection_stats: dict[str, int] = field(default_factory=dict)


@dataclass
class WorkerConfig:
    output_dir: Path
    fps: float = 1.0
    sharpness_threshold: float = 75.0
    min_face_px: int = 80
    temporal_window_s: float = 2.0
    face_dedup_threshold: int = 8
    frame_dedup_threshold: int = 8
    quality_threshold: float = 0.0
    max_per_file: int = 5
    uprighter_confidence: float = 0.95
    refine_window_s: float = 0.5


# --- Uprighter helpers -------------------------------------------------------

def _letterbox_pil(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def _squish_pil(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.BILINEAR)


def _center_crop_pil(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    cropped = img.crop((left, top, left + s, top + s))
    return cropped.resize((size, size), Image.BILINEAR)


_UPRIGHTER_STRATEGIES = (_letterbox_pil, _squish_pil, _center_crop_pil)
_UPRIGHTER_NORMALIZE = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _uprighter_predict(
    model: torch.nn.Module, pil_img: Image.Image, device: torch.device,
    use_tta: bool,
) -> tuple[int, float]:
    """Return (degrees_cw, confidence). degrees_cw in {0,90,180,270}."""
    with torch.inference_mode():
        if use_tta:
            tensors = torch.stack([
                _UPRIGHTER_NORMALIZE(fn(pil_img, UPRIGHTER_INPUT_SIZE))
                for fn in _UPRIGHTER_STRATEGIES
            ]).to(device)
            logits = model(tensors).mean(dim=0, keepdim=True)
        else:
            tensor = _UPRIGHTER_NORMALIZE(
                _squish_pil(pil_img, UPRIGHTER_INPUT_SIZE),
            ).unsqueeze(0).to(device)
            logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return int(UPRIGHTER_LABELS[pred_idx]), float(probs[pred_idx])


_CV2_ROTATE_CW = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _rotate_bgr_cw(bgr: np.ndarray, degrees: int) -> np.ndarray:
    flag = _CV2_ROTATE_CW.get(degrees)
    return bgr if flag is None else cv2.rotate(bgr, flag)


def _rotate_pil_cw(pil_img: Image.Image, degrees: int) -> Image.Image:
    if degrees == 0:
        return pil_img
    if degrees == 90:
        return pil_img.transpose(Image.ROTATE_270)
    if degrees == 180:
        return pil_img.transpose(Image.ROTATE_180)
    if degrees == 270:
        return pil_img.transpose(Image.ROTATE_90)
    return pil_img


def _maybe_uprighten_bgr(
    bgr: np.ndarray, models: Models, threshold: float, use_tta: bool,
) -> tuple[np.ndarray, int, float]:
    """Apply uprighter rotation if confident. Returns (frame, applied_deg, confidence)."""
    if models.uprighter_model is None:
        return bgr, 0, 0.0
    pil_img = _bgr_to_pil(bgr)
    pred_deg, conf = _uprighter_predict(
        models.uprighter_model, pil_img, models.device, use_tta=use_tta,
    )
    if conf >= threshold and pred_deg != 0:
        return _rotate_bgr_cw(bgr, pred_deg), pred_deg, conf
    return bgr, 0, conf


# --- Scoring helpers ---------------------------------------------------------

def _face_sharpness_bgr(
    bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float,
    padding: int = FACE_SHARPNESS_PADDING,
) -> float:
    h, w = bgr.shape[:2]
    cx1 = max(0, int(x1) - padding)
    cy1 = max(0, int(y1) - padding)
    cx2 = min(w, int(x2) + padding)
    cy2 = min(h, int(y2) + padding)
    if cx2 <= cx1 or cy2 <= cy1:
        return 0.0
    crop = bgr[cy1:cy2, cx1:cx2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _score_aesthetics_batch(
    pil_images: list[Image.Image], models: Models,
) -> np.ndarray:
    """Return raw aesthetic scores [N], roughly in [1, 10]."""
    if not pil_images:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(len(pil_images), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(pil_images), AESTHETICS_BATCH_SIZE):
            batch = pil_images[start:start + AESTHETICS_BATCH_SIZE]
            inputs = models.aesthetic_preprocessor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(
                device=models.device, dtype=models.aesthetic_dtype,
            )
            logits = models.aesthetic_model(pixel_values=pixel_values).logits
            scores = logits.squeeze(-1).float().cpu().numpy()
            out[start:start + len(batch)] = scores
    return out


_CLASSIFIER_TRAIN_TF = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(
        degrees=15, interpolation=T.InterpolationMode.BICUBIC, fill=0,
    ),
    T.RandomResizedCrop(
        size=FACE_QUALITY_INPUT_SIZE,
        scale=(0.80, 1.00),
        ratio=(0.9, 1.1),
        interpolation=T.InterpolationMode.BICUBIC,
    ),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _score_classifier_batch(
    face_crops: list[Image.Image], models: Models, n_tta: int = FACE_QUALITY_TTA_PASSES,
) -> np.ndarray:
    """Return softmax probs [N, 4] over (none, bad, okay, good), TTA-averaged."""
    n = len(face_crops)
    if n == 0 or models.face_quality_model is None:
        return np.zeros((n, len(FACE_QUALITY_LABELS)), dtype=np.float32)
    accum = np.zeros((n, len(FACE_QUALITY_LABELS)), dtype=np.float32)
    with torch.inference_mode():
        for _ in range(max(n_tta, 1)):
            batch = torch.stack([_CLASSIFIER_TRAIN_TF(c) for c in face_crops]).to(
                models.device,
            )
            logits = models.face_quality_model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            accum += probs
    return accum / max(n_tta, 1)


# --- Dedup helpers (in-memory) ----------------------------------------------

def _dhash_pil(pil_img: Image.Image) -> imagehash.ImageHash:
    return imagehash.dhash(pil_img, hash_size=8)


def _dedup_indices(
    hashes: list, order: list[int], threshold: int,
) -> list[int]:
    """Greedy dHash dedup: walk `order`, drop any hash within `threshold` of a kept one.

    Hashes that are None (e.g. candidates with no passing face crop) are always
    kept and never compared against — they contribute no signal to face-dedup.
    """
    kept: list[imagehash.ImageHash] = []
    keep_idx: list[int] = []
    for i in order:
        h = hashes[i]
        if h is None:
            keep_idx.append(i)
            continue
        if any((h - kh) <= threshold for kh in kept):
            continue
        kept.append(h)
        keep_idx.append(i)
    return keep_idx


# --- Per-candidate dataclass -------------------------------------------------

@dataclass
class _Candidate:
    timestamp_s: float
    frame_index: int
    bgr: np.ndarray
    faces: list[Any]                       # post-rejection passing faces (largest first)
    rejected_faces: list[tuple[Any, str]]  # [(face, reason), ...]
    sharpness_center: float
    uprighter_pred_deg: int
    uprighter_confidence: float


def _detect_qualifying_faces(face_app, bgr: np.ndarray, min_face_px: int) -> list:
    """Return all qualifying faces, largest first. Empty list if none."""
    try:
        faces = face_app.get(bgr)
    except Exception as e:
        logger.warning("InsightFace failed: %s", e)
        return []
    if not faces:
        return []
    qualifying = [f for f in faces if float(f.bbox[2] - f.bbox[0]) >= min_face_px]
    qualifying.sort(
        key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        reverse=True,
    )
    return qualifying


# --- Face rejection heuristics ----------------------------------------------

def _check_face_rejection(
    x1: float, y1: float, x2: float, y2: float,
    frame_w: int, frame_h: int,
) -> str | None:
    """Return a rejection reason string, or None if the face passes."""
    face_area_frac = ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h)

    if face_area_frac < FACE_MIN_AREA_FRAC:
        return f"too_small({face_area_frac:.4f}<{FACE_MIN_AREA_FRAC})"

    if face_area_frac < FACE_EDGE_IMMUNE_AREA_FRAC:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        edge_x = frame_w * FACE_EDGE_ZONE_FRAC
        edge_y = frame_h * FACE_EDGE_ZONE_FRAC
        if cx < edge_x or cx > frame_w - edge_x or cy < edge_y or cy > frame_h - edge_y:
            return f"small_and_edge({face_area_frac:.4f})"

    return None


def _rejection_kind(reason: str) -> str:
    """`too_small(0.0021<0.004)` -> `too_small`. Used for stat keys and filenames."""
    idx = reason.find("(")
    return reason[:idx] if idx > 0 else reason


def _split_rejected_faces(
    faces: list, frame_w: int, frame_h: int,
) -> tuple[list, list[tuple[Any, str]]]:
    """Split faces into (passing, [(face, reason), ...])."""
    passing: list = []
    rejected: list[tuple[Any, str]] = []
    for face in faces:
        x1 = float(face.bbox[0])
        y1 = float(face.bbox[1])
        x2 = float(face.bbox[2])
        y2 = float(face.bbox[3])
        reason = _check_face_rejection(x1, y1, x2, y2, frame_w, frame_h)
        if reason is None:
            passing.append(face)
        else:
            rejected.append((face, reason))
    return passing, rejected


def _write_rejected_face_crop(
    rejected_dir: Path, bgr: np.ndarray, face, reason: str,
    stem: str, timestamp_s: float, face_idx: int,
) -> None:
    """Save a padded face crop for visual inspection. Quality 88."""
    rejected_dir.mkdir(parents=True, exist_ok=True)
    pil_img = _bgr_to_pil(bgr)
    try:
        crop = extract_face_crop_from_image(
            pil_img,
            face.bbox[0], face.bbox[1], face.bbox[2], face.bbox[3],
            FACE_CROP_PADDING,
            kps=[[float(x), float(y)] for x, y in face.kps],
        )
    except Exception as e:
        logger.warning("Failed to extract rejected face crop: %s", e)
        return
    kind = _rejection_kind(reason)
    out_path = rejected_dir / f"{kind}_{stem}_{timestamp_s:.3f}_{face_idx}.jpg"
    try:
        crop.save(out_path, "JPEG", quality=88)
    except Exception as e:
        logger.warning("Failed to write rejected crop %s: %s", out_path, e)


def _build_rejected_columns(rejected: list[tuple[Any, str]]) -> dict:
    """Build the rejection-related parquet columns for one keeper row."""
    reasons = [reason for _, reason in rejected]
    bboxes = []
    for face, reason in rejected:
        bboxes.append({
            "x1": int(face.bbox[0]),
            "y1": int(face.bbox[1]),
            "x2": int(face.bbox[2]),
            "y2": int(face.bbox[3]),
            "reason": reason,
        })
    return {
        "rejected_face_count": int(len(rejected)),
        "rejected_face_reasons": json.dumps(reasons),
        "rejected_faces_json": json.dumps(bboxes),
    }


def _empty_rejected_columns() -> dict:
    return {
        "rejected_face_count": 0,
        "rejected_face_reasons": "[]",
        "rejected_faces_json": "[]",
    }


def _update_rejection_stats(
    stats: dict[str, int], passing: list, rejected: list[tuple[Any, str]],
) -> None:
    """Mutate stats with counts from one frame's rejection split."""
    for _, reason in rejected:
        stats["total_faces_rejected"] = stats.get("total_faces_rejected", 0) + 1
        kind = _rejection_kind(reason)
        stats[kind] = stats.get(kind, 0) + 1
    if rejected and not passing:
        stats["frames_with_all_faces_rejected"] = (
            stats.get("frames_with_all_faces_rejected", 0) + 1
        )


# --- Image and video processing ---------------------------------------------

def _face_slot_columns(face, class_probs: np.ndarray | None, slot: int) -> dict:
    """Per-slot columns face_{slot}_*. `face`/`class_probs` may be None for empty slots."""
    prefix = f"face_{slot}_"
    if face is None:
        return {
            f"{prefix}x1": None,
            f"{prefix}y1": None,
            f"{prefix}x2": None,
            f"{prefix}y2": None,
            f"{prefix}det_score": None,
            f"{prefix}kps": None,
            f"{prefix}kps_anomalous": None,
            f"{prefix}embedding": None,
            f"{prefix}p_none": None,
            f"{prefix}p_bad": None,
            f"{prefix}p_okay": None,
            f"{prefix}p_good": None,
            f"{prefix}pred_label": None,
            f"{prefix}pred_confidence": None,
        }
    bbox = face.bbox
    if class_probs is None:
        p_none = p_bad = p_okay = p_good = float("nan")
        pred_label = ""
        pred_confidence = float("nan")
    else:
        p_none = float(class_probs[0])
        p_bad = float(class_probs[1])
        p_okay = float(class_probs[2])
        p_good = float(class_probs[3])
        idx = int(np.argmax(class_probs))
        pred_label = FACE_QUALITY_LABELS[idx]
        pred_confidence = float(class_probs[idx])
    kps_list = [[float(x), float(y)] for x, y in face.kps]
    anomalous, _ = is_keypoint_anomalous(
        kps_list, (bbox[0], bbox[1], bbox[2], bbox[3]),
    )
    return {
        f"{prefix}x1": int(bbox[0]),
        f"{prefix}y1": int(bbox[1]),
        f"{prefix}x2": int(bbox[2]),
        f"{prefix}y2": int(bbox[3]),
        f"{prefix}det_score": float(face.det_score),
        f"{prefix}kps": json.dumps(kps_list),
        f"{prefix}kps_anomalous": bool(anomalous),
        f"{prefix}embedding": json.dumps([float(v) for v in face.normed_embedding]),
        f"{prefix}p_none": p_none,
        f"{prefix}p_bad": p_bad,
        f"{prefix}p_okay": p_okay,
        f"{prefix}p_good": p_good,
        f"{prefix}pred_label": pred_label,
        f"{prefix}pred_confidence": pred_confidence,
    }


def _build_keeper_dict(
    *,
    cfg: WorkerConfig,
    video_path: Path,
    source_type: str,
    timestamp_s: float,
    refined_timestamp_s: float,
    frame_index: int,
    bgr: np.ndarray,
    faces_top: list,                       # length MAX_FACE_SLOTS, padded with None
    faces_class_probs: list,               # length MAX_FACE_SLOTS, padded with None
    face_count: int,
    sharpness_center: float,
    refined_sharpness: float,
    aesthetics_norm: float,
    composite: float,
    uprighter_pred_deg: int,
    uprighter_confidence: float,
    kept_path: Path,
    source_fps: float | None,
    file_size_bytes: int,
    source_year: int,
    source_month: int,
    date_source: str,
    rejected_columns: dict,
) -> dict:
    h, w = bgr.shape[:2]
    face_1 = faces_top[0]
    face_1_probs = faces_class_probs[0]

    face_1_cols = _face_slot_columns(face_1, face_1_probs, slot=1)
    face_2_cols = _face_slot_columns(faces_top[1], faces_class_probs[1], slot=2)
    face_3_cols = _face_slot_columns(faces_top[2], faces_class_probs[2], slot=3)

    # Legacy face_* columns mirror face_1_* exactly. When face_1 is None
    # (all detected faces were rejected by heuristics), columns are null.
    if face_1 is None:
        legacy_face_cols = {
            "face_x1": None,
            "face_y1": None,
            "face_x2": None,
            "face_y2": None,
            "face_w": None,
            "face_det_score": None,
            "kps": None,
            "embedding": None,
            "p_none": None,
            "p_bad": None,
            "p_okay": None,
            "p_good": None,
            "pred_label": None,
            "pred_confidence": None,
        }
        best_pair_score: float | None = None
    else:
        legacy_face_cols = {
            "face_x1": face_1_cols["face_1_x1"],
            "face_y1": face_1_cols["face_1_y1"],
            "face_x2": face_1_cols["face_1_x2"],
            "face_y2": face_1_cols["face_1_y2"],
            "face_w": int(face_1.bbox[2] - face_1.bbox[0]),
            "face_det_score": face_1_cols["face_1_det_score"],
            "kps": face_1_cols["face_1_kps"],
            "embedding": json.dumps([float(v) for v in face_1.normed_embedding]),
            "p_none": face_1_cols["face_1_p_none"],
            "p_bad": face_1_cols["face_1_p_bad"],
            "p_okay": face_1_cols["face_1_p_okay"],
            "p_good": face_1_cols["face_1_p_good"],
            "pred_label": face_1_cols["face_1_pred_label"],
            "pred_confidence": face_1_cols["face_1_pred_confidence"],
        }
        if (
            faces_top[1] is not None
            and faces_class_probs[0] is not None
            and faces_class_probs[1] is not None
        ):
            best_pair_score = float(
                (faces_class_probs[0][3] + faces_class_probs[1][3]) / 2.0,
            )
        else:
            best_pair_score = None

    out: dict = {
        "video_path": str(video_path.resolve()),
        "video_stem": video_path.stem,
        "source_type": source_type,
        "timestamp_s": float(timestamp_s),
        "refined_timestamp_s": float(refined_timestamp_s),
        "frame_index": int(frame_index),
        "frame_w": int(w),
        "frame_h": int(h),
        **legacy_face_cols,
        "sharpness_center": float(sharpness_center),
        "refined_sharpness": float(refined_sharpness),
        "sharpness_delta": float(refined_sharpness - sharpness_center),
        "aesthetics_norm": float(aesthetics_norm),
        "composite": float(composite),
        "uprighter_pred": int(uprighter_pred_deg),
        "uprighter_confidence": float(uprighter_confidence),
        "kept_path": str(kept_path.resolve()),
        # New schema additions:
        "face_count": int(face_count),
        **face_1_cols,
        **face_2_cols,
        **face_3_cols,
        "best_pair_score": best_pair_score,
        "source_fps": source_fps,
        "file_size_bytes": int(file_size_bytes),
        "source_year": int(source_year),
        "source_month": int(source_month),
        "date_source": str(date_source),
        **rejected_columns,
    }
    return out


def _write_keeper_jpeg(
    kept_dir: Path, video_stem: str, timestamp_s: float, composite: float,
    bgr: np.ndarray,
) -> Path:
    kept_dir.mkdir(parents=True, exist_ok=True)
    out_path = kept_dir / f"{composite:.4f}_{video_stem}_{timestamp_s:.3f}.jpg"
    # perf: cv2.imwrite default JPEG quality is 95; 88 is imperceptible for these
    # crops and meaningfully reduces file size and encode time.
    cv2.imwrite(str(out_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return out_path


def _rank_faces_by_p_good(
    faces: list, class_probs_list: list,
) -> tuple[list, list]:
    """Sort faces by p_good descending. Returns (sorted_faces, sorted_class_probs).

    `class_probs_list` is parallel to `faces`. Entries may be None if the
    classifier did not run; those rank last.
    """
    def key(i: int) -> float:
        cp = class_probs_list[i]
        return -float(cp[3]) if cp is not None else 0.0
    order = sorted(range(len(faces)), key=key)
    return [faces[i] for i in order], [class_probs_list[i] for i in order]


def _pad_to_slots(seq: list, n: int) -> list:
    out = list(seq[:n])
    while len(out) < n:
        out.append(None)
    return out


def _process_image(
    row: pd.Series, models: Models, cfg: WorkerConfig, timer: StageTimer,
    rejection_stats: dict[str, int],
) -> list[dict]:
    image_path = Path(row["file_path"])
    file_size_bytes = int(image_path.stat().st_size)
    source_year, source_month, date_source = extract_source_date(image_path)
    with timer("frame_sampling"):
        try:
            with Image.open(image_path) as raw:
                pil_img = ImageOps.exif_transpose(raw).convert("RGB")
        except Exception as e:
            logger.warning("Failed to open image %s: %s", image_path, e)
            return []

    # Uprighter (3-strategy TTA for images; cheap since N=1).
    uprighter_pred_deg, uprighter_confidence = 0, 0.0
    if models.uprighter_model is not None:
        with timer("uprighter"):
            pred_deg, conf = _uprighter_predict(
                models.uprighter_model, pil_img, models.device, use_tta=True,
            )
            if conf >= cfg.uprighter_confidence and pred_deg != 0:
                pil_img = _rotate_pil_cw(pil_img, pred_deg)
                uprighter_pred_deg = pred_deg
            uprighter_confidence = conf

    with timer("frame_sampling"):
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    with timer("sharpness"):
        sharp = sharpness_score(bgr)
    if sharp < cfg.sharpness_threshold:
        logger.debug("%s sharpness=%.2f DROP-sharpness", image_path.name, sharp)
        return []

    with timer("face_detect"):
        detected = _detect_qualifying_faces(models.face_app, bgr, cfg.min_face_px)
    if not detected:
        logger.debug("%s DROP-face", image_path.name)
        return []

    fh, fw = bgr.shape[:2]
    passing, rejected = _split_rejected_faces(detected, fw, fh)
    _update_rejection_stats(rejection_stats, passing, rejected)
    for fi, (face, reason) in enumerate(rejected):
        _write_rejected_face_crop(
            cfg.output_dir / "rejected", bgr, face, reason,
            image_path.stem, 0.0, fi,
        )
    rejected_columns = _build_rejected_columns(rejected)

    faces = passing
    face_count = len(faces)

    with timer("aesthetics"):
        aes_raw = _score_aesthetics_batch([pil_img], models)[0]
        aesthetics_norm = float(np.clip((aes_raw - 1.0) / 9.0, 0.0, 1.0))

    face_class_probs: list = [None] * len(faces)
    if faces and models.face_quality_model is not None:
        with timer("classifier"):
            crops = []
            for face in faces:
                crop = extract_face_crop_from_image(
                    pil_img,
                    face.bbox[0], face.bbox[1], face.bbox[2], face.bbox[3],
                    FACE_CROP_PADDING,
                    kps=[[float(x), float(y)] for x, y in face.kps],
                ).resize(
                    (FACE_QUALITY_INPUT_SIZE, FACE_QUALITY_INPUT_SIZE),
                    Image.BICUBIC,
                )
                crops.append(crop)
            all_probs = _score_classifier_batch(crops, models)
            face_class_probs = [all_probs[i] for i in range(len(faces))]

    faces_ranked, probs_ranked = _rank_faces_by_p_good(faces, face_class_probs)
    face_1_probs = probs_ranked[0] if probs_ranked else None
    if face_1_probs is not None:
        p_good = float(face_1_probs[3])
        composite = (
            CLASSIFIER_BLEND_WEIGHT * p_good
            + (1.0 - CLASSIFIER_BLEND_WEIGHT) * aesthetics_norm
        )
    else:
        composite = aesthetics_norm

    if composite < cfg.quality_threshold:
        return []

    if faces_ranked:
        face_1 = faces_ranked[0]
        with timer("refinement"):
            face_sharp = _face_sharpness_bgr(
                bgr, face_1.bbox[0], face_1.bbox[1], face_1.bbox[2], face_1.bbox[3],
            )
    else:
        face_sharp = sharp
    with timer("jpeg_write"):
        kept_dir = cfg.output_dir / "kept"
        kept_path = _write_keeper_jpeg(kept_dir, image_path.stem, 0.0, composite, bgr)

    return [_build_keeper_dict(
        cfg=cfg,
        video_path=image_path,
        source_type="image",
        timestamp_s=0.0,
        refined_timestamp_s=0.0,
        frame_index=0,
        bgr=bgr,
        faces_top=_pad_to_slots(faces_ranked, MAX_FACE_SLOTS),
        faces_class_probs=_pad_to_slots(probs_ranked, MAX_FACE_SLOTS),
        face_count=face_count,
        sharpness_center=sharp,
        refined_sharpness=face_sharp,
        aesthetics_norm=aesthetics_norm,
        composite=composite,
        uprighter_pred_deg=uprighter_pred_deg,
        uprighter_confidence=uprighter_confidence,
        kept_path=kept_path,
        source_fps=None,
        file_size_bytes=file_size_bytes,
        source_year=source_year,
        source_month=source_month,
        date_source=date_source,
        rejected_columns=rejected_columns,
    )]


def _parse_windows(value) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and value == "":
        return None
    return json.loads(value) if isinstance(value, str) else list(value)


def _is_truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def _refine_video_keeper(
    video_path: Path, candidate: _Candidate, anchor_face, cfg: WorkerConfig,
    rotation: int,
) -> tuple[np.ndarray, float, float]:
    """Decode ±refine_window_s around candidate, pick highest face-crop sharpness."""
    if anchor_face is None:
        return candidate.bgr, candidate.timestamp_s, candidate.sharpness_center
    frames = decode_window(
        video_path, candidate.timestamp_s, cfg.refine_window_s, rotation=rotation,
    )
    if not frames:
        return candidate.bgr, candidate.timestamp_s, candidate.sharpness_center

    x1, y1 = float(anchor_face.bbox[0]), float(anchor_face.bbox[1])
    x2, y2 = float(anchor_face.bbox[2]), float(anchor_face.bbox[3])
    # Frames decoded here are in original (pre-uprighter) orientation. Apply the
    # same uprighter rotation we applied to the candidate so the face bbox lines
    # up with the refined frame.
    up_deg = candidate.uprighter_pred_deg
    sharps: list[float] = []
    aligned: list[tuple[float, np.ndarray]] = []
    for ts, frame in frames:
        rotated = _rotate_bgr_cw(frame, up_deg) if up_deg else frame
        sharps.append(_face_sharpness_bgr(rotated, x1, y1, x2, y2))
        aligned.append((ts, rotated))
    best_idx = int(np.argmax(sharps))
    best_ts, best_frame = aligned[best_idx]
    return best_frame, best_ts, float(sharps[best_idx])


def _process_video(
    row: pd.Series, models: Models, cfg: WorkerConfig, timer: StageTimer,
    rejection_stats: dict[str, int],
) -> list[dict]:
    video_path = Path(row["file_path"])
    file_size_bytes = int(video_path.stat().st_size)
    source_year, source_month, date_source = extract_source_date(video_path)
    source_fps = get_video_fps(video_path)
    with timer("frame_sampling"):
        rotation = get_video_rotation(video_path)

        windows = _parse_windows(row.get("sample_windows_s"))
        if windows is not None and len(windows) > 0:
            frame_iter = sample_frames_windowed(video_path, windows, cfg.fps, rotation=rotation)
        else:
            frame_iter = sample_frames(video_path, cfg.fps, rotation=rotation)
        iter_obj = iter(frame_iter)

    candidates: list[_Candidate] = []
    accepted_times: list[float] = []

    while True:
        with timer("frame_sampling"):
            try:
                item = next(iter_obj)
            except StopIteration:
                item = None
            except Exception as e:
                logger.warning("Failed to iterate frames for %s: %s", video_path, e)
                item = None
        if item is None:
            break
        frame_index, timestamp_s, bgr = item

        with timer("temporal_dedup"):
            skip = any(
                abs(timestamp_s - t) < cfg.temporal_window_s for t in accepted_times
            )
        if skip:
            continue

        try:
            with timer("uprighter"):
                up_bgr, up_deg, up_conf = _maybe_uprighten_bgr(
                    bgr, models, cfg.uprighter_confidence, use_tta=False,
                )

            with timer("sharpness"):
                sharp = sharpness_score(up_bgr)
            if sharp < cfg.sharpness_threshold:
                continue

            with timer("face_detect"):
                detected = _detect_qualifying_faces(
                    models.face_app, up_bgr, cfg.min_face_px,
                )
            if not detected:
                continue

            fh, fw = up_bgr.shape[:2]
            passing, rejected = _split_rejected_faces(detected, fw, fh)
            _update_rejection_stats(rejection_stats, passing, rejected)
            for fi, (face, reason) in enumerate(rejected):
                _write_rejected_face_crop(
                    cfg.output_dir / "rejected", up_bgr, face, reason,
                    video_path.stem, timestamp_s, fi,
                )
        except Exception as e:
            logger.warning("Per-frame processing failed for %s: %s", video_path, e)
            break

        accepted_times.append(timestamp_s)
        candidates.append(_Candidate(
            timestamp_s=timestamp_s,
            frame_index=frame_index,
            bgr=up_bgr,
            faces=passing,
            rejected_faces=rejected,
            sharpness_center=sharp,
            uprighter_pred_deg=up_deg,
            uprighter_confidence=up_conf,
        ))

    if not candidates:
        return []

    n = len(candidates)
    face_counts = [len(c.faces) for c in candidates]

    # Batch score: aesthetics on whole frames; classifier on every qualifying face.
    pil_frames = [_bgr_to_pil(c.bgr) for c in candidates]
    with timer("aesthetics"):
        aes_raw = _score_aesthetics_batch(pil_frames, models)
        aes_norm = np.clip((aes_raw - 1.0) / 9.0, 0.0, 1.0).astype(np.float32)

    # Classifier: build one crop per face, batched across candidates.
    per_cand_face_probs: list[list] = [[None] * len(c.faces) for c in candidates]
    if models.face_quality_model is not None:
        with timer("classifier"):
            all_crops: list = []
            crop_index: list[tuple[int, int]] = []  # (cand_idx, face_idx)
            for ci, (c, pil_img) in enumerate(zip(candidates, pil_frames)):
                for fi, face in enumerate(c.faces):
                    crop = extract_face_crop_from_image(
                        pil_img,
                        face.bbox[0], face.bbox[1], face.bbox[2], face.bbox[3],
                        FACE_CROP_PADDING,
                        kps=[[float(x), float(y)] for x, y in face.kps],
                    ).resize(
                        (FACE_QUALITY_INPUT_SIZE, FACE_QUALITY_INPUT_SIZE),
                        Image.BICUBIC,
                    )
                    all_crops.append(crop)
                    crop_index.append((ci, fi))
            if all_crops:
                all_probs = _score_classifier_batch(all_crops, models)
                for k, (ci, fi) in enumerate(crop_index):
                    per_cand_face_probs[ci][fi] = all_probs[k]

    # Rank each candidate's faces by p_good descending and keep top MAX_FACE_SLOTS.
    ranked_faces: list[list] = []
    ranked_probs: list[list] = []
    for ci, c in enumerate(candidates):
        f_sorted, p_sorted = _rank_faces_by_p_good(c.faces, per_cand_face_probs[ci])
        ranked_faces.append(f_sorted[:MAX_FACE_SLOTS])
        ranked_probs.append(p_sorted[:MAX_FACE_SLOTS])

    # Composite from face_1's p_good (the highest-p_good face after ranking).
    composite = np.zeros(n, dtype=np.float32)
    for i in range(n):
        face_1_probs = ranked_probs[i][0] if ranked_probs[i] else None
        if face_1_probs is not None:
            p_good = float(face_1_probs[3])
            composite[i] = (
                CLASSIFIER_BLEND_WEIGHT * p_good
                + (1.0 - CLASSIFIER_BLEND_WEIGHT) * aes_norm[i]
            )
        else:
            composite[i] = aes_norm[i]

    survivors = list(range(n))

    with timer("dhash_dedup"):
        # Face dHash dedup, anchored on face_1. Candidates with no passing face
        # contribute no signal (hash=None) and skip face-dedup.
        face_hashes: list = []
        for i, (c, pil_img) in enumerate(zip(candidates, pil_frames)):
            if not ranked_faces[i]:
                face_hashes.append(None)
                continue
            f1 = ranked_faces[i][0]
            crop = extract_face_crop_from_image(
                pil_img,
                f1.bbox[0], f1.bbox[1], f1.bbox[2], f1.bbox[3],
                FACE_CROP_PADDING,
            )
            face_hashes.append(_dhash_pil(crop))
        order = sorted(survivors, key=lambda i: -float(composite[i]))
        survivors = _dedup_indices(face_hashes, order, cfg.face_dedup_threshold)

        # Frame dHash dedup over remaining survivors only.
        survivor_set = set(survivors)
        full_hashes: dict[int, imagehash.ImageHash] = {
            i: _dhash_pil(pil_frames[i]) for i in survivor_set
        }
        order = sorted(survivor_set, key=lambda i: -float(composite[i]))
        kept: list[imagehash.ImageHash] = []
        new_survivors: list[int] = []
        for i in order:
            h = full_hashes[i]
            if any((h - kh) <= cfg.frame_dedup_threshold for kh in kept):
                continue
            kept.append(h)
            new_survivors.append(i)
        survivors = new_survivors

    # Quality threshold + per-file cap
    survivors = [i for i in survivors if float(composite[i]) >= cfg.quality_threshold]
    survivors.sort(key=lambda i: -float(composite[i]))
    survivors = survivors[:cfg.max_per_file]
    if not survivors:
        return []

    # Refine and write
    kept_dir = cfg.output_dir / "kept"
    out: list[dict] = []
    for i in survivors:
        c = candidates[i]
        face_1 = ranked_faces[i][0] if ranked_faces[i] else None
        with timer("refinement"):
            best_bgr, best_ts, refined_sharp = _refine_video_keeper(
                video_path, c, face_1, cfg, rotation=rotation,
            )
            # TTA uprighter confidence for the final frame metadata.
            up_conf_final = c.uprighter_confidence
            if models.uprighter_model is not None:
                _, up_conf_final = _uprighter_predict(
                    models.uprighter_model, _bgr_to_pil(best_bgr), models.device,
                    use_tta=True,
                )
        with timer("jpeg_write"):
            kept_path = _write_keeper_jpeg(
                kept_dir, video_path.stem, best_ts, float(composite[i]), best_bgr,
            )
        out.append(_build_keeper_dict(
            cfg=cfg,
            video_path=video_path,
            source_type="video",
            timestamp_s=c.timestamp_s,
            refined_timestamp_s=best_ts,
            frame_index=c.frame_index,
            bgr=best_bgr,
            faces_top=_pad_to_slots(ranked_faces[i], MAX_FACE_SLOTS),
            faces_class_probs=_pad_to_slots(ranked_probs[i], MAX_FACE_SLOTS),
            face_count=face_counts[i],
            sharpness_center=c.sharpness_center,
            refined_sharpness=refined_sharp,
            aesthetics_norm=float(aes_norm[i]),
            composite=float(composite[i]),
            uprighter_pred_deg=c.uprighter_pred_deg,
            uprighter_confidence=up_conf_final,
            kept_path=kept_path,
            source_fps=source_fps,
            file_size_bytes=file_size_bytes,
            source_year=source_year,
            source_month=source_month,
            date_source=date_source,
            rejected_columns=_build_rejected_columns(c.rejected_faces),
        ))
    return out


def process_file(row: pd.Series, models: Models, cfg: WorkerConfig) -> FileResult:
    """Process one manifest row end-to-end. Never raises; returns empty keepers on failure.

    Returns a `FileResult` with both keepers and per-stage wall-clock seconds.
    `stage_times_s["total"]` is the outermost wall time for the call.
    """
    timer = StageTimer()
    t0 = time.perf_counter()
    keepers: list[dict] = []
    rejection_stats: dict[str, int] = {
        "total_faces_rejected": 0,
        "too_small": 0,
        "small_and_edge": 0,
        "frames_with_all_faces_rejected": 0,
    }
    try:
        if _is_truthy(row.get("is_duplicate", False)):
            pass
        else:
            file_type = row.get("file_type", "")
            if file_type == "image":
                keepers = _process_image(row, models, cfg, timer, rejection_stats)
            elif file_type == "video":
                keepers = _process_video(row, models, cfg, timer, rejection_stats)
            else:
                logger.warning(
                    "Unknown file_type %r for %s", file_type, row.get("file_path"),
                )
    except Exception:
        logger.exception("process_file failed for %s", row.get("file_path"))
        keepers = []
    times = {k: timer.times.get(k, 0.0) for k in STAGE_KEYS}
    times["total"] = time.perf_counter() - t0
    return FileResult(
        keepers=keepers, stage_times_s=times, rejection_stats=rejection_stats,
    )
