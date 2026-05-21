"""Per-file worker: decode → rotate → score → dedup → refine → write keepers.

The orchestrator (`pipeline.py`) iterates the manifest and calls `process_file`
once per row. Each call processes one source video or image entirely in memory;
only the final keeper JPEGs are written to disk. Errors are caught and logged
so a single bad file never aborts a run.
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imagehash
import numpy as np
import pandas as pd
import pillow_heif
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps

from still_extractor.constants import (
    CLASSIFIER_BLEND_WEIGHT,
    FACE_CROP_PADDING,
    FACE_QUALITY_INPUT_SIZE,
    FACE_QUALITY_LABELS,
    FACE_SHARPNESS_PADDING,
    IMAGENET_MEAN,
    IMAGENET_STD,
    UPRIGHTER_INPUT_SIZE,
    UPRIGHTER_LABELS,
)
from still_extractor.face_crop import extract_face_crop_from_image
from still_extractor.models import Models
from still_extractor.sampling import (
    _apply_rotation,
    decode_window,
    get_video_rotation,
    sample_frames,
    sample_frames_windowed,
    sharpness_score,
)

pillow_heif.register_heif_opener()

logger = logging.getLogger(__name__)

FACE_QUALITY_TTA_PASSES = 3
AESTHETICS_BATCH_SIZE = 16


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
    hashes: list[imagehash.ImageHash], order: list[int], threshold: int,
) -> list[int]:
    """Greedy dHash dedup: walk `order`, drop any hash within `threshold` of a kept one."""
    kept: list[imagehash.ImageHash] = []
    keep_idx: list[int] = []
    for i in order:
        h = hashes[i]
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
    face: Any                  # InsightFace Face object
    sharpness_center: float
    uprighter_pred_deg: int
    uprighter_confidence: float


def _detect_largest_face(face_app, bgr: np.ndarray, min_face_px: int):
    try:
        faces = face_app.get(bgr)
    except Exception as e:
        logger.warning("InsightFace failed: %s", e)
        return None
    if not faces:
        return None
    qualifying = [f for f in faces if float(f.bbox[2] - f.bbox[0]) >= min_face_px]
    if not qualifying:
        return None
    return max(qualifying, key=lambda f: float(
        (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    ))


# --- Image and video processing ---------------------------------------------

def _build_keeper_dict(
    *,
    cfg: WorkerConfig,
    video_path: Path,
    source_type: str,
    timestamp_s: float,
    refined_timestamp_s: float,
    frame_index: int,
    bgr: np.ndarray,
    face,
    sharpness_center: float,
    refined_sharpness: float,
    aesthetics_norm: float,
    composite: float,
    class_probs: np.ndarray | None,
    uprighter_pred_deg: int,
    uprighter_confidence: float,
    kept_path: Path,
) -> dict:
    h, w = bgr.shape[:2]
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
    return {
        "video_path": str(video_path.resolve()),
        "video_stem": video_path.stem,
        "source_type": source_type,
        "timestamp_s": float(timestamp_s),
        "refined_timestamp_s": float(refined_timestamp_s),
        "frame_index": int(frame_index),
        "frame_w": int(w),
        "frame_h": int(h),
        "face_x1": int(bbox[0]),
        "face_y1": int(bbox[1]),
        "face_x2": int(bbox[2]),
        "face_y2": int(bbox[3]),
        "face_w": int(bbox[2] - bbox[0]),
        "face_det_score": float(face.det_score),
        "kps": json.dumps([[float(x), float(y)] for x, y in face.kps]),
        "embedding": json.dumps([float(v) for v in face.normed_embedding]),
        "sharpness_center": float(sharpness_center),
        "refined_sharpness": float(refined_sharpness),
        "sharpness_delta": float(refined_sharpness - sharpness_center),
        "aesthetics_norm": float(aesthetics_norm),
        "composite": float(composite),
        "p_none": p_none,
        "p_bad": p_bad,
        "p_okay": p_okay,
        "p_good": p_good,
        "pred_label": pred_label,
        "pred_confidence": pred_confidence,
        "uprighter_pred": int(uprighter_pred_deg),
        "uprighter_confidence": float(uprighter_confidence),
        "kept_path": str(kept_path.resolve()),
    }


def _write_keeper_jpeg(
    kept_dir: Path, video_stem: str, timestamp_s: float, composite: float,
    bgr: np.ndarray,
) -> Path:
    kept_dir.mkdir(parents=True, exist_ok=True)
    out_path = kept_dir / f"{composite:.4f}_{video_stem}_{timestamp_s:.3f}.jpg"
    cv2.imwrite(str(out_path), bgr)
    return out_path


def _process_image(
    row: pd.Series, models: Models, cfg: WorkerConfig,
) -> list[dict]:
    image_path = Path(row["file_path"])
    try:
        with Image.open(image_path) as raw:
            pil_img = ImageOps.exif_transpose(raw).convert("RGB")
    except Exception as e:
        logger.warning("Failed to open image %s: %s", image_path, e)
        return []

    # Uprighter (3-strategy TTA for images; cheap since N=1).
    uprighter_pred_deg, uprighter_confidence = 0, 0.0
    if models.uprighter_model is not None:
        pred_deg, conf = _uprighter_predict(
            models.uprighter_model, pil_img, models.device, use_tta=True,
        )
        if conf >= cfg.uprighter_confidence and pred_deg != 0:
            pil_img = _rotate_pil_cw(pil_img, pred_deg)
            uprighter_pred_deg = pred_deg
        uprighter_confidence = conf

    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    sharp = sharpness_score(bgr)
    if sharp < cfg.sharpness_threshold:
        logger.debug("%s sharpness=%.2f DROP-sharpness", image_path.name, sharp)
        return []

    face = _detect_largest_face(models.face_app, bgr, cfg.min_face_px)
    if face is None:
        logger.debug("%s DROP-face", image_path.name)
        return []

    aes_raw = _score_aesthetics_batch([pil_img], models)[0]
    aesthetics_norm = float(np.clip((aes_raw - 1.0) / 9.0, 0.0, 1.0))

    class_probs = None
    composite = aesthetics_norm
    if models.face_quality_model is not None:
        crop = extract_face_crop_from_image(
            pil_img,
            face.bbox[0], face.bbox[1], face.bbox[2], face.bbox[3],
            FACE_CROP_PADDING,
            kps=[[float(x), float(y)] for x, y in face.kps],
        ).resize((FACE_QUALITY_INPUT_SIZE, FACE_QUALITY_INPUT_SIZE), Image.BICUBIC)
        class_probs = _score_classifier_batch([crop], models)[0]
        p_good = float(class_probs[3])
        composite = (
            CLASSIFIER_BLEND_WEIGHT * p_good
            + (1.0 - CLASSIFIER_BLEND_WEIGHT) * aesthetics_norm
        )

    if composite < cfg.quality_threshold:
        return []

    face_sharp = _face_sharpness_bgr(
        bgr, face.bbox[0], face.bbox[1], face.bbox[2], face.bbox[3],
    )
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
        face=face,
        sharpness_center=sharp,
        refined_sharpness=face_sharp,
        aesthetics_norm=aesthetics_norm,
        composite=composite,
        class_probs=class_probs,
        uprighter_pred_deg=uprighter_pred_deg,
        uprighter_confidence=uprighter_confidence,
        kept_path=kept_path,
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
    video_path: Path, candidate: _Candidate, cfg: WorkerConfig, rotation: int,
) -> tuple[np.ndarray, float, float]:
    """Decode ±refine_window_s around candidate, pick highest face-crop sharpness."""
    frames = decode_window(
        video_path, candidate.timestamp_s, cfg.refine_window_s, rotation=rotation,
    )
    if not frames:
        return candidate.bgr, candidate.timestamp_s, candidate.sharpness_center

    x1, y1 = float(candidate.face.bbox[0]), float(candidate.face.bbox[1])
    x2, y2 = float(candidate.face.bbox[2]), float(candidate.face.bbox[3])
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
    row: pd.Series, models: Models, cfg: WorkerConfig,
) -> list[dict]:
    video_path = Path(row["file_path"])
    rotation = get_video_rotation(video_path)

    windows = _parse_windows(row.get("sample_windows_s"))
    if windows is not None and len(windows) > 0:
        frame_iter = sample_frames_windowed(video_path, windows, cfg.fps, rotation=rotation)
    else:
        frame_iter = sample_frames(video_path, cfg.fps, rotation=rotation)

    candidates: list[_Candidate] = []
    accepted_times: list[float] = []

    try:
        for frame_index, timestamp_s, bgr in frame_iter:
            # Temporal dedup: skip early before running any models.
            if any(abs(timestamp_s - t) < cfg.temporal_window_s for t in accepted_times):
                continue

            up_bgr, up_deg, up_conf = _maybe_uprighten_bgr(
                bgr, models, cfg.uprighter_confidence, use_tta=False,
            )

            sharp = sharpness_score(up_bgr)
            if sharp < cfg.sharpness_threshold:
                continue

            face = _detect_largest_face(models.face_app, up_bgr, cfg.min_face_px)
            if face is None:
                continue

            accepted_times.append(timestamp_s)
            candidates.append(_Candidate(
                timestamp_s=timestamp_s,
                frame_index=frame_index,
                bgr=up_bgr,
                face=face,
                sharpness_center=sharp,
                uprighter_pred_deg=up_deg,
                uprighter_confidence=up_conf,
            ))
    except Exception as e:
        logger.warning("Failed to iterate frames for %s: %s", video_path, e)

    if not candidates:
        return []

    # Batch score: aesthetics + classifier
    pil_frames = [_bgr_to_pil(c.bgr) for c in candidates]
    aes_raw = _score_aesthetics_batch(pil_frames, models)
    aes_norm = np.clip((aes_raw - 1.0) / 9.0, 0.0, 1.0).astype(np.float32)

    if models.face_quality_model is not None:
        face_crops = []
        for c, pil_img in zip(candidates, pil_frames):
            crop = extract_face_crop_from_image(
                pil_img,
                c.face.bbox[0], c.face.bbox[1], c.face.bbox[2], c.face.bbox[3],
                FACE_CROP_PADDING,
                kps=[[float(x), float(y)] for x, y in c.face.kps],
            ).resize((FACE_QUALITY_INPUT_SIZE, FACE_QUALITY_INPUT_SIZE), Image.BICUBIC)
            face_crops.append(crop)
        class_probs = _score_classifier_batch(face_crops, models)
        composite = (
            CLASSIFIER_BLEND_WEIGHT * class_probs[:, 3]
            + (1.0 - CLASSIFIER_BLEND_WEIGHT) * aes_norm
        )
    else:
        class_probs = None
        composite = aes_norm

    n = len(candidates)
    survivors = list(range(n))

    # Face dHash dedup
    face_hashes: list[imagehash.ImageHash] = []
    for c, pil_img in zip(candidates, pil_frames):
        crop = extract_face_crop_from_image(
            pil_img,
            c.face.bbox[0], c.face.bbox[1], c.face.bbox[2], c.face.bbox[3],
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
        best_bgr, best_ts, refined_sharp = _refine_video_keeper(
            video_path, c, cfg, rotation=rotation,
        )
        # TTA uprighter confidence for the final frame metadata.
        up_conf_final = c.uprighter_confidence
        if models.uprighter_model is not None:
            _, up_conf_final = _uprighter_predict(
                models.uprighter_model, _bgr_to_pil(best_bgr), models.device,
                use_tta=True,
            )
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
            face=c.face,
            sharpness_center=c.sharpness_center,
            refined_sharpness=refined_sharp,
            aesthetics_norm=float(aes_norm[i]),
            composite=float(composite[i]),
            class_probs=class_probs[i] if class_probs is not None else None,
            uprighter_pred_deg=c.uprighter_pred_deg,
            uprighter_confidence=up_conf_final,
            kept_path=kept_path,
        ))
    return out


def process_file(row: pd.Series, models: Models, cfg: WorkerConfig) -> list[dict]:
    """Process one manifest row end-to-end. Never raises; returns [] on failure."""
    if _is_truthy(row.get("is_duplicate", False)):
        return []
    file_type = row.get("file_type", "")
    try:
        if file_type == "image":
            return _process_image(row, models, cfg)
        if file_type == "video":
            return _process_video(row, models, cfg)
        logger.warning("Unknown file_type %r for %s", file_type, row.get("file_path"))
        return []
    except Exception:
        logger.exception("process_file failed for %s", row.get("file_path"))
        return []
