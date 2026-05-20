"""Pass 2: score indexed frames, dedup, emit top-K JPEGs and a CSV."""

import json
import logging
import shutil
from argparse import ArgumentParser
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
import torch
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from PIL import Image
from tqdm import tqdm

from still_extractor.face_crop import extract_face_crop

logger = logging.getLogger(__name__)

AESTHETICS_BATCH_SIZE = 32


def _load_index(index_file: Path) -> pd.DataFrame:
    df = pd.read_parquet(index_file)
    logger.info("Loaded %d rows from %s", len(df), index_file)
    df["kps_parsed"] = df["kps"].map(json.loads)
    df["embedding_parsed"] = df["embedding"].map(json.loads)
    return df


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    logger.warning("CUDA unavailable; aesthetics scoring will run on CPU and be slow")
    return torch.device("cpu")


def score_aesthetics(df: pd.DataFrame, device: torch.device) -> pd.Series:
    """Return aesthetics score normalized to [0, 1] (raw scale is ~1-10)."""
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    predictor, preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True, torch_dtype=dtype,
    )
    predictor = predictor.to(device).eval()

    raw_scores = np.zeros(len(df), dtype=np.float32)
    paths = df["frame_path"].tolist()

    with torch.inference_mode():
        for batch_start in tqdm(
            range(0, len(paths), AESTHETICS_BATCH_SIZE), desc="aesthetics",
        ):
            batch_paths = paths[batch_start:batch_start + AESTHETICS_BATCH_SIZE]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = preprocessor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
            logits = predictor(pixel_values=pixel_values).logits
            scores = logits.squeeze(-1).float().cpu().numpy()
            raw_scores[batch_start:batch_start + len(batch_paths)] = scores

    normed = np.clip((raw_scores - 1.0) / 9.0, 0.0, 1.0)
    logger.info(
        "Aesthetics raw: mean=%.2f min=%.2f max=%.2f frac>5.5=%.2f",
        float(raw_scores.mean()), float(raw_scores.min()), float(raw_scores.max()),
        float((raw_scores > 5.5).mean()),
    )
    logger.info(
        "Aesthetics norm: mean=%.3f min=%.3f max=%.3f frac>0.5=%.2f",
        float(normed.mean()), float(normed.min()), float(normed.max()),
        float((normed > 0.5).mean()),
    )

    del predictor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return pd.Series(normed, index=df.index, name="aesthetics_norm")


def _face_crop_laplacian(row: pd.Series, padding: int = 10) -> float:
    img = cv2.imread(row["frame_path"])
    if img is None:
        logger.warning("Could not read %s for face sharpness", row["frame_path"])
        return 0.0
    h, w = img.shape[:2]
    x1 = max(0, int(row["face_x1"]) - padding)
    y1 = max(0, int(row["face_y1"]) - padding)
    x2 = min(w, int(row["face_x2"]) + padding)
    y2 = min(h, int(row["face_y2"]) + padding)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _normalize_min_max(values: np.ndarray) -> np.ndarray:
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def score_face_sharpness(df: pd.DataFrame) -> pd.Series:
    raw = np.array(
        [_face_crop_laplacian(row) for _, row in tqdm(
            df.iterrows(), total=len(df), desc="face-sharpness",
        )],
        dtype=np.float32,
    )
    normed = _normalize_min_max(raw)
    logger.info(
        "Face sharpness raw: mean=%.1f min=%.1f max=%.1f",
        float(raw.mean()), float(raw.min()), float(raw.max()),
    )
    return pd.Series(normed, index=df.index, name="face_sharpness_norm")


def score_eye_openness(df: pd.DataFrame) -> pd.Series:
    # InsightFace 5-point landmarks lack eyelid points, so a true EAR isn't
    # possible. Use inter-eye distance / face height as a coarse frontality
    # proxy. Weak signal -- down-weighted by default (--eye-weight 0.5).
    ratios = np.zeros(len(df), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        kps = row["kps_parsed"]
        left_eye = np.array(kps[0], dtype=np.float32)
        right_eye = np.array(kps[1], dtype=np.float32)
        inter_eye = float(np.linalg.norm(left_eye - right_eye))
        face_h = float(row["face_y2"] - row["face_y1"])
        ratios[i] = inter_eye / face_h if face_h > 1e-6 else 0.0

    normed = _normalize_min_max(ratios)
    logger.info(
        "Eye-spread ratio: mean=%.3f min=%.3f max=%.3f",
        float(ratios.mean()), float(ratios.min()), float(ratios.max()),
    )
    return pd.Series(normed, index=df.index, name="eye_norm")


def _composite(
    df: pd.DataFrame, aesthetics_weight: float, face_sharpness_weight: float,
    eye_weight: float,
) -> pd.Series:
    total = aesthetics_weight + face_sharpness_weight + eye_weight
    weighted = (
        aesthetics_weight * df["aesthetics_norm"]
        + face_sharpness_weight * df["face_sharpness_norm"]
        + eye_weight * df["eye_norm"]
    ) / total
    return weighted.rename("composite")


def dedup_temporal(df: pd.DataFrame, window_s: float) -> pd.Series:
    """Greedy temporal dedup within each video. True = keep."""
    keep = pd.Series(False, index=df.index)
    for _, group in df.groupby("video_path", sort=False):
        sorted_group = group.sort_values("composite", ascending=False)
        kept_times: list[float] = []
        for idx, row in sorted_group.iterrows():
            ts = float(row["timestamp_s"])
            if any(abs(ts - kt) < window_s for kt in kept_times):
                continue
            kept_times.append(ts)
            keep.loc[idx] = True
    return keep


def dedup_face_crop(df: pd.DataFrame, threshold: int) -> pd.Series:
    """Dedup by face-crop dHash, iterating in df order. True = keep."""
    keep = pd.Series(False, index=df.index)
    kept_hashes: list[imagehash.ImageHash] = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="dedup-face"):
        crop = extract_face_crop(
            Path(row["frame_path"]),
            row["face_x1"], row["face_y1"], row["face_x2"], row["face_y2"],
            padding=20,
        )
        phash = imagehash.dhash(crop, hash_size=8)
        if any((phash - kept) <= threshold for kept in kept_hashes):
            continue
        kept_hashes.append(phash)
        keep.loc[idx] = True
    return keep


def dedup_full_frame(df: pd.DataFrame, threshold: int) -> pd.Series:
    """Dedup by full-frame dHash, iterating in df order. True = keep."""
    keep = pd.Series(False, index=df.index)
    kept_hashes: list[imagehash.ImageHash] = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="dedup-frame"):
        phash = imagehash.dhash(Image.open(row["frame_path"]), hash_size=8)
        if any((phash - kept) <= threshold for kept in kept_hashes):
            continue
        kept_hashes.append(phash)
        keep.loc[idx] = True
    return keep


def _copy_top_k(
    df: pd.DataFrame, top_k: int, top_frames_dir: Path,
) -> int:
    top_frames_dir.mkdir(parents=True, exist_ok=True)
    top = df.head(top_k)
    for _, row in top.iterrows():
        dest = top_frames_dir / (
            f"{row['composite']:.4f}_{row['video_stem']}_{row['timestamp_s']:.2f}.jpg"
        )
        shutil.copy2(row["frame_path"], dest)
        logger.debug("Copied %s -> %s", row["frame_path"], dest)
    return len(top)


def _write_scores_csv(df: pd.DataFrame, csv_path: Path) -> None:
    drop_cols = [c for c in ("kps_parsed", "embedding_parsed") if c in df.columns]
    out = df.drop(columns=drop_cols)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    logger.info("Wrote %d rows to %s", len(out), csv_path)


def main() -> None:
    parser = ArgumentParser(
        description="Score indexed frames, deduplicate, and emit top-K JPEGs.",
    )
    parser.add_argument("--index-file", type=Path, default=Path("data/index.parquet"),
                        help="Parquet file from Pass 1.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"),
                        help="Root output dir; JPEGs go to {output-dir}/top_frames/.")
    parser.add_argument("--top-k", type=int, default=200,
                        help="Number of top frames to emit.")
    parser.add_argument("--aesthetics-weight", type=float, default=1.0,
                        help="Composite weight for aesthetics sub-score.")
    parser.add_argument("--face-sharpness-weight", type=float, default=1.0,
                        help="Composite weight for face sharpness sub-score.")
    parser.add_argument("--eye-weight", type=float, default=0.5,
                        help="Composite weight for eye-openness sub-score.")
    parser.add_argument("--temporal-window-s", type=float, default=2.0,
                        help="Seconds: same-video frames within this window are temporal duplicates.")
    parser.add_argument("--face-dedup-threshold", type=int, default=8,
                        help="dHash Hamming distance threshold for face-crop dedup; <= is duplicate.")
    parser.add_argument("--frame-dedup-threshold", type=int, default=8,
                        help="dHash Hamming distance threshold for full-frame dedup; <= is duplicate.")
    parser.add_argument("--dedup-threshold", type=int, default=None,
                        help="DEPRECATED: alias for --frame-dedup-threshold.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    if args.dedup_threshold is not None:
        logger.warning(
            "--dedup-threshold is deprecated; use --frame-dedup-threshold instead.",
        )
        args.frame_dedup_threshold = args.dedup_threshold

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    df = _load_index(args.index_file)
    if df.empty:
        logger.warning("Index has no rows; nothing to score")
        return

    device = _select_device()

    df["aesthetics_norm"] = score_aesthetics(df, device)
    df["face_sharpness_norm"] = score_face_sharpness(df)
    df["eye_norm"] = score_eye_openness(df)
    df["composite"] = _composite(
        df, args.aesthetics_weight, args.face_sharpness_weight, args.eye_weight,
    )

    for _, row in df.iterrows():
        logger.debug(
            "%s t=%.2f composite=%.4f aes=%.3f face=%.3f eye=%.3f",
            row["video_stem"], row["timestamp_s"], row["composite"],
            row["aesthetics_norm"], row["face_sharpness_norm"], row["eye_norm"],
        )

    df = df.sort_values("composite", ascending=False).reset_index(drop=True)

    before = len(df)

    temporal_mask = dedup_temporal(df, args.temporal_window_s)
    df["kept_after_temporal"] = temporal_mask
    s1_kept = int(temporal_mask.sum())
    logger.info(
        "dedup stage 1 (temporal): %d kept, %d dropped",
        s1_kept, before - s1_kept,
    )

    s1_survivors = df[df["kept_after_temporal"]]
    face_mask_sub = dedup_face_crop(s1_survivors, args.face_dedup_threshold)
    face_mask = pd.Series(False, index=df.index)
    face_mask.loc[face_mask_sub.index] = face_mask_sub
    df["kept_after_face_dedup"] = face_mask
    s2_kept = int(face_mask.sum())
    logger.info(
        "dedup stage 2 (face crop): %d kept, %d dropped",
        s2_kept, s1_kept - s2_kept,
    )

    s2_survivors = df[df["kept_after_face_dedup"]]
    frame_mask_sub = dedup_full_frame(s2_survivors, args.frame_dedup_threshold)
    frame_mask = pd.Series(False, index=df.index)
    frame_mask.loc[frame_mask_sub.index] = frame_mask_sub
    df["kept_after_frame_dedup"] = frame_mask
    s3_kept = int(frame_mask.sum())
    logger.info(
        "dedup stage 3 (full frame): %d kept, %d dropped",
        s3_kept, s2_kept - s3_kept,
    )

    df["dedup_kept"] = df["kept_after_frame_dedup"]
    logger.info("dedup total: %d kept from %d candidates", s3_kept, before)

    top_frames_dir = args.output_dir / "top_frames"
    deduped = df[df["dedup_kept"]].reset_index(drop=True)
    copied = _copy_top_k(deduped, args.top_k, top_frames_dir)
    logger.info("Copied top %d frames to %s", copied, top_frames_dir)

    _write_scores_csv(df, args.output_dir / "scores.csv")


if __name__ == "__main__":
    main()
