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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from still_extractor.face_crop import extract_face_crop

logger = logging.getLogger(__name__)

AESTHETICS_BATCH_SIZE = 32

FACE_QUALITY_LABELS = ["none", "bad", "okay", "good"]
N_FACE_QUALITY_CLASSES = len(FACE_QUALITY_LABELS)
FACE_QUALITY_CROP_PADDING = 20
FACE_QUALITY_INPUT_SIZE = 128
FACE_QUALITY_TTA_PASSES = 3
FACE_QUALITY_BATCH_SIZE = 32
CLASSIFIER_BLEND_WEIGHT = 0.8
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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


def _parse_kps(value) -> list | None:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return None


def _build_face_quality_model() -> nn.Module:
    backbone = tv_models.mobilenet_v3_small(weights=None)
    in_features = backbone.classifier[3].in_features
    backbone.classifier[3] = nn.Linear(in_features, N_FACE_QUALITY_CLASSES)
    return backbone


def _load_face_quality_model(model_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    model = _build_face_quality_model()
    model.load_state_dict(state_dict)
    return model.eval().to(device)


def _build_face_quality_transforms() -> tuple[T.Compose, T.Compose]:
    val_tf = T.Compose([
        T.Resize((FACE_QUALITY_INPUT_SIZE, FACE_QUALITY_INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    train_tf = T.Compose([
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
    return train_tf, val_tf


def _extract_face_quality_crop(row: pd.Series) -> Image.Image | None:
    frame_path = row.get("frame_path")
    if not isinstance(frame_path, str) or not frame_path:
        return None
    p = Path(frame_path)
    if not p.exists():
        return None
    try:
        crop = extract_face_crop(
            p,
            row["face_x1"], row["face_y1"], row["face_x2"], row["face_y2"],
            FACE_QUALITY_CROP_PADDING,
            kps=_parse_kps(row.get("kps")),
        )
    except Exception as e:
        logger.warning("Failed to crop %s for classifier: %s", p, e)
        return None
    return crop.resize(
        (FACE_QUALITY_INPUT_SIZE, FACE_QUALITY_INPUT_SIZE), Image.BICUBIC,
    )


class _FaceQualityCropDataset(Dataset):
    def __init__(self, crops: list[Image.Image], transform: T.Compose) -> None:
        self.crops = crops
        self.transform = transform

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(self.crops[idx])


@torch.no_grad()
def _run_face_quality_pass(
    model: nn.Module, crops: list[Image.Image], transform: T.Compose,
    device: torch.device, batch_size: int,
) -> np.ndarray:
    ds = _FaceQualityCropDataset(crops, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out: list[np.ndarray] = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        out.append(probs)
    return np.concatenate(out, axis=0) if out else np.zeros((0, N_FACE_QUALITY_CLASSES))


def run_classifier_inference(
    df: pd.DataFrame, model_path: Path, device: torch.device,
    n_tta: int = FACE_QUALITY_TTA_PASSES,
    batch_size: int = FACE_QUALITY_BATCH_SIZE,
) -> pd.DataFrame:
    """Run the face-quality classifier on each row's face crop, with TTA.

    Returns a DataFrame indexed identically to `df` with columns
    `p_none_tta`, `p_bad_tta`, `p_okay_tta`, `p_good_tta`,
    `pred_label`, `pred_confidence`. Rows whose face crop could not be
    extracted are filled with NaN.
    """
    model = _load_face_quality_model(model_path, device)
    train_tf, _ = _build_face_quality_transforms()

    crops: list[Image.Image | None] = []
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="classifier-crops",
    ):
        crops.append(_extract_face_quality_crop(row))

    valid_positions = [i for i, c in enumerate(crops) if c is not None]
    valid_crops = [crops[i] for i in valid_positions]
    if not valid_crops:
        logger.warning("Classifier: no valid face crops to score")
        empty = pd.DataFrame(
            index=df.index,
            columns=[
                f"p_{lbl}_tta" for lbl in FACE_QUALITY_LABELS
            ] + ["pred_label", "pred_confidence"],
        )
        return empty

    accum = np.zeros((len(valid_crops), N_FACE_QUALITY_CLASSES), dtype=np.float32)
    n_passes = max(n_tta, 1)
    for k in range(n_passes):
        logger.info("Classifier TTA pass %d/%d", k + 1, n_passes)
        accum += _run_face_quality_pass(
            model, valid_crops, train_tf, device, batch_size,
        ).astype(np.float32)
    tta_probs = accum / n_passes

    columns = [f"p_{lbl}_tta" for lbl in FACE_QUALITY_LABELS] + [
        "pred_label", "pred_confidence",
    ]
    out = pd.DataFrame(index=df.index, columns=columns, dtype=object)
    for c, lbl in enumerate(FACE_QUALITY_LABELS):
        out[f"p_{lbl}_tta"] = np.nan
    out["pred_confidence"] = np.nan
    out["pred_label"] = ""
    for pos, row_pos in enumerate(valid_positions):
        idx = df.index[row_pos]
        for c, lbl in enumerate(FACE_QUALITY_LABELS):
            out.at[idx, f"p_{lbl}_tta"] = float(tta_probs[pos, c])
        c_best = int(np.argmax(tta_probs[pos]))
        out.at[idx, "pred_label"] = FACE_QUALITY_LABELS[c_best]
        out.at[idx, "pred_confidence"] = float(tta_probs[pos, c_best])
    for lbl in FACE_QUALITY_LABELS:
        out[f"p_{lbl}_tta"] = out[f"p_{lbl}_tta"].astype(float)
    out["pred_confidence"] = out["pred_confidence"].astype(float)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


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


def _copy_selected(df: pd.DataFrame, top_frames_dir: Path) -> int:
    if top_frames_dir.exists():
        shutil.rmtree(top_frames_dir)
    top_frames_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        dest = top_frames_dir / (
            f"{row['composite']:.4f}_{row['video_stem']}_{row['timestamp_s']:.2f}.jpg"
        )
        shutil.copy2(row["frame_path"], dest)
        logger.debug("Copied %s -> %s", row["frame_path"], dest)
    return len(df)


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
    parser.add_argument("--top-k-per-file", type=int, default=5,
                        help="Max frames to select per source file.")
    parser.add_argument("--top-k-global", type=int, default=0,
                        help="Optional global cap after per-file selection; 0 = no cap.")
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
    parser.add_argument("--classifier-model", type=Path,
                        default=Path("models/face_quality/best_model.pt"),
                        help="Path to trained face quality classifier model weights.")
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

    if args.classifier_model and args.classifier_model.exists():
        logger.info("[CLASSIFIER] Loaded model from %s", args.classifier_model)
        clf_scores = run_classifier_inference(df, args.classifier_model, device)
        df = df.join(clf_scores)
        df["composite_old"] = df["composite"]
        old_min = float(df["composite_old"].min())
        old_max = float(df["composite_old"].max())
        blended = (
            CLASSIFIER_BLEND_WEIGHT * df["p_good_tta"]
            + (1.0 - CLASSIFIER_BLEND_WEIGHT) * df["composite_old"]
        )
        df["composite"] = blended.where(df["p_good_tta"].notna(), df["composite_old"])
        n_scored = int(df["p_good_tta"].notna().sum())
        logger.info("[CLASSIFIER] Ran inference on %d candidates", n_scored)
        new_min = float(df["composite"].min())
        new_max = float(df["composite"].max())
        logger.info(
            "[CLASSIFIER] composite score range: [%.3f, %.3f] (was [%.3f, %.3f])",
            new_min, new_max, old_min, old_max,
        )
    else:
        if args.classifier_model:
            logger.warning(
                "[CLASSIFIER] Model not found at %s; proceeding with original composite",
                args.classifier_model,
            )
        else:
            logger.info(
                "[CLASSIFIER] No classifier model specified; using original composite",
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

    per_file_top = (
        df[df["dedup_kept"]]
        .sort_values("composite", ascending=False)
        .groupby("video_stem", sort=False)
        .head(args.top_k_per_file)
    )
    df["top_per_file"] = False
    df.loc[per_file_top.index, "top_per_file"] = True
    all_stems = set(df["video_stem"].unique())
    contributing_stems = set(per_file_top["video_stem"].unique())
    zero_contrib = len(all_stems - contributing_stems)
    logger.info(
        "[SELECTION] Per-file top-%d: %d rows from %d source files (%d files contributed 0)",
        args.top_k_per_file, len(per_file_top), len(contributing_stems), zero_contrib,
    )

    per_file_sorted = per_file_top.sort_values("composite", ascending=False)
    global_dedup_sub = dedup_full_frame(per_file_sorted, args.frame_dedup_threshold)
    df["global_dedup_kept"] = False
    df.loc[global_dedup_sub.index, "global_dedup_kept"] = global_dedup_sub.values
    global_kept = int(df["global_dedup_kept"].sum())
    logger.info(
        "[SELECTION] Cross-file dedup: %d rows kept, %d dropped",
        global_kept, len(per_file_top) - global_kept,
    )

    df["final_selection"] = df["global_dedup_kept"]
    if args.top_k_global > 0:
        capped_idx = (
            df[df["global_dedup_kept"]]
            .sort_values("composite", ascending=False)
            .head(args.top_k_global)
            .index
        )
        df["final_selection"] = False
        df.loc[capped_idx, "final_selection"] = True
        logger.info(
            "[SELECTION] Final selection: %d rows (capped at %d)",
            int(df["final_selection"].sum()), args.top_k_global,
        )
    else:
        logger.info(
            "[SELECTION] Final selection: %d rows (no global cap)",
            int(df["final_selection"].sum()),
        )

    top_frames_dir = args.output_dir / "top_frames"
    final = (
        df[df["final_selection"]]
        .sort_values("composite", ascending=False)
        .reset_index(drop=True)
    )
    copied = _copy_selected(final, top_frames_dir)
    logger.info("[SELECTION] Copied %d frames to %s", copied, top_frames_dir)

    _write_scores_csv(df, args.output_dir / "scores.csv")


if __name__ == "__main__":
    main()
