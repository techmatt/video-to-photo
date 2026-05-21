"""Export face crops for each labeled frame in labels.json.

Reads `save/labels.json` (keys `{video_stem}/{kept_filename}` -> `good|okay|bad|none`),
joins against `results.parquet` to recover `(video_path, timestamp_s, kept_path,
face bbox, kps)`, crops the face from the keeper JPEG using the same roll-corrected
crop the labeling UI showed, and writes a self-contained `labels/` folder.
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import pandas as pd
from PIL import Image

from still_extractor.constants import FACE_CROP_PADDING, card_key
from still_extractor.face_crop import extract_face_crop_from_image
from still_extractor.inventory import RunConfig
from still_extractor.utils import (
    parse_kps,
    safe_float as _safe_float,
    to_fwd_slash as _to_fwd_slash,
)

logger = logging.getLogger(__name__)


LABEL_TITLE_CASE = {"good": "Good", "okay": "Okay", "bad": "Bad", "none": "None"}


def _safe_str(v) -> str | None:
    if not isinstance(v, str) or not v or pd.isna(v):
        return None
    return v


def _row_card_key(row: pd.Series) -> str | None:
    """Match the key build_faces_review.py uses: `{video_stem}/{Path(kept_path).name}`."""
    stem = _safe_str(row.get("video_stem"))
    kept = _safe_str(row.get("kept_path"))
    if stem is None or kept is None:
        return None
    return card_key(stem, kept)


def _crop_face(img: Image.Image, row: pd.Series) -> Image.Image:
    """Crop to the parquet face bbox with padding and kps-based roll correction.
    If bbox is invalid, return the full image."""
    x1 = _safe_float(row.get("face_x1"))
    y1 = _safe_float(row.get("face_y1"))
    x2 = _safe_float(row.get("face_x2"))
    y2 = _safe_float(row.get("face_y2"))
    if None in (x1, y1, x2, y2):
        return img
    return extract_face_crop_from_image(
        img, x1, y1, x2, y2, FACE_CROP_PADDING, kps=parse_kps(row.get("kps")),
    )


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def build_lookup(results_df: pd.DataFrame) -> dict[str, int]:
    """Map `{stem}/{kept_filename}` -> row index in results_df."""
    lookup: dict[str, int] = {}
    collisions = 0
    for idx, row in results_df.iterrows():
        key = _row_card_key(row)
        if key is None:
            continue
        if key in lookup:
            collisions += 1
            continue
        lookup[key] = idx
    if collisions:
        logger.warning("%d duplicate card keys in results.parquet; kept first occurrence", collisions)
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export face crops for labeled frames listed in labels.json.",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Run YAML config. When provided, --results defaults to "
                             "{output_dir}/results.parquet. Explicit flag still overrides.")
    parser.add_argument("--labels-json", type=Path, default=Path("save/labels.json"),
                        help="Path to labels.json (keys: '{stem}/{kept_filename}').")
    parser.add_argument("--results", type=Path, default=None,
                        help="Path to results.parquet with kept_path + face bbox.")
    parser.add_argument("--output-dir", type=Path, default=Path("labels"),
                        help="Output directory (will contain faces/ and labels.json).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    if args.config is not None:
        cfg = RunConfig.from_yaml(args.config)
        if args.results is None:
            args.results = cfg.output_dir / "results.parquet"
    if args.results is None:
        parser.error("--results is required when --config is not provided")

    labels_in = json.loads(args.labels_json.read_text(encoding="utf-8"))
    logger.info("Loaded %d labels from %s", len(labels_in), args.labels_json)

    results_df = pd.read_parquet(args.results)
    logger.info("Results parquet %s: %d rows", args.results, len(results_df))

    lookup = build_lookup(results_df)
    logger.info("Built lookup with %d unique card keys", len(lookup))

    faces_dir = args.output_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict] = []
    missed: list[str] = []
    skipped_image: list[str] = []

    for key, raw_label in labels_in.items():
        label_lc = str(raw_label).strip().lower()
        if label_lc not in LABEL_TITLE_CASE:
            logger.warning("Skipping %s: unknown label %r", key, raw_label)
            continue

        row_idx = lookup.get(key)
        if row_idx is None:
            missed.append(key)
            continue
        row = results_df.iloc[row_idx]

        video_path = _safe_str(row.get("video_path"))
        stem = _safe_str(row.get("video_stem")) or ""
        timestamp_s = _safe_float(row.get("timestamp_s"))
        if video_path is None or timestamp_s is None:
            logger.warning("Skipping %s: missing video_path or timestamp_s", key)
            missed.append(key)
            continue

        kept = _safe_str(row.get("kept_path"))
        if kept is None:
            logger.warning("Skipping %s: missing kept_path", key)
            skipped_image.append(key)
            continue
        img_path = Path(kept)
        if not img_path.exists():
            logger.warning("Skipping %s: kept_path missing on disk: %s", key, img_path)
            skipped_image.append(key)
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            crop = _crop_face(img, row)
        except Exception as e:
            logger.warning("Skipping %s: failed to open/crop %s (%s)", key, img_path, e)
            skipped_image.append(key)
            continue

        natural_key = f"{video_path}|{timestamp_s}"
        sha256_12 = hashlib.sha256(natural_key.encode("utf-8")).hexdigest()[:12]
        out_name = f"{sha256_12}_{stem}_{timestamp_s:.3f}.jpg"
        out_path = faces_dir / out_name

        try:
            crop.save(out_path, format="JPEG", quality=92)
        except Exception as e:
            logger.warning("Skipping %s: failed to save crop %s (%s)", key, out_path, e)
            skipped_image.append(key)
            continue

        jpeg_sha256 = _sha256_hex(out_path.read_bytes())
        exported.append({
            "video_path": video_path,
            "timestamp_s": timestamp_s,
            "label": LABEL_TITLE_CASE[label_lc],
            "face_crop_path": _to_fwd_slash(out_path),
            "sha256": jpeg_sha256,
        })

    out_labels_json = args.output_dir / "labels.json"
    _atomic_write_json(out_labels_json, exported)

    total = len(labels_in)
    matched = total - len(missed)
    logger.info("=" * 60)
    logger.info("Total in labels.json: %d", total)
    logger.info("Matched to a results-parquet row: %d", matched)
    logger.info("Exported: %d", len(exported))
    logger.info("Missed (no matching row): %d", len(missed))
    if skipped_image:
        logger.info("Skipped (image read/crop/save failed): %d", len(skipped_image))
    logger.info("Wrote %s", out_labels_json)
    logger.info("Face crops in %s", faces_dir)

    if missed:
        logger.info("--- missed keys (%d) ---", len(missed))
        for k in missed:
            logger.info("  %s", k)


if __name__ == "__main__":
    main()
