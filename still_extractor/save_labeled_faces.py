"""Export face crops for each labeled frame in labels.json.

Reads `save/labels.json` (keys `{video_stem}/{frame_filename}` -> `good|okay|bad|none`),
joins against `refined_scores.csv` to recover `(video_path, timestamp_s, refined_frame_path,
face bbox)`, crops the face from the frame JPEG, and writes a self-contained
`labels/` folder.

Standalone: only uses pandas + Pillow + stdlib.
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


LABEL_TITLE_CASE = {"good": "Good", "okay": "Okay", "bad": "Bad", "none": "None"}


def _to_fwd_slash(p: str | Path) -> str:
    return str(p).replace("\\", "/")


def _safe_float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


def _safe_str(v) -> str | None:
    if not isinstance(v, str) or not v or pd.isna(v):
        return None
    return v


def _row_card_key(row: pd.Series) -> str | None:
    """Match the key build_index_html.py uses: `{video_stem}/{Path(frame_col).name}`,
    where `frame_col` is `refined_frame_path` if present-and-nonempty, else `frame_path`."""
    stem = _safe_str(row.get("video_stem"))
    if not stem:
        return None
    refined = _safe_str(row.get("refined_frame_path"))
    frame_path = _safe_str(row.get("frame_path"))
    chosen = refined if refined is not None else frame_path
    if chosen is None:
        return None
    return f"{stem}/{Path(chosen).name}"


def _resolve_image_for_row(row: pd.Series) -> Path | None:
    refined = _safe_str(row.get("refined_frame_path"))
    if refined is not None:
        rp = Path(refined)
        if rp.exists():
            return rp
        logger.warning("refined_frame_path missing on disk: %s", rp)
    frame_path = _safe_str(row.get("frame_path"))
    if frame_path is not None:
        fp = Path(frame_path)
        if fp.exists():
            return fp
        logger.warning("frame_path missing on disk: %s", fp)
    return None


def _crop_face(img: Image.Image, row: pd.Series) -> Image.Image:
    """Crop to the parquet face bbox. If bbox is invalid, return the full image."""
    x1 = _safe_float(row.get("face_x1"))
    y1 = _safe_float(row.get("face_y1"))
    x2 = _safe_float(row.get("face_x2"))
    y2 = _safe_float(row.get("face_y2"))
    if None in (x1, y1, x2, y2):
        return img
    w, h = img.size
    cx1 = max(0, int(x1))
    cy1 = max(0, int(y1))
    cx2 = min(w, int(x2))
    cy2 = min(h, int(y2))
    if cx2 <= cx1 or cy2 <= cy1:
        return img
    return img.crop((cx1, cy1, cx2, cy2))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def build_lookup(scores_df: pd.DataFrame) -> dict[str, int]:
    """Map `{stem}/{frame_filename}` -> row index in scores_df."""
    lookup: dict[str, int] = {}
    collisions = 0
    for idx, row in scores_df.iterrows():
        key = _row_card_key(row)
        if key is None:
            continue
        if key in lookup:
            collisions += 1
            continue
        lookup[key] = idx
    if collisions:
        logger.warning("%d duplicate card keys in scores CSV; kept first occurrence", collisions)
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export face crops for labeled frames listed in labels.json.",
    )
    parser.add_argument("--labels-json", type=Path, default=Path("save/labels.json"),
                        help="Path to labels.json (keys: '{stem}/{frame_filename}').")
    parser.add_argument("--parquet", type=Path, default=Path("data/mini/index.parquet"),
                        help="Pass 1 index parquet (schema printed for reference).")
    parser.add_argument("--scores-csv", type=Path, default=Path("data/mini/refined_scores.csv"),
                        help="refined_scores.csv with refined_frame_path + face bbox.")
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

    labels_in = json.loads(args.labels_json.read_text(encoding="utf-8"))
    logger.info("Loaded %d labels from %s", len(labels_in), args.labels_json)

    parquet_df = pd.read_parquet(args.parquet)
    logger.info("Parquet %s: %d rows", args.parquet, len(parquet_df))
    logger.info("Parquet columns: %s", list(parquet_df.columns))
    logger.info("Parquet dtypes:\n%s", parquet_df.dtypes.to_string())

    scores_df = pd.read_csv(args.scores_csv)
    logger.info("Scores CSV %s: %d rows", args.scores_csv, len(scores_df))

    lookup = build_lookup(scores_df)
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
        row = scores_df.iloc[row_idx]

        video_path = _safe_str(row.get("video_path"))
        stem = _safe_str(row.get("video_stem")) or ""
        timestamp_s = _safe_float(row.get("timestamp_s"))
        if video_path is None or timestamp_s is None:
            logger.warning("Skipping %s: missing video_path or timestamp_s", key)
            missed.append(key)
            continue

        img_path = _resolve_image_for_row(row)
        if img_path is None:
            logger.warning("Skipping %s: no readable frame image", key)
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
    logger.info("Matched to a scores-CSV row: %d", matched)
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
