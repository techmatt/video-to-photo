"""Export face crops for each labeled frame in face_labels.json.

Reads `{output_dir}/face_labels.json` (keys `{video_stem}/{kept_filename}` ->
`good|okay|bad|none`), joins against `results.parquet` to recover
`(video_path, timestamp_s, kept_path, face bbox, kps)`, crops the face from the
keeper JPEG using the same roll-corrected crop the labeling UI showed, and
writes a self-contained `faces/` folder plus an appended `labels.json` manifest.

Idempotent: a companion `seen_hashes.json` tracks SHA-256 of every exported JPEG
so repeated runs (or runs across multiple corpora sharing an output dir) only
add new crops.

The core logic lives in `run_export()` so the HTTP export server can reuse it.
"""

import argparse
import hashlib
import io
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


def _load_seen_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not parse %s (%s); treating as empty", path, e)
        return set()
    if not isinstance(data, list):
        logger.warning("%s is not a JSON list; treating as empty", path)
        return set()
    return {str(h) for h in data if isinstance(h, str)}


def _load_existing_labels(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not parse %s (%s); starting fresh", path, e)
        return []
    if not isinstance(data, list):
        logger.warning("%s is not a JSON list; starting fresh", path)
        return []
    return data


def _derive_corpus(cfg: RunConfig | None, results_path: Path) -> str:
    if cfg is not None:
        return cfg.name
    stem = results_path.stem
    if stem == "results":
        return results_path.parent.name
    return stem


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


def run_export(
    labels_json_path: Path,
    results_path: Path,
    output_dir: Path,
    corpus_name: str,
) -> dict:
    """Run the full face-crop export pipeline.

    Reads labels from `labels_json_path`, joins against `results_path`, writes new
    face crops to `output_dir/faces/`, updates `output_dir/labels.json` and
    `output_dir/seen_hashes.json` (both idempotent via SHA-256 dedup).

    Returns counts: `new`, `skipped_already_exported`, `skipped_no_match`,
    `skipped_image_error`, `total_in_store`, `corpus`.
    """
    labels_in = json.loads(Path(labels_json_path).read_text(encoding="utf-8"))
    logger.info("Loaded %d labels from %s", len(labels_in), labels_json_path)

    results_df = pd.read_parquet(results_path)
    logger.info("Results parquet %s: %d rows", results_path, len(results_df))

    lookup = build_lookup(results_df)
    logger.info("Built lookup with %d unique card keys", len(lookup))

    output_dir = Path(output_dir)
    faces_dir = output_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes_path = output_dir / "seen_hashes.json"
    seen_hashes = _load_seen_hashes(seen_hashes_path)
    logger.info("Loaded %d hashes from %s", len(seen_hashes), seen_hashes_path)

    out_labels_json = output_dir / "labels.json"
    exported: list[dict] = _load_existing_labels(out_labels_json)
    logger.info("Loaded %d existing entries from %s", len(exported), out_labels_json)

    already_exported_keys: set[tuple[str, str, float]] = set()
    for e in exported:
        if not isinstance(e, dict):
            continue
        c = e.get("corpus")
        vp = e.get("video_path")
        ts = e.get("timestamp_s")
        if isinstance(c, str) and isinstance(vp, str) and isinstance(ts, (int, float)):
            already_exported_keys.add((c, vp, float(ts)))

    new_count = 0
    dedup_skipped = 0
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

        if (corpus_name, video_path, timestamp_s) in already_exported_keys:
            logger.debug("Skipping %s: already in manifest for corpus", key)
            dedup_skipped += 1
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

        try:
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=92)
            jpeg_bytes = buf.getvalue()
        except Exception as e:
            logger.warning("Skipping %s: failed to encode JPEG (%s)", key, e)
            skipped_image.append(key)
            continue

        jpeg_sha256 = _sha256_hex(jpeg_bytes)
        if jpeg_sha256 in seen_hashes:
            logger.debug("Skipping %s: already exported (sha256 match)", key)
            dedup_skipped += 1
            continue

        natural_key = f"{video_path}|{timestamp_s}"
        sha256_12 = hashlib.sha256(natural_key.encode("utf-8")).hexdigest()[:12]
        out_name = f"{sha256_12}_{stem}_{timestamp_s:.3f}.jpg"
        out_path = faces_dir / out_name

        try:
            out_path.write_bytes(jpeg_bytes)
        except Exception as e:
            logger.warning("Skipping %s: failed to save crop %s (%s)", key, out_path, e)
            skipped_image.append(key)
            continue

        seen_hashes.add(jpeg_sha256)
        exported.append({
            "video_path": video_path,
            "timestamp_s": timestamp_s,
            "label": LABEL_TITLE_CASE[label_lc],
            "face_crop_path": _to_fwd_slash(out_path),
            "sha256": jpeg_sha256,
            "corpus": corpus_name,
        })
        new_count += 1

    _atomic_write_json(out_labels_json, exported)
    _atomic_write_json(seen_hashes_path, sorted(seen_hashes))

    total = len(labels_in)
    matched = total - len(missed)
    logger.info("=" * 60)
    logger.info("Corpus: %s", corpus_name)
    logger.info("Total in face_labels.json: %d", total)
    logger.info("Matched to a results-parquet row: %d", matched)
    logger.info("Already exported (skipped): %d", dedup_skipped)
    logger.info("New this run: %d", new_count)
    logger.info("Total in seen_hashes.json: %d", len(seen_hashes))
    logger.info("Missed (no matching row): %d", len(missed))
    if skipped_image:
        logger.info("Skipped (image read/crop/save failed): %d", len(skipped_image))
    logger.info("Wrote %s (%d entries)", out_labels_json, len(exported))
    logger.info("Wrote %s", seen_hashes_path)
    logger.info("Face crops in %s", faces_dir)

    if missed:
        logger.info("--- missed keys (%d) ---", len(missed))
        for k in missed:
            logger.info("  %s", k)

    return {
        "new": new_count,
        "skipped_already_exported": dedup_skipped,
        "skipped_no_match": len(missed),
        "skipped_image_error": len(skipped_image),
        "total_in_store": len(seen_hashes),
        "corpus": corpus_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export face crops for labeled frames listed in face_labels.json.",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Run YAML config. When provided, --results defaults to "
                             "{output_dir}/results.parquet and --labels-json defaults to "
                             "{output_dir}/face_labels.json. Explicit flags still override.")
    parser.add_argument("--labels-json", type=Path, default=None,
                        help="Path to face_labels.json (keys: '{stem}/{kept_filename}'). "
                             "Defaults to {output_dir}/face_labels.json when --config is given.")
    parser.add_argument("--results", type=Path, default=None,
                        help="Path to results.parquet with kept_path + face bbox.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/face_labels"),
                        help="Output directory (will contain faces/, labels.json, seen_hashes.json).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    cfg: RunConfig | None = None
    if args.config is not None:
        cfg = RunConfig.from_yaml(args.config)
        if args.results is None:
            args.results = cfg.output_dir / "results.parquet"
        if args.labels_json is None:
            args.labels_json = cfg.output_dir / "face_labels.json"
    if args.results is None:
        parser.error("--results is required when --config is not provided")
    if args.labels_json is None:
        parser.error("--labels-json is required when --config is not provided")

    corpus = _derive_corpus(cfg, args.results)

    run_export(
        labels_json_path=args.labels_json,
        results_path=args.results,
        output_dir=args.output_dir,
        corpus_name=corpus,
    )


if __name__ == "__main__":
    main()
