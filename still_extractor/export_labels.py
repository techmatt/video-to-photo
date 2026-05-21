"""Export labeled frames + metadata to a self-contained snapshot folder.

Snapshot is keyed by the natural key (video_path, timestamp_s), which is stable
across Pass 1 re-indexes. The snapshot can later be re-merged into a fresh
refined_scores.csv via import_labels.py.
"""

import hashlib
import json
import logging
import shutil
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tiff", ".tif", ".bmp"}
VALID_LABELS = {"good", "okay", "bad", "none"}


def _normalize_label(raw: object) -> str | None:
    """Return canonical 'Good' / 'Okay' / 'Bad' / 'None', or None if raw is not a usable label."""
    if raw is None:
        return None
    if isinstance(raw, float) and pd.isna(raw):
        return None
    s = str(raw).strip().lower()
    if s in VALID_LABELS:
        return s.capitalize()
    return None


def _label_key(video_path: str, refined_frame_path: str) -> str:
    """Build the save/labels.json key: '{video_stem}/{refined_frame_filename}'."""
    return f"{Path(video_path).stem}/{Path(refined_frame_path).name}"


def _natural_key_prefix(video_path: str, timestamp_s: float) -> str:
    raw = f"{video_path}|{timestamp_s}".encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


def _safe_int(v: object) -> int | None:
    f = _safe_float(v)
    return int(f) if f is not None else None


def _is_image_source(video_path: str, timestamp_s: float, frame_index: int) -> bool:
    suffix = Path(video_path).suffix.lower()
    return suffix in IMAGE_EXTENSIONS and timestamp_s == 0.0 and frame_index == 0


def main() -> None:
    parser = ArgumentParser(
        description="Snapshot all labeled frames + metadata to a self-contained folder.",
    )
    parser.add_argument("--scores-csv", type=Path, required=True,
                        help="Path to refined_scores.csv.")
    parser.add_argument("--labels-json", type=Path, default=Path("save/labels.json"),
                        help="Path to labels.json (keyed by '{video_stem}/{frame_filename}').")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Snapshot directory to create; receives frames/, labels.json, manifest.json.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    df = pd.read_csv(args.scores_csv)
    logger.info("Loaded %d rows from %s", len(df), args.scores_csv)

    labels_dict_raw: dict = json.loads(args.labels_json.read_text(encoding="utf-8"))
    logger.info("Loaded %d label entries from %s", len(labels_dict_raw), args.labels_json)

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    label_counts: dict[str, int] = {"Good": 0, "Okay": 0, "Bad": 0, "None": 0}
    entries: list[dict] = []
    skipped_missing_path = 0
    skipped_missing_file = 0
    total_seen_labeled = 0

    for _, row in df.iterrows():
        refined = row.get("refined_frame_path")
        video_path_raw = row.get("video_path")
        if not isinstance(video_path_raw, str) or not video_path_raw:
            continue
        if not isinstance(refined, str) or not refined:
            continue

        key = _label_key(video_path_raw, refined)
        raw_label = labels_dict_raw.get(key)
        if raw_label is None:
            continue

        norm = _normalize_label(raw_label)
        if norm is None:
            logger.warning(
                "Unknown label value %r for key %s; skipping", raw_label, key,
            )
            continue
        total_seen_labeled += 1

        src = Path(refined)
        if not src.exists():
            skipped_missing_file += 1
            logger.warning("Frame file missing on disk: %s; skipping", src)
            continue

        video_path = video_path_raw
        timestamp_s = float(row["timestamp_s"])
        frame_index = int(row["frame_index"])

        prefix = _natural_key_prefix(video_path, timestamp_s)
        dest_name = f"{prefix}_{src.name}"
        dest = frames_dir / dest_name
        shutil.copy2(src, dest)
        digest = _file_sha256(dest)

        source_type = "image" if _is_image_source(video_path, timestamp_s, frame_index) else "video"

        entries.append({
            "video_path": video_path,
            "timestamp_s": timestamp_s,
            "frame_index": frame_index,
            "label": norm,
            "source_type": source_type,
            "frame_w": _safe_int(row.get("frame_w")),
            "frame_h": _safe_int(row.get("frame_h")),
            "aesthetic_score": _safe_float(row.get("aesthetic_score")),
            "pred_confidence": _safe_float(row.get("pred_confidence")),
            "coverage": _safe_float(row.get("coverage")),
            "exported_filename": dest_name,
            "sha256": digest,
        })
        label_counts[norm] += 1

    (args.output_dir / "labels.json").write_text(
        json.dumps(entries, indent=2), encoding="utf-8",
    )
    manifest = {
        "exported_at": datetime.now(UTC).isoformat(),
        "scores_csv": str(args.scores_csv),
        "labels_json": str(args.labels_json),
        "total_labeled": len(entries),
        "label_counts": label_counts,
        "natural_key": ["video_path", "timestamp_s"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )

    print(f"Exported {len(entries)} labeled frames to {args.output_dir}")
    print(
        f"  Good={label_counts['Good']} "
        f"Okay={label_counts['Okay']} "
        f"Bad={label_counts['Bad']} "
        f"None={label_counts['None']}",
    )
    total_skipped = skipped_missing_path + skipped_missing_file
    if total_skipped:
        print(
            f"  Skipped {total_skipped} row(s) "
            f"(missing refined_frame_path: {skipped_missing_path}, "
            f"file not found: {skipped_missing_file}) "
            f"of {total_seen_labeled} labeled rows",
        )


if __name__ == "__main__":
    main()
