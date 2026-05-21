"""Re-merge a labels snapshot (produced by export_labels.py) into a fresh refined_scores.csv.

The snapshot is keyed by (video_path, timestamp_s). For each entry, the
matching row in the new CSV gets its 'label' column populated. If the JPEG
referenced by refined_frame_path no longer matches the exported sha256, a
warning is emitted but the label is still applied (the frame content may have
shifted between re-indexes, but the natural key still identifies the moment).
"""

import hashlib
import json
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

KEY_PRECISION = 6  # decimals to round timestamp_s to when building the lookup


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _key(video_path: str, timestamp_s: float) -> tuple[str, float]:
    return (str(video_path), round(float(timestamp_s), KEY_PRECISION))


def main() -> None:
    parser = ArgumentParser(
        description="Re-merge a label snapshot back into a refined_scores.csv.",
    )
    parser.add_argument("--scores-csv", type=Path, required=True,
                        help="Fresh refined_scores.csv (post re-index).")
    parser.add_argument("--label-export", type=Path, required=True,
                        help="Snapshot directory produced by export_labels.py.")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Where to write merged CSV. Defaults to --scores-csv (in-place); "
                             "in-place requires --yes.")
    parser.add_argument("--yes", action="store_true",
                        help="Confirm overwriting --scores-csv in place.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    output_csv: Path = args.output_csv if args.output_csv is not None else args.scores_csv
    in_place = output_csv.resolve() == args.scores_csv.resolve()
    if in_place and not args.yes:
        logger.error(
            "Refusing to overwrite %s in-place without --yes. "
            "Re-run with --yes, or pass --output-csv <other path>.",
            args.scores_csv,
        )
        sys.exit(2)

    labels_path = args.label_export / "labels.json"
    entries = json.loads(labels_path.read_text(encoding="utf-8"))
    logger.info("Loaded %d label entries from %s", len(entries), labels_path)

    df = pd.read_csv(args.scores_csv)
    logger.info("Loaded %d rows from %s", len(df), args.scores_csv)

    if "label" not in df.columns:
        df["label"] = pd.NA

    lookup: dict[tuple[str, float], int] = {}
    for idx in range(len(df)):
        try:
            key = _key(df.at[idx, "video_path"], df.at[idx, "timestamp_s"])
        except (TypeError, ValueError):
            continue
        lookup[key] = idx

    applied = 0
    sha_mismatches = 0
    misses: list[tuple[str, float]] = []

    for entry in entries:
        vp = str(entry["video_path"])
        ts = float(entry["timestamp_s"])
        row_idx = lookup.get(_key(vp, ts))
        if row_idx is None:
            misses.append((vp, ts))
            continue

        df.at[row_idx, "label"] = entry["label"]
        applied += 1

        current_path = df.at[row_idx, "refined_frame_path"]
        expected_sha = entry.get("sha256")
        if isinstance(current_path, str) and current_path and expected_sha:
            p = Path(current_path)
            if p.exists():
                actual_sha = _file_sha256(p)
                if actual_sha != expected_sha:
                    sha_mismatches += 1
                    logger.warning(
                        "SHA256 mismatch for %s @ %.6fs (label applied anyway)",
                        vp, ts,
                    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Wrote %d rows to %s", len(df), output_csv)

    print(f"Labels re-applied: {applied}")
    print(f"SHA256 mismatches (content changed, label applied anyway): {sha_mismatches}")
    print(f"Misses (key not found in new index): {len(misses)}")
    for vp, ts in misses:
        print(f"  MISS  {vp} @ {ts:.6f}s")


if __name__ == "__main__":
    main()
