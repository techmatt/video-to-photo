"""Compute combined quality_score + quality_bucket for each frame in results.parquet.

Reads `{output_dir}/results.parquet` (produced by `pipeline.py`), computes a
single 0-1 quality score per row that combines:

- Per-face classifier probabilities (`p_okay`, `p_good`) -- face #1 is in the
  bare `p_*` columns; faces #2 and #3 live in `face_2_p_*` / `face_3_p_*`.
- Aesthetic network output (`aesthetics_norm`).
- Largest-face area fraction, computed from the bbox of the highest-scoring
  face and `frame_w * frame_h`.

Multi-face frames boost the score with a softmax-with-bonus: a small linear
bonus is added for each additional face whose `face_score` clears a threshold,
capped so a crowd never explodes the score.

Zero-face frames get `aesthetics_norm * zero_face_cap` -- they cannot exceed
`zero_face_cap` regardless of aesthetics, so they cannot land in "great".

Thresholds for the bucket assignment are read from
`{output_dir}/score_thresholds.json` when present, otherwise the defaults at
the top of this module.

Writes two new columns back to the parquet in place: `quality_score` (float)
and `quality_bucket` ("low" | "medium" | "high" | "great").
"""

import json
import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from still_extractor.inventory import RunConfig

logger = logging.getLogger(__name__)


FACE_BONUS_ALPHA = 0.15
FACE_BONUS_TAU = 0.4
FACE_BONUS_CAP = 0.25

WEIGHT_FACE = 0.60
WEIGHT_AESTHETIC = 0.30
WEIGHT_AREA = 0.10

DEFAULT_THRESHOLDS = {
    "low_medium": 0.35,
    "medium_high": 0.55,
    "high_great": 0.75,
    "zero_face_cap": 0.72,
}

FACE_SLOTS: list[tuple[str, str, str, str, str, str]] = [
    ("p_okay", "p_good", "face_x1", "face_y1", "face_x2", "face_y2"),
    ("face_2_p_okay", "face_2_p_good", "face_2_x1", "face_2_y1", "face_2_x2", "face_2_y2"),
    ("face_3_p_okay", "face_3_p_good", "face_3_x1", "face_3_y1", "face_3_x2", "face_3_y2"),
]


@dataclass
class Thresholds:
    low_medium: float
    medium_high: float
    high_great: float
    zero_face_cap: float

    @classmethod
    def load(cls, path: Path) -> "Thresholds":
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                logger.info("Loaded thresholds from %s", path)
                return cls(
                    low_medium=float(data["low_medium"]),
                    medium_high=float(data["medium_high"]),
                    high_great=float(data["high_great"]),
                    zero_face_cap=float(data["zero_face_cap"]),
                )
            except Exception as e:
                logger.warning("Failed to parse %s (%s); using defaults", path, e)
        else:
            logger.info("No %s found; using default thresholds", path)
        return cls(**DEFAULT_THRESHOLDS)


def _f(val) -> float | None:
    """Float or None (treat NaN as None)."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return f


def _face_score(p_okay: float | None, p_good: float | None) -> float | None:
    if p_okay is None and p_good is None:
        return None
    o = p_okay if p_okay is not None else 0.0
    g = p_good if p_good is not None else 0.0
    return 0.5 * o + 1.0 * g


def _area_frac(
    x1: float | None,
    y1: float | None,
    x2: float | None,
    y2: float | None,
    frame_w: float | None,
    frame_h: float | None,
) -> float | None:
    if None in (x1, y1, x2, y2, frame_w, frame_h):
        return None
    if frame_w <= 0 or frame_h <= 0:
        return None
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    frac = (w * h) / (frame_w * frame_h)
    return max(0.0, min(1.0, frac))


def _compute_face_q(face_scores: list[float]) -> float:
    """Soft-max-with-bonus combination of per-face scores."""
    if not face_scores:
        return 0.0
    ordered = sorted(face_scores, reverse=True)
    best = ordered[0]
    bonus = sum(FACE_BONUS_ALPHA * s for s in ordered[1:] if s >= FACE_BONUS_TAU)
    bonus = min(bonus, FACE_BONUS_CAP)
    return min(best + bonus, 1.0)


def compute_quality_score(
    row: pd.Series,
    thresholds: Thresholds,
    area_missing_warned: list[bool],
) -> float:
    """Compute combined 0-1 quality_score for a single row."""
    aesthetic = _f(row.get("aesthetics_norm")) or 0.0
    aesthetic = max(0.0, min(1.0, aesthetic))

    per_face: list[tuple[float, int]] = []  # (face_score, slot_index)
    for idx, (okay_col, good_col, *_) in enumerate(FACE_SLOTS):
        score = _face_score(_f(row.get(okay_col)), _f(row.get(good_col)))
        if score is not None:
            per_face.append((max(0.0, min(1.0, score)), idx))

    if not per_face:
        return aesthetic * thresholds.zero_face_cap

    per_face.sort(key=lambda t: t[0], reverse=True)
    face_scores = [s for s, _ in per_face]
    face_q = _compute_face_q(face_scores)

    best_slot = per_face[0][1]
    x1c, y1c, x2c, y2c = FACE_SLOTS[best_slot][2:]
    frame_w = _f(row.get("frame_w"))
    frame_h = _f(row.get("frame_h"))
    area = _area_frac(
        _f(row.get(x1c)), _f(row.get(y1c)),
        _f(row.get(x2c)), _f(row.get(y2c)),
        frame_w, frame_h,
    )
    if area is None:
        if not area_missing_warned[0]:
            logger.warning(
                "Missing face bbox or frame size on at least one row; "
                "treating area fraction as 0.0 for those rows",
            )
            area_missing_warned[0] = True
        area = 0.0

    return (
        WEIGHT_FACE * face_q
        + WEIGHT_AESTHETIC * aesthetic
        + WEIGHT_AREA * area
    )


def assign_bucket(score: float, thresholds: Thresholds) -> str:
    if score < thresholds.low_medium:
        return "low"
    if score < thresholds.medium_high:
        return "medium"
    if score < thresholds.high_great:
        return "high"
    return "great"


def score_dataframe(df: pd.DataFrame, thresholds: Thresholds) -> pd.DataFrame:
    area_missing_warned = [False]
    scores: list[float] = []
    buckets: list[str] = []
    for _, row in df.iterrows():
        s = compute_quality_score(row, thresholds, area_missing_warned)
        s = max(0.0, min(1.0, s))
        scores.append(s)
        buckets.append(assign_bucket(s, thresholds))
    df = df.copy()
    df["quality_score"] = scores
    df["quality_bucket"] = buckets
    return df


def _print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    print(f"Total rows: {total}")
    print(f"quality_score  min={df['quality_score'].min():.4f}  "
          f"max={df['quality_score'].max():.4f}  "
          f"mean={df['quality_score'].mean():.4f}")
    print()
    print("Bucket distribution:")
    for bucket in ("low", "medium", "high", "great"):
        count = int((df["quality_bucket"] == bucket).sum())
        pct = 100.0 * count / total if total else 0.0
        print(f"  {bucket:<7} {count:>6}  ({pct:5.1f}%)")


def _spot_check(df: pd.DataFrame, n: int = 3) -> None:
    print()
    print("Spot-check (3 rows per bucket, sorted by quality_score):")
    for bucket in ("low", "medium", "high", "great"):
        sub = df[df["quality_bucket"] == bucket].sort_values("quality_score")
        if sub.empty:
            print(f"  [{bucket}] (no rows)")
            continue
        sample = sub.head(n)
        print(f"  [{bucket}] (showing {len(sample)} of {len(sub)})")
        for _, r in sample.iterrows():
            stem = r.get("video_stem", "?")
            ts = r.get("refined_timestamp_s") or r.get("timestamp_s") or 0.0
            print(
                f"    score={r['quality_score']:.3f}  "
                f"faces={int(r.get('face_count', 0) or 0)}  "
                f"aes={_f(r.get('aesthetics_norm')) or 0.0:.3f}  "
                f"{stem} @ {float(ts):.2f}s"
            )


def main() -> None:
    parser = ArgumentParser(
        description="Compute quality_score + quality_bucket and write them back to results.parquet.",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Run YAML config. When provided, --results defaults to "
                             "{output_dir}/results.parquet and the thresholds file "
                             "defaults to {output_dir}/score_thresholds.json.")
    parser.add_argument("--results", type=Path, default=None,
                        help="Path to results.parquet (overrides --config).")
    parser.add_argument("--thresholds", type=Path, default=None,
                        help="Path to score_thresholds.json (overrides --config).")
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
        if args.thresholds is None:
            args.thresholds = cfg.output_dir / "score_thresholds.json"
    if args.results is None:
        parser.error("--results is required when --config is not provided")
    if args.thresholds is None:
        parser.error("--thresholds is required when --config is not provided")

    thresholds = Thresholds.load(args.thresholds)
    logger.info(
        "Thresholds: low_medium=%.3f medium_high=%.3f high_great=%.3f zero_face_cap=%.3f",
        thresholds.low_medium, thresholds.medium_high,
        thresholds.high_great, thresholds.zero_face_cap,
    )

    df = pd.read_parquet(args.results)
    logger.info("Loaded %d rows from %s", len(df), args.results)

    df = score_dataframe(df, thresholds)
    df.to_parquet(args.results, index=False)
    logger.info("Wrote quality_score + quality_bucket back to %s", args.results)

    _print_summary(df)
    _spot_check(df)


if __name__ == "__main__":
    main()
