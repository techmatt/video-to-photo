"""Diagnostic HTML sorted by face detection confidence ascending.

Helps identify the threshold below which false-positive face detections dominate.
"""

import base64
import html
import io
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from still_extractor.face_crop import extract_face_crop

logger = logging.getLogger(__name__)


FACE_CROP_PADDING = 20
JPEG_QUALITY = 85


def _b64_face_crop(image_path: Path, x1, y1, x2, y2, kps=None) -> str:
    img = extract_face_crop(image_path, x1, y1, x2, y2, FACE_CROP_PADDING, kps=kps)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_kps(value) -> list | None:
    if not isinstance(value, str) or not value or pd.isna(value):
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return None


def _resolve_frame_for_row(row: pd.Series) -> Path | None:
    """Prefer refined_frame_path, fall back to frame_path."""
    refined = row.get("refined_frame_path", "")
    if isinstance(refined, str) and refined and not pd.isna(refined):
        rp = Path(refined)
        if rp.exists():
            return rp
    raw = row.get("frame_path", "")
    if not isinstance(raw, str) or not raw or pd.isna(raw):
        return None
    fp = Path(raw)
    if not fp.exists():
        return None
    return fp


def _confidence_class(score: float) -> str:
    if score < 0.50:
        return "low"
    if score <= 0.70:
        return "med"
    return "high"


CSS = """
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 16px;
  background: #111;
  color: #eee;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 13px;
}
header {
  position: sticky;
  top: 0;
  background: #111;
  padding: 12px 0;
  margin-bottom: 16px;
  border-bottom: 1px solid #333;
  z-index: 10;
}
header h1 { margin: 0; font-size: 18px; font-weight: 600; }
header .hint { color: #888; margin-top: 4px; font-size: 12px; }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 12px;
}
.card {
  background: #1a1a1a;
  border-radius: 6px;
  padding: 8px;
}
.card img {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 4px;
  background: #000;
}
.score {
  margin-top: 6px;
  font-size: 22px;
  font-weight: 700;
  line-height: 1.1;
}
.score.low { color: #e05555; }
.score.med { color: #e0a030; }
.score.high { color: #55bb55; }
.meta {
  margin-top: 4px;
  font-size: 11px;
  line-height: 1.4;
  color: #bbb;
  word-break: break-all;
}
.meta .composite { color: #ccc; font-size: 12px; }
.meta .sub { color: #888; }
"""


def _build_card(row: pd.Series, b64: str) -> str:
    score = float(row["face_det_score"])
    score_class = _confidence_class(score)
    composite = float(row["composite"])
    face_w = int(round(float(row["face_w"])))
    video_stem = html.escape(str(row["video_stem"]))
    return f"""<div class="card">
  <img src="data:image/jpeg;base64,{b64}" alt="">
  <div class="score {score_class}">{score:.3f}</div>
  <div class="meta">
    <div class="composite">composite {composite:.4f}</div>
    <div class="sub">{video_stem} &middot; {face_w}px</div>
  </div>
</div>"""


def main() -> None:
    parser = ArgumentParser(
        description="Build a diagnostic HTML sorted by face_det_score ascending.",
    )
    parser.add_argument("--parquet", type=Path, required=True,
                        help="Path to index.parquet (Pass 1 output).")
    parser.add_argument("--scores-csv", type=Path, required=True,
                        help="Path to refined_scores.csv.")
    parser.add_argument("--output-html", type=Path, required=True,
                        help="Path to write the HTML file.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    parquet_df = pd.read_parquet(args.parquet)
    scores_df = pd.read_csv(args.scores_csv)
    logger.info("Loaded %d parquet rows, %d refined_scores rows",
                len(parquet_df), len(scores_df))

    scores_df = scores_df[scores_df["final_selection"].astype(bool)].copy()

    parquet_subset = parquet_df[[
        "frame_path", "face_det_score", "video_stem", "frame_index",
        "face_x1", "face_y1", "face_x2", "face_y2", "kps",
    ]]
    scores_subset = scores_df[[
        "frame_path", "refined_frame_path", "composite",
        "face_w", "sharpness_center",
    ]]

    merged = parquet_subset.merge(scores_subset, on="frame_path", how="inner")
    logger.info("Joined: %d rows (final_selection)", len(merged))

    merged = merged.sort_values("face_det_score", ascending=True).reset_index(drop=True)

    cards: list[str] = []
    skipped = 0
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="cards"):
        img_path = _resolve_frame_for_row(row)
        if img_path is None:
            logger.warning("Missing image for %s frame %s",
                           row.get("video_stem", "?"), row.get("frame_index", "?"))
            skipped += 1
            continue
        try:
            b64 = _b64_face_crop(
                img_path,
                row["face_x1"], row["face_y1"], row["face_x2"], row["face_y2"],
                kps=_parse_kps(row.get("kps")),
            )
        except Exception as e:
            logger.warning("Failed to crop %s: %s", img_path, e)
            skipped += 1
            continue
        cards.append(_build_card(row, b64))

    if skipped:
        logger.info("Skipped %d rows with missing/unreadable images", skipped)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Confidence Review &mdash; {len(cards)} frames</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Confidence Review &mdash; {len(cards)} frames</h1>
  <div class="hint">Sorted by face_det_score ascending. Scroll from top (lowest confidence) to find the false-positive threshold.</div>
</header>
<div class="grid">
{chr(10).join(cards)}
</div>
</body>
</html>
"""

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html_doc, encoding="utf-8")
    logger.info("Built %s — %d cards, %.1f MB",
                args.output_html.name, len(cards),
                args.output_html.stat().st_size / (1024 * 1024))


if __name__ == "__main__":
    main()
