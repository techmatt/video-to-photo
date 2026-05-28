"""Build a self-contained HTML calibration viewer for combined quality scores.

Renders every keeper frame in `results.parquet` as a small base64-embedded
thumbnail, sorted ascending by the combined `quality_score` computed by
`score_frames.py`. The viewer ships the per-card score components (per-face
scores, aesthetic, area fraction, face count) in `data-*` attributes so the
threshold inputs in the sidebar can recompute scores and bucket labels live
without a server roundtrip.

Output: `{output_dir}/score_calibration.html`. A separate JSON summary lands
at `{output_dir}/build_score_calibration_summary.json`.

A four-input sidebar lets the user dial in `low_medium`, `medium_high`,
`high_great`, and `zero_face_cap`. The "Copy thresholds" button writes the
current values to the clipboard as JSON -- paste that into
`{output_dir}/score_thresholds.json` and rerun `score_frames.py` to persist.

Schema (relevant columns):
- aesthetic network score: `aesthetics_norm`
- face #1 classifier probs: `p_okay`, `p_good` (mirrored as `face_1_p_*`)
- face #2/#3 classifier probs: `face_2_p_*`, `face_3_p_*`
- bboxes: `face_x1/y1/x2/y2`, `face_2_*`, `face_3_*`
- frame size: `frame_w`, `frame_h`
- face count: `face_count`
- keeper jpeg path: `kept_path`
"""

import base64
import html
import io
import json
import logging
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from still_extractor.inventory import RunConfig
from still_extractor.score_frames import (
    DEFAULT_THRESHOLDS,
    FACE_BONUS_ALPHA,
    FACE_BONUS_CAP,
    FACE_BONUS_TAU,
    FACE_SLOTS,
    Thresholds,
    WEIGHT_AESTHETIC,
    WEIGHT_AREA,
    WEIGHT_FACE,
    _f,
    _area_frac,
    _face_score,
    score_dataframe,
)

logger = logging.getLogger(__name__)


THUMB_WIDTH = 200
JPEG_QUALITY = 78


def _b64_thumbnail(image_path: Path) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w > THUMB_WIDTH:
        new_h = max(1, int(round(h * THUMB_WIDTH / w)))
        img = img.resize((THUMB_WIDTH, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _row_components(row: pd.Series) -> dict:
    """Pull the score components needed for live recomputation in JS."""
    aesthetic = _f(row.get("aesthetics_norm")) or 0.0
    aesthetic = max(0.0, min(1.0, aesthetic))

    face_scores: list[float] = []
    best_slot = -1
    best_score = -1.0
    for idx, (okay_col, good_col, *_) in enumerate(FACE_SLOTS):
        s = _face_score(_f(row.get(okay_col)), _f(row.get(good_col)))
        if s is None:
            continue
        s = max(0.0, min(1.0, s))
        face_scores.append(s)
        if s > best_score:
            best_score = s
            best_slot = idx

    area = 0.0
    if best_slot >= 0:
        x1c, y1c, x2c, y2c = FACE_SLOTS[best_slot][2:]
        frame_w = _f(row.get("frame_w"))
        frame_h = _f(row.get("frame_h"))
        a = _area_frac(
            _f(row.get(x1c)), _f(row.get(y1c)),
            _f(row.get(x2c)), _f(row.get(y2c)),
            frame_w, frame_h,
        )
        area = a if a is not None else 0.0

    return {
        "face_scores": face_scores,
        "aesthetic": aesthetic,
        "area": area,
        "face_count": int(_f(row.get("face_count")) or 0),
    }


CSS = """
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 0;
  background: #111;
  color: #eee;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 13px;
}
.layout { display: flex; min-height: 100vh; }
aside {
  width: 280px;
  flex: 0 0 280px;
  background: #181818;
  border-right: 1px solid #333;
  padding: 16px;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
}
aside h2 { margin: 0 0 12px 0; font-size: 15px; }
aside .field { margin-bottom: 10px; }
aside label {
  display: block;
  color: #aaa;
  font-size: 11px;
  margin-bottom: 3px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
aside input[type="number"] {
  width: 100%;
  background: #222;
  color: #eee;
  border: 1px solid #444;
  border-radius: 3px;
  padding: 5px 7px;
  font-size: 13px;
  font-family: inherit;
}
aside button {
  width: 100%;
  background: #2a4a7a;
  color: #eee;
  border: 1px solid #4a7fd0;
  padding: 7px;
  border-radius: 3px;
  cursor: pointer;
  font-size: 12px;
  margin-top: 6px;
}
aside button:hover { background: #355faa; }
aside .status {
  color: #7ad97a;
  font-size: 11px;
  margin-top: 6px;
  min-height: 14px;
}
aside .stats { color: #aaa; font-size: 11px; line-height: 1.7; margin-top: 12px; }
aside .stats .lbl { color: #777; }
aside .stats .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
aside .stats .sw-low { background: #888; }
aside .stats .sw-medium { background: #5a8fe0; }
aside .stats .sw-high { background: #e0a040; }
aside .stats .sw-great { background: #22C55E; }
main { flex: 1; padding: 16px; }
main h1 { margin: 0 0 12px 0; font-size: 16px; font-weight: 600; }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 10px;
}
.card {
  background: #1a1a1a;
  border: 2px solid transparent;
  border-radius: 5px;
  padding: 6px;
  position: relative;
}
.card.bucket-low { border-color: #888; }
.card.bucket-medium { border-color: #5a8fe0; }
.card.bucket-high { border-color: #e0a040; }
.card.bucket-great { border-color: #22C55E; }
.card img {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 3px;
  background: #000;
}
.card .meta { margin-top: 5px; font-size: 11px; line-height: 1.45; }
.card .meta .score { font-size: 14px; font-weight: 600; color: #fff; }
.card .meta .bucket {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 9px;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #111;
}
.card .meta .bucket.bucket-low { background: #888; }
.card .meta .bucket.bucket-medium { background: #5a8fe0; }
.card .meta .bucket.bucket-high { background: #e0a040; }
.card .meta .bucket.bucket-great { background: #22C55E; }
.card .meta .zf {
  display: inline-block;
  padding: 1px 5px;
  border-radius: 9px;
  font-size: 10px;
  background: #553;
  color: #fff;
  margin-left: 4px;
}
.card .meta .sub { color: #888; word-break: break-all; }
"""


def _js_constants() -> str:
    return (
        f"const W_FACE = {WEIGHT_FACE};\n"
        f"const W_AES  = {WEIGHT_AESTHETIC};\n"
        f"const W_AREA = {WEIGHT_AREA};\n"
        f"const FACE_ALPHA = {FACE_BONUS_ALPHA};\n"
        f"const FACE_TAU = {FACE_BONUS_TAU};\n"
        f"const FACE_BONUS_CAP = {FACE_BONUS_CAP};\n"
    )


JS = """
function getThresholds() {
  return {
    low_medium: parseFloat(document.getElementById('thr-low-medium').value),
    medium_high: parseFloat(document.getElementById('thr-medium-high').value),
    high_great: parseFloat(document.getElementById('thr-high-great').value),
    zero_face_cap: parseFloat(document.getElementById('thr-zero-face').value),
  };
}

function computeScore(comp, thr) {
  if (!comp.face_scores || comp.face_scores.length === 0) {
    return Math.max(0, Math.min(1, comp.aesthetic * thr.zero_face_cap));
  }
  const sorted = [...comp.face_scores].sort((a, b) => b - a);
  const best = sorted[0];
  let bonus = 0;
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i] >= FACE_TAU) bonus += FACE_ALPHA * sorted[i];
  }
  bonus = Math.min(bonus, FACE_BONUS_CAP);
  const faceQ = Math.min(best + bonus, 1.0);
  const score = W_FACE * faceQ + W_AES * comp.aesthetic + W_AREA * comp.area;
  return Math.max(0, Math.min(1, score));
}

function assignBucket(score, thr) {
  if (score < thr.low_medium) return 'low';
  if (score < thr.medium_high) return 'medium';
  if (score < thr.high_great) return 'high';
  return 'great';
}

function recompute() {
  const thr = getThresholds();
  const cards = Array.from(document.querySelectorAll('.card'));
  const counts = { low: 0, medium: 0, high: 0, great: 0 };
  for (const card of cards) {
    const comp = JSON.parse(card.dataset.components);
    const score = computeScore(comp, thr);
    const bucket = assignBucket(score, thr);
    counts[bucket]++;
    card.classList.remove('bucket-low', 'bucket-medium', 'bucket-high', 'bucket-great');
    card.classList.add('bucket-' + bucket);
    card.dataset.score = score.toFixed(6);
    card.querySelector('.score').textContent = score.toFixed(2);
    const bEl = card.querySelector('.bucket');
    bEl.textContent = bucket;
    bEl.classList.remove('bucket-low', 'bucket-medium', 'bucket-high', 'bucket-great');
    bEl.classList.add('bucket-' + bucket);
  }
  cards.sort((a, b) => parseFloat(a.dataset.score) - parseFloat(b.dataset.score));
  const grid = document.querySelector('.grid');
  for (const c of cards) grid.appendChild(c);
  const total = cards.length || 1;
  const pct = (n) => (100 * n / total).toFixed(1) + '%';
  document.getElementById('stat-low').textContent = counts.low + ' (' + pct(counts.low) + ')';
  document.getElementById('stat-medium').textContent = counts.medium + ' (' + pct(counts.medium) + ')';
  document.getElementById('stat-high').textContent = counts.high + ' (' + pct(counts.high) + ')';
  document.getElementById('stat-great').textContent = counts.great + ' (' + pct(counts.great) + ')';
}

function copyThresholds() {
  const thr = getThresholds();
  const json = JSON.stringify(thr, null, 2);
  navigator.clipboard.writeText(json).then(() => {
    const s = document.getElementById('copy-status');
    s.textContent = 'Copied! Paste into score_thresholds.json.';
    setTimeout(() => { s.textContent = ''; }, 4000);
  }, () => {
    const s = document.getElementById('copy-status');
    s.textContent = 'Clipboard blocked -- see console.';
    console.log(json);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  ['thr-low-medium', 'thr-medium-high', 'thr-high-great', 'thr-zero-face']
    .forEach(id => document.getElementById(id).addEventListener('input', recompute));
  document.getElementById('copy-btn').addEventListener('click', copyThresholds);
  recompute();
});
"""


def _card_html(b64: str, comp: dict, row: pd.Series) -> str:
    stem = html.escape(str(row.get("video_stem") or "?"))
    ts_val = _f(row.get("refined_timestamp_s"))
    if ts_val is None:
        ts_val = _f(row.get("timestamp_s"))
    ts_str = f"{ts_val:.2f}s" if ts_val is not None else "—"
    face_count = comp["face_count"]
    zero_face = face_count == 0 or not comp["face_scores"]
    zf_badge = '<span class="zf">no face</span>' if zero_face else ""
    components_json = html.escape(json.dumps(comp), quote=True)
    return f"""<div class="card" data-components="{components_json}">
  <img src="data:image/jpeg;base64,{b64}" alt="">
  <div class="meta">
    <div><span class="score">0.00</span> <span class="bucket bucket-low">low</span>{zf_badge}</div>
    <div class="sub">{stem} @ {ts_str}</div>
    <div class="sub">faces: {face_count}</div>
  </div>
</div>"""


def main() -> None:
    parser = ArgumentParser(
        description="Build a self-contained HTML viewer for tuning quality_score thresholds.",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Run YAML config. When provided, --results defaults to "
                             "{output_dir}/results.parquet, --output-html to "
                             "{output_dir}/score_calibration.html, and --thresholds "
                             "to {output_dir}/score_thresholds.json.")
    parser.add_argument("--results", type=Path, default=None,
                        help="Path to results.parquet (overrides --config).")
    parser.add_argument("--output-html", type=Path, default=None,
                        help="Path to write the HTML file (overrides --config).")
    parser.add_argument("--thresholds", type=Path, default=None,
                        help="Path to score_thresholds.json (overrides --config). "
                             "Used to seed the threshold inputs only.")
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
        if args.output_html is None:
            args.output_html = cfg.output_dir / "score_calibration.html"
        if args.thresholds is None:
            args.thresholds = cfg.output_dir / "score_thresholds.json"
    if args.results is None or args.output_html is None or args.thresholds is None:
        parser.error("--results, --output-html, and --thresholds are required without --config")

    thresholds = Thresholds.load(args.thresholds)

    df = pd.read_parquet(args.results)
    logger.info("Loaded %d rows from %s", len(df), args.results)

    if "quality_score" not in df.columns:
        logger.info("quality_score column missing; computing in-memory for sort order")
        df = score_dataframe(df, thresholds)

    df = df.sort_values("quality_score").reset_index(drop=True)

    cards: list[str] = []
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="thumbs"):
        kept = row.get("kept_path")
        if not isinstance(kept, str) or not kept:
            skipped += 1
            continue
        img_path = Path(kept)
        if not img_path.exists():
            logger.warning("Keeper missing on disk: %s", img_path)
            skipped += 1
            continue
        try:
            b64 = _b64_thumbnail(img_path)
        except Exception as e:
            logger.warning("Failed to thumbnail %s: %s", img_path, e)
            skipped += 1
            continue
        comp = _row_components(row)
        cards.append(_card_html(b64, comp, row))

    if skipped:
        logger.info("Skipped %d rows with missing/unreadable images", skipped)

    seed = asdict(thresholds)

    sidebar = f"""<aside>
  <h2>Threshold calibration</h2>
  <div class="field">
    <label>low &rarr; medium</label>
    <input type="number" id="thr-low-medium" step="0.01" min="0" max="1" value="{seed['low_medium']}">
  </div>
  <div class="field">
    <label>medium &rarr; high</label>
    <input type="number" id="thr-medium-high" step="0.01" min="0" max="1" value="{seed['medium_high']}">
  </div>
  <div class="field">
    <label>high &rarr; great</label>
    <input type="number" id="thr-high-great" step="0.01" min="0" max="1" value="{seed['high_great']}">
  </div>
  <div class="field">
    <label>zero-face cap</label>
    <input type="number" id="thr-zero-face" step="0.01" min="0" max="1" value="{seed['zero_face_cap']}">
  </div>
  <button id="copy-btn">Copy thresholds JSON</button>
  <div class="status" id="copy-status"></div>
  <div class="stats">
    <div><span class="swatch sw-low"></span><span class="lbl">low:</span> <span id="stat-low">0</span></div>
    <div><span class="swatch sw-medium"></span><span class="lbl">medium:</span> <span id="stat-medium">0</span></div>
    <div><span class="swatch sw-high"></span><span class="lbl">high:</span> <span id="stat-high">0</span></div>
    <div><span class="swatch sw-great"></span><span class="lbl">great:</span> <span id="stat-great">0</span></div>
    <div class="lbl" style="margin-top:8px;">total frames: {len(cards)}</div>
  </div>
</aside>"""

    js_block = _js_constants() + JS

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Still Extractor &mdash; Score Calibration</title>
<style>{CSS}</style>
</head>
<body>
<div class="layout">
{sidebar}
<main>
  <h1>Score calibration &mdash; {len(cards)} frames (sorted by quality_score ascending)</h1>
  <div class="grid">
{chr(10).join(cards)}
  </div>
</main>
</div>
<script>
{js_block}
</script>
</body>
</html>
"""

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html_doc, encoding="utf-8")
    file_size_mb = args.output_html.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%d cards, %.1f MB)",
                args.output_html, len(cards), file_size_mb)

    summary = {
        "stage": "build_score_calibration",
        "config": str(args.config) if args.config is not None else None,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "card_count": len(cards),
        "skipped": skipped,
        "output_html": str(args.output_html),
        "file_size_mb": round(file_size_mb, 2),
        "seed_thresholds": seed,
        "defaults": DEFAULT_THRESHOLDS,
    }
    summary_path = args.output_html.parent / "build_score_calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote summary to %s", summary_path)

    print(f"Frames in HTML: {len(cards)}  ({skipped} skipped)")
    print(f"Open: {args.output_html}")


if __name__ == "__main__":
    main()
