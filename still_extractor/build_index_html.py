"""Build a self-contained HTML labeling UI from Pass 2/3 scores CSV."""

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


def _resolve_image_path(raw: str, image_root: Path | None) -> Path:
    p = Path(raw)
    if p.is_absolute() or image_root is None:
        return p
    return image_root / p


def _b64_face_crop(image_path: Path, x1, y1, x2, y2) -> str:
    img = extract_face_crop(image_path, x1, y1, x2, y2, FACE_CROP_PADDING)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _pick_frame_column(df: pd.DataFrame) -> str:
    if "refined_frame_path" in df.columns and df["refined_frame_path"].notna().any():
        return "refined_frame_path"
    return "frame_path"


def _resolve_frame_for_row(
    row: pd.Series, has_refined: bool, image_root: Path | None,
) -> tuple[Path | None, str]:
    """Return (image_path, column_used) with fallback to frame_path.

    If refined_frame_path is missing/unreadable, fall back to frame_path.
    Returns (None, "") if both are missing/unreadable.
    """
    if has_refined:
        refined = row.get("refined_frame_path", "")
        if isinstance(refined, str) and refined and not pd.isna(refined):
            refined_path = _resolve_image_path(refined, image_root)
            if refined_path.exists():
                return refined_path, "refined_frame_path"
        logger.warning(
            "refined_frame_path missing/unreadable for %s, falling back to frame_path",
            row.get("video_stem", "?"),
        )

    raw = row.get("frame_path", "")
    if not isinstance(raw, str) or not raw or pd.isna(raw):
        return None, ""
    fp = _resolve_image_path(raw, image_root)
    if not fp.exists():
        return None, ""
    return fp, "frame_path"


def _safe_float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


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
header h1 { margin: 0 0 8px 0; font-size: 18px; font-weight: 600; }
.toolbar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.toolbar button {
  background: #222;
  color: #eee;
  border: 1px solid #444;
  padding: 6px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}
.toolbar button:hover { background: #2a2a2a; }
.toolbar button.active { background: #3a5fb0; border-color: #4a7fd0; }
.toolbar .summary { margin-left: auto; color: #aaa; font-size: 12px; }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 12px;
}
.card {
  position: relative;
  background: #1a1a1a;
  border: 2px solid transparent;
  border-radius: 6px;
  padding: 8px;
  outline: none;
  transition: border-color 0.15s;
}
.card:focus { border-color: #5a8fe0; }
.card.good { border-color: #22C55E; }
.card.okay { border-color: #F59E0B; }
.card.bad { border-color: #FF1111; }
.card.none { border-color: #8B0000; }
.card-hovered::after {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(255, 255, 255, 0.12);
  pointer-events: none;
  border-radius: inherit;
}
.card img {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 4px;
  background: #000;
}
.card .meta {
  margin-top: 6px;
  font-size: 11px;
  line-height: 1.4;
  color: #bbb;
  word-break: break-all;
}
.card .meta .composite { color: #fff; font-weight: 600; font-size: 13px; }
.card .meta .sub { color: #888; }
.card .actions { margin-top: 6px; display: flex; gap: 6px; }
.card .actions button {
  flex: 1;
  background: #222;
  color: #eee;
  border: 1px solid #444;
  padding: 4px 0;
  border-radius: 3px;
  cursor: pointer;
  font-size: 11px;
}
.card .actions button.good-btn:hover { background: #2a5a2a; }
.card .actions button.okay-btn:hover { background: #5a4a1a; }
.card .actions button.bad-btn:hover { background: #5a2a2a; }
.card .actions button.none-btn:hover { background: #3a1010; }
.card a { color: #6af; text-decoration: none; }
.card a:hover { text-decoration: underline; }
.legend { color: #888; font-size: 12px; margin-top: 6px; }
.legend kbd {
  background: #2a2a2a;
  border: 1px solid #444;
  border-radius: 3px;
  padding: 1px 5px;
  font-family: inherit;
  font-size: 11px;
  color: #ddd;
}
"""


JS = """
const VALID_LABELS = ['good', 'okay', 'bad', 'none'];

let currentFilter = 'all';
let activeCard = null;
let userHasHovered = false;

function getLabel(card) {
  const v = localStorage.getItem(card.dataset.filename);
  return VALID_LABELS.includes(v) ? v : null;
}

function applyLabel(card, label) {
  card.classList.remove('good', 'okay', 'bad', 'none');
  if (label === 'clear') {
    localStorage.removeItem(card.dataset.filename);
  } else {
    localStorage.setItem(card.dataset.filename, label);
    card.classList.add(label);
  }
  updateSummary();
  applyFilter();
}

function restoreLabels() {
  document.querySelectorAll('.card').forEach(card => {
    const lbl = getLabel(card);
    if (lbl) card.classList.add(lbl);
  });
}

function updateSummary() {
  const counts = { good: 0, okay: 0, bad: 0, none: 0, unreviewed: 0 };
  const cards = document.querySelectorAll('.card');
  cards.forEach(card => {
    const lbl = getLabel(card);
    if (lbl) counts[lbl]++;
    else counts.unreviewed++;
  });
  console.log('Label counts:', counts, 'cards:', cards.length);
  document.getElementById('summary').textContent =
    `${counts.none} none · ${counts.bad} bad · ${counts.okay} okay · ${counts.good} good · ${counts.unreviewed} unreviewed · ${cards.length} total`;
}

function applyFilter() {
  document.querySelectorAll('.card').forEach(card => {
    const lbl = getLabel(card);
    let show = true;
    if (currentFilter === 'good') show = lbl === 'good';
    else if (currentFilter === 'okay') show = lbl === 'okay';
    else if (currentFilter === 'bad') show = lbl === 'bad';
    else if (currentFilter === 'none') show = lbl === 'none';
    else if (currentFilter === 'unreviewed') show = !lbl;
    card.style.display = show ? '' : 'none';
  });
}

function setFilter(f) {
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.filter === f);
  });
  applyFilter();
}

function exportLabels() {
  const out = {};
  document.querySelectorAll('.card').forEach(card => {
    const lbl = getLabel(card);
    if (lbl) out[card.dataset.filename] = lbl;
  });
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'labels.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function focusedCard() {
  return document.activeElement && document.activeElement.classList.contains('card')
    ? document.activeElement : null;
}

function visibleCards() {
  return Array.from(document.querySelectorAll('.card')).filter(c => c.style.display !== 'none');
}

function moveFocus(dx, dy) {
  const cards = visibleCards();
  if (cards.length === 0) return;
  const current = focusedCard();
  if (!current) { cards[0].focus(); return; }
  const rect = current.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  let best = null, bestDist = Infinity;
  for (const c of cards) {
    if (c === current) continue;
    const r = c.getBoundingClientRect();
    const ox = r.left + r.width / 2 - cx;
    const oy = r.top + r.height / 2 - cy;
    if (dx !== 0 && Math.sign(ox) !== Math.sign(dx)) continue;
    if (dy !== 0 && Math.sign(oy) !== Math.sign(dy)) continue;
    const d = Math.hypot(ox, oy);
    if (d < bestDist) { bestDist = d; best = c; }
  }
  if (best) best.focus();
}

document.addEventListener('keydown', e => {
  if (!userHasHovered) return;
  const focused = focusedCard();
  const labelTarget = activeCard || focused;
  const k = e.key.toLowerCase();
  if (labelTarget) {
    if (k === '1' || k === 'n') { applyLabel(labelTarget, 'none'); e.preventDefault(); return; }
    if (k === '2' || k === 'b') { applyLabel(labelTarget, 'bad'); e.preventDefault(); return; }
    if (k === '3' || k === 'o') { applyLabel(labelTarget, 'okay'); e.preventDefault(); return; }
    if (k === '4' || k === 'g') { applyLabel(labelTarget, 'good'); e.preventDefault(); return; }
    if (k === 'x') { applyLabel(labelTarget, 'clear'); e.preventDefault(); return; }
  }
  if (focused) {
    if (e.key === 'ArrowLeft') { moveFocus(-1, 0); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { moveFocus(1, 0); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { moveFocus(0, -1); e.preventDefault(); }
    else if (e.key === 'ArrowDown') { moveFocus(0, 1); e.preventDefault(); }
  }
});

document.addEventListener('DOMContentLoaded', () => {
  restoreLabels();
  updateSummary();
  document.querySelectorAll('.card').forEach(card => {
    card.addEventListener('mousemove', function () {
      if (activeCard !== this) {
        if (activeCard) activeCard.classList.remove('card-hovered');
        activeCard = this;
        activeCard.classList.add('card-hovered');
      }
      userHasHovered = true;
    });
    card.addEventListener('mouseleave', function () {
      this.classList.remove('card-hovered');
    });
  });
  document.querySelectorAll('.filter-btn').forEach(b => {
    b.addEventListener('click', () => setFilter(b.dataset.filter));
  });
  document.getElementById('export-btn').addEventListener('click', exportLabels);
  document.querySelectorAll('.none-btn').forEach(b => {
    b.addEventListener('click', e => {
      e.stopPropagation();
      applyLabel(b.closest('.card'), 'none');
    });
  });
  document.querySelectorAll('.bad-btn').forEach(b => {
    b.addEventListener('click', e => {
      e.stopPropagation();
      applyLabel(b.closest('.card'), 'bad');
    });
  });
  document.querySelectorAll('.okay-btn').forEach(b => {
    b.addEventListener('click', e => {
      e.stopPropagation();
      applyLabel(b.closest('.card'), 'okay');
    });
  });
  document.querySelectorAll('.good-btn').forEach(b => {
    b.addEventListener('click', e => {
      e.stopPropagation();
      applyLabel(b.closest('.card'), 'good');
    });
  });
});
"""


def _build_card(row: pd.Series, b64: str, frame_col: str) -> str:
    frame_path = str(row[frame_col])
    fp = Path(frame_path)
    card_key = fp.name
    href = html.escape(frame_path)
    composite = _safe_float(row.get("composite"))
    aes = _safe_float(row.get("aesthetics_norm"))
    fs = _safe_float(row.get("face_sharpness_norm"))
    eye = _safe_float(row.get("eye_norm"))
    ts = _safe_float(row.get("refined_timestamp_s")) or _safe_float(row.get("timestamp_s"))
    video_stem = html.escape(str(row.get("video_stem", "")))

    composite_str = f"{composite:.4f}" if composite is not None else "—"
    aes_str = f"{aes:.3f}" if aes is not None else "—"
    fs_str = f"{fs:.3f}" if fs is not None else "—"
    eye_str = f"{eye:.3f}" if eye is not None else "—"
    ts_str = f"{ts:.3f}s" if ts is not None else "—"

    return f"""<div class="card" tabindex="0" data-filename="{html.escape(card_key)}">
  <a href="{href}" target="_blank"><img src="data:image/jpeg;base64,{b64}" alt=""></a>
  <div class="meta">
    <div class="composite">{composite_str}</div>
    <div>{video_stem} @ {ts_str}</div>
    <div class="sub">aes {aes_str} · face {fs_str} · eye {eye_str}</div>
  </div>
  <div class="actions">
    <button class="none-btn">None (1)</button>
    <button class="bad-btn">Bad (2)</button>
    <button class="okay-btn">Okay (3)</button>
    <button class="good-btn">Good (4)</button>
  </div>
</div>"""


def main() -> None:
    parser = ArgumentParser(
        description="Build a self-contained HTML labeling UI for refined or scored frames.",
    )
    parser.add_argument("--scores-csv", type=Path, default=Path("data/refined_scores.csv"),
                        help="Path to refined_scores.csv (or scores.csv if Pass 3 not run).")
    parser.add_argument("--output-html", type=Path, default=Path("data/index.html"),
                        help="Path to write the HTML file.")
    parser.add_argument("--image-root", type=Path, default=None,
                        help="Root directory for resolving relative image paths.")
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

    if "dedup_kept" in df.columns:
        df = df[df["dedup_kept"].astype(bool)].copy()
        logger.info("Filtered to %d dedup-kept rows", len(df))

    if "composite" in df.columns:
        df = df.sort_values("composite", ascending=False).reset_index(drop=True)

    frame_col = _pick_frame_column(df)
    logger.info("Using image column: %s (with per-row fallback to frame_path)", frame_col)
    has_refined = frame_col == "refined_frame_path"

    cards: list[str] = []
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="cards"):
        img_path, col_used = _resolve_frame_for_row(row, has_refined, args.image_root)
        if img_path is None:
            logger.error(
                "Both refined_frame_path and frame_path missing/unreadable for %s; skipping",
                row.get("video_stem", "?"),
            )
            skipped += 1
            continue
        try:
            b64 = _b64_face_crop(
                img_path,
                row["face_x1"], row["face_y1"], row["face_x2"], row["face_y2"],
            )
        except Exception as e:
            logger.warning("Failed to crop %s: %s", img_path, e)
            skipped += 1
            continue
        cards.append(_build_card(row, b64, col_used))

    if skipped:
        logger.info("Skipped %d rows with missing/unreadable images", skipped)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Still Extractor — Review</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Still Extractor — Review ({len(cards)} frames)</h1>
  <div class="toolbar">
    <button class="filter-btn active" data-filter="all">All</button>
    <button class="filter-btn" data-filter="none">None</button>
    <button class="filter-btn" data-filter="bad">Bad</button>
    <button class="filter-btn" data-filter="okay">Okay</button>
    <button class="filter-btn" data-filter="good">Good</button>
    <button class="filter-btn" data-filter="unreviewed">Unreviewed</button>
    <button id="export-btn">Export Labels</button>
    <span class="summary" id="summary"></span>
  </div>
  <div class="legend">
    Shortcuts: <kbd>1</kbd>/<kbd>N</kbd> None &middot; <kbd>2</kbd>/<kbd>B</kbd> Bad &middot; <kbd>3</kbd>/<kbd>O</kbd> Okay &middot; <kbd>4</kbd>/<kbd>G</kbd> Good &middot; <kbd>X</kbd> Clear &middot; <kbd>&larr;</kbd><kbd>&rarr;</kbd> Navigate
  </div>
</header>
<div class="grid">
{chr(10).join(cards)}
</div>
<script>{JS}</script>
</body>
</html>
"""

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html_doc, encoding="utf-8")
    logger.info("Wrote %s (%d cards, %.1f MB)",
                args.output_html, len(cards),
                args.output_html.stat().st_size / (1024 * 1024))


if __name__ == "__main__":
    main()
