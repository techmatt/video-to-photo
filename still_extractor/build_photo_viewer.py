"""Build a self-contained HTML photo viewer for browsing and flagging refined frames."""

import html
import json
import logging
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from PIL import ExifTags, Image

from still_extractor.inventory import RunConfig

logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tiff", ".tif", ".bmp"}

_EXIF_ORIENTATION_TAG = next(
    k for k, v in ExifTags.TAGS.items() if v == "Orientation"
)
# EXIF orientation tag -> clockwise rotation needed for correct display
_EXIF_ORIENTATION_TO_DEG = {1: 0, 3: 180, 6: 90, 8: 270}


def _safe_float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


def _to_fwd_slash(p: str | Path) -> str:
    return str(p).replace("\\", "/")


def get_image_rotation_deg(image_path: str | Path) -> int:
    """Return clockwise degrees (0/90/180/270) to rotate the image for correct display."""
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if not exif:
                return 0
            return _EXIF_ORIENTATION_TO_DEG.get(exif.get(_EXIF_ORIENTATION_TAG, 1), 0)
    except Exception:
        return 0


def _make_img_src(img_path: Path, html_dir: Path) -> str:
    """Return a URL-safe src for the image. Use relative path when possible, else file://."""
    img_abs = Path(img_path).resolve()
    try:
        rel = img_abs.relative_to(html_dir.resolve())
        return quote(_to_fwd_slash(rel))
    except ValueError:
        return "file:///" + quote(_to_fwd_slash(img_abs), safe="/:")


CSS = """
* { box-sizing: border-box; }
html, body { min-width: 1400px; }
body {
  margin: 0;
  padding: 16px;
  background: #1a1a1a;
  color: #fff;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 13px;
}
header {
  position: sticky;
  top: 0;
  background: #1a1a1a;
  padding: 12px 0;
  margin-bottom: 16px;
  border-bottom: 1px solid #333;
  z-index: 10;
}
header h1 { margin: 0 0 8px 0; font-size: 18px; font-weight: 600; }
.toolbar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 6px; }
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
.toolbar .toolbar-label { color: #888; font-size: 12px; margin-right: 4px; }
.toolbar .toolbar-sep { color: #444; padding: 0 4px; }

.grid {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  align-content: flex-start;
  width: 100%;
}
.photo-card {
  position: relative;
  background: #111;
  border: 3px solid transparent;
  border-radius: 6px;
  overflow: hidden;
  transition: border-color 0.12s;
  /* width and height set by JS (justifyGrid) */
}
.photo-card.flagged { border-color: #F59E0B; }
.photo-card img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
  background: #000;
  cursor: zoom-in;
  transform-origin: center center;
}
.photo-card .overlay {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  padding: 8px 10px;
  background: linear-gradient(to top, rgba(0,0,0,0.85), rgba(0,0,0,0));
  color: #fff;
  font-size: 12px;
  line-height: 1.4;
  opacity: 0;
  transition: opacity 0.12s;
  pointer-events: none;
}
.photo-card:hover .overlay { opacity: 1; }
.photo-card .overlay .stem { color: #bbb; font-size: 11px; word-break: break-all; }

.video-badge {
  position: absolute;
  top: 4px;
  right: 4px;
  width: 20px;
  height: 20px;
  background: rgba(0,0,0,0.55);
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: none;
  z-index: 10;
}
.photo-card[data-source-type="image"] .video-badge { display: none; }

.flag-btn {
  position: absolute;
  top: 6px;
  right: 6px;
  background: rgba(20, 20, 20, 0.7);
  color: #eee;
  border: 1px solid #555;
  padding: 4px 8px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  z-index: 2;
}
.flag-btn:hover { background: rgba(40, 40, 40, 0.9); }
.photo-card.flagged .flag-btn {
  background: #F59E0B;
  color: #1a1a1a;
  border-color: #F59E0B;
  font-weight: 600;
}

#lightbox {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.92);
  z-index: 100;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  padding: 24px;
}
#lightbox.open { display: flex; }
#lightbox img {
  max-width: 90vw;
  max-height: 85vh;
  object-fit: contain;
  background: #000;
}
#lightbox-meta {
  margin-top: 12px;
  color: #ddd;
  font-size: 13px;
  text-align: center;
  word-break: break-all;
  max-width: 90vw;
}
#lightbox-close {
  position: absolute;
  top: 16px;
  right: 16px;
  background: #222;
  color: #eee;
  border: 1px solid #444;
  padding: 6px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

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
const FILTER_KEY = 'photoViewer.filter';
const SOURCE_FILTER_KEY = 'photoViewer.sourceFilter';
const SORT_KEY = 'photoViewer.sort';
const FLAG_PREFIX = 'flag:';

let currentFilter = 'good';
let currentSourceFilter = 'all';
let currentSort = 'confidence';
let lightboxIndex = -1;
let lastLayoutContainerWidth = -1;

function flagKey(card) { return FLAG_PREFIX + card.dataset.exportPath; }
function isFlagged(card) { return localStorage.getItem(flagKey(card)) === '1'; }

function setFlagged(card, flag) {
  if (flag) {
    localStorage.setItem(flagKey(card), '1');
    card.dataset.flagged = 'true';
    card.classList.add('flagged');
  } else {
    localStorage.removeItem(flagKey(card));
    card.dataset.flagged = 'false';
    card.classList.remove('flagged');
  }
  updateFlagCount();
}

function toggleFlag(card) { setFlagged(card, !isFlagged(card)); }

function restoreFlags() {
  document.querySelectorAll('.photo-card').forEach(card => {
    if (isFlagged(card)) {
      card.dataset.flagged = 'true';
      card.classList.add('flagged');
    }
  });
}

function clearAllFlags() {
  if (!confirm('Clear all flags? This cannot be undone.')) return;
  document.querySelectorAll('.photo-card').forEach(card => setFlagged(card, false));
}

function flagAllVisible() {
  visibleCards().forEach(card => setFlagged(card, true));
}

function updateFlagCount() {
  const n = document.querySelectorAll('.photo-card.flagged').length;
  document.getElementById('export-btn').textContent = `Export Flagged (${n})`;
}

function passesFilter(card) {
  if (currentFilter !== 'all') {
    if ((card.dataset.predLabel || '').toLowerCase() !== currentFilter) return false;
  }
  if (currentSourceFilter !== 'all') {
    if (card.dataset.sourceType !== currentSourceFilter) return false;
  }
  return true;
}

function visibleCards() {
  return Array.from(document.querySelectorAll('.photo-card')).filter(c => c.style.display !== 'none');
}

function updateShownCount() {
  const n = visibleCards().length;
  document.getElementById('shown-count').textContent = n;
}

function applyFilter() {
  document.querySelectorAll('.photo-card').forEach(card => {
    card.style.display = passesFilter(card) ? '' : 'none';
  });
  updateShownCount();
  relayout(true);
}

const TARGET_ROW_HEIGHT = 220;
const GRID_SPACING = 4;

function justifyGrid(cards, containerWidth, targetRowHeight, spacing) {
  // READ PASS: compute layout entirely in memory, no DOM writes.
  const layout = []; // [{ row: [cardObj,...], widths: [int,...], height: int }]
  let row = [];
  let rowAspectSum = 0;

  for (const card of cards) {
    row.push(card);
    rowAspectSum += card.aspectRatio;
    const totalSpacing = spacing * (row.length - 1);
    const rowHeight = (containerWidth - totalSpacing) / rowAspectSum;
    if (rowHeight <= targetRowHeight) {
      const widths = new Array(row.length);
      let usedWidth = 0;
      for (let i = 0; i < row.length; i++) {
        let w;
        if (i === row.length - 1) {
          // last card absorbs rounding remainder so total fits exactly
          w = containerWidth - totalSpacing - usedWidth;
        } else {
          w = Math.floor(row[i].aspectRatio * rowHeight);
          usedWidth += w;
        }
        widths[i] = w;
      }
      layout.push({ row, widths, height: Math.round(rowHeight) });
      row = [];
      rowAspectSum = 0;
    }
  }
  if (row.length > 0) {
    // trailing partial row: left-justified at target row height
    const widths = row.map(c => Math.round(c.aspectRatio * targetRowHeight));
    layout.push({ row, widths, height: Math.round(targetRowHeight) });
  }

  // WRITE PASS: apply all style changes with no interleaved reads.
  for (const { row: rowCards, widths, height } of layout) {
    const heightPx = height + 'px';
    for (let i = 0; i < rowCards.length; i++) {
      const s = rowCards[i].element.style;
      s.width = widths[i] + 'px';
      s.height = heightPx;
      s.flexShrink = '0';
    }
  }
}

function relayout(force) {
  const grid = document.querySelector('.grid');
  if (!grid) return;
  const containerWidth = grid.clientWidth;
  if (containerWidth <= 0) return;
  if (!force && containerWidth === lastLayoutContainerWidth) return;
  lastLayoutContainerWidth = containerWidth;
  const cards = visibleCards().map(el => ({
    element: el,
    aspectRatio: parseFloat(el.dataset.aspect) || 1.0,
  }));
  justifyGrid(cards, containerWidth, TARGET_ROW_HEIGHT, GRID_SPACING);
  applyRotations();
}

function applyRotations() {
  document.querySelectorAll('.photo-card').forEach(card => {
    const deg = parseInt(card.dataset.rotation || '0', 10);
    const img = card.querySelector('img');
    if (!img) return;
    if (deg === 0) {
      img.style.transform = '';
      img.style.width = '';
      img.style.height = '';
      return;
    }
    img.style.transform = `rotate(${deg}deg)`;
    if (deg === 90 || deg === 270) {
      // After CSS-rotating 90/270, the image's natural axis is perpendicular to
      // the card. Set its layout width/height to the card's opposite dimension
      // so the rotated bitmap fills the card.
      img.style.width = card.style.height;
      img.style.height = card.style.width;
    }
  });
}

function setFilter(f) {
  currentFilter = f;
  localStorage.setItem(FILTER_KEY, f);
  document.querySelectorAll('.filter-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.filter === f);
  });
  applyFilter();
}

function setSourceFilter(f) {
  currentSourceFilter = f;
  localStorage.setItem(SOURCE_FILTER_KEY, f);
  document.querySelectorAll('.source-filter-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.sourceFilter === f);
  });
  applyFilter();
}

function sortCards(mode) {
  currentSort = mode;
  localStorage.setItem(SORT_KEY, mode);
  const grid = document.querySelector('.grid');
  const cards = Array.from(grid.querySelectorAll('.photo-card'));
  const key = mode === 'aesthetic' ? 'aesthetic' : (mode === 'coverage' ? 'coverage' : 'predConfidence');
  cards.sort((a, b) => {
    const va = parseFloat(a.dataset[key]);
    const vb = parseFloat(b.dataset[key]);
    const fa = isNaN(va) ? -Infinity : va;
    const fb = isNaN(vb) ? -Infinity : vb;
    return fb - fa;
  });
  cards.forEach(c => grid.appendChild(c));
  document.querySelectorAll('.sort-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.sort === mode);
  });
  relayout(true);
}

function openLightbox(idx) {
  const cards = visibleCards();
  if (idx < 0 || idx >= cards.length) return;
  lightboxIndex = idx;
  const card = cards[idx];
  const img = card.querySelector('img');
  document.getElementById('lightbox-img').src = img.src;
  const pl = card.dataset.predLabel || '-';
  const pc = parseFloat(card.dataset.predConfidence);
  const aes = parseFloat(card.dataset.aesthetic);
  const cov = parseFloat(card.dataset.coverage);
  const stem = card.dataset.videoStem || '';
  const flagged = card.classList.contains('flagged') ? ' [FLAGGED]' : '';
  document.getElementById('lightbox-meta').textContent =
    `pred: ${pl} (${isNaN(pc) ? '-' : pc.toFixed(2)})` +
    `  |  aes: ${isNaN(aes) ? '-' : aes.toFixed(2)}` +
    `  |  coverage: ${isNaN(cov) ? '-' : Math.round(cov * 100) + '%'}` +
    `  |  ${stem}${flagged}`;
  document.getElementById('lightbox').classList.add('open');
}

function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
  document.getElementById('lightbox-img').src = '';
  lightboxIndex = -1;
}

function lightboxNav(dir) {
  if (lightboxIndex < 0) return;
  const cards = visibleCards();
  const next = lightboxIndex + dir;
  if (next < 0 || next >= cards.length) return;
  openLightbox(next);
}

function activeLightboxCard() {
  if (lightboxIndex < 0) return null;
  const cards = visibleCards();
  if (lightboxIndex >= cards.length) return null;
  return cards[lightboxIndex];
}

function exportFlagged() {
  const paths = Array.from(document.querySelectorAll('.photo-card.flagged'))
    .map(c => c.dataset.exportPath);
  if (paths.length === 0) {
    alert('No cards flagged. Use the flag button on each card, or "Flag Visible".');
    return;
  }
  const data = {
    export_paths: paths,
    exported_at: new Date().toISOString(),
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'flagged.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

document.addEventListener('DOMContentLoaded', () => {
  restoreFlags();
  const savedFilter = localStorage.getItem(FILTER_KEY);
  const savedSourceFilter = localStorage.getItem(SOURCE_FILTER_KEY);
  const savedSort = localStorage.getItem(SORT_KEY);
  sortCards(savedSort && ['confidence', 'aesthetic', 'coverage'].includes(savedSort) ? savedSort : 'confidence');
  currentSourceFilter = savedSourceFilter && ['all', 'video', 'image'].includes(savedSourceFilter) ? savedSourceFilter : 'all';
  document.querySelectorAll('.source-filter-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.sourceFilter === currentSourceFilter);
  });
  setFilter(savedFilter && ['all', 'good', 'okay', 'bad', 'none'].includes(savedFilter) ? savedFilter : 'good');
  updateFlagCount();

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => relayout(false), 250);
  });

  document.querySelectorAll('.filter-btn').forEach(b => {
    b.addEventListener('click', () => setFilter(b.dataset.filter));
  });
  document.querySelectorAll('.source-filter-btn').forEach(b => {
    b.addEventListener('click', () => setSourceFilter(b.dataset.sourceFilter));
  });
  document.querySelectorAll('.sort-btn').forEach(b => {
    b.addEventListener('click', () => sortCards(b.dataset.sort));
  });
  document.getElementById('export-btn').addEventListener('click', exportFlagged);
  document.getElementById('clear-flags-btn').addEventListener('click', clearAllFlags);
  document.getElementById('flag-visible-btn').addEventListener('click', flagAllVisible);

  document.querySelectorAll('.photo-card').forEach(card => {
    card.querySelector('.flag-btn').addEventListener('click', e => {
      e.stopPropagation();
      toggleFlag(card);
    });
    card.querySelector('img').addEventListener('click', e => {
      e.stopPropagation();
      const cards = visibleCards();
      const i = cards.indexOf(card);
      if (i >= 0) openLightbox(i);
    });
  });

  document.getElementById('lightbox').addEventListener('click', e => {
    if (e.target.id === 'lightbox') closeLightbox();
  });
  document.getElementById('lightbox-close').addEventListener('click', closeLightbox);

  document.addEventListener('keydown', e => {
    if (lightboxIndex >= 0) {
      if (e.key === 'Escape') { closeLightbox(); e.preventDefault(); }
      else if (e.key === 'ArrowLeft') { lightboxNav(-1); e.preventDefault(); }
      else if (e.key === 'ArrowRight') { lightboxNav(1); e.preventDefault(); }
      else if (e.key === ' ') {
        const card = activeLightboxCard();
        if (card) { toggleFlag(card); openLightbox(lightboxIndex); }
        e.preventDefault();
      }
    }
  });
});
"""


VIDEO_BADGE_HTML = (
    '<div class="video-badge" title="Video source">'
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="14" height="14" fill="white">'
    '<path d="M8 5v14l11-7z"/>'
    '</svg>'
    '</div>'
)


def _build_card(
    row: pd.Series,
    thumb_src: str,
    export_path: str,
    aspect: float,
    rotation: int,
    is_image_source: bool,
) -> str:
    pred_raw = row.get("pred_label")
    pred_label = (
        pred_raw
        if isinstance(pred_raw, str) and pred_raw and not pd.isna(pred_raw)
        else None
    )
    pred_conf = _safe_float(row.get("pred_confidence"))
    aes = _safe_float(row.get("aesthetics_norm"))
    coverage = _safe_float(row.get("face_coverage"))
    video_stem = str(row.get("video_stem", "") or "")

    pred_label_attr = pred_label or ""
    pred_conf_attr = f"{pred_conf:.6f}" if pred_conf is not None else ""
    aes_attr = f"{aes:.6f}" if aes is not None else ""
    coverage_attr = f"{coverage:.6f}" if coverage is not None else ""

    pred_str = (
        f"{pred_label} ({pred_conf:.2f})"
        if pred_label and pred_conf is not None
        else "-"
    )
    aes_str = f"{aes:.2f}" if aes is not None else "-"
    coverage_str = f"{round(coverage * 100)}%" if coverage is not None else "-"

    source_type = "image" if is_image_source else "video"
    badge_html = "" if is_image_source else VIDEO_BADGE_HTML

    return f"""<div class="photo-card"
     data-export-path="{html.escape(export_path)}"
     data-source-type="{source_type}"
     data-pred-label="{html.escape(pred_label_attr)}"
     data-pred-confidence="{pred_conf_attr}"
     data-aesthetic="{aes_attr}"
     data-coverage="{coverage_attr}"
     data-aspect="{aspect:.4f}"
     data-rotation="{rotation}"
     data-video-stem="{html.escape(video_stem)}"
     data-flagged="false">
  <button class="flag-btn" title="Toggle flag (space in lightbox)">Flag</button>
  {badge_html}
  <img src="{thumb_src}" loading="lazy" alt="">
  <div class="overlay">
    <div>pred: {html.escape(pred_str)}  |  aes: {aes_str}  |  coverage: {coverage_str}</div>
    <div class="stem">{html.escape(video_stem)}</div>
  </div>
</div>"""


def main() -> None:
    parser = ArgumentParser(
        description="Build a self-contained HTML photo viewer for refined frames.",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Run YAML config. When provided, --scores-csv, "
                             "--output-html and --parquet default to "
                             "{output_dir}/refined_scores.csv, "
                             "{output_dir}/index_photos.html and "
                             "{output_dir}/index.parquet. Explicit flags still override.")
    parser.add_argument("--scores-csv", type=Path, default=None,
                        help="Path to refined_scores.csv.")
    parser.add_argument("--output-html", type=Path, default=None,
                        help="Path to write the HTML file.")
    parser.add_argument("--parquet", type=Path, default=None,
                        help="Pass 1 index parquet (for video_path / frame_w / frame_h).")
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
        if args.scores_csv is None:
            args.scores_csv = cfg.output_dir / "refined_scores.csv"
        if args.output_html is None:
            args.output_html = cfg.output_dir / "index_photos.html"
        if args.parquet is None:
            args.parquet = cfg.output_dir / "index.parquet"
    if args.scores_csv is None or args.output_html is None or args.parquet is None:
        parser.error(
            "--scores-csv, --output-html, and --parquet are required when "
            "--config is not provided",
        )

    df = pd.read_csv(args.scores_csv)
    logger.info("Loaded %d rows from %s", len(df), args.scores_csv)

    idx = pd.read_parquet(
        args.parquet, columns=["frame_path", "video_path", "frame_w", "frame_h"],
    )
    logger.info("Loaded %d rows from %s", len(idx), args.parquet)

    # Replace any pre-existing copies of these columns with the parquet values.
    drop_cols = [c for c in ["video_path", "frame_w", "frame_h"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = df.merge(idx, on="frame_path", how="left")

    missing = df["video_path"].isna().sum()
    if missing:
        logger.warning("%d rows have no parquet match on frame_path", int(missing))

    face_h = df["face_y2"] - df["face_y1"]
    df["face_coverage"] = (
        (df["face_w"] * face_h) / (df["frame_w"] * df["frame_h"])
    ).clip(lower=0.0, upper=1.0)

    html_dir = args.output_html.parent
    html_dir.mkdir(parents=True, exist_ok=True)

    cards: list[str] = []
    skipped = 0
    image_source_count = 0
    for _, row in df.iterrows():
        video_path = row.get("video_path")
        if not isinstance(video_path, str) or not video_path or pd.isna(video_path):
            skipped += 1
            continue
        is_image_source = Path(video_path).suffix.lower() in IMAGE_EXTENSIONS

        refined = row.get("refined_frame_path")
        if not isinstance(refined, str) or not refined or pd.isna(refined):
            refined = row.get("frame_path", "")
        if not isinstance(refined, str) or not refined:
            skipped += 1
            continue
        refined_p = Path(refined)
        if not refined_p.exists():
            logger.warning("Thumbnail missing on disk: %s", refined_p)
            skipped += 1
            continue

        if is_image_source:
            export_path = _to_fwd_slash(video_path)
            image_source_count += 1
        else:
            export_path = _to_fwd_slash(refined_p.resolve())

        thumb_src = _make_img_src(refined_p, html_dir)

        # Image-source cards read EXIF orientation from the original file.
        # Video-source cards already have rotation baked into the JPEG by
        # pass1_index.py, so they always render at 0 degrees here.
        rotation = get_image_rotation_deg(video_path) if is_image_source else 0
        fw = _safe_float(row.get("frame_w"))
        fh = _safe_float(row.get("frame_h"))
        if fw and fh and fw > 0 and fh > 0:
            aspect = (fh / fw) if rotation in (90, 270) else (fw / fh)
        else:
            aspect = 1.0

        cards.append(_build_card(row, thumb_src, export_path, aspect, rotation, is_image_source))

    if skipped:
        logger.info("Skipped %d rows (missing video_path or thumbnail)", skipped)
    logger.info(
        "Built %d cards (%d image-source, %d video-source)",
        len(cards), image_source_count, len(cards) - image_source_count,
    )

    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1400">
<title>Still Extractor - Photos</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Still Extractor &mdash; Photos (<span id="shown-count">{len(cards)}</span> shown)</h1>
  <div class="toolbar">
    <span class="toolbar-label">Filter:</span>
    <button class="filter-btn" data-filter="all">All</button>
    <button class="filter-btn" data-filter="good">Good</button>
    <button class="filter-btn" data-filter="okay">Okay</button>
    <button class="filter-btn" data-filter="bad">Bad</button>
    <button class="filter-btn" data-filter="none">None</button>
    <span class="toolbar-sep">|</span>
    <span class="toolbar-label">Source:</span>
    <button class="source-filter-btn" data-source-filter="all">All</button>
    <button class="source-filter-btn" data-source-filter="video">Video</button>
    <button class="source-filter-btn" data-source-filter="image">Image</button>
    <span class="toolbar-sep">|</span>
    <span class="toolbar-label">Sort:</span>
    <button class="sort-btn" data-sort="confidence">Pred Confidence &darr;</button>
    <button class="sort-btn" data-sort="aesthetic">Aesthetic &darr;</button>
    <button class="sort-btn" data-sort="coverage">Coverage &darr;</button>
  </div>
  <div class="toolbar">
    <span class="toolbar-label">Flag:</span>
    <button id="flag-visible-btn">Flag Visible</button>
    <button id="clear-flags-btn">Clear All Flags</button>
    <span class="toolbar-sep">|</span>
    <button id="export-btn">Export Flagged (0)</button>
  </div>
  <div class="legend">
    Click image to open lightbox. In lightbox: <kbd>&larr;</kbd><kbd>&rarr;</kbd> navigate, <kbd>Space</kbd> flag, <kbd>Esc</kbd> close.
  </div>
</header>
<div class="grid">
{chr(10).join(cards)}
</div>
<div id="lightbox">
  <button id="lightbox-close">Close (Esc)</button>
  <img id="lightbox-img" src="" alt="">
  <div id="lightbox-meta"></div>
</div>
<script>{JS}</script>
</body>
</html>
"""

    args.output_html.write_text(body, encoding="utf-8")
    file_size_mb = args.output_html.stat().st_size / (1024 * 1024)
    logger.info(
        "Wrote %s (%d cards, %.2f MB)",
        args.output_html, len(cards), file_size_mb,
    )

    summary = {
        "stage": "build_photo_viewer",
        "config": str(args.config) if args.config is not None else None,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "card_count": len(cards),
        "video_source": len(cards) - image_source_count,
        "image_source": image_source_count,
        "output_html": str(args.output_html),
        "file_size_mb": round(file_size_mb, 2),
    }
    summary_path = args.output_html.parent / "build_photo_viewer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
