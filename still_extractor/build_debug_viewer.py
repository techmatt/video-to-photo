"""Build a self-contained HTML debug viewer with a fullscreen overlay (bbox, kps, scores)."""

import html
import json
import logging
import os
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from still_extractor.build_photo_viewer import (
    VIDEO_BADGE_HTML,
    _make_img_src,
    get_image_rotation_deg,
)
from still_extractor.constants import (
    IMAGE_EXTENSIONS,
    UPRIGHTER_CONFIDENCE_THRESHOLD,
    card_key,
)
from still_extractor.inventory import RunConfig
from still_extractor.utils import (
    parse_kps as _parse_kps,
    safe_float as _safe_float,
    to_fwd_slash as _to_fwd_slash,
)

logger = logging.getLogger(__name__)


IDENTITIES_DIR = Path("data/identities")


def _parse_embedding(val) -> np.ndarray | None:
    """Parse a JSON-encoded face embedding column to a numpy array, or None."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    try:
        arr = json.loads(val) if isinstance(val, str) else val
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(arr, (list, tuple)) or len(arr) == 0:
        return None
    try:
        out = np.asarray(arr, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if out.ndim != 1:
        return None
    return out


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v
    return v / norm


def _load_identities(index_path: Path) -> list[dict]:
    """Return identities with parsed centroids, or [] if file missing/invalid."""
    if not index_path.exists():
        return []
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not parse %s: %s", index_path, e)
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        centroid_raw = entry.get("centroid")
        if not isinstance(name, str) or not isinstance(centroid_raw, list):
            continue
        try:
            centroid = _l2_normalize(np.asarray(centroid_raw, dtype=np.float32))
        except (TypeError, ValueError):
            continue
        if centroid.ndim != 1 or centroid.shape[0] == 0:
            continue
        display = entry.get("display_name")
        portrait = entry.get("portrait_path")
        out.append({
            "name": name,
            "display_name": display if isinstance(display, str) and display else name,
            "centroid": centroid,
            "portrait_path": (
                portrait if isinstance(portrait, str) and portrait
                else f"data/identities/{name}.png"
            ),
        })
    return out


def _load_cluster_membership(clusters_path: Path) -> dict[str, set[str]]:
    """Return a map identity-name -> set of card_keys. Empty dict if missing."""
    if not clusters_path.exists():
        return {}
    try:
        raw = json.loads(clusters_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not parse %s: %s", clusters_path, e)
        return {}
    out: dict[str, set[str]] = {}
    for cluster in raw.get("clusters", []) if isinstance(raw, dict) else []:
        if not isinstance(cluster, dict):
            continue
        name = cluster.get("identity")
        frame_ids = cluster.get("frame_ids", [])
        if not isinstance(name, str) or not isinstance(frame_ids, list):
            continue
        out[name] = {fid for fid in frame_ids if isinstance(fid, str)}
    return out


def _portrait_relpath(portrait_path: str, html_dir: Path) -> str | None:
    """Return relpath from html_dir to the given portrait_path (project-relative), or None."""
    portrait = Path(portrait_path).resolve()
    if not portrait.exists():
        return None
    try:
        rel = os.path.relpath(portrait, html_dir.resolve())
    except ValueError:
        return None
    return _to_fwd_slash(rel)


def _nearest_identity(
    emb: np.ndarray, identities: list[dict],
) -> tuple[int, float] | None:
    """Return (identity_index, cosine_distance) for the nearest centroid, or None."""
    if not identities or emb.size == 0:
        return None
    emb_n = _l2_normalize(emb)
    centroids = np.stack([ident["centroid"] for ident in identities], axis=0)
    if centroids.shape[1] != emb_n.shape[0]:
        return None
    sims = centroids @ emb_n
    best = int(np.argmax(sims))
    dist = float(1.0 - sims[best])
    return best, dist


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

/* ---------- Debug overlay ---------- */

#overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.92);
  z-index: 1000;
}
#overlay.open { display: block; }
#overlay-inner {
  display: flex;
  flex-direction: row;
  height: 100vh;
  width: 100vw;
}
#overlay-left {
  flex: 0 0 65%;
  position: relative;
  overflow: hidden;
}
#overlay-img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #000;
}
#overlay-canvas {
  position: absolute;
  left: 0;
  top: 0;
  pointer-events: none;
}
#overlay-right {
  flex: 0 0 35%;
  background: #15171c;
  border-left: 1px solid #2a2d36;
  padding: 1.5rem;
  overflow-y: auto;
  color: #ddd;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: 13px;
  line-height: 1.55;
}
#overlay-scores .section { margin-bottom: 1.2rem; }
#overlay-scores .section:last-child { margin-bottom: 0; }
#overlay-scores .label {
  color: #6b7280;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 4px;
}
#overlay-scores .value { color: #f3f4f6; }
#overlay-scores .big {
  font-size: 24px;
  font-weight: 600;
  color: #f9fafb;
}
#overlay-scores .small { color: #9ca3af; font-size: 12px; }
#overlay-scores .source-name { color: #f3f4f6; word-break: break-all; }

.toggle-rejected-btn {
  background: #222;
  color: #ddd;
  border: 1px solid #444;
  padding: 1px 6px;
  margin-left: 6px;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 11px;
}
.toggle-rejected-btn:hover { background: #2a2a2a; }
.toggle-rejected-btn.active {
  background: rgba(239, 68, 68, 0.25);
  border-color: rgba(239, 68, 68, 0.7);
  color: #fff;
}

.bar-row {
  display: grid;
  grid-template-columns: 56px 1fr 56px;
  align-items: center;
  gap: 8px;
  margin: 3px 0;
  color: #cbd5e1;
}
.bar-row.winner { color: #f9fafb; font-weight: 600; }
.bar-row .name { font-size: 12px; }
.bar-row .num { font-size: 12px; text-align: right; color: #d1d5db; font-variant-numeric: tabular-nums; }
.bar-row .track {
  position: relative;
  height: 12px;
  background: #1f2330;
  border-radius: 3px;
  overflow: hidden;
}
.bar-row .fill {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  border-radius: 3px;
}

#overlay-close, #overlay-nav-prev, #overlay-nav-next {
  position: absolute;
  background: rgba(20, 20, 20, 0.7);
  color: #eee;
  border: 1px solid #444;
  border-radius: 6px;
  cursor: pointer;
  z-index: 2;
  font-size: 18px;
  line-height: 1;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}
#overlay-close:hover, #overlay-nav-prev:hover, #overlay-nav-next:hover {
  background: rgba(40, 40, 40, 0.95);
}
#overlay-close {
  top: 14px;
  right: 14px;
  width: 36px;
  height: 36px;
  font-size: 20px;
}
#overlay-nav-prev, #overlay-nav-next {
  top: 50%;
  transform: translateY(-50%);
  width: 44px;
  height: 64px;
  font-size: 26px;
}
#overlay-nav-prev { left: 14px; }
#overlay-nav-next { right: calc(35% + 14px); }

/* ---------- Face table ---------- */

.face-table {
  display: grid;
  grid-template-columns: 18px 56px 38px 22px 1fr 38px 56px;
  align-items: center;
  gap: 4px 6px;
  font-size: 12px;
  margin-top: 6px;
  font-variant-numeric: tabular-nums;
}
.face-table .ft-head {
  color: #6b7280;
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.face-row {
  display: contents;
  cursor: pointer;
}
.face-row.highlighted .ft-cell { background: rgba(74, 127, 208, 0.18); }
.face-row .ft-cell {
  padding: 3px 2px;
}
.face-row .ft-num { color: #d1d5db; }
.face-row .ft-quality { font-weight: 600; }
.face-row .ft-qscore { color: #9ca3af; text-align: right; }
.face-row .ft-portrait img {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  object-fit: cover;
  display: block;
}
.face-row .ft-identity { color: #f3f4f6; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.face-row .ft-identity.nearest-only { color: #94a3b8; font-style: italic; }
.face-row .ft-iscore { color: #cbd5e1; text-align: right; }
.face-row .ft-confbar {
  position: relative;
  height: 9px;
  background: #1f2330;
  border-radius: 2px;
  overflow: hidden;
}
.face-row .ft-confbar .seg {
  position: absolute;
  top: 0;
  bottom: 0;
  background: #4ade80;
  opacity: 0.85;
}
.face-row .ft-confbar.nearest-only .seg { background: #6b7280; }

/* ---------- Bbox hover tooltip ---------- */

#bbox-tooltip {
  position: absolute;
  pointer-events: none;
  background: rgba(15, 18, 24, 0.95);
  border: 1px solid #2a2d36;
  border-radius: 5px;
  padding: 6px 9px;
  font: 12px/1.4 ui-monospace, Menlo, Consolas, monospace;
  color: #f3f4f6;
  z-index: 5;
  max-width: 320px;
  display: none;
  white-space: nowrap;
}
#bbox-tooltip .tt-title { color: #f9fafb; font-weight: 600; margin-bottom: 2px; }
#bbox-tooltip .tt-row { color: #cbd5e1; }
#bbox-tooltip .tt-row.muted { color: #94a3b8; font-style: italic; }
#bbox-tooltip .tt-row.warn { color: #f97316; }
"""


JS_TEMPLATE = """
const FILTER_KEY = 'photoViewer.filter';
const SOURCE_FILTER_KEY = 'photoViewer.sourceFilter';
const SORT_KEY = 'photoViewer.sort';
const FLAG_PREFIX = 'flag:';

const FRAMES_DATA = __FRAMES_DATA__;
const IDENTITY_PORTRAITS = __IDENTITY_PORTRAITS__;

const LABEL_COLORS = {
  good: '#4ade80',
  okay: '#facc15',
  bad:  '#f97316',
  none: '#ef4444',
};

let currentFilter = 'good';
let currentSourceFilter = 'all';
let currentSort = 'confidence';
let overlayIndex = -1;
let lastLayoutContainerWidth = -1;
let showRejected = false;
let highlightedFace = -1;
let hoveredAcceptedFace = -1;
let hoveredRejectedFace = -1;
const SHOW_REJECTED_KEY = 'photoViewer.showRejected';

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
  const layout = [];
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
    const widths = row.map(c => Math.round(c.aspectRatio * targetRowHeight));
    layout.push({ row, widths, height: Math.round(targetRowHeight) });
  }

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

// ---------- Debug overlay ----------

function fmtNum(v, digits) {
  if (v == null || isNaN(v)) return '-';
  return Number(v).toFixed(digits);
}

function escHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function buildClassifierBars(frame) {
  const face1 = (frame.faces && frame.faces[0]) || null;
  if (!face1) return '';
  const order = ['good', 'okay', 'bad', 'none'];
  const probs = { good: face1.p_good, okay: face1.p_okay, bad: face1.p_bad, none: face1.p_none };
  const winner = face1.pred_label;
  let out = '';
  for (const label of order) {
    const p = probs[label];
    if (p == null) continue;
    const pct = Math.max(0, Math.min(100, p * 100));
    const color = LABEL_COLORS[label] || '#888';
    const cls = (label === winner) ? 'bar-row winner' : 'bar-row';
    out += `<div class="${cls}">`
        +  `<div class="name" style="color:${color}">${label}</div>`
        +  `<div class="track"><div class="fill" style="width:${pct.toFixed(1)}%;background:${color};"></div></div>`
        +  `<div class="num">${fmtNum(p, 2)}</div>`
        +  `</div>`;
  }
  return out;
}

function buildConfBar(conf, nearestOnly) {
  const segs = 10;
  const filled = Math.max(0, Math.min(segs, Math.round(conf * segs)));
  let html = '';
  for (let i = 0; i < segs; i++) {
    if (i >= filled) continue;
    const left = (i * (100 / segs)).toFixed(1);
    const width = (100 / segs - 1).toFixed(2);
    html += `<div class="seg" style="left:${left}%;width:${width}%;"></div>`;
  }
  const cls = nearestOnly ? 'ft-confbar nearest-only' : 'ft-confbar';
  return `<div class="${cls}">${html}</div>`;
}

function buildFaceTable(frame) {
  const faces = Array.isArray(frame.faces) ? frame.faces : [];
  const identities = Array.isArray(frame.face_identities) ? frame.face_identities : [];
  const anyFace = faces.some(f => f && f.x1 != null);
  if (!anyFace) return '';
  const hasIdentities = identities.some(i => i);

  let head = `<div class="face-table">`
    + `<div class="ft-head">#</div>`
    + `<div class="ft-head">Quality</div>`
    + `<div class="ft-head"></div>`;
  if (hasIdentities) {
    head += `<div class="ft-head"></div>`
      +    `<div class="ft-head">Identity</div>`
      +    `<div class="ft-head"></div>`
      +    `<div class="ft-head"></div>`;
  } else {
    head += `<div class="ft-head" style="grid-column: span 4;"></div>`;
  }

  let rows = '';
  for (let i = 0; i < faces.length; i++) {
    const face = faces[i];
    if (!face || face.x1 == null) continue;
    const label = face.pred_label || 'none';
    const color = LABEL_COLORS[label] || '#888';
    const conf = face.pred_confidence;
    const highlighted = (highlightedFace === i) ? ' highlighted' : '';
    const ident = identities[i] || null;

    let identCells;
    if (!hasIdentities) {
      identCells = `<div class="ft-cell" style="grid-column: span 4;"></div>`;
    } else if (ident) {
      const portraitUrl = IDENTITY_PORTRAITS[ident.identity];
      const portraitImg = portraitUrl
        ? `<img src="${escHtml(portraitUrl)}" alt="" onerror="this.style.display='none'">`
        : '';
      const nearestOnly = !ident.assigned;
      const cls = nearestOnly ? 'ft-identity nearest-only' : 'ft-identity';
      const suffix = nearestOnly ? ' (nearest)' : '';
      const displayName = ident.display_name || ident.identity;
      identCells =
        `<div class="ft-cell ft-portrait">${portraitImg}</div>`
        + `<div class="ft-cell ${cls}" title="${escHtml(displayName)}${suffix}">`
        +   `${escHtml(displayName)}${suffix}`
        + `</div>`
        + `<div class="ft-cell ft-iscore">${fmtNum(ident.confidence, 2)}</div>`
        + `<div class="ft-cell">${buildConfBar(ident.confidence, nearestOnly)}</div>`;
    } else {
      identCells =
        `<div class="ft-cell"></div>`
        + `<div class="ft-cell" style="color:#6b7280">-</div>`
        + `<div class="ft-cell"></div>`
        + `<div class="ft-cell"></div>`;
    }

    rows += `<div class="face-row${highlighted}" data-face-idx="${i}">`
      + `<div class="ft-cell ft-num">${i + 1}</div>`
      + `<div class="ft-cell ft-quality" style="color:${color}">${label}</div>`
      + `<div class="ft-cell ft-qscore">${fmtNum(conf, 2)}</div>`
      + identCells
      + `</div>`;
  }
  return head + rows + `</div>`;
}

function currentFrame() {
  const card = visibleCards()[overlayIndex];
  if (!card) return null;
  const frameIdx = parseInt(card.dataset.frameIdx, 10);
  if (isNaN(frameIdx)) return null;
  return FRAMES_DATA[frameIdx] || null;
}

function hideTooltip() {
  const tt = document.getElementById('bbox-tooltip');
  if (tt) tt.style.display = 'none';
}

function showTooltip(html, clientX, clientY) {
  const tt = document.getElementById('bbox-tooltip');
  const left = document.getElementById('overlay-left');
  if (!tt || !left) return;
  const rect = left.getBoundingClientRect();
  tt.innerHTML = html;
  tt.style.display = 'block';
  const ttRect = tt.getBoundingClientRect();
  let x = clientX - rect.left + 14;
  let y = clientY - rect.top + 14;
  if (x + ttRect.width > rect.width) x = clientX - rect.left - ttRect.width - 14;
  if (y + ttRect.height > rect.height) y = clientY - rect.top - ttRect.height - 14;
  tt.style.left = Math.max(0, x) + 'px';
  tt.style.top = Math.max(0, y) + 'px';
}

function buildAcceptedTooltipHtml(face, ident, idx) {
  const label = face.pred_label || 'none';
  const conf = (face.pred_confidence != null) ? fmtNum(face.pred_confidence, 2) : '-';
  const color = LABEL_COLORS[label] || '#888';
  let h = `<div class="tt-title">Face ${idx + 1}: <span style="color:${color}">${escHtml(label)}</span> ${conf}</div>`;
  if (ident) {
    const name = escHtml(ident.display_name || ident.identity);
    const c = fmtNum(ident.confidence, 2);
    if (ident.assigned) {
      h += `<div class="tt-row">identity: ${name} ${c}</div>`;
    } else {
      h += `<div class="tt-row muted">nearest: ${name} ${c}</div>`;
    }
  }
  if (face.kps_anomalous) {
    h += `<div class="tt-row warn">kps anomalous</div>`;
  }
  return h;
}

function buildRejectedTooltipHtml(rej, idx) {
  return `<div class="tt-title">Rejected face</div>`
    + `<div class="tt-row warn">${escHtml(rej.reason || 'rejected')}</div>`;
}

function imageDisplayRect() {
  const img = document.getElementById('overlay-img');
  const left = document.getElementById('overlay-left');
  if (!img || !left) return null;
  const containerW = img.clientWidth;
  const containerH = img.clientHeight;
  const natW = img.naturalWidth;
  const natH = img.naturalHeight;
  if (!natW || !natH || !containerW || !containerH) return null;
  const scale = Math.min(containerW / natW, containerH / natH);
  const dispW = natW * scale;
  const dispH = natH * scale;
  const offsetX = (containerW - dispW) / 2;
  const offsetY = (containerH - dispH) / 2;
  return { dispW, dispH, offsetX, offsetY, natW, natH, scale };
}

function hitTestFace(clientX, clientY) {
  const frame = currentFrame();
  if (!frame) return { accepted: -1, rejected: -1 };
  const img = document.getElementById('overlay-img');
  if (!img) return { accepted: -1, rejected: -1 };
  const r = img.getBoundingClientRect();
  const rect = imageDisplayRect();
  if (!rect) return { accepted: -1, rejected: -1 };
  const lx = clientX - r.left - rect.offsetX;
  const ly = clientY - r.top - rect.offsetY;
  if (lx < 0 || ly < 0 || lx > rect.dispW || ly > rect.dispH) {
    return { accepted: -1, rejected: -1 };
  }
  const fw = frame.frame_w || rect.natW;
  const fh = frame.frame_h || rect.natH;
  const ix = lx * (fw / rect.dispW);
  const iy = ly * (fh / rect.dispH);

  let accepted = -1;
  const faces = Array.isArray(frame.faces) ? frame.faces : [];
  for (let i = 0; i < faces.length; i++) {
    const f = faces[i];
    if (!f || f.x1 == null) continue;
    if (ix >= f.x1 && ix <= f.x2 && iy >= f.y1 && iy <= f.y2) { accepted = i; break; }
  }
  let rejected = -1;
  if (showRejected && accepted < 0) {
    const rs = Array.isArray(frame.rejected_faces) ? frame.rejected_faces : [];
    for (let i = 0; i < rs.length; i++) {
      const r2 = rs[i];
      if (r2.x1 == null) continue;
      if (ix >= r2.x1 && ix <= r2.x2 && iy >= r2.y1 && iy <= r2.y2) { rejected = i; break; }
    }
  }
  return { accepted, rejected };
}

function onOverlayMouseMove(e) {
  if (overlayIndex < 0) return;
  const frame = currentFrame();
  if (!frame) { hideTooltip(); return; }
  const { accepted, rejected } = hitTestFace(e.clientX, e.clientY);
  if (accepted >= 0) {
    const face = frame.faces[accepted];
    const ident = (frame.face_identities && frame.face_identities[accepted]) || null;
    showTooltip(buildAcceptedTooltipHtml(face, ident, accepted), e.clientX, e.clientY);
  } else if (rejected >= 0) {
    const rej = frame.rejected_faces[rejected];
    showTooltip(buildRejectedTooltipHtml(rej, rejected), e.clientX, e.clientY);
  } else {
    hideTooltip();
  }
  if (hoveredAcceptedFace !== accepted || hoveredRejectedFace !== rejected) {
    hoveredAcceptedFace = accepted;
    hoveredRejectedFace = rejected;
    drawOverlayAnnotations();
  }
}

function onOverlayMouseLeave() {
  hoveredAcceptedFace = -1;
  hoveredRejectedFace = -1;
  hideTooltip();
  drawOverlayAnnotations();
}

function buildScoresHtml(frame) {
  const ts = frame.refined_timestamp_s != null ? frame.refined_timestamp_s : frame.timestamp_s;
  const tsRaw = frame.timestamp_s;
  const tsRef = frame.refined_timestamp_s;
  const refinedDifferent = (tsRaw != null && tsRef != null && Math.abs(tsRaw - tsRef) > 1e-6);

  const isVideo = frame.source_type === 'video';
  const srcName = frame.video_basename || '';
  let sourceLine = `<div class="source-name">${escHtml(srcName)}`;
  if (isVideo && ts != null) sourceLine += ` @ ${fmtNum(ts, 3)}s`;
  sourceLine += `</div>`;
  let refinedLine = '';
  if (isVideo && refinedDifferent) {
    const delta = tsRef - tsRaw;
    const sign = delta >= 0 ? '+' : '';
    refinedLine = `<div class="small">(refined from ${fmtNum(tsRaw, 3)}s, &Delta;${sign}${fmtNum(delta, 3)}s)</div>`;
  }

  const composite = fmtNum(frame.composite, 3);
  const classifierBars = buildClassifierBars(frame);

  const sharpCenter = fmtNum(frame.sharpness_center, 1);
  const sharpRef = fmtNum(frame.refined_sharpness, 1);
  let sharpDelta = frame.sharpness_delta;
  if (sharpDelta == null && frame.refined_sharpness != null && frame.sharpness_center != null) {
    sharpDelta = frame.refined_sharpness - frame.sharpness_center;
  }
  const sharpDeltaSign = (sharpDelta != null && sharpDelta >= 0) ? '+' : '';
  const sharpDeltaStr = sharpDelta == null ? '-' : `${sharpDeltaSign}${fmtNum(sharpDelta, 1)}`;

  const aes = fmtNum(frame.aesthetics_norm, 2);

  let upLine = '';
  if (frame.uprighter_pred) {
    const map = { '90cw': '90&deg; CW', '180': '180&deg;', '270cw': '270&deg; CW' };
    const display = map[frame.uprighter_pred] || frame.uprighter_pred;
    const conf = fmtNum(frame.uprighter_confidence, 2);
    upLine =
      `<div class="section">`
      + `<div class="label">Uprighter</div>`
      + `<div class="value">${display} <span class="small">(conf ${conf})</span></div>`
      + `</div>`;
  }

  const faceCount = (frame.face_count != null) ? frame.face_count : '-';
  const bestPair = frame.best_pair_score;
  const rejectedCount = (frame.rejected_face_count != null) ? frame.rejected_face_count : 0;
  let facesSection =
      `<div class="section">`
      + `<div class="label">Faces Detected</div>`
      + `<div class="value">${faceCount}</div>`;
  const toggleLabel = showRejected ? 'Hide Rejected' : 'Show Rejected';
  const toggleClass = showRejected ? 'toggle-rejected-btn active' : 'toggle-rejected-btn';
  facesSection +=
      `<div class="small">rejected: ${rejectedCount} `
      + `<button class="${toggleClass}" id="toggle-rejected-btn">${toggleLabel}</button>`
      + `</div>`;
  if (bestPair != null) {
    facesSection += `<div class="small">best pair score: ${fmtNum(bestPair, 2)}</div>`;
  }
  facesSection += buildFaceTable(frame);
  facesSection += `</div>`;

  return ''
    + `<div class="section">`
    +   `<div class="label">Source</div>`
    +   sourceLine
    +   refinedLine
    + `</div>`
    + `<div class="section">`
    +   `<div class="label">Composite</div>`
    +   `<div class="big">${composite}</div>`
    + `</div>`
    + facesSection
    + `<div class="section">`
    +   `<div class="label">Classifier (face 1)</div>`
    +   classifierBars
    + `</div>`
    + `<div class="section">`
    +   `<div class="label">Sharpness</div>`
    +   `<div class="value">center: ${sharpCenter} &rarr; refined: ${sharpRef}</div>`
    +   `<div class="small">delta: ${sharpDeltaStr}</div>`
    + `</div>`
    + `<div class="section">`
    +   `<div class="label">Aesthetics</div>`
    +   `<div class="value">${aes}</div>`
    + `</div>`
    + upLine;
}

function drawOverlayAnnotations() {
  const img = document.getElementById('overlay-img');
  const canvas = document.getElementById('overlay-canvas');
  if (!img || !canvas) return;
  const card = visibleCards()[overlayIndex];
  if (!card) return;
  const frameIdx = parseInt(card.dataset.frameIdx, 10);
  if (isNaN(frameIdx)) return;
  const frame = FRAMES_DATA[frameIdx];
  if (!frame) return;

  const containerW = img.clientWidth;
  const containerH = img.clientHeight;
  const natW = img.naturalWidth || frame.frame_w;
  const natH = img.naturalHeight || frame.frame_h;
  if (!natW || !natH || !containerW || !containerH) return;

  const scale = Math.min(containerW / natW, containerH / natH);
  const dispW = natW * scale;
  const dispH = natH * scale;
  const offsetX = (containerW - dispW) / 2;
  const offsetY = (containerH - dispH) / 2;

  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = dispW + 'px';
  canvas.style.height = dispH + 'px';
  canvas.style.left = offsetX + 'px';
  canvas.style.top = offsetY + 'px';
  canvas.width = Math.round(dispW * dpr);
  canvas.height = Math.round(dispH * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, dispW, dispH);

  const faces = Array.isArray(frame.faces) ? frame.faces : [];
  if (faces.length === 0 || !faces[0]) return;
  const fw = frame.frame_w;
  const fh = frame.frame_h;
  if (!fw || !fh) return;
  const sx = dispW / fw;
  const sy = dispH / fh;

  const KPS_COLORS = ['#60a5fa', '#60a5fa', '#4ade80', '#f87171', '#f87171'];

  for (let slot = 0; slot < faces.length; slot++) {
    const face = faces[slot];
    if (!face || face.x1 == null) continue;
    const isPrimary = (slot === 0);
    const isEmphasized = (slot === highlightedFace) || (slot === hoveredAcceptedFace);
    const alpha = isEmphasized ? 1.0 : (isPrimary ? 1.0 : 0.7);
    const label = face.pred_label || 'none';
    const color = LABEL_COLORS[label] || '#888';

    const x1 = face.x1 * sx;
    const y1 = face.y1 * sy;
    const x2 = face.x2 * sx;
    const y2 = face.y2 * sy;

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.lineWidth = isEmphasized ? 4 : (isPrimary ? 3 : 2);
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    if (isPrimary) {
      const conf = (face.pred_confidence != null) ? face.pred_confidence.toFixed(2) : '-';
      const text = `${label} ${conf}`;
      ctx.font = '600 14px ui-monospace, Menlo, Consolas, monospace';
      const padX = 6;
      const textW = ctx.measureText(text).width;
      const pillH = 20;
      const pillW = textW + padX * 2;
      let pillX = x1;
      let pillY = y1 - pillH - 4;
      if (pillY < 0) pillY = y1 + 4;
      ctx.fillStyle = 'rgba(0,0,0,0.65)';
      ctx.beginPath();
      const r = 4;
      ctx.moveTo(pillX + r, pillY);
      ctx.lineTo(pillX + pillW - r, pillY);
      ctx.quadraticCurveTo(pillX + pillW, pillY, pillX + pillW, pillY + r);
      ctx.lineTo(pillX + pillW, pillY + pillH - r);
      ctx.quadraticCurveTo(pillX + pillW, pillY + pillH, pillX + pillW - r, pillY + pillH);
      ctx.lineTo(pillX + r, pillY + pillH);
      ctx.quadraticCurveTo(pillX, pillY + pillH, pillX, pillY + pillH - r);
      ctx.lineTo(pillX, pillY + r);
      ctx.quadraticCurveTo(pillX, pillY, pillX + r, pillY);
      ctx.closePath();
      ctx.fill();
      ctx.fillStyle = color;
      ctx.textBaseline = 'middle';
      ctx.fillText(text, pillX + padX, pillY + pillH / 2);
    }

    if (face.kps && face.kps.length === 5) {
      for (let i = 0; i < 5; i++) {
        const [kx, ky] = face.kps[i];
        ctx.fillStyle = KPS_COLORS[i];
        ctx.beginPath();
        ctx.arc(kx * sx, ky * sy, isPrimary ? 5 : 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  }

  if (showRejected) {
    const rejected = Array.isArray(frame.rejected_faces) ? frame.rejected_faces : [];
    for (let ri = 0; ri < rejected.length; ri++) {
      const r = rejected[ri];
      if (r.x1 == null || r.x2 == null || r.y1 == null || r.y2 == null) continue;
      const x1 = r.x1 * sx;
      const y1 = r.y1 * sy;
      const x2 = r.x2 * sx;
      const y2 = r.y2 * sy;
      const isHovered = (ri === hoveredRejectedFace);
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = isHovered ? 'rgba(239, 68, 68, 1.0)' : 'rgba(239, 68, 68, 0.5)';
      ctx.lineWidth = isHovered ? 3 : 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(239, 68, 68, 0.85)';
      ctx.font = '500 11px ui-monospace, Menlo, Consolas, monospace';
      ctx.textBaseline = 'bottom';
      const labelY = y1 - 2 >= 12 ? y1 - 2 : y2 + 12;
      ctx.fillText(r.reason || 'rejected', x1, labelY);
      ctx.restore();
    }
  }
}

function openOverlay(idx) {
  const cards = visibleCards();
  if (idx < 0 || idx >= cards.length) return;
  overlayIndex = idx;
  highlightedFace = -1;
  hoveredAcceptedFace = -1;
  hoveredRejectedFace = -1;
  hideTooltip();
  const card = cards[idx];
  const frameIdx = parseInt(card.dataset.frameIdx, 10);
  const frame = FRAMES_DATA[frameIdx];
  if (!frame) return;

  const img = document.getElementById('overlay-img');
  const cardImg = card.querySelector('img');
  const canvas = document.getElementById('overlay-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  img.onload = drawOverlayAnnotations;
  img.src = cardImg.src;
  if (img.complete && img.naturalWidth > 0) {
    drawOverlayAnnotations();
  }

  renderOverlayScores(frame);
  document.getElementById('overlay').classList.add('open');
}

function renderOverlayScores(frame) {
  document.getElementById('overlay-scores').innerHTML = buildScoresHtml(frame);
  const toggleBtn = document.getElementById('toggle-rejected-btn');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', e => {
      e.stopPropagation();
      showRejected = !showRejected;
      localStorage.setItem(SHOW_REJECTED_KEY, showRejected ? '1' : '0');
      renderOverlayScores(frame);
      drawOverlayAnnotations();
    });
  }
  document.querySelectorAll('.face-row').forEach(row => {
    row.addEventListener('click', e => {
      e.stopPropagation();
      const idx = parseInt(row.dataset.faceIdx, 10);
      if (isNaN(idx)) return;
      highlightedFace = (highlightedFace === idx) ? -1 : idx;
      renderOverlayScores(frame);
      drawOverlayAnnotations();
    });
  });
}

function closeOverlay() {
  document.getElementById('overlay').classList.remove('open');
  document.getElementById('overlay-img').src = '';
  const canvas = document.getElementById('overlay-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  overlayIndex = -1;
  highlightedFace = -1;
  hoveredAcceptedFace = -1;
  hoveredRejectedFace = -1;
  hideTooltip();
}

function overlayNav(dir) {
  if (overlayIndex < 0) return;
  const cards = visibleCards();
  const next = overlayIndex + dir;
  if (next < 0 || next >= cards.length) return;
  openOverlay(next);
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
  showRejected = localStorage.getItem(SHOW_REJECTED_KEY) === '1';
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
    resizeTimer = setTimeout(() => {
      relayout(false);
      if (overlayIndex >= 0) drawOverlayAnnotations();
    }, 250);
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
    const open = e => {
      e.stopPropagation();
      const cards = visibleCards();
      const i = cards.indexOf(card);
      if (i >= 0) openOverlay(i);
    };
    card.querySelector('img').addEventListener('click', open);
    card.addEventListener('click', e => {
      if (e.target.closest('.flag-btn')) return;
      open(e);
    });
  });

  document.getElementById('overlay').addEventListener('click', e => {
    if (e.target.id === 'overlay') closeOverlay();
  });
  document.getElementById('overlay-close').addEventListener('click', closeOverlay);
  document.getElementById('overlay-nav-prev').addEventListener('click', () => overlayNav(-1));
  document.getElementById('overlay-nav-next').addEventListener('click', () => overlayNav(1));

  const overlayLeft = document.getElementById('overlay-left');
  if (overlayLeft) {
    overlayLeft.addEventListener('mousemove', onOverlayMouseMove);
    overlayLeft.addEventListener('mouseleave', onOverlayMouseLeave);
  }

  document.addEventListener('keydown', e => {
    if (overlayIndex >= 0) {
      if (e.key === 'Escape') { closeOverlay(); e.preventDefault(); }
      else if (e.key === 'ArrowLeft') { overlayNav(-1); e.preventDefault(); }
      else if (e.key === 'ArrowRight') { overlayNav(1); e.preventDefault(); }
      else if (e.key === ' ') {
        const card = visibleCards()[overlayIndex];
        if (card) { toggleFlag(card); }
        e.preventDefault();
      }
    }
  });
});
"""


_UPRIGHTER_LABEL_MAP = {90: "90cw", 180: "180", 270: "270cw"}


def _opt_int(val) -> int | None:
    try:
        i = int(val)
    except (TypeError, ValueError):
        return None
    return i


def _face_slot_payload(row: pd.Series, slot: int) -> dict | None:
    """Return per-slot face dict for the debug JSON, or None if slot is empty."""
    x1 = _opt_int(row.get(f"face_{slot}_x1"))
    if x1 is None:
        return None
    pred_raw = row.get(f"face_{slot}_pred_label")
    pred_label = (
        pred_raw
        if isinstance(pred_raw, str) and pred_raw and not pd.isna(pred_raw)
        else None
    )
    return {
        "x1": x1,
        "y1": _opt_int(row.get(f"face_{slot}_y1")),
        "x2": _opt_int(row.get(f"face_{slot}_x2")),
        "y2": _opt_int(row.get(f"face_{slot}_y2")),
        "det_score": _safe_float(row.get(f"face_{slot}_det_score")),
        "kps": _parse_kps(row.get(f"face_{slot}_kps")),
        "kps_anomalous": _opt_bool(row.get(f"face_{slot}_kps_anomalous")),
        "p_none": _safe_float(row.get(f"face_{slot}_p_none")),
        "p_bad":  _safe_float(row.get(f"face_{slot}_p_bad")),
        "p_okay": _safe_float(row.get(f"face_{slot}_p_okay")),
        "p_good": _safe_float(row.get(f"face_{slot}_p_good")),
        "pred_label": pred_label,
        "pred_confidence": _safe_float(row.get(f"face_{slot}_pred_confidence")),
    }


def _opt_bool(val) -> bool | None:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    return bool(val)


def _face_identity_payload(
    row: pd.Series,
    slot: int,
    identities: list[dict],
    membership: dict[str, set[str]],
    has_embedding_cols: bool,
) -> dict | None:
    """Compute the per-face identity entry for face slot, or None."""
    if not identities or not has_embedding_cols:
        return None
    if _opt_int(row.get(f"face_{slot}_x1")) is None:
        return None
    emb = _parse_embedding(row.get(f"face_{slot}_embedding"))
    if emb is None:
        return None
    nearest = _nearest_identity(emb, identities)
    if nearest is None:
        return None
    idx, dist = nearest
    ident = identities[idx]
    confidence = max(0.0, min(1.0, 1.0 - dist))
    stem = row.get("video_stem")
    kept = row.get("kept_path")
    assigned = False
    if (
        isinstance(stem, str) and stem
        and isinstance(kept, str) and kept
    ):
        ck = card_key(stem, kept)
        assigned = ck in membership.get(ident["name"], set())
    return {
        "identity": ident["name"],
        "display_name": ident["display_name"],
        "confidence": float(confidence),
        "distance": float(dist),
        "assigned": bool(assigned),
    }


def _parse_rejected_faces(value) -> list[dict]:
    """Parse rejected_faces_json column into a list of {x1,y1,x2,y2,reason} dicts."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if not isinstance(value, str) or not value:
        return []
    try:
        data = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        out.append({
            "x1": _opt_int(d.get("x1")),
            "y1": _opt_int(d.get("y1")),
            "x2": _opt_int(d.get("x2")),
            "y2": _opt_int(d.get("y2")),
            "reason": str(d.get("reason", "")),
        })
    return out


def _build_frame_json(
    row: pd.Series,
    kept_path: Path,
    is_image_source: bool,
    identities: list[dict],
    membership: dict[str, set[str]],
    has_embedding_cols: bool,
) -> dict:
    """Per-frame debug payload baked into the page as JSON."""
    f = _safe_float

    up_deg = _opt_int(row.get("uprighter_pred")) or 0
    up_conf = f(row.get("uprighter_confidence")) or 0.0
    up_str: str | None = None
    if up_deg in _UPRIGHTER_LABEL_MAP and up_conf >= UPRIGHTER_CONFIDENCE_THRESHOLD:
        up_str = _UPRIGHTER_LABEL_MAP[up_deg]

    video_path = row.get("video_path") or ""
    video_basename = Path(video_path).name if video_path else ""

    faces = [_face_slot_payload(row, slot) for slot in (1, 2, 3)]
    face_identities = [
        _face_identity_payload(row, slot, identities, membership, has_embedding_cols)
        for slot in (1, 2, 3)
    ]
    face_count = _opt_int(row.get("face_count"))
    best_pair_score = f(row.get("best_pair_score"))
    rejected_faces = _parse_rejected_faces(row.get("rejected_faces_json"))
    rejected_face_count = _opt_int(row.get("rejected_face_count")) or 0

    return {
        "video_basename": video_basename,
        "video_stem": row.get("video_stem") or "",
        "kept_basename": kept_path.name,
        "source_type": "image" if is_image_source else "video",
        "frame_w": _opt_int(row.get("frame_w")),
        "frame_h": _opt_int(row.get("frame_h")),
        "faces": faces,
        "face_identities": face_identities,
        "face_count": face_count,
        "best_pair_score": best_pair_score,
        "rejected_faces": rejected_faces,
        "rejected_face_count": rejected_face_count,
        "sharpness_center": f(row.get("sharpness_center")),
        "refined_sharpness": f(row.get("refined_sharpness")),
        "sharpness_delta": f(row.get("sharpness_delta")),
        "aesthetics_norm": f(row.get("aesthetics_norm")),
        "composite": f(row.get("composite")),
        "timestamp_s": f(row.get("timestamp_s")),
        "refined_timestamp_s": f(row.get("refined_timestamp_s")),
        "uprighter_pred": up_str,
        "uprighter_confidence": up_conf if up_str is not None else None,
    }


def _build_card(
    row: pd.Series,
    thumb_src: str,
    export_path: str,
    aspect: float,
    rotation: int,
    is_image_source: bool,
    frame_idx: int,
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
     data-frame-idx="{frame_idx}"
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
  <button class="flag-btn" title="Toggle flag (space in overlay)">Flag</button>
  {badge_html}
  <img src="{thumb_src}" loading="lazy" alt="">
  <div class="overlay">
    <div>pred: {html.escape(pred_str)}  |  aes: {aes_str}  |  coverage: {coverage_str}</div>
    <div class="stem">{html.escape(video_stem)}</div>
  </div>
</div>"""


def main() -> None:
    parser = ArgumentParser(
        description="Build a self-contained HTML debug viewer with annotated overlay.",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Run YAML config. When provided, --results and "
                             "--output-html default to {output_dir}/results.parquet "
                             "and {output_dir}/index_photos_debug.html. Explicit "
                             "flags still override.")
    parser.add_argument("--results", type=Path, default=None,
                        help="Path to results.parquet.")
    parser.add_argument("--output-html", type=Path, default=None,
                        help="Path to write the HTML file.")
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
            args.output_html = cfg.output_dir / "index_photos_debug.html"
    else:
        cfg = None
    if args.results is None or args.output_html is None:
        parser.error(
            "--results and --output-html are required when --config is not provided",
        )

    df = pd.read_parquet(args.results)
    logger.info("Loaded %d rows from %s", len(df), args.results)

    face_h = df["face_y2"] - df["face_y1"]
    df["face_coverage"] = (
        (df["face_w"] * face_h) / (df["frame_w"] * df["frame_h"])
    ).clip(lower=0.0, upper=1.0)

    html_dir = args.output_html.parent
    html_dir.mkdir(parents=True, exist_ok=True)

    # --- Identity setup ---------------------------------------------------
    has_embedding_cols = all(
        f"face_{s}_embedding" in df.columns for s in (1, 2, 3)
    )
    identities: list[dict] = []
    membership: dict[str, set[str]] = {}
    portraits: dict[str, str] = {}
    if not has_embedding_cols:
        logger.warning(
            "Per-face embeddings not found in results.parquet "
            "(missing face_N_embedding columns) -- re-run pipeline to enable "
            "identity display. Building viewer without identity data.",
        )
    else:
        index_path = IDENTITIES_DIR / "index.json"
        identities = _load_identities(index_path)
        if identities:
            logger.info("Loaded %d identities from %s", len(identities), index_path)
            clusters_path = (
                cfg.output_dir / "clusters.json" if cfg is not None
                else args.output_html.parent / "clusters.json"
            )
            membership = _load_cluster_membership(clusters_path)
            if not membership:
                logger.info(
                    "clusters.json missing or empty at %s -- identities will be "
                    "shown as nearest-only (no hard assignments).", clusters_path,
                )
            for ident in identities:
                rel = _portrait_relpath(ident["portrait_path"], html_dir)
                if rel is not None:
                    portraits[ident["name"]] = rel
        else:
            logger.info(
                "No identities found at %s -- skipping identity display",
                IDENTITIES_DIR / "index.json",
            )

    cards: list[str] = []
    frames: list[dict] = []
    skipped = 0
    image_source_count = 0

    for _, row in df.iterrows():
        video_path = row.get("video_path")
        if not isinstance(video_path, str) or not video_path or pd.isna(video_path):
            skipped += 1
            continue
        is_image_source = Path(video_path).suffix.lower() in IMAGE_EXTENSIONS

        kept = row.get("kept_path")
        if not isinstance(kept, str) or not kept or pd.isna(kept):
            skipped += 1
            continue
        kept_p = Path(kept)
        if not kept_p.exists():
            logger.warning("Keeper missing on disk: %s", kept_p)
            skipped += 1
            continue

        if is_image_source:
            export_path = _to_fwd_slash(video_path)
            image_source_count += 1
        else:
            export_path = _to_fwd_slash(kept_p.resolve())

        thumb_src = _make_img_src(kept_p, html_dir)
        rotation = get_image_rotation_deg(video_path) if is_image_source else 0
        fw = _safe_float(row.get("frame_w"))
        fh = _safe_float(row.get("frame_h"))
        if fw and fh and fw > 0 and fh > 0:
            aspect = (fh / fw) if rotation in (90, 270) else (fw / fh)
        else:
            aspect = 1.0

        frame_idx = len(frames)
        frames.append(_build_frame_json(
            row, kept_p, is_image_source,
            identities, membership, has_embedding_cols,
        ))
        cards.append(_build_card(
            row, thumb_src, export_path, aspect, rotation, is_image_source, frame_idx,
        ))

    if skipped:
        logger.info("Skipped %d rows (missing video_path or keeper)", skipped)
    logger.info(
        "Built %d cards (%d image-source, %d video-source)",
        len(cards), image_source_count, len(cards) - image_source_count,
    )

    frames_json = json.dumps(frames, separators=(",", ":"), allow_nan=False)
    portraits_json = json.dumps(portraits, separators=(",", ":"))
    js_source = (
        JS_TEMPLATE
        .replace("__FRAMES_DATA__", frames_json)
        .replace("__IDENTITY_PORTRAITS__", portraits_json)
    )

    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1400">
<title>Still Extractor - Debug Photos</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Still Extractor &mdash; Debug Photos (<span id="shown-count">{len(cards)}</span> shown)</h1>
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
    Click any card to open the debug overlay. In overlay: <kbd>&larr;</kbd><kbd>&rarr;</kbd> navigate, <kbd>Space</kbd> flag, <kbd>Esc</kbd> close.
  </div>
</header>
<div class="grid">
{chr(10).join(cards)}
</div>
<div id="overlay">
  <div id="overlay-inner">
    <div id="overlay-left">
      <img id="overlay-img" src="" alt="">
      <canvas id="overlay-canvas"></canvas>
      <div id="bbox-tooltip"></div>
    </div>
    <div id="overlay-right">
      <div id="overlay-scores"></div>
    </div>
  </div>
  <button id="overlay-nav-prev" title="Previous (Left arrow)">&larr;</button>
  <button id="overlay-nav-next" title="Next (Right arrow)">&rarr;</button>
  <button id="overlay-close" title="Close (Esc)">&times;</button>
</div>
<script>{js_source}</script>
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
        "stage": "build_debug_viewer",
        "config": str(args.config) if args.config is not None else None,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "card_count": len(cards),
        "video_source": len(cards) - image_source_count,
        "image_source": image_source_count,
        "output_html": str(args.output_html),
        "file_size_mb": round(file_size_mb, 2),
        "has_embedding_cols": bool(has_embedding_cols),
        "identity_count": len(identities),
        "identities_with_portraits": len(portraits),
    }
    summary_path = args.output_html.parent / "build_debug_viewer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
