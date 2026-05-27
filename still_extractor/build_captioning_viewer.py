"""Build a self-contained viewer for captioned photos.

Reads ``results.parquet`` (caption_* columns written by ``caption_photos.py``)
and the raw model output sidecar at ``caption_experiments/raw_outputs_A.jsonl``
and writes ``{output_dir}/captioning_viewer.html`` -- a single file that opens
directly from disk (file://) with no server.
"""

import argparse
import html
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from still_extractor.inventory import RunConfig
from still_extractor.utils import to_fwd_slash as _to_fwd_slash

logger = logging.getLogger(__name__)


MIN_QUALITY_ALLOWED: dict[str, set[str]] = {
    "good": {"good"},
    "okay": {"good", "okay"},
}


def _make_img_src(img_path: Path, html_dir: Path) -> str:
    """Relative path if possible, else file:// absolute URL."""
    img_abs = Path(img_path).resolve()
    try:
        rel = img_abs.relative_to(html_dir.resolve())
        return quote(_to_fwd_slash(rel))
    except ValueError:
        return "file:///" + quote(_to_fwd_slash(img_abs), safe="/:")


def _opt_str(val) -> str | None:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val)
    return s if s else None


def _opt_int(val) -> int | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _opt_float(val) -> float | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f


def _load_raw_outputs(jsonl_path: Path) -> dict[str, dict]:
    """Index raw_outputs_A.jsonl by kept_path."""
    if not jsonl_path.exists():
        logger.info("Raw outputs file not found: %s", jsonl_path)
        return {}
    out: dict[str, dict] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            kp = obj.get("kept_path")
            if isinstance(kp, str) and kp:
                out[kp] = obj
    logger.info("Loaded %d raw output entries from %s", len(out), jsonl_path)
    return out


def _build_photo(row: pd.Series, raw_index: dict[str, dict]) -> dict:
    kept = str(row["kept_path"])
    raw = raw_index.get(kept, {})
    return {
        "path": kept,
        "aesthetic": _opt_int(row.get("caption_aesthetic_score")),
        "aesthetic_solo": _opt_int(row.get("caption_aesthetic_score_solo")),
        "setting": _opt_str(row.get("caption_setting")),
        "activity": _opt_str(row.get("caption_activity")),
        "people": _opt_str(row.get("caption_people")),
        "mood": _opt_str(row.get("caption_mood")),
        "framing": _opt_str(row.get("caption_framing")),
        "description": _opt_str(row.get("caption_description")),
        "model": _opt_str(row.get("caption_model")),
        "composite": _opt_float(row.get("composite")),
        "face_pred": _opt_str(row.get("face_1_pred_label")),
        "raw_p1": _opt_str(raw.get("prompt1_raw")),
        "raw_p2": _opt_str(raw.get("prompt2_raw")),
        "raw_p3": _opt_str(raw.get("prompt3_raw")),
    }


def _sort_photos(photos: list[dict]) -> list[dict]:
    """Aesthetic desc with nulls last; composite desc as tiebreaker."""
    def key(p: dict) -> tuple[int, int, float]:
        aes = p["aesthetic"]
        comp = p["composite"] if p["composite"] is not None else 0.0
        if aes is None:
            return (1, 0, -comp)
        return (0, -aes, -comp)
    return sorted(photos, key=key)


# ---------------------------------------------------------------------------
# HTML / CSS / JS
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

:root {
  --bg: #ffffff;
  --surface: #f8f8f8;
  --panel: #fafbfc;
  --border: #e0e0e0;
  --text-primary: #202124;
  --text-secondary: #5f6368;
  --text-muted: #9aa0a6;
  --accent: #1a73e8;
  --selected-ring: #1a73e8;
  --chip-bg: #f1f3f4;
  --good: #0f9d58;
  --okay: #f4b400;
  --bad: #db4437;
  --none: #9aa0a6;
}

* { box-sizing: border-box; }

html, body {
  margin: 0;
  padding: 0;
  height: 100vh;
  background: var(--bg);
  color: var(--text-primary);
  font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 14px;
  overflow: hidden;
}

#app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

header {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 10px 20px;
  border-bottom: 1px solid var(--border);
  background: var(--bg);
}
.title {
  font-weight: 600;
  font-size: 15px;
  color: var(--text-primary);
}
.stats {
  font-size: 12px;
  color: var(--text-secondary);
}

.cols {
  flex: 1 1 auto;
  display: flex;
  min-height: 0;
}

/* ---------- Left column ---------- */

.left {
  flex: 0 0 35%;
  border-right: 1px solid var(--border);
  overflow-y: auto;
  padding: 12px;
  background: var(--surface);
}
.grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}
@media (max-width: 900px) {
  .grid { grid-template-columns: repeat(2, 1fr); }
}

.card {
  position: relative;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
  cursor: pointer;
  outline: 2px solid transparent;
  outline-offset: -2px;
  transition: outline-color 0.1s, box-shadow 0.1s;
}
.card:hover { box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
.card.selected {
  outline-color: var(--selected-ring);
  box-shadow: 0 1px 6px rgba(26,115,232,0.25);
}
.card .thumb-wrap {
  position: relative;
  width: 100%;
  height: 180px;
  background: #f1f3f4;
}
.card img.thumb {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.card .aes-badge {
  position: absolute;
  top: 6px;
  right: 6px;
  min-width: 30px;
  height: 30px;
  padding: 0 6px;
  border-radius: 6px;
  background: rgba(0,0,0,0.6);
  color: white;
  font-weight: 600;
  font-size: 14px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 1px 3px rgba(0,0,0,0.4);
}
.card .aes-badge.aes-high  { background: #0f9d58; }
.card .aes-badge.aes-mid   { background: #f4b400; color: #2a2300; }
.card .aes-badge.aes-low   { background: #db4437; }
.card .aes-badge.aes-none  { background: #9aa0a6; }
.card .meta {
  padding: 6px 8px;
  font-size: 12px;
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.card .meta .mood {
  color: var(--accent);
  font-weight: 500;
  margin-right: 6px;
}

/* ---------- Right column ---------- */

.right {
  flex: 1 1 65%;
  overflow-y: auto;
  padding: 20px 24px;
  background: var(--bg);
}
.detail-empty {
  color: var(--text-secondary);
  font-style: italic;
  padding-top: 80px;
  text-align: center;
}
.detail-image-wrap {
  text-align: center;
  background: #15171c;
  border-radius: 6px;
  padding: 8px;
  margin-bottom: 14px;
}
.detail-image-wrap a {
  display: inline-block;
  cursor: zoom-in;
}
.detail-image-wrap img {
  max-width: 100%;
  max-height: 50vh;
  object-fit: contain;
  display: block;
}
.detail-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 18px;
  padding: 10px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 14px;
  font-size: 13px;
}
.stat-item .stat-label {
  color: var(--text-secondary);
  margin-right: 4px;
  text-transform: uppercase;
  font-size: 10px;
  letter-spacing: 0.06em;
  font-weight: 600;
}
.stat-item .stat-value { color: var(--text-primary); font-weight: 500; }
.stat-item .stat-value.aes-high { color: var(--good); }
.stat-item .stat-value.aes-mid  { color: #b07a00; }
.stat-item .stat-value.aes-low  { color: var(--bad); }
.stat-item .stat-value.aes-none { color: var(--text-muted); }

.section-label {
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-secondary);
  font-weight: 600;
  margin: 0 0 8px 0;
}
.fields-table {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 4px 16px;
  margin-bottom: 16px;
  font-size: 13px;
}
.fields-table .f-key {
  color: var(--text-secondary);
  text-transform: capitalize;
}
.fields-table .f-val { color: var(--text-primary); }
.fields-table .f-val.missing { color: var(--text-muted); font-style: italic; }

.description-block {
  background: var(--panel);
  border-left: 3px solid var(--accent);
  padding: 10px 14px;
  margin-bottom: 16px;
  font-size: 14px;
  line-height: 1.5;
  color: var(--text-primary);
  border-radius: 0 4px 4px 0;
}
.description-block.missing { color: var(--text-muted); font-style: italic; }

.raw-section {
  border-top: 1px solid var(--border);
  padding-top: 12px;
  margin-top: 4px;
}
.raw-toggle {
  background: transparent;
  border: none;
  padding: 0;
  cursor: pointer;
  font: 600 10px/1 'DM Sans', sans-serif;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-secondary);
}
.raw-toggle .caret { display: inline-block; margin-left: 4px; transition: transform 0.15s; }
.raw-toggle.open .caret { transform: rotate(180deg); }
.raw-body { display: none; margin-top: 10px; }
.raw-body.open { display: block; }
.raw-prompt {
  margin-bottom: 10px;
}
.raw-prompt-name {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 4px;
}
.raw-prompt pre {
  margin: 0;
  padding: 8px 10px;
  background: #15171c;
  color: #d0d4dc;
  border-radius: 4px;
  font: 12px/1.45 ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  white-space: pre-wrap;
  word-break: break-word;
}
.raw-prompt.missing pre {
  color: var(--text-muted);
  font-style: italic;
  background: var(--panel);
}

.empty-state {
  padding: 60px 24px;
  text-align: center;
  color: var(--text-secondary);
}
.empty-state code {
  display: inline-block;
  margin-top: 12px;
  padding: 8px 12px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  font-size: 13px;
}
"""


JS_TEMPLATE = """
const PHOTOS = __PHOTOS__;

const grid = document.getElementById('grid');
const detail = document.getElementById('detail');

let selectedIdx = -1;

function aesClass(a) {
  if (a == null) return 'aes-none';
  if (a >= 8) return 'aes-high';
  if (a >= 6) return 'aes-mid';
  return 'aes-low';
}

function aesDisplay(a) {
  return a == null ? '—' : String(a);
}

function esc(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function fileUrl(absPath) {
  // Convert backslashes to forward slashes; build file:/// URL.
  let p = String(absPath).replace(/\\\\/g, '/');
  // Encode each segment, preserving slashes and the drive colon.
  return 'file:///' + encodeURI(p).replace(/#/g, '%23');
}

function renderGrid() {
  const html = PHOTOS.map((p, i) => {
    const aes = aesDisplay(p.aesthetic);
    const cls = aesClass(p.aesthetic);
    const mood = p.mood ? `<span class="mood">${esc(p.mood)}</span>` : '';
    const setting = p.setting ? esc(p.setting) : '';
    const metaLine = (mood || setting) ? `${mood}${setting}` : '&nbsp;';
    return `
      <div class="card" data-idx="${i}">
        <div class="thumb-wrap">
          <img class="thumb" src="${esc(p.src)}" loading="lazy" alt="">
          <div class="aes-badge ${cls}">${aes}</div>
        </div>
        <div class="meta">${metaLine}</div>
      </div>`;
  }).join('');
  grid.innerHTML = html;
  grid.querySelectorAll('.card').forEach(el => {
    el.addEventListener('click', () => selectIdx(parseInt(el.dataset.idx, 10)));
  });
}

function fieldRow(label, value) {
  if (value == null || value === '') {
    return `<div class="f-key">${label}</div><div class="f-val missing">—</div>`;
  }
  return `<div class="f-key">${label}</div><div class="f-val">${esc(value)}</div>`;
}

function rawBlock(name, text) {
  if (text == null || text === '') {
    return `
      <div class="raw-prompt missing">
        <div class="raw-prompt-name">${name}</div>
        <pre>(no output available)</pre>
      </div>`;
  }
  return `
    <div class="raw-prompt">
      <div class="raw-prompt-name">${name}</div>
      <pre>${esc(text)}</pre>
    </div>`;
}

function renderDetail(idx) {
  if (idx < 0 || idx >= PHOTOS.length) {
    detail.innerHTML = '<div class="detail-empty">Select a photo on the left.</div>';
    return;
  }
  const p = PHOTOS[idx];
  const url = fileUrl(p.path);
  const aes = aesDisplay(p.aesthetic);
  const aesCls = aesClass(p.aesthetic);
  const solo = p.aesthetic_solo != null ? ` <span style="color:var(--text-secondary);font-weight:400">(solo: ${p.aesthetic_solo}/10)</span>` : '';
  const comp = p.composite != null ? p.composite.toFixed(3) : '—';
  const facePred = p.face_pred ? esc(p.face_pred) : '—';

  const hasAnyRaw = (p.raw_p1 != null) || (p.raw_p2 != null) || (p.raw_p3 != null);
  const rawSection = hasAnyRaw ? `
    <div class="raw-section">
      <button class="raw-toggle" id="raw-toggle"><span>Raw prompt outputs</span><span class="caret">▾</span></button>
      <div class="raw-body" id="raw-body">
        ${rawBlock('Prompt 1 (structured)', p.raw_p1)}
        ${rawBlock('Prompt 2 (description)', p.raw_p2)}
        ${rawBlock('Prompt 3 (aesthetic)', p.raw_p3)}
      </div>
    </div>` : '';

  const description = (p.description == null || p.description === '')
    ? '<div class="description-block missing">(no description)</div>'
    : `<div class="description-block">${esc(p.description)}</div>`;

  detail.innerHTML = `
    <div class="detail-image-wrap">
      <a href="${esc(url)}" target="_blank" rel="noopener" title="Open in new tab">
        <img src="${esc(p.src)}" alt="">
      </a>
    </div>
    <div class="detail-stats">
      <div class="stat-item">
        <span class="stat-label">Aesthetic</span>
        <span class="stat-value ${aesCls}">${aes}/10</span>${solo}
      </div>
      <div class="stat-item">
        <span class="stat-label">Composite</span>
        <span class="stat-value">${comp}</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">Face pred</span>
        <span class="stat-value">${facePred}</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">Model</span>
        <span class="stat-value">${esc(p.model || '—')}</span>
      </div>
    </div>

    <div class="section-label">Structured fields</div>
    <div class="fields-table">
      ${fieldRow('setting', p.setting)}
      ${fieldRow('activity', p.activity)}
      ${fieldRow('people', p.people)}
      ${fieldRow('mood', p.mood)}
      ${fieldRow('framing', p.framing)}
    </div>

    <div class="section-label">Description</div>
    ${description}

    ${rawSection}
  `;

  const tog = document.getElementById('raw-toggle');
  if (tog) {
    tog.addEventListener('click', () => {
      tog.classList.toggle('open');
      const body = document.getElementById('raw-body');
      if (body) body.classList.toggle('open');
    });
  }
}

function selectIdx(idx) {
  if (idx < 0 || idx >= PHOTOS.length) return;
  if (idx === selectedIdx) return;
  selectedIdx = idx;
  grid.querySelectorAll('.card').forEach(el => {
    el.classList.toggle('selected', parseInt(el.dataset.idx, 10) === idx);
  });
  renderDetail(idx);
  const sel = grid.querySelector(`.card[data-idx="${idx}"]`);
  if (sel) sel.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

function gridCols() {
  const w = grid.clientWidth;
  return w < 600 ? 2 : 3;
}

window.addEventListener('keydown', (e) => {
  if (PHOTOS.length === 0) return;
  const cols = gridCols();
  let next = selectedIdx;
  if (e.key === 'ArrowRight') next = Math.min(PHOTOS.length - 1, (selectedIdx < 0 ? 0 : selectedIdx + 1));
  else if (e.key === 'ArrowLeft') next = Math.max(0, (selectedIdx < 0 ? 0 : selectedIdx - 1));
  else if (e.key === 'ArrowDown') next = Math.min(PHOTOS.length - 1, (selectedIdx < 0 ? 0 : selectedIdx + cols));
  else if (e.key === 'ArrowUp') next = Math.max(0, (selectedIdx < 0 ? 0 : selectedIdx - cols));
  else if (e.key === 'Enter' || e.key === ' ') {
    if (selectedIdx >= 0) {
      e.preventDefault();
      window.open(fileUrl(PHOTOS[selectedIdx].path), '_blank');
    }
    return;
  } else {
    return;
  }
  e.preventDefault();
  selectIdx(next);
});

renderGrid();
if (PHOTOS.length > 0) selectIdx(0);
else renderDetail(-1);
"""


def _build_html(photos: list[dict], stats_line: str, title: str) -> str:
    if not photos:
        body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div id="app">
  <header>
    <span class="title">{html.escape(title)}</span>
    <span class="stats">No captioned photos yet.</span>
  </header>
  <div class="empty-state">
    <p>No captioned photos yet. Run:</p>
    <code>uv run python -m still_extractor.caption_photos --config &lt;config&gt;.yaml</code>
  </div>
</div>
</body>
</html>
"""
        return body

    photos_json = json.dumps(photos, separators=(",", ":"), allow_nan=False)
    js = JS_TEMPLATE.replace("__PHOTOS__", photos_json)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div id="app">
  <header>
    <span class="title">{html.escape(title)}</span>
    <span class="stats">{html.escape(stats_line)}</span>
  </header>
  <div class="cols">
    <div class="left">
      <div class="grid" id="grid"></div>
    </div>
    <div class="right" id="detail">
      <div class="detail-empty">Select a photo on the left.</div>
    </div>
  </div>
</div>
<script>{js}</script>
</body>
</html>
"""


def _stats_line(photos: list[dict]) -> str:
    n = len(photos)
    aes = [p["aesthetic"] for p in photos if p["aesthetic"] is not None]
    if aes:
        mean = sum(aes) / len(aes)
        sorted_aes = sorted(aes)
        mid = len(sorted_aes) // 2
        median = (
            sorted_aes[mid] if len(sorted_aes) % 2 == 1
            else (sorted_aes[mid - 1] + sorted_aes[mid]) / 2
        )
        pct = round(100.0 * len(aes) / n)
        return (
            f"{n} photos | Aesthetic: mean {mean:.1f}, median {median:g} | "
            f"Scored: {len(aes)}/{n} ({pct}%) | Sorted by aesthetic ↓"
        )
    return f"{n} photos | No aesthetic scores | Sorted by composite ↓"


def _try_open(path: Path) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        logger.debug("Open failed for %s: %s", path, e)


def build(
    cfg: RunConfig,
    min_quality: str,
    output_path: Path | None,
) -> Path:
    results_path = cfg.output_dir / "results.parquet"
    if not results_path.exists():
        raise FileNotFoundError(f"results.parquet not found: {results_path}")

    df = pd.read_parquet(results_path)
    logger.info("Loaded %d rows from %s", len(df), results_path)

    out_path = output_path if output_path is not None else cfg.output_dir / "captioning_viewer.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html_dir = out_path.parent

    title = f"Captioning Viewer - {cfg.name}"

    # Filter: caption_setting must exist as a column AND be non-null on the row,
    # AND pred_label matches the requested minimum quality.
    if "caption_setting" not in df.columns:
        logger.warning("No 'caption_setting' column in results.parquet -- nothing captioned yet")
        out_path.write_text(_build_html([], "", title), encoding="utf-8")
        print(f"captioning_viewer.html written: 0 captioned photos")
        print(f"Path: {out_path}")
        return out_path

    allowed = MIN_QUALITY_ALLOWED[min_quality]
    mask = (
        df["kept_path"].notna()
        & df["caption_setting"].notna()
        & df["pred_label"].isin(allowed)
    )
    sub = df.loc[mask]
    logger.info(
        "Filtered to %d captioned rows (pred_label in %s)",
        len(sub), sorted(allowed),
    )

    raw_path = cfg.output_dir / "caption_experiments" / "raw_outputs_A.jsonl"
    raw_index = _load_raw_outputs(raw_path)

    photos: list[dict] = []
    for _, row in sub.iterrows():
        kept_p = Path(str(row["kept_path"]))
        if not kept_p.exists():
            logger.debug("Skipping missing keeper on disk: %s", kept_p)
            continue
        p = _build_photo(row, raw_index)
        p["src"] = _make_img_src(kept_p, html_dir)
        photos.append(p)

    photos = _sort_photos(photos)
    stats = _stats_line(photos)

    if not photos:
        out_path.write_text(_build_html([], stats, title), encoding="utf-8")
        print(f"captioning_viewer.html written: 0 captioned photos")
        print(f"Path: {out_path}")
        return out_path

    body = _build_html(photos, stats, title)
    out_path.write_text(body, encoding="utf-8")

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%d photos, %.2f MB)", out_path, len(photos), file_size_mb)

    summary = {
        "stage": "build_captioning_viewer",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config_name": cfg.name,
        "min_quality": min_quality,
        "results_parquet": str(results_path),
        "raw_outputs": str(raw_path) if raw_path.exists() else None,
        "raw_outputs_matched": sum(1 for p in photos if p["raw_p1"] is not None or p["raw_p2"] is not None),
        "captioned_photos": len(photos),
        "aesthetic_scored": sum(1 for p in photos if p["aesthetic"] is not None),
        "output_html": str(out_path),
        "file_size_mb": round(file_size_mb, 2),
    }
    summary_path = out_path.parent / "build_captioning_viewer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"captioning_viewer.html written: {len(photos)} captioned photos")
    print(f"Path: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a captioning viewer HTML from results.parquet."
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Run YAML config (results.parquet under cfg.output_dir).")
    parser.add_argument("--min-quality", choices=["good", "okay"], default="good",
                        help="Minimum pred_label to include (matches caption_photos.py).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Override output HTML path "
                             "(default: {output_dir}/captioning_viewer.html).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--no-open", action="store_true",
                        help="Do not attempt to open the file in a browser.")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    cfg = RunConfig.from_yaml(args.config)
    out_path = build(cfg, args.min_quality, args.output)
    if not args.no_open:
        _try_open(out_path)


if __name__ == "__main__":
    main()
