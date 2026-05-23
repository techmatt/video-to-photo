"""HTML viewer for visualizing anomalous face keypoint geometry.

Reads keypoint_diagnostics.parquet (auto-runs diagnose_keypoints if missing)
and renders each face's full keeper JPEG with the 5-point landmarks plus
face bbox overlaid via a canvas. Anomalous frames come first sorted by
descending min_centroid_dist so the worst ArcFace outliers float to the
top. Filter chips select by anomaly reason; a Show Normal toggle appends
non-anomalous frames after the anomalous ones for comparison.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from still_extractor.diagnose_keypoints import run_diagnostics
from still_extractor.inventory import RunConfig
from still_extractor.utils import safe_float, to_fwd_slash

logger = logging.getLogger(__name__)


def _img_src(img_path: Path, html_dir: Path) -> str:
    img_abs = Path(img_path).resolve()
    try:
        rel = img_abs.relative_to(html_dir.resolve())
        return quote(to_fwd_slash(rel))
    except ValueError:
        return "file:///" + quote(to_fwd_slash(img_abs), safe="/:")


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
.toolbar .summary { margin-left: auto; color: #aaa; font-size: 12px; }
.section-label {
  width: 100%;
  margin: 12px 0 4px 0;
  color: #888;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 12px;
}
.card {
  background: #1a1a1a;
  border: 1px solid #2a2a2a;
  border-radius: 6px;
  padding: 8px;
}
.card.anomalous { border-color: #F59E0B; }
.card.normal { border-color: #2a2a2a; }
.img-wrap {
  position: relative;
  width: 100%;
  background: #000;
  border-radius: 4px;
  overflow: hidden;
}
.img-wrap img {
  display: block;
  width: 100%;
  height: auto;
}
.img-wrap canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}
.badges { margin-top: 6px; display: flex; gap: 4px; flex-wrap: wrap; }
.badge {
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 3px;
  background: #333;
  color: #ddd;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.badge.order { background: #7c2d12; color: #fed7aa; }
.badge.ratio { background: #713f12; color: #fde68a; }
.badge.span { background: #4c1d95; color: #ddd6fe; }
.meta {
  margin-top: 6px;
  font-size: 11px;
  line-height: 1.5;
  color: #bbb;
}
.meta .label { color: #888; }
.meta .v { color: #fff; }
.meta .stem { color: #777; word-break: break-all; font-size: 10px; }
"""


JS = r"""
function drawOverlay(card) {
  const img = card.querySelector('img');
  const canvas = card.querySelector('canvas');
  if (!img || !canvas) return;
  const w = img.naturalWidth, h = img.naturalHeight;
  if (!w || !h) return;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  let kps = [];
  try { kps = JSON.parse(card.dataset.kps || '[]'); } catch (e) { kps = []; }
  let bbox = null;
  try { bbox = JSON.parse(card.dataset.bbox || 'null'); } catch (e) { bbox = null; }

  const anomalous = card.dataset.anomalous === '1';
  const lineW = Math.max(2, Math.round(Math.min(w, h) / 400));
  const r = Math.max(3, Math.round(Math.min(w, h) / 250));

  if (bbox && bbox.length === 4) {
    ctx.lineWidth = lineW;
    ctx.strokeStyle = anomalous ? '#F59E0B' : '#22C55E';
    const [x1, y1, x2, y2] = bbox;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }

  if (kps.length >= 5) {
    const [le, re, no, lm, rm] = kps;
    const eye_mid = [(le[0] + re[0]) / 2, (le[1] + re[1]) / 2];
    const mouth_mid = [(lm[0] + rm[0]) / 2, (lm[1] + rm[1]) / 2];

    ctx.lineWidth = lineW;
    // left_eye -> right_eye (blue)
    ctx.strokeStyle = '#3b82f6';
    ctx.beginPath();
    ctx.moveTo(le[0], le[1]); ctx.lineTo(re[0], re[1]);
    ctx.stroke();
    // eye_mid -> nose (green)
    ctx.strokeStyle = '#22C55E';
    ctx.beginPath();
    ctx.moveTo(eye_mid[0], eye_mid[1]); ctx.lineTo(no[0], no[1]);
    ctx.stroke();
    // nose -> mouth_mid (red)
    ctx.strokeStyle = '#ef4444';
    ctx.beginPath();
    ctx.moveTo(no[0], no[1]); ctx.lineTo(mouth_mid[0], mouth_mid[1]);
    ctx.stroke();

    function dot(p, fill) {
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
      ctx.fill();
    }
    dot(le, '#3b82f6');
    dot(re, '#3b82f6');
    dot(no, '#22C55E');
    dot(lm, '#ef4444');
    dot(rm, '#ef4444');
  }
}

function attachImageHandlers() {
  document.querySelectorAll('.card').forEach(card => {
    const img = card.querySelector('img');
    if (!img) return;
    if (img.complete && img.naturalWidth) {
      drawOverlay(card);
    } else {
      img.addEventListener('load', () => drawOverlay(card));
    }
  });
}

let currentFilter = 'all_anomalous';
let showNormal = false;

function applyFilters() {
  document.querySelectorAll('.card').forEach(card => {
    const anomalous = card.dataset.anomalous === '1';
    const reasons = (card.dataset.reasons || '').split(',').filter(Boolean);
    let show = false;
    if (anomalous) {
      if (currentFilter === 'all' || currentFilter === 'all_anomalous') show = true;
      else if (reasons.includes(currentFilter)) show = true;
    } else {
      if (currentFilter === 'all') show = true;
      else if (showNormal) show = true;
    }
    card.style.display = show ? '' : 'none';
  });
  document.querySelectorAll('.section-label').forEach(el => {
    const which = el.dataset.section;
    let visible = false;
    if (which === 'anomalous') {
      visible = currentFilter !== 'normal_only';
    } else if (which === 'normal') {
      visible = currentFilter === 'all' || showNormal;
    }
    el.style.display = visible ? '' : 'none';
  });
  updateSummary();
}

function setFilter(f) {
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.filter === f);
  });
  applyFilters();
}

function toggleShowNormal() {
  showNormal = !showNormal;
  const btn = document.getElementById('show-normal-btn');
  btn.classList.toggle('active', showNormal);
  btn.textContent = showNormal ? 'Hide Normal' : 'Show Normal';
  applyFilters();
}

function updateSummary() {
  let total = 0, shown = 0, anomShown = 0, normalShown = 0;
  document.querySelectorAll('.card').forEach(card => {
    total++;
    if (card.style.display !== 'none') {
      shown++;
      if (card.dataset.anomalous === '1') anomShown++;
      else normalShown++;
    }
  });
  document.getElementById('summary').textContent =
    `${anomShown} anomalous + ${normalShown} normal = ${shown} shown / ${total} total`;
}

document.addEventListener('DOMContentLoaded', () => {
  attachImageHandlers();
  document.querySelectorAll('.filter-btn').forEach(b => {
    b.addEventListener('click', () => setFilter(b.dataset.filter));
  });
  document.getElementById('show-normal-btn').addEventListener(
    'click', toggleShowNormal,
  );
  applyFilters();
});
"""


def _card_html(row: pd.Series, src: str) -> str:
    anomalous = bool(row.get("anomalous"))
    reasons_csv = row.get("anomaly_reasons") or ""
    reasons = [r for r in reasons_csv.split(",") if r]

    kps_raw = row.get("kps_json")
    try:
        kps_list = json.loads(kps_raw) if isinstance(kps_raw, str) else []
    except Exception:
        kps_list = []
    bbox = [
        safe_float(row.get("face_x1")), safe_float(row.get("face_y1")),
        safe_float(row.get("face_x2")), safe_float(row.get("face_y2")),
    ]
    bbox_json = json.dumps(bbox) if all(v is not None for v in bbox) else "null"
    kps_json = json.dumps(kps_list)

    ratio = safe_float(row.get("ratio"))
    span_frac = safe_float(row.get("kps_span_frac"))
    dist = safe_float(row.get("min_centroid_dist"))
    cluster = row.get("assigned_cluster") or "unknown"
    stem = row.get("video_stem") or ""

    def fmt(v, p=3):
        return f"{v:.{p}f}" if v is not None else "n/a"

    badges_html = "".join(
        f'<span class="badge {r}">{r}</span>' for r in reasons
    )

    klass = "anomalous" if anomalous else "normal"

    return (
        f'<div class="card {klass}" '
        f'data-anomalous="{1 if anomalous else 0}" '
        f'data-reasons="{",".join(reasons)}" '
        f"data-kps='{kps_json}' "
        f"data-bbox='{bbox_json}' "
        f'data-dist="{fmt(dist, 4) if dist is not None else ""}" '
        f'>'
        f'<div class="img-wrap"><img src="{src}" alt=""><canvas></canvas></div>'
        f'<div class="badges">{badges_html}</div>'
        f'<div class="meta">'
        f'<span class="label">ratio</span> <span class="v">{fmt(ratio)}</span> '
        f'&middot; <span class="label">span</span> <span class="v">{fmt(span_frac)}</span> '
        f'&middot; <span class="label">dist</span> <span class="v">{fmt(dist)}</span><br>'
        f'<span class="label">cluster</span> <span class="v">{cluster}</span><br>'
        f'<span class="stem">{stem}</span>'
        f'</div>'
        f'</div>'
    )


def build_html(cfg: RunConfig) -> Path | None:
    diag_path = cfg.output_dir / "keypoint_diagnostics.parquet"
    if not diag_path.exists():
        logger.info(
            "%s missing; running diagnose_keypoints first", diag_path,
        )
        if run_diagnostics(cfg) is None:
            return None
    if not diag_path.exists():
        logger.error("Diagnostics parquet still missing after run; aborting")
        return None

    df = pd.read_parquet(diag_path)
    logger.info("Loaded %d rows from %s", len(df), diag_path)

    html_path = cfg.output_dir / "keypoint_debug.html"
    html_dir = html_path.parent

    # Anomalous first, sorted by min_centroid_dist desc (NaN last)
    anom = df[df["anomalous"] == True].copy()  # noqa: E712
    norm = df[df["anomalous"] != True].copy()  # noqa: E712
    anom = anom.sort_values(
        by="min_centroid_dist", ascending=False, na_position="last",
    )
    norm = norm.sort_values(
        by="min_centroid_dist", ascending=False, na_position="last",
    )

    anom_cards: list[str] = []
    for _, row in anom.iterrows():
        kept = row.get("kept_path")
        if not isinstance(kept, str) or not kept:
            continue
        p = Path(kept)
        if not p.exists():
            logger.warning("Missing keeper on disk: %s", p)
            continue
        anom_cards.append(_card_html(row, _img_src(p, html_dir)))

    norm_cards: list[str] = []
    for _, row in norm.iterrows():
        kept = row.get("kept_path")
        if not isinstance(kept, str) or not kept:
            continue
        p = Path(kept)
        if not p.exists():
            continue
        norm_cards.append(_card_html(row, _img_src(p, html_dir)))

    total_anom = len(anom_cards)
    total_norm = len(norm_cards)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Still Extractor - Keypoint Debug ({cfg.name})</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Keypoint Debug - {cfg.name} ({total_anom} anomalous / {total_anom + total_norm} total)</h1>
  <div class="toolbar">
    <span class="toolbar-label">Filter:</span>
    <button class="filter-btn active" data-filter="all_anomalous">All anomalous</button>
    <button class="filter-btn" data-filter="order">vertical_order</button>
    <button class="filter-btn" data-filter="ratio">ratio</button>
    <button class="filter-btn" data-filter="span">span</button>
    <button class="filter-btn" data-filter="all">All</button>
    <button id="show-normal-btn">Show Normal</button>
    <span class="summary" id="summary"></span>
  </div>
</header>
<div class="grid">
  <div class="section-label" data-section="anomalous">Anomalous (sorted by worst centroid distance)</div>
  {chr(10).join(anom_cards)}
  <div class="section-label" data-section="normal">Normal (control)</div>
  {chr(10).join(norm_cards)}
</div>
<script>{JS}</script>
</body>
</html>
"""

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_doc, encoding="utf-8")
    size_mb = html_path.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%d anom + %d normal cards, %.2f MB)",
                html_path, total_anom, total_norm, size_mb)

    summary = {
        "stage": "build_keypoint_debug",
        "config_name": cfg.name,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "anomalous_cards": total_anom,
        "normal_cards": total_norm,
        "output_html": str(html_path),
        "file_size_mb": round(size_mb, 2),
    }
    (cfg.output_dir / "build_keypoint_debug_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build HTML viewer for anomalous face keypoint geometry.",
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Run YAML config (e.g. configs/june27.yaml).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    cfg = RunConfig.from_yaml(args.config)
    out = build_html(cfg)
    if out is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
