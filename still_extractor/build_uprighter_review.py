"""Build a self-contained HTML review tool for uprighter training frames.

Crawls a frames directory recursively, embeds every JPEG as a card, and lets the
user click (or hover + X/Delete) to toggle "rejected". Exports a JSON array of
repo-root-relative paths for the rejected set.
"""

import html
import json
import logging
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from PIL import Image

logger = logging.getLogger(__name__)


def _to_fwd_slash(p: str | Path) -> str:
    return str(p).replace("\\", "/")


CSS = """
* { box-sizing: border-box; }
html, body { min-width: 1400px; }
body {
  margin: 0;
  padding: 12px;
  background: #1a1a1a;
  color: #fff;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 13px;
}
header {
  position: sticky;
  top: 0;
  background: #1a1a1a;
  padding: 10px 0;
  margin-bottom: 12px;
  border-bottom: 1px solid #333;
  z-index: 10;
}
.toolbar {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}
.toolbar h1 { margin: 0; font-size: 17px; font-weight: 600; flex: 1 1 auto; }
.toolbar .stat { color: #ccc; font-size: 13px; }
.toolbar .stat .num { color: #fff; font-weight: 600; }
.toolbar .stat.kept .num { color: #6fdc8c; }
.toolbar .stat.rejected .num { color: #ff6b6b; }
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
.toolbar .toolbar-sep { color: #444; }
.legend {
  color: #888;
  font-size: 12px;
  margin-top: 6px;
}
.legend kbd {
  background: #2a2a2a;
  border: 1px solid #444;
  border-radius: 3px;
  padding: 1px 5px;
  font-family: inherit;
  font-size: 11px;
  color: #ddd;
}

.grid {
  display: flex;
  flex-wrap: wrap;
  gap: 3px;
  align-content: flex-start;
  width: 100%;
}
.card {
  position: relative;
  background: #111;
  overflow: hidden;
  cursor: pointer;
  /* width/height set by justifyGrid */
}
.card img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
  background: #000;
}
.card::after {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(0, 255, 0, 0.08);
  pointer-events: none;
  transition: background 0.1s;
}
.card.rejected::after { background: rgba(255, 0, 0, 0.25); }
.reject-badge {
  position: absolute;
  top: 4px;
  left: 50%;
  transform: translateX(-50%);
  width: 22px;
  height: 22px;
  border-radius: 11px;
  background: rgba(220, 38, 38, 0.95);
  color: #fff;
  font-weight: 700;
  font-size: 14px;
  line-height: 22px;
  text-align: center;
  display: none;
  z-index: 2;
  pointer-events: none;
}
.card.rejected .reject-badge { display: block; }
"""


JS = """
const REJECTED_KEY = 'uprighter_review.rejected';
const TARGET_ROW_HEIGHT = 180;
const GRID_SPACING = 3;

let rejected = new Set();
let hoveredIdx = -1;
let lastLayoutContainerWidth = -1;

function loadRejected() {
  try {
    const raw = localStorage.getItem(REJECTED_KEY);
    if (raw) rejected = new Set(JSON.parse(raw));
  } catch (e) { /* ignore */ }
}

function saveRejected() {
  localStorage.setItem(REJECTED_KEY, JSON.stringify([...rejected]));
}

function updateCounts() {
  const total = FRAMES.length;
  const r = rejected.size;
  document.getElementById('kept-count').textContent = String(total - r);
  document.getElementById('rejected-count').textContent = String(r);
}

function buildGrid() {
  const grid = document.querySelector('.grid');
  const frag = document.createDocumentFragment();
  for (let i = 0; i < FRAMES.length; i++) {
    const f = FRAMES[i];
    const card = document.createElement('div');
    card.className = 'card';
    card.dataset.idx = String(i);
    card.dataset.aspect = String(f.a);
    if (rejected.has(f.p)) card.classList.add('rejected');

    const img = document.createElement('img');
    img.src = f.s;
    img.loading = 'lazy';
    img.alt = '';
    card.appendChild(img);

    const badge = document.createElement('div');
    badge.className = 'reject-badge';
    badge.textContent = '✕';
    card.appendChild(badge);

    card.addEventListener('click', () => toggleCard(i));
    card.addEventListener('mouseenter', () => { hoveredIdx = i; });
    card.addEventListener('mouseleave', () => { if (hoveredIdx === i) hoveredIdx = -1; });

    frag.appendChild(card);
  }
  grid.appendChild(frag);
}

function toggleCard(i) {
  const f = FRAMES[i];
  const card = document.querySelector('.card[data-idx="' + i + '"]');
  if (!card) return;
  if (rejected.has(f.p)) {
    rejected.delete(f.p);
    card.classList.remove('rejected');
  } else {
    rejected.add(f.p);
    card.classList.add('rejected');
  }
  saveRejected();
  updateCounts();
}

function justifyGrid(cards, containerWidth, targetRowHeight, spacing) {
  // READ PASS: compute layout entirely in memory, no DOM writes.
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

  // WRITE PASS
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
  const cards = Array.from(grid.querySelectorAll('.card')).map(el => ({
    element: el,
    aspectRatio: parseFloat(el.dataset.aspect) || 1.0,
  }));
  justifyGrid(cards, containerWidth, TARGET_ROW_HEIGHT, GRID_SPACING);
}

function exportRejected() {
  const arr = [...rejected].sort();
  const blob = new Blob([JSON.stringify(arr, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'rejected.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

document.addEventListener('DOMContentLoaded', () => {
  loadRejected();
  buildGrid();
  updateCounts();
  relayout(true);

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => relayout(false), 250);
  });

  document.getElementById('export-btn').addEventListener('click', exportRejected);

  document.addEventListener('keydown', e => {
    const tgt = e.target;
    if (tgt && (tgt.tagName === 'INPUT' || tgt.tagName === 'TEXTAREA')) return;
    if (e.key === 'x' || e.key === 'X' || e.key === 'Delete') {
      if (hoveredIdx >= 0) {
        toggleCard(hoveredIdx);
        e.preventDefault();
      }
    }
  });
});
"""


def _read_aspect(path: Path) -> float | None:
    try:
        with Image.open(path) as img:
            w, h = img.size
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None
    if h <= 0 or w <= 0:
        return None
    return max(0.1, min(10.0, w / h))


def main() -> None:
    parser = ArgumentParser(
        description="Build a self-contained HTML reviewer for uprighter training frames.",
    )
    parser.add_argument("--frames-dir", type=Path, required=True,
                        help="Directory to crawl recursively for *.jpg frames.")
    parser.add_argument("--output-html", type=Path, required=True,
                        help="Path to write the self-contained HTML file.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    frames_dir = args.frames_dir
    if not frames_dir.exists():
        parser.error(f"--frames-dir does not exist: {frames_dir}")

    output_html = args.output_html
    output_html.parent.mkdir(parents=True, exist_ok=True)

    abs_frames_dir = frames_dir.resolve()
    abs_html_dir = output_html.resolve().parent
    repo_root = Path.cwd().resolve()

    files = sorted(
        abs_frames_dir.rglob("*.jpg"),
        key=lambda p: (p.parent.name, p.name),
    )
    logger.info("Found %d JPEG files under %s", len(files), abs_frames_dir)

    frames: list[dict] = []
    skipped = 0
    for i, fp in enumerate(files):
        if i % 1000 == 0 and i > 0:
            logger.info("Read %d / %d aspect ratios", i, len(files))
        aspect = _read_aspect(fp)
        if aspect is None:
            skipped += 1
            continue
        try:
            src_rel = fp.relative_to(abs_html_dir)
        except ValueError:
            logger.warning("Frame %s not under html dir %s; skipping", fp, abs_html_dir)
            skipped += 1
            continue
        try:
            export_rel = fp.relative_to(repo_root)
        except ValueError:
            export_rel = fp
        frames.append({
            "s": quote(_to_fwd_slash(src_rel)),
            "p": _to_fwd_slash(export_rel),
            "a": round(aspect, 4),
        })

    if skipped:
        logger.info("Skipped %d files (unreadable or outside html dir)", skipped)
    logger.info("Embedding %d frames into %s", len(frames), output_html)

    frames_json = json.dumps(frames, separators=(",", ":"))
    title_text = f"Uprighter Review &mdash; {len(frames)} frames"

    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1400">
<title>{html.escape(f"Uprighter Review - {len(frames)} frames")}</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <div class="toolbar">
    <h1>{title_text}</h1>
    <span class="stat kept">Kept: <span class="num" id="kept-count">{len(frames)}</span></span>
    <span class="stat rejected">Rejected: <span class="num" id="rejected-count">0</span></span>
    <span class="toolbar-sep">|</span>
    <button id="export-btn">Export Rejected</button>
  </div>
  <div class="legend">
    Click a card to toggle reject. Hover + <kbd>X</kbd> or <kbd>Delete</kbd> to sweep quickly.
    Default = kept (green tint), rejected = red tint with <kbd>&#x2715;</kbd> badge.
    Progress is auto-saved to localStorage.
  </div>
</header>
<div class="grid"></div>
<script>
const FRAMES = {frames_json};
{JS}
</script>
</body>
</html>
"""

    output_html.write_text(body, encoding="utf-8")
    file_size_mb = output_html.stat().st_size / (1024 * 1024)
    logger.info(
        "Wrote %s (%d frames, %.2f MB)",
        output_html, len(frames), file_size_mb,
    )

    summary = {
        "stage": "build_uprighter_review",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "frame_count": len(frames),
        "skipped": skipped,
        "frames_dir": str(frames_dir),
        "output_html": str(output_html),
        "file_size_mb": round(file_size_mb, 2),
    }
    summary_path = output_html.parent / "build_uprighter_review_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
