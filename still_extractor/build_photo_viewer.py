"""Build a self-contained Google Photos-inspired viewer for keeper frames.

Single output file (`index_photos.html`) replaces the prior split between
`index_photos.html` and `index_photos_debug.html` — debug features
(scores, face bboxes, video badge) are now toggled via the in-app Settings
panel and persisted to localStorage.
"""

import base64
import html
import json
import logging
import os
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from PIL import ExifTags, Image

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
UNKNOWN_CHIP_ID = "__unknown__"

_EXIF_ORIENTATION_TAG = next(
    k for k, v in ExifTags.TAGS.items() if v == "Orientation"
)
_EXIF_ORIENTATION_TO_DEG = {1: 0, 3: 180, 6: 90, 8: 270}

_MONTH_NAMES_FULL = [
    "Unknown", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTH_NAMES_SHORT = [
    "?", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

_UPRIGHTER_LABEL_MAP = {90: "90cw", 180: "180", 270: "270cw"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

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


def _opt_int(val) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _opt_bool(val) -> bool | None:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    return bool(val)


def _section_key(year: int, month: int) -> str:
    """localStorage section key: '2025-05', '2024-12', '0-0' (unknown)."""
    return f"{year}-{month:02d}"


def _section_label(year: int, month: int) -> str:
    """Human-readable section label: 'May 2025', 'December 2024', 'Unknown'."""
    if year == 0 and month == 0:
        return "Unknown"
    if month == 0:
        return f"Unknown {year}"
    name = _MONTH_NAMES_FULL[month] if 1 <= month <= 12 else "Unknown"
    return f"{name} {year}"


# ---------------------------------------------------------------------------
# Frame dimensions sidecar cache
# ---------------------------------------------------------------------------

def _load_frame_dimensions_cache(cache_path: Path) -> dict[str, list[int]]:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[int]] = {}
    for k, v in data.items():
        if (
            isinstance(k, str) and isinstance(v, list) and len(v) == 2
            and all(isinstance(n, int) and n > 0 for n in v)
        ):
            out[k] = [int(v[0]), int(v[1])]
    return out


def _save_frame_dimensions_cache(cache_path: Path, dims: dict[str, list[int]]) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(dims, separators=(",", ":")), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to write %s: %s", cache_path, e)


def _read_image_dimensions(image_path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            if w > 0 and h > 0:
                return int(w), int(h)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Identity / cluster data loading (ported from build_debug_viewer)
# ---------------------------------------------------------------------------

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v
    return v / norm


def _parse_embedding(val) -> np.ndarray | None:
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


def _portrait_data_uri(portrait_path: Path) -> str | None:
    try:
        data = portrait_path.read_bytes()
    except Exception:
        return None
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _portrait_relpath(portrait_path: str, html_dir: Path) -> str | None:
    portrait = Path(portrait_path).resolve()
    if not portrait.exists():
        return None
    try:
        rel = os.path.relpath(portrait, html_dir.resolve())
    except ValueError:
        return None
    return _to_fwd_slash(rel)


def _load_identity_index(index_path: Path) -> dict[str, dict]:
    """Read identities/index.json -> {name: {display_name, portrait_path, centroid}}.

    Identities without a centroid are still indexed so display names resolve.
    """
    if not index_path.exists():
        return {}
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to parse %s (%s); using raw identity names", index_path, e)
        return {}
    if not isinstance(raw, list):
        return {}
    out: dict[str, dict] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        display = entry.get("display_name")
        portrait = entry.get("portrait_path")
        centroid_raw = entry.get("centroid")
        centroid: np.ndarray | None = None
        if isinstance(centroid_raw, list):
            try:
                vec = _l2_normalize(np.asarray(centroid_raw, dtype=np.float32))
                if vec.ndim == 1 and vec.shape[0] > 0:
                    centroid = vec
            except (TypeError, ValueError):
                pass
        out[name] = {
            "name": name,
            "display_name": display if isinstance(display, str) and display else name,
            "portrait_path": (
                portrait if isinstance(portrait, str) and portrait
                else f"data/identities/{name}.png"
            ),
            "centroid": centroid,
        }
    return out


def _load_cluster_artifacts(clusters_path: Path) -> tuple[dict[str, set[str]], set[str], list[dict], dict[str, set[str]]]:
    """Read clusters.json -> (frame_to_identities, frames_with_unknown, clusters_list, membership).

    frame_to_identities: card_key -> set of identity names (used for People filter)
    frames_with_unknown: set of card_keys with unclustered faces
    clusters_list: raw cluster list (for member_counts in chips)
    membership: identity name -> set of card_keys (used for "assigned" flag in debug overlay)
    """
    if not clusters_path.exists():
        return {}, set(), [], {}
    try:
        data = json.loads(clusters_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to parse %s (%s); skipping face filter", clusters_path, e)
        return {}, set(), [], {}

    frame_to_idents: dict[str, set[str]] = {}
    membership: dict[str, set[str]] = {}
    clusters_list = data.get("clusters", []) if isinstance(data, dict) else []
    for c in clusters_list:
        if not isinstance(c, dict):
            continue
        name = c.get("identity")
        if not isinstance(name, str):
            continue
        ck_set: set[str] = set()
        for fid in c.get("frame_ids", []) or []:
            if isinstance(fid, str):
                frame_to_idents.setdefault(fid, set()).add(name)
                ck_set.add(fid)
        membership[name] = ck_set

    frames_with_unknown: set[str] = {
        fid for fid in (data.get("unknown_frame_ids", []) or [])
        if isinstance(fid, str)
    }
    return frame_to_idents, frames_with_unknown, clusters_list, membership


def _nearest_identity(
    emb: np.ndarray, identities: list[dict],
) -> tuple[int, float] | None:
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


# ---------------------------------------------------------------------------
# Per-frame debug payload (ported / consolidated from build_debug_viewer)
# ---------------------------------------------------------------------------

def _face_slot_payload(row: pd.Series, slot: int) -> dict | None:
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


def _face_identity_payload(
    row: pd.Series,
    slot: int,
    identities_with_centroids: list[dict],
    membership: dict[str, set[str]],
    has_embedding_cols: bool,
) -> dict | None:
    if not identities_with_centroids or not has_embedding_cols:
        return None
    if _opt_int(row.get(f"face_{slot}_x1")) is None:
        return None
    emb = _parse_embedding(row.get(f"face_{slot}_embedding"))
    if emb is None:
        return None
    nearest = _nearest_identity(emb, identities_with_centroids)
    if nearest is None:
        return None
    idx, dist = nearest
    ident = identities_with_centroids[idx]
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
    identities_with_centroids: list[dict],
    membership: dict[str, set[str]],
    has_embedding_cols: bool,
    natural_w: int,
    natural_h: int,
) -> dict:
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
        _face_identity_payload(row, slot, identities_with_centroids, membership, has_embedding_cols)
        for slot in (1, 2, 3)
    ]
    return {
        "video_basename": video_basename,
        "video_stem": row.get("video_stem") or "",
        "kept_basename": kept_path.name,
        "source_type": "image" if is_image_source else "video",
        "frame_w": _opt_int(row.get("frame_w")),
        "frame_h": _opt_int(row.get("frame_h")),
        "w": int(natural_w),
        "h": int(natural_h),
        "faces": faces,
        "face_identities": face_identities,
        "face_count": _opt_int(row.get("face_count")),
        "best_pair_score": f(row.get("best_pair_score")),
        "rejected_faces": _parse_rejected_faces(row.get("rejected_faces_json")),
        "rejected_face_count": _opt_int(row.get("rejected_face_count")) or 0,
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


# ---------------------------------------------------------------------------
# HTML building
# ---------------------------------------------------------------------------

VIDEO_BADGE_HTML = (
    '<div class="video-badge" title="Video source">'
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" '
    'fill="white" aria-hidden="true">'
    '<path d="M8 5v14l11-7z"/>'
    '</svg>'
    '</div>'
)


def _build_card(
    row: pd.Series,
    thumb_src: str,
    export_path: str,
    rotation: int,
    is_image_source: bool,
    frame_idx: int,
    year: int,
    month: int,
    ckey: str,
    identities: list[str],
    has_unknown: bool,
) -> str:
    pred_raw = row.get("pred_label")
    pred_label = (
        pred_raw
        if isinstance(pred_raw, str) and pred_raw and not pd.isna(pred_raw)
        else "none"
    )
    video_stem = str(row.get("video_stem", "") or "")
    source_type = "image" if is_image_source else "video"
    badge_html = "" if is_image_source else VIDEO_BADGE_HTML
    identities_attr = "|".join(identities)

    return f"""<div class="photo-card"
     data-card-key="{html.escape(ckey)}"
     data-frame-idx="{frame_idx}"
     data-export-path="{html.escape(export_path)}"
     data-source-type="{source_type}"
     data-quality="{html.escape(pred_label)}"
     data-year="{year}"
     data-month="{month}"
     data-rotation="{rotation}"
     data-video-stem="{html.escape(video_stem)}"
     data-identities="{html.escape(identities_attr)}"
     data-has-unknown="{'true' if has_unknown else 'false'}">
  {badge_html}
  <button class="card-check" aria-label="Select photo" tabindex="-1">
    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
      <path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z" fill="currentColor"/>
    </svg>
  </button>
  <button class="card-remove" aria-label="Remove from export" tabindex="-1">
    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
      <path d="M19 6.4L17.6 5 12 10.6 6.4 5 5 6.4 10.6 12 5 17.6 6.4 19 12 13.4 17.6 19 19 17.6 13.4 12z" fill="currentColor"/>
    </svg>
  </button>
  <img src="{thumb_src}" loading="lazy" alt="">
</div>"""


def _year_pills_html(
    years: list[int],
    months_by_year: dict[int, list[int]],
    year_counts: dict[int, int],
    month_counts: dict[tuple[int, int], int],
) -> str:
    """Render the two-level year/month pill structure.

    For each year:
      - One year pill (with tri-state checkbox)
      - A month-pills row directly after, hidden by default (CSS max-height anim)
    """
    blocks: list[str] = []
    for y in years:
        y_label = "Unknown" if y == 0 else str(y)
        y_count = year_counts.get(y, 0)
        months = months_by_year.get(y, [])

        # Month pills row: independent toggles per month
        month_pills: list[str] = []
        for m in months:
            if m == 0:
                m_short = "?"
                m_full = "Unknown"
            else:
                m_short = _MONTH_NAMES_SHORT[m] if 1 <= m <= 12 else "?"
                m_full = _MONTH_NAMES_FULL[m] if 1 <= m <= 12 else "Unknown"
            mc = month_counts.get((y, m), 0)
            sec = _section_key(y, m)
            month_pills.append(
                f'<button class="month-pill" data-section="{sec}" data-year="{y}" '
                f'data-month="{m}" title="{m_full} {y_label} ({mc} photos)">'
                f'<span class="month-pill-check">'
                '<svg viewBox="0 0 24 24" width="10" height="10" aria-hidden="true">'
                '<path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z" fill="currentColor"/></svg>'
                f'</span>{m_short}</button>'
            )

        blocks.append(
            f'<div class="year-pill-group" data-year="{y}">'
            f'<div class="year-pill" data-year="{y}" title="{y_label} ({y_count} photos)">'
            f'<button class="year-pill-check-btn" data-year="{y}" aria-label="Toggle all months for {y_label}">'
            f'<span class="year-pill-check">'
            '<svg viewBox="0 0 24 24" width="11" height="11" aria-hidden="true" class="check-svg">'
            '<path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z" fill="currentColor"/></svg>'
            '<svg viewBox="0 0 24 24" width="11" height="11" aria-hidden="true" class="dash-svg">'
            '<path d="M5 11h14v2H5z" fill="currentColor"/></svg>'
            f'</span></button>'
            f'<button class="year-pill-label" data-year="{y}">{y_label}'
            f' <span class="year-pill-caret">&#9662;</span></button>'
            f'</div>'
            f'<div class="month-pills-wrap" data-year="{y}">'
            f'<div class="month-pills">{"".join(month_pills)}</div>'
            f'</div>'
            f'</div>'
        )
    return "".join(blocks)


def _identity_chips_html(
    clusters_list: list[dict],
    frames_with_unknown: set[str],
    identity_index: dict[str, dict],
) -> str:
    chips: list[str] = []
    sorted_clusters = sorted(
        (c for c in clusters_list if isinstance(c, dict) and isinstance(c.get("identity"), str)),
        key=lambda c: int(c.get("member_count", 0) or 0),
        reverse=True,
    )
    for c in sorted_clusters:
        name = c["identity"]
        info = identity_index.get(name, {})
        label = info.get("display_name", name)
        count = int(c.get("member_count", 0) or 0)
        chips.append(
            f'<button class="identity-chip" data-identity="{html.escape(name)}" '
            f'title="{html.escape(label)} - {count} photos">'
            f'<span class="identity-name">{html.escape(label)}</span>'
            f'<span class="identity-count">{count}</span></button>'
        )
    if frames_with_unknown:
        chips.append(
            f'<button class="identity-chip" data-identity="{UNKNOWN_CHIP_ID}" '
            f'title="Photos with unclustered faces - {len(frames_with_unknown)} photos">'
            f'<span class="identity-name">Unknown</span>'
            f'<span class="identity-count">{len(frames_with_unknown)}</span></button>'
        )
    return "".join(chips)


# ---------------------------------------------------------------------------
# CSS / JS (template strings)
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

:root {
  --bg: #ffffff;
  --surface: #f8f8f8;
  --border: #e0e0e0;
  --text-primary: #202124;
  --text-secondary: #5f6368;
  --accent: #1a73e8;
  --accent-hover: #1765cc;
  --accent-light: #e8f0fe;
  --selected-ring: #1a73e8;
  --year-header: #202124;
  --chip-bg: #f1f3f4;
  --chip-selected: #d2e3fc;
  --chip-selected-text: #1a73e8;
  --shadow: 0 1px 2px rgba(60,64,67,0.10), 0 2px 6px rgba(60,64,67,0.08);
}

* { box-sizing: border-box; }

html, body {
  margin: 0;
  padding: 0;
  background: var(--bg);
  color: var(--text-primary);
  font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 14px;
}

/* ---------- Header ---------- */

header {
  position: sticky;
  top: 0;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  z-index: 50;
}
.header-row {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px 24px;
  min-height: 56px;
}
.title {
  font-weight: 500;
  font-size: 16px;
  color: var(--text-primary);
  flex-shrink: 0;
  margin-right: 8px;
}
.year-pills {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: 6px 8px;
  flex: 1 1 auto;
  min-width: 0;
}
.year-pill-group {
  display: inline-flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
  flex-shrink: 0;
}
.year-pill {
  display: inline-flex;
  align-items: stretch;
  background: var(--chip-bg);
  border-radius: 14px;
  font: 500 12px/1 'DM Sans', sans-serif;
  color: var(--text-secondary);
  overflow: hidden;
  transition: background 0.12s, color 0.12s;
}
.year-pill:hover { background: #e8eaed; }
.year-pill.checked {
  background: var(--chip-selected);
  color: var(--chip-selected-text);
}
.year-pill-check-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: none;
  color: inherit;
  padding: 4px 4px 4px 8px;
  cursor: pointer;
  font: inherit;
}
.year-pill-label {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  background: transparent;
  border: none;
  color: inherit;
  padding: 4px 10px 4px 4px;
  cursor: pointer;
  font: inherit;
  white-space: nowrap;
}
.year-pill-caret {
  font-size: 9px;
  opacity: 0.6;
  transition: transform 0.18s ease;
  display: inline-block;
}
.year-pill-group.expanded .year-pill-caret { transform: rotate(180deg); }
.year-pill .year-pill-check {
  width: 13px;
  height: 13px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  border: 1.5px solid currentColor;
  flex-shrink: 0;
  color: var(--text-secondary);
}
.year-pill .year-pill-check svg { display: none; }
.year-pill.checked .year-pill-check {
  background: var(--accent);
  border-color: var(--accent);
  color: white;
}
.year-pill.checked .year-pill-check .check-svg { display: block; }
.year-pill.indeterminate .year-pill-check {
  background: var(--accent);
  border-color: var(--accent);
  color: white;
}
.year-pill.indeterminate .year-pill-check .dash-svg { display: block; }

/* Month pills row: animated via max-height, hidden by default. */
.month-pills-wrap {
  overflow: hidden;
  max-height: 0;
  transition: max-height 0.18s ease, opacity 0.18s ease;
  opacity: 0;
}
.year-pill-group.expanded .month-pills-wrap {
  max-height: 60px;
  opacity: 1;
}
.month-pills {
  display: inline-flex;
  flex-wrap: wrap;
  gap: 4px;
  padding-top: 2px;
}
.month-pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: var(--chip-bg);
  border: none;
  color: var(--text-secondary);
  padding: 3px 9px 3px 6px;
  border-radius: 11px;
  font: 500 11px/1 'DM Sans', sans-serif;
  cursor: pointer;
  transition: background 0.12s, color 0.12s;
}
.month-pill:hover { background: #e8eaed; }
.month-pill.checked {
  background: var(--chip-selected);
  color: var(--chip-selected-text);
}
.month-pill .month-pill-check {
  width: 11px;
  height: 11px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  border: 1.5px solid currentColor;
  flex-shrink: 0;
  color: var(--text-secondary);
}
.month-pill .month-pill-check svg { display: none; }
.month-pill.checked .month-pill-check {
  background: var(--accent);
  border-color: var(--accent);
  color: white;
}
.month-pill.checked .month-pill-check svg { display: block; }

.header-controls {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-left: auto;
  flex-shrink: 0;
}
.dropdown-trigger {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-primary);
  padding: 6px 12px;
  border-radius: 18px;
  font: 500 13px/1 'DM Sans', sans-serif;
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s;
}
.dropdown-trigger:hover { background: var(--surface); }
.dropdown-trigger.open { background: var(--surface); border-color: var(--text-secondary); }
.dropdown-trigger .caret {
  font-size: 10px;
  margin-left: 2px;
  color: var(--text-secondary);
}
.dropdown-trigger.icon-only {
  width: 36px;
  height: 36px;
  padding: 0;
  justify-content: center;
  border-radius: 50%;
}

#export-btn {
  background: var(--chip-bg);
  border: none;
  color: var(--text-secondary);
  padding: 8px 16px;
  border-radius: 20px;
  font: 500 13px/1 'DM Sans', sans-serif;
  cursor: pointer;
  transition: background 0.12s, color 0.12s;
  flex-shrink: 0;
}
#export-btn.active {
  background: var(--accent);
  color: white;
}
#export-btn.active:hover { background: var(--accent-hover); }
#export-btn:not(.active):hover { background: #e8eaed; color: var(--text-primary); }

/* ---------- Dropdown panels ---------- */

.dropdown-panel {
  display: none;
  position: absolute;
  right: 24px;
  top: 100%;
  margin-top: 6px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  box-shadow: var(--shadow);
  padding: 16px;
  min-width: 260px;
  max-width: 460px;
  max-height: 70vh;
  overflow-y: auto;
  z-index: 60;
}
.dropdown-panel.open { display: block; }
.dropdown-section + .dropdown-section { margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--border); }
.dropdown-label {
  display: block;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-secondary);
  margin-bottom: 8px;
}

/* Layout radio toggles */
.layout-radios { display: flex; flex-direction: column; gap: 6px; }
.layout-radio {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 10px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.12s;
}
.layout-radio:hover { background: var(--surface); }
.layout-radio input[type="radio"] { margin: 0; cursor: pointer; }
.layout-radio .layout-name { flex: 1; font-size: 13px; }
.layout-radio .layout-hint { color: var(--text-secondary); font-size: 11px; }

/* Quality toggles */
.quality-toggles { display: flex; flex-direction: column; gap: 6px; }
.quality-toggle {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 10px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.12s;
}
.quality-toggle:hover { background: var(--surface); }
.quality-toggle input[type="checkbox"] { margin: 0; cursor: pointer; }
.quality-toggle .quality-name { flex: 1; font-size: 13px; }
.quality-toggle .quality-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

/* People panel */
.mode-toggle { display: inline-flex; gap: 4px; }
.mode-btn {
  background: var(--chip-bg);
  border: none;
  color: var(--text-secondary);
  padding: 4px 12px;
  border-radius: 12px;
  font: 500 12px/1 'DM Sans', sans-serif;
  cursor: pointer;
}
.mode-btn.active {
  background: var(--accent);
  color: white;
}
.identity-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 6px;
}
.identity-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--chip-bg);
  border: none;
  color: var(--text-primary);
  padding: 4px 10px;
  border-radius: 14px;
  font: 500 12px/1 'DM Sans', sans-serif;
  cursor: pointer;
  transition: background 0.12s;
}
.identity-chip:hover { background: #e8eaed; }
.identity-chip.active {
  background: var(--chip-selected);
  color: var(--chip-selected-text);
}
.identity-chip .identity-count {
  color: var(--text-secondary);
  font-size: 11px;
  font-weight: 400;
}
.identity-chip.active .identity-count { color: var(--chip-selected-text); }

/* Settings */
.settings-toggle {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 10px;
  border-radius: 6px;
  cursor: pointer;
}
.settings-toggle:hover { background: var(--surface); }
.settings-toggle input[type="checkbox"] { margin: 0; cursor: pointer; }
.settings-toggle .toggle-label { flex: 1; font-size: 13px; }
.settings-toggle .toggle-hint { color: var(--text-secondary); font-size: 11px; }

/* ---------- Year sections ---------- */

main { padding: 8px 24px 40px 24px; }
.year-section { margin-bottom: 4px; }
.year-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 18px 0 10px 0;
  font-size: 15px;
  font-weight: 500;
  color: var(--year-header);
  cursor: pointer;
  user-select: none;
}
.year-arrow {
  display: inline-block;
  color: var(--text-secondary);
  font-size: 10px;
  width: 14px;
  text-align: center;
  transition: transform 0.18s ease;
}
.year-section.collapsed .year-arrow { transform: rotate(-90deg); }
.year-label { font-weight: 500; }
.year-stats {
  font-size: 13px;
  font-weight: 400;
  color: var(--text-secondary);
}
.year-stats .selected-count {
  color: var(--accent);
  margin-left: 4px;
}

.year-grid-wrap {
  overflow: hidden;
  transition: max-height 0.2s ease;
  max-height: 100000px;
}
.year-section.collapsed .year-grid-wrap {
  max-height: 0;
}

/* Crop layout (default): square grid */
body.layout-crop .section-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 4px;
}

/* Justified layout: rows of flex containers (heights set inline by JS) */
body.layout-justified .section-grid {
  display: block;
}
body.layout-justified .j-row {
  display: flex;
  gap: 4px;
  margin-bottom: 4px;
}
body.layout-justified .j-row:last-child { margin-bottom: 0; }
body.layout-justified .j-row .photo-card {
  aspect-ratio: auto;
  flex-shrink: 0;
}

/* ---------- Photo cards ---------- */

.photo-card {
  position: relative;
  aspect-ratio: 1 / 1;
  background: #f1f3f4;
  cursor: pointer;
  overflow: hidden;
  outline: 3px solid transparent;
  outline-offset: -3px;
  transition: outline-color 0.1s;
}
.photo-card img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.photo-card:hover::before {
  content: "";
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.08);
  pointer-events: none;
  z-index: 1;
}
.photo-card.selected { outline-color: var(--selected-ring); }

/* Checkbox (top-left) */
.card-check {
  position: absolute;
  top: 8px;
  left: 8px;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  border: 2px solid white;
  background: transparent;
  display: none;
  align-items: center;
  justify-content: center;
  padding: 0;
  cursor: pointer;
  color: transparent;
  filter: drop-shadow(0 0 1px rgba(0,0,0,0.4));
  z-index: 3;
}
.card-check svg { width: 12px; height: 12px; }
.photo-card:hover .card-check,
.photo-card.selected .card-check,
body.selection-mode .card-check {
  display: inline-flex;
}
.photo-card.selected .card-check {
  background: var(--accent);
  border-color: white;
  color: white;
}

/* Card remove ✕ (export overlay only) */
.card-remove {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: rgba(0,0,0,0.55);
  color: white;
  border: none;
  display: none;
  align-items: center;
  justify-content: center;
  padding: 0;
  cursor: pointer;
  z-index: 3;
}
.card-remove svg { width: 12px; height: 12px; }
#export-grid .card-remove { display: inline-flex; }
#export-grid .card-check { display: none !important; }

/* Video badge */
.video-badge {
  position: absolute;
  top: 6px;
  right: 6px;
  width: 16px;
  height: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  pointer-events: none;
  z-index: 2;
  opacity: 0.85;
  filter: drop-shadow(0 1px 2px rgba(0,0,0,0.55));
}
#export-grid .video-badge { display: none !important; }

/* ---------- Lightbox ---------- */

#lightbox {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.94);
  z-index: 1000;
  flex-direction: row;
}
#lightbox.open { display: flex; }
.lightbox-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  position: relative;
  min-width: 0;
}
.lightbox-image-area {
  flex: 1;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  padding: 24px;
}
#lightbox-img {
  display: block;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}
#lightbox-canvas {
  position: absolute;
  pointer-events: none;
  display: none;
}
#lightbox.show-faces #lightbox-canvas { display: block; }
.lightbox-bottom {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 24px 20px 24px;
  color: #ddd;
  font-size: 13px;
}
#lightbox-meta {
  flex: 1;
  color: #d0d0d0;
  word-break: break-all;
}
#lightbox-select-btn {
  background: var(--accent);
  color: white;
  border: none;
  padding: 8px 18px;
  border-radius: 20px;
  font: 500 13px/1 'DM Sans', sans-serif;
  cursor: pointer;
}
#lightbox-select-btn:not(.selected) {
  background: transparent;
  border: 1px solid #888;
  color: #ddd;
}
#lightbox-select-btn.export-mode {
  background: transparent;
  border: 1px solid #ef4444;
  color: #ef4444;
}
.lightbox-nav, .lightbox-close {
  position: absolute;
  background: rgba(60,60,60,0.5);
  border: none;
  color: white;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 5;
  font-size: 18px;
}
.lightbox-nav:hover, .lightbox-close:hover { background: rgba(100,100,100,0.7); }
.lightbox-nav.prev { left: 16px; top: 50%; transform: translateY(-50%); }
.lightbox-nav.next { right: 16px; top: 50%; transform: translateY(-50%); }
.lightbox-close { top: 16px; right: 16px; }

#lightbox.show-debug .lightbox-nav.next { right: calc(360px + 16px); }

.lightbox-side {
  display: none;
  width: 360px;
  flex-shrink: 0;
  background: #15171c;
  color: #ddd;
  padding: 24px;
  overflow-y: auto;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: 13px;
  line-height: 1.55;
}
#lightbox.show-debug .lightbox-side { display: block; }
#lightbox-scores .section { margin-bottom: 1.2rem; }
#lightbox-scores .section:last-child { margin-bottom: 0; }
#lightbox-scores .label {
  color: #6b7280;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 4px;
}
#lightbox-scores .value { color: #f3f4f6; }
#lightbox-scores .big {
  font-size: 24px;
  font-weight: 600;
  color: #f9fafb;
}
#lightbox-scores .small { color: #9ca3af; font-size: 12px; }
#lightbox-scores .source-name { color: #f3f4f6; word-break: break-all; }

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
.face-row .ft-cell { padding: 3px 2px; }
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

/* ---------- Export overlay ---------- */

#export-overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: var(--bg);
  z-index: 900;
  flex-direction: column;
}
#export-overlay.open { display: flex; }
.export-header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 14px 24px;
  border-bottom: 1px solid var(--border);
  background: var(--bg);
  position: sticky;
  top: 0;
  z-index: 1;
}
.export-title {
  flex: 1;
  font-size: 16px;
  font-weight: 500;
  color: var(--text-primary);
}
.export-actions { display: flex; gap: 8px; }
.export-btn {
  background: var(--chip-bg);
  border: none;
  color: var(--text-primary);
  padding: 8px 16px;
  border-radius: 18px;
  font: 500 13px/1 'DM Sans', sans-serif;
  cursor: pointer;
}
.export-btn.primary { background: var(--accent); color: white; }
.export-btn.primary:hover { background: var(--accent-hover); }
.export-btn:not(.primary):hover { background: #e8eaed; }
#export-grid {
  flex: 1;
  overflow-y: auto;
  padding: 16px 24px 40px 24px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 4px;
  align-content: start;
}
#export-grid .photo-card { outline-color: var(--selected-ring); cursor: pointer; }
.export-empty {
  grid-column: 1 / -1;
  text-align: center;
  color: var(--text-secondary);
  padding: 80px 16px;
}
"""


JS_TEMPLATE = """
const FRAMES_DATA = __FRAMES_DATA__;
const IDENTITY_PORTRAITS = __IDENTITY_PORTRAITS__;

const KEY_SELECTED = 'se_selected';
const KEY_YEARS_COLLAPSED = 'se_years_collapsed';        // legacy
const KEY_SECTIONS_COLLAPSED = 'se_sections_collapsed';  // new
const KEY_YEARS_EXPANDED = 'se_years_expanded';
const KEY_QUALITY_FILTER = 'se_quality_filter';
const KEY_SOURCE_FILTER = 'se_source_filter';
const KEY_PEOPLE_FILTER = 'se_people_filter';
const KEY_SHOW_DEBUG = 'se_debug_show_debug';
const KEY_SHOW_FACES = 'se_debug_show_faces';
const KEY_SHOW_REJECTED = 'se_debug_show_rejected';

const UNKNOWN_CHIP_ID = '__unknown__';

// Tunable constants for justified layout.
const TARGET_ROW_HEIGHT = 200;   // px
const CONTAINER_GAP = 4;         // px (matches crop gap)
const MIN_ROW_HEIGHT = 120;      // px (last row pinned to this if very short)
const MAX_ROW_HEIGHT = 350;      // px (cap stretched rows)
const HARD_MIN_ROW_HEIGHT = 80;  // px (absolute floor)
const RESIZE_DEBOUNCE_MS = 100;
const FILTER_RELAYOUT_DEBOUNCE_MS = 50;

const LABEL_COLORS = {
  good: '#0f9d58',
  okay: '#f4b400',
  bad:  '#db4437',
  none: '#9aa0a6',
};

let selected = new Set();
let collapsedSections = new Set();
let expandedYears = new Set();
let layout = 'justified';
let qualityFilter = { good: true, okay: false, bad: false, none: false };
let sourceFilter = 'all';
let peopleFilter = { mode: 'AND', identities: [] };
let debugFlags = { show_debug: false, show_faces: false };
let showRejected = false;

let lightboxOrder = [];
let lightboxIndex = -1;
let lightboxOrigin = 'main';

let highlightedFace = -1;
let hoveredAcceptedFace = -1;
let hoveredRejectedFace = -1;

// === Persistence ===
function loadState() {
  try {
    const s = JSON.parse(localStorage.getItem(KEY_SELECTED) || '[]');
    if (Array.isArray(s)) selected = new Set(s.filter(x => typeof x === 'string'));
  } catch (_) {}

  // Sections-collapsed (with migration from legacy years-collapsed).
  let loadedSections = false;
  try {
    const raw = localStorage.getItem(KEY_SECTIONS_COLLAPSED);
    if (raw != null) {
      const c = JSON.parse(raw);
      if (Array.isArray(c)) {
        collapsedSections = new Set(c.filter(x => typeof x === 'string'));
        loadedSections = true;
      }
    }
  } catch (_) {}
  if (!loadedSections) {
    try {
      const legacy = JSON.parse(localStorage.getItem(KEY_YEARS_COLLAPSED) || '[]');
      if (Array.isArray(legacy) && legacy.length > 0) {
        const legacyYears = new Set(legacy.map(String));
        document.querySelectorAll('.year-section').forEach(section => {
          if (legacyYears.has(String(section.dataset.year))) {
            collapsedSections.add(section.dataset.section);
          }
        });
        localStorage.setItem(KEY_SECTIONS_COLLAPSED, JSON.stringify(Array.from(collapsedSections)));
      }
    } catch (_) {}
  }

  try {
    const e = JSON.parse(localStorage.getItem(KEY_YEARS_EXPANDED) || '[]');
    if (Array.isArray(e)) expandedYears = new Set(e.map(x => parseInt(x, 10)).filter(n => !isNaN(n)));
  } catch (_) {}

  const sf = localStorage.getItem(KEY_SOURCE_FILTER);
  if (sf === 'images' || sf === 'videos') sourceFilter = sf;
  else sourceFilter = 'all';

  try {
    const q = JSON.parse(localStorage.getItem(KEY_QUALITY_FILTER) || 'null');
    if (q && typeof q === 'object') {
      ['good', 'okay', 'bad', 'none'].forEach(k => {
        if (typeof q[k] === 'boolean') qualityFilter[k] = q[k];
      });
    }
  } catch (_) {}
  try {
    const p = JSON.parse(localStorage.getItem(KEY_PEOPLE_FILTER) || 'null');
    if (p && typeof p === 'object') {
      peopleFilter.mode = p.mode === 'OR' ? 'OR' : 'AND';
      peopleFilter.identities = Array.isArray(p.identities)
        ? p.identities.filter(x => typeof x === 'string') : [];
    }
  } catch (_) {}
  debugFlags.show_debug = localStorage.getItem(KEY_SHOW_DEBUG) === 'true';
  debugFlags.show_faces = localStorage.getItem(KEY_SHOW_FACES) === 'true';
  showRejected = localStorage.getItem(KEY_SHOW_REJECTED) === 'true';
}

function saveSelected() { localStorage.setItem(KEY_SELECTED, JSON.stringify(Array.from(selected))); }
function saveCollapsedSections() { localStorage.setItem(KEY_SECTIONS_COLLAPSED, JSON.stringify(Array.from(collapsedSections))); }
function saveExpandedYears() { localStorage.setItem(KEY_YEARS_EXPANDED, JSON.stringify(Array.from(expandedYears))); }
function saveQualityFilter() { localStorage.setItem(KEY_QUALITY_FILTER, JSON.stringify(qualityFilter)); }
function saveSourceFilter() { localStorage.setItem(KEY_SOURCE_FILTER, sourceFilter); }
function savePeopleFilter() { localStorage.setItem(KEY_PEOPLE_FILTER, JSON.stringify(peopleFilter)); }
function saveDebugFlags() {
  localStorage.setItem(KEY_SHOW_DEBUG, String(debugFlags.show_debug));
  localStorage.setItem(KEY_SHOW_FACES, String(debugFlags.show_faces));
}

// === Card helpers ===
function cardIdentities(card) {
  const raw = card.dataset.identities || '';
  if (!raw) return [];
  return raw.split('|').filter(Boolean);
}

function passesQuality(card) {
  const q = (card.dataset.quality || 'none').toLowerCase();
  if (q in qualityFilter) return !!qualityFilter[q];
  return !!qualityFilter.none;
}

function passesPeople(card) {
  const sels = peopleFilter.identities;
  if (sels.length === 0) return true;
  const ids = new Set(cardIdentities(card));
  const hasUnknown = card.dataset.hasUnknown === 'true';
  const has = (sel) => sel === UNKNOWN_CHIP_ID ? hasUnknown : ids.has(sel);
  return peopleFilter.mode === 'AND' ? sels.every(has) : sels.some(has);
}

function passesSource(card) {
  if (sourceFilter === 'all') return true;
  const t = card.dataset.sourceType;
  if (sourceFilter === 'images') return t === 'image';
  if (sourceFilter === 'videos') return t === 'video';
  return true;
}

function passesFilter(card) { return passesQuality(card) && passesPeople(card) && passesSource(card); }

// === Apply filters & counts ===
function applyFilters() {
  document.querySelectorAll('.year-section').forEach(section => {
    let shown = 0;
    let sel = 0;
    section.querySelectorAll('.photo-card').forEach(card => {
      const visible = passesFilter(card);
      card.style.display = visible ? '' : 'none';
      if (visible) {
        shown++;
        if (selected.has(card.dataset.cardKey)) sel++;
      }
    });
    const photoCount = section.querySelector('.photo-count');
    const selCount = section.querySelector('.selected-count');
    if (photoCount) photoCount.textContent = shown + ' photos';
    if (selCount) {
      if (sel > 0) { selCount.textContent = '· ' + sel + ' selected'; selCount.style.display = ''; }
      else { selCount.textContent = ''; selCount.style.display = 'none'; }
    }
    section.style.display = shown === 0 ? 'none' : '';
  });
  updateExportButton();
  scheduleJustifiedRelayout();
}

// === Layout (justified only; crop CSS retained as dead code) ===
function applyLayout() {
  document.body.classList.add('layout-justified');
  layoutAllVisibleJustified();
}

function unwrapAllJustifiedRows() {
  // Restore the flat .photo-card children of .section-grid (remove .j-row wrappers).
  document.querySelectorAll('.section-grid').forEach(grid => {
    const rows = grid.querySelectorAll(':scope > .j-row');
    if (rows.length === 0) return;
    const cards = [];
    rows.forEach(row => {
      row.querySelectorAll(':scope > .photo-card').forEach(c => {
        c.style.width = '';
        c.style.height = '';
        cards.push(c);
      });
      row.remove();
    });
    cards.forEach(c => grid.appendChild(c));
  });
}

function layoutAllVisibleJustified() {
  document.querySelectorAll('.year-section').forEach(section => {
    if (collapsedSections.has(section.dataset.section)) return;
    if (section.style.display === 'none') return;
    const grid = section.querySelector(':scope > .year-grid-wrap > .section-grid');
    if (!grid) return;
    justifySection(grid);
  });
}

function visibleCardsInGrid(grid) {
  // Collect direct child .photo-cards plus those inside any existing .j-row children.
  const out = [];
  grid.querySelectorAll(':scope > .photo-card, :scope > .j-row > .photo-card').forEach(c => {
    if (c.style.display !== 'none') out.push(c);
  });
  return out;
}

function cardAspectRatio(card) {
  const frame = currentFrame(card);
  if (frame && frame.w && frame.h && frame.w > 0 && frame.h > 0) {
    return frame.w / frame.h;
  }
  return 4 / 3;
}

/**
 * Lay out a single section in justified rows.
 * Greedy packing: accumulate cards until adding the next would exceed
 * the container width at TARGET_ROW_HEIGHT, then scale that row to fit.
 * Last row is left-aligned at TARGET_ROW_HEIGHT (not stretched).
 */
function justifySection(grid) {
  const cards = visibleCardsInGrid(grid);
  // Tear down any previous row wrappers; we will rebuild.
  grid.querySelectorAll(':scope > .j-row').forEach(r => {
    while (r.firstChild) grid.appendChild(r.firstChild);
    r.remove();
  });

  if (cards.length === 0) return;
  const containerW = grid.clientWidth;
  if (!containerW || containerW <= 0) return;

  // Greedy pack: build rows whose natural width at TARGET exceeds container.
  const rows = [];
  let currentRow = [];
  let currentSumAr = 0;
  for (const card of cards) {
    const ar = cardAspectRatio(card);
    const candidateRow = [...currentRow, card];
    const candidateSumAr = currentSumAr + ar;
    const candidateNaturalW = candidateSumAr * TARGET_ROW_HEIGHT + (candidateRow.length - 1) * CONTAINER_GAP;
    if (candidateNaturalW > containerW && currentRow.length > 0) {
      rows.push({ cards: currentRow, sumAr: currentSumAr });
      currentRow = [card];
      currentSumAr = ar;
    } else {
      currentRow = candidateRow;
      currentSumAr = candidateSumAr;
    }
  }
  if (currentRow.length > 0) {
    rows.push({ cards: currentRow, sumAr: currentSumAr, isLast: true });
  }

  // Build DOM: one .j-row per row.
  const frag = document.createDocumentFragment();
  rows.forEach((row, rowIdx) => {
    const gapTotal = (row.cards.length - 1) * CONTAINER_GAP;
    let h;
    if (row.isLast) {
      h = TARGET_ROW_HEIGHT;
    } else {
      h = (containerW - gapTotal) / row.sumAr;
      h = Math.max(HARD_MIN_ROW_HEIGHT, Math.min(MAX_ROW_HEIGHT, h));
    }
    if (row.isLast && row.cards.length <= 2) {
      // Cap last-row width to 50% of container if only 1-2 cards.
      const maxRowW = containerW * 0.5;
      const naturalW = row.sumAr * h + gapTotal;
      if (naturalW > maxRowW) h = (maxRowW - gapTotal) / row.sumAr;
    }
    const rowEl = document.createElement('div');
    rowEl.className = 'j-row';
    rowEl.style.height = Math.round(h) + 'px';
    row.cards.forEach(card => {
      const ar = cardAspectRatio(card);
      const w = ar * h;
      card.style.width = Math.round(w) + 'px';
      card.style.height = Math.round(h) + 'px';
      rowEl.appendChild(card);
    });
    frag.appendChild(rowEl);
  });
  grid.appendChild(frag);
}

let justifiedRelayoutTimer = null;
function scheduleJustifiedRelayout() {
  if (layout !== 'justified') return;
  if (justifiedRelayoutTimer) clearTimeout(justifiedRelayoutTimer);
  justifiedRelayoutTimer = setTimeout(() => {
    justifiedRelayoutTimer = null;
    layoutAllVisibleJustified();
  }, FILTER_RELAYOUT_DEBOUNCE_MS);
}

let resizeRelayoutTimer = null;
function scheduleJustifiedResize() {
  if (layout !== 'justified') return;
  if (resizeRelayoutTimer) clearTimeout(resizeRelayoutTimer);
  resizeRelayoutTimer = setTimeout(() => {
    resizeRelayoutTimer = null;
    layoutAllVisibleJustified();
  }, RESIZE_DEBOUNCE_MS);
}

function applySectionCollapse() {
  document.querySelectorAll('.year-section').forEach(section => {
    const sec = section.dataset.section;
    const collapsed = collapsedSections.has(sec);
    section.classList.toggle('collapsed', collapsed);
  });
  applyMonthPillStates();
  applyYearPillStates();
}

function applyMonthPillStates() {
  document.querySelectorAll('.month-pill').forEach(pill => {
    const sec = pill.dataset.section;
    pill.classList.toggle('checked', !collapsedSections.has(sec));
  });
}

function applyYearPillStates() {
  document.querySelectorAll('.year-pill-group').forEach(group => {
    const year = group.dataset.year;
    const sections = Array.from(document.querySelectorAll('.year-section[data-year="' + year + '"]'));
    let totalSec = sections.length;
    let visibleSec = 0;
    sections.forEach(s => { if (!collapsedSections.has(s.dataset.section)) visibleSec++; });
    const pill = group.querySelector('.year-pill');
    if (!pill) return;
    pill.classList.remove('checked', 'indeterminate');
    if (totalSec === 0) return;
    if (visibleSec === totalSec) pill.classList.add('checked');
    else if (visibleSec > 0) pill.classList.add('indeterminate');
    // 0 visible -> neither class (unchecked)
    group.classList.toggle('expanded', expandedYears.has(parseInt(year, 10)));
  });
}

function toggleSection(sec) {
  if (collapsedSections.has(sec)) collapsedSections.delete(sec);
  else collapsedSections.add(sec);
  saveCollapsedSections();
  applySectionCollapse();
  scheduleJustifiedRelayout();
}

function toggleYearAllMonths(year) {
  const sections = Array.from(document.querySelectorAll('.year-section[data-year="' + year + '"]'));
  let anyVisible = false;
  sections.forEach(s => { if (!collapsedSections.has(s.dataset.section)) anyVisible = true; });
  if (anyVisible) {
    // Collapse all months for this year.
    sections.forEach(s => collapsedSections.add(s.dataset.section));
  } else {
    // Expand all months for this year.
    sections.forEach(s => collapsedSections.delete(s.dataset.section));
  }
  saveCollapsedSections();
  applySectionCollapse();
  scheduleJustifiedRelayout();
}

function toggleYearExpanded(year) {
  const y = parseInt(year, 10);
  if (isNaN(y)) return;
  if (expandedYears.has(y)) expandedYears.delete(y);
  else expandedYears.add(y);
  saveExpandedYears();
  applyYearPillStates();
}

// === Selection ===
function setSelected(card, on) {
  const key = card.dataset.cardKey;
  if (on) selected.add(key); else selected.delete(key);
  card.classList.toggle('selected', on);
  saveSelected();
}

function toggleSelected(card) { setSelected(card, !selected.has(card.dataset.cardKey)); }

function applySelectionState() {
  document.querySelectorAll('.photo-card').forEach(card => {
    card.classList.toggle('selected', selected.has(card.dataset.cardKey));
  });
  document.body.classList.toggle('selection-mode', selected.size > 0);
}

function updateExportButton() {
  const n = selected.size;
  const btn = document.getElementById('export-btn');
  btn.textContent = 'Export (' + n + ')';
  btn.classList.toggle('active', n > 0);
}

// === Shift-click range ===
let lastClickedCard = null;
function rangeSelect(fromCard, toCard) {
  if (!fromCard || !toCard) return;
  const fromYear = fromCard.closest('.year-section');
  const toYear = toCard.closest('.year-section');
  if (!fromYear || fromYear !== toYear) {
    setSelected(toCard, true);
    return;
  }
  const cards = Array.from(fromYear.querySelectorAll('.photo-card')).filter(c => c.style.display !== 'none');
  const a = cards.indexOf(fromCard);
  const b = cards.indexOf(toCard);
  if (a < 0 || b < 0) { setSelected(toCard, true); return; }
  const [lo, hi] = a < b ? [a, b] : [b, a];
  for (let i = lo; i <= hi; i++) setSelected(cards[i], true);
}

// === Dropdowns ===
function closeAllDropdowns() {
  document.querySelectorAll('.dropdown-panel').forEach(p => p.classList.remove('open'));
  document.querySelectorAll('.dropdown-trigger').forEach(t => t.classList.remove('open'));
}

function toggleDropdown(name) {
  const panel = document.getElementById('dropdown-' + name);
  const trigger = document.querySelector('.dropdown-trigger[data-dropdown="' + name + '"]');
  if (!panel) return;
  const isOpen = panel.classList.contains('open');
  closeAllDropdowns();
  if (!isOpen) {
    panel.classList.add('open');
    if (trigger) trigger.classList.add('open');
  }
}

// === Settings / debug ===
function applyDebugFlags() {
  ['show_debug', 'show_faces'].forEach(k => {
    const cb = document.getElementById('toggle-' + k);
    if (cb) cb.checked = !!debugFlags[k];
  });
}

// === Lightbox ===
function currentFrame(card) {
  const i = parseInt(card.dataset.frameIdx, 10);
  if (isNaN(i)) return null;
  return FRAMES_DATA[i] || null;
}

function openLightbox(card, origin) {
  lightboxOrigin = origin || 'main';
  if (lightboxOrigin === 'export') {
    lightboxOrder = Array.from(document.querySelectorAll('#export-grid .photo-card'));
  } else {
    lightboxOrder = visibleCards();
  }
  lightboxIndex = lightboxOrder.indexOf(card);
  if (lightboxIndex < 0) return;
  highlightedFace = -1;
  hoveredAcceptedFace = -1;
  hoveredRejectedFace = -1;
  hideTooltip();
  renderLightbox();
  document.getElementById('lightbox').classList.add('open');
}

function closeLightbox() {
  const lb = document.getElementById('lightbox');
  lb.classList.remove('open');
  document.getElementById('lightbox-img').src = '';
  lightboxIndex = -1;
  hideTooltip();
  const c = document.getElementById('lightbox-canvas');
  if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height);
}

function lightboxNav(dir) {
  if (lightboxIndex < 0) return;
  const next = lightboxIndex + dir;
  if (next < 0 || next >= lightboxOrder.length) return;
  lightboxIndex = next;
  highlightedFace = -1;
  hoveredAcceptedFace = -1;
  hoveredRejectedFace = -1;
  hideTooltip();
  renderLightbox();
}

function renderLightbox() {
  const card = lightboxOrder[lightboxIndex];
  if (!card) return;
  const frame = currentFrame(card);
  const lb = document.getElementById('lightbox');
  const img = document.getElementById('lightbox-img');
  const srcImg = card.querySelector('img');
  img.onload = () => { if (debugFlags.show_faces) drawFaceOverlay(); };
  img.src = srcImg.src;
  const deg = parseInt(card.dataset.rotation || '0', 10);
  img.style.transform = deg ? 'rotate(' + deg + 'deg)' : '';

  lb.classList.toggle('show-debug', debugFlags.show_debug);
  lb.classList.toggle('show-faces', debugFlags.show_faces && !!frame);
  lb.classList.toggle('export-mode', lightboxOrigin === 'export');

  const selBtn = document.getElementById('lightbox-select-btn');
  const sel = selected.has(card.dataset.cardKey);
  selBtn.classList.toggle('selected', sel);
  if (lightboxOrigin === 'export') {
    selBtn.textContent = 'Remove from Export';
    selBtn.classList.add('export-mode');
  } else {
    selBtn.classList.remove('export-mode');
    selBtn.textContent = sel ? '✓ Selected for Export' : 'Select for Export';
  }

  const stem = (frame && frame.video_basename) || card.dataset.videoStem || '';
  const year = card.dataset.year;
  const parts = [];
  if (stem) parts.push(stem);
  if (year && year !== '0') parts.push(year);
  if (frame && frame.source_type === 'video' && frame.refined_timestamp_s != null) {
    parts.push(fmtNum(frame.refined_timestamp_s, 2) + 's');
  }
  document.getElementById('lightbox-meta').textContent = parts.join(' · ');

  if (debugFlags.show_debug && frame) {
    document.getElementById('lightbox-scores').innerHTML = buildScoresHtml(frame);
    wireScoresPanel(frame);
  } else {
    document.getElementById('lightbox-scores').innerHTML = '';
  }

  if (debugFlags.show_faces && frame && img.complete && img.naturalWidth > 0) {
    drawFaceOverlay();
  } else if (!debugFlags.show_faces) {
    const c = document.getElementById('lightbox-canvas');
    if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height);
  }
}

function lightboxToggleSelection() {
  const card = lightboxOrder[lightboxIndex];
  if (!card) return;
  if (lightboxOrigin === 'export') {
    setSelected(card, false);
    applyFilters();
    refreshExportGrid();
    // Re-locate current card in updated lightboxOrder; close if gone
    lightboxOrder = Array.from(document.querySelectorAll('#export-grid .photo-card'));
    if (lightboxOrder.length === 0) { closeLightbox(); closeExportOverlay(); return; }
    if (lightboxIndex >= lightboxOrder.length) lightboxIndex = lightboxOrder.length - 1;
    renderLightbox();
  } else {
    toggleSelected(card);
    applyFilters();
    applySelectionState();
    renderLightbox();
  }
}

// === Export overlay ===
function openExportOverlay() {
  if (selected.size === 0) return;
  refreshExportGrid();
  document.getElementById('export-overlay').classList.add('open');
  document.getElementById('export-count').textContent = selected.size;
}

function closeExportOverlay() {
  document.getElementById('export-overlay').classList.remove('open');
  document.getElementById('export-grid').innerHTML = '';
}

function refreshExportGrid() {
  const grid = document.getElementById('export-grid');
  grid.innerHTML = '';
  document.getElementById('export-count').textContent = selected.size;
  if (selected.size === 0) {
    grid.innerHTML = '<div class="export-empty">No photos selected.</div>';
    return;
  }
  document.querySelectorAll('main .photo-card').forEach(card => {
    if (!selected.has(card.dataset.cardKey)) return;
    const clone = card.cloneNode(true);
    // Strip justified-layout inline sizing so the export grid uses square cells.
    clone.style.width = '';
    clone.style.height = '';
    clone.style.display = '';
    grid.appendChild(clone);
  });
}

function visibleCards() {
  return Array.from(document.querySelectorAll('main .photo-card')).filter(c => c.style.display !== 'none');
}

// === Debug overlay helpers ===
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
    out += '<div class="' + cls + '">'
        +  '<div class="name" style="color:' + color + '">' + label + '</div>'
        +  '<div class="track"><div class="fill" style="width:' + pct.toFixed(1) + '%;background:' + color + ';"></div></div>'
        +  '<div class="num">' + fmtNum(p, 2) + '</div>'
        +  '</div>';
  }
  return out;
}

function buildConfBar(conf, nearestOnly) {
  const segs = 10;
  const filled = Math.max(0, Math.min(segs, Math.round((conf || 0) * segs)));
  let h = '';
  for (let i = 0; i < segs; i++) {
    if (i >= filled) continue;
    const left = (i * (100 / segs)).toFixed(1);
    const width = (100 / segs - 1).toFixed(2);
    h += '<div class="seg" style="left:' + left + '%;width:' + width + '%;"></div>';
  }
  const cls = nearestOnly ? 'ft-confbar nearest-only' : 'ft-confbar';
  return '<div class="' + cls + '">' + h + '</div>';
}

function buildFaceTable(frame) {
  const faces = Array.isArray(frame.faces) ? frame.faces : [];
  const identities = Array.isArray(frame.face_identities) ? frame.face_identities : [];
  const anyFace = faces.some(f => f && f.x1 != null);
  if (!anyFace) return '';
  const hasIdentities = identities.some(i => i);

  let head = '<div class="face-table">'
    + '<div class="ft-head">#</div>'
    + '<div class="ft-head">Quality</div>'
    + '<div class="ft-head"></div>';
  if (hasIdentities) {
    head += '<div class="ft-head"></div>'
      + '<div class="ft-head">Identity</div>'
      + '<div class="ft-head"></div>'
      + '<div class="ft-head"></div>';
  } else {
    head += '<div class="ft-head" style="grid-column: span 4;"></div>';
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
      identCells = '<div class="ft-cell" style="grid-column: span 4;"></div>';
    } else if (ident) {
      const portraitUrl = IDENTITY_PORTRAITS[ident.identity];
      const portraitImg = portraitUrl
        ? '<img src="' + escHtml(portraitUrl) + '" alt="" onerror="this.style.display=\\'none\\'">'
        : '';
      const nearestOnly = !ident.assigned;
      const cls = nearestOnly ? 'ft-identity nearest-only' : 'ft-identity';
      const suffix = nearestOnly ? ' (nearest)' : '';
      const displayName = ident.display_name || ident.identity;
      identCells =
        '<div class="ft-cell ft-portrait">' + portraitImg + '</div>'
        + '<div class="ft-cell ' + cls + '" title="' + escHtml(displayName) + suffix + '">'
        + escHtml(displayName) + suffix + '</div>'
        + '<div class="ft-cell ft-iscore">' + fmtNum(ident.confidence, 2) + '</div>'
        + '<div class="ft-cell">' + buildConfBar(ident.confidence, nearestOnly) + '</div>';
    } else {
      identCells =
        '<div class="ft-cell"></div>'
        + '<div class="ft-cell" style="color:#6b7280">-</div>'
        + '<div class="ft-cell"></div>'
        + '<div class="ft-cell"></div>';
    }

    rows += '<div class="face-row' + highlighted + '" data-face-idx="' + i + '">'
      + '<div class="ft-cell ft-num">' + (i + 1) + '</div>'
      + '<div class="ft-cell ft-quality" style="color:' + color + '">' + label + '</div>'
      + '<div class="ft-cell ft-qscore">' + fmtNum(conf, 2) + '</div>'
      + identCells
      + '</div>';
  }
  return head + rows + '</div>';
}

function buildScoresHtml(frame) {
  const ts = frame.refined_timestamp_s != null ? frame.refined_timestamp_s : frame.timestamp_s;
  const tsRaw = frame.timestamp_s;
  const tsRef = frame.refined_timestamp_s;
  const refinedDifferent = (tsRaw != null && tsRef != null && Math.abs(tsRaw - tsRef) > 1e-6);
  const isVideo = frame.source_type === 'video';
  const srcName = frame.video_basename || '';
  let sourceLine = '<div class="source-name">' + escHtml(srcName);
  if (isVideo && ts != null) sourceLine += ' @ ' + fmtNum(ts, 3) + 's';
  sourceLine += '</div>';
  let refinedLine = '';
  if (isVideo && refinedDifferent) {
    const delta = tsRef - tsRaw;
    const sign = delta >= 0 ? '+' : '';
    refinedLine = '<div class="small">(refined from ' + fmtNum(tsRaw, 3) + 's, &Delta;' + sign + fmtNum(delta, 3) + 's)</div>';
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
  const sharpDeltaStr = sharpDelta == null ? '-' : sharpDeltaSign + fmtNum(sharpDelta, 1);

  const aes = fmtNum(frame.aesthetics_norm, 2);

  let upLine = '';
  if (frame.uprighter_pred) {
    const map = { '90cw': '90&deg; CW', '180': '180&deg;', '270cw': '270&deg; CW' };
    const display = map[frame.uprighter_pred] || frame.uprighter_pred;
    const conf = fmtNum(frame.uprighter_confidence, 2);
    upLine = '<div class="section">'
      + '<div class="label">Uprighter</div>'
      + '<div class="value">' + display + ' <span class="small">(conf ' + conf + ')</span></div>'
      + '</div>';
  }

  const faceCount = (frame.face_count != null) ? frame.face_count : '-';
  const bestPair = frame.best_pair_score;
  const rejectedCount = (frame.rejected_face_count != null) ? frame.rejected_face_count : 0;
  let facesSection = '<div class="section">'
    + '<div class="label">Faces Detected</div>'
    + '<div class="value">' + faceCount + '</div>';
  const toggleLabel = showRejected ? 'Hide Rejected' : 'Show Rejected';
  const toggleClass = showRejected ? 'toggle-rejected-btn active' : 'toggle-rejected-btn';
  facesSection += '<div class="small">rejected: ' + rejectedCount + ' '
    + '<button class="' + toggleClass + '" id="toggle-rejected-btn">' + toggleLabel + '</button>'
    + '</div>';
  if (bestPair != null) facesSection += '<div class="small">best pair score: ' + fmtNum(bestPair, 2) + '</div>';
  facesSection += buildFaceTable(frame);
  facesSection += '</div>';

  return ''
    + '<div class="section">'
    +   '<div class="label">Source</div>'
    +   sourceLine
    +   refinedLine
    + '</div>'
    + '<div class="section">'
    +   '<div class="label">Composite</div>'
    +   '<div class="big">' + composite + '</div>'
    + '</div>'
    + facesSection
    + '<div class="section">'
    +   '<div class="label">Classifier (face 1)</div>'
    +   classifierBars
    + '</div>'
    + '<div class="section">'
    +   '<div class="label">Sharpness</div>'
    +   '<div class="value">center: ' + sharpCenter + ' &rarr; refined: ' + sharpRef + '</div>'
    +   '<div class="small">delta: ' + sharpDeltaStr + '</div>'
    + '</div>'
    + '<div class="section">'
    +   '<div class="label">Aesthetics</div>'
    +   '<div class="value">' + aes + '</div>'
    + '</div>'
    + upLine;
}

function wireScoresPanel(frame) {
  const toggleBtn = document.getElementById('toggle-rejected-btn');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', e => {
      e.stopPropagation();
      showRejected = !showRejected;
      localStorage.setItem(KEY_SHOW_REJECTED, showRejected ? 'true' : 'false');
      document.getElementById('lightbox-scores').innerHTML = buildScoresHtml(frame);
      wireScoresPanel(frame);
      drawFaceOverlay();
    });
  }
  document.querySelectorAll('.face-row').forEach(row => {
    row.addEventListener('click', e => {
      e.stopPropagation();
      const idx = parseInt(row.dataset.faceIdx, 10);
      if (isNaN(idx)) return;
      highlightedFace = (highlightedFace === idx) ? -1 : idx;
      document.getElementById('lightbox-scores').innerHTML = buildScoresHtml(frame);
      wireScoresPanel(frame);
      drawFaceOverlay();
    });
  });
}

// === Canvas face overlay ===
function imageDisplayRect() {
  const img = document.getElementById('lightbox-img');
  const area = document.querySelector('.lightbox-image-area');
  if (!img || !area) return null;
  const containerW = area.clientWidth - 48; // padding 24
  const containerH = area.clientHeight - 48;
  const natW = img.naturalWidth;
  const natH = img.naturalHeight;
  if (!natW || !natH || !containerW || !containerH) return null;
  const scale = Math.min(containerW / natW, containerH / natH);
  const dispW = natW * scale;
  const dispH = natH * scale;
  const rect = img.getBoundingClientRect();
  const areaRect = area.getBoundingClientRect();
  return {
    dispW, dispH,
    offsetX: rect.left - areaRect.left,
    offsetY: rect.top - areaRect.top,
    natW, natH, scale,
  };
}

function drawFaceOverlay() {
  if (!debugFlags.show_faces) return;
  const canvas = document.getElementById('lightbox-canvas');
  const card = lightboxOrder[lightboxIndex];
  if (!canvas || !card) return;
  const frame = currentFrame(card);
  if (!frame) return;

  const rect = imageDisplayRect();
  if (!rect) return;
  const { dispW, dispH, offsetX, offsetY } = rect;

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

  const fw = frame.frame_w;
  const fh = frame.frame_h;
  if (!fw || !fh) return;
  const sx = dispW / fw;
  const sy = dispH / fh;
  const faces = Array.isArray(frame.faces) ? frame.faces : [];
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
      const text = label + ' ' + conf;
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

// === Tooltip on bbox hover ===
function hideTooltip() {
  const tt = document.getElementById('bbox-tooltip');
  if (tt) tt.style.display = 'none';
}

function showTooltip(html, clientX, clientY) {
  const tt = document.getElementById('bbox-tooltip');
  const area = document.querySelector('.lightbox-image-area');
  if (!tt || !area) return;
  const rect = area.getBoundingClientRect();
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

function hitTestFace(clientX, clientY) {
  if (!debugFlags.show_faces) return { accepted: -1, rejected: -1 };
  const card = lightboxOrder[lightboxIndex];
  if (!card) return { accepted: -1, rejected: -1 };
  const frame = currentFrame(card);
  if (!frame) return { accepted: -1, rejected: -1 };
  const rect = imageDisplayRect();
  if (!rect) return { accepted: -1, rejected: -1 };
  const area = document.querySelector('.lightbox-image-area');
  const ar = area.getBoundingClientRect();
  const lx = (clientX - ar.left) - rect.offsetX;
  const ly = (clientY - ar.top) - rect.offsetY;
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

function buildAcceptedTooltipHtml(face, ident, idx) {
  const label = face.pred_label || 'none';
  const conf = (face.pred_confidence != null) ? fmtNum(face.pred_confidence, 2) : '-';
  const color = LABEL_COLORS[label] || '#888';
  let h = '<div class="tt-title">Face ' + (idx + 1) + ': <span style="color:' + color + '">' + escHtml(label) + '</span> ' + conf + '</div>';
  if (ident) {
    const name = escHtml(ident.display_name || ident.identity);
    const c = fmtNum(ident.confidence, 2);
    if (ident.assigned) h += '<div class="tt-row">identity: ' + name + ' ' + c + '</div>';
    else h += '<div class="tt-row muted">nearest: ' + name + ' ' + c + '</div>';
  }
  if (face.kps_anomalous) h += '<div class="tt-row warn">kps anomalous</div>';
  return h;
}

function buildRejectedTooltipHtml(rej) {
  return '<div class="tt-title">Rejected face</div>'
    + '<div class="tt-row warn">' + escHtml(rej.reason || 'rejected') + '</div>';
}

function onLightboxMouseMove(e) {
  if (lightboxIndex < 0) return;
  if (!debugFlags.show_faces) { hideTooltip(); return; }
  const card = lightboxOrder[lightboxIndex];
  if (!card) { hideTooltip(); return; }
  const frame = currentFrame(card);
  if (!frame) { hideTooltip(); return; }
  const { accepted, rejected } = hitTestFace(e.clientX, e.clientY);
  if (accepted >= 0) {
    const face = frame.faces[accepted];
    const ident = (frame.face_identities && frame.face_identities[accepted]) || null;
    showTooltip(buildAcceptedTooltipHtml(face, ident, accepted), e.clientX, e.clientY);
  } else if (rejected >= 0) {
    showTooltip(buildRejectedTooltipHtml(frame.rejected_faces[rejected]), e.clientX, e.clientY);
  } else {
    hideTooltip();
  }
  if (hoveredAcceptedFace !== accepted || hoveredRejectedFace !== rejected) {
    hoveredAcceptedFace = accepted;
    hoveredRejectedFace = rejected;
    drawFaceOverlay();
  }
}

// === Wiring ===
document.addEventListener('DOMContentLoaded', () => {
  loadState();

  // Quality checkboxes
  document.querySelectorAll('.quality-toggle input[type="checkbox"]').forEach(cb => {
    const k = cb.dataset.quality;
    if (k in qualityFilter) cb.checked = qualityFilter[k];
    cb.addEventListener('change', () => {
      qualityFilter[k] = cb.checked;
      saveQualityFilter();
      applyFilters();
    });
  });

  // People mode + chips
  document.querySelectorAll('.mode-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === peopleFilter.mode);
    b.addEventListener('click', () => {
      peopleFilter.mode = b.dataset.mode;
      savePeopleFilter();
      document.querySelectorAll('.mode-btn').forEach(x => x.classList.toggle('active', x.dataset.mode === peopleFilter.mode));
      applyFilters();
    });
  });
  document.querySelectorAll('.identity-chip').forEach(chip => {
    const id = chip.dataset.identity;
    chip.classList.toggle('active', peopleFilter.identities.includes(id));
    chip.addEventListener('click', () => {
      const i = peopleFilter.identities.indexOf(id);
      if (i >= 0) peopleFilter.identities.splice(i, 1);
      else peopleFilter.identities.push(id);
      savePeopleFilter();
      chip.classList.toggle('active');
      applyFilters();
    });
  });

  // Settings checkboxes
  ['show_debug', 'show_faces'].forEach(k => {
    const cb = document.getElementById('toggle-' + k);
    if (!cb) return;
    cb.checked = !!debugFlags[k];
    cb.addEventListener('change', () => {
      debugFlags[k] = cb.checked;
      saveDebugFlags();
      applyDebugFlags();
      if (document.getElementById('lightbox').classList.contains('open')) renderLightbox();
    });
  });

  applyDebugFlags();

  // Source radios
  document.querySelectorAll('input[name="source-radio"]').forEach(r => {
    r.checked = (r.value === sourceFilter);
    r.addEventListener('change', () => {
      if (!r.checked) return;
      sourceFilter = r.value;
      saveSourceFilter();
      applyFilters();
    });
  });

  // Year pill: checkbox area toggles all months for that year
  document.querySelectorAll('.year-pill-check-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      toggleYearAllMonths(btn.dataset.year);
    });
  });
  // Year pill: label area expands/collapses month-pills row
  document.querySelectorAll('.year-pill-label').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      toggleYearExpanded(btn.dataset.year);
    });
  });
  // Month pills: toggle a single section
  document.querySelectorAll('.month-pill').forEach(p => {
    p.addEventListener('click', e => {
      e.stopPropagation();
      toggleSection(p.dataset.section);
    });
  });

  // Section header click: toggle that section
  document.querySelectorAll('.year-header').forEach(h => {
    h.addEventListener('click', () => {
      const sec = h.closest('.year-section').dataset.section;
      if (sec) toggleSection(sec);
    });
  });

  applyLayout();
  applySectionCollapse();
  applySelectionState();
  applyFilters();

  // ResizeObserver to re-run justified layout when container width changes.
  const mainEl = document.querySelector('main');
  if (mainEl && typeof ResizeObserver !== 'undefined') {
    const ro = new ResizeObserver(() => scheduleJustifiedResize());
    ro.observe(mainEl);
  }

  // Dropdown triggers
  document.querySelectorAll('.dropdown-trigger').forEach(t => {
    t.addEventListener('click', e => {
      e.stopPropagation();
      toggleDropdown(t.dataset.dropdown);
    });
  });
  document.addEventListener('click', e => {
    if (e.target.closest('.dropdown-panel')) return;
    if (e.target.closest('.dropdown-trigger')) return;
    closeAllDropdowns();
  });

  // Card click / checkbox
  document.querySelectorAll('main .photo-card').forEach(card => {
    card.addEventListener('click', e => {
      if (e.target.closest('.card-check')) {
        e.stopPropagation();
        if (e.shiftKey && lastClickedCard) {
          rangeSelect(lastClickedCard, card);
        } else {
          toggleSelected(card);
        }
        lastClickedCard = card;
        applyFilters();
        applySelectionState();
        return;
      }
      openLightbox(card, 'main');
    });
  });

  // Export button
  document.getElementById('export-btn').addEventListener('click', openExportOverlay);

  // Export overlay buttons
  document.getElementById('export-done-btn').addEventListener('click', closeExportOverlay);
  document.getElementById('export-zip-btn').addEventListener('click', () => {
    alert('ZIP export not yet implemented.');
  });
  document.getElementById('export-grid').addEventListener('click', e => {
    const card = e.target.closest('.photo-card');
    if (!card) return;
    if (e.target.closest('.card-remove')) {
      e.stopPropagation();
      setSelected(card, false);
      // Remove the live card and the clone
      const liveCard = document.querySelector('main .photo-card[data-card-key="' + card.dataset.cardKey + '"]');
      if (liveCard) liveCard.classList.remove('selected');
      card.remove();
      applyFilters();
      applySelectionState();
      document.getElementById('export-count').textContent = selected.size;
      if (selected.size === 0) {
        const grid = document.getElementById('export-grid');
        grid.innerHTML = '<div class="export-empty">No photos selected.</div>';
      }
      return;
    }
    openLightbox(card, 'export');
  });

  // Lightbox controls
  document.getElementById('lightbox-close').addEventListener('click', closeLightbox);
  document.getElementById('lightbox-prev').addEventListener('click', () => lightboxNav(-1));
  document.getElementById('lightbox-next').addEventListener('click', () => lightboxNav(1));
  document.getElementById('lightbox-select-btn').addEventListener('click', lightboxToggleSelection);
  document.getElementById('lightbox').addEventListener('click', e => {
    // Only close when clicking pure backdrop on left side (not on image area or side panel)
    if (e.target.id === 'lightbox') closeLightbox();
  });
  const area = document.querySelector('.lightbox-image-area');
  if (area) {
    area.addEventListener('mousemove', onLightboxMouseMove);
    area.addEventListener('mouseleave', () => {
      hideTooltip();
      hoveredAcceptedFace = -1;
      hoveredRejectedFace = -1;
      if (debugFlags.show_faces) drawFaceOverlay();
    });
  }

  // Keyboard
  document.addEventListener('keydown', e => {
    if (document.getElementById('lightbox').classList.contains('open')) {
      if (e.key === 'Escape') { closeLightbox(); e.preventDefault(); }
      else if (e.key === 'ArrowLeft') { lightboxNav(-1); e.preventDefault(); }
      else if (e.key === 'ArrowRight') { lightboxNav(1); e.preventDefault(); }
      else if (e.key === ' ') { lightboxToggleSelection(); e.preventDefault(); }
    } else if (document.getElementById('export-overlay').classList.contains('open')) {
      if (e.key === 'Escape') { closeExportOverlay(); e.preventDefault(); }
    }
  });

  // Redraw face overlay on resize
  let resizeTimer = null;
  window.addEventListener('resize', () => {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (debugFlags.show_faces && document.getElementById('lightbox').classList.contains('open')) {
        drawFaceOverlay();
      }
    }, 120);
  });
});
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = ArgumentParser(
        description="Build a self-contained Google Photos-inspired viewer for keeper frames.",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Run YAML config. When provided, --results and "
                             "--output-html default to {output_dir}/results.parquet "
                             "and {output_dir}/index_photos.html. Explicit flags still override.")
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

    cfg: RunConfig | None = None
    if args.config is not None:
        cfg = RunConfig.from_yaml(args.config)
        if args.results is None:
            args.results = cfg.output_dir / "results.parquet"
        if args.output_html is None:
            args.output_html = cfg.output_dir / "index_photos.html"
    if args.results is None or args.output_html is None:
        parser.error(
            "--results and --output-html are required when --config is not provided",
        )

    df = pd.read_parquet(args.results)
    logger.info("Loaded %d rows from %s", len(df), args.results)

    missing_date_cols = [c for c in ("source_year", "source_month") if c not in df.columns]
    if missing_date_cols:
        raise ValueError(
            "results.parquet is missing source_year/source_month columns. "
            "Re-run the pipeline to generate updated results.",
        )

    clusters_path: Path | None = None
    if cfg is not None:
        clusters_path = cfg.output_dir / "clusters.json"
        if "embedding" in df.columns:
            try:
                from still_extractor.build_clusters import run_clustering
                run_clustering(cfg)
            except Exception as e:
                logger.warning(
                    "Identity clustering failed (%s); building viewer without face filter", e,
                )
        else:
            logger.warning("No 'embedding' column in results.parquet; skipping clustering")

    frame_to_identities, frames_with_unknown, clusters_list, membership = (
        _load_cluster_artifacts(clusters_path) if clusters_path else ({}, set(), [], {})
    )

    identity_index = _load_identity_index(IDENTITIES_DIR / "index.json")
    identities_with_centroids = [
        info for info in identity_index.values() if info.get("centroid") is not None
    ]

    html_dir = args.output_html.parent
    html_dir.mkdir(parents=True, exist_ok=True)

    portraits: dict[str, str] = {}
    for ident in identities_with_centroids:
        rel = _portrait_relpath(ident["portrait_path"], html_dir)
        if rel is not None:
            portraits[ident["name"]] = rel

    has_embedding_cols = all(
        f"face_{s}_embedding" in df.columns for s in (1, 2, 3)
    )
    if identities_with_centroids and not has_embedding_cols:
        logger.warning(
            "Identity centroids loaded but per-face embedding columns missing -- "
            "identity will not be shown in debug overlay.",
        )

    # Group rows by (year, month), preserving sort order (composite desc).
    # Sort by composite desc as default within-section ordering.
    if "composite" in df.columns:
        df = df.sort_values("composite", ascending=False, kind="mergesort").reset_index(drop=True)

    # Frame dimensions sidecar cache (avoid re-reading every image each build).
    cache_path: Path | None = None
    if cfg is not None:
        cache_path = cfg.output_dir / "frame_dimensions.json"
    elif args.results is not None:
        cache_path = args.results.parent / "frame_dimensions.json"
    dims_cache: dict[str, list[int]] = (
        _load_frame_dimensions_cache(cache_path) if cache_path else {}
    )
    dims_cache_original_size = len(dims_cache)

    # Per section: list of (video_stem, sort_timestamp_s, card_html). Sorted after the
    # loop so frames from the same source file appear together in temporal order.
    cards_by_section: dict[tuple[int, int], list[tuple[str, float, str]]] = {}
    frames: list[dict] = []
    skipped = 0
    image_source_count = 0
    dim_reads = 0

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

        stem_str = str(row.get("video_stem", "") or "")
        ckey = card_key(stem_str, kept_p) if stem_str else f"_/{kept_p.name}"
        identities = sorted(frame_to_identities.get(ckey, set()))
        has_unknown = ckey in frames_with_unknown

        # Natural pixel dimensions for justified layout, with sidecar cache.
        cached = dims_cache.get(ckey)
        if cached is not None:
            w_nat, h_nat = cached[0], cached[1]
        else:
            wh = _read_image_dimensions(kept_p)
            if wh is None:
                w_nat, h_nat = 4, 3
            else:
                w_nat, h_nat = wh
                dims_cache[ckey] = [w_nat, h_nat]
                dim_reads += 1

        frame_idx = len(frames)
        frames.append(_build_frame_json(
            row, kept_p, is_image_source,
            identities_with_centroids, membership, has_embedding_cols,
            w_nat, h_nat,
        ))

        year = int(row["source_year"])
        month = int(row["source_month"])
        card_html = _build_card(
            row, thumb_src, export_path, rotation, is_image_source,
            frame_idx, year, month, ckey, identities, has_unknown,
        )
        ts_val = row.get("timestamp_s")
        if is_image_source or pd.isna(ts_val):
            sort_ts = 0.0
        else:
            sort_ts = float(ts_val)
        cards_by_section.setdefault((year, month), []).append((stem_str, sort_ts, card_html))

    if skipped:
        logger.info("Skipped %d rows (missing video_path or keeper)", skipped)

    # Sort within each (year, month) section so frames from the same source file
    # are grouped together. Ordering between groups uses the group's earliest
    # timestamp; within a group, ascending timestamp.
    for items in cards_by_section.values():
        group_first_ts: dict[str, float] = {}
        for stem, ts, _ in items:
            prev = group_first_ts.get(stem)
            if prev is None or ts < prev:
                group_first_ts[stem] = ts
        items.sort(key=lambda it: (group_first_ts[it[0]], it[0], it[1]))

    if cache_path is not None and len(dims_cache) != dims_cache_original_size:
        _save_frame_dimensions_cache(cache_path, dims_cache)
    logger.info(
        "Frame dimensions: %d cached, %d freshly read",
        dims_cache_original_size, dim_reads,
    )

    total_cards = sum(len(v) for v in cards_by_section.values())

    # Section ordering: reverse chronological, with (0, 0) [Unknown] last.
    # Any (y, 0) where y != 0 sorts before (0, 0) but after the dated months of y.
    def _section_sort_key(ym: tuple[int, int]) -> tuple[int, int, int]:
        y, m = ym
        if y == 0 and m == 0:
            return (2, 0, 0)
        return (0, -y, -m)

    sorted_sections = sorted(cards_by_section.keys(), key=_section_sort_key)
    section_counts = {ym: len(cards_by_section[ym]) for ym in sorted_sections}

    # Derived: year list (deduped, ordered), months per year, year totals.
    sorted_years: list[int] = []
    seen_years: set[int] = set()
    for y, _ in sorted_sections:
        if y not in seen_years:
            seen_years.add(y)
            sorted_years.append(y)

    months_by_year: dict[int, list[int]] = {}
    for y, m in sorted_sections:
        months_by_year.setdefault(y, []).append(m)
    # Months already in reverse order from section sort; nothing more to do.

    year_counts: dict[int, int] = {}
    for (y, m), n in section_counts.items():
        year_counts[y] = year_counts.get(y, 0) + n

    logger.info(
        "Built %d cards (%d image-source, %d video-source) across %d sections / %d years",
        total_cards, image_source_count, total_cards - image_source_count,
        len(sorted_sections), len(sorted_years),
    )

    # Build month+year sections.
    year_sections: list[str] = []
    for ym in sorted_sections:
        y, m = ym
        sec = _section_key(y, m)
        label = _section_label(y, m)
        section_cards = "\n".join(html_part for _, _, html_part in cards_by_section[ym])
        year_sections.append(f"""<section class="year-section" data-section="{sec}" data-year="{y}" data-month="{m}">
  <h2 class="year-header">
    <span class="year-arrow">&#9662;</span>
    <span class="year-label">{label}</span>
    <span class="year-stats">
      <span class="photo-count">{len(cards_by_section[ym])} photos</span>
      <span class="selected-count" style="display:none"></span>
    </span>
  </h2>
  <div class="year-grid-wrap"><div class="section-grid">
{section_cards}
  </div></div>
</section>""")

    year_pills_html = _year_pills_html(
        sorted_years, months_by_year, year_counts, section_counts,
    )
    identity_chips_html = _identity_chips_html(clusters_list, frames_with_unknown, identity_index)

    # Hide People dropdown trigger if no face filter data exists.
    show_people_dropdown = bool(identity_chips_html)

    frames_json = json.dumps(frames, separators=(",", ":"), allow_nan=False)
    portraits_json = json.dumps(portraits, separators=(",", ":"))
    js_source = (
        JS_TEMPLATE
        .replace("__FRAMES_DATA__", frames_json)
        .replace("__IDENTITY_PORTRAITS__", portraits_json)
    )

    source_panel = """
<div class="dropdown-panel" id="dropdown-source">
  <div class="dropdown-section">
    <span class="dropdown-label">Source</span>
    <div class="layout-radios">
      <label class="layout-radio"><input type="radio" name="source-radio" value="all"><span class="layout-name">All</span></label>
      <label class="layout-radio"><input type="radio" name="source-radio" value="images"><span class="layout-name">Images only</span></label>
      <label class="layout-radio"><input type="radio" name="source-radio" value="videos"><span class="layout-name">Videos only</span></label>
    </div>
  </div>
</div>"""

    quality_panel = """
<div class="dropdown-panel" id="dropdown-quality">
  <div class="dropdown-section">
    <span class="dropdown-label">Quality</span>
    <div class="quality-toggles">
      <label class="quality-toggle"><input type="checkbox" data-quality="good"><span class="quality-dot" style="background:#0f9d58"></span><span class="quality-name">Good</span></label>
      <label class="quality-toggle"><input type="checkbox" data-quality="okay"><span class="quality-dot" style="background:#f4b400"></span><span class="quality-name">Okay</span></label>
      <label class="quality-toggle"><input type="checkbox" data-quality="bad"><span class="quality-dot" style="background:#db4437"></span><span class="quality-name">Bad</span></label>
      <label class="quality-toggle"><input type="checkbox" data-quality="none"><span class="quality-dot" style="background:#9aa0a6"></span><span class="quality-name">None</span></label>
    </div>
  </div>
</div>"""

    people_panel = f"""
<div class="dropdown-panel" id="dropdown-people">
  <div class="dropdown-section">
    <span class="dropdown-label">Match Mode</span>
    <div class="mode-toggle">
      <button class="mode-btn" data-mode="AND">AND</button>
      <button class="mode-btn" data-mode="OR">OR</button>
    </div>
  </div>
  <div class="dropdown-section">
    <span class="dropdown-label">People</span>
    <div class="identity-chips">{identity_chips_html}</div>
  </div>
</div>""" if show_people_dropdown else ""

    settings_panel = """
<div class="dropdown-panel" id="dropdown-settings">
  <div class="dropdown-section">
    <span class="dropdown-label">Debug</span>
    <label class="settings-toggle"><input type="checkbox" id="toggle-show_debug"><span class="toggle-label">Show debug scores</span><span class="toggle-hint">side panel</span></label>
    <label class="settings-toggle"><input type="checkbox" id="toggle-show_faces"><span class="toggle-label">Show face overlays</span><span class="toggle-hint">bboxes + kps</span></label>
  </div>
</div>"""

    people_trigger = (
        '<button class="dropdown-trigger" data-dropdown="people">People <span class="caret">&#9662;</span></button>'
        if show_people_dropdown else ""
    )

    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Still Extractor</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <div class="header-row">
    <span class="title">Still Extractor</span>
    <div class="year-pills">{year_pills_html}</div>
    <div class="header-controls">
      <button class="dropdown-trigger" data-dropdown="source">Source <span class="caret">&#9662;</span></button>
      <button class="dropdown-trigger" data-dropdown="quality">Quality <span class="caret">&#9662;</span></button>
      {people_trigger}
      <button class="dropdown-trigger icon-only" data-dropdown="settings" title="Settings">&#9881;</button>
    </div>
    <button id="export-btn">Export (0)</button>
  </div>
  {source_panel}
  {quality_panel}
  {people_panel}
  {settings_panel}
</header>
<main>
{chr(10).join(year_sections)}
</main>
<div id="lightbox">
  <div class="lightbox-main">
    <div class="lightbox-image-area">
      <img id="lightbox-img" src="" alt="">
      <canvas id="lightbox-canvas"></canvas>
      <div id="bbox-tooltip"></div>
    </div>
    <div class="lightbox-bottom">
      <div id="lightbox-meta"></div>
      <button id="lightbox-select-btn">Select for Export</button>
    </div>
    <button class="lightbox-nav prev" id="lightbox-prev" title="Previous (←)">&#8592;</button>
    <button class="lightbox-nav next" id="lightbox-next" title="Next (→)">&#8594;</button>
    <button class="lightbox-close" id="lightbox-close" title="Close (Esc)">&times;</button>
  </div>
  <div class="lightbox-side">
    <div id="lightbox-scores"></div>
  </div>
</div>
<div id="export-overlay">
  <div class="export-header">
    <div class="export-title">Selected for Export (<span id="export-count">0</span>)</div>
    <div class="export-actions">
      <button class="export-btn primary" id="export-zip-btn">Export ZIP</button>
      <button class="export-btn" id="export-done-btn">Done</button>
    </div>
  </div>
  <div id="export-grid"></div>
</div>
<script>{js_source}</script>
</body>
</html>
"""

    args.output_html.write_text(body, encoding="utf-8")
    file_size_mb = args.output_html.stat().st_size / (1024 * 1024)
    logger.info(
        "Wrote %s (%d cards, %.2f MB)",
        args.output_html, total_cards, file_size_mb,
    )

    summary = {
        "stage": "build_photo_viewer",
        "config": str(args.config) if args.config is not None else None,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "card_count": total_cards,
        "video_source": total_cards - image_source_count,
        "image_source": image_source_count,
        "year_counts": {str(y): year_counts[y] for y in sorted_years},
        "section_counts": {_section_key(y, m): section_counts[(y, m)] for (y, m) in sorted_sections},
        "identity_count": len(identities_with_centroids),
        "identities_with_portraits": len(portraits),
        "has_embedding_cols": bool(has_embedding_cols),
        "output_html": str(args.output_html),
        "file_size_mb": round(file_size_mb, 2),
    }
    summary_path = args.output_html.parent / "build_photo_viewer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
