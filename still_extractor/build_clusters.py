"""Cluster face identities from results.parquet using ArcFace embeddings.

Runs DBSCAN (cosine metric) over per-frame embeddings, matches each cluster
centroid against a persistent global identity store (data/identities/), and
writes a per-run clusters.json keyed by card_key() so the photo viewer can
filter frames by identity. New identities get placeholder names (personA,
personB, ...); the user can rename portraits in data/identities/ by editing
index.json + renaming the matching PNG.
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

from still_extractor.constants import FACE_CROP_PADDING, card_key
from still_extractor.face_crop import extract_face_crop_from_image
from still_extractor.inventory import RunConfig
from still_extractor.utils import parse_kps, safe_float as _safe_float

logger = logging.getLogger(__name__)


DBSCAN_EPS = 0.4
DBSCAN_MIN_SAMPLES = 5  # minimum cluster size
IDENTITY_MATCH_THRESHOLD = 0.5  # max cosine distance to match existing identity
# Orphan recovery is more lenient: the user explicitly opted in by renaming
# the PNG, so we trust borderline matches more than blind centroid matching.
ORPHAN_MATCH_THRESHOLD = 0.6

PORTRAIT_SIZE = 256


def _parse_embedding(val) -> np.ndarray | None:
    """Parse the JSON-encoded embedding column to a numpy array, or None."""
    if val is None:
        return None
    try:
        arr = json.loads(val) if isinstance(val, str) else val
    except Exception:
        return None
    if not isinstance(arr, (list, tuple)) or len(arr) == 0:
        return None
    try:
        out = np.asarray(arr, dtype=np.float32)
    except Exception:
        return None
    if out.ndim != 1:
        return None
    return out


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v
    return v / norm


def _next_placeholder_name(existing: set[str]) -> str:
    """Return the next unused personA/personB/.../personAA placeholder name."""
    # Enumerate base-26 names: A, B, ..., Z, AA, AB, ..., ZZ, AAA, ...
    def to_letters(n: int) -> str:
        s = ""
        n += 1  # 1-indexed
        while n > 0:
            n, r = divmod(n - 1, 26)
            s = chr(ord("A") + r) + s
        return s

    i = 0
    while True:
        name = f"person{to_letters(i)}"
        if name not in existing:
            return name
        i += 1


def _load_identity_index(index_path: Path) -> list[dict]:
    if not index_path.exists():
        return []
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not parse %s (%s); starting fresh", index_path, e)
        return []
    if not isinstance(data, list):
        logger.warning("%s is not a list; starting fresh", index_path)
        return []
    return data


def _sha256_file(path: Path) -> str | None:
    """Return hex sha256 of file contents, or None on read failure."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        logger.warning("Failed to hash %s: %s", path, e)
        return None


def _save_portrait(
    kept_path: Path,
    bbox: tuple[float, float, float, float],
    kps,
    out_path: Path,
) -> str | None:
    """Crop the representative face, resize, write PNG. Return sha256 or None on failure."""
    try:
        img = Image.open(kept_path).convert("RGB")
    except Exception as e:
        logger.warning("Failed to open %s for portrait: %s", kept_path, e)
        return None
    x1, y1, x2, y2 = bbox
    try:
        crop = extract_face_crop_from_image(
            img, x1, y1, x2, y2, FACE_CROP_PADDING, kps=kps,
        )
        crop = crop.resize((PORTRAIT_SIZE, PORTRAIT_SIZE), Image.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path, format="PNG")
    except Exception as e:
        logger.warning("Failed to write portrait %s: %s", out_path, e)
        return None
    return _sha256_file(out_path)


def _sync_identity_names(identities_dir: Path, existing: list[dict]) -> None:
    """Reflect portrait-file renames into `existing` entries (in place).

    For each PNG under `identities_dir`, try to link it to an entry by:
      1. matching the stored `portrait_sha256` (strong link), else
      2. matching the filename stem against the entry's `display_name` or `name`.

    On match, updates the entry's `display_name` (file stem), `portrait_path`
    (the actual file, project-relative), and refreshes `portrait_sha256`.

    Unmatched PNGs are logged as orphans. Each entry can be claimed by at most
    one PNG; later candidates for an already-claimed entry are skipped.
    """
    if not identities_dir.exists():
        return
    pngs = sorted(identities_dir.glob("*.png"))
    if not pngs:
        return

    by_hash: dict[str, dict] = {}
    by_name: dict[str, dict] = {}
    by_display: dict[str, dict] = {}
    for entry in existing:
        h = entry.get("portrait_sha256")
        if isinstance(h, str) and h:
            by_hash.setdefault(h, entry)
        n = entry.get("name")
        if isinstance(n, str) and n:
            by_name.setdefault(n, entry)
        d = entry.get("display_name")
        if isinstance(d, str) and d:
            by_display.setdefault(d, entry)

    claimed: set[int] = set()
    for png in pngs:
        h = _sha256_file(png)
        entry = None
        match_type = None
        if h is not None and h in by_hash:
            entry = by_hash[h]
            match_type = "hash"
        elif png.stem in by_display:
            entry = by_display[png.stem]
            match_type = "stem(display)"
        elif png.stem in by_name:
            entry = by_name[png.stem]
            match_type = "stem(name)"

        if entry is None:
            logger.info("Orphan portrait (no hash or stem match): %s", png)
            continue
        if id(entry) in claimed:
            logger.warning(
                "Portrait %s would re-link identity %s via %s, but it is "
                "already claimed by another file; ignoring.",
                png, entry.get("name"), match_type,
            )
            continue
        claimed.add(id(entry))

        new_display = png.stem
        new_path = f"data/identities/{png.name}"
        old_display = entry.get("display_name")
        old_path = entry.get("portrait_path")
        if old_display != new_display or old_path != new_path:
            logger.info(
                "Identity %s: '%s' (%s) -> '%s' (%s) via %s",
                entry.get("name"), old_display, old_path,
                new_display, new_path, match_type,
            )
        entry["display_name"] = new_display
        entry["portrait_path"] = new_path
        if h is not None:
            entry["portrait_sha256"] = h


def _recover_orphans_by_embedding(
    identities_dir: Path, existing: list[dict],
) -> int:
    """Match orphan portrait PNGs to unclaimed identity centroids via InsightFace.

    Last-resort bootstrap when sha256 and stem matching have both failed (e.g. a
    rename + content-edit, or a first migration where index.json never stored
    sha256). For each orphan PNG: detect the largest face, embed it, and
    Hungarian-assign to the nearest unclaimed centroid under
    IDENTITY_MATCH_THRESHOLD. Mutates `existing` in place. Returns count linked.
    """
    if not identities_dir.exists():
        return 0
    pngs = sorted(identities_dir.glob("*.png"))
    if not pngs:
        return 0

    claimed_hashes = {
        e.get("portrait_sha256") for e in existing
        if isinstance(e.get("portrait_sha256"), str)
    }
    orphans: list[Path] = []
    for png in pngs:
        h = _sha256_file(png)
        if h is not None and h in claimed_hashes:
            continue
        orphans.append(png)
    if not orphans:
        return 0

    unclaimed: list[dict] = [
        e for e in existing
        if not isinstance(e.get("portrait_sha256"), str)
        or not e["portrait_sha256"]
    ]
    if not unclaimed:
        logger.info(
            "Found %d orphan portrait(s) but no unclaimed identities to match.",
            len(orphans),
        )
        return 0

    logger.info(
        "Attempting face-embedding recovery for %d orphan portrait(s) against "
        "%d unclaimed centroid(s).",
        len(orphans), len(unclaimed),
    )

    import cv2
    from insightface.app import FaceAnalysis

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    centroids = np.stack(
        [_l2_normalize(np.asarray(e["centroid"], dtype=np.float32))
         for e in unclaimed], axis=0,
    )

    def _pad_for_detect(bgr: np.ndarray, factor: float = 2.0) -> np.ndarray:
        """Gray-pad a tight face crop so the detector has surrounding context."""
        h, w = bgr.shape[:2]
        nh, nw = int(h * factor), int(w * factor)
        canvas = np.full((nh, nw, 3), 128, dtype=np.uint8)
        y0, x0 = (nh - h) // 2, (nw - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = bgr
        return canvas

    cost = np.full((len(orphans), len(unclaimed)), 1.0, dtype=np.float32)
    embeddings: list[np.ndarray | None] = [None] * len(orphans)
    for oi, png in enumerate(orphans):
        bgr = cv2.imread(str(png))
        if bgr is None:
            continue
        # Portraits are 256x256 face crops; the detector won't fire on those
        # without context, so pad before detection.
        faces = face_app.get(_pad_for_detect(bgr, 2.0))
        if not faces:
            logger.info("No face detected in orphan %s; skipping.", png.name)
            continue
        faces.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        emb = _l2_normalize(np.asarray(faces[0].normed_embedding, dtype=np.float32))
        embeddings[oi] = emb
        cost[oi, :] = 1.0 - (centroids @ emb)

    row_ind, col_ind = linear_sum_assignment(cost)
    linked = 0
    for r, c in zip(row_ind, col_ind):
        if embeddings[r] is None:
            continue
        d = float(cost[r, c])
        png = orphans[r]
        entry = unclaimed[c]
        if d > ORPHAN_MATCH_THRESHOLD:
            logger.info(
                "Orphan %s: nearest unclaimed identity %s d=%.3f > %.2f; skipping.",
                png.name, entry["name"], d, ORPHAN_MATCH_THRESHOLD,
            )
            continue
        new_display = png.stem
        new_path = f"data/identities/{png.name}"
        new_hash = _sha256_file(png)
        logger.info(
            "Recovered orphan: %s -> %s (display='%s', d=%.3f)",
            png.name, entry["name"], new_display, d,
        )
        entry["display_name"] = new_display
        entry["portrait_path"] = new_path
        if new_hash is not None:
            entry["portrait_sha256"] = new_hash
        linked += 1
    return linked


def _collect_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    """Return (N, 512) embedding matrix and list of source row indices in df."""
    embeddings: list[np.ndarray] = []
    row_indices: list[int] = []
    for idx, row in df.iterrows():
        face_count = row.get("face_count")
        if face_count is None or pd.isna(face_count) or int(face_count) <= 0:
            continue
        emb = _parse_embedding(row.get("embedding"))
        if emb is None:
            continue
        embeddings.append(emb)
        row_indices.append(idx)
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), []
    return np.stack(embeddings, axis=0), row_indices


def _match_to_existing(
    new_centroids: np.ndarray,
    existing: list[dict],
) -> dict[int, int]:
    """Match new cluster centroids to existing identities via Hungarian + threshold.

    Returns dict mapping new-cluster-index -> existing-identity-index for accepted matches.
    """
    if len(existing) == 0 or new_centroids.shape[0] == 0:
        return {}
    existing_centroids = np.stack(
        [np.asarray(e["centroid"], dtype=np.float32) for e in existing], axis=0,
    )
    # cosine distance = 1 - cos_similarity (assumes both inputs are L2-normalized)
    cos_sim = new_centroids @ existing_centroids.T
    cost = 1.0 - cos_sim
    row_ind, col_ind = linear_sum_assignment(cost)
    matches: dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < IDENTITY_MATCH_THRESHOLD:
            matches[int(r)] = int(c)
    return matches


def run_clustering(cfg: RunConfig) -> dict | None:
    """Cluster faces in `{cfg.output_dir}/results.parquet`.

    Writes:
      - `{cfg.output_dir}/clusters.json`
      - `data/identities/index.json` (global, updated in place)
      - `data/identities/{name}.png` (one per identity, overwritten each run)

    Returns the cluster summary dict, or None if clustering was skipped/errored.
    """
    results_path = cfg.output_dir / "results.parquet"
    if not results_path.exists():
        logger.error("results.parquet not found at %s", results_path)
        return None

    df = pd.read_parquet(results_path)
    logger.info("Loaded %d rows from %s", len(df), results_path)

    if "embedding" not in df.columns:
        logger.error(
            "results.parquet has no 'embedding' column - rerun the pipeline "
            "with a build that writes embeddings before clustering.",
        )
        return None

    embeddings, row_indices = _collect_embeddings(df)
    logger.info("Collected %d embeddings from rows with face_count > 0", embeddings.shape[0])
    if embeddings.shape[0] == 0:
        logger.warning("No embeddings to cluster; skipping")
        return None

    # InsightFace embeddings come from normed_embedding so they are already L2-normalized,
    # but re-normalize defensively.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms

    logger.info(
        "Running DBSCAN (eps=%.2f, min_samples=%d, metric=cosine) on %d embeddings",
        DBSCAN_EPS, DBSCAN_MIN_SAMPLES, embeddings.shape[0],
    )
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine")
    labels = db.fit_predict(embeddings)

    unknown_mask = labels == -1
    unknown_face_count = int(unknown_mask.sum())
    unknown_row_indices = [row_indices[i] for i in np.where(unknown_mask)[0]]

    cluster_labels = sorted(int(l) for l in set(labels) if l != -1)
    logger.info(
        "DBSCAN found %d clusters and %d noise points",
        len(cluster_labels), unknown_face_count,
    )

    # Build cluster -> member row indices, plus centroid + representative
    clusters_raw: list[dict] = []
    for lbl in cluster_labels:
        member_local = np.where(labels == lbl)[0]
        if len(member_local) < DBSCAN_MIN_SAMPLES:
            continue
        member_rows = [row_indices[i] for i in member_local]
        member_embs = embeddings[member_local]
        centroid = _l2_normalize(member_embs.mean(axis=0))

        # Pick representative: highest composite (fall back to det_score then any)
        best_row_idx = None
        best_composite = -float("inf")
        for row_idx in member_rows:
            comp = _safe_float(df.iloc[row_idx].get("composite"))
            if comp is None:
                comp = _safe_float(df.iloc[row_idx].get("face_det_score")) or 0.0
            if comp > best_composite:
                best_composite = comp
                best_row_idx = row_idx
        if best_row_idx is None:
            best_row_idx = member_rows[0]

        clusters_raw.append({
            "label": lbl,
            "member_rows": member_rows,
            "centroid": centroid,
            "representative_row": best_row_idx,
        })

    # Identity matching
    identities_dir = Path("data/identities")
    identities_dir.mkdir(parents=True, exist_ok=True)
    index_path = identities_dir / "index.json"
    existing = _load_identity_index(index_path)
    # Reflect any file renames the user did since last run so subsequent steps
    # see the up-to-date display_name / portrait_path.
    _sync_identity_names(identities_dir, existing)
    # Last-resort: link any remaining orphan portraits by re-embedding their faces.
    _recover_orphans_by_embedding(identities_dir, existing)
    existing_names = {e["name"] for e in existing if isinstance(e.get("name"), str)}

    new_centroids = (
        np.stack([c["centroid"] for c in clusters_raw], axis=0)
        if clusters_raw else np.empty((0, embeddings.shape[1]), dtype=np.float32)
    )
    matches = _match_to_existing(new_centroids, existing)

    matched_existing = 0
    new_identities = 0
    cluster_entries: list[dict] = []

    for ci, cluster in enumerate(clusters_raw):
        if ci in matches:
            ident_idx = matches[ci]
            entry = existing[ident_idx]
            name = entry["name"]
            entry["centroid"] = cluster["centroid"].tolist()
            entry["member_count"] = len(cluster["member_rows"])
            # Backfill display_name/portrait_path for entries from older runs
            # that pre-date these fields. Don't overwrite user-edited values
            # (renames are propagated by _sync_identity_names before this loop).
            entry.setdefault("display_name", name)
            entry.setdefault("portrait_path", f"data/identities/{name}.png")
            matched_existing += 1
            is_new = False
        else:
            name = _next_placeholder_name(existing_names)
            existing_names.add(name)
            entry = {
                "name": name,
                "display_name": name,
                "centroid": cluster["centroid"].tolist(),
                "member_count": len(cluster["member_rows"]),
                "portrait_path": f"data/identities/{name}.png",
            }
            existing.append(entry)
            new_identities += 1
            is_new = True

        rep_row = df.iloc[cluster["representative_row"]]
        rep_kept = rep_row.get("kept_path")
        bbox = (
            _safe_float(rep_row.get("face_x1")),
            _safe_float(rep_row.get("face_y1")),
            _safe_float(rep_row.get("face_x2")),
            _safe_float(rep_row.get("face_y2")),
        )

        # Save a portrait only for newly-created identities; matched ones keep
        # the user's curated PNG sticky across runs.
        if is_new and (
            isinstance(rep_kept, str)
            and rep_kept
            and not pd.isna(rep_kept)
            and None not in bbox
        ):
            sha = _save_portrait(
                Path(rep_kept),
                bbox,
                parse_kps(rep_row.get("kps")),
                identities_dir / f"{name}.png",
            )
            if sha is not None:
                entry["portrait_sha256"] = sha

        # Build frame_ids for this cluster
        frame_ids: list[str] = []
        for row_idx in cluster["member_rows"]:
            r = df.iloc[row_idx]
            stem = r.get("video_stem")
            kept = r.get("kept_path")
            if (
                isinstance(stem, str) and stem
                and isinstance(kept, str) and kept
                and not pd.isna(stem) and not pd.isna(kept)
            ):
                frame_ids.append(card_key(stem, kept))

        cluster_entries.append({
            "identity": name,
            "member_count": len(cluster["member_rows"]),
            "representative_kept_path": (
                str(rep_kept) if isinstance(rep_kept, str) else None
            ),
            "representative_face_bbox": [
                float(b) if b is not None else None for b in bbox
            ],
            "frame_ids": frame_ids,
        })

    # Build unknown frame_ids (noise faces)
    unknown_frame_ids: list[str] = []
    for row_idx in unknown_row_indices:
        r = df.iloc[row_idx]
        stem = r.get("video_stem")
        kept = r.get("kept_path")
        if (
            isinstance(stem, str) and stem
            and isinstance(kept, str) and kept
            and not pd.isna(stem) and not pd.isna(kept)
        ):
            unknown_frame_ids.append(card_key(stem, kept))

    clusters_summary = {
        "run_name": cfg.name,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "cluster_count": len(cluster_entries),
        "unknown_face_count": unknown_face_count,
        "matched_existing": matched_existing,
        "new_identities": new_identities,
        "clusters": cluster_entries,
        "unknown_frame_ids": unknown_frame_ids,
    }

    clusters_path = cfg.output_dir / "clusters.json"
    clusters_path.write_text(json.dumps(clusters_summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", clusters_path)

    index_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d identities)", index_path, len(existing))

    print()
    print("Identity clustering complete")
    print(f"  Clusters found:   {len(cluster_entries)}")
    print(f"  Unknown faces:    {unknown_face_count}")
    print(f"  Matched existing: {matched_existing}")
    print(f"  New identities:   {new_identities}")

    return clusters_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster face identities from results.parquet.",
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
    result = run_clustering(cfg)
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
