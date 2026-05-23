"""Cluster face identities from results.parquet using ArcFace embeddings.

Runs DBSCAN (cosine metric) over per-frame embeddings, matches each cluster
centroid against a persistent global identity store (data/identities/), and
writes a per-run clusters.json keyed by card_key() so the photo viewer can
filter frames by identity. New identities get placeholder names (personA,
personB, ...); the user can rename portraits in data/identities/ by editing
index.json + renaming the matching PNG.
"""

import argparse
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


def _save_portrait(
    kept_path: Path,
    bbox: tuple[float, float, float, float],
    kps,
    out_path: Path,
) -> bool:
    """Crop the representative face, resize to PORTRAIT_SIZE, write PNG. Return success."""
    try:
        img = Image.open(kept_path).convert("RGB")
    except Exception as e:
        logger.warning("Failed to open %s for portrait: %s", kept_path, e)
        return False
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
        return False
    return True


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
            entry["portrait_path"] = f"data/identities/{name}.png"
            # Preserve user-edited display_name; backfill for entries from
            # older runs that pre-date the field.
            entry.setdefault("display_name", name)
            matched_existing += 1
        else:
            name = _next_placeholder_name(existing_names)
            existing_names.add(name)
            existing.append({
                "name": name,
                "display_name": name,
                "centroid": cluster["centroid"].tolist(),
                "member_count": len(cluster["member_rows"]),
                "portrait_path": f"data/identities/{name}.png",
            })
            new_identities += 1

        # Save portrait PNG
        rep_row = df.iloc[cluster["representative_row"]]
        rep_kept = rep_row.get("kept_path")
        bbox = (
            _safe_float(rep_row.get("face_x1")),
            _safe_float(rep_row.get("face_y1")),
            _safe_float(rep_row.get("face_x2")),
            _safe_float(rep_row.get("face_y2")),
        )
        if (
            isinstance(rep_kept, str)
            and rep_kept
            and not pd.isna(rep_kept)
            and None not in bbox
        ):
            _save_portrait(
                Path(rep_kept),
                bbox,
                parse_kps(rep_row.get("kps")),
                identities_dir / f"{name}.png",
            )

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
