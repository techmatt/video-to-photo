"""Diagnose anomalous face keypoint geometry and its impact on ArcFace clustering.

For each row in results.parquet with a detected face_1, compute simple
geometry sanity checks on the 5-point landmark layout (eyes, nose, mouth
corners). Flag rows whose nose isn't vertically between eyes and mouth,
whose eye-to-nose / eye-to-mouth ratio is off, or whose keypoints span too
little of the face bbox. Then compare the ArcFace cosine distance from
those anomalous embeddings against cluster centroids vs. the same distance
for non-anomalous embeddings -- a higher mean distance + lower cluster
assignment rate for anomalous frames is evidence that bad keypoints (via
the alignment they drive) are degrading identity clustering.
"""

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from still_extractor.constants import card_key
from still_extractor.inventory import RunConfig
from still_extractor.utils import parse_kps, safe_float

logger = logging.getLogger(__name__)


RATIO_LOW = 0.25
RATIO_HIGH = 0.75
KPS_SPAN_FRAC_MIN = 0.25
IDENTITY_INDEX_PATH = Path("data/identities/index.json")


def _parse_embedding(val) -> np.ndarray | None:
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


def _load_centroids(index_path: Path) -> tuple[np.ndarray, list[str]] | None:
    """Return (M, D) centroid matrix and parallel list of names, or None if unavailable."""
    if not index_path.exists():
        return None
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not parse %s (%s)", index_path, e)
        return None
    if not isinstance(data, list) or not data:
        return None
    names: list[str] = []
    rows: list[np.ndarray] = []
    for e in data:
        c = e.get("centroid") if isinstance(e, dict) else None
        n = e.get("name") if isinstance(e, dict) else None
        if not isinstance(c, list) or not isinstance(n, str):
            continue
        try:
            v = np.asarray(c, dtype=np.float32)
        except Exception:
            continue
        if v.ndim != 1:
            continue
        rows.append(_l2_normalize(v))
        names.append(n)
    if not rows:
        return None
    return np.stack(rows, axis=0), names


def _load_cluster_assignment(clusters_path: Path) -> dict[str, str]:
    """Map card_key -> identity name from clusters.json's frame_ids lists."""
    if not clusters_path.exists():
        return {}
    try:
        data = json.loads(clusters_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not parse %s (%s)", clusters_path, e)
        return {}
    out: dict[str, str] = {}
    for c in data.get("clusters", []) or []:
        ident = c.get("identity")
        if not isinstance(ident, str):
            continue
        for fid in c.get("frame_ids", []) or []:
            if isinstance(fid, str):
                out[fid] = ident
    return out


def _kp_geometry(kps: list[list[float]], bbox_h: float | None) -> dict:
    """Compute geometry diagnostics for one 5-point landmark set.

    Layout: 0=left_eye, 1=right_eye, 2=nose, 3=left_mouth, 4=right_mouth.
    """
    pts = [(float(p[0]), float(p[1])) for p in kps[:5]]
    le, re_, no, lm, rm = pts
    eye_mid = ((le[0] + re_[0]) * 0.5, (le[1] + re_[1]) * 0.5)
    mouth_mid = ((lm[0] + rm[0]) * 0.5, (lm[1] + rm[1]) * 0.5)

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    eye_to_nose = dist(eye_mid, no)
    nose_to_mouth = dist(no, mouth_mid)
    eye_to_mouth = dist(eye_mid, mouth_mid)
    ratio = eye_to_nose / eye_to_mouth if eye_to_mouth > 0 else float("nan")

    ys = [p[1] for p in pts]
    kps_span = max(ys) - min(ys)
    span_frac = (kps_span / bbox_h) if (bbox_h and bbox_h > 0) else float("nan")

    vertical_ok = (eye_mid[1] < no[1] < mouth_mid[1])

    reasons: list[str] = []
    if not vertical_ok:
        reasons.append("order")
    if math.isfinite(ratio) and (ratio < RATIO_LOW or ratio > RATIO_HIGH):
        reasons.append("ratio")
    if math.isfinite(span_frac) and span_frac < KPS_SPAN_FRAC_MIN:
        reasons.append("span")

    return {
        "eye_to_nose_dist": eye_to_nose,
        "nose_to_mouth_dist": nose_to_mouth,
        "eye_mouth_dist": eye_to_mouth,
        "ratio": ratio,
        "kps_span": kps_span,
        "kps_span_frac": span_frac,
        "vertical_order_ok": vertical_ok,
        "anomaly_reasons": reasons,
        "anomalous": bool(reasons),
    }


def run_diagnostics(cfg: RunConfig) -> dict | None:
    results_path = cfg.output_dir / "results.parquet"
    if not results_path.exists():
        logger.error("results.parquet not found at %s", results_path)
        return None

    df = pd.read_parquet(results_path)
    logger.info("Loaded %d rows from %s", len(df), results_path)

    centroids_data = _load_centroids(IDENTITY_INDEX_PATH)
    if centroids_data is None:
        logger.warning(
            "No identity centroids at %s -- skipping centroid distance computation",
            IDENTITY_INDEX_PATH,
        )
        centroids = None
        centroid_names: list[str] = []
    else:
        centroids, centroid_names = centroids_data
        logger.info("Loaded %d identity centroids", len(centroid_names))

    clusters_path = cfg.output_dir / "clusters.json"
    cluster_assignment = _load_cluster_assignment(clusters_path)
    if cluster_assignment:
        logger.info("Loaded %d frame->identity assignments from %s",
                    len(cluster_assignment), clusters_path)
    else:
        logger.warning("No cluster assignments available from %s", clusters_path)

    out_rows: list[dict] = []

    total_faces = 0
    anomalous_count = 0
    reason_counts = {"order": 0, "ratio": 0, "span": 0}

    anomalous_dists: list[float] = []
    normal_dists: list[float] = []
    anomalous_assigned = 0
    normal_assigned = 0
    anomalous_with_dist = 0
    normal_with_dist = 0

    for _, row in df.iterrows():
        face_count = safe_float(row.get("face_count"))
        if face_count is None or face_count <= 0:
            continue
        kps = parse_kps(row.get("face_1_kps"))
        if kps is None or len(kps) < 5:
            continue

        y1 = safe_float(row.get("face_1_y1"))
        y2 = safe_float(row.get("face_1_y2"))
        bbox_h = (y2 - y1) if (y1 is not None and y2 is not None) else None

        geom = _kp_geometry(kps, bbox_h)
        total_faces += 1
        if geom["anomalous"]:
            anomalous_count += 1
            for r in geom["anomaly_reasons"]:
                reason_counts[r] = reason_counts.get(r, 0) + 1

        kept = row.get("kept_path")
        stem = row.get("video_stem")
        key = None
        if isinstance(stem, str) and stem and isinstance(kept, str) and kept:
            key = card_key(stem, kept)

        min_dist: float | None = None
        assigned: str = "no_embedding"
        if centroids is not None:
            emb = _parse_embedding(row.get("embedding"))
            if emb is not None:
                emb_n = _l2_normalize(emb)
                cos_sim = centroids @ emb_n
                # cosine distance = 1 - cos_sim
                dists = 1.0 - cos_sim
                idx = int(np.argmin(dists))
                min_dist = float(dists[idx])
                assigned = cluster_assignment.get(key, "unknown") if key else "unknown"

                if geom["anomalous"]:
                    anomalous_dists.append(min_dist)
                    anomalous_with_dist += 1
                    if assigned != "unknown":
                        anomalous_assigned += 1
                else:
                    normal_dists.append(min_dist)
                    normal_with_dist += 1
                    if assigned != "unknown":
                        normal_assigned += 1
        else:
            assigned = "no_centroids"

        out_rows.append({
            "kept_path": str(kept) if isinstance(kept, str) else None,
            "card_key": key,
            "video_stem": stem if isinstance(stem, str) else None,
            "face_x1": safe_float(row.get("face_1_x1")),
            "face_y1": y1,
            "face_x2": safe_float(row.get("face_1_x2")),
            "face_y2": y2,
            "kps_json": json.dumps(kps),
            "anomalous": geom["anomalous"],
            "anomaly_reasons": ",".join(geom["anomaly_reasons"]),
            "ratio": geom["ratio"] if math.isfinite(geom["ratio"]) else None,
            "kps_span_frac": (
                geom["kps_span_frac"] if math.isfinite(geom["kps_span_frac"]) else None
            ),
            "vertical_order_ok": geom["vertical_order_ok"],
            "min_centroid_dist": min_dist,
            "assigned_cluster": assigned,
        })

    out_df = pd.DataFrame(out_rows)
    out_path = cfg.output_dir / "keypoint_diagnostics.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(out_df))

    pct = (100.0 * anomalous_count / total_faces) if total_faces else 0.0

    def _mean(xs: list[float]) -> float | None:
        return float(np.mean(xs)) if xs else None

    a_mean = _mean(anomalous_dists)
    n_mean = _mean(normal_dists)
    a_assigned_pct = (
        100.0 * anomalous_assigned / anomalous_with_dist
        if anomalous_with_dist else None
    )
    n_assigned_pct = (
        100.0 * normal_assigned / normal_with_dist
        if normal_with_dist else None
    )

    print()
    print("Keypoint anomaly diagnostics")
    print(f"  Total faces analyzed:     {total_faces}")
    print(f"  Anomalous keypoints:      {anomalous_count}  ({pct:.1f}%)")
    print(f"    vertical_order_ok=False: {reason_counts.get('order', 0)}")
    print(f"    ratio out of range:      {reason_counts.get('ratio', 0)}")
    print(f"    kps_span too small:      {reason_counts.get('span', 0)}")
    print(f"    (a face can trip more than one rule; sums may exceed anomalous count)")
    print()
    if centroids is None:
        print("ArcFace embedding impact: skipped (no identity centroids found)")
    else:
        print("ArcFace embedding impact (anomalous vs normal):")
        a_dist_s = f"{a_mean:.3f}" if a_mean is not None else "n/a"
        n_dist_s = f"{n_mean:.3f}" if n_mean is not None else "n/a"
        a_asg_s = f"{a_assigned_pct:.1f}%" if a_assigned_pct is not None else "n/a"
        n_asg_s = f"{n_assigned_pct:.1f}%" if n_assigned_pct is not None else "n/a"
        print(f"  Anomalous - mean min_centroid_dist: {a_dist_s}  assigned_to_cluster: {a_asg_s}")
        print(f"  Normal    - mean min_centroid_dist: {n_dist_s}  assigned_to_cluster: {n_asg_s}")

    summary = {
        "stage": "diagnose_keypoints",
        "config_name": cfg.name,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "total_faces": total_faces,
        "anomalous_count": anomalous_count,
        "anomalous_pct": pct,
        "reason_counts": reason_counts,
        "anomalous_mean_min_centroid_dist": a_mean,
        "normal_mean_min_centroid_dist": n_mean,
        "anomalous_assigned_pct": a_assigned_pct,
        "normal_assigned_pct": n_assigned_pct,
        "centroids_available": centroids is not None,
        "clusters_available": bool(cluster_assignment),
        "output_path": str(out_path),
    }
    summary_path = cfg.output_dir / "diagnose_keypoints_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote summary to %s", summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose face keypoint geometry anomalies and their effect "
                    "on ArcFace cluster distances.",
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
    result = run_diagnostics(cfg)
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
