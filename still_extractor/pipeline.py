"""V2 pipeline orchestrator: runs inventory, loads models, processes files end-to-end.

Replaces the three legacy pass scripts. Each manifest row is handled in one
shot by `worker.process_file`; we only persist final keepers + a status log.
Cross-file dedup and the per-video cap are applied at the end over the union
of (this run's keepers) ∪ (keepers loaded from a prior `results.parquet`).
"""

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from still_extractor.constants import (
    DEFAULT_FACE_QUALITY_MODEL,
    DEFAULT_UPRIGHTER_MODEL,
    FACE_QUALITY_LABELS,
    UPRIGHTER_CONFIDENCE_THRESHOLD,
)
from still_extractor.inventory import RunConfig, run_inventory
from still_extractor.models import load_models
from still_extractor.worker import (
    STAGE_KEYS,
    WorkerConfig,
    _dhash_pil,
    process_file,
)

import imagehash
from PIL import Image

logger = logging.getLogger(__name__)

STATUS_COLUMNS = ["file_path", "status", "keepers", "elapsed_s", "processed_at"]


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    logger.warning("CUDA unavailable; pipeline will run on CPU (slow)")
    return torch.device("cpu")


def _load_done_paths(status_path: Path) -> set[str]:
    if not status_path.exists():
        return set()
    df = pd.read_csv(status_path)
    if df.empty or "status" not in df.columns:
        return set()
    return set(df.loc[df["status"] == "done", "file_path"].astype(str).tolist())


def _append_status(
    status_path: Path, file_path: str, status: str, keepers: int, elapsed_s: float,
) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not status_path.exists()
    with open(status_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "file_path": file_path,
            "status": status,
            "keepers": keepers,
            "elapsed_s": f"{elapsed_s:.3f}",
            "processed_at": datetime.now(timezone.utc).isoformat(),
        })


def _is_truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def _truncate_for_testing(
    manifest_df: pd.DataFrame, max_videos: int, max_images: int,
) -> pd.DataFrame:
    """Test-mode truncation. When >0, cap that type to first N (smallest size_bytes).
    When ==0, exclude that type entirely. (Both 0 means caller shouldn't call this.)
    """
    non_dup = manifest_df[~manifest_df["is_duplicate"].map(_is_truthy)].copy()
    keep_paths: set[str] = set()
    if max_videos > 0:
        videos = non_dup[non_dup["file_type"] == "video"].sort_values("size_bytes")
        keep_paths.update(videos.head(max_videos)["file_path"].astype(str).tolist())
    if max_images > 0:
        images = non_dup[non_dup["file_type"] == "image"].sort_values("size_bytes")
        keep_paths.update(images.head(max_images)["file_path"].astype(str).tolist())
    truncated = non_dup[non_dup["file_path"].astype(str).isin(keep_paths)]
    dropped = len(non_dup) - len(truncated)
    logger.info(
        "Test mode: keeping %d/%d non-duplicate files (dropped %d)",
        len(truncated), len(non_dup), dropped,
    )
    return truncated


def _cross_file_dedup(keepers: list[dict], threshold: int) -> tuple[list[dict], int]:
    """Apply frame dHash dedup across all keepers from all files.

    Higher composite wins. Returns (survivors, n_dropped).
    """
    if not keepers:
        return [], 0
    order = sorted(range(len(keepers)), key=lambda i: -float(keepers[i]["composite"]))
    kept_hashes: list[imagehash.ImageHash] = []
    survivor_indices: list[int] = []
    for i in tqdm(order, desc="cross-file dedup"):
        kp = Path(keepers[i]["kept_path"])
        if not kp.exists():
            survivor_indices.append(i)
            continue
        try:
            h = _dhash_pil(Image.open(kp))
        except Exception as e:
            logger.warning("Failed to hash keeper %s: %s", kp, e)
            survivor_indices.append(i)
            continue
        if any((h - kh) <= threshold for kh in kept_hashes):
            continue
        kept_hashes.append(h)
        survivor_indices.append(i)
    survivor_indices.sort()
    survivors = [keepers[i] for i in survivor_indices]
    return survivors, len(keepers) - len(survivors)


def _apply_video_cap(
    keepers: list[dict], max_per_video: int,
) -> tuple[list[dict], int]:
    if max_per_video <= 0 or not keepers:
        return keepers, 0
    by_video: dict[str, list[int]] = {}
    for i, k in enumerate(keepers):
        by_video.setdefault(k["video_path"], []).append(i)
    survivors: list[int] = []
    videos_capped = 0
    for vp, indices in by_video.items():
        indices.sort(key=lambda i: -float(keepers[i]["composite"]))
        if len(indices) > max_per_video:
            videos_capped += 1
        survivors.extend(indices[:max_per_video])
    survivors.sort()
    return [keepers[i] for i in survivors], videos_capped


def _aggregate_stage_times(
    per_file: list[dict[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Aggregate per-file stage timings into total/mean/max plus a % breakdown.

    Returns (stage_times_s, stage_times_pct). `stage_times_pct` is each stage's
    share of the sum of all per-stage totals (excluding `total`), avoiding the
    overlap confusion between stage time and end-to-end wall time.
    """
    n = len(per_file)
    keys = list(STAGE_KEYS) + ["total"]
    block: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [d.get(key, 0.0) for d in per_file]
        total = float(sum(values))
        max_v = float(max(values)) if values else 0.0
        mean_v = total / n if n else 0.0
        block[key] = {
            "total": total,
            "mean_per_file": mean_v,
            "max_per_file": max_v,
        }
    stage_sum = sum(block[k]["total"] for k in STAGE_KEYS)
    pct: dict[str, float] = {}
    for key in STAGE_KEYS:
        pct[key] = 100.0 * block[key]["total"] / stage_sum if stage_sum > 0 else 0.0
    return block, pct


def _build_viewers(config_path: Path) -> None:
    """Invoke build_faces_review then build_photo_viewer as subprocesses.

    Subprocess gives clean isolation from the pipeline's loaded models and
    matches how users invoke these stages manually. Failures are logged but
    do not abort — the parquet is already on disk and the user can re-run.
    """
    viewers = [
        ("build_faces_review", "still_extractor.build_faces_review"),
        ("build_photo_viewer", "still_extractor.build_photo_viewer"),
    ]
    for name, module in viewers:
        print(f"───── {name} ─────")
        try:
            subprocess.run(
                [sys.executable, "-m", module, "--config", str(config_path)],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.warning("%s failed with exit code %d", name, e.returncode)


def _print_stage_timing_table(
    stage_times_s: dict[str, dict[str, float]],
    stage_times_pct: dict[str, float],
) -> None:
    rows = sorted(
        STAGE_KEYS, key=lambda k: -stage_times_s[k]["total"],
    )
    print("Stage              Total(s)   Mean/file(s)   % of total")
    print("──────────────────────────────────────────────────────")
    for key in rows:
        b = stage_times_s[key]
        print(
            f"{key:<18}{b['total']:>9.2f}     {b['mean_per_file']:>9.4f}     {stage_times_pct[key]:>6.2f}%",
        )
    total_block = stage_times_s.get("total", {"total": 0.0, "mean_per_file": 0.0})
    print("──────────────────────────────────────────────────────")
    print(
        f"{'total (wall)':<18}{total_block['total']:>9.2f}     {total_block['mean_per_file']:>9.4f}",
    )
    print("───────────────────────────────────────")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V2 pipeline: inventory → per-file worker → cross-file dedup → results.parquet.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--sharpness-threshold", type=float, default=75.0)
    parser.add_argument("--min-face-px", type=int, default=80)
    parser.add_argument("--temporal-window-s", type=float, default=2.0)
    parser.add_argument("--face-dedup-threshold", type=int, default=8)
    parser.add_argument("--frame-dedup-threshold", type=int, default=8)
    parser.add_argument("--quality-threshold", type=float, default=0.0)
    parser.add_argument("--max-per-file", type=int, default=5)
    parser.add_argument("--max-per-video", type=int, default=10)
    parser.add_argument("--refine-window-s", type=float, default=0.5)
    parser.add_argument("--uprighter-confidence", type=float,
                        default=UPRIGHTER_CONFIDENCE_THRESHOLD)
    parser.add_argument("--face-quality-model", type=Path,
                        default=DEFAULT_FACE_QUALITY_MODEL)
    parser.add_argument("--uprighter-model", type=Path,
                        default=DEFAULT_UPRIGHTER_MODEL)
    parser.add_argument("--rescan", action="store_true",
                        help="Reprocess all files even if marked done in pipeline_status.csv.")
    parser.add_argument("--max-videos", type=int, default=0,
                        help="Test mode: cap number of non-duplicate videos processed.")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Test mode: cap number of non-duplicate images processed.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
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
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "pipeline_status.csv"
    results_path = output_dir / "results.parquet"

    print("═══════════════════════════════════════")
    print(f"Still Extractor — Pipeline ({cfg.name})")
    print(f"Config:  {args.config}")
    print(f"Output:  {output_dir}")
    print("═══════════════════════════════════════")

    manifest_df = run_inventory(cfg, rescan=False)
    if manifest_df is None or manifest_df.empty:
        logger.warning("Empty manifest; nothing to do.")
        return

    device = _select_device()
    models = load_models(args.face_quality_model, args.uprighter_model, device)

    worker_cfg = WorkerConfig(
        output_dir=output_dir,
        fps=args.fps,
        sharpness_threshold=args.sharpness_threshold,
        min_face_px=args.min_face_px,
        temporal_window_s=args.temporal_window_s,
        face_dedup_threshold=args.face_dedup_threshold,
        frame_dedup_threshold=args.frame_dedup_threshold,
        quality_threshold=args.quality_threshold,
        max_per_file=args.max_per_file,
        uprighter_confidence=args.uprighter_confidence,
        refine_window_s=args.refine_window_s,
    )

    non_dup_df = manifest_df[~manifest_df["is_duplicate"].map(_is_truthy)].copy()
    n_dup = len(manifest_df) - len(non_dup_df)
    test_mode = args.max_videos > 0 or args.max_images > 0
    if test_mode:
        non_dup_df = _truncate_for_testing(manifest_df, args.max_videos, args.max_images)

    non_dup_df = non_dup_df.sort_values("size_bytes")

    done_paths = set() if args.rescan else _load_done_paths(status_path)
    to_process_df = non_dup_df[
        ~non_dup_df["file_path"].astype(str).isin(done_paths)
    ]
    n_skipped_done = len(non_dup_df) - len(to_process_df)

    print(
        f"Manifest: {len(manifest_df)} files "
        f"({n_dup} duplicates skipped, {n_skipped_done} already done, "
        f"{len(to_process_df)} to process)"
    )

    fresh_keepers: list[dict] = []
    n_processed = 0
    per_file_stage_times: list[dict[str, float]] = []
    rejection_stats: dict[str, int] = {
        "total_faces_rejected": 0,
        "too_small": 0,
        "small_and_edge": 0,
        "frames_with_all_faces_rejected": 0,
    }
    for _, row in tqdm(
        to_process_df.iterrows(), total=len(to_process_df), desc="files",
    ):
        start = time.monotonic()
        result = process_file(row, models, worker_cfg)
        elapsed = time.monotonic() - start
        n_processed += 1
        keepers = result.keepers
        per_file_stage_times.append(result.stage_times_s)
        for k, v in result.rejection_stats.items():
            rejection_stats[k] = rejection_stats.get(k, 0) + int(v)
        status = "done"
        fresh_keepers.extend(keepers)
        # Don't write status rows when test-truncating, so a real run will redo them.
        if not test_mode:
            _append_status(
                status_path, str(row["file_path"]), status, len(keepers), elapsed,
            )
        logger.info(
            "%s -> %d keeper(s) in %.1fs",
            Path(row["file_path"]).name, len(keepers), elapsed,
        )

    # Load prior keepers from results.parquet so cross-file dedup sees them too.
    prior_keepers: list[dict] = []
    if not args.rescan and not test_mode and results_path.exists():
        prior_df = pd.read_parquet(results_path)
        processed_paths = set(to_process_df["file_path"].astype(str).tolist())
        prior_df = prior_df[~prior_df["video_path"].astype(str).isin(processed_paths)]
        prior_keepers = prior_df.to_dict(orient="records")
        logger.info("Loaded %d prior keepers from %s", len(prior_keepers), results_path)

    all_keepers = prior_keepers + fresh_keepers
    keepers_before_global_dedup = len(all_keepers)
    survivors, n_dropped_global = _cross_file_dedup(
        all_keepers, args.frame_dedup_threshold,
    )
    keepers_after_global_dedup = len(survivors)

    capped_survivors, videos_capped = _apply_video_cap(survivors, args.max_per_video)
    keepers_after_video_cap = len(capped_survivors)

    # In test mode, write to side files so we don't poison the production outputs.
    parquet_out = output_dir / ("results_test.parquet" if test_mode else "results.parquet")
    summary_path = output_dir / (
        "pipeline_summary_test.json" if test_mode else "pipeline_summary.json"
    )
    # date_source is tracked per keeper for the summary breakdown but not stored
    # in parquet — old parquets won't have it on reload anyway.
    date_source_counts: dict[str, int] = {
        "exif_primary": 0,
        "exif_fallback": 0,
        "path_regex": 0,
        "mtime": 0,
        "unknown": 0,
    }
    for k in capped_survivors:
        ds = k.pop("date_source", None)
        if isinstance(ds, str) and ds in date_source_counts:
            date_source_counts[ds] += 1

    if capped_survivors:
        results_df = pd.DataFrame(capped_survivors)
        results_df.to_parquet(parquet_out, index=False)
        logger.info("Wrote %d keepers to %s", len(results_df), parquet_out)
    else:
        logger.warning("No keepers survived; not writing %s", parquet_out.name)

    pred_label_counts = {lbl: 0 for lbl in FACE_QUALITY_LABELS}
    uprighter_corrections = 0
    kps_anomalous_count = 0
    for k in capped_survivors:
        lbl = k.get("pred_label") or ""
        if lbl in pred_label_counts:
            pred_label_counts[lbl] += 1
        if int(k.get("uprighter_pred", 0)) != 0:
            uprighter_corrections += 1
        if k.get("face_1_kps_anomalous") is True:
            kps_anomalous_count += 1
    kps_anomalous_pct = (
        100.0 * kps_anomalous_count / len(capped_survivors)
        if capped_survivors else 0.0
    )

    stage_times_block, stage_times_pct = _aggregate_stage_times(per_file_stage_times)

    summary = {
        "config": str(args.config),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "total_files": int(len(manifest_df)),
        "files_processed": int(n_processed),
        "files_skipped_duplicate": int(n_dup),
        "files_skipped_done": int(n_skipped_done),
        "keepers_before_global_dedup": int(keepers_before_global_dedup),
        "keepers_after_global_dedup": int(keepers_after_global_dedup),
        "keepers_after_video_cap": int(keepers_after_video_cap),
        "videos_capped": int(videos_capped),
        "pred_label_counts": pred_label_counts,
        "uprighter_corrections_applied": int(uprighter_corrections),
        "kps_anomalous_count": int(kps_anomalous_count),
        "kps_anomalous_pct": float(kps_anomalous_pct),
        "rejection_stats": rejection_stats,
        "date_source_counts": date_source_counts,
        "stage_times_s": stage_times_block,
        "stage_times_pct": stage_times_pct,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote summary to %s", summary_path)

    print("───────────────────────────────────────")
    print(f"Files processed:        {n_processed}")
    print(f"Keepers (pre-dedup):    {keepers_before_global_dedup}")
    print(f"Keepers (post-dedup):   {keepers_after_global_dedup}")
    print(f"Keepers (post-cap):     {keepers_after_video_cap}")
    print(f"Videos capped:          {videos_capped}")
    print(f"Uprighter corrections:  {uprighter_corrections}")
    print(f"Kps anomalous:          {kps_anomalous_count}  ({kps_anomalous_pct:.1f}%)")
    print(f"pred_label_counts:      {pred_label_counts}")
    print(
        f"Rejected faces:         total={rejection_stats['total_faces_rejected']} "
        f"too_small={rejection_stats['too_small']} "
        f"small_and_edge={rejection_stats['small_and_edge']} "
        f"frames_all_rejected={rejection_stats['frames_with_all_faces_rejected']}"
    )
    print(
        "Date sources:           "
        + "  ".join(f"{k}={v}" for k, v in date_source_counts.items())
    )
    print("───────────────────────────────────────")
    _print_stage_timing_table(stage_times_block, stage_times_pct)

    if not test_mode and parquet_out.exists():
        _build_viewers(args.config)


if __name__ == "__main__":
    main()
