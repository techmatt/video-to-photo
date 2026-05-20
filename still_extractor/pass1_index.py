"""Pass 1: manifest-driven indexing of candidate frames from videos and images."""

import csv
import json
import logging
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import pillow_heif
from insightface.app import FaceAnalysis
from PIL import Image
from tqdm import tqdm

from still_extractor.inventory import RunConfig

pillow_heif.register_heif_opener()

logger = logging.getLogger(__name__)

STATUS_COLUMNS = [
    "file_path",
    "file_hash",
    "status",
    "frames_sampled",
    "frames_failed_sharpness",
    "frames_failed_face_detect",
    "frames_failed_face_size",
    "frames_written",
    "sharpness_mean",
    "sharpness_min",
    "sharpness_max",
    "faces_detected",
    "faces_failed_size",
    "elapsed_s",
    "processed_at",
]

_STATS_COLUMNS = (
    "frames_sampled",
    "frames_failed_sharpness",
    "frames_failed_face_detect",
    "frames_failed_face_size",
    "frames_written",
    "sharpness_mean",
    "sharpness_min",
    "sharpness_max",
    "faces_detected",
    "faces_failed_size",
)


@dataclass
class FileStats:
    frames_sampled: int = 0
    frames_failed_sharpness: int = 0
    frames_failed_face_detect: int = 0
    frames_failed_face_size: int = 0
    frames_written: int = 0
    sharpness_values: list[float] = field(default_factory=list)
    faces_detected: int = 0
    faces_failed_size: int = 0

    def sharpness_mean(self) -> float | None:
        return float(sum(self.sharpness_values) / len(self.sharpness_values)) if self.sharpness_values else None

    def sharpness_min(self) -> float | None:
        return float(min(self.sharpness_values)) if self.sharpness_values else None

    def sharpness_max(self) -> float | None:
        return float(max(self.sharpness_values)) if self.sharpness_values else None

# Module-level FaceAnalysis used by worker processes (set via pool initializer).
_face_app: FaceAnalysis | None = None


def _create_face_app() -> FaceAnalysis:
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _worker_init() -> None:
    global _face_app
    _face_app = _create_face_app()


def center_crop_70(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    crop_h = int(h * 0.7)
    crop_w = int(w * 0.7)
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    return img[y0:y0 + crop_h, x0:x0 + crop_w]


def sharpness_score(img: np.ndarray) -> float:
    cropped = center_crop_70(img)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _bbox_width(bbox: np.ndarray) -> float:
    return float(bbox[2] - bbox[0])


def _bbox_area(bbox: np.ndarray) -> float:
    return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def _video_duration_seconds(container: "av.container.InputContainer", stream) -> float | None:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    return None


def sample_frames(video_path: Path, fps: float):
    """Yield (frame_index, timestamp_seconds, bgr_array) by seeking across the full duration."""
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base
        if time_base is None:
            logger.warning("No time_base for %s; skipping", video_path)
            return

        duration_sec = _video_duration_seconds(container, stream)
        if duration_sec is None or duration_sec <= 0:
            logger.warning("No usable duration for %s; skipping", video_path)
            return

        time_base_f = float(time_base)
        target_interval = 1.0 / fps
        num_samples = max(1, int(duration_sec * fps))

        last_pts = -1
        for i in range(num_samples):
            target_sec = i * target_interval
            target_pts = int(target_sec / time_base_f)

            try:
                container.seek(target_pts, stream=stream, any_frame=False, backward=True)
            except Exception as e:
                logger.warning("Seek failed at %.2fs in %s: %s", target_sec, video_path.name, e)
                continue

            chosen = None
            try:
                for decoded in container.decode(stream):
                    if decoded.pts is None or decoded.pts <= last_pts:
                        continue
                    actual_sec = float(decoded.pts * time_base)
                    if actual_sec + 1e-6 >= target_sec:
                        chosen = decoded
                        break
            except Exception as e:
                logger.warning("Decode failed near %.2fs in %s: %s", target_sec, video_path.name, e)
                continue

            if chosen is None:
                continue

            last_pts = chosen.pts
            actual_sec = float(chosen.pts * time_base)
            img = chosen.to_ndarray(format="bgr24")
            yield i, actual_sec, img
    finally:
        container.close()


def sample_frames_windowed(video_path: Path, fps: float, windows: list[int]):
    """Yield (frame_index, timestamp_seconds, bgr_array) within [t, t+1.0) for each t in windows."""
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base
        if time_base is None:
            logger.warning("No time_base for %s; skipping", video_path)
            return

        time_base_f = float(time_base)
        target_interval = 1.0 / fps
        window_len = 1.0
        per_window = max(1, int(window_len * fps))

        idx = 0
        last_pts = -1
        for window_start in windows:
            for i in range(per_window):
                target_sec = float(window_start) + i * target_interval
                if target_sec >= window_start + window_len:
                    break
                target_pts = int(target_sec / time_base_f)

                try:
                    container.seek(target_pts, stream=stream, any_frame=False, backward=True)
                except Exception as e:
                    logger.warning("Seek failed at %.2fs in %s: %s", target_sec, video_path.name, e)
                    continue

                chosen = None
                try:
                    for decoded in container.decode(stream):
                        if decoded.pts is None or decoded.pts <= last_pts:
                            continue
                        actual_sec = float(decoded.pts * time_base)
                        if actual_sec + 1e-6 >= target_sec:
                            chosen = decoded
                            break
                except Exception as e:
                    logger.warning("Decode failed near %.2fs in %s: %s", target_sec, video_path.name, e)
                    continue

                if chosen is None:
                    continue

                last_pts = chosen.pts
                actual_sec = float(chosen.pts * time_base)
                yield idx, actual_sec, chosen.to_ndarray(format="bgr24")
                idx += 1
    finally:
        container.close()


def _parse_windows(value) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str) and value == "":
        return None
    return json.loads(value)


def _build_row(
    *,
    source_path: Path,
    frame_index: int,
    timestamp_s: float,
    frame_path: Path,
    frame_w: int,
    frame_h: int,
    sharpness: float,
    face,
) -> dict:
    bbox = face.bbox
    return {
        "video_path": str(source_path.resolve()),
        "video_stem": source_path.stem,
        "frame_index": int(frame_index),
        "timestamp_s": float(timestamp_s),
        "frame_path": str(frame_path.resolve()),
        "frame_w": int(frame_w),
        "frame_h": int(frame_h),
        "sharpness_center": sharpness,
        "face_x1": float(bbox[0]),
        "face_y1": float(bbox[1]),
        "face_x2": float(bbox[2]),
        "face_y2": float(bbox[3]),
        "face_w": float(bbox[2] - bbox[0]),
        "face_det_score": float(face.det_score),
        "kps": json.dumps([[float(x), float(y)] for x, y in face.kps]),
        "embedding": json.dumps([float(v) for v in face.normed_embedding]),
    }


def process_video(
    row: pd.Series,
    output_dir: Path,
    face_app: FaceAnalysis,
    fps: float,
    sharpness_threshold: float,
    min_face_px: int,
    stats: FileStats,
) -> tuple[list[dict], FileStats]:
    video_path = Path(row["file_path"])
    frames_dir = output_dir / "frames" / video_path.stem

    windows = _parse_windows(row.get("sample_windows_s"))
    if windows is not None and len(windows) > 0:
        frame_iter = sample_frames_windowed(video_path, fps, windows)
    else:
        frame_iter = sample_frames(video_path, fps)

    rows: list[dict] = []
    try:
        for frame_index, timestamp_sec, img in frame_iter:
            stats.frames_sampled += 1
            score = sharpness_score(img)
            stats.sharpness_values.append(score)
            if score < sharpness_threshold:
                stats.frames_failed_sharpness += 1
                logger.debug(
                    "%s frame=%d t=%.3fs sharpness=%.2f DROP-sharpness",
                    video_path.name, frame_index, timestamp_sec, score,
                )
                continue

            try:
                faces = face_app.get(img)
            except Exception as e:
                logger.warning(
                    "InsightFace failed on %s frame=%d: %s",
                    video_path.name, frame_index, e,
                )
                stats.frames_failed_face_detect += 1
                continue

            stats.faces_detected += len(faces)
            if not faces:
                stats.frames_failed_face_detect += 1
                logger.debug(
                    "%s frame=%d t=%.3fs sharpness=%.2f DROP-face-detect",
                    video_path.name, frame_index, timestamp_sec, score,
                )
                continue

            qualifying = [f for f in faces if _bbox_width(f.bbox) >= min_face_px]
            stats.faces_failed_size += len(faces) - len(qualifying)
            if not qualifying:
                stats.frames_failed_face_size += 1
                logger.debug(
                    "%s frame=%d t=%.3fs sharpness=%.2f DROP-face-size",
                    video_path.name, frame_index, timestamp_sec, score,
                )
                continue

            largest = max(qualifying, key=lambda f: _bbox_area(f.bbox))

            out_path = frames_dir / f"{frame_index:06d}_{timestamp_sec:.3f}.jpg"
            if not frames_dir.exists():
                frames_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), img)
            stats.frames_written += 1

            h, w = img.shape[:2]
            rows.append(_build_row(
                source_path=video_path,
                frame_index=frame_index,
                timestamp_s=timestamp_sec,
                frame_path=out_path,
                frame_w=w,
                frame_h=h,
                sharpness=score,
                face=largest,
            ))
    except Exception as e:
        logger.warning("Failed to process %s: %s", video_path, e)

    return rows, stats


def process_image(
    row: pd.Series,
    output_dir: Path,
    face_app: FaceAnalysis,
    sharpness_threshold: float,
    min_face_px: int,
    stats: FileStats,
) -> tuple[list[dict], FileStats]:
    image_path = Path(row["file_path"])

    try:
        with Image.open(image_path) as img_pil:
            img = np.array(img_pil.convert("RGB"))[:, :, ::-1].copy()
    except Exception as e:
        logger.warning("Failed to open image %s: %s", image_path, e)
        return [], stats

    stats.frames_sampled += 1
    score = sharpness_score(img)
    stats.sharpness_values.append(score)
    if score < sharpness_threshold:
        stats.frames_failed_sharpness += 1
        logger.debug("%s sharpness=%.2f DROP-sharpness", image_path.name, score)
        return [], stats

    try:
        faces = face_app.get(img)
    except Exception as e:
        logger.warning("InsightFace failed on image %s: %s", image_path, e)
        stats.frames_failed_face_detect += 1
        return [], stats

    stats.faces_detected += len(faces)
    if not faces:
        stats.frames_failed_face_detect += 1
        logger.debug("%s sharpness=%.2f DROP-face-detect", image_path.name, score)
        return [], stats

    qualifying = [f for f in faces if _bbox_width(f.bbox) >= min_face_px]
    stats.faces_failed_size += len(faces) - len(qualifying)
    if not qualifying:
        stats.frames_failed_face_size += 1
        logger.debug("%s sharpness=%.2f DROP-face-size", image_path.name, score)
        return [], stats

    largest = max(qualifying, key=lambda f: _bbox_area(f.bbox))

    frames_dir = output_dir / "frames" / image_path.stem
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_path = frames_dir / "00000_0.000.jpg"
    cv2.imwrite(str(out_path), img)
    stats.frames_written += 1

    h, w = img.shape[:2]
    return [_build_row(
        source_path=image_path,
        frame_index=0,
        timestamp_s=0.0,
        frame_path=out_path,
        frame_w=w,
        frame_h=h,
        sharpness=score,
        face=largest,
    )], stats


def append_status(
    status_csv: Path,
    *,
    file_path: str,
    file_hash: str,
    status: str,
    elapsed_s: float,
    processed_at: str,
    stats: FileStats | None,
) -> None:
    status_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not status_csv.exists()
    if stats is None:
        stat_cols = {col: None for col in _STATS_COLUMNS}
    else:
        stat_cols = {
            "frames_sampled": stats.frames_sampled,
            "frames_failed_sharpness": stats.frames_failed_sharpness,
            "frames_failed_face_detect": stats.frames_failed_face_detect,
            "frames_failed_face_size": stats.frames_failed_face_size,
            "frames_written": stats.frames_written,
            "sharpness_mean": stats.sharpness_mean(),
            "sharpness_min": stats.sharpness_min(),
            "sharpness_max": stats.sharpness_max(),
            "faces_detected": stats.faces_detected,
            "faces_failed_size": stats.faces_failed_size,
        }
    with open(status_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "file_path": file_path,
            "file_hash": file_hash,
            "status": status,
            **stat_cols,
            "elapsed_s": f"{elapsed_s:.3f}",
            "processed_at": processed_at,
        })


def _process_one(
    row: pd.Series,
    output_dir: Path,
    face_app: FaceAnalysis,
    fps: float,
    sharpness_threshold: float,
    min_face_px: int,
) -> tuple[dict, FileStats, list[dict]]:
    file_path = row["file_path"]
    file_type = row["file_type"]
    stats = FileStats()
    start = time.monotonic()
    try:
        if file_type == "video":
            rows, stats = process_video(
                row, output_dir, face_app, fps, sharpness_threshold, min_face_px, stats,
            )
        elif file_type == "image":
            rows, stats = process_image(
                row, output_dir, face_app, sharpness_threshold, min_face_px, stats,
            )
        else:
            raise ValueError(f"Unknown file_type: {file_type!r}")
        status = "done"
    except Exception:
        logger.exception("Failed to process %s", file_path)
        rows = []
        status = "failed"
    elapsed = time.monotonic() - start
    meta = {
        "file_path": file_path,
        "file_hash": row.get("hash", ""),
        "status": status,
        "elapsed_s": elapsed,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    if status == "done":
        passed_sharpness = stats.frames_sampled - stats.frames_failed_sharpness
        passed_face_detect = passed_sharpness - stats.frames_failed_face_detect
        logger.info(
            "%s - sampled=%d sharp=%d faces=%d written=%d (%.1fs)",
            Path(file_path).name,
            stats.frames_sampled, passed_sharpness, passed_face_detect,
            stats.frames_written, elapsed,
        )
    return meta, stats, rows


def _worker_process_video(
    row_dict: dict,
    output_dir_str: str,
    fps: float,
    sharpness_threshold: float,
    min_face_px: int,
) -> tuple[dict, FileStats, list[dict]]:
    if _face_app is None:
        raise RuntimeError("FaceAnalysis app not initialized in worker process")
    row = pd.Series(row_dict)
    return _process_one(
        row, Path(output_dir_str), _face_app, fps, sharpness_threshold, min_face_px,
    )


def _write_index(
    index_file: Path,
    existing_df: pd.DataFrame | None,
    new_rows: list[dict],
) -> int:
    if not new_rows and existing_df is None:
        logger.info("No rows to write to %s", index_file)
        return 0

    new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()
    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    if "frame_path" in combined.columns:
        combined = combined.drop_duplicates(subset=["frame_path"], keep="last")

    index_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(index_file, index=False)
    return len(combined)


def _load_done_paths(status_path: Path) -> set[str]:
    if not status_path.exists():
        return set()
    df = pd.read_csv(status_path)
    if df.empty or "status" not in df.columns:
        return set()
    return set(df.loc[df["status"] == "done", "file_path"].astype(str).tolist())


def _ensure_status_csv_schema(status_path: Path) -> None:
    """Rewrite an old-schema pass1_status.csv with current STATUS_COLUMNS so appends don't corrupt it."""
    if not status_path.exists():
        return
    df = pd.read_csv(status_path)
    if list(df.columns) == STATUS_COLUMNS:
        return
    if "rows_written" in df.columns and "frames_written" not in df.columns:
        df = df.rename(columns={"rows_written": "frames_written"})
    for col in STATUS_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[STATUS_COLUMNS]
    df.to_csv(status_path, index=False)
    logger.info("Upgraded %s to new status schema (%d rows)", status_path, len(df))


def main() -> None:
    parser = ArgumentParser(description="Manifest-driven Pass 1: index candidate frames from videos and images.")
    parser.add_argument("--config", type=Path, required=True, help="Path to run YAML config file.")
    parser.add_argument("--ffmpeg-path", type=str, default="ffmpeg",
                        help="ffmpeg executable path. Accepted but unused (PyAV uses bundled libs).")
    parser.add_argument("--min-face-px", type=int, default=80,
                        help="Minimum face bounding box width in pixels.")
    parser.add_argument("--sharpness-threshold", type=float, default=75.0,
                        help="Laplacian variance threshold for pre-filter.")
    parser.add_argument("--fps", type=float, default=3.0,
                        help="Sample rate in frames per second for short videos.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel video workers.")
    parser.add_argument("--rescan", action="store_true",
                        help="Reprocess all files even if already marked done in pass1_status.csv.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity.")
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

    if args.ffmpeg_path != "ffmpeg":
        logger.debug("--ffmpeg-path=%s provided but unused (PyAV uses bundled libs)", args.ffmpeg_path)

    cfg = RunConfig.from_yaml(args.config)
    output_dir = cfg.output_dir
    manifest_path = output_dir / "manifest.csv"
    index_path = output_dir / "index.parquet"
    status_path = output_dir / "pass1_status.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run inventory.py first."
        )

    manifest_df = pd.read_csv(manifest_path)
    manifest_df = manifest_df[manifest_df["is_duplicate"].astype(str) != "True"].copy()

    _ensure_status_csv_schema(status_path)
    done_paths = set() if args.rescan else _load_done_paths(status_path)
    if args.rescan:
        to_process = manifest_df
    else:
        to_process = manifest_df[~manifest_df["file_path"].astype(str).isin(done_paths)]

    n_done_already = len(manifest_df) - len(to_process)
    n_remaining = len(to_process)

    print("═══════════════════════════════════════")
    print("Still Extractor — Pass 1")
    print(f"Run: {cfg.name}")
    print(f"Config: {args.config}")
    print(f"Manifest: {manifest_path}")
    print(f"Status:   {status_path}")
    print(f"Index:    {index_path}")
    print(f"{n_done_already} files already processed, {n_remaining} remaining.")
    print("═══════════════════════════════════════")

    if n_remaining == 0:
        print("Nothing to do.")
        return

    videos_df = to_process[to_process["file_type"] == "video"].copy()
    images_df = to_process[to_process["file_type"] == "image"].copy()

    existing_index_df: pd.DataFrame | None = None
    if index_path.exists():
        existing_index_df = pd.read_parquet(index_path)
        logger.info("Resume: %s already has %d rows", index_path, len(existing_index_df))

    output_dir.mkdir(parents=True, exist_ok=True)

    all_new_rows: list[dict] = []
    n_videos_done = 0
    n_images_done = 0
    n_failed = 0
    video_rows_total = 0
    image_rows_total = 0
    main_face_app: FaceAnalysis | None = None

    # Videos: parallel pool when workers > 1, otherwise in-process
    if len(videos_df) > 0 and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
            futures = {
                pool.submit(
                    _worker_process_video,
                    row.to_dict(), str(output_dir),
                    args.fps, args.sharpness_threshold, args.min_face_px,
                ): row["file_path"]
                for _, row in videos_df.iterrows()
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="videos"):
                file_path = futures[future]
                try:
                    meta, stats, rows = future.result()
                except Exception:
                    logger.exception("Worker crashed for %s", file_path)
                    meta = {
                        "file_path": file_path,
                        "file_hash": "",
                        "status": "failed",
                        "elapsed_s": 0.0,
                        "processed_at": datetime.now(timezone.utc).isoformat(),
                    }
                    stats = None
                    rows = []
                append_status(status_path, **meta, stats=stats)
                all_new_rows.extend(rows)
                if meta["status"] == "done":
                    n_videos_done += 1
                    video_rows_total += len(rows)
                else:
                    n_failed += 1
    elif len(videos_df) > 0:
        main_face_app = _create_face_app()
        for _, row in tqdm(videos_df.iterrows(), total=len(videos_df), desc="videos"):
            meta, stats, rows = _process_one(
                row, output_dir, main_face_app,
                args.fps, args.sharpness_threshold, args.min_face_px,
            )
            append_status(status_path, **meta, stats=stats)
            all_new_rows.extend(rows)
            if meta["status"] == "done":
                n_videos_done += 1
                video_rows_total += len(rows)
            else:
                n_failed += 1

    # Images: always single-threaded in main
    if len(images_df) > 0:
        if main_face_app is None:
            main_face_app = _create_face_app()
        for _, row in tqdm(images_df.iterrows(), total=len(images_df), desc="images"):
            meta, stats, rows = _process_one(
                row, output_dir, main_face_app,
                args.fps, args.sharpness_threshold, args.min_face_px,
            )
            append_status(status_path, **meta, stats=stats)
            all_new_rows.extend(rows)
            if meta["status"] == "done":
                n_images_done += 1
                image_rows_total += len(rows)
            else:
                n_failed += 1

    total_index_rows = _write_index(index_path, existing_index_df, all_new_rows)

    print("───────────────────────────────────────")
    print("Pass 1 complete")
    print(f"Videos processed: {n_videos_done}  →  {video_rows_total:,} candidate frames")
    print(f"Images processed: {n_images_done}  →  {image_rows_total:,} candidate frames")
    print(f"Failed: {n_failed} (see {status_path})")
    print(f"Parquet written: {index_path} ({total_index_rows:,} rows)")
    print("───────────────────────────────────────")


if __name__ == "__main__":
    main()
