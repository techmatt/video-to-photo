"""Pass 1: walk source videos and index candidate frames."""

import json
import logging
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from tqdm import tqdm

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

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


def find_videos(video_dir: Path) -> list[Path]:
    return sorted(
        p for p in video_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def center_crop_70(img: np.ndarray) -> np.ndarray:
    # Center 70% of width and height -> ~50% of area.
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


def _video_duration_seconds(container: "av.container.InputContainer", stream) -> float | None:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    return None


def sample_frames(video_path: Path, fps: float):
    """Yield (frame_index, timestamp_seconds, bgr_array) by seeking to target timestamps."""
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


def _bbox_width(bbox: np.ndarray) -> float:
    return float(bbox[2] - bbox[0])


def _bbox_area(bbox: np.ndarray) -> float:
    return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def process_video(
    video_path: Path,
    output_dir: Path,
    fps: float,
    sharpness_threshold: float,
    min_face_px: float,
    face_app: FaceAnalysis | None = None,
) -> tuple[dict[str, int | float | str], list[dict]]:
    """Process one video, returning (stats, rows)."""
    if face_app is None:
        face_app = _face_app
    if face_app is None:
        raise RuntimeError("FaceAnalysis app not initialized in this process")

    frames_dir = output_dir / "frames" / video_path.stem

    if frames_dir.exists() and any(frames_dir.iterdir()):
        logger.warning("Skipping %s: %s exists and is non-empty", video_path.name, frames_dir)
        return {
            "video": str(video_path),
            "frames_sampled": 0,
            "frames_kept": 0,
            "faces_found": 0,
            "elapsed_seconds": 0.0,
            "skipped": 1,
        }, []

    frames_dir.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    frames_sampled = 0
    frames_kept = 0
    faces_found = 0
    rows: list[dict] = []
    video_abs = str(video_path.resolve())

    try:
        for frame_index, timestamp_sec, img in sample_frames(video_path, fps):
            frames_sampled += 1
            score = sharpness_score(img)
            if score < sharpness_threshold:
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
                continue

            qualifying = [f for f in faces if _bbox_width(f.bbox) >= min_face_px]
            if not qualifying:
                logger.debug(
                    "%s frame=%d t=%.3fs sharpness=%.2f DROP-no-face",
                    video_path.name, frame_index, timestamp_sec, score,
                )
                continue

            largest = max(qualifying, key=lambda f: _bbox_area(f.bbox))

            out_path = frames_dir / f"{frame_index:06d}_{timestamp_sec:.3f}.jpg"
            cv2.imwrite(str(out_path), img)
            frames_kept += 1
            faces_found += 1

            h, w = img.shape[:2]
            bbox = largest.bbox
            kps = largest.kps
            embedding = largest.normed_embedding

            rows.append({
                "video_path": video_abs,
                "video_stem": video_path.stem,
                "frame_index": int(frame_index),
                "timestamp_s": float(timestamp_sec),
                "frame_path": str(out_path.resolve()),
                "frame_w": int(w),
                "frame_h": int(h),
                "sharpness_center": score,
                "face_x1": float(bbox[0]),
                "face_y1": float(bbox[1]),
                "face_x2": float(bbox[2]),
                "face_y2": float(bbox[3]),
                "face_w": float(bbox[2] - bbox[0]),
                "face_det_score": float(largest.det_score),
                "kps": json.dumps([[float(x), float(y)] for x, y in kps]),
                "embedding": json.dumps([float(v) for v in embedding]),
            })

            logger.debug(
                "%s frame=%d t=%.3fs sharpness=%.2f KEEP faces=%d largest_w=%.1f",
                video_path.name, frame_index, timestamp_sec, score,
                len(qualifying), _bbox_width(bbox),
            )
    except Exception as e:
        logger.warning("Failed to process %s: %s", video_path, e)

    elapsed = time.monotonic() - start
    logger.info(
        "%s: sampled=%d kept=%d faces=%d elapsed=%.1fs",
        video_path.name, frames_sampled, frames_kept, faces_found, elapsed,
    )
    return {
        "video": str(video_path),
        "frames_sampled": frames_sampled,
        "frames_kept": frames_kept,
        "faces_found": faces_found,
        "elapsed_seconds": elapsed,
        "skipped": 0,
    }, rows


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


def main() -> None:
    parser = ArgumentParser(description="Walk source videos and index candidate frames.")
    parser.add_argument("--video-dir", type=Path, required=True,
                        help="Directory to scan recursively for videos.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Root output directory; frames go to {output-dir}/frames/{video_stem}/.")
    parser.add_argument("--ffmpeg-path", type=str, default="ffmpeg",
                        help="ffmpeg executable path. Reserved for future use; PyAV uses its bundled libs.")
    parser.add_argument("--fps", type=float, default=3.0,
                        help="Sample rate in frames per second.")
    parser.add_argument("--sharpness-threshold", type=float, default=100.0,
                        help="Laplacian variance threshold; frames below this are dropped.")
    parser.add_argument("--min-face-px", type=float, default=80.0,
                        help="Minimum face bounding box width in pixels; faces below this are dropped.")
    parser.add_argument("--index-file", type=Path, default=Path("data/index.parquet"),
                        help="Output Parquet file for per-frame index rows.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel video workers.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity. DEBUG prints per-frame keep/drop decisions.")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    if args.ffmpeg_path != "ffmpeg":
        logger.debug("--ffmpeg-path=%s provided but unused (PyAV uses bundled libs)", args.ffmpeg_path)

    existing_df: pd.DataFrame | None = None
    if args.index_file.exists():
        existing_df = pd.read_parquet(args.index_file)
        logger.info("Resume: %s already has %d rows", args.index_file, len(existing_df))

    videos = find_videos(args.video_dir)
    if not videos:
        logger.warning("No videos found in %s", args.video_dir)
        return

    logger.info("Found %d videos in %s", len(videos), args.video_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_sampled = 0
    total_kept = 0
    total_faces = 0
    total_skipped = 0
    all_rows: list[dict] = []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
            futures = {
                pool.submit(
                    process_video,
                    v, args.output_dir, args.fps, args.sharpness_threshold, args.min_face_px,
                ): v for v in videos
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="videos"):
                v = futures[future]
                try:
                    stats, rows = future.result()
                except Exception:
                    logger.exception("Worker failed for %s", v)
                    continue
                total_sampled += int(stats["frames_sampled"])
                total_kept += int(stats["frames_kept"])
                total_faces += int(stats["faces_found"])
                total_skipped += int(stats["skipped"])
                all_rows.extend(rows)
    else:
        face_app = _create_face_app()
        for v in tqdm(videos, desc="videos"):
            try:
                stats, rows = process_video(
                    v, args.output_dir, args.fps, args.sharpness_threshold,
                    args.min_face_px, face_app,
                )
            except Exception:
                logger.exception("Failed processing %s", v)
                continue
            total_sampled += int(stats["frames_sampled"])
            total_kept += int(stats["frames_kept"])
            total_faces += int(stats["faces_found"])
            total_skipped += int(stats["skipped"])
            all_rows.extend(rows)

    total_index_rows = _write_index(args.index_file, existing_df, all_rows)

    logger.info(
        "Summary: videos=%d sampled=%d kept=%d faces=%d skipped=%d index_rows=%d",
        len(videos), total_sampled, total_kept, total_faces, total_skipped, total_index_rows,
    )
    logger.info("Wrote %d total rows to %s", total_index_rows, args.index_file)


if __name__ == "__main__":
    main()
