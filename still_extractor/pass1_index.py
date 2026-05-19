"""Pass 1: walk source videos and index candidate frames."""

import logging
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


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


def process_video(
    video_path: Path,
    output_dir: Path,
    fps: float,
    sharpness_threshold: float,
) -> dict[str, int | float | str]:
    """Process one video, returning a stats dict."""
    frames_dir = output_dir / "frames" / video_path.stem

    if frames_dir.exists() and any(frames_dir.iterdir()):
        logger.warning("Skipping %s: %s exists and is non-empty", video_path.name, frames_dir)
        return {
            "video": str(video_path),
            "frames_sampled": 0,
            "frames_kept": 0,
            "elapsed_seconds": 0.0,
            "skipped": 1,
        }

    frames_dir.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    frames_sampled = 0
    frames_kept = 0

    try:
        for frame_index, timestamp_sec, img in sample_frames(video_path, fps):
            frames_sampled += 1
            score = sharpness_score(img)
            keep = score >= sharpness_threshold
            logger.debug(
                "%s frame=%d t=%.3fs sharpness=%.2f %s",
                video_path.name, frame_index, timestamp_sec, score,
                "KEEP" if keep else "DROP",
            )
            if not keep:
                continue
            out_path = frames_dir / f"{frame_index:06d}_{timestamp_sec:.3f}.jpg"
            cv2.imwrite(str(out_path), img)
            frames_kept += 1
    except Exception as e:
        logger.warning("Failed to process %s: %s", video_path, e)

    elapsed = time.monotonic() - start
    logger.info(
        "%s: sampled=%d kept=%d elapsed=%.1fs",
        video_path.name, frames_sampled, frames_kept, elapsed,
    )
    return {
        "video": str(video_path),
        "frames_sampled": frames_sampled,
        "frames_kept": frames_kept,
        "elapsed_seconds": elapsed,
        "skipped": 0,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
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
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel video workers.")
    args = parser.parse_args()

    if args.ffmpeg_path != "ffmpeg":
        logger.debug("--ffmpeg-path=%s provided but unused (PyAV uses bundled libs)", args.ffmpeg_path)

    videos = find_videos(args.video_dir)
    if not videos:
        logger.warning("No videos found in %s", args.video_dir)
        return

    logger.info("Found %d videos in %s", len(videos), args.video_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_sampled = 0
    total_kept = 0
    total_skipped = 0

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_video, v, args.output_dir, args.fps, args.sharpness_threshold,
                ): v for v in videos
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="videos"):
                v = futures[future]
                try:
                    stats = future.result()
                except Exception:
                    logger.exception("Worker failed for %s", v)
                    continue
                total_sampled += int(stats["frames_sampled"])
                total_kept += int(stats["frames_kept"])
                total_skipped += int(stats["skipped"])
    else:
        for v in tqdm(videos, desc="videos"):
            try:
                stats = process_video(v, args.output_dir, args.fps, args.sharpness_threshold)
            except Exception:
                logger.exception("Failed processing %s", v)
                continue
            total_sampled += int(stats["frames_sampled"])
            total_kept += int(stats["frames_kept"])
            total_skipped += int(stats["skipped"])

    logger.info(
        "Summary: videos=%d sampled=%d kept=%d skipped=%d",
        len(videos), total_sampled, total_kept, total_skipped,
    )


if __name__ == "__main__":
    main()
