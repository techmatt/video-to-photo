"""Pass 3: micro-window refinement around top-K candidates from Pass 2."""

import logging
from argparse import ArgumentParser
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _face_sharpness_bgr(
    img: np.ndarray, x1: float, y1: float, x2: float, y2: float, padding: int = 10,
) -> float:
    h, w = img.shape[:2]
    cx1 = max(0, int(x1) - padding)
    cy1 = max(0, int(y1) - padding)
    cx2 = min(w, int(x2) + padding)
    cy2 = min(h, int(y2) + padding)
    if cx2 <= cx1 or cy2 <= cy1:
        return 0.0
    crop = img[cy1:cy2, cx1:cx2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _decode_window(
    video_path: Path, target_s: float, window_s: float,
) -> list[tuple[float, np.ndarray]]:
    """Return [(actual_ts_s, bgr_frame), ...] for frames in [target_s - window_s, target_s + window_s]."""
    out: list[tuple[float, np.ndarray]] = []
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base
        if time_base is None:
            logger.warning("No time_base for %s; skipping", video_path)
            return out

        time_base_f = float(time_base)
        start_s = max(0.0, target_s - window_s)
        end_s = target_s + window_s
        seek_pts = int(start_s / time_base_f)

        try:
            container.seek(seek_pts, stream=stream, any_frame=False, backward=True)
        except Exception as e:
            logger.warning("Seek failed at %.3fs in %s: %s", start_s, video_path.name, e)
            return out

        try:
            for decoded in container.decode(stream):
                if decoded.pts is None:
                    continue
                actual_s = float(decoded.pts * time_base)
                if actual_s < start_s - 1e-6:
                    continue
                if actual_s > end_s + 1e-6:
                    break
                out.append((actual_s, decoded.to_ndarray(format="bgr24")))
        except Exception as e:
            logger.warning(
                "Decode failed near %.3fs in %s: %s", target_s, video_path.name, e,
            )
    finally:
        container.close()
    return out


def _refine_candidate(
    row: pd.Series, window_s: float, refined_dir: Path,
) -> dict | None:
    video_path = Path(row["video_path"])
    if not video_path.exists():
        logger.warning("Video not found: %s; skipping", video_path)
        return None

    try:
        frames = _decode_window(video_path, float(row["timestamp_s"]), window_s)
    except Exception as e:
        logger.warning("Failed to decode window for %s: %s", video_path, e)
        return None

    if not frames:
        logger.warning(
            "No frames decoded in window for %s @ %.3fs", video_path.name, row["timestamp_s"],
        )
        return None

    x1, y1, x2, y2 = (
        float(row["face_x1"]), float(row["face_y1"]),
        float(row["face_x2"]), float(row["face_y2"]),
    )

    sharpness = [_face_sharpness_bgr(img, x1, y1, x2, y2) for _, img in frames]
    best_idx = int(np.argmax(sharpness))
    best_ts, best_img = frames[best_idx]
    best_sharp = float(sharpness[best_idx])

    original_ts = float(row["timestamp_s"])
    closest_idx = int(np.argmin([abs(ts - original_ts) for ts, _ in frames]))
    original_sharp = float(sharpness[closest_idx])

    if abs(best_idx - closest_idx) <= 1:
        logger.debug(
            "No meaningful improvement for %s @ %.3fs (best_idx=%d closest_idx=%d)",
            video_path.name, original_ts, best_idx, closest_idx,
        )

    out_path = refined_dir / (
        f"{float(row['composite']):.4f}_{row['video_stem']}_"
        f"{original_ts:.3f}_refined.jpg"
    )
    cv2.imwrite(str(out_path), best_img)

    return {
        "refined_frame_path": str(out_path.resolve()),
        "refined_timestamp_s": best_ts,
        "refined_sharpness": best_sharp,
        "original_sharpness": original_sharp,
        "sharpness_delta": best_sharp - original_sharp,
    }


def main() -> None:
    parser = ArgumentParser(
        description="Refine top-K candidates by picking the sharpest frame in a micro-window.",
    )
    parser.add_argument("--scores-csv", type=Path, default=Path("data/scores.csv"),
                        help="Path to scores.csv from Pass 2.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"),
                        help="Root output dir; refined JPEGs go to {output-dir}/refined/.")
    parser.add_argument("--ffmpeg-path", type=str, default="ffmpeg",
                        help="ffmpeg executable path. Reserved; PyAV uses bundled libs.")
    parser.add_argument("--window-s", type=float, default=0.5,
                        help="Half-window in seconds around candidate timestamp.")
    parser.add_argument("--top-k", type=int, default=200,
                        help="How many top candidates to refine (by composite score).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    if args.ffmpeg_path != "ffmpeg":
        logger.debug(
            "--ffmpeg-path=%s provided but unused (PyAV uses bundled libs)",
            args.ffmpeg_path,
        )

    df = pd.read_csv(args.scores_csv)
    logger.info("Loaded %d rows from %s", len(df), args.scores_csv)

    if "dedup_kept" not in df.columns:
        logger.error("scores.csv missing dedup_kept column")
        return

    kept = df[df["dedup_kept"].astype(bool)].copy()
    kept = kept.sort_values("composite", ascending=False).head(args.top_k).reset_index(drop=True)
    logger.info("Refining top %d candidates (post-dedup)", len(kept))

    refined_dir = args.output_dir / "refined"
    refined_dir.mkdir(parents=True, exist_ok=True)

    refined_cols = {
        "refined_frame_path": [],
        "refined_timestamp_s": [],
        "refined_sharpness": [],
        "original_sharpness": [],
        "sharpness_delta": [],
    }

    for _, row in tqdm(kept.iterrows(), total=len(kept), desc="refine"):
        result = _refine_candidate(row, args.window_s, refined_dir)
        if result is None:
            for k in refined_cols:
                refined_cols[k].append(None)
        else:
            for k in refined_cols:
                refined_cols[k].append(result[k])

    for k, vals in refined_cols.items():
        kept[k] = vals

    deltas = kept["sharpness_delta"].dropna().astype(float)
    if len(deltas):
        logger.info(
            "sharpness_delta: mean=%.2f max=%.2f min=%.2f n=%d",
            float(deltas.mean()), float(deltas.max()), float(deltas.min()), len(deltas),
        )
    else:
        logger.info("sharpness_delta: no successful refinements")

    out_csv = args.output_dir / "refined_scores.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_csv, index=False)
    logger.info("Wrote %d rows to %s", len(kept), out_csv)


if __name__ == "__main__":
    main()
