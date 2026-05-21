"""Frame sampling, video rotation, sharpness — shared by worker and trainers."""

import logging
import math
import random
import struct
from pathlib import Path
from typing import Iterator

import av
import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Video display-matrix rotation
#
# PyAV 17 / current ffmpeg builds no longer expose `stream.metadata["rotate"]`
# or display-matrix side data. iPhone clips encode rotation in the QuickTime
# tkhd display matrix, so we read the box tree directly.

_MOOV_SCAN_BYTES = 8 * 1024 * 1024  # bytes to scan from each end of file


def _read_atom_header(buf: bytes, i: int, end: int):
    if i + 8 > end:
        return None
    size = struct.unpack(">I", buf[i:i + 4])[0]
    atype = buf[i + 4:i + 8]
    header = 8
    if size == 1:
        if i + 16 > end:
            return None
        size = struct.unpack(">Q", buf[i + 8:i + 16])[0]
        header = 16
    elif size == 0:
        size = end - i
    if size < header or i + size > end:
        return None
    return atype, header, size


def _iter_children(buf: bytes, start: int, end: int):
    i = start
    while i < end:
        h = _read_atom_header(buf, i, end)
        if h is None:
            return
        atype, header, size = h
        yield atype, i + header, i + size
        i += size


def _find_boxes(buf: bytes, start: int, end: int, want: bytes, recurse_into=None):
    for atype, bstart, bend in _iter_children(buf, start, end):
        if atype == want:
            yield bstart, bend
        elif recurse_into and atype in recurse_into:
            yield from _find_boxes(buf, bstart, bend, want, recurse_into)


def _moov_buffer(path: Path) -> bytes | None:
    """Return enough bytes to cover the moov atom (head, or tail for iPhone files)."""
    sz = path.stat().st_size
    with open(path, "rb") as f:
        head_len = min(sz, _MOOV_SCAN_BYTES)
        head = f.read(head_len)
        if any(True for _ in _find_boxes(head, 0, len(head), b"moov")):
            return head
        tail_len = min(sz, _MOOV_SCAN_BYTES)
        f.seek(sz - tail_len)
        tail = f.read()
        magic = tail.find(b"moov")
        if magic < 4:
            return None
        return tail[magic - 4:]


def get_video_rotation(path: Path) -> int:
    """Return clockwise rotation (0/90/180/270) for the video track, 0 if none."""
    try:
        buf = _moov_buffer(path)
        if buf is None:
            return 0
        for mstart, mend in _find_boxes(buf, 0, len(buf), b"moov"):
            for tstart, tend in _find_boxes(buf, mstart, mend, b"trak"):
                handler_type = None
                for hstart, hend in _find_boxes(
                    buf, tstart, tend, b"hdlr", recurse_into={b"mdia"},
                ):
                    body = buf[hstart:hend]
                    if len(body) >= 16:
                        handler_type = body[8:12]
                    break
                if handler_type != b"vide":
                    continue
                for kstart, kend in _find_boxes(buf, tstart, tend, b"tkhd"):
                    body = buf[kstart:kend]
                    if not body:
                        continue
                    version = body[0]
                    matrix_offset = 40 if version == 0 else 52
                    if len(body) < matrix_offset + 36:
                        continue
                    m = struct.unpack(">9i", body[matrix_offset:matrix_offset + 36])
                    a = m[0] / 65536.0
                    b = m[1] / 65536.0
                    if abs(a) < 1e-6 and abs(b) < 1e-6:
                        return 0
                    deg = math.degrees(math.atan2(b, a))
                    return int(round(deg / 90)) * 90 % 360
    except Exception as e:
        logger.warning("Failed to read rotation for %s: %s", path.name, e)
    return 0


_CV2_ROTATE_FOR_DEG = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _apply_rotation(img: np.ndarray, rotation_deg: int) -> np.ndarray:
    flag = _CV2_ROTATE_FOR_DEG.get(rotation_deg)
    if flag is None:
        return img
    return cv2.rotate(img, flag)


def _center_crop_70(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    crop_h = int(h * 0.7)
    crop_w = int(w * 0.7)
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    return img[y0:y0 + crop_h, x0:x0 + crop_w]


def sharpness_score(img: np.ndarray) -> float:
    """Laplacian variance over the center 70% crop."""
    cropped = _center_crop_70(img)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _video_duration_seconds(container, stream) -> float | None:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    return None


def sample_frames(
    video_path: Path, fps: float, rotation: int | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_s, bgr_ndarray) over the full duration at `fps`.

    Rotation comes from the QuickTime tkhd display matrix; if `rotation` is None
    we probe it from the file. Frames are returned already in display orientation.
    """
    if rotation is None:
        rotation = get_video_rotation(video_path)
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
            img = _apply_rotation(img, rotation)
            yield i, actual_sec, img
    finally:
        container.close()


def sample_frames_windowed(
    video_path: Path, windows: list[int], fps: float, rotation: int | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield frames within [t, t+1.0) for each t in `windows` at `fps`."""
    if rotation is None:
        rotation = get_video_rotation(video_path)
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
                img = chosen.to_ndarray(format="bgr24")
                img = _apply_rotation(img, rotation)
                yield idx, actual_sec, img
                idx += 1
    finally:
        container.close()


def decode_window(
    video_path: Path, target_s: float, window_s: float, rotation: int | None = None,
) -> list[tuple[float, np.ndarray]]:
    """Return [(actual_ts_s, bgr_frame), ...] for frames in [target_s - window_s, target_s + window_s].

    Used by micro-window refinement. Applies container rotation to each decoded frame.
    """
    if rotation is None:
        rotation = get_video_rotation(video_path)
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
                img = decoded.to_ndarray(format="bgr24")
                img = _apply_rotation(img, rotation)
                out.append((actual_s, img))
        except Exception as e:
            logger.warning("Decode failed near %.3fs in %s: %s", target_s, video_path.name, e)
    finally:
        container.close()
    return out


def compute_sample_windows(
    duration_s: float, n_windows: int, min_spacing_s: float, seed_hash: int,
) -> list[int]:
    """Pre-compute deterministic 1-second window starts for long videos."""
    last_candidate = int(duration_s) - 1
    if last_candidate <= 0:
        return []
    candidates = list(range(0, last_candidate))
    rng = random.Random(seed_hash)
    rng.shuffle(candidates)
    accepted: list[int] = []
    for c in candidates:
        if all(abs(c - a) >= min_spacing_s for a in accepted):
            accepted.append(c)
            if len(accepted) >= n_windows:
                break
    accepted.sort()
    return accepted
