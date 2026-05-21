"""Inventory pass: crawl source directories and build manifest.csv."""

import csv
import hashlib
import json
import logging
import random
import statistics
import sys
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import av
import pillow_heif
import yaml
from tqdm import tqdm

from still_extractor.constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

pillow_heif.register_heif_opener()

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = [
    "file_path",
    "file_type",
    "extension",
    "size_bytes",
    "hash",
    "is_duplicate",
    "canonical_path",
    "duration_s",
    "is_long_video",
    "sample_windows_s",
    "run_name",
    "scanned_at",
]


@dataclass
class RunConfig:
    name: str
    dirs_file: Path
    long_video_threshold_s: float
    long_video_windows: int
    long_video_min_spacing_s: float
    output_dir: Path

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            name=data["name"],
            dirs_file=Path(data["dirs_file"]),
            long_video_threshold_s=float(data.get("long_video_threshold_s", 60)),
            long_video_windows=int(data.get("long_video_windows", 20)),
            long_video_min_spacing_s=float(data.get("long_video_min_spacing_s", 5)),
            output_dir=Path(data["output_dir"]),
        )


def load_dirs(dirs_file: Path) -> list[Path]:
    dirs: list[Path] = []
    with open(dirs_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            dirs.append(Path(line))
    return dirs


def classify_file(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    return None


def crawl_directory(directory: Path) -> list[Path]:
    found: list[Path] = []
    for p in directory.rglob("*"):
        if not p.is_file():
            continue
        if classify_file(p) is None:
            continue
        found.append(p.resolve())
    return found


def file_hash(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read(65536)).hexdigest()


def probe_video_duration(path: Path) -> float | None:
    try:
        with av.open(str(path)) as container:
            if container.duration is None:
                return None
            return float(container.duration) / av.time_base
    except Exception as e:
        logger.warning("Failed to probe duration for %s: %s", path, e)
        return None


def compute_sample_windows(
    duration_s: float,
    n_windows: int,
    min_spacing_s: float,
    seed: int,
) -> list[int]:
    last_candidate = int(duration_s) - 1
    if last_candidate <= 0:
        return []
    candidates = list(range(0, last_candidate))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    accepted: list[int] = []
    for c in candidates:
        if all(abs(c - a) >= min_spacing_s for a in accepted):
            accepted.append(c)
            if len(accepted) >= n_windows:
                break
    accepted.sort()
    return accepted


def load_existing_manifest(manifest_path: Path) -> tuple[list[dict], set[str]]:
    if not manifest_path.exists():
        return [], set()
    rows: list[dict] = []
    paths: set[str] = set()
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            paths.add(row["file_path"])
    return rows, paths


def write_manifest(manifest_path: Path, rows: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_bytes(n: int) -> str:
    for unit, threshold in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if n >= threshold:
            v = n / threshold
            return f"{v:.1f} {unit}" if v < 10 else f"{int(round(v))} {unit}"
    return f"{n} B"


def _to_bool_str(v) -> str:
    if isinstance(v, str):
        return "True" if v == "True" else "False"
    return "True" if v else "False"


def _serialize_row(row: dict) -> dict:
    out = dict(row)
    if out.get("duration_s") is None:
        out["duration_s"] = ""
    if out.get("sample_windows_s") is None:
        out["sample_windows_s"] = ""
    out["is_duplicate"] = _to_bool_str(out["is_duplicate"])
    out["is_long_video"] = _to_bool_str(out["is_long_video"])
    return out


def main() -> None:
    parser = ArgumentParser(description="Build a manifest of all video and image files for a run.")
    parser.add_argument("--config", type=Path, required=True, help="Path to run YAML config file.")
    parser.add_argument("--rescan", action="store_true",
                        help="Re-crawl all directories and rebuild the manifest from scratch.")
    args = parser.parse_args()

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    cfg = RunConfig.from_yaml(args.config)
    manifest_path = cfg.output_dir / "manifest.csv"

    print("═══════════════════════════════════════")
    print("Still Extractor — Inventory")
    print(f"Run: {cfg.name}")
    print(f"Dirs file: {cfg.dirs_file}")
    print(f"Output: {manifest_path}")
    print("═══════════════════════════════════════")

    existing_rows: list[dict] = []
    existing_paths: set[str] = set()
    if not args.rescan:
        existing_rows, existing_paths = load_existing_manifest(manifest_path)

    dirs = load_dirs(cfg.dirs_file)
    all_paths: list[Path] = []
    for d in dirs:
        print(f"📁  Scanning: {d}")
        if not d.exists():
            logger.warning("Directory does not exist: %s", d)
            continue
        all_paths.extend(crawl_directory(d))

    seen_in_crawl: set[str] = set()
    unique_paths: list[Path] = []
    for p in all_paths:
        sp = str(p)
        if sp in seen_in_crawl:
            continue
        seen_in_crawl.add(sp)
        unique_paths.append(p)

    n_images_total = sum(1 for p in unique_paths if classify_file(p) == "image")
    n_videos_total = sum(1 for p in unique_paths if classify_file(p) == "video")

    new_paths = [p for p in unique_paths if str(p) not in existing_paths]
    n_already = len(unique_paths) - len(new_paths)

    print(
        f"Found {len(unique_paths)} files "
        f"({n_images_total} images, {n_videos_total} videos) "
        f"across {len(dirs)} directories."
    )
    print(f"Already in manifest: {n_already}. New files to process: {len(new_paths)}.")

    if not new_paths:
        print("Nothing new to process.")
        rows_out = [_serialize_row(r) for r in existing_rows]
        rows_out.sort(key=lambda r: int(r["size_bytes"]))
        write_manifest(manifest_path, rows_out)
        return

    new_paths.sort(key=lambda p: str(p))

    hashes: dict[Path, str] = {}
    sizes: dict[Path, int] = {}
    for p in tqdm(new_paths, desc="Hashing files"):
        try:
            hashes[p] = file_hash(p)
            sizes[p] = p.stat().st_size
        except Exception as e:
            logger.warning("Failed to hash %s: %s", p, e)

    hashable_paths = [p for p in new_paths if p in hashes]

    existing_hash_to_canonical: dict[str, str] = {}
    for r in existing_rows:
        h = r.get("hash")
        if not h:
            continue
        is_dup = r.get("is_duplicate", "False") == "True"
        if not is_dup and h not in existing_hash_to_canonical:
            existing_hash_to_canonical[h] = r["file_path"]

    hash_groups: dict[str, list[Path]] = defaultdict(list)
    for p in hashable_paths:
        hash_groups[hashes[p]].append(p)

    is_duplicate: dict[Path, bool] = {}
    canonical_path: dict[Path, str] = {}
    duplicate_report: list[tuple[str, str, list[str]]] = []

    for h, group in hash_groups.items():
        group_sorted = sorted(group, key=lambda p: str(p))
        existing_canonical = existing_hash_to_canonical.get(h)
        if existing_canonical is not None:
            for p in group_sorted:
                is_duplicate[p] = True
                canonical_path[p] = existing_canonical
            if len(group_sorted) >= 1:
                duplicate_report.append((
                    h,
                    existing_canonical,
                    [str(p) for p in group_sorted],
                ))
        else:
            canonical = group_sorted[0]
            for i, p in enumerate(group_sorted):
                is_duplicate[p] = i != 0
                canonical_path[p] = str(canonical)
            if len(group_sorted) > 1:
                duplicate_report.append((
                    h,
                    str(canonical),
                    [str(p) for p in group_sorted[1:]],
                ))

    n_dup_files = sum(1 for p in hashable_paths if is_duplicate.get(p, False))
    n_dup_hash_groups = sum(1 for _, _, skipped in duplicate_report if len(skipped) >= 1)

    print(
        f"Duplicates found: {n_dup_files} files "
        f"({n_dup_hash_groups} unique hashes with 2+ copies)."
    )
    for h, kept, skipped in duplicate_report[:20]:
        copies = len(skipped) + 1
        skipped_str = ", ".join(skipped[:3])
        if len(skipped) > 3:
            skipped_str += f", ... (+{len(skipped) - 3} more)"
        print(f"  hash {h[:8]}... → {copies} copies: kept {kept}, skipped {skipped_str}")
    if len(duplicate_report) > 20:
        print(f"  ... and {len(duplicate_report) - 20} more duplicate groups")

    canonical_videos = [
        p for p in hashable_paths
        if classify_file(p) == "video" and not is_duplicate[p]
    ]
    duration_by_path: dict[Path, float | None] = {}
    for p in tqdm(canonical_videos, desc="Probing videos"):
        duration_by_path[p] = probe_video_duration(p)

    long_video_report: list[tuple[Path, float, int]] = []
    windows_by_path: dict[Path, list[int]] = {}
    for p in canonical_videos:
        d = duration_by_path.get(p)
        if d is None or d <= cfg.long_video_threshold_s:
            continue
        seed = int(hashes[p][:8], 16) % (2 ** 32)
        windows = compute_sample_windows(
            duration_s=d,
            n_windows=cfg.long_video_windows,
            min_spacing_s=cfg.long_video_min_spacing_s,
            seed=seed,
        )
        windows_by_path[p] = windows
        long_video_report.append((p, d, len(windows)))

    n_long = len(long_video_report)
    n_total_canonical_videos = len(canonical_videos)
    print(f"Long videos (> {int(cfg.long_video_threshold_s)}s): {n_long} of {n_total_canonical_videos}")
    for p, d, n_win in long_video_report[:20]:
        print(f"  {p.name}  {int(d)}s → {n_win} windows pre-computed")
    if n_long > 20:
        print(f"  ... and {n_long - 20} more long videos")

    scanned_at = datetime.now(timezone.utc).isoformat()
    new_rows: list[dict] = []
    for p in hashable_paths:
        file_type = classify_file(p)
        duration = duration_by_path.get(p)
        is_long = (
            file_type == "video"
            and duration is not None
            and duration > cfg.long_video_threshold_s
            and not is_duplicate[p]
        )
        windows = windows_by_path.get(p) if is_long else None
        new_rows.append({
            "file_path": str(p),
            "file_type": file_type,
            "extension": p.suffix.lower(),
            "size_bytes": sizes[p],
            "hash": hashes[p],
            "is_duplicate": is_duplicate[p],
            "canonical_path": canonical_path[p],
            "duration_s": duration if file_type == "video" else None,
            "is_long_video": is_long,
            "sample_windows_s": json.dumps(windows) if windows is not None else None,
            "run_name": cfg.name,
            "scanned_at": scanned_at,
        })

    all_rows = existing_rows + [_serialize_row(r) for r in new_rows]
    all_rows.sort(key=lambda r: int(r["size_bytes"]))
    write_manifest(manifest_path, all_rows)

    def _not_dup(r: dict) -> bool:
        v = r["is_duplicate"]
        return v is False or v == "False"

    n_unique = sum(1 for r in all_rows if _not_dup(r))
    n_images_unique = sum(
        1 for r in all_rows if r["file_type"] == "image" and _not_dup(r)
    )
    n_videos_unique = sum(
        1 for r in all_rows if r["file_type"] == "video" and _not_dup(r)
    )
    n_dup_total = sum(1 for r in all_rows if not _not_dup(r))
    n_long_windowed = n_long
    n_short_full = n_total_canonical_videos - n_long

    print("───────────────────────────────────────")
    print(f"Manifest written: {manifest_path}")
    print(
        f"Total unique files: {n_unique} "
        f"({n_images_unique} images, {n_videos_unique} videos — {n_dup_total} duplicates skipped)"
    )
    print(f"Long videos: {n_long_windowed} windowed, {n_short_full} processed in full")
    sizes_all = [int(r["size_bytes"]) for r in all_rows]
    if sizes_all:
        median_size = int(statistics.median(sizes_all))
        print(
            f"File size range: {_format_bytes(min(sizes_all))} → "
            f"{_format_bytes(max(sizes_all))} (median {_format_bytes(median_size)})"
        )
    print("───────────────────────────────────────")


if __name__ == "__main__":
    main()
