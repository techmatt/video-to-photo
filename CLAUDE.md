# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Windows + NVIDIA GPU, Python 3.12+, `uv` for env management. `uv sync` pulls the CUDA 12.4 build of torch/torchvision/torchaudio via the explicit `pytorch-cu124` index defined in `pyproject.toml` — don't replace those entries with PyPI defaults or you'll silently lose GPU support.

InsightFace `buffalo_l` weights are non-commercial research only. Personal/family use is the design target; don't redistribute outputs.

## Running the pipeline

Each stage is a module under `still_extractor/`, run via:

```
uv run python -m still_extractor.<module> --config configs/<run>.yaml
```

Order, with the intermediate artifact each one consumes/produces (all under `<output_dir>/` from the YAML config):

| Stage | Module | Reads | Writes |
|---|---|---|---|
| 0 | `inventory` | `dirs_file` from YAML | `manifest.csv` |
| 1 | `pass1_index` | `manifest.csv` | `index.parquet`, `pass1_status.csv`, `pass1_summary.json`, `frames/<stem>/*.jpg` |
| 1b | `cluster_faces` | `index.parquet` (embeddings) | identity clusters (stub today) |
| 2 | `pass2_score` | `index.parquet` | `scores.csv`, `pass2_summary.json`, `top_frames/*.jpg` |
| 3 | `pass3_refine` | `scores.csv` | `refined_scores.csv`, `pass3_summary.json` (+ refined frames) |
| 4 | `build_index_html` | `refined_scores.csv` (falls back to `scores.csv`) | `index.html` (self-contained, base64-embedded face crops), `build_index_summary.json` |
| 5 | `build_photo_viewer` | `refined_scores.csv` + `index.parquet` | `index_photos.html`, `build_photo_viewer_summary.json` |

## Configuration model

Every stage accepts `--config <yaml>` (e.g. `configs/mini.yaml`) loaded via `RunConfig.from_yaml`. The YAML names a run, points at a `dirs_file` (one source directory per line, `#` for comments), sets long-video sampling parameters, and chooses the output directory. Per-stage path flags (`--index-file`, `--scores-csv`, `--output-dir`, `--output-html`, `--parquet`) are optional overrides that default from `output_dir` when `--config` is supplied; per-stage tuning knobs (`--top-k-per-file`, `--classifier-model`, `--window-s`, …) remain explicit flags only.

## Architecture notes worth knowing before editing

**Manifest is the contract for what to process.** `inventory.py` crawls source dirs, hashes the first 64 KiB of each file with MD5, marks duplicates (same hash) with a canonical-path pointer, probes video durations, and pre-computes `sample_windows_s` for long videos (>`long_video_threshold_s`). Pass 1 reads `manifest.csv`, filters `is_duplicate == True`, and processes the rest. Sort order matters: the manifest is sorted by `size_bytes` ascending so small files process first and give early signal.

**Pass 1 sampling has two modes**, selected per row:
- Short videos: `sample_frames()` walks the whole duration at `--fps` (default 3 fps).
- Long videos: `sample_frames_windowed()` decodes 1-second windows at the pre-computed `sample_windows_s` offsets only. Windows are seeded from the file hash so the same file gets the same windows across runs.

**Resume is driven by `pass1_status.csv`.** Each processed file appends one row with per-gate counters (sharpness/face-detect/face-size), sharpness diagnostics, and `status` (`done`/`failed`). On startup pass 1 skips any `file_path` whose latest row is `done` unless `--rescan` is set. If an older-schema status CSV is present, `_ensure_status_csv_schema()` rewrites it with the current `STATUS_COLUMNS` before any append — necessary because `csv.DictWriter` would otherwise produce a column-count mismatch.

**Gate ordering in pass 1 is sharpness → face-detect → face-size**, and every gate has a dedicated counter on the `FileStats` dataclass. When debugging "why did this file produce zero rows", read the relevant `pass1_status.csv` row — `frames_failed_*` tells you which gate dropped them.

**Pass 1 GPU parallelism.** Videos can run through a `ProcessPoolExecutor` (`--workers N`) where each worker initializes its own `FaceAnalysis` via `_worker_init`; images always run single-threaded in the main process. The module-level `_face_app` is the per-process singleton — don't try to share it across workers.

**The Parquet index is the join key for everything downstream.** Pass 1 dedups on `frame_path` when merging existing + new rows (`drop_duplicates(subset=["frame_path"], keep="last")`), so re-running a file safely overwrites its old rows.

## Coding conventions in use

- Python 3.12+ syntax: `|` unions, built-in generics (`list[...]`, `dict[...]`), `pathlib.Path` everywhere.
- Logging via module-level `logger = logging.getLogger(__name__)`, configured once in `main()` with `force=True`. INFO-level per-file lines use ASCII separators (not em dashes) because logging writes to stderr, which isn't reconfigured to UTF-8.
- `argparse` directly; no click/typer.
- Flat module layout under `still_extractor/`; shared helpers live in their own small modules (e.g. `face_crop.py`) rather than a `utils` grab-bag.
- One CLI entry point per stage. Each module's `main()` is the entry; tests/scripts can import the helpers directly.

## Docs to consult

- `family_video_still_extractor_design.md` — full design rationale (why aesthetics-only fails, scoring model, identity-clustering plan).
- `docs/prompt*.md` — the historical prompt series that built each stage. New stage work tends to follow the same prompt-doc pattern.
