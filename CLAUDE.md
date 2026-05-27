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
| 1 | `pipeline` | `manifest.csv` | `results.parquet`, `pipeline_status.csv`, `pipeline_summary.json`, `kept/*.jpg` |
| 2 | `build_faces_review` | `results.parquet` | `faces_review.html` (self-contained, base64-embedded face crops), `build_faces_review_summary.json` |
| 3 | `build_photo_viewer` | `results.parquet` | `index_photos.html`, `build_photo_viewer_summary.json` |

`train_classifier.py` and `save_labeled_faces.py` also consume `results.parquet`; both accept `--config <yaml>` and derive the parquet path from `output_dir`.

## Configuration model

Every stage accepts `--config <yaml>` (e.g. `configs/mini.yaml`) loaded via `RunConfig.from_yaml`. The YAML names a run, points at a `dirs_file` (one source directory per line, `#` for comments), sets long-video sampling parameters, and chooses the output directory. Per-stage path flags (`--results`, `--output-dir`, `--output-html`) are optional overrides that default from `output_dir` when `--config` is supplied; per-stage tuning knobs (`--fps`, `--max-per-file`, `--max-per-video`, `--quality-threshold`, …) remain explicit flags only.

## Architecture notes worth knowing before editing

**Manifest is the contract for what to process.** `inventory.py` crawls source dirs, hashes the first 64 KiB of each file with MD5, marks duplicates (same hash) with a canonical-path pointer, probes video durations, and pre-computes `sample_windows_s` for long videos (>`long_video_threshold_s`). `pipeline.py` reads `manifest.csv`, filters `is_duplicate == True`, and processes the rest. Sort order matters: the manifest is sorted by `size_bytes` ascending so small files process first and give early signal.

**Per-file worker is the unit of work.** `pipeline.py` iterates manifest rows and calls `worker.process_file` once per row. Each call handles one video or image end-to-end: decode → uprighter rotation → sharpness/face-detect gates → aesthetic + classifier scoring → per-file dHash dedup (face then full frame) → quality threshold → per-file cap → refine pass on ±`refine_window_s` for video → write keeper JPEGs. The orchestrator never writes intermediate per-frame artifacts; only final keepers land on disk under `kept/`.

**Sampling has two modes**, selected per row:
- Short videos: `sample_frames()` walks the whole duration at `--fps` (default 1 fps).
- Long videos: `sample_frames_windowed()` decodes 1-second windows at the pre-computed `sample_windows_s` offsets only. Windows are seeded from the file hash so the same file gets the same windows across runs.

**Resume is driven by `pipeline_status.csv`.** Each processed file appends one row (`file_path, status, keepers, elapsed_s, processed_at`). On startup the orchestrator skips any `file_path` whose latest row is `done` unless `--rescan` is set. The `--max-videos` / `--max-images` flags cap how many of each type are processed in a single invocation (smaller-size-first) but otherwise behave like a normal run — they write to `results.parquet` / `pipeline_status.csv` and run the viewers, so a smoke-test invocation composes naturally with later invocations via the resume mechanism (e.g. `--max-videos 10 --max-images 10`, then re-run with no caps to process the rest).

**Cross-file dedup + per-video cap run at the end** over the union of (this run's fresh keepers) ∪ (keepers loaded from any prior `results.parquet`). `_cross_file_dedup` greedily drops frame-dHash neighbors within `--frame-dedup-threshold` (higher composite wins); `_apply_video_cap` keeps the top `--max-per-video` composite per source `video_path`. The final survivors are what gets written to `results.parquet`.

**`results.parquet` is the single downstream artifact.** Notable columns: `video_path`, `video_stem`, `source_type` (`video`|`image`), `timestamp_s`, `refined_timestamp_s`, `frame_w`/`frame_h`, `face_x1/y1/x2/y2`, `face_w`, `face_det_score`, `kps` (JSON-encoded 5-point landmarks), `embedding` (JSON-encoded 512-d face embedding), `sharpness_center`, `refined_sharpness`, `aesthetics_norm`, `composite`, `p_none`/`p_bad`/`p_okay`/`p_good`, `pred_label`, `pred_confidence`, `uprighter_pred`, `uprighter_confidence`, `kept_path` (absolute path to the keeper JPEG under `kept/`).

**Card key is the join contract between Python and browser localStorage.** `constants.card_key(video_stem, kept_path)` returns `{video_stem}/{Path(kept_path).name}`. This key is used by `build_faces_review.py`'s `data-filename`, by `train_classifier.py`'s label join, and by `save_labeled_faces.py`'s lookup. Keep all four in sync — if you rename the key format, every reader needs the same change.

## Coding conventions in use

- Python 3.12+ syntax: `|` unions, built-in generics (`list[...]`, `dict[...]`), `pathlib.Path` everywhere.
- Logging via module-level `logger = logging.getLogger(__name__)`, configured once in `main()` with `force=True`. INFO-level per-file lines use ASCII separators (not em dashes) because logging writes to stderr, which isn't reconfigured to UTF-8.
- `argparse` directly; no click/typer.
- Flat module layout under `still_extractor/`; shared helpers live in their own small modules (e.g. `face_crop.py`, `constants.py`, `utils.py`) rather than a `utils` grab-bag.
- One CLI entry point per stage. Each module's `main()` is the entry; tests/scripts can import the helpers directly.

## Docs to consult

- `family_video_still_extractor_design.md` — full design rationale (why aesthetics-only fails, scoring model, identity-clustering plan).
- `docs/prompt*.md` — the historical prompt series that built each stage. New stage work tends to follow the same prompt-doc pattern.
