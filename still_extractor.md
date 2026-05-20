# still_extractor — Project Handoff Document

*Authoritative reference for new Claude sessions. Read this before anything else.*

---

## What this project is

A local CLI pipeline that extracts a curated set of high-quality still frames from a family video/photo corpus — suitable for photo albums. Target output: a few hundred well-curated JPEGs ranked by quality. Heavy **precision bias**: 200 great frames beats 2000 okay frames.

**Hardware**: Windows + NVIDIA GPU (RTX 2060 SUPER, CUDA 12.4, torch 2.6.0+cu124). Local processing only. Multi-hour/day runs are acceptable.

**Owner**: Matt Fisher (Principal Research Scientist, Adobe Research). Personal use only.

---

## Repo location and stack

- Repo: `C:\Code\video-to-photo\`
- Module: `still_extractor/` (flat layout)
- Python 3.12, `uv` for env management
- PyTorch via `[tool.uv.sources]` in `pyproject.toml` pinned to the CUDA 12.4 index URL — do **not** use `install_torch.bat` or suggest putting torch in a regular pip install; uv sync will undo it
- Style: `pathlib.Path`, `|` unions, built-in generics, `argparse`, `logging`
- Key deps: `pyav`, `insightface`, `onnxruntime-gpu`, `aesthetic-predictor-v2-5`, `open_clip_torch`, `opencv-python`, `imagehash`, `pandas`, `pyarrow`, `scikit-learn`, `hdbscan`, `Pillow`, `pillow-heif`, `pyyaml`, `tqdm`

---

## Corpus

- ~10,000 videos + images total at full scale; ~1,500 in the current mini test dataset
- Roughly 50/50 videos and images
- Video formats: `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`
- Image formats: `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, `.tiff`, `.tif`, `.bmp`
- HEIC is common (iPhone). `pillow-heif` registered at module load handles this transparently
- Long videos: up to ~5 minutes. Videos >60s get windowed sampling (see below)
- All files on the same Windows machine; some on external drives

---

## Pipeline overview

Six scripts, each independently callable. All read from a **run config YAML** (e.g. `configs/mini.yaml`).

### Run config format

```yaml
name: mini
dirs_file: configs/dirs_mini.txt
long_video_threshold_s: 60
long_video_windows: 20
long_video_min_spacing_s: 5
output_dir: data/mini
```

`dirs_file` is a text file of directory paths to crawl, one per line, `#` for comments.

---

### Script 1: `inventory.py`

**Purpose**: crawl directories, hash files, deduplicate, probe video durations, pre-compute sampling windows for long videos. Writes `manifest.csv`. Run once (or `--rescan` to pick up new files).

**CLI**: `python -m still_extractor.inventory --config configs/mini.yaml [--rescan]`

**Key behaviors**:
- Hashes first 64KB (MD5) for dedup. First-seen (lexicographic path order) is canonical; duplicates marked `is_duplicate=True` and skipped in all downstream passes.
- Long videos (>60s): pre-computes N non-overlapping 1-second sampling windows with minimum spacing, stored as JSON in `sample_windows_s` column. Seed derived from file hash for reproducibility.
- **Sorts manifest by `size_bytes` ascending** so Pass 1 processes small files first.
- Verbose console output with tqdm bars and emoji headers. Uses `print()` for milestones, `logging` for warnings.

**Manifest schema** (key columns): `file_path`, `file_type` (video/image), `extension`, `size_bytes`, `hash`, `is_duplicate`, `canonical_path`, `duration_s`, `is_long_video`, `sample_windows_s`, `run_name`, `scanned_at`

**Output**: `{output_dir}/manifest.csv`

---

### Script 2: `pass1_index.py`

**Purpose**: the slow indexing pass. Reads manifest, processes every non-duplicate file through sharpness filter + InsightFace face detection, writes surviving frames to `index.parquet`. Resumable via `pass1_status.csv`.

**CLI**: `python -m still_extractor.pass1_index --config configs/mini.yaml [--rescan] [--sharpness-threshold 75.0] [--min-face-px 80] [--fps 3.0] [--workers 1]`

**Key behaviors**:

*Videos*:
- Short videos (<60s): sample at `--fps` across full duration using PyAV
- Long videos: decode only the pre-computed windows from `sample_windows_s` (1-second windows at `--fps` each)
- Sharpness pre-filter: Laplacian variance on center 70%×70% crop. **Default threshold: 75.0** (lowered from 100.0 — important)
- Face detection: InsightFace `buffalo_l`, `det_size=(640,640)`, `ctx_id=0`. Drop frames with no face ≥ `--min-face-px` wide. Keep largest face per frame.

*Images*:
- Load via Pillow (HEIC transparent). Convert to BGR numpy array.
- Same sharpness filter + face detection gates.
- Single row per image: `frame_index=0`, `timestamp_s=0.0`
- Written as JPEG to `{output_dir}/frames/{stem}/00000_0.000.jpg`

*Lazy directory creation*: frames directory only created immediately before first frame write. No empty directories.

*Resume*: skip files already in `pass1_status.csv` with `status="done"` unless `--rescan`.

*Workers*: videos use `ProcessPoolExecutor` when `--workers > 1` (FaceAnalysis not picklable — uses module-level initializer). Images always single-threaded.

**Parquet schema** (key columns): `video_path`, `video_stem`, `frame_index`, `timestamp_s`, `frame_path`, `frame_w`, `frame_h`, `sharpness_center`, `face_x1/y1/x2/y2`, `face_w`, `face_det_score`, `kps` (JSON string, 5-point landmarks), `embedding` (JSON string, 512-d L2-normalized)

**Status CSV schema**: `file_path`, `file_hash`, `status`, `frames_sampled`, `frames_failed_sharpness`, `frames_failed_face_detect`, `frames_failed_face_size`, `frames_written`, `sharpness_mean/min/max`, `faces_detected`, `faces_failed_size`, `elapsed_s`, `processed_at`

**Output**: `{output_dir}/index.parquet`, `{output_dir}/pass1_status.csv`, `{output_dir}/frames/{stem}/*.jpg`

---

### Script 3: `pass2_score.py`

**Purpose**: score all candidates, deduplicate, emit top-K. Fast — iterate freely without re-running Pass 1.

**CLI**: `python -m still_extractor.pass2_score --config configs/mini.yaml [--top-k-per-file 5] [--temporal-window-s 2.0] [--face-dedup-threshold 8] [--frame-dedup-threshold 8]`

**Scoring sub-components**:
1. **Aesthetics**: Aesthetic Predictor V2.5 (SigLIP-based, fp16 on CUDA, batches of 32). Score 1–10, normalized to [0,1].
2. **Face sharpness**: Laplacian variance on face bbox crop (10px padding). Per-corpus min-max normalized.
3. **Eye-openness**: inter-eye distance / face height ratio (weak proxy with 5-point landmarks, down-weighted). Per-corpus normalized.

**Composite score**: weighted average of the three normalized sub-scores.

**Three-stage deduplication** (in order, each operates on survivors of previous):
1. **Temporal gate**: same video + within `--temporal-window-s` → keep higher composite. Pure DataFrame operation, no I/O.
2. **Face-crop dHash**: imagehash dHash on face crop. Hamming distance threshold.
3. **Full-frame dHash**: imagehash dHash on full frame. Hamming distance threshold.

Status columns in scores.csv: `kept_after_temporal`, `kept_after_face_dedup`, `kept_after_frame_dedup`, `dedup_kept` (derived AND of all three).

**Top-K selection**: **per-file**, not global. Default 5 per file. Images naturally contribute at most 1 (only one frame per image in the Parquet). This ensures corpus coverage — global top-K would be dominated by a handful of well-lit clips.

**Output**: `{output_dir}/top_frames/*.jpg` (named `{composite:.4f}_{stem}_{ts:.2f}.jpg`), `{output_dir}/scores.csv`

---

### Script 4: `pass3_refine.py`

**Purpose**: micro-window refinement. For each top-K candidate, decode ±`--window-s` seconds at native frame rate and pick the sharpest face frame. Fixes "3fps sampling missed the actual best moment."

**CLI**: `python -m still_extractor.pass3_refine --config configs/mini.yaml [--window-s 0.5] [--top-k 200]`

**Output**: `{output_dir}/refined/*.jpg`, `{output_dir}/refined_scores.csv` (adds `refined_frame_path`, `refined_timestamp_s`, `refined_sharpness`, `original_sharpness`, `sharpness_delta`)

**Diagnostic**: logs mean and max `sharpness_delta` — tells you whether Pass 3 is buying anything on your corpus. In testing: mean +15, max +22, consistently positive.

---

### Script 5: `build_index_html.py`

**Purpose**: self-contained HTML review + labeling page. Shows face crops in a grid, enables 3-category labeling, exports `labels.json`.

**CLI**: `python -m still_extractor.build_index_html --scores-csv data/mini/refined_scores.csv --output-html data/mini/index.html`

**Key behaviors**:
- Shows **face crops** (not full frames), base64-embedded, generated by Pillow at build time
- Each card links to full-resolution JPEG
- Sorted by composite score descending

**3-category labeling UI**:
- **Bad** (`1` or `B`): red border, dimmed
- **Okay** (`2` or `O`): amber border
- **Good** (`3` or `G`): green border
- `X`: clear label
- Arrow keys: navigate between cards
- `1/2/3` auto-advances focus to next card for fast keyboard labeling
- State persists in `localStorage` keyed by filename
- Filter bar: All / Good / Okay / Bad / Unreviewed
- **Export Labels** button: downloads `labels.json` as `{"filename": "good"|"okay"|"bad"}`

**Shared helper**: `face_crop.py` — `extract_face_crop(image_path, x1, y1, x2, y2, padding) -> Image`

---

### Script 6: `cluster_faces.py`

**Status**: stub only. Deferred to v2.

**Planned purpose**: cluster face embeddings from `index.parquet` using HDBSCAN or agglomerative clustering. Emit cluster grids for manual labeling. Not needed for v1.

---

## Key design decisions and their rationale

**Why not global top-K**: global top-K would be dominated by a handful of well-lit clips. Per-file top-5 ensures coverage across the whole corpus. Some files contribute 0 (no good frames), some contribute up to 5.

**Why Parquet not SQLite**: embeddings are float arrays — natural in Parquet, awkward in SQLite (blob serialization). Pass 2 is a pandas/numpy operation on a dataframe. Cross-video queries (identity clustering) are just "read the whole file."

**Why three-stage dedup**: each stage catches different failure modes. Temporal gate catches adjacent 3fps samples from the same shot (free, no I/O). Face-crop dHash catches near-identical faces with different backgrounds. Full-frame dHash catches identical compositions. Running all three in order is cheap.

**Why lower sharpness threshold to 75**: at 100, ~102 of 158 zero-row files were killed entirely at the sharpness gate. Median sharpness_max for zero-row files was 76 — just below threshold. 75 unlocks ~80 more files while face detection remains the next gate, preventing blurry frames from reaching Pass 2.

**Why per-file top-5 not per-file top-1 for images**: images are already a single frame, so per-file cap of 5 naturally resolves to 1. No special casing needed.

**Why windowed sampling for long videos**: 5-minute videos at 3fps = 900 frames, dominating runtime. 20 random non-overlapping 1-second windows (min 5s spacing) gives ~60 frames — good coverage without bias toward any segment. Seed from file hash ensures reproducibility.

**InsightFace license**: `buffalo_l` weights are non-commercial research only. This tool is personal use — acceptable. Do not redistribute outputs commercially.

---

## Current status (as of this document)

**Completed and working**:
- `inventory.py` — fully implemented, tested on ~1500 file mini corpus (980 images, 503 videos, 43 long videos with windowed sampling)
- `pass1_index.py` — fully implemented including manifest-driven routing, image support, HEIC, windowed sampling, lazy dirs, per-gate status logging
- `pass2_score.py` — fully implemented including three-stage dedup
- `pass3_refine.py` — fully implemented
- `build_index_html.py` — implemented, pending Prompt 10 (3-category labeling upgrade)
- `face_crop.py` — shared helper, implemented

**In progress**:
- Pass 1 full mini-corpus run in progress with new sharpness threshold of 75.0
- Prompt 10 (3-category HTML labeling UI) written but not yet executed

**Pending / v2**:
- Per-file top-K enforcement in Pass 2 (currently global top-K — needs patch prompt; agreed on top-5 per file)
- `cluster_faces.py` — deferred
- Classifier training script — v2. Plan: label face crops Bad/Okay/Good (3 categories), train ordinal classifier on frozen InsightFace embeddings (already in Parquet), replace composite score in Pass 2
- Source-original copy for top-K images (Pass 2 currently copies the JPEG-converted frame; for images the original file should be used)

---

## Active threads / open questions

1. **Pass 2 per-file top-K patch**: needs a new prompt. Top-5 per file agreed. Global dedup pass after per-file selection catches cross-file near-duplicates. Write this prompt after full mini-corpus Pass 1 completes.

2. **Gate analysis on full corpus**: once Pass 1 finishes, aggregate `pass1_status.csv` to see gate breakdown across all 1500 files including images. Key question: are images mostly passing face detection (expected for family photos), or is something wrong with the image routing?

3. **Disk management**: intermediate `frames/` directories can be large. Plan discussed: delete `frames/` after Pass 2 completes, keeping only `index.parquet` and top-K refined JPEGs. Not yet implemented.

4. **Labeling plan**: label face crops (not full frames) on Bad/Okay/Good 3-category scale. HTML UI handles this after Prompt 10. Target: 1000+ labeled examples. Train ordinal classifier on embeddings already in Parquet.

5. **Classifier architecture**: small head on frozen InsightFace `normed_embedding` (512-d, already in Parquet — no re-encoding needed). Ordinal loss: two binary classifiers "≥ Okay" and "≥ Good", combined. Slots into Pass 2 as replacement or additional factor in composite score.

---

## Prompts written (in order)

| Prompt | File | Status |
|---|---|---|
| 1 — Repo scaffold | `prompt1_scaffold.md` | Done (with uv sources deviation noted) |
| 2 — Pass 1 sampling + sharpness | `prompt2_pass1_sampling.md` | Done |
| 3 — Pass 1 face detection + Parquet | `prompt3_pass1_faces.md` | Done |
| 4 — Pass 2 scoring + dedup + top-K | `prompt4_pass2_scoring.md` | Done |
| 4b — Dedup patch (3-stage) | `prompt4b_dedup_patch.md` | Done |
| 5 — Pass 3 + HTML labeling UI | `prompt5_pass3_and_html.md` | Done |
| 6 — Inventory script | `prompt6_inventory.md` | Done |
| 7 — Pass 1 manifest-driven rewrite + images | `prompt7_pass1_manifest.md` | Done |
| 8 — Richer status logging + inventory sort | `prompt8_status_logging.md` | Done |
| 9 — Sharpness threshold 75 + lazy dirs | `prompt9_threshold_and_lazy_dirs.md` | Done |
| 10 — HTML 3-category labeling UI | `prompt10_html_3category.md` | Written, not yet executed |

---

## Notes for future Claude sessions

- Always write Claude Code prompts as `.md` files to `/mnt/user-data/outputs/` and present them — never print inline
- Do not suggest re-running inventory unless new files have been added or `--rescan` is explicitly needed
- The uv/torch CUDA setup is working — do not suggest changing it
- InsightFace `buffalo_l` weights download to `%USERPROFILE%\.insightface\models\` on first run
- `onnxruntime` and `onnxruntime-gpu` conflict — only `onnxruntime-gpu` should be installed
- ffmpeg path: `--ffmpeg-path` argument exists for CLI stability but PyAV uses bundled libs; ffmpeg binary not actually needed
- When Matt asks for status analysis, read `pass1_status.csv` via bash/pandas — do not ask him to paste contents
