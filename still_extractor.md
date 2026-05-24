# still_extractor — Project Handoff Document

*Authoritative reference for new Claude sessions. Read this before anything else.*

---

## Project overview

`still_extractor` is a Python pipeline that extracts high-quality portrait frames from a corpus of family videos and photos. The output is used to build a face quality classifier (MobileNetV3), a labeling UI for generating training data, and a photo viewer for selecting photos for print/photo book use. Located at `C:\Code\video-to-photo\`.

Downstream goal: select a top ~200 frames from a corpus, caption them with a local VLM, cluster identities via ArcFace embeddings, and feed structured annotations to Claude to assemble a storybook. See **Downstream: Storybook Pipeline** section.

---

## Architecture (v2)

A single per-file pipeline replaces the old pass1/pass2/pass3 architecture. Each source file is processed end-to-end in memory; only final keeper JPEGs are written to disk.

**Per-file flow (videos):**
sample frames → temporal dedup (early, before models) → uprighter → sharpness gate → face detect → **rejection heuristics** → batch score (aesthetic + classifier) → face/frame dHash dedup → top-K selection → micro-window refinement → write keeper JPEG

**Per-file flow (images):**
EXIF orient → uprighter → sharpness gate → face detect → **rejection heuristics** → score → write keeper JPEG

**Orchestrator** handles: inventory → per-file workers → cross-file dedup → per-video cap → `results.parquet`

`process_file()` returns a `FileResult` dataclass with `keepers: list[dict]` and `stage_times_s: dict[str, float]`.

---

## Module inventory (`still_extractor/`)

| Module | Role |
|---|---|
| `pipeline.py` | Main orchestrator — runs full pipeline from config |
| `worker.py` | Per-file processor: `process_file(row, models, cfg) -> FileResult` |
| `models.py` | Model loading: `load_models() -> Models` dataclass |
| `sampling.py` | Frame sampling, rotation detection (tkhd parser), sharpness |
| `inventory.py` | File crawl, dedup, manifest, `RunConfig.from_yaml()` |
| `face_crop.py` | `extract_face_crop_from_image()` (in-memory), `extract_face_crop()` (path-based wrapper) |
| `constants.py` | All shared constants + `card_key(video_stem, kept_path)` |
| `utils.py` | `safe_float`, `to_fwd_slash`, `parse_kps` |
| `build_photo_viewer.py` | Full-frame justified-grid viewer with selection/export workflow |
| `build_faces_review.py` | Face-crop labeling UI — filters out null-face rows automatically |
| `train_classifier.py` | Face quality classifier training (MobileNetV3) |
| `train_uprighter.py` | Uprighter training (MobileNetV3, 4-class rotation) |
| `build_uprighter_review.py` | Uprighter training data review HTML |
| `save_labeled_faces.py` | Exports labeled face crops to global store `data/face_labels/` |
| `launch_faces_export_server.py` | Local HTTP server (port 7432) — Export Labels button in faces_review.html POSTs here |
| `compare_classifiers.py` | Compares all `best_model_v*.pt` checkpoints; auto-globs from `models/face_quality/` |
| `export_flagged.py` | Copies flagged frames from photo viewer to output dir |
| `build_clusters.py` | DBSCAN identity clustering on ArcFace embeddings; updates `data/identities/` |
| `diagnose_keypoints.py` | Flags anomalous face keypoint geometry; emits `keypoint_diagnostics.parquet` |
| `build_keypoint_debug.py` | HTML viewer for anomalous keypoint frames |
| `diagnose_dates.py` | Diagnostic for frames with unknown source month |

Note: `build_debug_viewer.py` has been deleted — debug functionality is now integrated into `index_photos.html` via debug flags.

---

## Key constants (`constants.py`)

- `FACE_QUALITY_LABELS = ["none", "bad", "okay", "good"]`
- `FACE_CROP_PADDING = 20`
- `FACE_QUALITY_INPUT_SIZE = 128`
- `UPRIGHTER_INPUT_SIZE = 224`
- `UPRIGHTER_CONFIDENCE_THRESHOLD = 0.95`
- `CLASSIFIER_BLEND_WEIGHT = 0.8`
- `FACE_MIN_AREA_FRAC = 0.004` — minimum face area / frame area to pass rejection
- `FACE_EDGE_IMMUNE_AREA_FRAC = 0.025` — faces above this size are immune from edge rejection
- `FACE_EDGE_ZONE_FRAC = 0.10` — fraction of frame width/height defining the edge zone
- `card_key(video_stem, kept_path)` → `"{video_stem}/{Path(kept_path).name}"` — must stay in sync across `build_faces_review.py`, `train_classifier.py`, `save_labeled_faces.py`, `build_clusters.py`, and browser localStorage

---

## Config files

Each corpus run has a YAML config:
```yaml
name: june27
dirs_file: configs/dirs_june27.txt
long_video_threshold_s: 60
long_video_windows: 20
long_video_min_spacing_s: 5
output_dir: data/june27
```

`--config` is supported by all pipeline stages.

---

## CLI reference

```bash
# Full pipeline
uv run python -m still_extractor.pipeline --config configs/june27.yaml

# Test run (safe — writes results_test.parquet and pipeline_summary_test.json)
uv run python -m still_extractor.pipeline --config configs/june27.yaml --max-videos 10 --max-images 0

# Build photo viewer (auto-invokes build_clusters if results.parquet has embedding column)
uv run python -m still_extractor.build_photo_viewer --config configs/june27.yaml

# Build faces review UI
uv run python -m still_extractor.build_faces_review --config configs/june27.yaml

# Start face export server (keep running while labeling)
uv run python -m still_extractor.launch_faces_export_server --config configs/june27.yaml

# Build identity clusters (also auto-run by build_photo_viewer)
uv run python -m still_extractor.build_clusters --config configs/june27.yaml

# Keypoint diagnostics
uv run python -m still_extractor.diagnose_keypoints --config configs/june27.yaml
uv run python -m still_extractor.build_keypoint_debug --config configs/june27.yaml

# Date extraction diagnostics
uv run python -m still_extractor.diagnose_dates --config configs/june27.yaml

# Train face quality classifier
uv run python -m still_extractor.train_classifier \
  --labels-store data/face_labels/labels.json \
  --results data/june27/results.parquet \
  --synthetic-none-ratio 0.25 \
  --sampler-boost-good 1.25 \
  --output models/face_quality/best_model_vN.pt

# Compare all classifier checkpoints
uv run python -m still_extractor.compare_classifiers \
  --labels-store data/face_labels/labels.json

# Export labeled face crops to global store (CLI path — prefer the Export Labels button)
uv run python -m still_extractor.save_labeled_faces \
  --config configs/june27.yaml \
  --output-dir data/face_labels

# Train uprighter
uv run python -m still_extractor.train_uprighter \
  --frames-json data/june27/uprighter_frames.json \
  --rejected-json labels/rejected.json \
  --output-dir models/uprighter
```

---

## Output artifacts

### Per-run (under `data/{run_name}/`)

| Artifact | Description |
|---|---|
| `manifest.csv` | File inventory with hashes, duplication info, sample windows |
| `pipeline_status.csv` | Per-file done-status for resumability |
| `pipeline_summary.json` | Run summary: keeper counts, label distribution, uprighter corrections, stage timing, rejection stats, date source breakdown |
| `results.parquet` | One row per keeper frame (see schema below) |
| `kept/` | Final keeper JPEGs: `{composite:.4f}_{stem}_{timestamp_s:.3f}.jpg` |
| `rejected/` | Face crop JPEGs rejected by heuristics: `{reason}_{stem}_{timestamp_s:.3f}_{face_idx}.jpg` |
| `faces_review.html` | Face-crop labeling UI (null-face rows filtered out) |
| `face_labels.json` | Labels exported from faces_review.html (input to export server) |
| `index_photos.html` | Full-frame photo viewer with selection/export workflow and optional debug overlays |
| `clusters.json` | Per-run identity cluster assignments (written by `build_clusters.py`) |
| `frame_dimensions.json` | Sidecar cache of (w, h) per card_key — avoids re-reading images on viewer rebuild |
| `keypoint_diagnostics.parquet` | Per-face keypoint anomaly flags and centroid distances (written by `diagnose_keypoints.py`) |

### Global (persistent across runs)

| Artifact | Description |
|---|---|
| `data/face_labels/labels.json` | List of exported labeled face crops with sha256, corpus, label |
| `data/face_labels/seen_hashes.json` | Dedup set — prevents re-exporting the same crop |
| `data/face_labels/faces/` | Exported face crop JPEGs |
| `data/identities/index.json` | Global identity store: stable name, display_name, centroid, member_count, portrait_path |
| `data/identities/{name}.png` | Representative 256×256 portrait per identity |
| `models/face_quality/best_model.pt` | Current best classifier (v5) |
| `models/face_quality/best_model_v3.pt … best_model_v7.pt` | Versioned checkpoints — never overwrite |
| `models/uprighter/best_model.pt` | Current uprighter checkpoint |

---

## `results.parquet` schema

```
card_key, source_type, source_path, video_stem,
timestamp_s, source_year, source_month,

# Top-3 faces (ranked by p_good descending; null-filled if fewer detected)
face_1_x1, face_1_y1, face_1_x2, face_1_y2,
face_1_kps,                    # 5×2 keypoints JSON
face_1_embedding,              # ArcFace embedding JSON (normed float32 array)
face_1_kps_anomalous,          # bool — True if keypoint geometry is anomalous
face_1_pred_label, face_1_pred_confidence,
face_2_* … face_3_*,           # same structure, null if not present

face_count,                    # total faces detected (before top-3 truncation)
best_pair_score,               # (face_1_p_good + face_2_p_good) / 2; null if <2 faces
rejected_face_count,
rejected_faces_json,           # JSON array of {x1,y1,x2,y2,reason}

sharpness_center, refined_sharpness, sharpness_delta,
aesthetics_norm, composite,
uprighter_pred, uprighter_confidence,
w, h,                          # natural pixel dimensions of keeper JPEG
kept_path
```

`kept_path` is absolute. `source_type` is `"video"` or `"image"`. Rows with `face_count=0` (all faces rejected) have null face columns — they are valid keeper frames but have no labelable face crop.

`source_year` and `source_month` are int (month=0 = unknown). Populated by `extract_source_date()` in `worker.py` using: EXIF DateTimeOriginal → EXIF DateTime → path regex → PyAV video metadata → mtime year only. **mtime is never used as a month source** (copy-from-camera resets mtime). The pipeline also emits `date_source_counts` in `pipeline_summary.json`.

---

## Face rejection heuristics

Applied before classifier inference. Controlled by constants in `constants.py`.

- **too_small**: face area / frame area < `FACE_MIN_AREA_FRAC` (0.004)
- **small_and_edge**: face area / frame area < `FACE_EDGE_IMMUNE_AREA_FRAC` (0.025) AND face center within `FACE_EDGE_ZONE_FRAC` (10%) of any frame edge

Rejected face crops are written to `data/{run_name}/rejected/` for audit.

---

## Roll correction and keypoint anomaly detection

`face_crop.py` applies roll correction (using keypoint geometry) before ArcFace embedding. When keypoints are badly localized, this correction degrades the crop. `is_keypoint_anomalous(kps, bbox)` detects bad geometry using three flags:

- **vertical_order**: nose tip not between eyes and mouth vertically
- **ratio**: `eye_to_nose_dist / eye_mouth_dist` outside `[0.25, 0.75]`
- **span**: keypoints clustered within top 25% of bbox height

When anomalous, roll correction is **skipped** (unrotated crop used instead). The anomaly flag is stored as `face_N_kps_anomalous` in `results.parquet`.

**Impact on june27**: 8.5% of faces are keypoint-anomalous. Before the fix, anomalous faces had mean centroid distance 0.476 with 41.5% cluster assignment rate vs 0.376 / 71.5% for normal faces — strong evidence the fix matters for clustering quality.

Constants in `face_crop.py`: `KPS_RATIO_MIN=0.25`, `KPS_RATIO_MAX=0.75`, `KPS_SPAN_MIN_FRAC=0.25`.

---

## Labeling workflow

1. Start the export server: `uv run python -m still_extractor.launch_faces_export_server --config configs/june27.yaml`
2. Run `build_faces_review.py` → open `faces_review.html` in browser
3. Label cards with keyboard: `1/N`=none, `2/B`=bad, `3/O`=okay, `4/G`=good, `X`=clear
4. Click **Export Labels** — POSTs to the export server, which runs `save_labeled_faces.run_export()` and shows an alert with counts: new faces added, already in store, skipped
5. The global store at `data/face_labels/` is updated atomically (dedup via `seen_hashes.json`)

**Important**: always use the Export Labels button (or run `save_labeled_faces.py`) after any significant labeling session before re-running the pipeline.

**Current label store** (as of this session): 2328 total — good: 491, okay: 655, bad: 600, none: 582. Multi-corpus: june27 (1427), Julia2Uncompressed (316), JuliaEllieMay2026 (585).

---

## Face quality classifier

### Architecture
MobileNetV3-Small, 4-class (none/bad/okay/good), 128×128 input, kps-based roll correction, ~1.5M params.

### Training (`train_classifier.py`)
- Data source: `data/face_labels/labels.json` (global store, direct from `face_crop_path` — no parquet join)
- Synthetic none augmentation: random background crops from keeper JPEGs, non-overlapping with face bboxes, per-epoch resampling, ratio controlled by `--synthetic-none-ratio` (default 0.25)
- Class balancing: `WeightedRandomSampler` with boost controlled by `--sampler-boost-good` (default 1.25)
- MixUp α=0.3; Good-Good pairs use Beta(0.5, 0.5)
- AdamW lr=1e-3, CosineAnnealingLR, batch=32, epochs=100, early stop patience=12
- Two-phase fine-tune: phase 1 head only, phase 2 unfreeze features.12/13+classifier
- Always start from ImageNet pretrained weights — never warm-start from a prior checkpoint
- Per-epoch logging: val_loss, val_acc, Good F1, Good precision, Good recall
- **Always save to a new versioned file** (`best_model_vN.pt`). Update `best_model.pt` only after explicit comparison.

### Checkpoint history

All numbers from same val split (seed=42, test_size=0.1, n=233).

| Version | Val Loss | Good F1 | Good Prec | Good Rec | Macro F1 | Status |
|---|---|---|---|---|---|---|
| v3 | 0.8146 | 0.699 | 0.581 | **0.878** | 0.738 | KEEP — highest Good recall |
| v4 | **0.7925** | 0.701 | 0.603 | 0.837 | **0.764** | KEEP — best val_loss + macro F1 |
| v5 | 0.8101 | 0.701 | **0.708** | 0.694 | 0.724 | KEEP — **current best_model.pt**, highest Good precision |
| v7 | 0.8513 | 0.686 | 0.643 | 0.735 | 0.714 | KEEP — Pareto-optimal on bad F1 + good recall vs v5 |

v1, v2, v6 deleted (strictly dominated).

**v5 is current `best_model.pt`** — highest Good precision (0.708). Good precision is the primary metric for `best_model.pt` since it minimizes false positives polluting identity clusters.

### Key finding for v8

v7's epoch 74 had p_good=0.667 / f1_good=0.717 — better than the saved epoch 86 (0.643 / 0.686) on all Good metrics. The checkpoint is saved on val_loss-min policy, which is wrong when Good precision is the goal. **For v8: change checkpoint save policy to save on `p_good` instead of `val_loss`.** This alone may close the gap with v5 without any other changes.

### Comparison tool
```bash
uv run python -m still_extractor.compare_classifiers --labels-store data/face_labels/labels.json
```
Auto-globs all `best_model_v*.pt` in `models/face_quality/`. Run after each new version.

---

## Identity clustering

### Overview

`build_clusters.py` clusters ArcFace embeddings from `results.parquet` using DBSCAN, matches clusters to the global identity store via Hungarian algorithm, and writes per-run and global outputs. Auto-invoked by `build_photo_viewer.py` if `results.parquet` has an `embedding` column.

### Constants (in `build_clusters.py`)
- `DBSCAN_EPS = 0.4`
- `DBSCAN_MIN_SAMPLES = 5`
- `IDENTITY_MATCH_THRESHOLD = 0.5`

### Global identity store (`data/identities/`)

- `index.json`: list of known identities with fields: `name` (stable ID, e.g. `"personA"`), `display_name` (human-readable, e.g. `"Julia"`), `centroid` (float32 array), `member_count`, `portrait_path`
- `{name}.png`: 256×256 representative portrait per identity

**Rename workflow**: edit `display_name` in `index.json` (leave `name` alone — it is the stable ID tied to the PNG filename). Then rebuild viewer: `uv run python -m still_extractor.build_photo_viewer --config configs/<run>.yaml`. No re-clustering needed.

### Per-run outputs
- `data/{run_name}/clusters.json`: cluster assignments with `identity`, `member_count`, `representative_kept_path`, `frame_ids` (list of card_keys)

### Known issue: identity oversplit

Current `data/identities/` has ~25 entries for ~8 actual people (personA–personP plus named entries). Root cause: `DBSCAN_EPS=0.4` is too tight — same person in different lighting/pose/age falls outside each other's neighborhood. Also: pipeline re-run with roll correction fix shifted embedding space, causing Hungarian matching to miss existing identities and create new placeholders.

**To fix**: consider increasing `DBSCAN_EPS` to 0.5–0.6 and re-running `build_clusters.py`, then manually merging/renaming via `display_name`. The named entries (Julia, Jason, Matt, etc.) and unnamed placeholders (personA–personP) need to be reconciled.

### Unknown faces

DBSCAN noise points (label=-1) are "unknown" — faces that don't cluster into any identity. Shown as a special "Unknown" chip in `index_photos.html`. Per-frame `has_unknown` boolean embedded in viewer JS data.

---

## Photo viewer (`index_photos.html`)

Single output replacing the old `index_photos.html` + `index_photos_debug.html`. Built by `build_photo_viewer.py`.

### Features
- **Layout**: justified (Google Photos-style, full aspect ratio rows, 200px target row height). Crop mode (square grid) exists in CSS but is not exposed in the UI.
- **Year/month sections**: reverse chronological, collapsible. Two-level header pills: year pills always visible, click year label to expand month pills. State persists to localStorage.
- **Filters**: Source (All/Images/Videos), Quality (Good/Okay/Bad/None), People (identity chips, AND/OR toggle). All compose with AND between filter types.
- **Selection/export**: Google Photos-style circular checkboxes, shift+click range select within section, blue selection ring. Export overlay shows selected photos; "Export ZIP" is stubbed (not yet implemented).
- **Debug flags** (⚙ Settings, off by default): `show_debug` (score panel + face identity table), `show_faces` (bbox/keypoint canvas overlay + hover tooltip). All debug data embedded in HTML regardless of flag state.
- **Card sort**: within each month section, cards sorted by `(group_first_timestamp, video_stem, timestamp_s)` — frames from the same source file appear together in temporal order.
- **Video badge**: always visible (small play icon, top-right of card).

### localStorage keys
| Key | Value |
|---|---|
| `se_selected` | JSON array of card_key strings |
| `se_sections_collapsed` | JSON array of section key strings ("2025-05") |
| `se_years_expanded` | JSON array of expanded year ints |
| `se_quality_filter` | `{good, okay, bad, none}` booleans |
| `se_people_filter` | `{mode: "AND"\|"OR", identities: [...]}` |
| `se_source_filter` | `"all"` \| `"images"` \| `"videos"` |
| `se_layout` | `"justified"` (default) |
| `se_debug_show_debug` | boolean string |
| `se_debug_show_faces` | boolean string |

### Per-frame JS data (key fields)
- `card_key`, `source_type`, `video_stem`, `timestamp_s`
- `w`, `h` (natural dimensions for justified layout — cached in `frame_dimensions.json`)
- `source_year`, `source_month`
- `composite`, `sharpness_center`, `refined_sharpness`, `aesthetics_norm`
- `face_count`, `face_identities` (per-face: `{identity, confidence, assigned}`)
- `has_unknown` (bool — frame has DBSCAN noise-point faces)
- `identities` (array of hard-assigned identity names present in frame)
- `face_N_kps` (keypoints for bbox/keypoint overlay)

---

## Date extraction (`extract_source_date` in `worker.py`)

Fallback chain per source file:
1. EXIF `DateTimeOriginal` (tag 36867) — images + HEIC/HEIF (piexif handles HEIC fine; only video extensions are skipped)
2. EXIF `DateTime` (tag 306) — fallback EXIF tag
3. PyAV video metadata — `com.apple.quicktime.creationdate` or `creation_time` from container metadata (for .mp4/.mov/.m4v)
4. Path regex — `YYYY-MM`, `YYYY_MM`, `YYYYMMDD`, month-name + optional day + year (e.g. "May 24 2026"), standalone `20XX`
5. mtime — **year only**; mtime month is never used (copy-from-camera resets mtime)
6. `(0, 0)` — unknown

`date_source_counts` reported in `pipeline_summary.json` and console. Video extensions skipped for EXIF: `.mp4 .mov .avi .mkv .m4v` (not `.heic/.heif` — those have valid EXIF).

---

## Pipeline timing

Stage timings aggregated into `pipeline_summary.json`. Approximate per-file times (june27 corpus):

| Stage | Mean/file | % of total |
|---|---|---|
| refinement | 0.110s | ~35% |
| frame_sampling | 0.053s | ~17% |
| face_detect | 0.042s | ~13% |
| aesthetics | 0.034s | ~11% |
| uprighter | 0.032s | ~10% |
| classifier | 0.018s | ~6% |
| others | — | ~8% |

**Non-trivial optimization opportunities (not yet implemented):**
- Refinement container reuse: opening av container once per keeper → ~0.8s/file saving
- TTA-uprighter re-run on keepers is redundant (rotation already applied in candidate phase)
- Aesthetic preprocessor: pre-downscale to ~512px before model input → ~150ms/file

---

## Models

### Face quality classifier (`models/face_quality/best_model.pt`)
- Current: v5 (val_loss 0.8101, Good precision 0.708)
- See checkpoint history table above

### Uprighter (`models/uprighter/best_model.pt`)
- MobileNetV3-Small, 4-class: 0°/90°/180°/270° CW
- Input: 224×224, mixed resize strategy, 3-strategy TTA at inference
- Val accuracy: 82.8% single-pass, 88.6% TTA
- **Known issue**: 90°↔270° confusion (~23% error rate) due to `RandomHorizontalFlip` in training augmentation. Fix: retrain without `RandomHorizontalFlip`. Deferred.
- Confidence threshold: 0.95

---

## Downstream: Storybook Pipeline (planned, not yet implemented)

Goal: select top ~200 frames from a corpus, annotate with captions and person identities, feed to Claude to organize into a storybook.

### Captioning
- Model: **SmolVLM2 (2B)** — fits in 8GB VRAM, ~2s/image, good structured output
- Prompt for structured short-form output: `"Describe this photo using only these fields: setting (one phrase), activity (one phrase), people (count and relationship), mood (one word), framing (close portrait / medium / wide action). Be brief."`

### Identity clustering
- ArcFace embeddings stored per-face in `results.parquet` (`face_N_embedding` columns)
- DBSCAN clustering + manual label review → identity manifest per image
- Use case: "make a storybook focused on Julia with a distribution of other people"

### Assembly
- Caption metadata + identity annotations fed to Claude as structured text
- Claude organizes into chapters/sections; layout step TBD

---

## Pending work

- **Retrain uprighter**: remove `RandomHorizontalFlip` from training augmentation to fix 90°↔270° confusion. Use existing training data.
- **Identity oversplit**: increase `DBSCAN_EPS` (try 0.5–0.6), re-run `build_clusters.py`, manually reconcile named + placeholder identities in `data/identities/index.json`.
- **v8 classifier**: change checkpoint save policy from val_loss-min to p_good-max. This is the highest-leverage change for the next training run. Consider also: further label collection for Good class (491 samples, still the smallest class).
- **ZIP export**: implement the Export ZIP button in `index_photos.html`. Currently stubs with `alert("not yet implemented")`. Will require a local server endpoint (similar to `launch_faces_export_server.py` pattern).
- **Storybook pipeline**: implement captioning script (SmolVLM2) + identity clustering → assembly.
- **Dedup keeper JPEGs**: some dedup-loser JPEGs may remain in `kept/` — pipeline doesn't delete them. Low priority.

---

## Session rules

- Always write Claude Code prompts as `.md` files to `/mnt/user-data/outputs/` and present them.
- Do not write ahead multiple prompts unless explicitly discussed. Write one, wait for results, then write the next.
- For debugging persistent bugs where the cause is ambiguous, write a diagnosis-first prompt. Otherwise give Claude Code high-level intent — it handles details correctly.
- When writing prompts for steps that could take >~30s, include instructions to estimate runtime and background the process.
- Always use the Export Labels button (or `save_labeled_faces.py`) after any significant labeling session before re-running the pipeline.
- **Never overwrite a versioned checkpoint** (`best_model_vN.pt`). Always save new training runs to the next version number. Update `best_model.pt` only after explicit comparison and decision.
- Do not prompt about updating the handoff document at end of each topic — Matt initiates updates.
- Test runs use `--max-videos N --max-images M` and write to `results_test.parquet` / `pipeline_summary_test.json` (safe, doesn't pollute production).
