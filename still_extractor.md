# still_extractor — Project Handoff Document

*Authoritative reference for new Claude sessions. Read this before anything else.*

---

## Project overview

`still_extractor` is a Python pipeline that extracts high-quality portrait frames from a corpus of family videos and photos. The output is used to build a face quality classifier (MobileNetV3), a labeling UI for generating training data, and a photo viewer for selecting photos for print/photo book use. Located at `C:\Code\video-to-photo\`.

Downstream goal: select a top ~200 frames from a corpus, caption them with a local VLM, cluster identities via ArcFace embeddings, and feed structured annotations to Claude to assemble a storybook. See **Downstream: Storybook Pipeline** section.

---

## Architecture (v2)

A single per-file pipeline replaces the old pass1/pass2/pass3 architecture. Each source file is processed end-to-end in memory; only final keeper JPEGs are written to disk. **v2 is complete and running.**

**Per-file flow (videos):**
sample frames → temporal dedup (early, before models) → uprighter → sharpness gate → face detect → **rejection heuristics** → batch score (aesthetic + classifier) → face/frame dHash dedup → top-K selection → micro-window refinement → write keeper JPEG

**Per-file flow (images):**
EXIF orient → uprighter → sharpness gate → face detect → **rejection heuristics** → score → write keeper JPEG

**Orchestrator** handles: inventory → per-file workers → cross-file dedup → per-video cap → `results.parquet`

`process_file()` returns a `FileResult` dataclass with `keepers: list[dict]` and `stage_times_s: dict[str, float]`.

---

## Directory structure

```
data/
  runs/                        # per-corpus pipeline outputs (reproducible)
    june27/
    JuliaEllieMay2026/
    julia2_uncompressed/
  ground_truth/                # accumulated hand-labeled data (irreplaceable)
    face_labels/               # labels.json lives here
    identities/                # identity clusters, named persons
models/
  face_quality/                # classifier checkpoints
  uprighter/                   # uprighter checkpoints
configs/                       # per-corpus YAML configs
still_extractor/               # source modules
```

**Important**: `data/runs/` is reproducible (delete and re-run pipeline). `data/ground_truth/` is not — back it up separately. Do not conflate them.

**Known debt**: `data/ground_truth/face_labels/labels.json` contains 2,084 entries with absolute paths (`C:/Code/video-to-photo/data/...`). These work on the current machine but will break if the project moves. A future `make_paths_relative` pass would fix this.

**Note**: `june27/` run data lives on a separate machine/drive and is not present in all checkouts. `JuliaEllieMay2026/` is the currently accessible run.

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
| `save_labeled_faces.py` | Exports labeled face crops to global store `data/ground_truth/face_labels/` |
| `launch_faces_export_server.py` | Local HTTP server (port 7432) — Export Labels button in faces_review.html POSTs here |
| `compare_classifiers.py` | Compares all `best_model_v*.pt` checkpoints; auto-globs from `models/face_quality/` |
| `export_flagged.py` | Copies flagged frames from photo viewer to output dir |
| `build_clusters.py` | DBSCAN identity clustering on ArcFace embeddings; updates `data/ground_truth/identities/` |
| `diagnose_keypoints.py` | Flags anomalous face keypoint geometry; emits `keypoint_diagnostics.parquet` |
| `build_keypoint_debug.py` | HTML viewer for anomalous keypoint frames |
| `diagnose_dates.py` | Diagnostic for frames with unknown source month |
| `caption_photos.py` | Captions good-quality keepers with SmolVLM2; writes caption columns to results.parquet |
| `build_captioning_viewer.py` | Two-column HTML viewer: photo grid sorted by aesthetic score + detail panel with full caption output |
| `tools/review_descriptions.py` | One-off analysis: compare two-prompt vs combined-prompt description quality (reusable for future captioning experiments) |

**Deleted**: `build_debug_viewer.py` (debug now in index_photos.html), `caption_photos_overnight.py` (experimental runner, superseded).

---

## Key constants (`constants.py`)

- `FACE_QUALITY_LABELS = ["none", "bad", "okay", "good"]`
- `FACE_CROP_PADDING = 20`
- `FACE_QUALITY_INPUT_SIZE = 128`
- `UPRIGHTER_INPUT_SIZE = 224`
- `UPRIGHTER_CONFIDENCE_THRESHOLD = 0.95`
- `CLASSIFIER_BLEND_WEIGHT = 0.8`
- `FACE_MIN_AREA_FRAC = 0.004`
- `FACE_EDGE_IMMUNE_AREA_FRAC = 0.025`
- `FACE_EDGE_ZONE_FRAC = 0.10`
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
output_dir: data/runs/june27
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

# Train face quality classifier (v7 recipe — see Classifier section)
uv run python -m still_extractor.train_classifier \
  --labels-store data/ground_truth/face_labels/labels.json \
  --results data/runs/june27/results.parquet \
  --synthetic-none-ratio 0.25 \
  --sampler-boost-good 1.25 \
  --output models/face_quality/best_model_vN.pt \
  2>&1 | tee models/face_quality/train_vN.log &

# Compare all classifier checkpoints
uv run python -m still_extractor.compare_classifiers

# Export labeled face crops to global store (prefer Export Labels button in UI)
uv run python -m still_extractor.save_labeled_faces \
  --config configs/june27.yaml \
  --output-dir data/ground_truth/face_labels

# Train uprighter
uv run python -m still_extractor.train_uprighter \
  --frames-json data/runs/june27/uprighter_frames.json \
  --rejected-json labels/rejected.json \
  --output-dir models/uprighter

# Caption good-quality keepers (three-prompt: structured fields + description + aesthetic)
uv run python -m still_extractor.caption_photos \
  --config configs/june27.yaml \
  --min-quality good

# Smoke test captioning (3 images only)
uv run python -m still_extractor.caption_photos \
  --config configs/june27.yaml \
  --min-quality good \
  --max-images 3

# Build captioning viewer
uv run python -m still_extractor.build_captioning_viewer --config configs/june27.yaml
```

---

## Output artifacts

### Per-run (under `data/runs/{run_name}/`)

| Artifact | Description |
|---|---|
| `results.parquet` | One row per keeper frame; all scores, embeddings, caption columns |
| `kept/` | Keeper JPEGs |
| `pipeline_summary.json` | Stage timing, counts, date source breakdown |
| `frame_dimensions.json` | Cached natural image dimensions for justified layout |
| `faces_review.html` | Labeling UI (rebuilt by `build_faces_review.py`) |
| `index_photos.html` | Photo viewer (rebuilt by `build_photo_viewer.py`) |
| `captioning_viewer.html` | Caption review viewer (rebuilt by `build_captioning_viewer.py`) |
| `caption_experiments/` | Overnight experiment artifacts (raw outputs, per-experiment parquets, logs) |
| `clusters.json` | Cluster assignments per frame |
| `keypoint_diagnostics.parquet` | Anomalous keypoint frames |

### Global ground truth (`data/ground_truth/`)

| Artifact | Description |
|---|---|
| `face_labels/labels.json` | Labeled face crops store — 2,328 entries, `is_val` field frozen |
| `identities/index.json` | Known identity list with name, display_name, centroid, portrait_path |
| `identities/{name}.png` | 256×256 representative portrait per identity |

---

## Face quality classifier

### Val set (frozen)

`data/ground_truth/face_labels/labels.json` has a permanently frozen val split via `is_val` field:
- **233 entries** `is_val=true` (val set — never trained on)
- **2,095 entries** `is_val=false` (train set)
- Set using `StratifiedShuffleSplit(test_size=0.1, random_state=42)` on the 2,328-label store, applied once. Do not re-derive.
- `train_classifier.py` and `compare_classifiers.py` both filter on `is_val` — do not use `StratifiedShuffleSplit` in these scripts.
- New labels added by `save_labeled_faces.py` default to `is_val=false`.

### Checkpoint history (evaluated on fixed val set)

| Version | val_acc | macro_f1 | p_good | f1_good | Notes |
|---|---|---|---|---|---|
| v3 | — | — | — | — | Pre-freeze; numbers unreliable (contaminated val) |
| v4 | — | — | — | — | Pre-freeze; numbers unreliable (contaminated val) |
| v5 | 0.738 | 0.740 | 0.727 | 0.688 | v5 recipe; current `best_model.pt` before v7 promotion |
| v7 | **0.841** | **0.840** | **0.745** | **0.788** | **Current best. `best_model.pt` → v7.** |
| v8 | — | — | — | — | EfficientNet experiment; ruled out (6× slower, worse) |
| v9 | — | — | — | — | sampler-boost=1.5 experiment; p_good regressed |
| v10 (onyx_fern) | 0.536 | 0.536 | 0.500 | — | EfficientNet-B0; significantly worse than all MobileNet |
| v11 (sable_spire) | 0.631 | 0.632 | 0.633 | 0.633 | Clean baseline on fixed val; v5 recipe on current data |

**Current best**: `best_model.pt` → `best_model_v7.pt`

### v7 recipe (the one to build on)

| Parameter | Value |
|---|---|
| Architecture | MobileNetV3-Small |
| Input size | 128×128 |
| `--sampler-boost-good` | 1.25 |
| `--synthetic-none-ratio` | 0.25 (use 0.20 if none ≥ 1.3 × good count) |
| AdamW lr | 1e-3 |
| CosineAnnealingLR | T_max=100 |
| Batch size | 32 |
| Max epochs | 100 |
| Early stop patience | 12 |
| Checkpoint save policy | `p_good` max |
| MixUp | α=0.3, Good-Good Beta(0.5,0.5) |
| Two-phase fine-tune | head-only → unfreeze features.12/13 + classifier |
| ImageNet pretrained | yes, never warm-start from another checkpoint |

v7 vs v5: patience 10→12, epochs 80→100, per-epoch Good precision logging added, conditional synthetic-none-ratio. Everything else identical.

### Codename + sidecar system (v8+)

Each training run gets an auto-generated codename (e.g. `onyx_fern`, `sable_spire`). Checkpoint filename: `best_model_vN_{codename}.pt`. Sidecar JSON at same path with `.json` extension — contains arch, hyperparams, label counts, val metrics at save epoch, timestamp.

**Record training decisions in the sidecar and/or handoff.** The sidecar is the authoritative record of what was trained and why. Do not rely on chat history alone.

### What we've learned

- **EfficientNet-B0 ruled out**: 6× slower per epoch, significantly worse at 128×128 input. MobileNetV3-Small is the right backbone for this scale.
- **Sampler boost direction**: values above 1.25 trade Good precision for Good recall (v9: boost=1.5 → p_good 0.615 vs v7's 0.745). Do not increase beyond 1.25 for precision-focused runs.
- **p_good is the right save metric**: val_loss-min and f1_good-max both produce worse precision than p_good-max at similar F1.
- **v4/v5's apparent strength was partly contamination**: on the fixed val set, v7 dominates across all four classes.
- **v11 result**: v5 recipe on current 2,328-label store gives p_good=0.633 — substantially below v7's 0.745. v7's recipe is strictly better.

### Next classifier directions (if continuing)

- Lower sampler boost to 1.0 or 1.1 (opposite direction of v9 — v4's implicit boost=1.0 may explain its historical strength)
- Focal loss term to penalize confident false-positive Goods
- Different base LR (5e-4 or 2e-3)
- More Good-class labels (currently smallest class at ~491 samples; val set is only 233 samples total — noise floor may limit measurable improvement)

---

## Uprighter

- MobileNetV3-Small, 4-class: 0°/90°/180°/270° CW
- Input: 224×224, mixed resize strategy, 3-strategy TTA at inference
- Val accuracy: 82.8% single-pass, 88.6% TTA
- Confidence threshold: 0.95
- **Known issue**: 90°↔270° confusion (~23% error rate) due to `RandomHorizontalFlip` in training augmentation.
- **Fix ready to implement**: retrain without `RandomHorizontalFlip`. v2 architecture is stable — this is now undeferred. Use existing training data.

---

## Downstream: Storybook Pipeline

### Status

`caption_photos.py` is implemented and running. Full 204-image captioning pass in progress (june27 corpus, `--min-quality good`). Identity clustering infrastructure exists; assembly step not yet started.

### Captioning (`caption_photos.py`)

**Three-prompt design** (fp16, SmolVLM2-2.2B-Instruct, left-padding for batched inference):

- **Prompt 1 — structured fields**: setting, activity, people, mood, framing
- **Prompt 2 — description**: one vivid sentence (two-prompt wins over combined; see experiments)
- **Prompt 3 — aesthetic score**: dedicated single-score prompt using Exp C rubric (100% coverage vs ~50% when combined with structured fields)

**Caption columns in results.parquet**: `caption_setting`, `caption_activity`, `caption_people`, `caption_mood`, `caption_framing`, `caption_aesthetic_score`, `caption_description`, `caption_model`, `caption_timestamp`

**Historical columns** (from overnight experiments, do not remove): `caption_lighting_score`, `caption_face_quality_score`, `caption_aesthetic_score_solo`

**Throughput**: ~6.4s/image on current hardware (18s/image for two-prompt; three-prompt ~27s/image estimated). Full 204-image run ≈ 3.5h. **Explore speed improvements**: image resize (96×96?), larger batch size, `torch.compile` — investigate before next full run.

**Filter**: `pred_label == "good"` (or `okay` with `--min-quality okay`). This matches `build_captioning_viewer.py` and `index_photos.html` quality bucketing — keep in sync.

**Idempotent**: skips already-captioned rows unless `--force`. Re-run after each pass to fill gaps.

### Captioning experiments (june27, completed)

Four prompt variants tested on 204 good-quality images:
- **Exp A** (two-prompt, structured + description): 28% all-3 score coverage — scores deprioritized when competing with structured text
- **Exp B** (scores-only): 100% coverage, but biased high (mean 8.71, poor discrimination)
- **Exp C** (aesthetic-only, dedicated rubric): 100% coverage, wider range (min=6), most discriminating → **adopted as Prompt 3**
- **Exp D** (single combined prompt): failed to emit lighting score on any of 204 images

Description format: two-prompt (A) beats combined (D) — more specific, more identifiable per-photo. Combined had 3× mode collapse. Two-prompt descriptions average 20 words ± 8.5 (combined: 12.7 ± 3.1).

Framing field coverage is low (~NaN for many images — model doesn't reliably emit it). Not blocking; only matters when storybook layout logic needs it.

Raw outputs archived in `data/runs/june27/caption_experiments/raw_outputs_A.jsonl` (prompt1_raw, prompt2_raw; prompt3 not yet archived for three-prompt runs).

### Captioning viewer (`captioning_viewer.html`)

Built by `build_captioning_viewer.py`. Two-column layout: scrollable photo grid (left, sorted by aesthetic score descending) + detail panel (right: full image, structured fields, description, collapsible raw outputs). Aesthetic score badge color-coded (green 8–10, amber 6–7, red 1–5). Keyboard navigation (arrow keys, Enter/Space).

Rebuild after each captioning pass:
```bash
uv run python -m still_extractor.build_captioning_viewer --config configs/june27.yaml
```

### Identity clustering

- ArcFace embeddings stored per-face in `results.parquet` (`face_N_embedding` columns)
- DBSCAN clustering + manual label review → identity manifest per image
- **Known issue**: identity oversplit (~25 entries for ~8 actual people). Fix: increase `DBSCAN_EPS` to 0.5–0.6, re-run `build_clusters.py`, manually reconcile.
- Use case: "make a storybook focused on Julia with a distribution of other people"

### Assembly

Not yet started. Plan: caption metadata + identity annotations fed to Claude as structured text; Claude organizes into chapters/sections.

---

## Photo viewer (`index_photos.html`)

Single output built by `build_photo_viewer.py`.

- **Layout**: justified (Google Photos-style, full aspect ratio rows, 200px target row height)
- **Year/month sections**: reverse chronological, collapsible
- **Filters**: Source (All/Images/Videos), Quality (Good/Okay/Bad/None), People (identity chips, AND/OR toggle)
- **Selection/export**: Google Photos-style circular checkboxes, shift+click range select, Export ZIP (stubbed)
- **Debug flags** (⚙ Settings, off by default): score panel, bbox/keypoint overlay
- **Pre-schema parquets**: `source_year`/`source_month` gracefully handled — missing columns coalesce to 0 → rendered as "Unknown" section

---

## Date extraction

Fallback chain per source file:
1. EXIF `DateTimeOriginal` (tag 36867)
2. EXIF `DateTime` (tag 306)
3. PyAV video metadata (`com.apple.quicktime.creationdate` or `creation_time`)
4. Path regex — `YYYY-MM`, `YYYY_MM`, `YYYYMMDD`, month-name + optional day + year, standalone `20XX`
5. mtime — **year only** (mtime month never used — copy-from-camera resets mtime)
6. `(0, 0)` — unknown

---

## Pipeline timing (approximate, june27 corpus)

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
- Refinement container reuse: ~0.8s/file saving
- TTA-uprighter re-run on keepers is redundant
- Aesthetic preprocessor: pre-downscale to ~512px → ~150ms/file

---

## Pending work

1. **Uprighter retrain**: remove `RandomHorizontalFlip` to fix 90°↔270° confusion. Ready to do — v2 is stable.
2. **Caption speed investigation**: benchmark image resize (96×96), batch size tuning, `torch.compile` before next full captioning run.
3. **Full captioning pass**: complete 204-image three-prompt run on june27 (in progress). Then rebuild `captioning_viewer.html`.
4. **Identity oversplit**: increase `DBSCAN_EPS` (0.5–0.6), re-run clustering, reconcile named + placeholder identities.
5. **ZIP export**: implement Export ZIP in `index_photos.html` (currently stubbed).
6. **Absolute path debt**: 2,084 entries in `labels.json` use `C:/Code/video-to-photo/...` absolute paths. Run a `make_paths_relative` migration if the project moves.
7. **Storybook assembly**: implement Claude-powered assembly step once captioning + identity clustering are complete.

---

## Session rules

- Always write Claude Code prompts as `.md` files to `/mnt/user-data/outputs/` and present them.
- Do not write ahead multiple prompts unless explicitly discussed. Write one, wait for results, then write the next.
- For debugging persistent bugs where the cause is ambiguous, write a diagnosis-first prompt. Otherwise give Claude Code high-level intent — it handles details correctly.
- When writing prompts for steps that could take >~30s, include instructions to estimate runtime and background the process.
- Always use the Export Labels button (or `save_labeled_faces.py`) after any significant labeling session.
- **Never overwrite a versioned checkpoint** (`best_model_vN.pt`). Always save new training runs to the next version number. Update `best_model.pt` only after explicit comparison and decision.
- **Record training decisions**: every training run should have a sidecar JSON (v8+) and be summarized in the handoff. Do not rely on chat history.
- Do not prompt about updating the handoff document — Matt initiates updates.
- Test runs use `--max-videos N --max-images M` and write to `results_test.parquet` / `pipeline_summary_test.json`.
