# still_extractor — Project Handoff Document

*Authoritative reference for new Claude sessions. Read this before anything else.*

---

## Project overview

`still_extractor` is a Python pipeline that extracts high-quality portrait frames from a corpus of family videos and photos. The output is used to build a face quality classifier (MobileNetV3) and a labeling UI for generating training data. Located at `C:\Code\video-to-photo\`.

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
| `build_photo_viewer.py` | Full-frame justified-grid viewer with flagging |
| `build_faces_review.py` | Face-crop labeling UI — filters out null-face rows automatically |
| `build_debug_viewer.py` | Debug overlay viewer: face bbox + keypoints + score breakdown per frame |
| `train_classifier.py` | Face quality classifier training (MobileNetV3) |
| `train_uprighter.py` | Uprighter training (MobileNetV3, 4-class rotation) |
| `build_uprighter_review.py` | Uprighter training data review HTML |
| `save_labeled_faces.py` | Exports labeled face crops to global store `data/face_labels/` |
| `launch_faces_export_server.py` | Local HTTP server (port 7432) — Export Labels button in faces_review.html POSTs here |
| `compare_classifiers.py` | Compares all `best_model_v*.pt` checkpoints; auto-globs from `models/face_quality/` |
| `export_flagged.py` | Copies flagged frames from photo viewer to output dir |

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
- `card_key(video_stem, kept_path)` → `"{video_stem}/{Path(kept_path).name}"` — must stay in sync across `build_faces_review.py`, `train_classifier.py`, `save_labeled_faces.py`, and browser localStorage

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

# Build viewers
uv run python -m still_extractor.build_photo_viewer --config configs/june27.yaml
uv run python -m still_extractor.build_faces_review --config configs/june27.yaml
uv run python -m still_extractor.build_debug_viewer --config configs/june27.yaml

# Start face export server (keep running while labeling)
uv run python -m still_extractor.launch_faces_export_server --config configs/june27.yaml

# Train face quality classifier
uv run python -m still_extractor.train_classifier \
  --labels-store data/face_labels/labels.json \
  --results data/june27/results.parquet \
  --synthetic-none-ratio 0.25 \
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
| `pipeline_summary.json` | Run summary: keeper counts, label distribution, uprighter corrections, stage timing, rejection stats |
| `results.parquet` | One row per keeper frame (see schema below) |
| `kept/` | Final keeper JPEGs: `{composite:.4f}_{stem}_{timestamp_s:.3f}.jpg` |
| `rejected/` | Face crop JPEGs rejected by heuristics: `{reason}_{stem}_{timestamp_s:.3f}_{face_idx}.jpg` |
| `faces_review.html` | Face-crop labeling UI (null-face rows filtered out) |
| `face_labels.json` | Labels exported from faces_review.html (input to export server) |
| `index_photos.html` | Full-frame photo viewer with flagging |
| `index_photos_debug.html` | Debug viewer: fullscreen overlay with bbox, keypoints, score breakdown |

### Global (persistent across runs)

| Artifact | Description |
|---|---|
| `data/face_labels/labels.json` | List of exported labeled face crops with sha256, corpus, label |
| `data/face_labels/seen_hashes.json` | Dedup set — prevents re-exporting the same crop |
| `data/face_labels/faces/` | Exported face crop JPEGs |
| `models/face_quality/best_model.pt` | Current best classifier (v5 as of last session) |
| `models/face_quality/best_model_v1.pt … best_model_v5.pt` | Versioned checkpoints — never overwrite |
| `models/uprighter/best_model.pt` | Current uprighter checkpoint |

---

## `results.parquet` schema

```
video_path, video_stem, source_type,
timestamp_s, refined_timestamp_s, frame_index,
frame_w, frame_h,
source_fps, file_size_bytes,

# Primary face (legacy columns — always mirrors face_1_*)
face_x1, face_y1, face_x2, face_y2, face_w, face_det_score,
kps, embedding,
p_none, p_bad, p_okay, p_good, pred_label, pred_confidence,

# Top-3 faces (ranked by p_good descending; null-filled if fewer detected)
face_1_x1 … face_1_pred_confidence,
face_2_x1 … face_2_pred_confidence,
face_3_x1 … face_3_pred_confidence,

face_count,               # total faces detected (before top-3 truncation)
best_pair_score,          # (face_1_p_good + face_2_p_good) / 2; null if <2 faces
rejected_face_count,      # faces rejected by heuristics
rejected_faces_json,      # JSON array of {x1,y1,x2,y2,reason} for rejected faces

sharpness_center, refined_sharpness, sharpness_delta,
aesthetics_norm, composite,
uprighter_pred, uprighter_confidence,
kept_path
```

`kept_path` is absolute. `source_type` is `"video"` or `"image"`. Rows with `face_count=0` (all faces rejected) have null face columns — they are valid keeper frames but have no labelable face crop.

---

## Face rejection heuristics

Applied before classifier inference. Controlled by constants in `constants.py`.

- **too_small**: face area / frame area < `FACE_MIN_AREA_FRAC` (0.004)
- **small_and_edge**: face area / frame area < `FACE_EDGE_IMMUNE_AREA_FRAC` (0.025) AND face center within `FACE_EDGE_ZONE_FRAC` (10%) of any frame edge

Rejected face crops are written to `data/{run_name}/rejected/` for audit. The debug viewer shows rejected faces as dashed red boxes with a "Show Rejected" toggle.

---

## Labeling workflow

1. Start the export server: `uv run python -m still_extractor.launch_faces_export_server --config configs/june27.yaml`
2. Run `build_faces_review.py` → open `faces_review.html` in browser
3. Label cards with keyboard: `1/N`=none, `2/B`=bad, `3/O`=okay, `4/G`=good, `X`=clear
4. Click **Export Labels** — POSTs to the export server, which runs `save_labeled_faces.run_export()` and shows an alert with counts: new faces added, already in store, skipped
5. The global store at `data/face_labels/` is updated atomically (dedup via `seen_hashes.json`)

**Important**: always use the Export Labels button (or run `save_labeled_faces.py`) after any significant labeling session before re-running the pipeline. Label loss has occurred before.

---

## Face quality classifier

### Architecture
MobileNetV3-Small, 4-class (none/bad/okay/good), 128×128 input, kps-based roll correction, ~1.5M params.

### Training (`train_classifier.py`)
- Data source: `data/face_labels/labels.json` (global store, direct from `face_crop_path` — no parquet join)
- Synthetic none augmentation: random background crops from keeper JPEGs, non-overlapping with face bboxes, per-epoch resampling, ratio controlled by `--synthetic-none-ratio` (default 0.25)
- Class balancing: `WeightedRandomSampler` with `SAMPLER_BOOST = {good: 1.75, others: 1.0}`
- MixUp α=0.3; Good-Good pairs use Beta(0.5, 0.5)
- AdamW lr=1e-3, CosineAnnealingLR, batch=32, epochs=80, early stop patience=10
- Two-phase fine-tune: phase 1 head only (epochs 1→40), phase 2 unfreeze features.12/13+classifier (epochs 41→80)
- Always start from ImageNet pretrained weights — never warm-start from a prior checkpoint
- **Always save to a new versioned file** (`best_model_vN.pt`). Update `best_model.pt` only after explicit comparison.

### Checkpoint history

| Version | Epoch | Val Loss | Good F1 | Good Precision | Notes |
|---|---|---|---|---|---|
| v1 | 59 | 1.157 | 0.537 | — | Old label distribution |
| v2 | 42 | 0.859 | 0.679 | — | Best macro F1 (0.707); different label split |
| v3 | 52 | 0.989 | 0.580 | 0.476 | SAMPLER_BOOST 1.75, MixUp α→0.3 |
| v4 | 58 | 0.954 | 0.522 | 0.429 | Static synthetic none (sampled once) |
| v5 | 47 | 0.958 | 0.520 | **0.565** | Per-epoch synthetic resampling, boost→1.25; **current best_model.pt** |

**v5 is current `best_model.pt`** — chosen for best Good precision (0.565), minimizing junk in top-200 selection. v2 has best macro F1 if overall quality matters more than precision on Good.

### Comparison tool
```bash
uv run python -m still_extractor.compare_classifiers --labels-store data/face_labels/labels.json
```
Auto-globs all `best_model_v*.pt` in `models/face_quality/`. Run after each new version.

---

## Pipeline timing

Stage timings are recorded per-file and aggregated into `pipeline_summary.json` under `stage_times_s` (total/mean/max per stage) and `stage_times_pct`. Printed as a sorted table at end of each run.

Approximate per-file times on current hardware (june27 corpus, 20-file sample):

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
- Current: v5 (epoch 47, val_loss 0.958)
- See checkpoint history table above

### Uprighter (`models/uprighter/best_model.pt`)
- MobileNetV3-Small, 4-class: 0°/90°/180°/270° CW
- Input: 224×224, mixed resize strategy, 3-strategy TTA at inference
- Val accuracy: 82.8% single-pass, 88.6% TTA
- **Known issue**: 90°↔270° confusion (~23% error rate) due to `RandomHorizontalFlip` in training. Retrain without it when ready — deferred until after other priorities.
- Confidence threshold: 0.95

---

## Downstream: Storybook Pipeline (planned, not yet implemented)

Goal: select top ~200 frames from a corpus, annotate with captions and person identities, feed to Claude to organize into a storybook.

### Captioning
- Model: **SmolVLM2 (2B)** — fits in 8GB VRAM, ~2s/image, good structured output
- Prompt for structured short-form output (not long prose):
  `"Describe this photo using only these fields: setting (one phrase), activity (one phrase), people (count and relationship), mood (one word), framing (close portrait / medium / wide action). Be brief."`
- Output per image: ~20 tokens of structured metadata

### Identity clustering
- ArcFace embeddings already stored in `results.parquet` (`embedding` column)
- Plan: DBSCAN or hierarchical clustering on embeddings → top-N identity clusters → manual label review grid → assign identifiers ("Julia", "Person B", etc.)
- Per-image person manifest fed to Claude alongside captions
- Use case: "make a storybook focused on Julia with a distribution of other people" — filter/score by identity composition

### Assembly
- Both caption metadata and identity annotations fed to Claude as structured text
- Claude organizes into chapters/sections; layout step TBD (PDF template or other)

---

## Pending work

- **Retrain uprighter** without `RandomHorizontalFlip` to fix 90°↔270° confusion
- **More Good labels** — Good class still has ~370 samples; more would improve Good F1 across all model versions. Per-epoch synthetic resampling (v5) helps None precision but hasn't closed the Good gap.
- **v6 idea**: combine per-epoch resampling (v5) with SAMPLER_BOOST back at 1.75 (v3 level) — the reduced boost in v5 (1.25) likely worked against Good recall; threading both together may improve Good F1 without sacrificing precision
- **Storybook pipeline**: implement captioning script and identity clustering (see above)
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
- Pre-playtest for Helium Hustle applies to that project only — not this one.
