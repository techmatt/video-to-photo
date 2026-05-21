# still_extractor — Project Handoff Document

*Authoritative reference for new Claude sessions. Read this before anything else.*

---

## Project overview

`still_extractor` is a Python pipeline that extracts high-quality portrait frames from a corpus of family videos and photos. The output is used to build a face quality classifier (MobileNetV3) and a labeling UI for generating training data. Located at `C:\Code\video-to-photo\`.

---

## Architecture (v2)

A single per-file pipeline replaces the old pass1/pass2/pass3 architecture. Each source file is processed end-to-end in memory; only final keeper JPEGs are written to disk.

**Per-file flow (videos):**
sample frames → temporal dedup (early, before models) → uprighter → sharpness gate → face detect → batch score (aesthetic + classifier) → face/frame dHash dedup → top-K selection → micro-window refinement → write keeper JPEG

**Per-file flow (images):**
EXIF orient → uprighter → sharpness gate → face detect → score → write keeper JPEG

**Orchestrator** handles: inventory → per-file workers → cross-file dedup → per-video cap → `results.parquet`

---

## Module inventory (`still_extractor/`)

| Module | Role |
|---|---|
| `pipeline.py` | Main orchestrator — runs full pipeline from config |
| `worker.py` | Per-file processor: `process_file(row, models, cfg) -> list[dict]` |
| `models.py` | Model loading: `load_models() -> Models` dataclass |
| `sampling.py` | Frame sampling, rotation detection (tkhd parser), sharpness |
| `inventory.py` | File crawl, dedup, manifest, `RunConfig.from_yaml()` |
| `face_crop.py` | `extract_face_crop_from_image()` (in-memory), `extract_face_crop()` (path-based wrapper) |
| `constants.py` | All shared constants + `card_key(video_stem, kept_path)` |
| `utils.py` | `safe_float`, `to_fwd_slash`, `parse_kps` |
| `build_photo_viewer.py` | Full-frame justified-grid viewer with flagging |
| `build_faces_review.py` | Face-crop labeling UI (renamed from build_index_html) |
| `train_classifier.py` | Face quality classifier training (MobileNetV3) |
| `train_uprighter.py` | Uprighter training (MobileNetV3, 4-class rotation) |
| `build_uprighter_review.py` | Uprighter training data review HTML |
| `save_labeled_faces.py` | Exports labeled face crops to `data/face_labels/` |
| `export_flagged.py` | Copies flagged frames from photo viewer to output dir |

---

## Key constants (`constants.py`)

- `FACE_QUALITY_LABELS = ["none", "bad", "okay", "good"]`
- `FACE_CROP_PADDING = 20`
- `FACE_QUALITY_INPUT_SIZE = 128`
- `UPRIGHTER_INPUT_SIZE = 224`
- `UPRIGHTER_CONFIDENCE_THRESHOLD = 0.95`
- `CLASSIFIER_BLEND_WEIGHT = 0.8`
- `card_key(video_stem, kept_path)` → `"{video_stem}/{Path(kept_path).name}"` — **must stay in sync** across `build_faces_review.py`, `train_classifier.py`, `save_labeled_faces.py`, and browser localStorage

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
# Full pipeline (inventory + per-file processing + dedup + results.parquet)
uv run python -m still_extractor.pipeline --config configs/june27.yaml

# Test run (10 videos, doesn't pollute results.parquet — writes results_test.parquet)
uv run python -m still_extractor.pipeline --config configs/june27.yaml --max-videos 10 --max-images 0

# Build viewers
uv run python -m still_extractor.build_photo_viewer --config configs/june27.yaml
uv run python -m still_extractor.build_faces_review --config configs/june27.yaml

# Train face quality classifier (after labeling in faces_review.html)
uv run python -m still_extractor.train_classifier \
  --results data/june27/results.parquet \
  --labels-json save/labels.json \
  --output-dir models/face_quality

# Export labeled face crops to global store
uv run python -m still_extractor.save_labeled_faces \
  --results data/june27/results.parquet \
  --labels-json save/labels.json \
  --output-dir data/face_labels

# Build uprighter training review HTML
uv run python -m still_extractor.build_uprighter_review \
  --frames-dir data/june27/frames \
  --output-html data/june27/uprighter_review.html

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
| `pipeline_summary.json` | Run summary: keeper counts, label distribution, uprighter corrections |
| `results.parquet` | One row per keeper frame (see schema below) |
| `kept/` | Final keeper JPEGs: `{composite:.4f}_{stem}_{timestamp_s:.3f}.jpg` |
| `faces_review.html` | Face-crop labeling UI |
| `index_photos.html` | Full-frame photo viewer with flagging |

### Global (persistent across runs)

| Artifact | Description |
|---|---|
| `data/face_labels/` | Labeled face crops + `labels.json` (output of `save_labeled_faces.py`) |
| `models/face_quality/best_model.pt` | Current face quality classifier checkpoint |
| `models/uprighter/best_model.pt` | Current uprighter checkpoint |

---

## `results.parquet` schema

```
video_path, video_stem, source_type,
timestamp_s, refined_timestamp_s, frame_index,
frame_w, frame_h,
face_x1, face_y1, face_x2, face_y2, face_w, face_det_score,
kps, embedding,
sharpness_center, refined_sharpness, sharpness_delta,
aesthetics_norm, composite,
p_none, p_bad, p_okay, p_good, pred_label, pred_confidence,
uprighter_pred, uprighter_confidence,
kept_path
```

`kept_path` is absolute. `source_type` is `"video"` or `"image"`.

---

## Labeling workflow

1. Run `build_faces_review.py` → open `faces_review.html` in browser
2. Label cards with keyboard: `1/N`=none, `2/B`=bad, `3/O`=okay, `4/G`=good, `X`=clear
3. Click "Export Labels" → save as `save/labels.json`
4. Run `train_classifier.py` to retrain
5. Run `save_labeled_faces.py` to snapshot face crops to `data/face_labels/`

**Important**: always run `save_labeled_faces.py` after a significant labeling session before re-running the pipeline. Label loss has occurred before due to `kept_path` changes across pipeline re-runs.

Labels in `save/labels.json` are keyed by `card_key(video_stem, kept_path)` = `{video_stem}/{Path(kept_path).name}`.

---

## Models

### Face quality classifier (`models/face_quality/best_model.pt`)
- MobileNetV3-Small, 4-class: none/bad/okay/good
- Input: 128×128 face crop, FACE_CROP_PADDING=20, kps-based roll correction
- Checkpoint keys: `model_state`, `epoch`, `val_loss`

### Uprighter (`models/uprighter/best_model.pt`)
- MobileNetV3-Small, 4-class: 0°/90°/180°/270° CW
- Input: 224×224, mixed resize strategy (letterbox/squish/center-crop), 3-strategy TTA at inference
- Val accuracy: 82.8% single-pass, 88.6% TTA
- Known issue: 90°↔270° confusion (~23% error rate) due to RandomHorizontalFlip in training. **Retrain without it** when ready.
- Confidence threshold: 0.95 (corrections only applied above this)

---

## Pending work

- **Run full june27 pipeline and retrain face classifier** — the v1 classifier had many errors; re-label from fresh `faces_review.html` and retrain.
- **Retrain uprighter** without `RandomHorizontalFlip` to fix 90°↔270° confusion.
- **Clean up remaining v1 artifacts** in `data/june27/`: `index.html`, `pass1.log`, `pass2.log`, `pass3.log`, `build_uprighter_review_summary.json`. Also `data/test_run/`.
- **Dedup keeper JPEGs**: 137 dedup-loser JPEGs remain in `kept/` — pipeline doesn't delete them. Low priority.

---

## Session rules

- Always write Claude Code prompts as `.md` files to `/mnt/user-data/outputs/` and present them.
- Do not write ahead multiple prompts unless explicitly discussed. Write one, wait for results, then write the next.
- For debugging persistent bugs where the cause is ambiguous, write a diagnosis-first prompt. Otherwise give Claude Code high-level intent — it handles details correctly.
- When writing prompts for steps that could take >~30s, include instructions to estimate runtime and background the process.
- Always run `save_labeled_faces.py` immediately after any significant labeling session before re-running the pipeline.
- Do not prompt about updating the handoff document at end of each topic — Matt initiates updates.
- Test runs use `--max-videos N --max-images M` and write to `results_test.parquet` (safe, doesn't pollute production).
