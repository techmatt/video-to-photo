# still_extractor — Project Handoff Document

*Authoritative reference for new Claude sessions. Read this before anything else. The original handoff document is superseded by this one.*

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
- Key deps: `pyav`, `insightface`, `onnxruntime-gpu`, `aesthetic-predictor-v2-5`, `open_clip_torch`, `opencv-python`, `imagehash`, `pandas`, `pyarrow`, `scikit-learn`, `hdbscan`, `Pillow`, `pillow-heif`, `pyyaml`, `tqdm`, `torch`, `torchvision`

---

## Corpus

- ~10,000 videos + images total at full scale; ~1,500 in the current mini test dataset
- Roughly 50/50 videos and images
- Video formats: `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`
- Image formats: `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, `.tiff`, `.tif`, `.bmp`
- HEIC is common (iPhone). `pillow-heif` registered at module load handles this transparently
- Long videos: up to ~5 minutes. Videos >60s get windowed sampling
- All files on the same Windows machine; some on external drives

---

## Pipeline overview

Eight scripts, each independently callable. All read from a **run config YAML** (e.g. `configs/mini.yaml`).

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

Crawls directories, hashes files, deduplicates, probes video durations, pre-computes sampling windows for long videos. Writes `manifest.csv`. Run once (or `--rescan` to pick up new files).

**CLI**: `python -m still_extractor.inventory --config configs/mini.yaml [--rescan]`

**Key behaviors**:
- Hashes first 64KB (MD5) for dedup. First-seen (lexicographic path order) is canonical.
- Long videos (>60s): pre-computes N non-overlapping 1-second sampling windows with minimum spacing, stored as JSON in `sample_windows_s` column.
- Sorts manifest by `size_bytes` ascending so Pass 1 processes small files first.

**Output**: `{output_dir}/manifest.csv`

---

### Script 2: `pass1_index.py`

The slow indexing pass. Reads manifest, processes every non-duplicate file through sharpness filter + InsightFace face detection, writes surviving frames to `index.parquet`. Resumable via `pass1_status.csv`.

**CLI**: `python -m still_extractor.pass1_index --config configs/mini.yaml [--rescan] [--sharpness-threshold 75.0] [--min-face-px 80] [--fps 3.0] [--workers 1]`

**Key behaviors**:
- Sharpness pre-filter: Laplacian variance on center 70%×70% crop. **Default threshold: 75.0** (deliberately lowered from 100.0 — do not raise without good reason; at 100.0 ~80 files were killed entirely at the sharpness gate).
- Face detection: InsightFace `buffalo_l`, `det_size=(640,640)`, `ctx_id=0`.
- Images: single row per image, `frame_index=0`, `timestamp_s=0.0`.
- Lazy directory creation: frames directory only created immediately before first frame write.
- Pass 1 writes **full-frame JPEGs** to `frames/`. Face crops are extracted on-the-fly at HTML build time — Pass 1 does not write face crop JPEGs.

**Parquet schema** (key columns): `video_path`, `video_stem`, `frame_index`, `timestamp_s`, `frame_path`, `frame_w`, `frame_h`, `sharpness_center`, `face_x1/y1/x2/y2`, `face_w`, `face_det_score`, `kps` (JSON string, 5-point landmarks), `embedding` (JSON string, 512-d L2-normalized)

**Output**: `{output_dir}/index.parquet`, `{output_dir}/pass1_status.csv`, `{output_dir}/frames/{stem}/*.jpg`

---

### Script 3: `pass2_score.py`

Scores all candidates, deduplicates, emits per-file top-K. Integrates trained classifier. Fast — iterate freely without re-running Pass 1.

**CLI**: `python -m still_extractor.pass2_score --config configs/mini.yaml [--top-k-per-file 5] [--top-k-global 0] [--classifier-model models/face_quality/best_model.pt]`

**Scoring**:
1. **Aesthetics**: Aesthetic Predictor V2.5, score 1–10 normalized to [0,1]
2. **Face sharpness**: Laplacian variance on face bbox crop, per-corpus normalized
3. **Eye-openness**: inter-eye distance / face height ratio, per-corpus normalized
4. **Composite (old)**: weighted average of the three sub-scores
5. **Classifier blend** (when `--classifier-model` provided and file exists):
   `composite = 0.8 * p_good_tta + 0.2 * composite_old`
   Classifier is MobileNetV3-Small, 3-pass TTA, loaded from `models/face_quality/best_model.pt`

**Three-stage deduplication**:
1. Temporal gate: same video + within `--temporal-window-s` → keep higher composite
2. Face-crop dHash: Hamming distance on face crop hash
3. Full-frame dHash: Hamming distance on full frame hash

**Selection**: per-file top-K (default 5) after dedup, then optional global cap (`--top-k-global`, default 0 = no cap), then cross-file dedup pass.

**Output columns added by classifier**: `composite_old`, `p_none_tta`, `p_bad_tta`, `p_okay_tta`, `p_good_tta`, `pred_label`, `pred_confidence`

**Output**: `{output_dir}/scores.csv`, `{output_dir}/top_frames/*.jpg`

---

### Script 4: `pass3_refine.py`

Micro-window refinement. For each `final_selection` candidate, decodes ±`--window-s` seconds at native frame rate and picks the sharpest face frame.

**CLI**: `python -m still_extractor.pass3_refine --config configs/mini.yaml [--window-s 0.5]`

**Key behaviors**:
- Reads `final_selection == True` rows from `scores.csv` (not `dedup_kept` — this was patched)
- Image-source rows (image extension + ts=0 + frame_index=0) are passed through unchanged: `refined_frame_path = frame_path`, `sharpness_delta = 0.0`
- Propagates all classifier columns (`p_*_tta`, `pred_label`, `pred_confidence`, `composite_old`) into `refined_scores.csv`

**Output**: `{output_dir}/refined/*.jpg`, `{output_dir}/refined_scores.csv`

---

### Script 5: `build_index_html.py`

Self-contained HTML review + labeling page. Shows **face crops** (not full frames), extracted on-the-fly with rotation correction from `kps` landmarks.

**CLI**: `python -m still_extractor.build_index_html --scores-csv data/mini/refined_scores.csv --output-html data/mini/index.html [--inference-csv data/mini/classifier/inference_scores.csv]`

**Key behaviors**:
- Face crops extracted via `face_crop.py` with `kps` for upright rotation correction (roll angle from eye landmarks, corrected if >2°)
- When classifier columns are present in `refined_scores.csv`, `--inference-csv` is not needed — predictions are read directly from the scores CSV
- Default `--inference-csv` path: `{scores_csv_dir}/classifier/inference_scores.csv`
- localStorage keyed by `"{video_stem}/{Path(refined_frame_path).name}"` — stem-prefixed bare filename. This is the only correct key format; any deviation causes counter bugs.
- No label seeding from files on page load — localStorage is the sole source of truth
- Labels exported to `save/labels.json` (repo root)

**4-category labeling UI** (settled taxonomy):
- `1/N` — **None**: not a face, InsightFace false positive (dark red `#8B0000`)
- `2/B` — **Bad**: real face, unusable (saturated red `#FF1111`)
- `3/O` — **Okay**: real face with issues, candidate-worthy (amber `#F59E0B`)
- `4/G` — **Good**: target quality (green `#22C55E`)
- `X` — clear label

**Hover-to-label**: mouse `mousemove` (not `mouseenter`) sets active card; `userHasHovered` gate prevents spurious keypresses on page load; no auto-advance on keypress.

**Prediction toolbar** (when classifier data present): Pred filter (All/None/Bad/Okay/Good/Uncertain) + Sort (Composite / Pred Confidence ↓), combinable with GT filter.

**Filter bar order**: All · None · Bad · Okay · Good · Unreviewed

---

### Script 6: `build_photo_viewer.py`

Full-frame photo viewer for browsing and selecting frames for export. Separate from the face-crop labeling HTML.

**CLI**: `python -m still_extractor.build_photo_viewer --scores-csv data/mini/refined_scores.csv --output-html data/mini/index_photos.html --parquet data/mini/index.parquet`

**Key behaviors**:
- Shows full refined JPEGs referenced by path (not base64-embedded) with `loading="lazy"`
- Justified (packed-row) grid layout — no dead space, rows fill container width
- CSS rotation correction from `kps` roll angle per card, using `object-fit: cover` + `overflow: hidden`
- Default filter: **Good** predicted label
- Sort options: Pred Confidence ↓, Aesthetic ↓, Coverage ↓
- Face coverage ratio: `(face_w * face_h) / (frame_w * frame_h)` from joined parquet
- Flagging: click flag button on card or Spacebar in lightbox; flagged cards get gold border
- Export Flagged: downloads `flagged.json` with list of export source paths
  - Image-source rows → export path = original `video_path`
  - Video-source rows → export path = `refined_frame_path`
- Lightbox: click image, arrow key navigation, Escape to close

**Output**: `data/mini/index_photos.html`

---

### Script 7: `export_flagged.py`

Copies flagged files to export directory.

**CLI**: `python -m still_extractor.export_flagged --flagged-json flagged.json --output-dir export/`

---

### Script 8: `train_classifier.py`

Trains 4-class face quality classifier (None/Bad/Okay/Good) on labeled face crops.

**CLI**: `python -m still_extractor.train_classifier --scores-csv data/mini/refined_scores.csv --labels-json save/labels.json --output-dir models/face_quality --epochs 60 --seed 42`

**Architecture**: MobileNetV3-Small (ImageNet pretrained), 4-class softmax head.

**Training strategy**:
- Phase 1 (epochs 1 to `epochs//2`): head only, **no early stopping** — runs unconditionally regardless of val loss plateau
- Phase 2 (epochs `epochs//2 + 1` to `epochs`): unfreeze last conv block + head, early stopping with patience=10 on val loss
- Best checkpoint saved during Phase 2 only
- Early stopping only applies in Phase 2 — this is critical; early stopping in Phase 1 prevents Phase 2 from ever running

**Augmentation stack** (training):
- RandomHorizontalFlip, RandomRotation ±15°, RandomResizedCrop (scale 0.8-1.0)
- RandomPerspective (distortion 0.15, p=0.3)
- ColorJitter (brightness/contrast 0.4, saturation 0.3, hue 0.05)
- RandomGrayscale (p=0.08), GaussianBlur (p=0.3), RandomErasing (p=0.3)
- JPEG re-compression simulation (quality 60-95)
- MixUp (alpha=0.2) with soft cross-entropy loss
- WeightedRandomSampler for class balance
- Label smoothing ε=0.1 on val loss

**Inference**: 3-pass TTA for Pass 2 integration (speed), 5-pass TTA for `inference_scores.csv`

**Label format**: `save/labels.json` keyed by `"{video_stem}/{Path(refined_frame_path).name}"`

**Model location**: `models/face_quality/best_model.pt` (permanent; not tied to data run dir)

**Current trained model**: epoch 59 of 60, val_loss 0.8407, val_acc ~0.72, trained on 1260 labeled mini-corpus frames. Phase 2 engaged and ran to completion.

---

### Shared helper: `face_crop.py`

`extract_face_crop(image_path, x1, y1, x2, y2, padding=10, kps=None) -> Image`

When `kps` provided and `abs(roll) > 2°`: expands crop region by `padding*3`, rotates by `-angle` with bicubic resampling, crops back to target size. Used by `build_index_html.py`, `build_photo_viewer.py`, and `train_classifier.py`.

---

### Deferred: `cluster_faces.py`

Stub only. Cluster face embeddings with HDBSCAN. Deferred to v2.

---

## Key design decisions

**Why per-file top-K not global**: global top-K is dominated by a handful of well-lit clips. Per-file top-5 ensures coverage across the whole corpus.

**Why 75 sharpness threshold**: at 100, ~80 files were killed entirely at the sharpness gate. 75 unlocks those files while face detection remains the next gate.

**Why train on face crops not embeddings**: InsightFace embeddings are optimized for identity, not quality. Quality-relevant features (sharpness, expression, occlusion, lighting) are better captured by a pretrained image backbone with direct pixel access.

**Why 4-category taxonomy**: None (false positive) and Bad (real face, poor quality) are qualitatively different signals for the classifier. Their embeddings cluster differently. Keeping them separate gives the model a cleaner training signal.

**Why 0.8/0.2 classifier/composite blend**: classifier signal dominates but composite (aesthetic + sharpness + eye-openness) provides a meaningful fallback for uncertain predictions.

**Why Pass 3 reads `final_selection` not `dedup_kept`**: `dedup_kept` reflects the 3-stage dedup survivors (~1422 rows); `final_selection` reflects the per-file top-K output (~1260 rows). Pass 3 should refine the curated set, not the full dedup survivors.

**Why image-source rows are passed through in Pass 3**: images have no video to decode for micro-window refinement. Passthrough copies `frame_path` → `refined_frame_path` with `sharpness_delta=0`.

**InsightFace license**: `buffalo_l` weights are non-commercial research only. This tool is personal use — acceptable.

---

## Current status

**Mini corpus** (~1,500 files): fully processed end-to-end.
- Inventory: 1,481 unique files (980 images + 501 videos, 43 long videos windowed)
- Pass 1: 4,965 candidate frames
- Pass 2: 1,261 final_selection frames with classifier blend active
- Pass 3: 1,261 refined frames (726 new JPEGs + 535 image-source passthroughs)
- Labeling: 1,260 frames labeled (160 none / 586 bad / 324 okay / 190 good)
- Classifier: trained, val_acc ~0.72, integrated into Pass 2
- Photo viewer: implemented with justified grid + CSS rotation (prompt 33, just executed)

**Pending**:
- Test on new (non-mini) corpus — not yet started
- Full ~10,000 file corpus run — pending new corpus validation
- Classifier v2: multi-face quality, scene composition, background aesthetics — deferred
- `cluster_faces.py` — deferred to v2
- Pass 3 JPEG re-encode quality: currently at Pillow default (~75). For video-source frames, could raise to 95 for higher quality export. Not yet implemented.

---

## Workflow process

This project is developed collaboratively between Matt and Claude (claude.ai). Matt designs systems and reviews outputs; implementation is done via **Claude Code prompts** — Markdown `.md` files written by Claude to `/mnt/user-data/outputs/` and executed by Claude Code in the repo. Claude never prints prompts inline. Each prompt is numbered sequentially and scoped to a single logical change. Prompts include cleanup steps (deleting stale outputs) unless deletion would be destructive. Long-running steps always include an instruction to estimate runtime and background the process. Debugging prompts always read the actual current code before proposing a fix — never assume the bug from symptoms alone. Matt initiates handoff document updates; Claude does not prompt for them.

---

## Notes for future Claude sessions

- Always write Claude Code prompts as `.md` files to `/mnt/user-data/outputs/` — never print inline
- Do not suggest re-running inventory unless new files have been added
- The uv/torch CUDA setup is working — do not suggest changing it
- `onnxruntime` and `onnxruntime-gpu` conflict — only `onnxruntime-gpu` should be installed
- InsightFace `buffalo_l` weights download to `%USERPROFILE%\.insightface\models\` on first run
- Pre-playtest project — skip all save migration text and effort
- localStorage key format is `"{video_stem}/{Path(refined_frame_path).name}"` — any deviation causes counter bugs
- The `build_index_html.py` counter has had repeated bugs; if counter numbers look wrong, write a diagnosis-first prompt that reads the actual JS before patching
- When Matt asks for status analysis, read CSVs via bash/pandas — do not ask him to paste contents
- `face_crop.py` rotation correction uses `kps` from parquet/scores CSV — always pass `kps` when calling `extract_face_crop` in any new script
- The photo viewer (`build_photo_viewer.py`) references images by file path, not base64 — do not change this
- Labels live at `save/labels.json` (repo root) — this is where the HTML Export Labels button writes, and where `train_classifier.py` reads from
