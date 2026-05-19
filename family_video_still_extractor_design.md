# Family Video Still-Frame Extractor — Design Handoff

## Context (read first)

I'm Matt. I have years of family videos of my daughters (10s–2min clips, mostly handheld phone footage) and want to extract a curated set of high-quality still frames for photo albums. Target hardware is Windows + NVIDIA GPU. Local processing only. I'm fine with multi-hour or multi-day runs.

This is the **planning phase**. We've already discussed the design and surveyed the prior art in a previous chat. The output of that should be a working CLI pipeline I run on my Windows machine. Personal use only (this matters for some model licenses below).

This document is the design summary. Please read it end-to-end before proposing implementation steps. Then we discuss the plan before writing any Claude Code prompts.

## Goals and non-goals

**Goals:**
- Output: a few hundred well-curated JPEGs across the whole corpus, ranked by quality, suitable for photo albums.
- Heavy precision bias. 200 great frames beats 2000 okay frames.
- Per-video count is not enforced. Some videos will yield three good frames; some will yield zero. That's fine.
- Reproducible: the same inputs should produce the same outputs.

**Non-goals:**
- No on-device training. Pretrained models only.
- No real-time anything.
- No GUI initially. Plain CLI + an HTML index page for review is fine.
- Not optimizing for last-mile decode throughput. Sampled-frame CPU decode is plenty fast for this corpus.

## Why the naive aesthetics-only approach fails

It's tempting to just score every frame with an aesthetics model and pick the maxima. We surveyed this — PerfectFrameAI on GitHub does exactly that with NIMA. The failure mode is well-defined: aesthetics models reward photographic quality (composition, color, exposure, bokeh) without caring whether your subject is actually visible and in focus. For family video, "is there a clear, in-focus face with eyes open" matters more than aesthetic score. Aesthetics ends up as one signal among several, not the gating signal.

The keyframe-extraction literature (Katna, LMSKE, etc.) solves the adjacent but wrong problem: representative coverage for video summarization, not "best stills." Don't get pulled toward clustering approaches.

The intersection (aesthetics + faces + identity + sharpness, packaged) doesn't appear to exist as open source. We're building it.

## Pipeline overview

Three passes, with persisted intermediate state so passes 2 and 3 can be re-run cheaply while iterating on scoring.

### Pass 1 — Index (slow, run once)

For each video, decode sampled frames and compute cheap features per frame. Persist to a SQLite DB or Parquet file.

1. **Sample frames at 3 fps** using ffmpeg or PyAV. Per-frame timestamp tracked.
2. **Sharpness pre-filter**: Laplacian variance on a center crop. Drop frames below a low threshold — motion-blurred frames are dead weight downstream.
3. **Face detection + landmarks + embedding** via InsightFace buffalo_l. Drop frames with no face at least N pixels wide (start with 80px).
4. **Persist per-surviving-frame**: video path, frame index, timestamp, sharpness score, face boxes, landmarks, face embeddings, frame size.

Most sampled frames get dropped at the face gate. This is the bulk of wall-clock time.

### Pass 2 — Score and rank (fast, iterate freely)

On surviving candidates:

1. **Aesthetics score** via Aesthetic Predictor V2.5 (SigLIP-based, scale 1–10, >5.5 is good).
2. **Sharpness score on face region** (not center crop). The face being sharp matters more than the background.
3. **Eye-openness** from InsightFace landmarks (eye aspect ratio).
4. **Identity match** (optional, recommended): one-time clustering of all face embeddings across the corpus → I hand-label clusters in a YAML file (`daughter_1`, `daughter_2`, `other`, `unknown`). Each candidate gets a flag for whether it contains a target identity.
5. **Composite score**. Start multiplicative so a low factor kills the frame: `aesthetics_norm * face_sharpness_norm * eye_open * (1.0 if has_target else 0.3)`. Tune from there.
6. **Local maxima within each video**, then **global ranking across the corpus**. Take top-K.
7. **Deduplicate**: near-duplicate top frames (consecutive sharp moments from the same shot are nearly identical) collapse to one. Use perceptual hash (`imagehash` library, dHash or pHash, Hamming distance threshold) on a small thumbnail, or cosine distance on CLIP embeddings if I want it heavier.

### Pass 3 — Fine refinement (cheap, on final survivors only)

For each top-K survivor:

1. Decode ±10 frames at native frame rate around the candidate timestamp.
2. Re-score sharpness and eye-openness on each.
3. Keep the best frame in the micro-window.

This fixes the "we sampled at 3 fps but the actual sharpest frame was at +0.07s" problem. Cheap because it only runs on a couple hundred candidates.

### Output

- `out/jpegs/` — full-resolution JPEGs named `{score:.3f}_{video_stem}_{timestamp_seconds:.2f}.jpg`, sorted by score.
- `out/index.html` — thumbnail grid with scores, source video filename, timestamp, "open original" link. For triage.
- `out/scores.csv` — full table of every surviving candidate with all sub-scores, for sanity-checking and re-ranking offline.

## Identity-clustering one-time step

Plan: after Pass 1 completes, run a separate script that:
1. Collects all face embeddings from the DB.
2. Clusters with HDBSCAN or simple agglomerative on cosine distance.
3. Emits `clusters/cluster_{id}.png` — a grid of representative face crops per cluster.
4. I manually edit `clusters.yaml` mapping cluster IDs to labels: `daughter_1`, `daughter_2`, `other`, `unknown`, `noise`.
5. Pass 2 consumes `clusters.yaml` for the identity-match factor.

Soft identity weighting (×0.3 for non-target, not ×0.0) is the recommended start — strict gating drops too many good frames where the face is slightly turned and the match is uncertain. Tune after seeing first output.

## Component library list (researched, current as of May 2026)

All Windows + NVIDIA compatible. Personal use okay on every license; commercial use would require care on some.

### Video frame sampling
- **ffmpeg** (https://www.gyan.dev/ffmpeg/builds/ on Windows) — CLI for sampled-timestamp decoding. Simplest.
- **PyAV** (https://github.com/PyAV-Org/PyAV) — Python bindings to ffmpeg. CPU decode is fine for sampled extraction; hardware decode via PyAV is finicky and not worth the trouble at our sample rates.
- (Skip PyNvVideoCodec and avcuda — overkill for sampled-frame work.)

### Face detection / landmarks / embeddings
- **InsightFace** (https://github.com/deepinsight/insightface) — `pip install insightface onnxruntime-gpu`. `buffalo_l` pack covers detection, 2D/3D landmarks, age/gender, and 512-D recognition embedding in one call. License caveat: **pretrained models are non-commercial research only**. Fine for my personal photo album use; do not redistribute outputs.
- Fallback if InsightFace license becomes a concern: OpenCV YuNet face detector (Apache 2.0) plus a separate landmark/embedding stage.

### Aesthetics scoring
- **Primary: Aesthetic Predictor V2.5** (https://github.com/discus0434/aesthetic-predictor-v2-5, `pip install aesthetic-predictor-v2-5`). SigLIP-based, single-package install, 1–10 scale, >5.5 is good. Newer (Dec 2024 release), handles a wider domain than V2. AGPLv3 license — fine for personal use, would matter if redistributing.
- **Fallback: improved-aesthetic-predictor** (https://github.com/christophschuhmann/improved-aesthetic-predictor). The classic CLIP ViT-L/14 + MLP head ("sac+logos+ava1-l14-linearMSE.pth"). Requires pairing with **open_clip_torch** (https://github.com/mlfoundations/open_clip, `pip install open_clip_torch`) for the CLIP backbone. Worth having as a sanity check or alternative score.

### Sharpness
- Just OpenCV (`pip install opencv-python`). `cv2.Laplacian` variance, or Tenengrad (Sobel gradient energy) per the [sharp-frame-extractor](https://github.com/cansik/sharp-frame-extractor) reference. Apply to face region in Pass 2, center crop in Pass 1.

### Deduplication
- **imagehash** (`pip install imagehash`) — dHash or pHash, Hamming distance threshold. Simple, fast. Good enough.
- Alternative: **imagededup** (https://github.com/idealo/imagededup, Apache 2.0) — heavier-weight, supports CNN-based near-dup detection. Probably overkill but available if pHash misses cases.

### Identity clustering
- `scikit-learn` agglomerative clustering, or **hdbscan** (`pip install hdbscan`) for automatic cluster-count selection. Either is fine.

### Reference projects (for code patterns, not adoption)
- **PerfectFrameAI** (https://github.com/BKDDFS/PerfectFrameAI) — for their NIMA-scorer batching pattern. Skip the Docker wrapper and HTTP server. Apache 2.0.
- **sharp-frame-extractor** (https://github.com/cansik/sharp-frame-extractor) — clean reference for the block-based sharpness selection pattern.

## Tooling preferences

- Python 3.12, `uv` for env management.
- Modern Python: `|` unions, built-in generics, `pathlib.Path` over `os.path`, `argparse`, `logging`.
- Flat module layout. No deeply nested packages.
- PyTorch with CUDA 12.x from the official PyTorch install selector — don't trust pip defaults.
- SQLite via stdlib `sqlite3` for the intermediate index. Parquet via `pyarrow` is fine if there's a reason.
- One CLI entry point per pass (`pass1_index.py`, `pass2_score.py`, `pass3_refine.py`, plus `cluster_faces.py` and `build_index_html.py`). All callable independently.

## Open questions to discuss before implementation

1. **InsightFace license**: comfortable using buffalo_l given non-commercial-research-only on the weights, since this is personal? Or pivot to YuNet + a separately licensed embedding model?
2. **Identity step: include in v1 or defer?** Adds the labeling step but is the biggest quality lever. My instinct is include it.
3. **Aesthetics model: V2.5 primary, V2 as fallback — or run both and combine?** Running both is cheap and gives a sanity check. Slightly more setup.
4. **Storage format for the intermediate index**: SQLite (single file, easy to query during iteration) vs Parquet (faster columnar, but more friction for ad-hoc queries). Leaning SQLite.
5. **First milestone**: get end-to-end on a single ~30-second test video with all stages, then scale up. Agreed?

## What I want from this conversation

Once you've read the above, please:
1. Push back on anything that looks wrong, missing, or over-engineered.
2. Propose any component swaps you'd recommend (with reasoning).
3. Once we converge, help me lay out the repo structure and the order of Claude Code prompts I'll need.

Don't write Claude Code prompts yet — we discuss the plan first, then I'll request prompts one at a time.
