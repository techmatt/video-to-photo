"""Caption photos with SmolVLM2 and write back to results.parquet.

Standalone tool, not part of the extraction pipeline. Run after a pipeline run:

    uv run python -m still_extractor.caption_photos \\
      --config configs/june27.yaml \\
      --min-quality good

Quality buckets come from the face quality classifier's ``pred_label`` (the same
field ``build_photo_viewer.py`` uses for its quality filter chips), not from a
composite-score threshold.
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from still_extractor.inventory import RunConfig

logger = logging.getLogger(__name__)

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
MODEL_SHORT_NAME = "SmolVLM2-2.2B-Instruct-3prompt"

MIN_QUALITY_ALLOWED: dict[str, set[str]] = {
    "good": {"good"},
    "okay": {"good", "okay"},
}

PROMPT_STRUCTURED = (
    "Describe this photo using exactly these fields and no others. "
    "Be brief - one phrase or word per field.\n"
    "\n"
    "setting: [where the photo was taken, e.g. playground, kitchen, beach]\n"
    "activity: [what is happening, e.g. eating, playing, laughing]\n"
    "people: [count and rough relationship, e.g. one child, two children, "
    "child and adult, group]\n"
    "mood: [one word, e.g. joyful, calm, silly, focused, tired]\n"
    "framing: [close portrait, medium, or wide action]\n"
    "\n"
    "Respond with only the fields above, nothing else."
)

PROMPT_DESCRIPTION = (
    "Describe this photo in one sentence. Focus on the people, what they are "
    "doing, and the setting. Be specific and vivid but concise. Do not start "
    'with "The photo shows" or "In this image".'
)

PROMPT_AESTHETIC = (
    "Rate the overall aesthetic quality of this photo for use in a family "
    "picture book. Consider: composition, lighting, sharpness, emotional "
    "warmth, and visual appeal. Respond with only a single integer from 1 "
    "to 10. Nothing else."
)

TEXT_FIELDS = ("setting", "activity", "people", "mood", "framing")
# Retained for import-compatibility with caption_photos_overnight.py (historical
# experiment runner). The new structured prompt no longer emits these fields.
SCORE_FIELDS = ("lighting", "face_quality", "aesthetic")

CAPTION_COLUMNS_STR = [
    "caption_setting", "caption_activity", "caption_people", "caption_mood",
    "caption_framing", "caption_description", "caption_model", "caption_timestamp",
]
CAPTION_COLUMNS_INT = [
    "caption_aesthetic_score",
]
ALL_CAPTION_COLUMNS = CAPTION_COLUMNS_STR + CAPTION_COLUMNS_INT

DIGIT_RE = re.compile(r"\d+")

MAX_IMAGE_LONGEST_SIDE = 1024

# torch.compile is only attempted when we have enough images to amortize its
# warmup cost. Below this threshold, the warmup pass alone takes longer than
# the savings it produces.
COMPILE_MIN_IMAGES = 50


def _load_image_capped(path: str) -> tuple[Image.Image, tuple[int, int], tuple[int, int]]:
    """Open an image as RGB and cap its longest side at MAX_IMAGE_LONGEST_SIDE.

    Returns (image, original_size, final_size) where sizes are (w, h).
    """
    img = Image.open(path).convert("RGB")
    orig = img.size
    longest = max(orig)
    if longest > MAX_IMAGE_LONGEST_SIDE:
        scale = MAX_IMAGE_LONGEST_SIDE / longest
        new_size = (max(1, int(round(orig[0] * scale))), max(1, int(round(orig[1] * scale))))
        img = img.resize(new_size, Image.LANCZOS)
    return img, orig, img.size


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_structured_output(text: str) -> dict:
    """Parse a structured prompt response into a dict of text caption fields.

    Returns a dict with keys matching TEXT_FIELDS (str | None). Robust to
    extra surrounding text and missing fields. Score fields are no longer
    part of the structured prompt; aesthetic comes from PROMPT_AESTHETIC.
    """
    out: dict[str, str | None] = {f: None for f in TEXT_FIELDS}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower().lstrip("-* ").rstrip()
        value = value.strip().strip("[]").strip()
        if not value:
            continue
        if key in TEXT_FIELDS:
            out[key] = value
    return out


def parse_aesthetic_score(text: str) -> int | None:
    """Parse the aesthetic-prompt response into an int in [1, 10] or None."""
    m = DIGIT_RE.search(text.strip())
    if not m:
        return None
    try:
        n = int(m.group(0))
    except ValueError:
        return None
    return max(1, min(10, n))


# ---------------------------------------------------------------------------
# Model + inference
# ---------------------------------------------------------------------------

def load_model(device: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if device == "auto":
        device_map = "auto"
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_map = device
        target_device = device

    t0 = time.time()
    # fp16, not bf16: Turing GPUs (RTX 20xx) emulate bf16 in software and run
    # ~order-of-magnitude slower than native fp16 tensor cores. Ampere+ has
    # both natively; fp16 is fine for inference everywhere.
    print(f"Loading {MODEL_ID} (device={target_device}, dtype=float16)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # Decoder-only generation needs left padding so batched outputs all start
    # at the same position; right padding silently produces empty completions
    # for non-longest items.
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model.eval()
    load_s = time.time() - t0

    real_device = str(next(model.parameters()).device)
    print(f"Model loaded in {load_s:.1f}s on {real_device}")
    return processor, model


def _build_messages(prompt: str) -> list[dict]:
    """Single user turn with one image placeholder + prompt text."""
    return [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]


@torch.inference_mode()
def run_prompt_batch(
    processor,
    model,
    images: list[Image.Image],
    prompt: str,
    max_new_tokens: int,
) -> list[str]:
    """Run one prompt over a batch of images. Returns a list of decoded outputs."""
    messages_per_image = [_build_messages(prompt) for _ in images]
    prompts = [
        processor.apply_chat_template(m, add_generation_prompt=True)
        for m in messages_per_image
    ]
    inputs = processor(
        text=prompts,
        images=[[img] for img in images],
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        k: (v.to(model.device, dtype=model.dtype) if v.is_floating_point() else v.to(model.device))
        for k, v in inputs.items()
    }
    input_len = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    new_tokens = output_ids[:, input_len:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Selection + I/O
# ---------------------------------------------------------------------------

def select_rows(
    df: pd.DataFrame, min_quality: str, force: bool, max_images: int | None,
) -> tuple[pd.Index, int]:
    """Return (selected_row_indices, n_already_captioned_in_filter).

    Selected rows are kept_path-present, pred_label in the allowed set, and
    (unless force) lack a non-null caption_setting.
    """
    allowed = MIN_QUALITY_ALLOWED[min_quality]
    base_mask = df["kept_path"].notna() & df["pred_label"].isin(allowed)

    n_already = 0
    if "caption_setting" in df.columns:
        already_mask = base_mask & df["caption_setting"].notna()
        n_already = int(already_mask.sum())

    if not force and "caption_setting" in df.columns:
        sel_mask = base_mask & df["caption_setting"].isna()
    else:
        sel_mask = base_mask

    selected_idx = df.index[sel_mask]
    if max_images is not None and max_images >= 0:
        selected_idx = selected_idx[:max_images]
    return selected_idx, n_already


def ensure_caption_columns(df: pd.DataFrame) -> None:
    """Add missing caption_* columns in place with appropriate nullable dtypes."""
    for col in CAPTION_COLUMNS_STR:
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df), dtype="object")
    for col in CAPTION_COLUMNS_INT:
        if col not in df.columns:
            df[col] = pd.Series([pd.NA] * len(df), dtype="Int8")


def write_parquet_atomic(df: pd.DataFrame, dest: Path) -> None:
    tmp = dest.with_name(dest.stem + "_captions_tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(dest)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    df: pd.DataFrame,
    captioned_idx: list[int],
    total_s: float,
) -> None:
    n = len(captioned_idx)
    if n == 0:
        print("No images captioned.")
        return

    mins, secs = divmod(int(total_s), 60)
    rate = total_s / max(n, 1)
    print()
    print(f"Captioned: {n} images in {mins}m {secs:02d}s ({rate:.2f}s/image)")
    print()

    sub = df.loc[captioned_idx]
    coverage_specs = [
        ("setting", "caption_setting"),
        ("activity", "caption_activity"),
        ("people", "caption_people"),
        ("mood", "caption_mood"),
        ("framing", "caption_framing"),
        ("aesthetic_score", "caption_aesthetic_score"),
        ("description", "caption_description"),
    ]
    print("Structured field coverage (non-null):")
    label_w = max(len(label) for label, _ in coverage_specs)
    for label, col in coverage_specs:
        present = int(sub[col].notna().sum())
        pct = (100.0 * present / n) if n else 0.0
        print(f"  {label:<{label_w}} {present}/{n} ({pct:.0f}%)")
    print()

    print("Score distributions:")
    for label, col in (
        ("aesthetic_score", "caption_aesthetic_score"),
    ):
        vals = pd.to_numeric(sub[col], errors="coerce").dropna()
        if len(vals) == 0:
            print(f"  {label}: (no values)")
            continue
        print(
            f"  {label}: mean={vals.mean():.1f}  median={int(vals.median())}  "
            f"min={int(vals.min())}  max={int(vals.max())}"
        )
    print()

    sample_n = min(5, n)
    print(f"Sample captions ({sample_n} random rows):")
    sample = sub.sample(n=sample_n, random_state=0) if n > sample_n else sub
    for _, row in sample.iterrows():
        kp = row.get("kept_path") or ""
        print(f"  - {Path(str(kp)).name}")
        print(f"      setting: {row.get('caption_setting')}")
        print(f"      mood:    {row.get('caption_mood')}")
        print(f"      aesthetic: {row.get('caption_aesthetic_score')}")
        desc = row.get("caption_description")
        if isinstance(desc, str) and desc:
            print(f"      desc: {desc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _compile_and_warmup(processor, model, sample_image: Image.Image, batch_size: int):
    """Compile the model with reduce-overhead mode and run one full-batch warmup.

    Caller is responsible for the size gate (COMPILE_MIN_IMAGES); this just does
    the work. Returns the compiled wrapper to use for subsequent generate calls.
    Warms at ``batch_size`` so the common steady-state shape hits the compiled path.
    """
    t0 = time.time()
    print("Compiling model (torch.compile mode=reduce-overhead) + warmup pass...")
    compiled = torch.compile(model, mode="reduce-overhead")
    warm_batch = [sample_image] * batch_size
    for prompt, max_new in (
        (PROMPT_STRUCTURED, 120),
        (PROMPT_DESCRIPTION, 120),
        (PROMPT_AESTHETIC, 8),
    ):
        run_prompt_batch(processor, compiled, warm_batch, prompt, max_new)
    print(f"Warmup complete in {time.time() - t0:.1f}s")
    return compiled


def caption_run(
    cfg: RunConfig,
    min_quality: str,
    batch_size: int,
    device: str,
    force: bool,
    max_images: int | None,
    compile_model: bool,
) -> None:
    results_path = cfg.output_dir / "results.parquet"
    if not results_path.exists():
        raise FileNotFoundError(f"results.parquet not found: {results_path}")

    df = pd.read_parquet(results_path)
    ensure_caption_columns(df)

    selected_idx, n_already = select_rows(df, min_quality, force, max_images)
    n_to_caption = len(selected_idx)

    n_in_filter = int(
        (df["kept_path"].notna() & df["pred_label"].isin(MIN_QUALITY_ALLOWED[min_quality])).sum()
    )
    print(f"Captioning {n_to_caption} images "
          f"(pred_label in {sorted(MIN_QUALITY_ALLOWED[min_quality])}, "
          f"min-quality={min_quality})")
    print(f"In filter: {n_in_filter}.  Already captioned: {n_already} "
          f"(skipping, use --force to re-run)")

    if n_to_caption == 0:
        print("Nothing to do.")
        return

    # Benchmark on RTX 2060 (still_extractor/tools/benchmark_captioning.py):
    # bs=4 -> 0.296 img/s, bs=2 -> 0.281, bs=1 -> 0.268. Use a per-image budget
    # that loosely tracks batch size, with a floor for very small batches.
    est_per_image_s = 3.4 if batch_size >= 4 else (3.6 if batch_size >= 2 else 3.7)
    est_s = n_to_caption * est_per_image_s
    est_m, est_sec = divmod(int(est_s), 60)
    print(f"Estimated runtime: ~{est_m}m {est_sec:02d}s "
          f"({n_to_caption} images x ~{est_per_image_s:.1f}s at batch_size={batch_size})")

    processor, model = load_model(device)

    idx_list = list(selected_idx)

    if compile_model:
        if n_to_caption < COMPILE_MIN_IMAGES:
            print(f"--compile requested but only {n_to_caption} images "
                  f"(< {COMPILE_MIN_IMAGES}); skipping compile.")
        else:
            try:
                warm_path = df.at[idx_list[0], "kept_path"]
                warm_img, _, _ = _load_image_capped(warm_path)
                model = _compile_and_warmup(processor, model, warm_img, batch_size)
                try:
                    warm_img.close()
                except Exception:
                    pass
            except Exception as e:
                logger.warning("Compile/warmup failed (%s); falling back to eager.", e)

    n_batches = (n_to_caption + batch_size - 1) // batch_size
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Note: we considered prompt-batched ordering (run all prompt-1s, then all
    # prompt-2s, then all prompt-3s across the full image set). Benchmarked in
    # still_extractor/tools/benchmark_captioning.py at this hardware (RTX 2060,
    # 8 GB) and it was substantially slower than the per-image-batch ordering
    # below, so we keep the per-batch 3-prompt loop.

    pbar = tqdm(total=n_to_caption, desc="captioning", unit="img")
    for b in range(n_batches):
        batch_idx = idx_list[b * batch_size : (b + 1) * batch_size]
        images: list[Image.Image] = []
        valid_idx: list[int] = []
        for i in batch_idx:
            path = df.at[i, "kept_path"]
            try:
                img, orig_size, final_size = _load_image_capped(path)
            except Exception as e:
                logger.warning("Failed to open %s (%s); skipping", path, e)
                continue
            if orig_size != final_size:
                print(f"  {Path(str(path)).name}: {orig_size[0]}x{orig_size[1]} "
                      f"-> {final_size[0]}x{final_size[1]}")
            else:
                print(f"  {Path(str(path)).name}: {orig_size[0]}x{orig_size[1]}")
            images.append(img)
            valid_idx.append(i)

        if not images:
            pbar.update(len(batch_idx))
            continue

        try:
            structured_out = run_prompt_batch(
                processor, model, images, PROMPT_STRUCTURED, max_new_tokens=120,
            )
        except Exception as e:
            logger.warning("Structured prompt failed on batch %d (%s); skipping", b, e)
            pbar.update(len(batch_idx))
            continue

        try:
            description_out = run_prompt_batch(
                processor, model, images, PROMPT_DESCRIPTION, max_new_tokens=120,
            )
        except Exception as e:
            logger.warning("Description prompt failed on batch %d (%s); skipping", b, e)
            description_out = [""] * len(images)

        try:
            aesthetic_out = run_prompt_batch(
                processor, model, images, PROMPT_AESTHETIC, max_new_tokens=8,
            )
        except Exception as e:
            logger.warning("Aesthetic prompt failed on batch %d (%s); skipping", b, e)
            aesthetic_out = [""] * len(images)

        for row_i, struct_text, desc_text, aes_text in zip(
            valid_idx, structured_out, description_out, aesthetic_out
        ):
            parsed = parse_structured_output(struct_text)
            if all(v is None for v in parsed.values()):
                logger.warning("Empty/unparseable structured output for %s: %r",
                               df.at[row_i, "kept_path"], struct_text[:200])
            df.at[row_i, "caption_setting"] = parsed["setting"]
            df.at[row_i, "caption_activity"] = parsed["activity"]
            df.at[row_i, "caption_people"] = parsed["people"]
            df.at[row_i, "caption_mood"] = parsed["mood"]
            df.at[row_i, "caption_framing"] = parsed["framing"]
            aes_score = parse_aesthetic_score(aes_text or "")
            if aes_score is None:
                logger.warning("No digit in aesthetic output for %s: %r",
                               df.at[row_i, "kept_path"], (aes_text or "")[:200])
            df.at[row_i, "caption_aesthetic_score"] = (
                pd.NA if aes_score is None else int(aes_score)
            )
            desc_clean = (desc_text or "").strip()
            df.at[row_i, "caption_description"] = desc_clean if desc_clean else None
            df.at[row_i, "caption_model"] = MODEL_SHORT_NAME
            df.at[row_i, "caption_timestamp"] = timestamp

        for img in images:
            try:
                img.close()
            except Exception:
                pass
        pbar.update(len(batch_idx))

    pbar.close()
    elapsed = time.time() - t0

    write_parquet_atomic(df, results_path)
    print(f"Wrote {results_path}")

    print_summary(df, idx_list, elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Caption keeper photos with SmolVLM2 and write columns back to results.parquet.",
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Run YAML config (sets output_dir -> results.parquet).")
    parser.add_argument("--min-quality", choices=["good", "okay"], default="good",
                        help="Minimum pred_label to caption. 'good' = only good; "
                             "'okay' = good + okay.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Images per batched forward pass. Default 4 (benchmark "
                             "winner on RTX 2060/8GB; bs=8 thrashes VRAM, bs=2 slower).")
    parser.add_argument("--device", default="auto",
                        help="cuda, cpu, or auto.")
    parser.add_argument("--force", action="store_true",
                        help="Re-caption rows that already have captions.")
    parser.add_argument("--compile", dest="compile_model", action="store_true",
                        help=f"Compile model with torch.compile (reduce-overhead). "
                             f"Only applied when image count >= {COMPILE_MIN_IMAGES}; "
                             f"warmup cost otherwise dominates. Benchmark on RTX 2060 "
                             f"showed no speedup, but the flag is available.")
    parser.add_argument("--max-images", type=int, default=None,
                        help=argparse.SUPPRESS)
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
    caption_run(
        cfg,
        min_quality=args.min_quality,
        batch_size=args.batch_size,
        device=args.device,
        force=args.force,
        max_images=args.max_images,
        compile_model=args.compile_model,
    )


if __name__ == "__main__":
    main()
