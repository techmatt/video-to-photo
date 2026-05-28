"""Caption prompt structure A/B/baseline experiment.

Compares three prompt configurations on a fixed 20-image sample from a run's
``results.parquet``:

  - current   : existing 3 prompts (structured w/ 5 fields, description, aesthetic).
  - variant_B : safe 3-prompt baseline. Structured trimmed to setting+activity;
                description and aesthetic prompts unchanged.
  - variant_A : experimental 2-prompt. setting+activity+description in one
                combined call; aesthetic unchanged.

Reuses ``caption_photos.run_prompt_batch`` and the sample-selection logic from
``benchmark_captioning`` (seed=42, filtered to good/okay pred_label). Runs at
bs=4, compile=False (benchmark winner).

Prints a timing comparison table, a per-image side-by-side of setting / activity
/ description / aesthetic across all three variants, and saves raw + parsed
outputs to ``<output_dir>/caption_prompt_experiment.json`` for later inspection.

Usage:
    uv run python -m still_extractor.tools.experiment_caption_prompts \
        --config configs/JuliaEllieMay2026.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from still_extractor.caption_photos import (
    PROMPT_AESTHETIC,
    PROMPT_DESCRIPTION,
    PROMPT_STRUCTURED,
    load_model,
    parse_aesthetic_score,
    parse_structured_output,
    run_prompt_batch,
)
from still_extractor.inventory import RunConfig
from still_extractor.tools.benchmark_captioning import load_all_images, select_sample

logger = logging.getLogger(__name__)

SAMPLE_SIZE = 20
SAMPLE_SEED = 42
BATCH_SIZE = 4

# Token budgets per prompt. Matches caption_photos.py for shared prompts; the
# combined variant-A prompt gets a larger budget since it emits structured
# fields *and* a one-sentence description in a single completion.
MAX_NEW_STRUCTURED = 120
MAX_NEW_STRUCTURED_B = 120
MAX_NEW_DESCRIPTION = 120
MAX_NEW_AESTHETIC = 8
MAX_NEW_COMBINED_A = 240

# Variant B: same wording as PROMPT_STRUCTURED but with people / mood / framing
# field lines removed. Keep the header and footer verbatim.
PROMPT_STRUCTURED_B = (
    "Describe this photo using exactly these fields and no others. "
    "Be brief - one phrase or word per field.\n"
    "\n"
    "setting: [where the photo was taken, e.g. playground, kitchen, beach]\n"
    "activity: [what is happening, e.g. eating, playing, laughing]\n"
    "\n"
    "Respond with only the fields above, nothing else."
)

# Variant A: one call that emits setting + activity (key/value form, as in the
# current structured prompt) plus a one-sentence description. The description
# guidance is the existing PROMPT_DESCRIPTION text embedded as the field
# description.
PROMPT_COMBINED_A = (
    "Describe this photo using exactly these fields and no others. "
    "Use one phrase or word for setting and activity, and one sentence for description.\n"
    "\n"
    "setting: [where the photo was taken, e.g. playground, kitchen, beach]\n"
    "activity: [what is happening, e.g. eating, playing, laughing]\n"
    "description: [one sentence about the people, what they are doing, and the "
    "setting. Be specific and vivid but concise. Do not start with "
    '"The photo shows" or "In this image"]\n'
    "\n"
    "Respond with only the fields above, nothing else."
)

# Recognized keys in the combined variant-A output, used to delimit the
# multi-line description value.
COMBINED_A_KEYS = ("setting", "activity", "description")
_KEY_LINE_RE = re.compile(
    r"^\s*[-*\s]*(" + "|".join(COMBINED_A_KEYS) + r")\s*:",
    re.IGNORECASE,
)


def parse_combined_a(text: str) -> dict[str, str | None]:
    """Parse variant-A combined output into setting / activity / description.

    Robust to extra prose and to descriptions that wrap onto multiple lines:
    the value for each key runs until the next recognized key line or end of
    string. Returns a dict with all three keys (str | None).
    """
    out: dict[str, str | None] = {k: None for k in COMBINED_A_KEYS}
    lines = text.splitlines()

    # First pass: identify which lines start a new key.
    key_starts: list[tuple[int, str]] = []
    for i, raw in enumerate(lines):
        m = _KEY_LINE_RE.match(raw)
        if m:
            key_starts.append((i, m.group(1).lower()))

    for idx, (line_no, key) in enumerate(key_starts):
        first_line = lines[line_no]
        _, _, after = first_line.partition(":")
        end_line = key_starts[idx + 1][0] if idx + 1 < len(key_starts) else len(lines)
        chunk_parts = [after]
        for j in range(line_no + 1, end_line):
            chunk_parts.append(lines[j])
        value = " ".join(p.strip() for p in chunk_parts if p.strip())
        value = value.strip().strip("[]").strip()
        if value:
            out[key] = value
    return out


@dataclass
class VariantResult:
    name: str
    n_prompts: int
    wall_s: float = 0.0
    # parsed[i] is the per-image output for the i-th sample image.
    parsed: list[dict] = field(default_factory=list)
    raw: list[dict] = field(default_factory=list)
    malformed_count: int = 0

    @property
    def img_per_s(self) -> float:
        if self.wall_s <= 0:
            return 0.0
        return len(self.parsed) / self.wall_s


def _has_malformed_a(parsed: dict) -> bool:
    return (
        parsed.get("setting") is None
        or parsed.get("activity") is None
        or parsed.get("description") is None
        or parsed.get("aesthetic_score") is None
    )


def _has_malformed_three_prompt(parsed: dict) -> bool:
    return (
        parsed.get("setting") is None
        or parsed.get("activity") is None
        or parsed.get("description") is None
        or parsed.get("aesthetic_score") is None
    )


def run_variant_current(
    processor, model, images: list[Image.Image], batch_size: int,
) -> VariantResult:
    """Existing 3-prompt pipeline (structured 5-field + description + aesthetic)."""
    n = len(images)
    parsed_all: list[dict] = []
    raw_all: list[dict] = []
    t0 = time.time()
    for start in range(0, n, batch_size):
        chunk = images[start : start + batch_size]
        struct_out = run_prompt_batch(processor, model, chunk, PROMPT_STRUCTURED, MAX_NEW_STRUCTURED)
        desc_out = run_prompt_batch(processor, model, chunk, PROMPT_DESCRIPTION, MAX_NEW_DESCRIPTION)
        aes_out = run_prompt_batch(processor, model, chunk, PROMPT_AESTHETIC, MAX_NEW_AESTHETIC)
        for s_text, d_text, a_text in zip(struct_out, desc_out, aes_out):
            structured = parse_structured_output(s_text)
            description = (d_text or "").strip() or None
            aesthetic = parse_aesthetic_score(a_text or "")
            parsed_all.append({
                "setting": structured["setting"],
                "activity": structured["activity"],
                "people": structured["people"],
                "mood": structured["mood"],
                "framing": structured["framing"],
                "description": description,
                "aesthetic_score": aesthetic,
            })
            raw_all.append({
                "structured": s_text,
                "description": d_text,
                "aesthetic": a_text,
            })
    wall = time.time() - t0
    result = VariantResult(name="current", n_prompts=3, wall_s=wall, parsed=parsed_all, raw=raw_all)
    result.malformed_count = sum(1 for p in parsed_all if _has_malformed_three_prompt(p))
    return result


def run_variant_b(
    processor, model, images: list[Image.Image], batch_size: int,
) -> VariantResult:
    """3-prompt baseline with structured trimmed to setting + activity."""
    n = len(images)
    parsed_all: list[dict] = []
    raw_all: list[dict] = []
    t0 = time.time()
    for start in range(0, n, batch_size):
        chunk = images[start : start + batch_size]
        struct_out = run_prompt_batch(processor, model, chunk, PROMPT_STRUCTURED_B, MAX_NEW_STRUCTURED_B)
        desc_out = run_prompt_batch(processor, model, chunk, PROMPT_DESCRIPTION, MAX_NEW_DESCRIPTION)
        aes_out = run_prompt_batch(processor, model, chunk, PROMPT_AESTHETIC, MAX_NEW_AESTHETIC)
        for s_text, d_text, a_text in zip(struct_out, desc_out, aes_out):
            structured = parse_structured_output(s_text)
            description = (d_text or "").strip() or None
            aesthetic = parse_aesthetic_score(a_text or "")
            parsed_all.append({
                "setting": structured["setting"],
                "activity": structured["activity"],
                "description": description,
                "aesthetic_score": aesthetic,
            })
            raw_all.append({
                "structured_b": s_text,
                "description": d_text,
                "aesthetic": a_text,
            })
    wall = time.time() - t0
    result = VariantResult(name="variant_B", n_prompts=3, wall_s=wall, parsed=parsed_all, raw=raw_all)
    result.malformed_count = sum(1 for p in parsed_all if _has_malformed_three_prompt(p))
    return result


def run_variant_a(
    processor, model, images: list[Image.Image], batch_size: int,
) -> VariantResult:
    """2-prompt experimental: combined structured+description, then aesthetic."""
    n = len(images)
    parsed_all: list[dict] = []
    raw_all: list[dict] = []
    t0 = time.time()
    for start in range(0, n, batch_size):
        chunk = images[start : start + batch_size]
        combined_out = run_prompt_batch(processor, model, chunk, PROMPT_COMBINED_A, MAX_NEW_COMBINED_A)
        aes_out = run_prompt_batch(processor, model, chunk, PROMPT_AESTHETIC, MAX_NEW_AESTHETIC)
        for c_text, a_text in zip(combined_out, aes_out):
            combined = parse_combined_a(c_text)
            aesthetic = parse_aesthetic_score(a_text or "")
            parsed_all.append({
                "setting": combined["setting"],
                "activity": combined["activity"],
                "description": combined["description"],
                "aesthetic_score": aesthetic,
            })
            raw_all.append({
                "combined": c_text,
                "aesthetic": a_text,
            })
    wall = time.time() - t0
    result = VariantResult(name="variant_A", n_prompts=2, wall_s=wall, parsed=parsed_all, raw=raw_all)
    result.malformed_count = sum(1 for p in parsed_all if _has_malformed_a(p))
    return result


def print_timing_table(results: list[VariantResult]) -> None:
    baseline = next((r for r in results if r.name == "current"), None)
    base_rate = baseline.img_per_s if baseline else None

    print()
    print("variant    | prompts | img/s  | vs current")
    print("-----------|---------|--------|-----------")
    for r in results:
        rate = r.img_per_s
        if base_rate and r.name != "current" and base_rate > 0:
            delta = (rate - base_rate) / base_rate * 100.0
            vs = f"{delta:+.1f}%"
        elif r.name == "current":
            vs = "baseline"
        else:
            vs = "  -   "
        print(f"{r.name:<10} | {r.n_prompts:<7} | {rate:<6.3f} | {vs}")
    print()


def _trunc(s: object, width: int) -> str:
    if s is None:
        return "(null)"
    text = str(s).replace("\n", " ").replace("\r", " ").strip()
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def print_side_by_side(
    paths: list[Path],
    current: VariantResult,
    variant_a: VariantResult,
    variant_b: VariantResult,
) -> None:
    col_w = 50
    field_w = 13
    sep_line = "-" * field_w + "|" + ("-" * (col_w + 2) + "|") * 2 + "-" * (col_w + 1)
    header = (
        f"{'field':<{field_w}}| "
        f"{'current':<{col_w}}| "
        f"{'variant_B':<{col_w}}| "
        f"{'variant_A':<{col_w}}"
    )
    for i, path in enumerate(paths):
        cur = current.parsed[i] if i < len(current.parsed) else {}
        vb = variant_b.parsed[i] if i < len(variant_b.parsed) else {}
        va = variant_a.parsed[i] if i < len(variant_a.parsed) else {}
        print()
        print(f"=== Image: {path} ===")
        print(header)
        print(sep_line)
        for label, key in (
            ("setting", "setting"),
            ("activity", "activity"),
            ("description", "description"),
            ("aesthetic", "aesthetic_score"),
        ):
            print(
                f"{label:<{field_w}}| "
                f"{_trunc(cur.get(key), col_w):<{col_w}}| "
                f"{_trunc(vb.get(key), col_w):<{col_w}}| "
                f"{_trunc(va.get(key), col_w):<{col_w}}"
            )


def save_results(
    out_path: Path,
    paths: list[Path],
    current: VariantResult,
    variant_a: VariantResult,
    variant_b: VariantResult,
    batch_size: int,
) -> None:
    payload = {
        "metadata": {
            "sample_size": len(paths),
            "sample_seed": SAMPLE_SEED,
            "batch_size": batch_size,
            "compile": False,
        },
        "timings": {
            "current": {
                "n_prompts": current.n_prompts,
                "wall_s": current.wall_s,
                "img_per_s": current.img_per_s,
                "malformed_count": current.malformed_count,
            },
            "variant_A": {
                "n_prompts": variant_a.n_prompts,
                "wall_s": variant_a.wall_s,
                "img_per_s": variant_a.img_per_s,
                "malformed_count": variant_a.malformed_count,
            },
            "variant_B": {
                "n_prompts": variant_b.n_prompts,
                "wall_s": variant_b.wall_s,
                "img_per_s": variant_b.img_per_s,
                "malformed_count": variant_b.malformed_count,
            },
        },
        "per_image": [
            {
                "path": str(paths[i]),
                "current": {
                    "parsed": current.parsed[i],
                    "raw": current.raw[i],
                },
                "variant_A": {
                    "parsed": variant_a.parsed[i],
                    "raw": variant_a.raw[i],
                },
                "variant_B": {
                    "parsed": variant_b.parsed[i],
                    "raw": variant_b.raw[i],
                },
            }
            for i in range(len(paths))
        ],
        "prompts": {
            "current_structured": PROMPT_STRUCTURED,
            "variant_b_structured": PROMPT_STRUCTURED_B,
            "variant_a_combined": PROMPT_COMBINED_A,
            "description": PROMPT_DESCRIPTION,
            "aesthetic": PROMPT_AESTHETIC,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 3-prompt vs 2-prompt caption structures on a fixed 20-image sample.",
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Run YAML config (used to locate results.parquet and output dir).")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None,
                        help="Override path for JSON output. Default: "
                             "<output_dir>/caption_prompt_experiment.json.")
    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s", force=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    cfg = RunConfig.from_yaml(args.config)
    paths = select_sample(cfg, args.sample_size, args.seed)
    print(f"Selected {len(paths)} sample images from {cfg.output_dir}/results.parquet "
          f"(seed={args.seed})")
    images = load_all_images(paths)
    print(f"Loaded {len(images)} images into memory")

    processor, model = load_model(args.device)

    print(f"\n--- Running current (3 prompts, structured 5-field) ---")
    current = run_variant_current(processor, model, images, args.batch_size)
    print(f"current: {current.wall_s:.1f}s, {current.img_per_s:.3f} img/s, "
          f"malformed={current.malformed_count}/{len(images)}")

    print(f"\n--- Running variant_B (3 prompts, structured 2-field) ---")
    variant_b = run_variant_b(processor, model, images, args.batch_size)
    print(f"variant_B: {variant_b.wall_s:.1f}s, {variant_b.img_per_s:.3f} img/s, "
          f"malformed={variant_b.malformed_count}/{len(images)}")

    print(f"\n--- Running variant_A (2 prompts, combined setting+activity+description) ---")
    variant_a = run_variant_a(processor, model, images, args.batch_size)
    print(f"variant_A: {variant_a.wall_s:.1f}s, {variant_a.img_per_s:.3f} img/s, "
          f"malformed={variant_a.malformed_count}/{len(images)}")

    print_timing_table([current, variant_b, variant_a])
    print_side_by_side(paths, current, variant_a, variant_b)

    threshold = 2
    flags: list[str] = []
    for r in (current, variant_b, variant_a):
        if r.malformed_count > threshold:
            flags.append(f"  {r.name}: {r.malformed_count}/{len(images)} images had >=1 unparseable field")
    if flags:
        print("\n!!! MODE COLLAPSE / PARSE FAILURE FLAG (>2/20 images affected) !!!")
        for f in flags:
            print(f)
    else:
        print("\nNo variant exceeded the 2/20 malformed threshold.")

    out_path = args.output if args.output else cfg.output_dir / "caption_prompt_experiment.json"
    save_results(out_path, paths, current, variant_a, variant_b, args.batch_size)


if __name__ == "__main__":
    main()
