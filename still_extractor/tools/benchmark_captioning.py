"""Benchmark SmolVLM2 captioning throughput.

Sweeps batch_size x torch.compile over a fixed 20-image sample drawn from
``results.parquet``, then runs a prompt-batched variant at the largest non-OOM
batch size. Reports wall-clock, images/second, and peak VRAM per configuration
and prints a recommended config (best img/s under a VRAM cap).

Usage:
    uv run python -m still_extractor.tools.benchmark_captioning \
        --config configs/JuliaEllieMay2026.yaml
"""

from __future__ import annotations

import argparse
import gc
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from still_extractor.caption_photos import (
    PROMPT_AESTHETIC,
    PROMPT_DESCRIPTION,
    PROMPT_STRUCTURED,
    _load_image_capped,
    load_model,
    run_prompt_batch,
)
from still_extractor.inventory import RunConfig

logger = logging.getLogger(__name__)

# (prompt_text, max_new_tokens) tuples in the same order the production pipeline runs them.
PROMPTS: list[tuple[str, int]] = [
    (PROMPT_STRUCTURED, 120),
    (PROMPT_DESCRIPTION, 120),
    (PROMPT_AESTHETIC, 8),
]

SAMPLE_SIZE = 20
SAMPLE_SEED = 42


@dataclass
class Result:
    label: str
    batch_size: int
    compile_enabled: bool
    img_per_s: float | None
    peak_vram_mb: float | None
    wall_s: float | None
    notes: str

    def as_row(self) -> str:
        if self.img_per_s is None:
            ips = "  -   "
            vram = "   -  "
        else:
            ips = f"{self.img_per_s:.3f}"
            vram = f"{int(round(self.peak_vram_mb))}"
        return (
            f"{self.label:<22} | {self.batch_size:<10} | "
            f"{('True' if self.compile_enabled else 'False'):<7} | "
            f"{ips:<6} | {vram:<12} | {self.notes}"
        )


def select_sample(cfg: RunConfig, n: int, seed: int) -> list[Path]:
    parquet = cfg.output_dir / "results.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"results.parquet not found at {parquet}")
    df = pd.read_parquet(parquet)
    mask = df["kept_path"].notna() & df["pred_label"].isin({"good", "okay"})
    eligible = df.loc[mask, "kept_path"].tolist()
    rng = random.Random(seed)
    rng.shuffle(eligible)
    picked: list[Path] = []
    for p in eligible:
        pp = Path(str(p))
        if pp.exists():
            picked.append(pp)
        if len(picked) >= n:
            break
    if len(picked) < n:
        raise RuntimeError(
            f"Only found {len(picked)} eligible images on disk (needed {n}). "
            f"Searched {len(eligible)} parquet rows."
        )
    return picked


def load_all_images(paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for p in paths:
        img, _, _ = _load_image_capped(str(p))
        images.append(img)
    return images


def _is_oom(exc: BaseException) -> bool:
    if torch.cuda.is_available() and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def _reset_peak_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_image_batched(processor, model, images: list[Image.Image], batch_size: int) -> None:
    """Original loop: per batch of images, run all 3 prompts."""
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        for prompt, max_new in PROMPTS:
            run_prompt_batch(processor, model, chunk, prompt, max_new)


def run_prompt_batched(processor, model, images: list[Image.Image], batch_size: int) -> None:
    """Prompt-major loop: for each prompt, walk all images in batches of batch_size."""
    for prompt, max_new in PROMPTS:
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            run_prompt_batch(processor, model, chunk, prompt, max_new)


def _maybe_compile(model, enabled: bool):
    if not enabled:
        return model
    return torch.compile(model, mode="reduce-overhead")


def _warmup(processor, model, images: list[Image.Image], batch_size: int) -> float:
    """Run one full 3-prompt pass on a single batch to absorb compile cost.
    Returns elapsed wall-clock seconds for the warmup."""
    warm = images[:batch_size]
    _sync()
    t0 = time.time()
    for prompt, max_new in PROMPTS:
        run_prompt_batch(processor, model, warm, prompt, max_new)
    _sync()
    return time.time() - t0


def _benchmark_image_batched(
    processor,
    base_model,
    images: list[Image.Image],
    batch_size: int,
    compile_enabled: bool,
) -> Result:
    label = "image-batched"
    _reset_peak_mem()
    notes_parts: list[str] = []
    try:
        model = _maybe_compile(base_model, compile_enabled)
        if compile_enabled:
            warm_s = _warmup(processor, model, images, batch_size)
            notes_parts.append(f"+warmup {warm_s:.0f}s")
            # Reset peak after warmup so reported VRAM reflects the steady-state pass.
            _reset_peak_mem()
        _sync()
        t0 = time.time()
        run_image_batched(processor, model, images, batch_size)
        _sync()
        elapsed = time.time() - t0
        return Result(
            label=label,
            batch_size=batch_size,
            compile_enabled=compile_enabled,
            img_per_s=len(images) / elapsed,
            peak_vram_mb=_peak_mb(),
            wall_s=elapsed,
            notes=" ".join(notes_parts),
        )
    except Exception as exc:  # noqa: BLE001
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if _is_oom(exc):
            return Result(label, batch_size, compile_enabled, None, None, None, "OOM")
        return Result(
            label, batch_size, compile_enabled, None, None, None,
            f"FAIL: {type(exc).__name__}: {str(exc)[:80]}",
        )


def _benchmark_prompt_batched(
    processor,
    base_model,
    images: list[Image.Image],
    batch_size: int,
) -> Result:
    _reset_peak_mem()
    try:
        _sync()
        t0 = time.time()
        run_prompt_batched(processor, base_model, images, batch_size)
        _sync()
        elapsed = time.time() - t0
        return Result(
            label="prompt-batched",
            batch_size=batch_size,
            compile_enabled=False,
            img_per_s=len(images) / elapsed,
            peak_vram_mb=_peak_mb(),
            wall_s=elapsed,
            notes=f"bs={batch_size}",
        )
    except Exception as exc:  # noqa: BLE001
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if _is_oom(exc):
            return Result("prompt-batched", batch_size, False, None, None, None, "OOM")
        return Result(
            "prompt-batched", batch_size, False, None, None, None,
            f"FAIL: {type(exc).__name__}: {str(exc)[:80]}",
        )


def print_table(results: list[Result]) -> None:
    header = (
        f"{'variant':<22} | {'batch_size':<10} | {'compile':<7} | "
        f"{'img/s':<6} | {'peak_VRAM_MB':<12} | notes"
    )
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)
    for r in results:
        print(r.as_row())
    print()


def pick_recommended(results: list[Result], vram_cap_mb: float) -> Result | None:
    candidates = [
        r for r in results
        if r.img_per_s is not None
        and r.peak_vram_mb is not None
        and r.peak_vram_mb <= vram_cap_mb
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.img_per_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SmolVLM2 captioning throughput.")
    parser.add_argument("--config", type=Path, required=True,
                        help="Run YAML config (used to locate results.parquet).")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--vram-cap-mb", type=float, default=7500.0,
                        help="Recommend the best config that peaks under this VRAM (MB).")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--full-run-images", type=int, default=200,
                        help="Image count to project total runtime for at the end.")
    parser.add_argument("--skip-compile", action="store_true",
                        help="Skip the torch.compile half of the sweep.")
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

    compile_axis = [False] if args.skip_compile else [False, True]

    results: list[Result] = []
    for bs in sorted(args.batch_sizes):
        for compile_enabled in compile_axis:
            tag = f"bs={bs} compile={compile_enabled}"
            print(f"\n--- Benchmarking image-batched {tag} ---")
            r = _benchmark_image_batched(processor, model, images, bs, compile_enabled)
            results.append(r)
            print(r.as_row())
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Prompt-batched variant: pick the largest non-OOM batch size from the
    # non-compile sweep (compile orthogonal to access-pattern question).
    non_compile_ok = [
        r for r in results
        if not r.compile_enabled and r.img_per_s is not None
    ]
    prompt_result: Result | None = None
    if non_compile_ok:
        best_bs = max(r.batch_size for r in non_compile_ok)
        print(f"\n--- Benchmarking prompt-batched bs={best_bs} ---")
        prompt_result = _benchmark_prompt_batched(processor, model, images, best_bs)
        print(prompt_result.as_row())
        results.append(prompt_result)

    print_table(results)

    recommended = pick_recommended(results, args.vram_cap_mb)
    if recommended is None:
        print(f"No configuration stayed under VRAM cap {args.vram_cap_mb:.0f} MB.")
        return

    is_prompt_batched = recommended.label == "prompt-batched"
    print(
        f"Recommended: variant={recommended.label}, batch_size={recommended.batch_size}, "
        f"compile={recommended.compile_enabled}, "
        f"img/s={recommended.img_per_s:.3f}, "
        f"peak_VRAM={int(round(recommended.peak_vram_mb))} MB"
    )
    est_s = args.full_run_images / recommended.img_per_s
    m, s = divmod(int(est_s), 60)
    print(f"Estimated wall-clock for {args.full_run_images}-image full run "
          f"at this config: ~{m}m {s:02d}s")

    # Also surface a useful comparison: img-batched best vs prompt-batched, if applicable.
    img_best = max(
        (r for r in results if r.label == "image-batched" and r.img_per_s is not None),
        key=lambda r: r.img_per_s,
        default=None,
    )
    if img_best and prompt_result and prompt_result.img_per_s is not None:
        delta = (prompt_result.img_per_s - img_best.img_per_s) / img_best.img_per_s * 100.0
        print(f"prompt-batched vs best image-batched: {delta:+.1f}% img/s")


if __name__ == "__main__":
    main()
