"""Overnight captioning experiments (A-E + cross-analysis + final parquet write).

Sequential, logs everything per-experiment. Imports model loading and inference
helpers from ``caption_photos`` directly; never shells out.

    uv run python -m still_extractor.caption_photos_overnight \
        --config configs/june27.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from still_extractor.caption_photos import (
    CAPTION_COLUMNS_INT,
    CAPTION_COLUMNS_STR,
    DIGIT_RE,
    MODEL_SHORT_NAME,
    PROMPT_DESCRIPTION,
    PROMPT_STRUCTURED,
    SCORE_FIELDS,
    TEXT_FIELDS,
    _load_image_capped,
    ensure_caption_columns,
    load_model,
    parse_structured_output,
    run_prompt_batch,
    write_parquet_atomic,
)
from still_extractor.inventory import RunConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts unique to the overnight experiments
# ---------------------------------------------------------------------------

PROMPT_B_SCORES_ONLY = (
    "Rate this photo on three dimensions. Respond with only these three lines, nothing else:\n"
    "\n"
    "lighting: [score 1-10, where 10 is ideal natural light, 1 is very dark or harsh]\n"
    "face_quality: [score 1-10, where 10 is sharp, well-lit, forward-facing face, "
    "1 is blurry or occluded]\n"
    "aesthetic: [score 1-10, where 10 is a beautiful picture-book quality photo, "
    "1 is poor quality]\n"
    "\n"
    "You must provide a number for every line. If uncertain, give your best estimate."
)

PROMPT_C_AESTHETIC_ONLY = (
    "Rate the overall aesthetic quality of this photo for use in a family picture book.\n"
    "Consider: composition, lighting, sharpness, emotional warmth, and visual appeal.\n"
    "Respond with only a single integer from 1 to 10. Nothing else."
)

PROMPT_D_COMBINED = (
    "Describe this photo using exactly these fields. Be brief - one phrase or word per "
    "field unless otherwise noted.\n"
    "\n"
    "setting: [where the photo was taken]\n"
    "activity: [what is happening]\n"
    "people: [count and rough relationship, e.g. one child, child and adult, group]\n"
    "mood: [one word]\n"
    "framing: [close portrait, medium, or wide action]\n"
    "lighting: [score 1-10]\n"
    "face_quality: [score 1-10]\n"
    "aesthetic: [score 1-10]\n"
    "description: [one vivid sentence about the people, action, and setting]\n"
    "\n"
    "You must include a number for every score. If uncertain, estimate.\n"
    "Respond with only the fields above, nothing else."
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class _Tee:
    """File-like that fans writes out to several streams (best-effort)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


@contextmanager
def tee_stdout(log_path: Path):
    """Mirror stdout into ``log_path`` for the duration of the with-block.

    tqdm writes to stderr, so progress bars stay only in master.log; the
    per-experiment log file gets the structured prints (headers, tables).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("a", encoding="utf-8")
    f.write(f"\n# {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
    orig = sys.stdout
    sys.stdout = _Tee(orig, f)
    try:
        yield
    finally:
        sys.stdout = orig
        f.close()


def _format_mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def _parse_single_int(text: str) -> int | None:
    m = DIGIT_RE.search(text.strip())
    if not m:
        return None
    try:
        return max(1, min(10, int(m.group(0))))
    except ValueError:
        return None


def _parse_combined_d(text: str) -> tuple[dict, str | None]:
    """Parse Experiment D output. Returns (structured_fields_dict, description)."""
    parsed = parse_structured_output(text)
    desc: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        normalized = key.strip().lower().lstrip("-* ").rstrip()
        if normalized == "description":
            v = value.strip().strip("[]").strip()
            if v:
                desc = v
            break
    return parsed, desc


# ---------------------------------------------------------------------------
# Inference loop shared by every experiment
# ---------------------------------------------------------------------------

def _run_prompt_over_paths(
    processor,
    model,
    paths: list[str],
    images_by_path: dict[str, Image.Image],
    prompt: str,
    max_new_tokens: int,
    batch_size: int,
    desc: str,
) -> list[tuple[str, str]]:
    """Run a single prompt over every path. Returns list of (path, raw_output)."""
    out: list[tuple[str, str]] = []
    pbar = tqdm(total=len(paths), desc=desc, unit="img")
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        batch_imgs = [images_by_path[p] for p in batch_paths]
        try:
            outs = run_prompt_batch(processor, model, batch_imgs, prompt, max_new_tokens)
        except Exception as e:
            logger.exception("Batch starting at %d failed: %s", i, e)
            outs = [""] * len(batch_imgs)
        for p, o in zip(batch_paths, outs):
            out.append((p, (o or "")))
        pbar.update(len(batch_paths))
    pbar.close()
    return out


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _coverage_block(df: pd.DataFrame, label_col_pairs) -> str:
    n = max(len(df), 1)
    label_w = max(len(label) for label, _ in label_col_pairs)
    lines = []
    for label, col in label_col_pairs:
        present = int(df[col].notna().sum()) if col in df.columns else 0
        pct = 100.0 * present / n
        lines.append(f"  {label:<{label_w}} {present}/{len(df)} ({pct:.0f}%)")
    return "\n".join(lines)


def _score_stats(series: pd.Series) -> dict | None:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return None
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "median": int(vals.median()),
        "min": int(vals.min()),
        "max": int(vals.max()),
        "std": float(vals.std()) if len(vals) > 1 else 0.0,
    }


def _print_score_distributions(df: pd.DataFrame, label_col_pairs) -> None:
    print("\nScore distributions:")
    for label, col in label_col_pairs:
        st = _score_stats(df[col])
        if st is None:
            print(f"  {label}: (no values)")
        else:
            print(f"  {label}: mean={st['mean']:.1f}  median={st['median']}  "
                  f"min={st['min']}  max={st['max']}  (n={st['n']})")


def _hist_line(series: pd.Series, label: str) -> str:
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    bins = {i: 0 for i in range(1, 11)}
    for v in vals:
        v = max(1, min(10, int(v)))
        bins[v] += 1
    parts = [f"{i}[{'#' * bins[i]}]" for i in range(1, 11)]
    return f"{label}: " + " ".join(parts)


def _pearson(x, y) -> float | None:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return None
    a, b = a[mask], b[mask]
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _fmt_r(r: float | None) -> str:
    return "n/a" if r is None else f"{r:.2f}"


# ---------------------------------------------------------------------------
# Experiment A: two-prompt baseline
# ---------------------------------------------------------------------------

def run_experiment_A(processor, model, paths, images, expts_dir, batch_size):
    log_path = expts_dir / "run_A.log"
    parquet_path = expts_dir / "results_A.parquet"
    raw_path = expts_dir / "raw_outputs_A.jsonl"
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with tee_stdout(log_path):
        print(f"\n=== Experiment A: full two-prompt run ({len(paths)} images) ===")
        struct = _run_prompt_over_paths(
            processor, model, paths, images,
            PROMPT_STRUCTURED, 180, batch_size, "A structured",
        )
        desc = _run_prompt_over_paths(
            processor, model, paths, images,
            PROMPT_DESCRIPTION, 120, batch_size, "A description",
        )

        rows = []
        with raw_path.open("w", encoding="utf-8") as fh:
            for (p1, s_raw), (p2, d_raw) in zip(struct, desc):
                assert p1 == p2, f"path mismatch: {p1} vs {p2}"
                fh.write(json.dumps({
                    "kept_path": p1,
                    "prompt1_raw": s_raw,
                    "prompt2_raw": d_raw,
                }, ensure_ascii=False) + "\n")
                parsed = parse_structured_output(s_raw)
                row = {"kept_path": p1}
                for f in TEXT_FIELDS:
                    row[f"caption_{f}"] = parsed[f]
                for f in SCORE_FIELDS:
                    row[f"caption_{f}_score"] = parsed[f]
                clean = (d_raw or "").strip()
                row["caption_description"] = clean if clean else None
                row["caption_model"] = MODEL_SHORT_NAME
                row["caption_timestamp"] = timestamp
                rows.append(row)

        df_a = pd.DataFrame(rows)
        df_a.to_parquet(parquet_path, index=False)
        print(f"Wrote {parquet_path} ({len(df_a)} rows)")

        coverage_specs = [
            ("setting", "caption_setting"),
            ("activity", "caption_activity"),
            ("people", "caption_people"),
            ("mood", "caption_mood"),
            ("framing", "caption_framing"),
            ("lighting_score", "caption_lighting_score"),
            ("face_quality_score", "caption_face_quality_score"),
            ("aesthetic_score", "caption_aesthetic_score"),
            ("description", "caption_description"),
        ]
        print("\nStructured field coverage (non-null):")
        print(_coverage_block(df_a, coverage_specs))
        _print_score_distributions(df_a, [
            ("lighting_score", "caption_lighting_score"),
            ("face_quality_score", "caption_face_quality_score"),
            ("aesthetic_score", "caption_aesthetic_score"),
        ])
        sample = df_a.sample(n=min(10, len(df_a)), random_state=0) if len(df_a) else df_a
        print("\n10 random sample rows (filename | setting | mood | aesthetic | desc[:60]):")
        for _, row in sample.iterrows():
            kp = Path(str(row["kept_path"])).name
            d = (row["caption_description"] or "")[:60]
            print(f"  {kp} | {row['caption_setting']} | {row['caption_mood']} | "
                  f"{row['caption_aesthetic_score']} | {d}")
        print("=== Experiment A complete ===")


# ---------------------------------------------------------------------------
# Experiment B: scores-only prompt
# ---------------------------------------------------------------------------

def run_experiment_B(processor, model, paths, images, expts_dir, batch_size):
    log_path = expts_dir / "run_B.log"
    parquet_path = expts_dir / "results_B.parquet"
    raw_path = expts_dir / "raw_outputs_B.jsonl"
    with tee_stdout(log_path):
        print(f"\n=== Experiment B: scores-only prompt ({len(paths)} images) ===")
        outs = _run_prompt_over_paths(
            processor, model, paths, images,
            PROMPT_B_SCORES_ONLY, 80, batch_size, "B scores",
        )
        rows = []
        with raw_path.open("w", encoding="utf-8") as fh:
            for p, raw in outs:
                fh.write(json.dumps({"kept_path": p, "raw": raw}, ensure_ascii=False) + "\n")
                parsed = parse_structured_output(raw)
                rows.append({
                    "kept_path": p,
                    "b_lighting_score": parsed["lighting"],
                    "b_face_quality_score": parsed["face_quality"],
                    "b_aesthetic_score": parsed["aesthetic"],
                })

        df_b = pd.DataFrame(rows)
        df_b.to_parquet(parquet_path, index=False)
        print(f"Wrote {parquet_path} ({len(df_b)} rows)")

        print("\nCoverage:")
        print(_coverage_block(df_b, [
            ("lighting", "b_lighting_score"),
            ("face_quality", "b_face_quality_score"),
            ("aesthetic", "b_aesthetic_score"),
        ]))
        _print_score_distributions(df_b, [
            ("lighting", "b_lighting_score"),
            ("face_quality", "b_face_quality_score"),
            ("aesthetic", "b_aesthetic_score"),
        ])
        present = df_b[["b_lighting_score", "b_face_quality_score", "b_aesthetic_score"]].notna().sum(axis=1)
        print("\nScores per row:")
        for k in (3, 2, 1, 0):
            print(f"  {k} score(s): {int((present == k).sum())}")
        print("=== Experiment B complete ===")


# ---------------------------------------------------------------------------
# Experiment C: single aesthetic integer
# ---------------------------------------------------------------------------

def run_experiment_C(processor, model, paths, images, expts_dir, batch_size):
    log_path = expts_dir / "run_C.log"
    parquet_path = expts_dir / "results_C.parquet"
    raw_path = expts_dir / "raw_outputs_C.jsonl"
    with tee_stdout(log_path):
        print(f"\n=== Experiment C: aesthetic-only prompt ({len(paths)} images) ===")
        outs = _run_prompt_over_paths(
            processor, model, paths, images,
            PROMPT_C_AESTHETIC_ONLY, 20, batch_size, "C aesthetic",
        )
        rows = []
        with raw_path.open("w", encoding="utf-8") as fh:
            for p, raw in outs:
                fh.write(json.dumps({"kept_path": p, "raw": raw}, ensure_ascii=False) + "\n")
                rows.append({"kept_path": p, "c_aesthetic_score": _parse_single_int(raw)})

        df_c = pd.DataFrame(rows)
        df_c.to_parquet(parquet_path, index=False)
        print(f"Wrote {parquet_path} ({len(df_c)} rows)")

        valid = df_c["c_aesthetic_score"].notna()
        n = max(len(df_c), 1)
        print(f"\nCoverage: {int(valid.sum())}/{len(df_c)} "
              f"({100.0*valid.sum()/n:.0f}%) produced a valid integer")
        st = _score_stats(df_c["c_aesthetic_score"])
        if st is not None:
            print(f"Distribution: mean={st['mean']:.2f}  median={st['median']}  "
                  f"min={st['min']}  max={st['max']}  std={st['std']:.2f}  (n={st['n']})")
        present_df = df_c.dropna(subset=["c_aesthetic_score"])
        if len(present_df):
            top = present_df.sort_values(
                "c_aesthetic_score", ascending=False).head(10)
            bot = present_df.sort_values(
                "c_aesthetic_score", ascending=True).head(10)
            print("\nTop 10 by aesthetic:")
            for _, r in top.iterrows():
                print(f"  {int(r['c_aesthetic_score'])}  {Path(str(r['kept_path'])).name}")
            print("\nBottom 10 by aesthetic:")
            for _, r in bot.iterrows():
                print(f"  {int(r['c_aesthetic_score'])}  {Path(str(r['kept_path'])).name}")
        print("=== Experiment C complete ===")


# ---------------------------------------------------------------------------
# Experiment D: single combined prompt
# ---------------------------------------------------------------------------

def run_experiment_D(processor, model, paths, images, expts_dir, batch_size):
    log_path = expts_dir / "run_D.log"
    parquet_path = expts_dir / "results_D.parquet"
    raw_path = expts_dir / "raw_outputs_D.jsonl"
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with tee_stdout(log_path):
        print(f"\n=== Experiment D: single combined prompt ({len(paths)} images) ===")
        outs = _run_prompt_over_paths(
            processor, model, paths, images,
            PROMPT_D_COMBINED, 220, batch_size, "D combined",
        )
        rows = []
        with raw_path.open("w", encoding="utf-8") as fh:
            for p, raw in outs:
                fh.write(json.dumps({"kept_path": p, "raw": raw}, ensure_ascii=False) + "\n")
                parsed, desc = _parse_combined_d(raw)
                row = {"kept_path": p}
                for f in TEXT_FIELDS:
                    row[f"caption_{f}"] = parsed[f]
                for f in SCORE_FIELDS:
                    row[f"caption_{f}_score"] = parsed[f]
                row["caption_description"] = desc
                row["caption_model"] = MODEL_SHORT_NAME
                row["caption_timestamp"] = timestamp
                rows.append(row)

        df_d = pd.DataFrame(rows)
        df_d.to_parquet(parquet_path, index=False)
        print(f"Wrote {parquet_path} ({len(df_d)} rows)")

        print("\nCoverage:")
        print(_coverage_block(df_d, [
            ("setting", "caption_setting"),
            ("activity", "caption_activity"),
            ("people", "caption_people"),
            ("mood", "caption_mood"),
            ("framing", "caption_framing"),
            ("lighting_score", "caption_lighting_score"),
            ("face_quality_score", "caption_face_quality_score"),
            ("aesthetic_score", "caption_aesthetic_score"),
            ("description", "caption_description"),
        ]))
        _print_score_distributions(df_d, [
            ("lighting_score", "caption_lighting_score"),
            ("face_quality_score", "caption_face_quality_score"),
            ("aesthetic_score", "caption_aesthetic_score"),
        ])
        print("=== Experiment D complete ===")


# ---------------------------------------------------------------------------
# Experiment E: description comparison (no model inference)
# ---------------------------------------------------------------------------

def _coerce_int(v) -> int | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def run_experiment_E(paths, expts_dir):
    log_path = expts_dir / "run_E.log"
    out_path = expts_dir / "description_comparison.jsonl"
    with tee_stdout(log_path):
        print(f"\n=== Experiment E: description quality deep-dive ===")
        rng = random.Random(42)
        sample_n = min(30, len(paths))
        sample = rng.sample(paths, sample_n)

        a_path = expts_dir / "results_A.parquet"
        d_path = expts_dir / "results_D.parquet"
        c_path = expts_dir / "results_C.parquet"
        a_map = pd.read_parquet(a_path).set_index("kept_path") if a_path.exists() else None
        d_map = pd.read_parquet(d_path).set_index("kept_path") if d_path.exists() else None
        c_map = pd.read_parquet(c_path).set_index("kept_path") if c_path.exists() else None

        rows = []
        with out_path.open("w", encoding="utf-8") as fh:
            for p in sample:
                a_row = a_map.loc[p].to_dict() if a_map is not None and p in a_map.index else {}
                d_row = d_map.loc[p].to_dict() if d_map is not None and p in d_map.index else {}
                c_row = c_map.loc[p].to_dict() if c_map is not None and p in c_map.index else {}
                entry = {
                    "kept_path": p,
                    "desc_two_prompt": a_row.get("caption_description"),
                    "desc_combined": d_row.get("caption_description"),
                    "a_aesthetic_score": _coerce_int(a_row.get("caption_aesthetic_score")),
                    "c_aesthetic_score": _coerce_int(c_row.get("c_aesthetic_score")),
                }
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                rows.append(entry)

        print(f"Wrote {out_path} ({len(rows)} rows)")
        print()
        for e in rows:
            print(f"Image: {e['kept_path']}")
            print(f"  Two-prompt:  {e['desc_two_prompt']}")
            print(f"  Combined:    {e['desc_combined']}")
            print(f"  A aesthetic: {e['a_aesthetic_score']}   "
                  f"C aesthetic: {e['c_aesthetic_score']}")
            print("---")
        print("=== Experiment E complete ===")


# ---------------------------------------------------------------------------
# Cross-experiment analysis
# ---------------------------------------------------------------------------

def cross_experiment_analysis(expts_dir: Path, results_parquet_path: Path):
    log_path = expts_dir / "cross_analysis.log"
    with tee_stdout(log_path):
        print("\n=== Cross-experiment analysis ===")
        a_path = expts_dir / "results_A.parquet"
        b_path = expts_dir / "results_B.parquet"
        c_path = expts_dir / "results_C.parquet"
        if not (a_path.exists() and b_path.exists() and c_path.exists()):
            print("Missing one of A/B/C parquets; skipping cross-experiment analysis.")
            print(f"  A: {a_path.exists()}  B: {b_path.exists()}  C: {c_path.exists()}")
            return
        a = pd.read_parquet(a_path)
        b = pd.read_parquet(b_path)
        c = pd.read_parquet(c_path)

        merged = (
            a[["kept_path", "caption_aesthetic_score", "caption_lighting_score",
               "caption_face_quality_score", "caption_framing"]]
            .merge(b, on="kept_path", how="outer")
            .merge(c, on="kept_path", how="outer")
        )

        complete = merged.dropna(subset=[
            "caption_aesthetic_score", "b_aesthetic_score", "c_aesthetic_score",
        ])
        n = len(complete)

        print(f"\nScore correlations across prompt variants (N={n} images with all scores)\n")
        print("Aesthetic:")
        print(f"  A (two-prompt) vs B (scores-only):   r = "
              f"{_fmt_r(_pearson(complete['caption_aesthetic_score'], complete['b_aesthetic_score']))}")
        print(f"  A (two-prompt) vs C (solo):          r = "
              f"{_fmt_r(_pearson(complete['caption_aesthetic_score'], complete['c_aesthetic_score']))}")
        print(f"  B (scores-only) vs C (solo):         r = "
              f"{_fmt_r(_pearson(complete['b_aesthetic_score'], complete['c_aesthetic_score']))}")
        ab = merged.dropna(subset=["caption_lighting_score", "b_lighting_score"])
        afq = merged.dropna(subset=["caption_face_quality_score", "b_face_quality_score"])
        print(f"\nLighting (A vs B):                     r = "
              f"{_fmt_r(_pearson(ab['caption_lighting_score'], ab['b_lighting_score']))}  (n={len(ab)})")
        print(f"Face quality (A vs B):                 r = "
              f"{_fmt_r(_pearson(afq['caption_face_quality_score'], afq['b_face_quality_score']))}  (n={len(afq)})")

        total = max(len(merged), 1)

        def pct(col):
            return 100.0 * merged[col].notna().sum() / total

        print(f"\nCoverage (% images with valid score, N={len(merged)}):\n")
        print("              Exp A      Exp B      Exp C")
        print("              (2-prompt) (scores)   (aesthetic only)")
        print(f"lighting       {pct('caption_lighting_score'):>4.0f}%      "
              f"{pct('b_lighting_score'):>4.0f}%      n/a")
        print(f"face_quality   {pct('caption_face_quality_score'):>4.0f}%      "
              f"{pct('b_face_quality_score'):>4.0f}%      n/a")
        print(f"aesthetic      {pct('caption_aesthetic_score'):>4.0f}%      "
              f"{pct('b_aesthetic_score'):>4.0f}%      "
              f"{pct('c_aesthetic_score'):>4.0f}%")
        all3_a = int(merged[["caption_lighting_score", "caption_face_quality_score",
                             "caption_aesthetic_score"]].notna().all(axis=1).sum())
        all3_b = int(merged[["b_lighting_score", "b_face_quality_score",
                             "b_aesthetic_score"]].notna().all(axis=1).sum())
        print(f"all_3_scores   {100.0*all3_a/total:>4.0f}%      "
              f"{100.0*all3_b/total:>4.0f}%      n/a")

        print("\nAesthetic score distribution (binned 1-10):\n")
        print(_hist_line(merged["caption_aesthetic_score"], "Exp A"))
        print(_hist_line(merged["b_aesthetic_score"],       "Exp B"))
        print(_hist_line(merged["c_aesthetic_score"],       "Exp C"))
        print()
        for label, col in (
            ("Exp A", "caption_aesthetic_score"),
            ("Exp B", "b_aesthetic_score"),
            ("Exp C", "c_aesthetic_score"),
        ):
            st = _score_stats(merged[col])
            if st is not None:
                print(f"  {label} aesthetic: mean={st['mean']:.2f}  std={st['std']:.2f}  "
                      f"min={st['min']}  max={st['max']}  (n={st['n']})")

        print("\nCoverage by image characteristics (Experiment A):")
        try:
            results_df = pd.read_parquet(results_parquet_path)
        except Exception as e:
            print(f"  (could not load {results_parquet_path}: {e})")
            print("=== Cross-experiment analysis complete ===")
            return

        joined = a.merge(
            results_df[["kept_path", "face_count", "composite"]],
            on="kept_path", how="left",
        )
        score_cols = ["caption_lighting_score", "caption_face_quality_score", "caption_aesthetic_score"]
        joined["_all3"] = joined[score_cols].notna().all(axis=1)

        print("\n  By face presence:")
        joined["_has_face"] = joined["face_count"].fillna(0).astype(int) > 0
        for has_face, sub in joined.groupby("_has_face"):
            label = "has_face" if has_face else "no_face"
            n_s = max(len(sub), 1)
            print(f"    {label:<10} n={len(sub)}  all3={100.0*sub['_all3'].sum()/n_s:.0f}%  "
                  f"aesth_only={100.0*sub['caption_aesthetic_score'].notna().sum()/n_s:.0f}%")

        print("\n  By composite quartile:")
        try:
            joined["_quartile"] = pd.qcut(
                joined["composite"], q=4,
                labels=["Q1", "Q2", "Q3", "Q4"],
                duplicates="drop",
            )
        except Exception:
            joined["_quartile"] = None
        for q, sub in joined.groupby("_quartile", dropna=False):
            n_s = max(len(sub), 1)
            print(f"    {str(q):<10} n={len(sub)}  all3={100.0*sub['_all3'].sum()/n_s:.0f}%  "
                  f"aesth_only={100.0*sub['caption_aesthetic_score'].notna().sum()/n_s:.0f}%")

        print("\n  By caption_framing:")
        framing = joined["caption_framing"].fillna("(none)")
        for fr, sub in joined.groupby(framing):
            n_s = max(len(sub), 1)
            print(f"    {str(fr):<24} n={len(sub)}  all3={100.0*sub['_all3'].sum()/n_s:.0f}%  "
                  f"aesth_only={100.0*sub['caption_aesthetic_score'].notna().sum()/n_s:.0f}%")
        print("=== Cross-experiment analysis complete ===")


# ---------------------------------------------------------------------------
# Final step: write Experiment A back to results.parquet
# ---------------------------------------------------------------------------

def write_captions_to_results(results_parquet_path: Path, expts_dir: Path) -> None:
    a_path = expts_dir / "results_A.parquet"
    c_path = expts_dir / "results_C.parquet"
    if not a_path.exists():
        print(f"Skipping final write: {a_path} does not exist.")
        return

    print(f"\n=== Final step: writing captions back to {results_parquet_path} ===")
    df = pd.read_parquet(results_parquet_path)
    ensure_caption_columns(df)
    if "caption_aesthetic_score_solo" not in df.columns:
        df["caption_aesthetic_score_solo"] = pd.Series([pd.NA] * len(df), dtype="Int8")

    a = pd.read_parquet(a_path).set_index("kept_path")
    c = (pd.read_parquet(c_path).set_index("kept_path")["c_aesthetic_score"]
         if c_path.exists() else pd.Series(dtype="Int8"))

    n_written_a = 0
    n_written_c = 0
    for i, row in df.iterrows():
        kp = row.get("kept_path")
        if kp is None or (isinstance(kp, float) and pd.isna(kp)):
            continue
        if kp in a.index:
            ar = a.loc[kp]
            for col in CAPTION_COLUMNS_STR:
                if col not in ar:
                    continue
                v = ar[col]
                df.at[i, col] = None if (v is None or (isinstance(v, float) and pd.isna(v))) else v
            for col in CAPTION_COLUMNS_INT:
                if col not in ar:
                    continue
                v = ar[col]
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    df.at[i, col] = pd.NA
                else:
                    df.at[i, col] = int(v)
            n_written_a += 1
        if kp in c.index:
            v = c.loc[kp]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                df.at[i, "caption_aesthetic_score_solo"] = pd.NA
            else:
                df.at[i, "caption_aesthetic_score_solo"] = int(v)
                n_written_c += 1

    write_parquet_atomic(df, results_parquet_path)
    print(f"Written to {results_parquet_path}")
    cap_cols = sorted(c for c in df.columns if c.startswith("caption_"))
    print(f"  Caption columns: {', '.join(cap_cols)}")
    print(f"  Captioned rows (A): {n_written_a} / {len(df)} total")
    print(f"  Solo aesthetic rows (C): {n_written_c} / {len(df)} total")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight captioning experiments.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Limit number of images for smoke testing (skips final write if set).",
    )
    parser.add_argument(
        "--skip", default="",
        help="Comma-separated experiment letters to skip (A,B,C,D,E,ANALYSIS,FINAL).",
    )
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
    results_path = cfg.output_dir / "results.parquet"
    expts_dir = cfg.output_dir / "caption_experiments"
    expts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(results_path)
    filt = df[(df["pred_label"] == "good") & df["kept_path"].notna()]
    paths_all = filt["kept_path"].tolist()
    if args.max_images is not None:
        paths_all = paths_all[: args.max_images]
    n_imgs = len(paths_all)
    if n_imgs == 0:
        print("No images to process.")
        return

    s_per = 2.8
    plan_a = round(n_imgs * 2 * s_per / 60)
    plan_b = round(n_imgs * s_per / 60)
    plan_c = round(n_imgs * s_per / 60)
    plan_d = round(n_imgs * s_per / 60)
    plan_e = 3
    total = plan_a + plan_b + plan_c + plan_d + plan_e
    print("Experiment plan:")
    print(f"  A. Full run (two-prompt, fp16)          {n_imgs} images x 2 x {s_per}s  ~  {plan_a} min")
    print(f"  B. Scores-only prompt                   {n_imgs} images x 1 x {s_per}s  ~  {plan_b} min")
    print(f"  C. Aesthetic-only prompt                {n_imgs} images x 1 x {s_per}s  ~  {plan_c} min")
    print(f"  D. Single combined prompt               {n_imgs} images x 1 x {s_per}s  ~  {plan_d} min")
    print(f"  E. Description quality sample (manual)  30 images              ~   {plan_e} min")
    print(f"  Total estimated:                                              ~  {total} min")
    print("  (Running sequentially overnight - actual time may vary)")
    print()

    print(f"Loading {n_imgs} images (longest side capped at 1024)...")
    images: dict[str, Image.Image] = {}
    failed: list[str] = []
    for p in paths_all:
        try:
            img, _, _ = _load_image_capped(p)
            images[p] = img
        except Exception as e:
            failed.append(p)
            logger.warning("Failed to open %s: %s", p, e)
    paths = [p for p in paths_all if p in images]
    print(f"Loaded {len(paths)} of {n_imgs}; {len(failed)} failed.\n")

    processor, model = load_model(args.device)

    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    times: dict[str, float] = {}

    def run_safely(letter: str, fn, *a, **kw):
        if letter in skip:
            print(f"Skipping experiment {letter}.")
            return
        t0 = time.time()
        try:
            fn(*a, **kw)
        except Exception as e:
            logger.exception("Experiment %s failed: %s", letter, e)
        finally:
            times[letter] = time.time() - t0

    overall_start = time.time()
    run_safely("A", run_experiment_A, processor, model, paths, images, expts_dir, args.batch_size)
    run_safely("B", run_experiment_B, processor, model, paths, images, expts_dir, args.batch_size)
    run_safely("C", run_experiment_C, processor, model, paths, images, expts_dir, args.batch_size)
    run_safely("D", run_experiment_D, processor, model, paths, images, expts_dir, args.batch_size)
    run_safely("E", run_experiment_E, paths, expts_dir)

    if "ANALYSIS" not in skip:
        try:
            cross_experiment_analysis(expts_dir, results_path)
        except Exception as e:
            logger.exception("Cross-experiment analysis failed: %s", e)

    if "FINAL" in skip:
        print("Skipping final results.parquet write (--skip FINAL).")
    elif args.max_images is not None:
        print("Skipping final results.parquet write because --max-images was set "
              "(smoke-test mode).")
    else:
        try:
            write_captions_to_results(results_path, expts_dir)
        except Exception as e:
            logger.exception("Final write failed: %s", e)

    total_s = time.time() - overall_start
    print("\n=== Overnight run complete ===")
    unit = {"A": "captioned", "B": "scored", "C": "aesthetic",
            "D": "combined", "E": "compared"}
    denom_for = {"A": len(paths), "B": len(paths), "C": len(paths),
                 "D": len(paths), "E": min(30, len(paths))}
    for letter in "ABCDE":
        secs = times.get(letter, 0.0)
        denom = denom_for[letter]
        print(f"{letter}: {denom}/{denom} {unit[letter]:<9}  ({_format_mmss(secs)})")
    print(f"Total wall time: {_format_mmss(total_s)}")
    print(f"Results in {expts_dir}")


if __name__ == "__main__":
    main()
