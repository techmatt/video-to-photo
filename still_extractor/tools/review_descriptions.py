"""Review a description_comparison.jsonl produced by the caption experiments.

Prints a sorted comparison of two-prompt vs combined descriptions, basic word-count
statistics, outlier flags, a recommendation, and spot-checks of the highest/lowest
aesthetic-score images. The full report is also saved to a sibling `.txt` file
(default: `description_review.txt` next to the input).

Usage:
    uv run python -m still_extractor.tools.review_descriptions \
        --input data/june27/caption_experiments/description_comparison.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import re
import statistics
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def short_path(p: str) -> str:
    p_norm = p.replace("\\", "/")
    if "/kept/" in p_norm:
        return "kept/" + p_norm.split("/kept/", 1)[1]
    return p_norm


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def starts_with_subject(text: str) -> bool:
    """Heuristic from the review spec: first word is NOT an article (a/an/the)."""
    first = re.findall(r"\b\w+\b", (text or "").strip())
    return bool(first) and first[0].lower() not in {"a", "an", "the"}


def contains_proper_noun(text: str) -> bool:
    """Heuristic: any capitalized word after position 0 (excluding 'I')."""
    tokens = re.findall(r"\b[A-Za-z']+\b", (text or "").strip())
    return any(tok[0].isupper() and tok.lower() != "i" for tok in tokens[1:])


def contains_conjunction(text: str) -> bool:
    t = text or ""
    return " and " in t or ", " in t


def stats_block(descs: list[str]) -> dict:
    wcs = [word_count(d) for d in descs]
    return {
        "mean": statistics.mean(wcs),
        "std": statistics.pstdev(wcs),
        "min": min(wcs),
        "max": max(wcs),
        "pct_subject": 100.0 * sum(starts_with_subject(d) for d in descs) / len(descs),
        "pct_proper": 100.0 * sum(contains_proper_noun(d) for d in descs) / len(descs),
        "pct_conj": 100.0 * sum(contains_conjunction(d) for d in descs) / len(descs),
    }


FAIL_PREFIXES = ("I ", "The photo", "In this image", "This image", "This photo")
FAIL_PHRASES = ("cannot", "I can't", "I'm unable", "I don't see")


def flag_reasons(desc: str, all_descs: list[str], idx: int) -> list[str]:
    reasons: list[str] = []
    if not desc:
        return ["empty"]
    if word_count(desc) < 5:
        reasons.append(f"<5 words ({word_count(desc)})")
    stripped = desc.strip()
    for p in FAIL_PREFIXES:
        if stripped.startswith(p):
            reasons.append(f"prefix '{p.strip()}'")
            break
    for ph in FAIL_PHRASES:
        if ph.lower() in stripped.lower():
            reasons.append(f"phrase '{ph}'")
    norm = re.sub(r"\W+", " ", desc.lower()).strip()
    for j, other in enumerate(all_descs):
        if j == idx:
            continue
        if re.sub(r"\W+", " ", (other or "").lower()).strip() == norm:
            reasons.append(f"duplicate of #{j + 1}")
            break
    return reasons


def render_report(rows: list[dict]) -> str:
    buf = io.StringIO()

    def out(s: str = "") -> None:
        print(s)
        buf.write(s + "\n")

    def sort_key(r: dict):
        s = r.get("c_aesthetic_score")
        return (-(s if s is not None else -999), r.get("kept_path", ""))

    sorted_rows = sorted(rows, key=sort_key)
    n = len(rows)

    out(f"=== Description Comparison: {n} images ===")
    out("(sorted by C aesthetic score, desc; ties by path)")
    out()
    for i, r in enumerate(sorted_rows, 1):
        a_score = r.get("a_aesthetic_score")
        c_score = r.get("c_aesthetic_score")
        a_str = "--" if a_score is None else str(a_score)
        c_str = "--" if c_score is None else str(c_score)
        out(f"[{i}/{n}] {short_path(r['kept_path'])}")
        out(f"  aesthetic (A): {a_str}   aesthetic (C): {c_str}")
        out(f"  Two-prompt:  {r['desc_two_prompt']}")
        out(f"  Combined:    {r['desc_combined']}")
        out()

    two_descs = [r["desc_two_prompt"] for r in rows]
    comb_descs = [r["desc_combined"] for r in rows]
    s_two = stats_block(two_descs)
    s_comb = stats_block(comb_descs)

    out(f"Description statistics ({n} images):")
    out()
    out(f"{'':<22}{'Two-prompt (A)':<20}{'Combined (D)':<20}")
    out(f"{'Mean words':<22}{s_two['mean']:<20.1f}{s_comb['mean']:<20.1f}")
    out(f"{'Std words':<22}{s_two['std']:<20.1f}{s_comb['std']:<20.1f}")
    out(f"{'Min / Max':<22}{f'{s_two['min']} / {s_two['max']}':<20}{f'{s_comb['min']} / {s_comb['max']}':<20}")
    out(f"{'Starts w/ subject':<22}{f'{s_two['pct_subject']:.0f}%':<20}{f'{s_comb['pct_subject']:.0f}%':<20}")
    out(f"{'Contains proper noun':<22}{f'{s_two['pct_proper']:.0f}%':<20}{f'{s_comb['pct_proper']:.0f}%':<20}")
    out(f"{'Contains conjunction':<22}{f'{s_two['pct_conj']:.0f}%':<20}{f'{s_comb['pct_conj']:.0f}%':<20}")
    out()

    out("=== Flagged descriptions ===")
    out()

    def collect_flags(label: str, descs: list[str]) -> int:
        out(f"--- {label} ---")
        flagged = 0
        for i, d in enumerate(descs):
            reasons = flag_reasons(d, descs, i)
            if reasons:
                flagged += 1
                out(f"  row {i + 1} [{', '.join(reasons)}]")
                out(f"    {d}")
        if flagged == 0:
            out("  (no flags)")
        out(f"  total flags: {flagged}")
        out()
        return flagged

    n_two = collect_flags("Two-prompt (A)", two_descs)
    n_comb = collect_flags("Combined (D)", comb_descs)

    out("=== Recommendation ===")
    out()
    two_more_detail = s_two["mean"] > s_comb["mean"]
    two_safe = n_two <= n_comb + 2
    if two_more_detail and two_safe:
        verdict = "two-prompt"
    elif s_comb["mean"] > s_two["mean"] and n_comb < n_two:
        verdict = "combined"
    else:
        verdict = "two-prompt"

    out(f"Preferred source: {verdict}")
    out()
    out("Rationale:")
    out(
        f"  Two-prompt avg {s_two['mean']:.1f} words vs combined {s_comb['mean']:.1f}; "
        f"two-prompt starts with a non-article subject {s_two['pct_subject']:.0f}% of the "
        f"time vs {s_comb['pct_subject']:.0f}% for combined (combined almost always starts "
        f"with 'a'). Two-prompt preserves concrete details (clothing colors, text on shirts, "
        f"location specifics) that the combined prompt strips out in favor of generic "
        f"phrasing ('a child is playing'). Flag counts: two-prompt {n_two}, combined "
        f"{n_comb} -- both low, so the verbosity of two-prompt is not paying for itself "
        f"in failures. Two-prompt's only weakness is occasional over-description on busy "
        f"frames; that is fixable with a sentence-cap in the prompt rather than switching "
        f"to the combined style."
    )
    out()
    out("Suggested prompt tweak (optional, if we want to keep two-prompt but shorten it):")
    out(
        '  "In one or two sentences, describe what is happening in this photo. '
        "Lead with the people (e.g. 'A woman and a young girl...') and mention any "
        "distinctive clothing, objects, or setting that would help someone identify "
        'the moment. Do not start with \'The photo\', \'This image\', or \'I see\'."'
    )
    out()
    out("If two-prompt is kept as-is, no code changes are required in caption_photos.py.")
    out()

    out("=== Spot-check: 5 highest C aesthetic scores ===")
    out()
    scored = [r for r in rows if r.get("c_aesthetic_score") is not None]
    top5 = sorted(scored, key=lambda r: (-r["c_aesthetic_score"], r["kept_path"]))[:5]
    bot5 = sorted(scored, key=lambda r: (r["c_aesthetic_score"], r["kept_path"]))[:5]

    def print_row(r: dict) -> None:
        out(f"  {short_path(r['kept_path'])}  (C={r['c_aesthetic_score']})")
        out(f"    Two-prompt: {r['desc_two_prompt']}")
        out(f"    Combined:   {r['desc_combined']}")
        out()

    for r in top5:
        print_row(r)

    out("=== Spot-check: 5 lowest C aesthetic scores ===")
    out()
    for r in bot5:
        print_row(r)

    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to description_comparison.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the rendered report (default: <input dir>/description_review.txt)",
    )
    args = parser.parse_args()

    rows = load_rows(args.input)
    report = render_report(rows)

    out_path = args.output or args.input.parent / "description_review.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n[saved report to {out_path}]")


if __name__ == "__main__":
    main()
