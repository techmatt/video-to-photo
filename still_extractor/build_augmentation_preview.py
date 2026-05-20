"""Build a self-contained HTML preview of the augmentation stack.

Renders a 15x6 grid: column 1 is the original 128x128 face crop (with its
ground-truth label), columns 2-6 are independent random augmentations of the
same crop. Used to eyeball the training-time augmentation pipeline before
committing to a training run.
"""

import base64
import html
import io
import json
import logging
import random
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from PIL import Image

from still_extractor.build_index_html import _parse_kps, _safe_float
from still_extractor.face_crop import extract_face_crop

logger = logging.getLogger(__name__)


FACE_CROP_PADDING = 20
DISPLAY_SIZE = 128
N_AUG_COLS = 5
LABELS_IN_PRIORITY = ("good", "okay", "bad", "none")
LABEL_COLORS = {
    "none": "#8B0000",
    "bad":  "#FF1111",
    "okay": "#F59E0B",
    "good": "#22C55E",
}


def _build_augmenter():
    try:
        import torchvision.transforms as T  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "torchvision is required for build_augmentation_preview but is not installed."
        ) from e

    import torchvision.transforms as T

    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(
            degrees=15,
            interpolation=T.InterpolationMode.BICUBIC,
            fill=0,
        ),
        T.RandomResizedCrop(
            size=DISPLAY_SIZE,
            scale=(0.80, 1.00),
            ratio=(0.9, 1.1),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomPerspective(distortion_scale=0.15, p=0.3),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        T.RandomGrayscale(p=0.08),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
        T.ToTensor(),
        T.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.3, 3.0), value=0),
        T.ToPILImage(),
    ])


def _jpeg_recompress(img: Image.Image, rng: random.Random) -> Image.Image:
    quality = rng.randint(60, 95)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def _augment_once(base: Image.Image, augmenter, rng: random.Random) -> Image.Image:
    out = augmenter(base)
    if out.mode != "RGB":
        out = out.convert("RGB")
    return _jpeg_recompress(out, rng)


def _b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _join_labels(df: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    df = df.copy()

    def key_for(row: pd.Series) -> str | None:
        stem = row.get("video_stem")
        if not isinstance(stem, str) or not stem:
            return None
        path_col = (
            "refined_frame_path"
            if "refined_frame_path" in row and isinstance(row.get("refined_frame_path"), str)
            and row.get("refined_frame_path")
            else "frame_path"
        )
        raw = row.get(path_col)
        if not isinstance(raw, str) or not raw:
            return None
        return f"{stem}/{Path(raw).name}"

    df["_label_key"] = df.apply(key_for, axis=1)
    df["_label"] = df["_label_key"].map(labels)
    df = df[df["_label"].isin(LABELS_IN_PRIORITY)].reset_index(drop=True)
    return df


def _stratified_sample(df: pd.DataFrame, n_rows: int, rng: random.Random) -> pd.DataFrame:
    present = [lbl for lbl in LABELS_IN_PRIORITY if (df["_label"] == lbl).any()]
    if not present:
        return df.sample(n=min(n_rows, len(df)), random_state=rng.randint(0, 2**31 - 1))

    per_stratum = n_rows // len(present)
    remainder = n_rows - per_stratum * len(present)
    picks: list[pd.DataFrame] = []
    for lbl in present:
        sub = df[df["_label"] == lbl]
        take = min(per_stratum, len(sub))
        if take > 0:
            picks.append(sub.sample(n=take, random_state=rng.randint(0, 2**31 - 1)))

    chosen = pd.concat(picks) if picks else df.iloc[0:0]
    if remainder > 0 or len(chosen) < n_rows:
        leftover = df.drop(chosen.index)
        need = n_rows - len(chosen)
        if need > 0 and len(leftover) > 0:
            extra = leftover.sample(
                n=min(need, len(leftover)),
                random_state=rng.randint(0, 2**31 - 1),
            )
            chosen = pd.concat([chosen, extra])

    if "composite" in chosen.columns:
        chosen = chosen.sort_values("composite", ascending=False)
    return chosen.reset_index(drop=True)


def _resolve_image_path(raw) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    p = Path(raw)
    return p if p.exists() else None


def _crop_for_row(row: pd.Series) -> Image.Image | None:
    img_path = _resolve_image_path(row.get("refined_frame_path")) or _resolve_image_path(
        row.get("frame_path"),
    )
    if img_path is None:
        return None
    try:
        crop = extract_face_crop(
            img_path,
            row["face_x1"], row["face_y1"], row["face_x2"], row["face_y2"],
            FACE_CROP_PADDING,
            kps=_parse_kps(row.get("kps")),
        )
    except Exception as e:
        logger.warning("Failed to crop %s: %s", img_path, e)
        return None
    return crop.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)


CSS = """
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 16px;
  background: #1a1a1a;
  color: #eee;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 13px;
}
h1 { margin: 0 0 16px 0; font-size: 18px; font-weight: 600; }
.subtitle { color: #888; margin: 0 0 16px 0; font-size: 12px; }
table { border-collapse: separate; border-spacing: 8px; }
th { font-weight: 600; color: #ccc; padding: 4px 8px; text-align: center; font-size: 12px; }
th.score-h { text-align: right; min-width: 60px; }
td { vertical-align: top; padding: 0; }
td.score {
  vertical-align: middle;
  text-align: right;
  padding-right: 8px;
  color: #fff;
  font-weight: 600;
  font-size: 13px;
}
.cell {
  width: 128px;
  height: 168px;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.cell img {
  width: 128px;
  height: 128px;
  display: block;
  border-radius: 4px;
  background: #000;
}
.cell.original img { box-shadow: 0 0 0 2px var(--label-color, transparent); }
.cell .label {
  margin-top: 6px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--label-color, #888);
}
.cell .key {
  margin-top: 2px;
  font-size: 10px;
  color: #666;
  max-width: 128px;
  word-break: break-all;
  text-align: center;
  line-height: 1.2;
}
"""


def _build_row(
    row: pd.Series,
    aug_imgs: list[Image.Image],
    original: Image.Image,
) -> str:
    label = row.get("_label", "")
    key = row.get("_label_key", "")
    composite = _safe_float(row.get("composite"))
    composite_str = f"{composite:.4f}" if composite is not None else "—"
    color = LABEL_COLORS.get(label, "#888")

    cells = [f'<td class="score">{composite_str}</td>']
    cells.append(
        f'<td><div class="cell original" style="--label-color: {color}">'
        f'<img src="data:image/png;base64,{_b64_png(original)}" alt="">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="key">{html.escape(str(key))}</div>'
        f'</div></td>'
    )
    for aug in aug_imgs:
        cells.append(
            f'<td><div class="cell">'
            f'<img src="data:image/png;base64,{_b64_png(aug)}" alt="">'
            f'</div></td>'
        )
    return "<tr>" + "".join(cells) + "</tr>"


def main() -> None:
    parser = ArgumentParser(
        description="Build a self-contained HTML grid previewing the augmentation stack.",
    )
    parser.add_argument("--scores-csv", type=Path,
                        default=Path("data/mini/refined_scores.csv"))
    parser.add_argument("--labels-json", type=Path, required=True,
                        help="Path to exported labels.json from the labeling UI.")
    parser.add_argument("--output-html", type=Path,
                        default=Path("data/mini/augmentation_preview.html"))
    parser.add_argument("--n-rows", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    rng = random.Random(args.seed)
    augmenter = _build_augmenter()

    labels = json.loads(args.labels_json.read_text(encoding="utf-8"))
    logger.info("Loaded %d labels from %s", len(labels), args.labels_json)

    df = pd.read_csv(args.scores_csv)
    logger.info("Loaded %d scored rows from %s", len(df), args.scores_csv)

    df = _join_labels(df, labels)
    logger.info("%d rows have a matching label", len(df))
    if len(df) == 0:
        raise SystemExit("No rows joined with labels.json; check stem/filename keys.")

    sampled = _stratified_sample(df, args.n_rows, rng)
    logger.info("Sampled %d rows for preview", len(sampled))
    counts = sampled["_label"].value_counts().to_dict()
    logger.info("Sample label distribution: %s", counts)

    rows_html: list[str] = []
    for _, row in sampled.iterrows():
        original = _crop_for_row(row)
        if original is None:
            logger.warning("Could not load crop for %s; skipping", row.get("_label_key"))
            continue
        augs = [_augment_once(original, augmenter, rng) for _ in range(N_AUG_COLS)]
        rows_html.append(_build_row(row, augs, original))

    header_cells = (
        '<th class="score-h">Score</th><th>Original</th>'
        + "".join(f"<th>Aug {i + 1}</th>" for i in range(N_AUG_COLS))
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Augmentation Preview</title>
<style>{CSS}</style>
</head>
<body>
<h1>Augmentation Preview ({len(rows_html)} rows x {N_AUG_COLS} aug)</h1>
<p class="subtitle">seed={args.seed} · column 1 is the original (label-coloured outline) · columns 2-{N_AUG_COLS + 1} are independent random augmentations</p>
<table>
<thead><tr>{header_cells}</tr></thead>
<tbody>
{chr(10).join(rows_html)}
</tbody>
</table>
</body>
</html>
"""

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html_doc, encoding="utf-8")
    size_mb = args.output_html.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%d rows, %.2f MB)", args.output_html, len(rows_html), size_mb)


if __name__ == "__main__":
    main()
