"""Score multiple face-quality classifier checkpoints on the same val split.

The val split is the frozen ``is_val=true`` subset of ``--labels-store``
(written one-time by the v11 prep step), so every checkpoint is scored on
exactly the same held-out entries regardless of how the label store grows.
Prints a per-checkpoint confusion matrix and per-class precision/recall/F1,
plus a final side-by-side summary.

By default, auto-discovers all ``best_model_v*.pt`` files under
``--checkpoint-dir`` (default ``models/face_quality/``) so new versions are
picked up automatically.

Usage:
    uv run python -m still_extractor.compare_classifiers \\
        --labels-store data/face_labels/labels.json
"""

import argparse
import json
import logging
import re
from pathlib import Path

# Import train_classifier first so pandas loads before torch — on Windows the
# reverse order can trigger a heap-corruption crash in pandas._libs.tslibs.
from still_extractor.train_classifier import (
    DEFAULT_LABELS_STORE,
    IDX_TO_LABEL,
    N_CLASSES,
    FaceCropDataset,
    _build_model,
    _build_val_transform,
    _collect_predictions,
    _load_labels_store,
    _print_validation_report,
)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from still_extractor.constants import FACE_QUALITY_LABELS

logger = logging.getLogger(__name__)


_DASH = "-"


def _load_sidecar(ckpt_path: Path) -> dict:
    """Return the sidecar JSON next to a checkpoint, or {} if absent/invalid."""
    sidecar_path = ckpt_path.with_suffix(".json")
    if not sidecar_path.exists():
        return {}
    try:
        return json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to read sidecar %s: %s", sidecar_path, e)
        return {}


def _eval_checkpoint(
    ckpt_path: Path, val_loader: DataLoader, device: torch.device,
) -> dict:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sidecar = _load_sidecar(ckpt_path)
    arch = sidecar.get("arch") or state.get("arch") or "mobilenet_v3_small"
    model = _build_model(arch).to(device)
    model.load_state_dict(state["model_state"])
    val_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.eval()
    total_loss = 0.0
    n_seen = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = val_criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
    val_loss = total_loss / max(n_seen, 1)

    y_true, y_pred = _collect_predictions(model, val_loader, device)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(N_CLASSES)), zero_division=0,
    )
    acc = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    return {
        "path": ckpt_path,
        "ckpt_epoch": state.get("epoch", None),
        "ckpt_val_loss": float(state.get("val_loss", float("nan"))),
        "val_loss": val_loss,
        "val_acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": support,
        "y_true": y_true,
        "y_pred": y_pred,
        "codename": sidecar.get("codename") or state.get("codename") or _DASH,
        "arch": arch if sidecar.get("arch") or state.get("arch") else _DASH,
        "save_policy": sidecar.get("checkpoint_save_policy") or _DASH,
        "notes": sidecar.get("notes") or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare face-quality classifier checkpoints on the same val split.",
    )
    parser.add_argument("--labels-store", type=Path, default=DEFAULT_LABELS_STORE)
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=None,
                        help="Checkpoint paths to compare. If omitted, "
                             "auto-discovers best_model_v*.pt in --checkpoint-dir.")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("models/face_quality"),
                        help="Directory scanned for best_model_v*.pt when "
                             "--checkpoints is not provided.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.checkpoints is None:
        # Match both legacy `best_model_v8.pt` and codename-tagged
        # `best_model_v8_iron_sparrow.pt` filenames.
        pattern = re.compile(
            r"best_model_v(\d+)(?:_[a-z][a-z_]*)?\.pt$", re.IGNORECASE,
        )
        discovered: list[tuple[int, str, Path]] = []
        for p in args.checkpoint_dir.glob("best_model_v*.pt"):
            m = pattern.search(p.name)
            if m:
                discovered.append((int(m.group(1)), p.name, p))
        if not discovered:
            raise SystemExit(
                f"No best_model_v*.pt checkpoints found under {args.checkpoint_dir}",
            )
        discovered.sort(key=lambda t: (t[0], t[1]))
        args.checkpoints = [p for _, _, p in discovered]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    items = _load_labels_store(args.labels_store)
    if len(items) < N_CLASSES * 2:
        raise SystemExit("Not enough labeled rows to evaluate.")
    val_items = [(p, l) for p, l, is_val in items if is_val]
    if not val_items:
        raise SystemExit(
            f"No is_val=true entries in {args.labels_store}; freeze the val "
            f"set first (see docs/prompt_v11_freeze_val.md).",
        )
    val_labels = [lbl for _, lbl in val_items]

    counts = [0] * N_CLASSES
    for y in val_labels:
        counts[y] += 1
    print(
        f"Val set: {len(val_items)} fixed entries (is_val=true); "
        f"class counts: "
        f"{ {IDX_TO_LABEL[c]: counts[c] for c in range(N_CLASSES)} }",
    )

    val_ds = FaceCropDataset(val_items, _build_val_transform())
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    results: list[dict] = []
    for ckpt in args.checkpoints:
        if not ckpt.exists():
            raise SystemExit(f"Checkpoint not found: {ckpt}")
        print(f"\n{'=' * 72}\n{ckpt}\n{'=' * 72}")
        r = _eval_checkpoint(ckpt, val_loader, device)
        print(
            f"Stored metadata: epoch={r['ckpt_epoch']}, "
            f"val_loss (at training time)={r['ckpt_val_loss']:.4f}"
        )
        print(
            f"Val on this split: loss={r['val_loss']:.4f}, "
            f"accuracy={r['val_acc']:.3f}"
        )
        _print_validation_report(r["y_true"], r["y_pred"])
        results.append(r)

    print(f"\n{'=' * 72}\nSummary\n{'=' * 72}")
    name_w = max(len(Path(r["path"]).name) for r in results) + 2
    codename_w = max([len(str(r["codename"])) for r in results] + [len("codename")]) + 2
    arch_w = max([len(str(r["arch"])) for r in results] + [len("arch")]) + 2
    print(
        f"{'checkpoint':<{name_w}} {'codename':<{codename_w}} "
        f"{'arch':<{arch_w}} {'epoch':>6} {'val_loss':>9} "
        f"{'val_acc':>8} {'macro_f1':>9}"
    )
    for r in results:
        macro_f1 = float(np.mean(r["f1"]))
        print(
            f"{Path(r['path']).name:<{name_w}} "
            f"{str(r['codename']):<{codename_w}} "
            f"{str(r['arch']):<{arch_w}} {str(r['ckpt_epoch']):>6} "
            f"{r['val_loss']:>9.4f} {r['val_acc']:>8.3f} {macro_f1:>9.3f}"
        )

    print(f"\nPer-class F1:")
    header = (
        f"{'checkpoint':<{name_w}} "
        + " ".join(f"{l:>8}" for l in FACE_QUALITY_LABELS)
    )
    print(header)
    for r in results:
        print(
            f"{Path(r['path']).name:<{name_w}} "
            + " ".join(f"{f:>8.3f}" for f in r["f1"])
        )

    print(f"\nPer-class recall:")
    print(header)
    for r in results:
        print(
            f"{Path(r['path']).name:<{name_w}} "
            + " ".join(f"{v:>8.3f}" for v in r["recall"])
        )

    print(f"\nPer-class precision:")
    print(header)
    for r in results:
        print(
            f"{Path(r['path']).name:<{name_w}} "
            + " ".join(f"{v:>8.3f}" for v in r["precision"])
        )

    print(f"\nPer-class F1 winner:")
    print(f"{'class':>10} {'winner':>22} {'F1':>8}")
    for c in range(N_CLASSES):
        best_idx = int(np.argmax([r["f1"][c] for r in results]))
        best = results[best_idx]
        print(
            f"{FACE_QUALITY_LABELS[c]:>10} "
            f"{Path(best['path']).name:>22} "
            f"{float(best['f1'][c]):>8.3f}"
        )

    if len(results) >= 2:
        a, b = results[0], results[-1]
        print(
            f"\nDelta ({Path(b['path']).name} - {Path(a['path']).name}):"
        )
        print(
            f"  val_loss:  {b['val_loss'] - a['val_loss']:+.4f}\n"
            f"  val_acc:   {b['val_acc'] - a['val_acc']:+.3f}\n"
            f"  macro_f1:  {float(np.mean(b['f1'])) - float(np.mean(a['f1'])):+.3f}"
        )


if __name__ == "__main__":
    main()
