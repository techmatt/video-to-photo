"""Score multiple face-quality classifier checkpoints on the same val split.

Reproduces the train/val split used by ``train_classifier`` (same labels store,
same StratifiedShuffleSplit seed and test_size), then evaluates each provided
checkpoint on the held-out val set. Prints a per-checkpoint confusion matrix
and per-class precision/recall/F1, plus a final side-by-side summary.

Usage:
    uv run python -m still_extractor.compare_classifiers \\
        --checkpoints models/face_quality/best_model_v1.pt \\
                      models/face_quality/best_model_v2.pt
"""

import argparse
import logging
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
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from still_extractor.constants import FACE_QUALITY_LABELS

logger = logging.getLogger(__name__)


def _eval_checkpoint(
    ckpt_path: Path, val_loader: DataLoader, device: torch.device,
) -> dict:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = _build_model().to(device)
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare face-quality classifier checkpoints on the same val split.",
    )
    parser.add_argument("--labels-store", type=Path, default=DEFAULT_LABELS_STORE)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True,
                        help="Two or more checkpoint paths to compare.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match the seed used by train_classifier "
                             "to reproduce the same val split.")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    items = _load_labels_store(args.labels_store)
    if len(items) < N_CLASSES * 2:
        raise SystemExit("Not enough labeled rows for stratified split.")
    item_labels = [lbl for _, lbl in items]
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.seed,
    )
    _, val_pos = next(splitter.split(np.zeros(len(items)), item_labels))
    val_items = [items[i] for i in val_pos]
    val_labels = [lbl for _, lbl in val_items]

    counts = [0] * N_CLASSES
    for y in val_labels:
        counts[y] += 1
    print(
        f"Val set: {len(val_items)} samples (seed={args.seed}, "
        f"test_size={args.test_size}); class counts: "
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
    print(
        f"{'checkpoint':<{name_w}} {'epoch':>6} {'val_loss':>9} "
        f"{'val_acc':>8} {'macro_f1':>9}"
    )
    for r in results:
        macro_f1 = float(np.mean(r["f1"]))
        print(
            f"{Path(r['path']).name:<{name_w}} {str(r['ckpt_epoch']):>6} "
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

    if len(results) >= 2:
        a, b = results[0], results[1]
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
