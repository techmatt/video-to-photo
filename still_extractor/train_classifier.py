"""Train a 4-class face quality classifier (None/Bad/Okay/Good) on labeled face crops.

Backbone: MobileNetV3-Small (ImageNet pretrained), trained in two phases —
first the classifier head only, then the last InvertedResidual block plus
the head. After training, runs inference on the full corpus (refined_scores.csv)
with and without 5-pass test-time augmentation, and writes soft labels to CSV.
"""

import argparse
import io
import json
import logging
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from still_extractor.build_index_html import _parse_kps
from still_extractor.face_crop import extract_face_crop

logger = logging.getLogger(__name__)


LABEL_TO_IDX = {"none": 0, "bad": 1, "okay": 2, "good": 3}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
N_CLASSES = 4
FACE_CROP_PADDING = 20
DISPLAY_SIZE = 128
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class JpegRecompress:
    """Re-encode the image as JPEG at a random quality, then decode. Matches Prompt 27."""

    def __init__(self, q_min: int = 60, q_max: int = 95) -> None:
        self.q_min = q_min
        self.q_max = q_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        quality = random.randint(self.q_min, self.q_max)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()


def _build_train_transform() -> T.Compose:
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BICUBIC, fill=0),
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
        JpegRecompress(60, 95),
        T.ToTensor(),
        T.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.3, 3.0), value=0),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _build_val_transform() -> T.Compose:
    return T.Compose([
        T.Resize((DISPLAY_SIZE, DISPLAY_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _row_key(row: pd.Series) -> str | None:
    stem = row.get("video_stem")
    if not isinstance(stem, str) or not stem:
        return None
    refined = row.get("refined_frame_path")
    if isinstance(refined, str) and refined:
        return f"{stem}/{Path(refined).name}"
    raw = row.get("frame_path")
    if isinstance(raw, str) and raw:
        return f"{stem}/{Path(raw).name}"
    return None


def _crop_for_row(row: pd.Series) -> Image.Image | None:
    refined = row.get("refined_frame_path")
    raw = row.get("frame_path")
    img_path: Path | None = None
    if isinstance(refined, str) and refined and Path(refined).exists():
        img_path = Path(refined)
    elif isinstance(raw, str) and raw and Path(raw).exists():
        img_path = Path(raw)
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


class FaceCropDataset(Dataset):
    def __init__(
        self,
        crops: list[Image.Image],
        labels: list[int] | None,
        transform,
    ) -> None:
        self.crops = crops
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int):
        img = self.transform(self.crops[idx])
        if self.labels is None:
            return img
        return img, self.labels[idx]


def mixup_batch(
    x: torch.Tensor, y_onehot: torch.Tensor, alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x_mix, y_mix


def _build_model() -> nn.Module:
    backbone = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    )
    in_features = backbone.classifier[3].in_features
    backbone.classifier[3] = nn.Linear(in_features, N_CLASSES)
    return backbone


def _apply_phase(model: nn.Module, phase: int) -> None:
    """Freeze/unfreeze parameters according to the training phase, and log trainables."""
    if phase == 1:
        for name, param in model.named_parameters():
            param.requires_grad = "classifier" in name
    elif phase == 2:
        for name, param in model.named_parameters():
            param.requires_grad = (
                "features.12" in name
                or "features.13" in name
                or "classifier" in name
            )
    else:
        raise ValueError(f"Unknown phase {phase}")
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Phase %d: %d trainable tensors, %d parameters",
        phase, len(trainable_names), n_params,
    )
    for name in trainable_names:
        logger.debug("  trainable: %s", name)


def _make_optimizer(
    model: nn.Module, lr: float, total_epochs: int, start_epoch: int = 0,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-5,
    )
    # Fast-forward the scheduler to match the current epoch when resuming after
    # a phase transition. CosineAnnealingLR computes LR purely from its internal
    # step counter, so manually stepping it is the simplest way to align without
    # needing to seed initial_lr on the freshly-created optimizer.
    if start_epoch > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(start_epoch):
                scheduler.step()
    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_seen = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_onehot = F.one_hot(y, num_classes=N_CLASSES).float()
        x_mix, y_mix = mixup_batch(x, y_onehot, alpha=0.2)
        logits = model(x_mix)
        loss = -(y_mix * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        n_seen += x.size(0)
    return total_loss / max(n_seen, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, list[float]]:
    model.eval()
    total_loss = 0.0
    n_seen = 0
    correct = 0
    per_class_correct = [0] * N_CLASSES
    per_class_total = [0] * N_CLASSES
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        n_seen += x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        for c in range(N_CLASSES):
            mask = y == c
            per_class_total[c] += int(mask.sum().item())
            per_class_correct[c] += int(((pred == y) & mask).sum().item())
    val_loss = total_loss / max(n_seen, 1)
    val_acc = correct / max(n_seen, 1)
    per_class_acc = [
        per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else float("nan")
        for c in range(N_CLASSES)
    ]
    return val_loss, val_acc, per_class_acc


def _run_inference_pass(
    model: nn.Module, crops: list[Image.Image], transform,
    device: torch.device, batch_size: int,
) -> np.ndarray:
    ds = FaceCropDataset(crops, labels=None, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_probs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def _format_counts(labels: list[int]) -> dict[str, int]:
    counts = [0] * N_CLASSES
    for y in labels:
        counts[y] += 1
    return {IDX_TO_LABEL[i]: counts[i] for i in range(N_CLASSES)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train face quality classifier and run inference on the full corpus.",
    )
    parser.add_argument("--scores-csv", type=Path,
                        default=Path("data/mini/refined_scores.csv"))
    parser.add_argument("--labels-json", type=Path,
                        default=Path("save/labels.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("models/face_quality"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tta-passes", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    labels_raw = json.loads(args.labels_json.read_text(encoding="utf-8"))
    df = pd.read_csv(args.scores_csv).reset_index(drop=True)
    logger.info(
        "Loaded %d scored rows from %s, %d labels from %s",
        len(df), args.scores_csv, len(labels_raw), args.labels_json,
    )

    keys: list[str | None] = [_row_key(df.iloc[i]) for i in range(len(df))]
    label_strs: list[str | None] = [
        labels_raw.get(k) if isinstance(k, str) else None for k in keys
    ]

    logger.info("Extracting %d face crops...", len(df))
    crops: list[Image.Image | None] = []
    for i in tqdm(range(len(df)), desc="crops", unit="img"):
        crops.append(_crop_for_row(df.iloc[i]))
    missing = sum(1 for c in crops if c is None)
    if missing:
        logger.warning("%d/%d rows had unreadable crops", missing, len(crops))

    labeled_positions = [
        i for i in range(len(df))
        if label_strs[i] in LABEL_TO_IDX and crops[i] is not None
    ]
    labeled_crops = [crops[i] for i in labeled_positions]
    labeled_labels = [LABEL_TO_IDX[label_strs[i]] for i in labeled_positions]
    logger.info(
        "Labeled rows usable for training: %d (of %d total labels)",
        len(labeled_positions), len(labels_raw),
    )
    logger.info("Full labeled class counts: %s", _format_counts(labeled_labels))
    if len(labeled_positions) < N_CLASSES * 2:
        raise SystemExit("Not enough labeled rows for stratified train/val split.")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=args.seed)
    train_pos, val_pos = next(
        splitter.split(np.zeros(len(labeled_labels)), labeled_labels),
    )
    train_crops = [labeled_crops[i] for i in train_pos]
    train_labels = [labeled_labels[i] for i in train_pos]
    val_crops = [labeled_crops[i] for i in val_pos]
    val_labels = [labeled_labels[i] for i in val_pos]
    logger.info("Train class counts: %s", _format_counts(train_labels))
    logger.info("Val   class counts: %s", _format_counts(val_labels))

    train_transform = _build_train_transform()
    val_transform = _build_val_transform()

    train_ds = FaceCropDataset(train_crops, train_labels, train_transform)
    val_ds = FaceCropDataset(val_crops, val_labels, val_transform)

    train_counts = [0] * N_CLASSES
    for y in train_labels:
        train_counts[y] += 1
    sample_weights = [
        1.0 / train_counts[y] if train_counts[y] > 0 else 0.0
        for y in train_labels
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    model = _build_model().to(device)
    _apply_phase(model, 1)
    optimizer, scheduler = _make_optimizer(model, args.lr, args.epochs, start_epoch=0)
    val_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    phase1_end = args.epochs // 2
    phase2_start = phase1_end + 1
    best_val_loss = float("inf")
    best_epoch = 0
    best_path = args.output_dir / "best_model.pt"
    log_rows: list[dict] = []

    header = (
        f"{'epoch':>5} {'ph':>2} {'lr':>9} "
        f"{'tr_loss':>9} {'val_loss':>9} {'val_acc':>8} | per-class acc (n/b/o/g)"
    )
    print(header)
    print("-" * len(header))

    def _run_epoch(epoch: int, phase: int) -> tuple[float, float]:
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, per_class_acc = evaluate(
            model, val_loader, device, val_criterion,
        )
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        pc_str = " ".join(
            f"{IDX_TO_LABEL[c][0]}={per_class_acc[c]:.2f}"
            for c in range(N_CLASSES)
        )
        marker = "  [best]" if phase == 2 and val_loss < best_val_loss else ""
        print(
            f"{epoch:>5d} {phase:>2d} {cur_lr:>9.2e} "
            f"{tr_loss:>9.4f} {val_loss:>9.4f} {val_acc:>8.3f} | {pc_str}{marker}"
        )
        log_rows.append({
            "epoch": epoch,
            "phase": phase,
            "lr": cur_lr,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            **{f"acc_{IDX_TO_LABEL[c]}": per_class_acc[c] for c in range(N_CLASSES)},
        })
        return tr_loss, val_loss

    logger.info(
        "=== Phase 1: training head only (epochs 1-%d) ===", phase1_end,
    )
    for epoch in range(1, phase1_end + 1):
        _run_epoch(epoch, phase=1)

    logger.info(
        "=== Phase 2: unfreezing last conv block (epochs %d-%d) ===",
        phase2_start, args.epochs,
    )
    _apply_phase(model, 2)
    optimizer, scheduler = _make_optimizer(
        model, args.lr, args.epochs, start_epoch=phase1_end,
    )
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = args.patience

    for epoch in range(phase2_start, args.epochs + 1):
        _, val_loss = _run_epoch(epoch, phase=2)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch,
                 "val_loss": val_loss},
                best_path,
            )
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(
                    "Early stop at epoch %d (best val_loss=%.4f @ epoch %d)",
                    epoch, best_val_loss, best_epoch,
                )
                break

    pd.DataFrame(log_rows).to_csv(args.output_dir / "training_log.csv", index=False)
    logger.info(
        "Wrote training_log.csv (%d epochs, best val_loss=%.4f @ epoch %d)",
        len(log_rows), best_val_loss, best_epoch,
    )

    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    logger.info("Reloaded best model from epoch %d", state["epoch"])

    valid_idx = [i for i, c in enumerate(crops) if c is not None]
    valid_crops = [crops[i] for i in valid_idx]
    logger.info(
        "Running inference on %d/%d rows (rest had unreadable crops)",
        len(valid_crops), len(crops),
    )

    base_probs = _run_inference_pass(
        model, valid_crops, val_transform, device, args.batch_size,
    )
    tta_accum = np.zeros_like(base_probs)
    n_tta = max(args.tta_passes, 1)
    for k in range(n_tta):
        logger.info("TTA pass %d/%d", k + 1, n_tta)
        tta_accum += _run_inference_pass(
            model, valid_crops, train_transform, device, args.batch_size,
        )
    tta_probs = tta_accum / n_tta

    out = df.copy()
    for c in range(N_CLASSES):
        out[f"p_{IDX_TO_LABEL[c]}"] = np.nan
        out[f"p_{IDX_TO_LABEL[c]}_tta"] = np.nan
    for pos, i in enumerate(valid_idx):
        for c in range(N_CLASSES):
            out.at[i, f"p_{IDX_TO_LABEL[c]}"] = float(base_probs[pos, c])
            out.at[i, f"p_{IDX_TO_LABEL[c]}_tta"] = float(tta_probs[pos, c])

    pred_labels: list[str] = []
    pred_conf: list[float] = []
    for i in range(len(df)):
        if crops[i] is None:
            pred_labels.append("")
            pred_conf.append(float("nan"))
            continue
        row_tta = np.array([
            out.at[i, f"p_{IDX_TO_LABEL[c]}_tta"] for c in range(N_CLASSES)
        ])
        c = int(np.argmax(row_tta))
        pred_labels.append(IDX_TO_LABEL[c])
        pred_conf.append(float(row_tta[c]))
    out["pred_label"] = pred_labels
    out["pred_confidence"] = pred_conf
    out["gt_label"] = [s if isinstance(s, str) else "" for s in label_strs]

    inference_path = args.output_dir / "inference_scores.csv"
    out.to_csv(inference_path, index=False)
    logger.info("Wrote %s (%d rows)", inference_path, len(out))

    # Also write alongside the scores CSV so the HTML builder finds it by default.
    scores_inference_path = args.scores_csv.parent / "classifier" / "inference_scores.csv"
    scores_inference_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(scores_inference_path, index=False)
    logger.info("Wrote %s (%d rows)", scores_inference_path, len(out))

    nonempty_preds = [p for p in pred_labels if p]
    pred_counts = pd.Series(nonempty_preds).value_counts().to_dict()
    finite_conf = [c for c in pred_conf if c == c]  # filter NaN
    mean_conf = float(np.mean(finite_conf)) if finite_conf else float("nan")
    logger.info("Prediction counts: %s", pred_counts)
    logger.info("Mean prediction confidence (TTA): %.3f", mean_conf)

    labeled_pairs = [
        (gt, pr) for gt, pr in zip(out["gt_label"], pred_labels)
        if gt in LABEL_TO_IDX and pr in LABEL_TO_IDX
    ]
    if labeled_pairs:
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        for gt, pr in labeled_pairs:
            cm[LABEL_TO_IDX[gt], LABEL_TO_IDX[pr]] += 1
        print("Confusion matrix on labeled rows (rows=gt, cols=pred):")
        col_header = "          " + " ".join(
            f"{IDX_TO_LABEL[c]:>6}" for c in range(N_CLASSES)
        )
        print(col_header)
        for r in range(N_CLASSES):
            row_str = " ".join(f"{cm[r, c]:>6d}" for c in range(N_CLASSES))
            print(f"{IDX_TO_LABEL[r]:>10} {row_str}")
        correct = int(np.trace(cm))
        total = int(cm.sum())
        print(f"Labeled accuracy: {correct}/{total} = {correct / max(total, 1):.3f}")


if __name__ == "__main__":
    main()
