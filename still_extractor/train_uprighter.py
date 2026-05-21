"""Train a 4-class image-rotation classifier (uprighter) on synthetic rotations.

Each source frame produces four training examples via on-the-fly 0°/90°/180°/270°
clockwise rotation. Three input-resize strategies (letterbox / squish / center crop)
are selected uniformly per example during training and averaged at inference time
(3-strategy TTA). Architecture: MobileNetV3-Small (ImageNet pretrained), trained
in two phases — head only, then last InvertedResidual block + head — mirroring
the pattern in `train_classifier.py`.
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from still_extractor.constants import (
    DEFAULT_UPRIGHTER_MODEL,
    IMAGENET_MEAN,
    IMAGENET_STD,
    UPRIGHTER_INPUT_SIZE,
)

logger = logging.getLogger(__name__)


N_CLASSES = 4
IMAGE_SIZE = UPRIGHTER_INPUT_SIZE
CACHE_MAX_DIM = 256  # All sources pre-decoded + downsized to max-dim 256 in RAM.
ROTATION_LABELS = [0, 90, 180, 270]


def _load_corpus(
    frames_json: Path, rejected_json: Path, repo_root: Path,
) -> tuple[list[Path], int, int]:
    frames = json.loads(frames_json.read_text(encoding="utf-8"))
    rejected = set(json.loads(rejected_json.read_text(encoding="utf-8")))
    kept = [f for f in frames if f not in rejected]
    paths = [repo_root / f for f in kept]
    logger.info(
        "Corpus: %d frames after filtering (loaded %d, rejected %d)",
        len(paths), len(frames), len(rejected),
    )
    return paths, len(frames), len(rejected)


def _build_image_cache(paths: list[Path], max_dim: int) -> list[Image.Image]:
    """Decode each source frame once and downsize so max(w,h)=max_dim, preserving aspect."""
    cache: list[Image.Image] = []
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            img = im.convert("RGB")
        w, h = img.size
        scale = max_dim / max(w, h)
        if scale < 1.0:
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.BILINEAR)
        cache.append(img)
        if (i + 1) % 500 == 0:
            logger.info("  cached %d / %d frames", i + 1, len(paths))
    total_px = sum(im.size[0] * im.size[1] for im in cache)
    logger.info(
        "Image cache built: %d images, ~%.1f MiB (RGB, max-dim %d)",
        len(cache), total_px * 3 / (1024 * 1024), max_dim,
    )
    return cache


# --- Resize strategies -------------------------------------------------------

def _letterbox(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def _squish(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.BILINEAR)


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    cropped = img.crop((left, top, left + s, top + s))
    return cropped.resize((size, size), Image.BILINEAR)


STRATEGY_FNS = (_letterbox, _squish, _center_crop)
STRATEGY_NAMES = ("letterbox", "squish", "center_crop")


class RandomResizeStrategy:
    """Pick one of the three resize strategies uniformly at random."""

    def __init__(self, size: int = IMAGE_SIZE) -> None:
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        fn = random.choice(STRATEGY_FNS)
        return fn(img, self.size)


def apply_all_strategies(img: Image.Image, size: int = IMAGE_SIZE) -> list[Image.Image]:
    return [fn(img, size) for fn in STRATEGY_FNS]


# --- Augmentation ------------------------------------------------------------

class RandomBorderCrop:
    """Crop 0–max_frac independently from each border, then resize back to original."""

    def __init__(self, max_frac: float = 0.05) -> None:
        self.max_frac = max_frac

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        fr_l = random.uniform(0.0, self.max_frac)
        fr_t = random.uniform(0.0, self.max_frac)
        fr_r = random.uniform(0.0, self.max_frac)
        fr_b = random.uniform(0.0, self.max_frac)
        left = int(round(fr_l * w))
        top = int(round(fr_t * h))
        right = w - int(round(fr_r * w))
        bottom = h - int(round(fr_b * h))
        if right - left < 1 or bottom - top < 1:
            return img
        cropped = img.crop((left, top, right, bottom))
        return cropped.resize((w, h), Image.BILINEAR)


def _build_train_post_resize_transform() -> T.Compose:
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
        RandomBorderCrop(max_frac=0.05),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _build_normalize_only() -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# --- Rotation ----------------------------------------------------------------

def _rotate_pil(img: Image.Image, rot_idx: int) -> Image.Image:
    """rot_idx in {0,1,2,3} = {0°, 90°, 180°, 270°} clockwise.

    PIL's ROTATE_90 / ROTATE_270 are counter-clockwise, so the mapping inverts
    for the 90° and 270° cases.
    """
    if rot_idx == 0:
        return img
    if rot_idx == 1:
        return img.transpose(Image.ROTATE_270)  # 90° CW
    if rot_idx == 2:
        return img.transpose(Image.ROTATE_180)
    if rot_idx == 3:
        return img.transpose(Image.ROTATE_90)   # 270° CW
    raise ValueError(f"Invalid rot_idx: {rot_idx}")


# --- Datasets ----------------------------------------------------------------

class UprighterDataset(Dataset):
    """Yields (tensor, label). Each cached source image contributes 4 examples."""

    def __init__(self, cached_images: list[Image.Image], training: bool) -> None:
        self.cached_images = cached_images
        self.training = training
        self.random_resize = RandomResizeStrategy(IMAGE_SIZE) if training else None
        self.transform = (
            _build_train_post_resize_transform() if training else _build_normalize_only()
        )

    def __len__(self) -> int:
        return len(self.cached_images) * N_CLASSES

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        src_idx, rot_idx = divmod(idx, N_CLASSES)
        img = _rotate_pil(self.cached_images[src_idx], rot_idx)
        if self.training:
            img = self.random_resize(img)
        else:
            img = _letterbox(img, IMAGE_SIZE)
        return self.transform(img), rot_idx


class ValTTADataset(Dataset):
    """Yields (stacked 3-strategy tensors, label) for 3-strategy TTA evaluation."""

    def __init__(self, cached_images: list[Image.Image]) -> None:
        self.cached_images = cached_images
        self.normalize = _build_normalize_only()

    def __len__(self) -> int:
        return len(self.cached_images) * N_CLASSES

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        src_idx, rot_idx = divmod(idx, N_CLASSES)
        img = _rotate_pil(self.cached_images[src_idx], rot_idx)
        tensors = torch.stack([self.normalize(v) for v in apply_all_strategies(img, IMAGE_SIZE)])
        return tensors, rot_idx


# --- Model -------------------------------------------------------------------

def _build_model() -> nn.Module:
    backbone = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    )
    in_features = backbone.classifier[3].in_features
    backbone.classifier[3] = nn.Linear(in_features, N_CLASSES)
    return backbone


def _apply_phase(model: nn.Module, phase: int) -> None:
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
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Phase %d: %d trainable tensors, %d parameters", phase, n_trainable, n_params)


# --- Train / eval loops ------------------------------------------------------

def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
    criterion: nn.Module, device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_seen = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        n_seen += x.size(0)
    return total_loss / max(n_seen, 1)


@torch.no_grad()
def evaluate_single(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n_seen = 0
    correct = 0
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        n_seen += x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        for gt, pr in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[int(gt), int(pr)] += 1
    return total_loss / max(n_seen, 1), correct / max(n_seen, 1), cm


@torch.no_grad()
def evaluate_tta(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    n_seen = 0
    correct = 0
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for tensors, y in loader:
        b, k, c, h, w = tensors.shape
        flat = tensors.view(b * k, c, h, w).to(device, non_blocking=True)
        logits = model(flat).view(b, k, N_CLASSES).mean(dim=1)
        pred = logits.argmax(dim=1)
        y_dev = y.to(device)
        correct += int((pred == y_dev).sum().item())
        n_seen += b
        for gt, pr in zip(y.numpy(), pred.cpu().numpy()):
            cm[int(gt), int(pr)] += 1
    return correct / max(n_seen, 1), cm


def _print_confusion(cm: np.ndarray, title: str) -> None:
    print(title)
    col_header = "          " + " ".join(f"{ROTATION_LABELS[c]:>6}d" for c in range(N_CLASSES))
    print(col_header)
    for r in range(N_CLASSES):
        row_str = " ".join(f"{cm[r, c]:>6d}" for c in range(N_CLASSES))
        print(f"{ROTATION_LABELS[r]:>8}d  {row_str}")
    per_class = [
        cm[r, r] / cm[r].sum() if cm[r].sum() > 0 else float("nan")
        for r in range(N_CLASSES)
    ]
    print("per-class acc: " + " ".join(
        f"{ROTATION_LABELS[c]}d={per_class[c]:.3f}" for c in range(N_CLASSES)
    ))


# --- Main --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a 4-class rotation (uprighter) classifier on synthetic rotations.",
    )
    parser.add_argument("--frames-json", type=Path,
                        default=Path("data/june27/uprighter_frames.json"))
    parser.add_argument("--rejected-json", type=Path,
                        default=Path("labels/rejected.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_UPRIGHTER_MODEL.parent)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr-phase1", type=float, default=1e-3)
    parser.add_argument("--lr-phase2", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
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

    paths, n_loaded, n_rejected = _load_corpus(
        args.frames_json, args.rejected_json, args.repo_root,
    )
    num_source = len(paths)
    if num_source < 10:
        raise SystemExit(f"Corpus too small ({num_source}) — refusing to train.")

    # Split by source frame so all 4 rotations of a frame go to the same split.
    # Classes are perfectly balanced by construction.
    rng = random.Random(args.seed)
    indices = list(range(num_source))
    rng.shuffle(indices)
    n_train = int(round(0.8 * num_source))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    train_paths = [paths[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    logger.info(
        "Split: %d train / %d val source frames (%d / %d examples after 4x rotation)",
        len(train_paths), len(val_paths),
        len(train_paths) * N_CLASSES, len(val_paths) * N_CLASSES,
    )

    logger.info("Pre-decoding train cache...")
    train_cache = _build_image_cache(train_paths, CACHE_MAX_DIM)
    logger.info("Pre-decoding val cache...")
    val_cache = _build_image_cache(val_paths, CACHE_MAX_DIM)

    train_ds = UprighterDataset(train_cache, training=True)
    val_ds = UprighterDataset(val_cache, training=False)
    val_tta_ds = ValTTADataset(val_cache)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )
    val_tta_loader = DataLoader(
        val_tta_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    model = _build_model().to(device)
    criterion = nn.CrossEntropyLoss()

    log_rows: list[dict] = []
    best_state: dict | None = None
    best_val_loss = float("inf")
    best_epoch = 0

    header = (
        f"{'epoch':>5} {'ph':>2} {'lr':>9} "
        f"{'tr_loss':>9} {'val_loss':>9} {'val_acc':>8}"
    )
    print(header)
    print("-" * len(header))

    phase1_end = max(1, args.epochs // 2)
    phase2_start = phase1_end + 1

    # === Phase 1: head only, no early stopping ===
    _apply_phase(model, 1)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_phase1,
    )
    logger.info("=== Phase 1: head only (epochs 1-%d) ===", phase1_end)
    for epoch in range(1, phase1_end + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _ = evaluate_single(model, val_loader, device, criterion)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5d} {1:>2d} {cur_lr:>9.2e} "
              f"{tr_loss:>9.4f} {val_loss:>9.4f} {val_acc:>8.3f}")
        log_rows.append({
            "epoch": epoch, "phase": 1, "lr": cur_lr,
            "train_loss": tr_loss, "val_loss": val_loss, "val_acc": val_acc,
        })

    # === Phase 2: unfreeze last conv block + head, early stopping on val loss ===
    _apply_phase(model, 2)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_phase2,
    )
    patience_left = args.patience
    logger.info(
        "=== Phase 2: unfreeze last conv block + head (epochs %d-%d) ===",
        phase2_start, args.epochs,
    )
    for epoch in range(phase2_start, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _ = evaluate_single(model, val_loader, device, criterion)
        cur_lr = optimizer.param_groups[0]["lr"]
        marker = ""
        if val_loss < best_val_loss:
            marker = "  [best]"
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        print(f"{epoch:>5d} {2:>2d} {cur_lr:>9.2e} "
              f"{tr_loss:>9.4f} {val_loss:>9.4f} {val_acc:>8.3f}{marker}")
        log_rows.append({
            "epoch": epoch, "phase": 2, "lr": cur_lr,
            "train_loss": tr_loss, "val_loss": val_loss, "val_acc": val_acc,
        })
        if patience_left <= 0:
            logger.info(
                "Early stop at epoch %d (best val_loss=%.4f @ epoch %d)",
                epoch, best_val_loss, best_epoch,
            )
            break

    if best_state is None:
        logger.warning("No Phase 2 epoch improved val_loss; saving final model state.")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = args.epochs

    # Reload best state for final evaluation.
    model.load_state_dict(best_state)
    val_loss_final, val_acc_single, cm_single = evaluate_single(
        model, val_loader, device, criterion,
    )
    val_acc_tta, cm_tta = evaluate_tta(model, val_tta_loader, device)

    best_path = args.output_dir / "best_model.pt"
    torch.save({
        "model_state_dict": best_state,
        "val_acc_single": float(val_acc_single),
        "val_acc_tta": float(val_acc_tta),
        "val_loss": float(val_loss_final),
        "epoch": int(best_epoch),
        "num_source_frames": int(num_source),
        "num_rejected": int(n_rejected),
    }, best_path)
    logger.info("Saved best model -> %s", best_path)

    log_path = args.output_dir / "training_log.json"
    log_path.write_text(json.dumps(log_rows, indent=2), encoding="utf-8")
    logger.info("Wrote training log -> %s (%d epochs)", log_path, len(log_rows))

    print()
    print(
        f"Best epoch: {best_epoch}  val_loss={val_loss_final:.4f}  "
        f"val_acc_single={val_acc_single:.4f}  val_acc_tta={val_acc_tta:.4f}"
    )
    print()
    _print_confusion(cm_single, "Confusion matrix - single-pass (letterbox, rows=gt, cols=pred):")
    print()
    _print_confusion(cm_tta, "Confusion matrix - 3-strategy TTA (rows=gt, cols=pred):")


if __name__ == "__main__":
    main()
