"""Train a 4-class face quality classifier (None/Bad/Okay/Good).

Training data comes from the global face labels store at
``data/ground_truth/face_labels/labels.json`` (a JSON list whose entries point at
already-extracted face crops via ``face_crop_path``). Backbone:
MobileNetV3-Small (ImageNet pretrained), fine-tuned in two phases — first the
classifier head, then the last InvertedResidual block plus the head. After
training, optionally runs inference on every row in ``results.parquet``
(when ``--results`` or ``--config`` is provided), with and without 5-pass
test-time augmentation, and writes soft labels to CSV.
"""

import argparse
import datetime as _dt
import io
import json
import logging
import random
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from still_extractor.constants import (
    DEFAULT_FACE_QUALITY_MODEL,
    FACE_CROP_PADDING,
    FACE_QUALITY_INPUT_SIZE,
    FACE_QUALITY_LABELS,
    FACE_SLOTS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_TO_IDX,
)
from still_extractor.face_crop import extract_face_crop_from_image
from still_extractor.inventory import RunConfig
from still_extractor.utils import parse_kps

logger = logging.getLogger(__name__)


IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
N_CLASSES = len(FACE_QUALITY_LABELS)
DISPLAY_SIZE = FACE_QUALITY_INPUT_SIZE

DEFAULT_LABELS_STORE: Path = Path("data/ground_truth/face_labels/labels.json")
TRAIN_LABEL_SMOOTHING: float = 0.1
MIXUP_ALPHA: float = 0.3
MIXUP_ALPHA_GOOD_PAIR: float = 0.5
SAMPLER_BOOST: dict[str, float] = {
    "none": 1.0, "bad": 1.0, "okay": 1.0, "good": 1.25,
}

ARCH_CHOICES: tuple[str, ...] = ("mobilenet_v3_small", "efficientnet_b0")

# Two-word codename pool for tagging runs. Kept intentionally tame (colors,
# materials, nature, animals) so checkpoint filenames stay legible.
_CODENAME_ADJECTIVES: tuple[str, ...] = (
    "iron", "cobalt", "silver", "amber", "jade", "copper", "ashen", "slate",
    "onyx", "crimson", "cerulean", "sable", "tawny", "russet", "gilt", "ivory",
    "obsidian", "azure", "scarlet", "viridian", "maroon", "opal", "flint",
    "hazel", "umber", "coral", "lichen", "pewter", "dusk", "ember",
)
_CODENAME_NOUNS: tuple[str, ...] = (
    "sparrow", "reef", "mesa", "anvil", "cedar", "shoal", "crest", "flint",
    "beacon", "prism", "ridge", "haven", "glyph", "forge", "delta", "spire",
    "lantern", "cairn", "basalt", "hollow", "mantle", "summit", "larch",
    "cirque", "grotto", "ledge", "solstice", "canopy", "comet", "fern",
)


def generate_codename() -> str:
    """Return a fresh `adjective_noun` codename. Reseeds Python's global RNG
    from OS entropy first so the choice is independent of any prior seeding."""
    random.seed()
    return f"{random.choice(_CODENAME_ADJECTIVES)}_{random.choice(_CODENAME_NOUNS)}"


def _resolve_output_path(
    output: Path | None, output_dir: Path, codename: str,
) -> Path:
    """Resolve the best-checkpoint path, auto-inserting the codename when the
    stem looks like a bare version tag (e.g. ``best_model_v8.pt``)."""
    if output is None:
        return output_dir / f"best_model_{codename}.pt"
    stem = output.stem
    if re.search(r"_v\d+$", stem) or stem in ("best_model",):
        return output.with_name(f"{stem}_{codename}{output.suffix}")
    return output


def _parse_version_tag(path: Path) -> str:
    """Extract ``v<N>`` from a checkpoint stem like ``best_model_v8_iron_sparrow``."""
    m = re.search(r"_v(\d+)(?:_|$)", path.stem)
    return f"v{m.group(1)}" if m else "unknown"


class JpegRecompress:
    """Re-encode the image as JPEG at a random quality, then decode."""

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
        # Down-up resize: attack the resolution-quality shortcut so large
        # "Good" source faces still see low-res inputs at training time.
        T.RandomApply([
            T.Resize(64, interpolation=T.InterpolationMode.BILINEAR),
            T.Resize(128, interpolation=T.InterpolationMode.BILINEAR),
        ], p=0.3),
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


class FaceCropDataset(Dataset):
    """Dataset of labeled face crops. Real items come from disk (opened on
    demand); synthetic items are in-memory PIL crops kept in a separate list so
    they can be regenerated per epoch via ``replace_synthetic_none``."""

    def __init__(
        self,
        items: list[tuple[Path | Image.Image, int]],
        transform,
    ) -> None:
        self.real_items: list[tuple[Path | Image.Image, int]] = list(items)
        self.synthetic_items: list[tuple[Image.Image, int]] = []
        self.transform = transform

    def __len__(self) -> int:
        return len(self.real_items) + len(self.synthetic_items)

    def __getitem__(self, idx: int):
        n_real = len(self.real_items)
        if idx < n_real:
            src, label = self.real_items[idx]
        else:
            src, label = self.synthetic_items[idx - n_real]
        if isinstance(src, Image.Image):
            img = src if src.mode == "RGB" else src.convert("RGB")
        else:
            img = Image.open(src).convert("RGB")
        return self.transform(img), label

    def replace_synthetic_none(self, crops: list[Image.Image]) -> None:
        """Replace the synthetic-none portion in-place. Real items untouched."""
        none_idx = LABEL_TO_IDX["none"]
        self.synthetic_items = [(img, none_idx) for img in crops]

    def current_labels(self) -> list[int]:
        return (
            [lbl for _, lbl in self.real_items]
            + [lbl for _, lbl in self.synthetic_items]
        )


class InMemoryCropDataset(Dataset):
    """In-memory PIL crops used by the inference pass."""

    def __init__(self, crops: list[Image.Image], transform) -> None:
        self.crops = crops
        self.transform = transform

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int):
        return self.transform(self.crops[idx])


def _resolve_crop_path(raw_path: str, labels_store: Path) -> Path:
    """Resolve a crop path from the labels store.

    Tries the path as-is (absolute, or relative to cwd) first; if that does
    not exist, falls back to resolving relative to the labels-store's parent's
    parent (so ``data/ground_truth/face_labels/labels.json`` + ``data/ground_truth/face_labels/foo.jpg``
    works regardless of cwd).
    """
    p = Path(raw_path)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return labels_store.parent.parent / p


def _load_labels_store(labels_store: Path) -> list[tuple[Path, int, bool]]:
    """Load the labels store as ``(crop_path, label_idx, is_val)`` tuples.

    Entries missing the ``is_val`` field are treated as ``is_val=False``
    (train-only) for forward compatibility with future labels added before the
    next freeze; a warning is emitted with the count.
    """
    raw = json.loads(labels_store.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(
            f"Expected JSON list at top level of {labels_store}, "
            f"got {type(raw).__name__}",
        )
    items: list[tuple[Path, int, bool]] = []
    missing_file = 0
    invalid_label = 0
    missing_is_val = 0
    for entry in raw:
        label_str = str(entry.get("label", "")).lower()
        if label_str not in LABEL_TO_IDX:
            invalid_label += 1
            continue
        crop_path = _resolve_crop_path(entry.get("face_crop_path", ""), labels_store)
        if not crop_path.exists():
            missing_file += 1
            continue
        if "is_val" in entry:
            is_val = bool(entry["is_val"])
        else:
            missing_is_val += 1
            is_val = False
        items.append((crop_path, LABEL_TO_IDX[label_str], is_val))
    if missing_file:
        logger.warning(
            "%d/%d labels skipped: face_crop_path does not exist on disk",
            missing_file, len(raw),
        )
    if invalid_label:
        logger.warning(
            "%d/%d labels skipped: label not in %s",
            invalid_label, len(raw), list(LABEL_TO_IDX.keys()),
        )
    if missing_is_val:
        logger.warning(
            "%d/%d labels missing is_val field; treating as train-only.",
            missing_is_val, len(raw),
        )
    return items


def _crop_overlaps_face(
    cx1: float, cy1: float, cx2: float, cy2: float,
    faces: list[tuple[float, float, float, float]],
    threshold: float = 0.10,
) -> bool:
    """True if any (expanded) face bbox covers more than `threshold` of the
    crop's area."""
    crop_area = max((cx2 - cx1) * (cy2 - cy1), 1.0)
    for fx1, fy1, fx2, fy2 in faces:
        ix1, iy1 = max(cx1, fx1), max(cy1, fy1)
        ix2, iy2 = min(cx2, fx2), min(cy2, fy2)
        if ix2 > ix1 and iy2 > iy1:
            if ((ix2 - ix1) * (iy2 - iy1)) / crop_area > threshold:
                return True
    return False


def sample_synthetic_none_crops(
    results_parquet: Path,
    n: int,
    rng: np.random.Generator,
    crop_size: int = FACE_QUALITY_INPUT_SIZE,
    min_crop_px: int = 48,
) -> list[Image.Image]:
    """Sample background (non-face) square crops from keeper JPEGs to use as
    synthetic 'none'-class training examples.

    Rows are sampled uniformly from ``results_parquet``. For each, all face
    bboxes (top-3 slots + rejected_faces_json) are expanded by
    ``FACE_CROP_PADDING`` and the function tries up to 10 random square
    placements per image; the first placement whose intersection with any
    face is below 10% of the crop area is accepted, resized to
    ``crop_size`` with PIL BILINEAR, and returned.
    """
    if n <= 0:
        return []
    df = pd.read_parquet(results_parquet)
    keepers = df[df["kept_path"].notna() & (df["kept_path"] != "")].reset_index(drop=True)
    if len(keepers) == 0:
        logger.warning(
            "No keeper rows in %s; cannot sample synthetic crops",
            results_parquet,
        )
        return []

    n_oversample = min(n * 3, len(keepers))
    sampled_idx = rng.choice(len(keepers), size=n_oversample, replace=False)

    out: list[Image.Image] = []
    skipped_unreadable = 0
    skipped_too_small = 0
    skipped_no_placement = 0
    for idx in sampled_idx:
        if len(out) >= n:
            break
        row = keepers.iloc[int(idx)]
        kept_path = Path(str(row["kept_path"]))
        if not kept_path.exists():
            skipped_unreadable += 1
            continue
        try:
            img = Image.open(kept_path).convert("RGB")
        except Exception as e:
            logger.debug("Failed to open %s: %s", kept_path, e)
            skipped_unreadable += 1
            continue
        img_w, img_h = img.size

        face_boxes: list[tuple[float, float, float, float]] = []
        for i in FACE_SLOTS:
            x1 = row.get(f"face_{i}_x1")
            if pd.isna(x1):
                continue
            y1 = row.get(f"face_{i}_y1")
            x2 = row.get(f"face_{i}_x2")
            y2 = row.get(f"face_{i}_y2")
            ex1 = max(0.0, float(x1) - FACE_CROP_PADDING)
            ey1 = max(0.0, float(y1) - FACE_CROP_PADDING)
            ex2 = min(float(img_w), float(x2) + FACE_CROP_PADDING)
            ey2 = min(float(img_h), float(y2) + FACE_CROP_PADDING)
            face_boxes.append((ex1, ey1, ex2, ey2))

        rj = row.get("rejected_faces_json")
        if isinstance(rj, str) and rj:
            try:
                for entry in json.loads(rj):
                    rx1 = float(entry["x1"]); ry1 = float(entry["y1"])
                    rx2 = float(entry["x2"]); ry2 = float(entry["y2"])
                    ex1 = max(0.0, rx1 - FACE_CROP_PADDING)
                    ey1 = max(0.0, ry1 - FACE_CROP_PADDING)
                    ex2 = min(float(img_w), rx2 + FACE_CROP_PADDING)
                    ey2 = min(float(img_h), ry2 + FACE_CROP_PADDING)
                    face_boxes.append((ex1, ey1, ex2, ey2))
            except Exception as e:
                logger.debug(
                    "Failed to parse rejected_faces_json on %s: %s", kept_path, e,
                )

        max_side = min(img_w, img_h) // 2
        if max_side < min_crop_px:
            skipped_too_small += 1
            continue

        accepted: Image.Image | None = None
        for _ in range(10):
            side = int(rng.integers(low=min_crop_px, high=max_side + 1))
            if side > img_w or side > img_h:
                continue
            cx1 = int(rng.integers(low=0, high=img_w - side + 1))
            cy1 = int(rng.integers(low=0, high=img_h - side + 1))
            cx2 = cx1 + side
            cy2 = cy1 + side
            if not _crop_overlaps_face(cx1, cy1, cx2, cy2, face_boxes, 0.10):
                accepted = img.crop((cx1, cy1, cx2, cy2)).resize(
                    (crop_size, crop_size), Image.BILINEAR,
                )
                break

        if accepted is not None:
            out.append(accepted)
        else:
            skipped_no_placement += 1

    if len(out) < n:
        logger.warning(
            "Synthetic none sampling collected %d/%d crops "
            "(skipped: %d unreadable, %d too-small, %d no-placement)",
            len(out), n,
            skipped_unreadable, skipped_too_small, skipped_no_placement,
        )
    return out


def mixup_batch(
    x: torch.Tensor,
    y_targets: torch.Tensor,
    y_labels: torch.Tensor,
    alpha: float = MIXUP_ALPHA,
    good_alpha: float = MIXUP_ALPHA_GOOD_PAIR,
    good_idx: int = LABEL_TO_IDX["good"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """MixUp with per-pair λ: Beta(good_alpha) when both partners are Good,
    Beta(alpha) otherwise. ``y_labels`` carries the original integer labels
    (before smoothing) and is used only to select the Beta distribution.
    """
    n = x.size(0)
    idx = torch.randperm(n, device=x.device)
    is_good_pair = (y_labels == good_idx) & (y_labels[idx] == good_idx)
    is_good_np = is_good_pair.detach().cpu().numpy()
    lam_default = np.random.beta(alpha, alpha, size=n)
    lam_good = np.random.beta(good_alpha, good_alpha, size=n)
    lam = np.where(is_good_np, lam_good, lam_default).astype(np.float32)
    lam_t = torch.from_numpy(lam).to(x.device)
    lam_x = lam_t.view(n, 1, 1, 1)
    lam_y = lam_t.view(n, 1)
    x_mix = lam_x * x + (1.0 - lam_x) * x[idx]
    y_mix = lam_y * y_targets + (1.0 - lam_y) * y_targets[idx]
    return x_mix, y_mix


def _build_model(arch: str = "mobilenet_v3_small") -> nn.Module:
    if arch == "mobilenet_v3_small":
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        )
        in_features = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Linear(in_features, N_CLASSES)
        return backbone
    if arch == "efficientnet_b0":
        return timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=N_CLASSES,
        )
    raise ValueError(f"Unknown arch {arch!r} (choices: {ARCH_CHOICES})")


def _apply_phase(model: nn.Module, phase: int, arch: str = "mobilenet_v3_small") -> None:
    """Freeze/unfreeze parameters according to the training phase."""
    if phase == 1:
        for name, param in model.named_parameters():
            param.requires_grad = "classifier" in name
    elif phase == 2:
        if arch == "mobilenet_v3_small":
            for name, param in model.named_parameters():
                param.requires_grad = (
                    "features.12" in name
                    or "features.13" in name
                    or "classifier" in name
                )
        elif arch == "efficientnet_b0":
            for name, param in model.named_parameters():
                param.requires_grad = (
                    name.startswith("blocks.6")
                    or name.startswith("classifier")
                )
        else:
            raise ValueError(f"Unknown arch {arch!r}")
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
    smoothing = TRAIN_LABEL_SMOOTHING
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_onehot = F.one_hot(y, num_classes=N_CLASSES).float()
        y_smooth = (1.0 - smoothing) * y_onehot + smoothing / N_CLASSES
        x_mix, y_mix = mixup_batch(x, y_smooth, y, alpha=MIXUP_ALPHA)
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
) -> tuple[float, float, list[float], list[float], list[float], list[float]]:
    model.eval()
    total_loss = 0.0
    n_seen = 0
    all_true: list[int] = []
    all_pred: list[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        n_seen += x.size(0)
        pred = logits.argmax(dim=1)
        all_true.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
    val_loss = total_loss / max(n_seen, 1)
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    val_acc = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    per_class_acc: list[float] = []
    for c in range(N_CLASSES):
        mask = y_true == c
        n = int(mask.sum())
        per_class_acc.append(
            float((y_pred[mask] == c).sum()) / n if n > 0 else float("nan"),
        )
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(N_CLASSES)), zero_division=0,
    )
    return (
        val_loss, val_acc, per_class_acc,
        [float(v) for v in f1], [float(v) for v in prec],
        [float(v) for v in rec],
    )


@torch.no_grad()
def _collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        all_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        all_true.extend(y.cpu().numpy().tolist())
    return np.array(all_true), np.array(all_pred)


def _print_validation_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    labels = list(range(N_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    print("\nValidation confusion matrix (rows=true, cols=pred):")
    col_header = "            " + " ".join(f"{IDX_TO_LABEL[c]:>6}" for c in labels)
    print(col_header)
    for r in labels:
        row_str = " ".join(f"{cm[r, c]:>6d}" for c in labels)
        print(f"{IDX_TO_LABEL[r]:>12} {row_str}")
    print("\nPer-class metrics (validation):")
    print(f"{'class':>10} {'prec':>8} {'recall':>8} {'f1':>8} {'support':>8}")
    for c in labels:
        print(
            f"{IDX_TO_LABEL[c]:>10} {prec[c]:>8.3f} {rec[c]:>8.3f} "
            f"{f1[c]:>8.3f} {int(support[c]):>8d}"
        )
    n = int(support.sum())
    overall_acc = float((np.array(y_true) == np.array(y_pred)).sum()) / max(n, 1)
    print(f"\nOverall validation accuracy: {overall_acc:.3f} ({n} samples)")


def _crop_for_row(row: pd.Series) -> Image.Image | None:
    kept = row.get("kept_path")
    if not isinstance(kept, str) or not kept:
        return None
    img_path = Path(kept)
    if not img_path.exists():
        return None
    try:
        img = Image.open(img_path).convert("RGB")
        crop = extract_face_crop_from_image(
            img,
            row["face_x1"], row["face_y1"], row["face_x2"], row["face_y2"],
            FACE_CROP_PADDING,
            kps=parse_kps(row.get("kps")),
        )
    except Exception as e:
        logger.warning("Failed to crop %s: %s", img_path, e)
        return None
    return crop.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)


def _run_inference_pass(
    model: nn.Module, crops: list[Image.Image], transform,
    device: torch.device, batch_size: int,
) -> np.ndarray:
    ds = InMemoryCropDataset(crops, transform=transform)
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
        description="Train face quality classifier from the global labels store, "
                    "then optionally score every row in a results.parquet.",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Run YAML config. When provided, --results defaults "
                             "to {output_dir}/results.parquet.")
    parser.add_argument("--labels-store", type=Path, default=DEFAULT_LABELS_STORE,
                        help="Path to the global face labels store (list JSON).")
    parser.add_argument("--results", type=Path, default=None,
                        help="Optional results.parquet to score after training. "
                             "Not used for training data loading.")
    parser.add_argument("--labels-json", type=Path, default=None,
                        help="(deprecated) Legacy per-run labels.json — ignored "
                             "when --labels-store is the active path.")
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_FACE_QUALITY_MODEL.parent)
    parser.add_argument("--output", type=Path, default=None,
                        help="Best-checkpoint path. Defaults to "
                             "{output-dir}/best_model.pt when omitted.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tta-passes", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--synthetic-none-ratio", type=float, default=0.25,
        help="Fraction of training 'none' count to add as synthetic non-face "
             "background crops sampled from --results keepers. 0 disables.",
    )
    parser.add_argument(
        "--sampler-boost-good", type=float, default=SAMPLER_BOOST["good"],
        help=f"WeightedRandomSampler boost multiplier for the 'good' class "
             f"(default: {SAMPLER_BOOST['good']})",
    )
    parser.add_argument(
        "--arch", default="mobilenet_v3_small", choices=list(ARCH_CHOICES),
        help="Backbone architecture. 'mobilenet_v3_small' (default) is the "
             "original lightweight model; 'efficientnet_b0' uses timm's "
             "ImageNet-pretrained EfficientNet-B0 with the same 128x128 input.",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    if args.labels_json is not None:
        logger.warning(
            "--labels-json is deprecated and ignored; training data is now read "
            "from --labels-store (%s).", args.labels_store,
        )

    if args.config is not None:
        cfg = RunConfig.from_yaml(args.config)
        if args.results is None:
            args.results = cfg.output_dir / "results.parquet"

    codename = generate_codename()
    logger.info("Codename: %s", codename)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    items = _load_labels_store(args.labels_store)
    if len(items) < N_CLASSES * 2:
        raise SystemExit("Not enough labeled rows for stratified train/val split.")
    item_labels = [lbl for _, lbl, _ in items]
    logger.info(
        "Loaded %d labeled crops from %s; class counts: %s",
        len(items), args.labels_store, _format_counts(item_labels),
    )

    train_items = [(p, l) for p, l, is_val in items if not is_val]
    val_items = [(p, l) for p, l, is_val in items if is_val]
    train_labels = [lbl for _, lbl in train_items]
    val_labels = [lbl for _, lbl in val_items]
    print(
        f"Val set: {len(val_items)} fixed entries (is_val=true)",
        flush=True,
    )
    print(
        f"Train set: {len(train_items)} entries "
        f"(is_val=false, before synthetic augmentation)",
        flush=True,
    )
    logger.info("Train class counts: %s", _format_counts(train_labels))
    logger.info("Val   class counts: %s", _format_counts(val_labels))

    n_synth_target = 0
    synth_rng: np.random.Generator | None = None
    if args.synthetic_none_ratio > 0:
        if args.results is None:
            raise SystemExit(
                "--results is required when --synthetic-none-ratio > 0 "
                "(synthetic crops are sampled from keeper JPEGs).",
            )
        none_idx = LABEL_TO_IDX["none"]
        train_none_count = sum(1 for y in train_labels if y == none_idx)
        n_synth_target = int(train_none_count * args.synthetic_none_ratio)
        if n_synth_target > 0:
            synth_rng = np.random.default_rng(args.seed)
            logger.info(
                "Per-epoch synthetic none resampling enabled: target n=%d "
                "(ratio=%.3f, real none train=%d)",
                n_synth_target, args.synthetic_none_ratio, train_none_count,
            )

    train_transform = _build_train_transform()
    val_transform = _build_val_transform()

    train_ds = FaceCropDataset(train_items, train_transform)
    val_ds = FaceCropDataset(val_items, val_transform)

    SAMPLER_BOOST["good"] = args.sampler_boost_good
    logger.info("Sampler boost: %s", SAMPLER_BOOST)

    def _rebuild_train_loader() -> DataLoader:
        labels = train_ds.current_labels()
        counts = [0] * N_CLASSES
        for y in labels:
            counts[y] += 1
        weights = [
            SAMPLER_BOOST[IDX_TO_LABEL[y]] / counts[y]
            if counts[y] > 0 else 0.0
            for y in labels
        ]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(labels),
            replacement=True,
        )
        return DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0,
        )

    train_loader = _rebuild_train_loader()
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    model = _build_model(args.arch).to(device)
    _apply_phase(model, 1, args.arch)
    optimizer, scheduler = _make_optimizer(model, args.lr, args.epochs, start_epoch=0)
    val_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    phase1_end = args.epochs // 2
    phase2_start = phase1_end + 1
    best_p_good = float("-inf")
    best_epoch = 0
    best_val_metrics: dict[str, float | int] = {}
    best_path = _resolve_output_path(args.output, args.output_dir, codename)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Best checkpoint will be saved to %s (selecting by p_good_max)",
        best_path,
    )
    log_rows: list[dict] = []

    header = (
        f"{'epoch':>5} {'ph':>2} {'lr':>9} "
        f"{'tr_loss':>9} {'val_loss':>9} {'val_acc':>8} "
        f"{'p_good':>8} {'f1_good':>8} "
        f"| per-class acc (n/b/o/g)"
    )
    print(header)
    print("-" * len(header))

    good_idx = LABEL_TO_IDX["good"]

    def _resample_synth(epoch: int) -> None:
        nonlocal train_loader
        if n_synth_target <= 0 or synth_rng is None:
            return
        crops = sample_synthetic_none_crops(
            args.results, n_synth_target, synth_rng,
        )
        train_ds.replace_synthetic_none(crops)
        train_loader = _rebuild_train_loader()
        logger.debug(
            "Epoch %d: resampled %d synthetic none crops", epoch, len(crops),
        )

    def _run_epoch(epoch: int, phase: int) -> dict[str, float]:
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        (val_loss, val_acc, per_class_acc, per_class_f1,
         per_class_prec, per_class_rec) = evaluate(
            model, val_loader, device, val_criterion,
        )
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        pc_str = " ".join(
            f"{IDX_TO_LABEL[c][0]}={per_class_acc[c]:.2f}"
            for c in range(N_CLASSES)
        )
        good_f1 = per_class_f1[good_idx]
        good_prec = per_class_prec[good_idx]
        good_rec = per_class_rec[good_idx]
        marker = "  [best]" if phase == 2 and good_prec > best_p_good else ""
        print(
            f"{epoch:>5d} {phase:>2d} {cur_lr:>9.2e} "
            f"{tr_loss:>9.4f} {val_loss:>9.4f} {val_acc:>8.3f} "
            f"{good_prec:>8.3f} {good_f1:>8.3f} "
            f"| {pc_str}{marker}",
            flush=True,
        )
        log_rows.append({
            "epoch": epoch,
            "phase": phase,
            "lr": cur_lr,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "p_good": good_prec,
            "f1_good": good_f1,
            **{f"acc_{IDX_TO_LABEL[c]}": per_class_acc[c] for c in range(N_CLASSES)},
            **{f"f1_{IDX_TO_LABEL[c]}": per_class_f1[c] for c in range(N_CLASSES)},
            **{f"p_{IDX_TO_LABEL[c]}": per_class_prec[c] for c in range(N_CLASSES)},
        })
        return {
            "tr_loss": tr_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "good_f1": good_f1,
            "good_precision": good_prec,
            "good_recall": good_rec,
            "macro_f1": float(np.mean(per_class_f1)),
        }

    logger.info("=== Phase 1: training head only (epochs 1-%d) ===", phase1_end)
    for epoch in range(1, phase1_end + 1):
        _resample_synth(epoch)
        _run_epoch(epoch, phase=1)

    logger.info(
        "=== Phase 2: unfreezing last conv block (epochs %d-%d) ===",
        phase2_start, args.epochs,
    )
    _apply_phase(model, 2, args.arch)
    optimizer, scheduler = _make_optimizer(
        model, args.lr, args.epochs, start_epoch=phase1_end,
    )
    best_p_good = float("-inf")
    best_epoch = 0
    patience_left = args.patience

    for epoch in range(phase2_start, args.epochs + 1):
        _resample_synth(epoch)
        epoch_metrics = _run_epoch(epoch, phase=2)
        good_prec = epoch_metrics["good_precision"]
        if good_prec > best_p_good:
            best_p_good = good_prec
            best_epoch = epoch
            best_val_metrics = {"epoch": epoch, **epoch_metrics}
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch,
                 "val_loss": epoch_metrics["val_loss"], "p_good": good_prec,
                 "arch": args.arch, "codename": codename,
                 "select_by": "p_good"},
                best_path,
            )
            logger.info(
                "New best @ epoch %d: p_good=%.4f (val_loss=%.4f)",
                epoch, good_prec, epoch_metrics["val_loss"],
            )
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(
                    "Early stop at epoch %d (best p_good=%.4f @ epoch %d)",
                    epoch, best_p_good, best_epoch,
                )
                break

    pd.DataFrame(log_rows).to_csv(args.output_dir / "training_log.csv", index=False)
    logger.info(
        "Wrote training_log.csv (%d epochs, best p_good=%.4f @ epoch %d)",
        len(log_rows), best_p_good, best_epoch,
    )

    sidecar_path = best_path.with_suffix(".json")
    sidecar = {
        "codename": codename,
        "version_tag": _parse_version_tag(best_path),
        "arch": args.arch,
        "input_size": DISPLAY_SIZE,
        "num_classes": N_CLASSES,
        "label_names": list(FACE_QUALITY_LABELS),
        "checkpoint_save_policy": "p_good_max",
        "hyperparams": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "early_stop_patience": args.patience,
            "mixup_alpha": MIXUP_ALPHA,
            "good_good_mixup_beta": MIXUP_ALPHA_GOOD_PAIR,
            "sampler_boost_good": args.sampler_boost_good,
            "synthetic_none_ratio": args.synthetic_none_ratio,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
        },
        "label_counts": {
            **_format_counts(item_labels),
            "total": len(item_labels),
        },
        "val_metrics_at_save": {
            "epoch": int(best_val_metrics.get("epoch", best_epoch)),
            "val_loss": float(best_val_metrics.get("val_loss", float("nan"))),
            "val_acc": float(best_val_metrics.get("val_acc", float("nan"))),
            "good_f1": float(best_val_metrics.get("good_f1", float("nan"))),
            "good_precision": float(best_val_metrics.get("good_precision", best_p_good)),
            "good_recall": float(best_val_metrics.get("good_recall", float("nan"))),
            "macro_f1": float(best_val_metrics.get("macro_f1", float("nan"))),
        },
        "trained_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "notes": "",
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    logger.info("Wrote sidecar metadata to %s", sidecar_path)

    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    logger.info("Reloaded best model from epoch %d", state["epoch"])

    y_true, y_pred = _collect_predictions(model, val_loader, device)
    _print_validation_report(y_true, y_pred)
    print(
        f"\nBest checkpoint: epoch {state['epoch']}, "
        f"val_loss={float(state['val_loss']):.4f}, "
        f"p_good={float(state['p_good']):.4f} (selected by p_good_max)",
    )

    if args.results is None:
        logger.info("--results not provided; skipping inference pass on corpus.")
        return

    df = pd.read_parquet(args.results).reset_index(drop=True)
    logger.info(
        "Loaded %d keeper rows from %s for inference", len(df), args.results,
    )

    logger.info("Extracting %d face crops for inference...", len(df))
    crops: list[Image.Image | None] = []
    for i in tqdm(range(len(df)), desc="crops", unit="img"):
        crops.append(_crop_for_row(df.iloc[i]))
    missing = sum(1 for c in crops if c is None)
    if missing:
        logger.warning("%d/%d rows had unreadable crops", missing, len(crops))

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

    inference_path = args.output_dir / "inference_scores.csv"
    out.to_csv(inference_path, index=False)
    logger.info("Wrote %s (%d rows)", inference_path, len(out))

    results_inference_path = args.results.parent / "classifier" / "inference_scores.csv"
    results_inference_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(results_inference_path, index=False)
    logger.info("Wrote %s (%d rows)", results_inference_path, len(out))

    nonempty_preds = [p for p in pred_labels if p]
    pred_counts = pd.Series(nonempty_preds).value_counts().to_dict()
    finite_conf = [c for c in pred_conf if c == c]
    mean_conf = float(np.mean(finite_conf)) if finite_conf else float("nan")
    logger.info("Prediction counts: %s", pred_counts)
    logger.info("Mean prediction confidence (TTA): %.3f", mean_conf)


if __name__ == "__main__":
    main()
