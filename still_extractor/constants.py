"""Single source of truth for shared constants across the pipeline."""

from pathlib import Path

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".heic", ".heif", ".tiff", ".tif", ".bmp",
})
VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".m4v",
})

FACE_QUALITY_LABELS: list[str] = ["none", "bad", "okay", "good"]
LABEL_TO_IDX: dict[str, int] = {l: i for i, l in enumerate(FACE_QUALITY_LABELS)}
FACE_QUALITY_INPUT_SIZE: int = 128
FACE_CROP_PADDING: int = 20
DEFAULT_FACE_QUALITY_MODEL: Path = Path("models/face_quality/best_model.pt")

# Top-3 faces schema. Each frame stores up to MAX_FACE_SLOTS faces ranked by
# p_good descending. face_1 is mirrored into the legacy face_* columns
# (face_x1, p_good, pred_label, etc.) — downstream code still reads those.
FACE_SLOT_COLUMNS: list[str] = [
    "x1", "y1", "x2", "y2", "det_score", "kps", "embedding",
    "p_none", "p_bad", "p_okay", "p_good", "pred_label", "pred_confidence",
]
FACE_SLOTS: list[int] = [1, 2, 3]
MAX_FACE_SLOTS: int = 3

UPRIGHTER_INPUT_SIZE: int = 224
DEFAULT_UPRIGHTER_MODEL: Path = Path("models/uprighter/best_model.pt")
UPRIGHTER_CONFIDENCE_THRESHOLD: float = 0.95
UPRIGHTER_LABELS: list[str] = ["0", "90", "180", "270"]

FACE_SHARPNESS_PADDING: int = 10
CLASSIFIER_BLEND_WEIGHT: float = 0.8

# Face rejection heuristics
# Minimum face area as fraction of frame area to pass at all
FACE_MIN_AREA_FRAC: float = 0.004
# Minimum face area fraction to be immune from edge rejection
FACE_EDGE_IMMUNE_AREA_FRAC: float = 0.025
# Edge zone: fraction of frame width/height defining the "near edge" region
FACE_EDGE_ZONE_FRAC: float = 0.10

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def card_key(video_stem: str, kept_path: "str | Path") -> str:
    """Stable join key between Python pipeline and browser label store."""
    return f"{video_stem}/{Path(kept_path).name}"
