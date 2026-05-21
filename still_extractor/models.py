"""Load and hold all ML models used by the pipeline.

`load_models()` is called once at pipeline startup; the returned `Models`
object is passed into every worker call. Optional models (face quality
classifier, uprighter) are loaded only if their weight file exists; the
worker treats `None` as "skip this stage".
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as tv_models
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from insightface.app import FaceAnalysis

from still_extractor.constants import FACE_QUALITY_LABELS, UPRIGHTER_LABELS

logger = logging.getLogger(__name__)


@dataclass
class Models:
    face_app: Any                       # InsightFace FaceAnalysis
    aesthetic_model: Any                # Aesthetic Predictor V2.5
    aesthetic_preprocessor: Any
    aesthetic_dtype: torch.dtype
    face_quality_model: nn.Module | None
    uprighter_model: nn.Module | None
    device: torch.device


def _build_mobilenet_v3_small(num_classes: int) -> nn.Module:
    backbone = tv_models.mobilenet_v3_small(weights=None)
    in_features = backbone.classifier[3].in_features
    backbone.classifier[3] = nn.Linear(in_features, num_classes)
    return backbone


def _load_mobilenet_state(
    model_path: Path, num_classes: int, device: torch.device,
) -> nn.Module:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model = _build_mobilenet_v3_small(num_classes)
    model.load_state_dict(state_dict)
    return model.eval().to(device)


def load_models(
    face_quality_path: Path | None,
    uprighter_path: Path | None,
    device: torch.device,
) -> Models:
    logger.info("Loading InsightFace buffalo_l...")
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    logger.info("Loading Aesthetic Predictor V2.5...")
    aesthetic_dtype = torch.float16 if device.type == "cuda" else torch.float32
    aesthetic_model, aesthetic_preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True, torch_dtype=aesthetic_dtype,
    )
    aesthetic_model = aesthetic_model.to(device).eval()

    face_quality_model: nn.Module | None = None
    if face_quality_path is not None and Path(face_quality_path).exists():
        logger.info("Loading face quality classifier from %s", face_quality_path)
        face_quality_model = _load_mobilenet_state(
            Path(face_quality_path), num_classes=len(FACE_QUALITY_LABELS), device=device,
        )
    else:
        logger.warning(
            "Face quality classifier not found at %s; classifier scoring disabled",
            face_quality_path,
        )

    uprighter_model: nn.Module | None = None
    if uprighter_path is not None and Path(uprighter_path).exists():
        logger.info("Loading uprighter model from %s", uprighter_path)
        uprighter_model = _load_mobilenet_state(
            Path(uprighter_path), num_classes=len(UPRIGHTER_LABELS), device=device,
        )
    else:
        logger.warning(
            "Uprighter model not found at %s; rotation correction disabled",
            uprighter_path,
        )

    return Models(
        face_app=face_app,
        aesthetic_model=aesthetic_model,
        aesthetic_preprocessor=aesthetic_preprocessor,
        aesthetic_dtype=aesthetic_dtype,
        face_quality_model=face_quality_model,
        uprighter_model=uprighter_model,
        device=device,
    )
