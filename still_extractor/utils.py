"""Small pure utility functions shared across the pipeline."""

import json
import math
from pathlib import Path


def safe_float(val) -> float | None:
    """Parse val to a float. Return None on type/value error or NaN."""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def to_fwd_slash(path: "str | Path") -> str:
    """Convert path separators to forward slashes (for HTML/JSON)."""
    return str(path).replace("\\", "/")


def parse_kps(kps_val) -> list[list[float]] | None:
    """Parse JSON kps string or already-list value to [[x,y], ...], or None on failure."""
    if kps_val is None:
        return None
    try:
        result = json.loads(kps_val) if isinstance(kps_val, str) else kps_val
        return result if isinstance(result, list) else None
    except Exception:
        return None
