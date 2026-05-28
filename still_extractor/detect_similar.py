"""Detect near-duplicate frames via CLIP ViT-B/32 embeddings.

Computes a 512-d CLIP embedding for every keeper frame in `results.parquet`,
caches embeddings to `{output_dir}/embeddings.npy` + `embeddings_index.json`
so re-runs only embed new frames, then groups frames whose pairwise cosine
similarity exceeds `similarity_threshold` (default 0.95) into connected
components. Within each component the frame with the highest `quality_score`
(falling back to largest file size) becomes the representative.

Writes two new columns back to `results.parquet`:
- `similarity_group_id`: Int64 (nullable), null for singletons.
- `is_group_representative`: bool, true for singletons and representatives.
"""

import json
import logging
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

import open_clip

from still_extractor.inventory import RunConfig
from still_extractor.utils import to_fwd_slash

logger = logging.getLogger(__name__)


CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
EMBEDDING_DIM = 512
BATCH_SIZE = 32
DEFAULT_SIMILARITY_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Embedding cache (paths -> N x 512 float32 matrix)
# ---------------------------------------------------------------------------

def _normalize_path_key(path: str | Path) -> str:
    """Stable cache key derived from the absolute path with forward slashes."""
    return to_fwd_slash(Path(path).resolve())


def _load_cache(
    npy_path: Path, index_path: Path,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Return (path -> embedding) and the ordered list of paths.

    Empty if either file is absent or malformed.
    """
    if not npy_path.exists() or not index_path.exists():
        return {}, []
    try:
        arr = np.load(npy_path)
    except Exception as e:
        logger.warning("Failed to load %s (%s); rebuilding cache", npy_path, e)
        return {}, []
    try:
        paths = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load %s (%s); rebuilding cache", index_path, e)
        return {}, []
    if not isinstance(paths, list) or arr.ndim != 2 or len(paths) != arr.shape[0]:
        logger.warning(
            "Cache inconsistent (%s rows vs %s paths); rebuilding",
            arr.shape[0] if arr.ndim == 2 else "?", len(paths) if isinstance(paths, list) else "?",
        )
        return {}, []
    return {p: arr[i] for i, p in enumerate(paths)}, list(paths)


def _save_cache(
    npy_path: Path, index_path: Path,
    paths: list[str], embeddings: np.ndarray,
) -> None:
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, embeddings.astype(np.float32))
    index_path.write_text(
        json.dumps(paths, separators=(",", ":")), encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLIP embedding
# ---------------------------------------------------------------------------

def _load_clip_model(
    device: torch.device,
) -> tuple[torch.nn.Module, "callable"]:
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device,
    )
    model.eval()
    return model, preprocess


def _compute_embeddings_for_paths(
    paths: list[str],
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    """Embed each path; skip ones that can't be opened. Return (ok_paths, embeddings)."""
    ok_paths: list[str] = []
    chunks: list[np.ndarray] = []

    pending_paths: list[str] = []
    pending_tensors: list[torch.Tensor] = []

    def flush() -> None:
        if not pending_tensors:
            return
        batch = torch.stack(pending_tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch).float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        chunks.append(feats.cpu().numpy().astype(np.float32))
        ok_paths.extend(pending_paths)
        pending_paths.clear()
        pending_tensors.clear()

    for p in tqdm(paths, desc="CLIP embed", unit="img"):
        try:
            with Image.open(p) as img:
                tensor = preprocess(img.convert("RGB"))
        except Exception as e:
            logger.warning("Failed to load %s (%s); skipping", p, e)
            continue
        pending_paths.append(p)
        pending_tensors.append(tensor)
        if len(pending_tensors) >= BATCH_SIZE:
            flush()
    flush()

    if not chunks:
        return [], np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    return ok_paths, np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Similarity grouping
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _group_by_similarity(
    embeddings: np.ndarray, threshold: float,
) -> list[int]:
    """Return component ids per row (size N), one int per embedding row."""
    n = embeddings.shape[0]
    if n == 0:
        return []
    # N x N cosine similarity (embeddings already unit-normalized).
    # NOTE: O(N^2) memory; ~4MB at float32 for N=1039. For N>10K consider chunking.
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -1.0)

    uf = UnionFind(n)
    # Iterate the upper triangle only (i < j).
    iu, ju = np.where(np.triu(sim > threshold, k=1))
    for i, j in zip(iu.tolist(), ju.tolist()):
        uf.union(i, j)

    return [uf.find(i) for i in range(n)]


# ---------------------------------------------------------------------------
# Representative selection
# ---------------------------------------------------------------------------

def _representative_score(
    df: pd.DataFrame, indices: list[int],
) -> int:
    """Pick row index of best representative for one group.

    Prefer the row with the highest non-null `quality_score`. If all rows
    have null `quality_score`, fall back to the largest file size on disk.
    """
    if "quality_score" in df.columns:
        scores = df.loc[indices, "quality_score"]
        valid = scores.dropna()
        if not valid.empty:
            return int(valid.idxmax())

    def file_size(idx: int) -> int:
        path = df.at[idx, "kept_path"]
        try:
            return Path(path).stat().st_size
        except (OSError, TypeError):
            return -1

    return max(indices, key=file_size)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _read_similarity_threshold(config_path: Path) -> float:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Failed to re-read %s for similarity_threshold (%s); using default", config_path, e)
        return DEFAULT_SIMILARITY_THRESHOLD
    val = data.get("similarity_threshold")
    if val is None:
        return DEFAULT_SIMILARITY_THRESHOLD
    try:
        return float(val)
    except (TypeError, ValueError):
        logger.warning("Invalid similarity_threshold=%r in %s; using default", val, config_path)
        return DEFAULT_SIMILARITY_THRESHOLD


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = ArgumentParser(
        description="Compute CLIP embeddings, group near-duplicate frames, and write similarity columns to results.parquet.",
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Run YAML config; output_dir provides parquet + cache paths.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    cfg = RunConfig.from_yaml(args.config)
    threshold = _read_similarity_threshold(args.config)
    results_path = cfg.output_dir / "results.parquet"
    npy_path = cfg.output_dir / "embeddings.npy"
    index_path = cfg.output_dir / "embeddings_index.json"

    df = pd.read_parquet(results_path)
    logger.info("Loaded %d rows from %s", len(df), results_path)
    if "kept_path" not in df.columns:
        raise SystemExit("results.parquet is missing 'kept_path'")

    # Resolve image paths in row order; skip rows whose file is missing.
    row_path_keys: list[str | None] = []
    missing_on_disk = 0
    for _, row in df.iterrows():
        kept = row.get("kept_path")
        if not isinstance(kept, str) or not kept:
            row_path_keys.append(None)
            continue
        p = Path(kept)
        if not p.exists():
            logger.warning("Keeper missing on disk, skipping: %s", p)
            row_path_keys.append(None)
            missing_on_disk += 1
            continue
        row_path_keys.append(_normalize_path_key(p))

    cache_map, cached_paths = _load_cache(npy_path, index_path)
    cached_before = len(cached_paths)

    # Determine which paths need embedding (preserve first occurrence order).
    new_paths: list[str] = []
    seen_new: set[str] = set()
    for key in row_path_keys:
        if key is None or key in cache_map or key in seen_new:
            continue
        new_paths.append(key)
        seen_new.add(key)

    if new_paths:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Computing CLIP embeddings for %d new path(s) on %s", len(new_paths), device,
        )
        model, preprocess = _load_clip_model(device)
        ok_new_paths, new_embs = _compute_embeddings_for_paths(
            new_paths, model, preprocess, device,
        )
        if ok_new_paths:
            for i, p in enumerate(ok_new_paths):
                cache_map[p] = new_embs[i]
            cached_paths.extend(ok_new_paths)
            cached_embs = np.stack([cache_map[p] for p in cached_paths], axis=0)
            _save_cache(npy_path, index_path, cached_paths, cached_embs)
            logger.info("Cache now has %d embeddings (added %d)", len(cached_paths), len(ok_new_paths))
        added_new = len(ok_new_paths)
    else:
        logger.info("All embeddings already cached; nothing new to compute.")
        added_new = 0

    # Build per-row embedding matrix for the rows we can group.
    embeddable_idx: list[int] = []  # df row index
    emb_rows: list[np.ndarray] = []
    for df_idx, key in enumerate(row_path_keys):
        if key is None:
            continue
        vec = cache_map.get(key)
        if vec is None:
            continue
        embeddable_idx.append(df_idx)
        emb_rows.append(vec)

    if not emb_rows:
        logger.warning("No embeddable rows; nothing to group.")
        return

    embeddings = np.stack(emb_rows, axis=0).astype(np.float32)
    # Ensure unit-normalized (cache could be stale from a prior, un-normalized version).
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    embeddings = embeddings / norms

    logger.info(
        "Grouping %d embeddings with cosine threshold %.3f", embeddings.shape[0], threshold,
    )
    components = _group_by_similarity(embeddings, threshold)

    # Map raw component id -> sequential group id, but only for size>=2.
    comp_to_indices: dict[int, list[int]] = {}
    for local_idx, comp_id in enumerate(components):
        comp_to_indices.setdefault(comp_id, []).append(embeddable_idx[local_idx])

    sequential_id = 0
    group_id_per_row: dict[int, int | None] = {}
    is_rep_per_row: dict[int, bool] = {}
    size_counter: Counter[int] = Counter()

    for indices in comp_to_indices.values():
        if len(indices) == 1:
            df_idx = indices[0]
            group_id_per_row[df_idx] = None
            is_rep_per_row[df_idx] = True
            size_counter[1] += 1
            continue
        sequential_id += 1
        size_counter[len(indices)] += 1
        rep_idx = _representative_score(df, indices)
        for df_idx in indices:
            group_id_per_row[df_idx] = sequential_id
            is_rep_per_row[df_idx] = (df_idx == rep_idx)

    # Default values for rows we couldn't embed (missing on disk, etc.).
    group_id_col: list[int | None] = []
    is_rep_col: list[bool] = []
    for df_idx in range(len(df)):
        if df_idx in is_rep_per_row:
            group_id_col.append(group_id_per_row[df_idx])
            is_rep_col.append(is_rep_per_row[df_idx])
        else:
            group_id_col.append(None)
            is_rep_col.append(True)

    df["similarity_group_id"] = pd.array(group_id_col, dtype="Int64")
    df["is_group_representative"] = pd.array(is_rep_col, dtype="bool")
    df.to_parquet(results_path, index=False)
    logger.info(
        "Wrote similarity_group_id + is_group_representative to %s", results_path,
    )

    # Summary
    total = embeddings.shape[0]
    multi_groups = {size: cnt for size, cnt in size_counter.items() if size >= 2}
    groups_total = sum(multi_groups.values())
    size2 = multi_groups.get(2, 0)
    size3 = multi_groups.get(3, 0)
    size4plus = sum(cnt for size, cnt in multi_groups.items() if size >= 4)
    non_rep = sum(1 for v in is_rep_per_row.values() if not v)
    rep = total - non_rep

    print(f"Embeddings: {total} total ({cached_before} cached, {added_new} new)")
    print(f"Similarity threshold: {threshold:.2f}")
    print(
        f"Groups found: {groups_total} "
        f"(size 2: {size2}, size 3: {size3}, size 4+: {size4plus})"
    )
    print(f"Frames deduplicated: {non_rep} (non-representative)")
    print(f"Representative frames: {rep} / {total}")
    if missing_on_disk:
        print(f"Skipped (missing on disk): {missing_on_disk}")


if __name__ == "__main__":
    main()
