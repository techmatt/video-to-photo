"""Copy files listed in flagged.json to an export directory."""

import json
import logging
import shutil
from argparse import ArgumentParser
from pathlib import Path

logger = logging.getLogger(__name__)


def _unique_dst(dst_dir: Path, name: str) -> Path:
    candidate = dst_dir / name
    if not candidate.exists():
        return candidate
    stem = Path(name).stem
    suffix = Path(name).suffix
    i = 2
    while True:
        candidate = dst_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def main() -> None:
    parser = ArgumentParser(
        description="Copy files listed in flagged.json to an export directory.",
    )
    parser.add_argument("--flagged-json", type=Path, default=Path("flagged.json"),
                        help="Path to flagged.json downloaded from the photo viewer.")
    parser.add_argument("--output-dir", type=Path, default=Path("export"),
                        help="Directory to copy flagged files into.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    payload = json.loads(args.flagged_json.read_text(encoding="utf-8"))
    paths = payload.get("export_paths", [])
    logger.info("Loaded %d export paths from %s", len(paths), args.flagged_json)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    for raw in paths:
        src = Path(raw)
        if not src.exists():
            logger.warning("Source missing, skipping: %s", src)
            missing += 1
            continue
        dst = _unique_dst(args.output_dir, src.name)
        shutil.copy2(src, dst)
        logger.info("Copied %s -> %s", src, dst)
        copied += 1

    logger.info(
        "%d files copied to %s/  (%d missing)", copied, args.output_dir, missing,
    )


if __name__ == "__main__":
    main()
