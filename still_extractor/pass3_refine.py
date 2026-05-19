"""Pass 3: refine top-scoring candidates with face and identity signals."""

import logging
from argparse import ArgumentParser

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(description="Refine top-scoring candidates with face/identity signals.")
    args = parser.parse_args()


if __name__ == "__main__":
    main()
