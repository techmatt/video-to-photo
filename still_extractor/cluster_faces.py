"""Cluster detected face embeddings into identities."""

import logging
from argparse import ArgumentParser

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(description="Cluster detected face embeddings into identities.")
    args = parser.parse_args()


if __name__ == "__main__":
    main()
