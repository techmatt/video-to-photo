"""Build a browsable HTML gallery of selected still frames."""

import logging
from argparse import ArgumentParser

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(description="Build a browsable HTML gallery of selected still frames.")
    args = parser.parse_args()


if __name__ == "__main__":
    main()
