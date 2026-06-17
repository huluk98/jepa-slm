"""Distributed training entrypoint for JEPA-SLM."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_training_config
from .trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JEPA-SLM distributed training entrypoint.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_training_config(args.config))


if __name__ == "__main__":
    main()
