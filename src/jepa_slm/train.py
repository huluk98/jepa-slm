"""Distributed training entrypoint placeholder.

This module validates the hardware/config path and records the intended launch
settings. The full model/training implementation should plug into this entrypoint
once the architecture moves from research scaffold to training code.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JEPA-SLM distributed training entrypoint.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if rank == 0:
        hardware = config.get("hardware", {})
        batching = config.get("batching", {})
        print("JEPA-SLM training scaffold")
        print(f"config: {args.config}")
        print(f"world_size: {world_size}")
        print(f"accelerator: {hardware.get('accelerator')}")
        print(f"precision: {hardware.get('precision')}")
        print(f"per_gpu_micro_batch_sequences: {batching.get('per_gpu_micro_batch_sequences')}")
        print("Replace src/jepa_slm/train.py with the full PyTorch training loop before long runs.")
    else:
        _ = local_rank


if __name__ == "__main__":
    main()
