#!/usr/bin/env python3
"""Resolve a training config's data source for run_training.sh.

Prints shell-evalable KEY=VALUE lines so the launcher can decide whether the
configured corpus is a local glob that must be cleaned first, or a streaming /
synthetic source that needs no local preparation. Intentionally torch-free so it
runs during preflight before the heavy imports.
"""

from __future__ import annotations

import glob
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jepa_slm.config import load_training_config


def _is_local(dataset: str) -> bool:
    if dataset in {"synthetic", ""}:
        return False
    return any(char in dataset for char in "*?[") or Path(dataset).exists()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: launch_preflight.py <config.yaml>")
    config = load_training_config(sys.argv[1])
    dataset = config.data.dataset or ""
    local = _is_local(dataset)
    glob_count = len(glob.glob(dataset)) if local else 0
    output_dir = str(Path(dataset).parent) if local else ""
    eval_dataset = config.data.eval_dataset or ""

    print(f"DATASET={shlex.quote(dataset)}")
    print(f"IS_LOCAL={1 if local else 0}")
    print(f"GLOB_COUNT={glob_count}")
    print(f"OUTPUT_DIR={shlex.quote(output_dir)}")
    print(f"SUBSET={shlex.quote(config.data.subset or 'sample-10BT')}")
    print(f"EVAL_DATASET={shlex.quote(eval_dataset)}")
    print(f"MAX_STEPS={config.runtime.max_steps}")
    print(f"SEQUENCE_PACKING={1 if config.batching.sequence_packing else 0}")


if __name__ == "__main__":
    main()
