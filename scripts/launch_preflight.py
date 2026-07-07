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


def _metadata_complete(output_dir: str) -> bool:
    """True when the corpus dir's metadata.json exists and says complete."""

    if not output_dir:
        return False
    metadata_path = Path(output_dir) / "metadata.json"
    if not metadata_path.exists():
        return False
    import json

    try:
        return json.loads(metadata_path.read_text(encoding="utf-8")).get("complete") is True
    except Exception:  # noqa: BLE001 - unreadable metadata = not complete
        return False


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: launch_preflight.py <config.yaml>")
    config = load_training_config(sys.argv[1])
    dataset = config.data.dataset or ""
    local = _is_local(dataset)
    glob_count = len(glob.glob(dataset)) if local else 0
    output_dir = str(Path(dataset).parent) if local else ""
    eval_dataset = config.data.eval_dataset or ""
    stop_file = config.runtime.stop_file
    devices = config.runtime.cuda_visible_devices

    # shlex.quote(str(...)) on EVERY value: these lines are eval'd by the
    # launcher, so an unquoted YAML surprise must never become a shell token.
    print(f"DATASET={shlex.quote(dataset)}")
    print(f"IS_LOCAL={1 if local else 0}")
    print(f"GLOB_COUNT={glob_count}")
    print(f"OUTPUT_DIR={shlex.quote(output_dir)}")
    print(f"SUBSET={shlex.quote(config.data.subset or 'sample-10BT')}")
    print(f"EVAL_DATASET={shlex.quote(eval_dataset)}")
    print(f"MAX_STEPS={shlex.quote(str(config.runtime.max_steps))}")
    print(f"SEQUENCE_PACKING={1 if config.batching.sequence_packing else 0}")
    print(f"STOP_FILE={shlex.quote('' if stop_file is None else str(stop_file))}")
    print(f"CFG_DEVICES={shlex.quote('' if devices is None else str(devices))}")
    print(f"METADATA_COMPLETE={1 if _metadata_complete(output_dir) else 0}")


if __name__ == "__main__":
    main()
