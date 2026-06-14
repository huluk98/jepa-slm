#!/usr/bin/env python3
"""Check that the local machine is ready for an 8x H20 training run."""

from __future__ import annotations

import importlib.util
import os
import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"torch import failed: {exc}")
        return 1

    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    if not torch.cuda.is_available():
        print("No CUDA device is visible. Run this on the H20 training node.")
        return 1

    count = torch.cuda.device_count()
    print(f"GPU count: {count}")
    if count < 8:
        print("Expected 8 visible GPUs for the H20 launch config.")
        return 1

    for idx in range(count):
        props = torch.cuda.get_device_properties(idx)
        memory_gb = props.total_memory / 1024**3
        print(
            f"[{idx}] {props.name} | capability {props.major}.{props.minor} | "
            f"{memory_gb:.1f} GiB"
        )

    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"flash_attn installed: {importlib.util.find_spec('flash_attn') is not None}")
    print(
        "transformer_engine installed: "
        f"{importlib.util.find_spec('transformer_engine') is not None}"
    )

    if not torch.cuda.is_bf16_supported():
        print("BF16 is not reported as supported; H20 training config expects BF16.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
