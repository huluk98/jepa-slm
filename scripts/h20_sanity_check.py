#!/usr/bin/env python3
"""Check that the local machine is ready for an H20 training run (4x or 8x).

By default this passes on any node with >=1 CUDA GPU and BF16 support. Set
EXPECT_GPUS=N to print a NOTE when fewer than N GPUs are visible, and
STRICT_GPUS=1 to make that count mismatch a hard failure (exit 1).
"""

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

    expect = int(os.environ.get("EXPECT_GPUS", "0") or 0)
    strict = os.environ.get("STRICT_GPUS", "0") == "1"
    if expect and count < expect:
        print(f"NOTE: {count} GPU(s) visible, fewer than EXPECT_GPUS={expect}.")
        if strict:
            print("STRICT_GPUS=1: failing on the GPU-count mismatch.")
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
