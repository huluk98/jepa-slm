"""Objective formulas as lightweight reference functions.

These helpers are deliberately dependency-free. They document the intended
training math and can be mirrored in a real PyTorch training loop.
"""

from __future__ import annotations


def linear_ramp(step: int, warmup_steps: int, peak: float) -> float:
    """Return a linearly ramped auxiliary loss weight."""
    if warmup_steps <= 0:
        return peak
    return min(peak, peak * max(0, step) / warmup_steps)


def ema_tau(step: int, total_steps: int, start: float = 0.99, end: float = 0.9995) -> float:
    """Return a simple linear EMA tau schedule."""
    if total_steps <= 0:
        return end
    progress = min(1.0, max(0.0, step / total_steps))
    return start + progress * (end - start)
