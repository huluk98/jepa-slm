"""Objective schedules and latent-objective helper math.

The scalar schedules are dependency-free so they can be unit-tested and reused
outside the training loop. The VICReg helpers operate on torch tensors and are
imported lazily by the model.
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


def jepa_lambda_schedule(
    step: int,
    max_steps: int,
    warmup_fraction: float,
    peak: float,
    final_weight: float = 0.05,
    final_phase_fraction: float = 0.20,
) -> float:
    """Three-phase JEPA weight: ramp up, plateau, then decay for CE polish.

    - Steps ``[0, warmup)``: linear ramp ``0 -> peak``.
    - Steps ``[warmup, decay_start)``: hold ``peak``.
    - Steps ``[decay_start, max_steps]``: linear decay ``peak -> final_weight``.

    This honors the documented phase-3 "low-or-zero JEPA polish" without
    touching the simpler :func:`linear_ramp` used elsewhere.
    """
    if max_steps <= 0:
        return peak
    warmup_steps = max(1, int(max_steps * warmup_fraction))
    if step < warmup_steps:
        return peak * max(0, step) / warmup_steps

    final_phase_fraction = min(max(final_phase_fraction, 0.0), 1.0)
    decay_steps = int(max_steps * final_phase_fraction)
    decay_start = max_steps - decay_steps
    if decay_steps <= 0 or step < decay_start:
        return peak
    progress = min(1.0, (step - decay_start) / max(1, decay_steps))
    return peak + progress * (final_weight - peak)
