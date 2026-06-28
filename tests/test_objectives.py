import pytest

from jepa_slm.objectives import ema_tau, jepa_lambda_schedule, linear_ramp


def test_linear_ramp_hits_peak() -> None:
    assert linear_ramp(0, 100, 0.25) == 0
    assert linear_ramp(50, 100, 0.25) == 0.125
    assert linear_ramp(100, 100, 0.25) == 0.25
    assert linear_ramp(200, 100, 0.25) == 0.25


def test_ema_tau_schedule_bounds() -> None:
    assert ema_tau(0, 100, 0.99, 0.999) == 0.99
    assert ema_tau(100, 100, 0.99, 0.999) == 0.999
    assert ema_tau(200, 100, 0.99, 0.999) == 0.999


def test_jepa_lambda_schedule_ramps_plateaus_then_decays() -> None:
    # Warmup over first 10% (10 steps), then plateau, then final-phase decay.
    assert jepa_lambda_schedule(0, 100, 0.1, 0.25, 0.05, 0.2) == 0.0
    assert jepa_lambda_schedule(5, 100, 0.1, 0.25, 0.05, 0.2) == 0.125
    assert jepa_lambda_schedule(10, 100, 0.1, 0.25, 0.05, 0.2) == 0.25
    # Plateau in the middle, decay only in the final phase.
    assert jepa_lambda_schedule(50, 100, 0.1, 0.25, 0.05, 0.2) == 0.25
    assert jepa_lambda_schedule(80, 100, 0.1, 0.25, 0.05, 0.2) == 0.25
    assert jepa_lambda_schedule(90, 100, 0.1, 0.25, 0.05, 0.2) < 0.25
    assert jepa_lambda_schedule(100, 100, 0.1, 0.25, 0.05, 0.2) == pytest.approx(0.05)
