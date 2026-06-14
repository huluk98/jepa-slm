from jepa_slm.objectives import ema_tau, linear_ramp


def test_linear_ramp_hits_peak() -> None:
    assert linear_ramp(0, 100, 0.25) == 0
    assert linear_ramp(50, 100, 0.25) == 0.125
    assert linear_ramp(100, 100, 0.25) == 0.25
    assert linear_ramp(200, 100, 0.25) == 0.25


def test_ema_tau_schedule_bounds() -> None:
    assert ema_tau(0, 100, 0.99, 0.999) == 0.99
    assert ema_tau(100, 100, 0.99, 0.999) == 0.999
    assert ema_tau(200, 100, 0.99, 0.999) == 0.999
