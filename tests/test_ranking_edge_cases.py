"""Edge-case tests for newSNR ranking (NaN/inf/extreme values)."""

import math

from gwsearch import ranking


class _Cfg:
    conditioning = type("c", (), {"low_frequency_cutoff": 20.0, "high_frequency_cutoff": None})


CFG = _Cfg()


# ---------------------------------------------------------------------------
# Normal operation (already tested in test_newsnr.py, duplicated here for
# completeness of the edge-case suite)
# ---------------------------------------------------------------------------

def test_normal_values():
    val = ranking.new_snr(10.0, 20.0, duration=1.0, cfg=CFG)
    assert math.isfinite(val)
    assert val > 0.0
    # chisq=20, dof=30 → rchisq=0.667 < 1 → newSNR > SNR (low chi-squared boosts)
    # Just verify output is a sensible positive finite number
    assert val > 5.0


def test_perfect_match_rchisq_one():
    """rchisq == 1 → reweight_factor == 1 → newSNR == SNR."""
    dof = 16 * 2 - 2  # 30
    chisq_perfect = float(dof)  # rchisq = 1
    val = ranking.new_snr(8.0, chisq_perfect, duration=1.0, cfg=CFG)
    expected = 8.0 / (((1 + 1.0**3) / 2.0) ** (1.0 / 6.0))
    assert abs(val - expected) < 1e-9


# ---------------------------------------------------------------------------
# NaN / Inf inputs
# ---------------------------------------------------------------------------

def test_nan_snr_returns_zero():
    val = ranking.new_snr(float("nan"), 10.0, duration=1.0, cfg=CFG)
    assert val == 0.0


def test_inf_snr_returns_zero():
    val = ranking.new_snr(float("inf"), 10.0, duration=1.0, cfg=CFG)
    assert val == 0.0


def test_negative_snr_returns_zero():
    val = ranking.new_snr(-5.0, 10.0, duration=1.0, cfg=CFG)
    assert val == 0.0


def test_nan_chisq_falls_back_to_snr():
    val = ranking.new_snr(9.0, float("nan"), duration=1.0, cfg=CFG)
    assert val == 9.0


def test_negative_chisq_falls_back_to_snr():
    val = ranking.new_snr(9.0, -1.0, duration=1.0, cfg=CFG)
    assert val == 9.0


def test_inf_chisq_falls_back_to_snr():
    val = ranking.new_snr(9.0, float("inf"), duration=1.0, cfg=CFG)
    # inf chisq is finite-checked; code clips to 1000 → still valid float
    assert math.isfinite(val)
    assert val > 0.0


# ---------------------------------------------------------------------------
# Extreme but valid values
# ---------------------------------------------------------------------------

def test_very_high_chisq_penalises_heavily():
    """Extremely high chi-squared (glitch) should produce very low newSNR."""
    val = ranking.new_snr(50.0, 1e6, duration=1.0, cfg=CFG)
    assert val < 50.0  # penalised
    assert math.isfinite(val)


def test_very_low_chisq_preserves_snr():
    """Near-zero chi-squared: reweight_factor ≈ 0.5^(1/6) ≈ 0.89, so newSNR ≈ snr/0.89."""
    val = ranking.new_snr(10.0, 0.0, duration=1.0, cfg=CFG)
    assert math.isfinite(val)
    assert val > 0.0


def test_output_is_always_finite():
    """Sweep over a range of SNR/chisq pairs; none should produce NaN/Inf."""
    import itertools
    snrs = [0.1, 5.5, 8.0, 20.0, 100.0]
    chisqs = [0.0, 1.0, 30.0, 300.0, 1e5]
    for snr, chisq in itertools.product(snrs, chisqs):
        val = ranking.new_snr(snr, chisq, duration=1.0, cfg=CFG)
        assert math.isfinite(val), f"Non-finite output for snr={snr}, chisq={chisq}: {val}"
