"""Tests for coincidence detection logic."""

import math

from gwsearch.coincidence import find_coincidences


def _trig(gps, new_snr, template_id="t0", ifo="H1"):
    return {"gps": gps, "new_snr": new_snr, "template_id": template_id, "ifo": ifo}


# ---------------------------------------------------------------------------
# Basic coincidence detection
# ---------------------------------------------------------------------------

def test_coincidence_within_window():
    trigs = {
        "H1": [_trig(100.000, 9.0)],
        "L1": [_trig(100.005, 8.5)],  # 5 ms apart – within 15 ms window
    }
    coincs = find_coincidences(trigs, window=0.015, require_same_template=True)
    assert len(coincs) == 1
    c = coincs[0]
    assert set(c["ifos"]) == {"H1", "L1"}
    assert abs(c["dt"]) < 0.015
    # network stat must be quadrature sum
    expected_net = math.sqrt(9.0**2 + 8.5**2)
    assert abs(c["network_stat"] - expected_net) < 1e-6


def test_no_coincidence_outside_window():
    trigs = {
        "H1": [_trig(100.000, 9.0)],
        "L1": [_trig(100.020, 8.5)],  # 20 ms apart – outside 15 ms window
    }
    coincs = find_coincidences(trigs, window=0.015)
    assert coincs == []


def test_same_template_required_filters_mismatched():
    trigs = {
        "H1": [_trig(100.000, 9.0, template_id="tA")],
        "L1": [_trig(100.005, 8.5, template_id="tB")],
    }
    coincs = find_coincidences(trigs, window=0.015, require_same_template=True)
    assert coincs == []


def test_same_template_not_required_allows_different():
    trigs = {
        "H1": [_trig(100.000, 9.0, template_id="tA")],
        "L1": [_trig(100.005, 8.5, template_id="tB")],
    }
    coincs = find_coincidences(trigs, window=0.015, require_same_template=False)
    assert len(coincs) == 1


def test_nan_snr_handled_gracefully():
    """NaN new_snr values should be treated as 0, not crash."""
    import math
    trigs = {
        "H1": [_trig(100.000, float("nan"))],
        "L1": [_trig(100.005, 8.5)],
    }
    coincs = find_coincidences(trigs, window=0.015, require_same_template=False)
    # Should produce a coincidence with NaN replaced by 0
    assert len(coincs) == 1
    assert math.isfinite(coincs[0]["network_stat"])


def test_inf_snr_handled_gracefully():
    """Inf new_snr should be caught and not propagate."""
    trigs = {
        "H1": [_trig(100.000, float("inf"))],
        "L1": [_trig(100.005, 8.5)],
    }
    coincs = find_coincidences(trigs, window=0.015, require_same_template=False)
    assert len(coincs) == 1
    assert math.isfinite(coincs[0]["network_stat"])


def test_single_ifo_returns_empty():
    trigs = {"H1": [_trig(100.0, 9.0)]}
    coincs = find_coincidences(trigs, window=0.015)
    assert coincs == []


def test_empty_triggers_returns_empty():
    coincs = find_coincidences({}, window=0.015)
    assert coincs == []


def test_multiple_triggers_multiple_coincidences():
    """Two separate events should each produce one coincidence."""
    trigs = {
        "H1": [_trig(100.000, 10.0, "t0"), _trig(200.000, 11.0, "t1")],
        "L1": [_trig(100.005, 9.5, "t0"), _trig(200.008, 10.5, "t1")],
    }
    coincs = find_coincidences(trigs, window=0.015, require_same_template=True)
    assert len(coincs) == 2
