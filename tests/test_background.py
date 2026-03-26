"""Tests for background (time-slide FAR) estimation."""

from gwsearch.background import compute_background


def _trig(gps, new_snr, template_id="t0"):
    return {"gps": gps, "new_snr": new_snr, "template_id": template_id}


# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------

def test_fewer_than_two_ifos_returns_null():
    result = compute_background(
        per_ifo_triggers={"H1": [_trig(100.0, 9.0)]},
        slides=10,
        window=0.5,
        slide_spacing=0.5,
    )
    assert result["far_hz"] is None
    assert result["ifar_days"] is None


def test_empty_triggers_returns_null():
    result = compute_background(
        per_ifo_triggers={"H1": [], "L1": []},
        slides=10,
        window=0.5,
        slide_spacing=0.5,
    )
    assert result["far_hz"] is None or result["samples"] == []


def test_no_background_coincidences_when_triggers_not_overlapping():
    """
    If H1 and L1 triggers are 1000 s apart, no slide of ±5 s should match.
    """
    h1 = [_trig(0.0, 9.0)]
    l1 = [_trig(1000.0, 8.5)]
    result = compute_background(
        per_ifo_triggers={"H1": h1, "L1": l1},
        slides=5,
        window=0.5,
        slide_spacing=1.0,
    )
    assert result["samples"] == []
    assert result["far_hz"] is None


def test_background_finds_accidental_coincidences():
    """
    Densely populated triggers → time slides should find many accidentals.
    """
    h1 = [_trig(float(t), 9.0) for t in range(0, 100, 1)]
    l1 = [_trig(float(t) + 0.01, 8.5) for t in range(0, 100, 1)]
    result = compute_background(
        per_ifo_triggers={"H1": h1, "L1": l1},
        slides=5,
        window=0.5,
        slide_spacing=1.0,
    )
    assert len(result["samples"]) > 0


def test_far_is_positive_when_accidentals_exist():
    h1 = [_trig(float(t), 9.0) for t in range(0, 50, 1)]
    l1 = [_trig(float(t) + 0.02, 8.0) for t in range(0, 50, 1)]
    result = compute_background(
        per_ifo_triggers={"H1": h1, "L1": l1},
        slides=3,
        window=0.5,
        slide_spacing=1.0,
    )
    if result["samples"]:
        assert result["far_hz"] is not None
        assert result["far_hz"] > 0
        assert result["ifar_days"] is not None
        assert result["ifar_days"] > 0


def test_metadata_returned():
    result = compute_background(
        per_ifo_triggers={"H1": [_trig(0.0, 9.0)], "L1": [_trig(0.01, 8.0)]},
        slides=10,
        window=0.5,
        slide_spacing=0.5,
    )
    assert "slides" in result
    assert "slide_spacing" in result
    assert result["slides"] == 10
    assert result["slide_spacing"] == 0.5


def test_more_slides_produce_more_background_samples():
    """More time slides → more trials → more background samples (or equal)."""
    h1 = [_trig(float(t), 9.0) for t in range(0, 30, 1)]
    l1 = [_trig(float(t) + 0.02, 8.0) for t in range(0, 30, 1)]
    r_few = compute_background({"H1": h1, "L1": l1}, slides=2, window=0.5, slide_spacing=1.0)
    r_many = compute_background({"H1": h1, "L1": l1}, slides=10, window=0.5, slide_spacing=1.0)
    assert len(r_many["samples"]) >= len(r_few["samples"])
