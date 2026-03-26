"""Tests for template bank generation (simple grid and spin coverage)."""

from gwsearch.bank import build_bank
from gwsearch.config import BankConfig


def _default_cfg(**kwargs) -> BankConfig:
    cfg = BankConfig(
        min_mass1=2.0,
        max_mass1=20.0,
        min_mass2=2.0,
        max_mass2=20.0,
        mismatch=0.03,
        approximant="IMRPhenomD",
        f_lower=20.0,
        quick=False,
        include_spins=False,
    )
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Basic structural checks
# ---------------------------------------------------------------------------

def test_bank_nonempty():
    cfg = _default_cfg()
    templates = build_bank(cfg)
    assert len(templates) > 0


def test_all_templates_have_required_keys():
    cfg = _default_cfg()
    required = {"mass1", "mass2", "spin1z", "spin2z", "approximant", "f_lower"}
    for t in build_bank(cfg):
        assert required.issubset(t.keys()), f"Missing keys in template: {t}"


def test_mass_ordering_m1_ge_m2():
    cfg = _default_cfg()
    for t in build_bank(cfg):
        assert t["mass1"] >= t["mass2"], f"m1 < m2 in template: {t}"


def test_total_mass_above_minimum():
    cfg = _default_cfg()
    for t in build_bank(cfg):
        assert t["mass1"] + t["mass2"] >= 4.0


def test_quick_mode_respects_cap():
    cfg = _default_cfg(quick=True, max_templates_quick=50)
    templates = build_bank(cfg)
    assert len(templates) <= 50


# ---------------------------------------------------------------------------
# Spin grid
# ---------------------------------------------------------------------------

def test_no_spins_when_disabled():
    cfg = _default_cfg(include_spins=False)
    for t in build_bank(cfg):
        assert t["spin1z"] == 0.0
        assert t["spin2z"] == 0.0


def test_spins_included_when_enabled():
    cfg = _default_cfg(include_spins=True)
    cfg.spin_grid_values = [-0.5, 0.0, 0.5]
    templates = build_bank(cfg)
    spin_pairs = {(t["spin1z"], t["spin2z"]) for t in templates}
    # At minimum there should be more than just (0, 0)
    assert len(spin_pairs) > 1


def test_spin_values_are_from_grid():
    cfg = _default_cfg(include_spins=True)
    cfg.spin_grid_values = [-0.5, 0.0, 0.5]
    allowed = {-0.5, 0.0, 0.5}
    for t in build_bank(cfg):
        assert t["spin1z"] in allowed, f"Unexpected spin1z: {t['spin1z']}"
        assert t["spin2z"] in allowed, f"Unexpected spin2z: {t['spin2z']}"


def test_spin_templates_more_than_nospin():
    """With a 3-value spin grid, bank should be ~9x bigger (3x3 spin combos)."""
    cfg_nospin = _default_cfg(include_spins=False)
    cfg_spin = _default_cfg(include_spins=True)
    cfg_spin.spin_grid_values = [-0.5, 0.0, 0.5]

    n_nospin = len(build_bank(cfg_nospin))
    n_spin = len(build_bank(cfg_spin))
    assert n_spin > n_nospin


# ---------------------------------------------------------------------------
# Approximant / frequency passthrough
# ---------------------------------------------------------------------------

def test_approximant_is_preserved():
    cfg = _default_cfg(approximant="SEOBNRv4")
    for t in build_bank(cfg):
        assert t["approximant"] == "SEOBNRv4"


def test_f_lower_is_preserved():
    cfg = _default_cfg(f_lower=30.0)
    for t in build_bank(cfg):
        assert t["f_lower"] == 30.0
