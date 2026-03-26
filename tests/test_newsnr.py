from gwsearch import ranking


def test_newsnr_equivalence():
    snr = 10.0
    chisq = 20.0
    class DummyCfg:
        conditioning = type("c", (), {"low_frequency_cutoff": 20.0})
    val = ranking.new_snr(snr, chisq, duration=1.0, cfg=DummyCfg)
    # direct formula: rho_new = rho / ((1 + (chisq/dof)**3)/2)**(1/6)
    dof = 16 * 2 - 2
    expected = snr / (((1 + (chisq / dof) ** 3) / 2) ** (1 / 6))
    assert abs(val - expected) < 1e-9
