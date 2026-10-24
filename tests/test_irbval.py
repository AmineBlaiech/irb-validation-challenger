import numpy as np
import pandas as pd
import pytest

from irbval.data import simulate_portfolio
from irbval.performance import calibration as cal
from irbval.performance import discrimination as disc
from irbval.performance import stability as stab
from irbval.quality import checks as dq
from irbval.rwa import irb

NO_FLOORS = irb.Regime("sans planchers", 1.0, irb.Regime.crr3().pd_floor, False)


# --- RWA --------------------------------------------------------------------------------

def test_corporate_rw_matches_bcbs_reference():
    # Valeur de référence du Comité de Bâle : PD 1 %, LGD 45 %, M 2,5 ans → RW 92,32 %
    rw = irb.risk_weight(0.01, 0.45, "corporate", regime=NO_FLOORS)[0]
    assert rw == pytest.approx(0.9232, abs=1e-4)


def test_mortgage_rw_reference():
    # Hypothèque résidentielle PD 1 %, LGD 45 % → RW 56,40 %
    rw = irb.risk_weight(0.01, 0.45, "retail_mortgage", regime=NO_FLOORS)[0]
    assert rw == pytest.approx(0.5640, abs=1e-4)


def test_crr2_scaling_factor():
    crr2 = irb.risk_weight(0.01, 0.45, "corporate", regime=irb.Regime.crr2())[0]
    crr3 = irb.risk_weight(0.01, 0.45, "corporate", regime=NO_FLOORS)[0]
    assert crr2 / crr3 == pytest.approx(1.06)


def test_pd_and_lgd_floors_crr3():
    floored = irb.capital_requirement(0.0001, 0.10, "retail_other")[0]
    manual = irb.capital_requirement(0.0005, 0.30, "retail_other", regime=NO_FLOORS)[0]
    assert floored == pytest.approx(manual)


def test_sme_adjustment_lowers_capital():
    big = irb.capital_requirement(0.02, 0.45, "corporate")[0]
    sme = irb.capital_requirement(0.02, 0.45, "corporate_sme", turnover_meur=10)[0]
    assert sme < big


def test_defaulted_exposure():
    k = irb.capital_requirement(1.0, 0.45, "retail_other", elbe=0.40)[0]
    assert k == pytest.approx(0.05)


def test_output_floor_binding():
    res = irb.output_floor(rwa_irb=50, rwa_sa=100, year=2030)
    assert res["rwa_final"] == pytest.approx(72.5) and res["binding"] == 1


def test_rw_increases_with_pd_and_lgd():
    pds = np.array([0.001, 0.01, 0.05])
    assert np.all(np.diff(irb.risk_weight(pds, 0.45, "corporate")) > 0)
    assert irb.risk_weight(0.01, 0.6, "corporate")[0] > irb.risk_weight(0.01, 0.3, "corporate")[0]


# --- Discrimination ---------------------------------------------------------------------

def test_auc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert disc.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert disc.auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_delong_close_to_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.random(4000) < 0.1
    s = rng.normal(y * 1.0, 1)
    se_delong = np.sqrt(disc.delong_variance(y, s))
    lo, hi = disc.auc_bootstrap_ci(y, s, n_boot=300)
    assert se_delong == pytest.approx((hi - lo) / (2 * 1.96), rel=0.25)


# --- Calibration ------------------------------------------------------------------------

def test_jeffreys_and_binomial_agree_on_large_samples():
    t = pd.DataFrame({"N": [10_000, 10_000], "D": [100, 160], "PD": [0.01, 0.01]})
    t["TD_observe"] = t["D"] / t["N"]
    assert cal.binomial_test(t)[1] < 0.001 and cal.jeffreys_test(t)[1] < 0.001
    assert cal.binomial_test(t)[0] > 0.4 and cal.jeffreys_test(t)[0] > 0.4


def test_vasicek_threshold_more_tolerant_than_binomial():
    t = pd.DataFrame({"N": [5_000], "D": [80], "PD": [0.01]})
    t["TD_observe"] = t["D"] / t["N"]
    v = cal.vasicek_binomial_test(t, rho=0.05)
    assert v["TD_critique_vasicek"][0] > v["TD_critique_binomial"][0]


# --- Stabilité ----------------------------------------------------------------------------

def test_psi_zero_on_identical_samples():
    x = pd.Series(np.random.default_rng(1).normal(size=5000))
    assert stab.psi(x, x) == pytest.approx(0, abs=1e-9)


# --- Qualité des données sur le portefeuille simulé ------------------------------------------

@pytest.fixture(scope="module")
def portfolio():
    return simulate_portfolio(n_dev=8_000, n_mon=6_000, seed=7)


def test_deterministic_rules_catch_all_injected_anomalies(portfolio):
    rep = dq.run_rules(portfolio, dq.default_rules())
    for label, rule in [("duplicate", "doublons"), ("age_999", "age_invalide"),
                        ("negative_ead", "ead_negative"),
                        ("origination_after_snapshot", "octroi_apres_arrete"),
                        ("default_without_date", "defaut_sans_date")]:
        injected = portfolio["_anomaly"].str.contains(label).to_numpy()
        assert rep.flags[rule].to_numpy()[injected].all(), rule


def test_missing_income_is_detected_as_informative(portfolio):
    res = dq.missingness_vs_target(portfolio[["income", "default_12m"]])
    assert res.loc[0, "p_value_chi2"] < 0.01


def test_leakage_scan_flags_future_variable(portfolio):
    res = dq.leakage_scan(portfolio)
    assert res.loc[0, "variable"] == "nb_unpaid_next_12m" and res.loc[0, "suspect_fuite"]
