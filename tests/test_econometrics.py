"""Tests for the Track A econometrics pipeline.

The load-bearing test in this file is `test_deviance_recovers_known_variance_function`.
Everything else guards a specific trap we actually hit while building Stage 6.

Why test the estimator against a synthetic process
--------------------------------------------------
Stage 6 returned a negative result: no step-3 model beat predicting the mean, on either
outcome. A negative result is only credible if the estimator is known to work. So the deviance
step is checked against data generated with a *known* variance function -- if it can recover a
variance structure that is there by construction, then its failure to find one in the real data
is a statement about the data, not about the code.

Tests that need the survey files are marked `dataset` and skip when they are absent, matching
the convention in the rest of the suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shrimp_risk.econometrics import data as D  # noqa: E402
from shrimp_risk.econometrics import just_pope as JP  # noqa: E402

IFPRI_DIR = ROOT / "data" / "raw" / "ifpri"
HASAN_FILE = ROOT / "data" / "raw" / "farm_economics" / "WSSVRiskFactor_rev.xlsx"

needs_ifpri = pytest.mark.skipif(
    not (IFPRI_DIR / "000_hh_commonvar.tab").exists(),
    reason="IFPRI survey files not present",
)
needs_hasan = pytest.mark.skipif(
    not HASAN_FILE.exists(), reason="Hasan & Haque workbook not present"
)


# ---------------------------------------------------------------------------
# The estimator itself, on data whose answer we know
# ---------------------------------------------------------------------------

def test_deviance_is_squared_residual():
    y = np.array([1.0, 2.0, 3.0])
    y_hat = np.array([1.5, 2.0, 1.0])
    np.testing.assert_allclose(JP.deviance(y, y_hat), [0.25, 0.0, 4.0])


def test_deviance_recovers_known_variance_function():
    """The estimator must find a variance structure that is present by construction.

    We generate y = 2x + noise, where the noise SD grows with x. The mean function is linear
    and correctly specified, so `E[d_i | x_i] = Var(y_i | x_i)` holds exactly and the log
    deviance must correlate positively with x. If this fails, every Stage 6 conclusion is
    suspect -- which is the point of testing it.
    """
    rng = np.random.default_rng(2025)
    n = 4000
    x = rng.uniform(1.0, 5.0, n)
    sigma = 0.5 * x                      # variance depends on the input, by construction
    y = 2.0 * x + rng.normal(0.0, sigma)

    frame = pd.DataFrame({"x": x, "y": y})
    res = JP.run_just_pope(frame, outcome="y", covariates=["x"], label="synthetic")

    corr = np.corrcoef(x, res.log_deviance)[0, 1]
    assert corr > 0.30, f"log-deviance should track the known variance function; got r={corr:.3f}"

    row = res.summary_row()
    assert row["breusch_pagan_p"] < 0.01, (
        "Breusch-Pagan must detect heteroskedasticity that was built into the data; "
        f"got p={row['breusch_pagan_p']:.4g}"
    )


def test_homoskedastic_data_is_not_flagged():
    """The mirror image: constant variance must NOT trigger the heteroskedasticity test.

    Without this, a test that always fires would make the disease-model finding
    (Breusch-Pagan p = 0.195, no heteroskedasticity) meaningless.
    """
    rng = np.random.default_rng(7)
    n = 4000
    x = rng.uniform(1.0, 5.0, n)
    y = 2.0 * x + rng.normal(0.0, 1.0, n)     # constant SD

    res = JP.run_just_pope(pd.DataFrame({"x": x, "y": y}), outcome="y",
                           covariates=["x"], label="homoskedastic")
    assert res.summary_row()["breusch_pagan_p"] > 0.05


def test_reset_detects_a_known_misspecification():
    """RESET must fire when non-linearity has been left in the conditional mean, by construction.

    The real mean is quadratic; we fit it linear. That is exactly the failure mode RESET exists
    to catch, and it is the failure Stage 6 turns out to have on both outcomes -- so the test
    that reports it has to be shown to work on data whose answer is known in advance.
    """
    rng = np.random.default_rng(2025)
    n = 2000
    x = rng.uniform(1.0, 5.0, n)
    y = 2.0 * x + 1.5 * x**2 + rng.normal(0.0, 1.0, n)   # quadratic truth, linear fit

    res = JP.run_just_pope(pd.DataFrame({"x": x, "y": y}), outcome="y",
                           covariates=["x"], label="misspecified")
    assert res.summary_row()["reset2_p"] < 0.01, (
        "RESET must reject when a quadratic term is omitted from the mean equation; "
        f"got p={res.summary_row()['reset2_p']:.4g}"
    )


def test_reset_does_not_fire_on_a_correctly_specified_mean_equation():
    """The mirror image, and the reason the Stage 6 RESET result carries information.

    A test that rejected everything would make `reset2_p = 4.3e-5` on the income equation
    meaningless. Same generator as above with the quadratic term removed.
    """
    rng = np.random.default_rng(7)
    n = 2000
    x = rng.uniform(1.0, 5.0, n)
    y = 2.0 * x + rng.normal(0.0, 1.0, n)                # linear truth, linear fit

    res = JP.run_just_pope(pd.DataFrame({"x": x, "y": y}), outcome="y",
                           covariates=["x"], label="correct")
    assert res.summary_row()["reset2_p"] > 0.05


def test_log_deviance_floor_is_counted_not_silent():
    """A residual of exactly zero would give log(0) = -inf and poison every downstream fit."""
    d = np.array([0.0, 1.0, np.e])
    logd, n_floored = JP.log_deviance(d)
    assert np.isfinite(logd).all(), "floor must remove -inf"
    assert n_floored == 1, "the floored observation must be counted, not hidden"


def test_log_transform_refuses_non_positive_outcome():
    """Refusing is correct: 11 farms report exactly 0 BDT of White Spot loss.

    Silently dropping them would change the question from 'what drives loss?' to
    'given a loss occurred, what drives its size?' -- a different model.
    """
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.0, 5.0, 7.0]})
    with pytest.raises(ValueError, match="non-positive"):
        JP.run_just_pope(frame, outcome="y", covariates=["x"], transform="log")

    res = JP.run_just_pope(frame, outcome="y", covariates=["x"], transform="log1p")
    assert res.summary_row()["n"] == 3, "log1p must keep all three rows"


# ---------------------------------------------------------------------------
# The c904 sentinel -- the single most damaging trap in this dataset
# ---------------------------------------------------------------------------

def test_c904_sentinel_becomes_missing():
    s = pd.Series([10.0, 30.0, 9997.0, 100.0])
    cleaned = D.clean_c904(s)
    assert cleaned.isna().sum() == 1
    assert cleaned.max() == 100.0


@needs_ifpri
def test_c904_sentinel_evidence_holds_on_real_data():
    """The diagnosis rests on an EMPTY range, not on 9997 merely looking odd.

    A genuine heavy tail would populate 101..9996. It is empty, which is what identifies
    9997 as a code. These are the numbers quoted in stage_06 and the PPT.
    """
    m = D.load_ifpri_module("020_hh_modc9_c902-c905")
    rep = D.sentinel_report(m["c904"])

    assert rep["n_between_101_and_9996"] == 0, "the empty range IS the evidence"
    assert rep["n_equal_sentinel"] == 94
    assert rep["max_cleaned"] <= 100.0
    # Uncleaned vs cleaned: the size of the mistake if the sentinel were treated as data.
    assert rep["mean_uncleaned"] > 800
    assert 32.0 < rep["mean_cleaned"] < 33.0


# ---------------------------------------------------------------------------
# Merge-grain traps -- long modules silently duplicate farms
# ---------------------------------------------------------------------------

def test_assert_one_row_per_farm_rejects_duplicates():
    dup = pd.DataFrame({"farmer_uid": ["a", "a", "b"], "v": [1, 2, 3]})
    with pytest.raises(Exception):
        D.assert_one_row_per_farm(dup, "duplicated")


def test_assert_one_row_per_farm_accepts_clean_table():
    ok = pd.DataFrame({"farmer_uid": ["a", "b"], "v": [1, 2]})
    assert len(D.assert_one_row_per_farm(ok, "clean")) == 2


@needs_ifpri
def test_white_spot_filter_is_one_row_per_farm():
    """492 rows, 492 farms -- no aggregation needed, and none silently applied."""
    m = D.load_ifpri_module("020_hh_modc9_c902-c905")
    w = D.filter_white_spot(m)
    assert len(w) == 492
    assert w["farmer_uid"].nunique() == 492


@needs_ifpri
def test_species_module_is_long_and_collapses_to_one_row_per_farm():
    raw = D.load_ifpri_module("006_hh_modc3")
    assert len(raw) > raw["farmer_uid"].nunique(), "modc3 must be long by farm x species"
    af = D.aggregate_species_module(raw)
    frame = af.frame
    assert frame["farmer_uid"].is_unique, "aggregation must yield one row per farm"


# ---------------------------------------------------------------------------
# Provenance -- a derived variable must never be presented as observed
# ---------------------------------------------------------------------------

@needs_ifpri
def test_revenue_is_labelled_derived_not_observed():
    af = D.build_income_frame()
    kinds = af.kinds() if callable(af.kinds) else af.kinds
    assert kinds["revenue_total_bdt"] == "derived", (
        "the survey has no income column; revenue is constructed and must say so"
    )


@needs_ifpri
def test_disease_outcome_is_observed():
    af = D.build_disease_frame()
    kinds = af.kinds() if callable(af.kinds) else af.kinds
    assert kinds["wsd_loss_bdt"] == "observed", "c905 is recorded by the survey, not constructed"


@needs_ifpri
def test_frames_have_the_expected_sample_sizes():
    """Attrition is documented in stage_06; this pins it so it cannot drift unnoticed."""
    assert len(D.build_income_frame().frame) == 1219
    assert len(D.build_disease_frame().frame) == 463


# ---------------------------------------------------------------------------
# External validation dataset
# ---------------------------------------------------------------------------

@needs_hasan
def test_hasan_haque_zero_inflation_and_missingness():
    """Both properties drive methodology decisions recorded in the dataset survey."""
    af = D.load_hasan_haque()
    f = af.frame
    assert len(f) == 233

    prevalence = f["CurrentPrevalance(%)"] if "CurrentPrevalance(%)" in f.columns else None
    if prevalence is not None:
        assert (prevalence == 0).sum() == 94, "40.3% exact zeros -- drives the hurdle-model question"
        assert prevalence.max() <= 100


# ---------------------------------------------------------------------------
# Sampling design -- the clustering sensitivity check (D-029)
# ---------------------------------------------------------------------------

@needs_ifpri
def test_cluster_robust_check_was_run_and_is_small():
    """IFPRI is a clustered sample; the reported HC3 errors assume it is not.

    The check exists so the answer is a number rather than a shrug. It pins that the design
    effect is modest -- if a re-run ever makes it large, the headline specification has to
    change and this test is the thing that says so.
    """
    path = ROOT / "reports" / "tables" / "exp03_cluster_robust.csv"
    if not path.exists():
        pytest.skip("exp03_cluster_robust.csv not generated yet; run scripts/exp03_track_a.py")
    df = pd.read_csv(path).set_index("risk_model")

    assert set(df.index) == {"income", "disease"}
    for model in ("income", "disease"):
        row = df.loc[model]
        assert str(row["note"]).startswith("sensitivity check"), (
            f"{model}: clustering was not actually measured -- {row['note']}"
        )
        assert row["n_clusters"] > 1, "a single cluster would make the check meaningless"
        assert row["se_ratio_median"] < 1.5, (
            f"{model}: cluster-robust errors are now {row['se_ratio_median']:.2f}x HC3. "
            "The design effect is no longer negligible -- the reported specification must change, "
            "not this threshold."
        )
