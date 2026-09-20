"""Tests for Stage 9 -- measurement-error propagation.

Why these are simulation tests against closed-form answers
-----------------------------------------------------------
Stage 9's claim is that mismeasuring the disease regressor attenuates its coefficient by a
predictable amount. The experiment measures that on IFPRI data, where the true coefficient is
unknown -- so if the measured attenuation came out at some unexpected value, there would be no way
to tell a real property of the data from a mistake in the perturbation code.

These tests remove that ambiguity. Each generates data where the attenuation is known by
construction and asserts that both the formula and the perturbation machinery recover it. This is
the same device `test_deviance_recovers_known_variance_function` provides for Stage 6 and
`test_acc_recovers_the_prevalence_exactly_from_a_known_confusion_matrix` for Stage 8.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shrimp_risk.integration import sensitivity as SEN  # noqa: E402


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Bivariate OLS slope. Written out rather than imported so the test depends on arithmetic."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    return float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))


# ---------------------------------------------------------------------------------------
# The attenuation formulas
# ---------------------------------------------------------------------------------------

def test_perfect_measurement_does_not_attenuate():
    assert SEN.binary_attenuation(p=0.4, fpr=0.0, fnr=0.0) == pytest.approx(1.0)
    assert SEN.classical_attenuation(np.array([0.0, 1.0, 2.0, 3.0]), error_sd=0.0) == 1.0


def test_a_coin_flip_measurement_destroys_the_coefficient_entirely():
    """`fpr = fnr = 0.5` means the observed value is independent of the truth."""
    assert SEN.binary_attenuation(p=0.4, fpr=0.5, fnr=0.5) == pytest.approx(0.0, abs=1e-12)


def test_attenuation_is_worse_for_a_rare_condition_at_the_same_error_rate():
    """The property that makes binary misclassification unlike the classical case.

    At a 10% false-positive rate, a condition present in 5% of farms is attenuated far harder than
    one present in 45%, because the false positives come from a much larger pool of true
    negatives. White Spot in the IFPRI sample is common, which works in this project's favour --
    and that is a fact about the sample, not a general property, so it is pinned here.
    """
    rare = SEN.binary_attenuation(p=0.05, fpr=0.10, fnr=0.10)
    common = SEN.binary_attenuation(p=0.45, fpr=0.10, fnr=0.10)
    assert rare < common < 1.0


def test_binary_attenuation_matches_a_simulation():
    """The formula against a data-generating process where the true slope is known.

    `y = 2 + 3x + noise` with `x` binary. Misclassify `x`, regress on the mismeasured version, and
    the slope must land on `3 * lambda`.
    """
    rng = np.random.default_rng(2025)
    n, p, beta = 200_000, 0.35, 3.0
    fpr, fnr = 0.12, 0.20

    x = (rng.random(n) < p).astype("int64")
    y = 2.0 + beta * x + rng.normal(0, 1.0, n)
    x_star = SEN.misclassify_binary(x, rng, fpr=fpr, fnr=fnr)

    predicted = beta * SEN.binary_attenuation(float(x.mean()), fpr, fnr)
    observed = ols_slope(x_star, y)

    assert observed == pytest.approx(predicted, rel=0.03)
    assert observed < beta, "misclassification must bias toward zero, never away from it"


def test_classical_attenuation_matches_a_simulation():
    """The continuous case, with Gaussian error that is classical by construction."""
    rng = np.random.default_rng(7)
    n, beta, error_sd = 200_000, 1.5, 0.25

    x = rng.normal(0.5, 0.20, n)
    y = 1.0 + beta * x + rng.normal(0, 0.5, n)
    x_star = SEN.perturb_continuous(x, rng, error_sd=error_sd, clip=None)

    predicted = beta * SEN.classical_attenuation(x, error_sd)
    observed = ols_slope(x_star, y)

    assert observed == pytest.approx(predicted, rel=0.03)
    assert observed < beta


def test_zero_error_leaves_the_coefficient_where_it_was():
    """The mirror of the two tests above -- the machinery must not attenuate what it should not."""
    rng = np.random.default_rng(3)
    n, beta = 50_000, 1.5
    x = rng.normal(0.5, 0.2, n)
    y = 1.0 + beta * x + rng.normal(0, 0.5, n)

    unchanged = SEN.perturb_continuous(x, rng, error_sd=0.0, clip=None)
    np.testing.assert_allclose(unchanged, x)
    assert ols_slope(unchanged, y) == pytest.approx(beta, rel=0.02)


# ---------------------------------------------------------------------------------------
# The perturbation machinery
# ---------------------------------------------------------------------------------------

def test_misclassification_hits_the_rates_it_was_given():
    rng = np.random.default_rng(11)
    x = np.array([0] * 50_000 + [1] * 50_000)
    x_star = SEN.misclassify_binary(x, rng, fpr=0.10, fnr=0.25)

    false_positive_rate = x_star[x == 0].mean()
    false_negative_rate = 1 - x_star[x == 1].mean()
    assert false_positive_rate == pytest.approx(0.10, abs=0.01)
    assert false_negative_rate == pytest.approx(0.25, abs=0.01)


def test_empirical_errors_are_resampled_rather_than_approximated_by_a_gaussian():
    """The version Stage 9 leads with: real classifier error is skewed and bounded.

    Feeding a one-sided error distribution must produce a one-sided shift, which a Gaussian of the
    same standard deviation would not.
    """
    rng = np.random.default_rng(5)
    x = np.full(20_000, 0.5)
    errors = np.array([0.10, 0.15, 0.20])  # always positive
    out = SEN.perturb_continuous(x, rng, error_sd=0.05, empirical_errors=errors)

    assert (out > 0.5).all(), "an all-positive error set cannot produce a downward shift"
    assert set(np.round(out - 0.5, 6)) <= {0.10, 0.15, 0.20}


def test_a_perturbed_prevalence_stays_inside_zero_and_one():
    """No pipeline can report a prevalence of 1.4, so no simulation may either."""
    rng = np.random.default_rng(9)
    x = np.array([0.02, 0.5, 0.98] * 1000)
    out = SEN.perturb_continuous(x, rng, error_sd=0.5)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_a_missing_coefficient_is_recorded_as_missing_not_as_zero():
    """A dropped collinear regressor must not read as 'estimated, and the effect was nil'."""
    class FakeEquation:
        coefficients = __import__("pandas").DataFrame(
            {"variable": ["const", "x"], "estimate": [1.0, 2.0],
             "std_error": [0.1, 0.2], "t": [10.0, 10.0], "p_value": [0.0, 0.0]}
        )

    present = SEN.coefficient_of(FakeEquation(), "x")
    absent = SEN.coefficient_of(FakeEquation(), "not_in_the_model")
    assert present["estimate"] == 2.0
    assert np.isnan(absent["estimate"])
