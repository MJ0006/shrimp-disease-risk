"""Sensitivity -- what measurement error in a prevalence regressor does to the risk estimates.

The question, and why it is the third contribution rather than a robustness check
---------------------------------------------------------------------------------
The base paper regresses farm outcomes on disease prevalence and treats that prevalence as
**observed**. It is not observed; it is what a farmer remembered and reported to an enumerator.
This project's whole premise is that a camera could measure it instead -- but a camera measures it
with error too, and a different kind of error. Replacing one mismeasured regressor with another is
only progress if the error gets smaller, and nobody has ever quantified either.

Classical errors-in-variables says a mismeasured regressor's coefficient is biased **toward zero**
by

    lambda = var(true) / (var(true) + var(error))

so an uncorrected prevalence does not merely give a wrong number -- it systematically *understates
how much disease matters to farm income*. That is a policy-relevant direction: the error makes the
problem look smaller than it is.

Two kinds of error, because the project produces two kinds
-----------------------------------------------------------
1. **Misclassification of a binary regressor.** The income equation as estimated in Experiment 03
   already contains `had_white_spot`, a self-reported 0/1. A vision system replacing that
   questionnaire item makes false-positive and false-negative errors at rates its confusion matrix
   gives directly. The attenuation formula is different from the classical one and is derived in
   `binary_attenuation`.
2. **Continuous error in a prevalence regressor.** The bridge's actual output is a fraction in
   [0, 1] with the error distribution Experiment 07 measured. This is the classical case, except
   that the error is not Gaussian, not independent of the true value, and not symmetric -- so the
   classical formula is used as a *benchmark to compare against*, never as the answer.

What this module can and cannot claim
--------------------------------------
It perturbs a regressor in **real IFPRI data** by an error magnitude **measured** on real
posteriors, and reports what happens to real coefficients. That is a propagation study, and a
strong one.

It is not an end-to-end measurement. The image corpus has no farm identifier (Master Plan
limitation 2) and the IFPRI farms are a different sample from a different year (limitation 3b), so
no shrimp in the photographs belongs to any farm in the survey. Nothing computed here may be
described as the effect of *our* classifier on *those* farms.

And `D-026` still stands underneath all of it: RESET rejects both mean equations, so every result
this module produces is conditional on a functional form the project itself flags as doubtful. The
sensitivity of a coefficient is measured within a specification that may be wrong; that is worth
knowing and is not the same as knowing the truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerturbationSpec:
    """One point on the error grid, with a name a table can carry.

    `source` records where the magnitude came from -- a measured Experiment 07 cell, or a
    deliberately synthetic level chosen to bracket it. The distinction matters enough to travel
    with every row: a reader must be able to tell the levels this pipeline actually produces from
    the ones included to show the shape of the curve.
    """

    label: str
    kind: str            # "classical", "empirical", or "misclassification"
    magnitude: float     # error SD for continuous kinds; unused for misclassification
    fpr: float = 0.0
    fnr: float = 0.0
    source: str = ""


def classical_attenuation(x: np.ndarray, error_sd: float) -> float:
    """`lambda = var(true) / (var(true) + var(error))` -- the textbook prediction.

    Reported next to every measured coefficient path so the two can be compared. Where they
    disagree, the disagreement is the finding: the classical formula assumes the error is
    independent of the true value and of the equation error, and a classifier's error is neither
    (it is largest where the classes are hardest to tell apart, which is not random with respect
    to the farm).
    """
    var_true = float(np.var(x, ddof=1))
    if var_true <= 0:
        return float("nan")
    return var_true / (var_true + error_sd ** 2)


def binary_attenuation(p: float, fpr: float, fnr: float) -> float:
    """Attenuation of a binary regressor's coefficient under misclassification.

    With `X` in {0, 1}, `P(X = 1) = p`, and observed `X*` where `fpr = P(X* = 1 | X = 0)` and
    `fnr = P(X* = 0 | X = 1)`::

        q    = P(X* = 1)     = p (1 - fnr) + (1 - p) fpr
        cov(X*, X)           = p [(1 - fnr) - q]
        var(X*)              = q (1 - q)
        lambda = cov / var   = p [(1 - fnr) - q] / [q (1 - q)]

    Two properties worth stating, because they are what make binary misclassification different
    from the classical continuous case:

    * With no error (`fpr = fnr = 0`) it returns exactly 1, and with `fpr = fnr = 0.5` -- a
      coin-flip measurement -- exactly 0.
    * The attenuation depends on `p` itself. A rare condition measured with a small false-positive
      rate is attenuated far more than a common one at the same rate, because the false positives
      are drawn from a much larger pool of true negatives. White Spot in the IFPRI sample is not
      rare, which works in this project's favour and should be said rather than assumed.
    """
    q = p * (1 - fnr) + (1 - p) * fpr
    if q <= 0 or q >= 1:
        return float("nan")
    return float(p * ((1 - fnr) - q) / (q * (1 - q)))


def misclassify_binary(
    x: np.ndarray, rng: np.random.Generator, fpr: float, fnr: float
) -> np.ndarray:
    """Flip a 0/1 regressor at the given false-positive and false-negative rates.

    The rates come from the classifier's own confusion matrix, so this simulates *replacing the
    questionnaire item with the vision system* rather than adding generic noise. It assumes the
    self-report is correct and the camera is not, which is the conservative direction: if the
    self-report is itself noisy -- and the project argues it is -- the real comparison is more
    favourable to the camera than this makes it look.
    """
    x = np.asarray(x).astype("int64")
    flip_up = (x == 0) & (rng.random(len(x)) < fpr)
    flip_down = (x == 1) & (rng.random(len(x)) < fnr)
    out = x.copy()
    out[flip_up] = 1
    out[flip_down] = 0
    return out


def perturb_continuous(
    x: np.ndarray,
    rng: np.random.Generator,
    error_sd: float,
    empirical_errors: np.ndarray | None = None,
    clip: tuple[float, float] | None = (0.0, 1.0),
) -> np.ndarray:
    """Add measurement error to a continuous regressor.

    With `empirical_errors`, the error is **resampled from the errors Experiment 07 actually
    measured** rather than drawn from a Gaussian. That is the version this project should lead
    with: a classifier's error is skewed and bounded -- an estimate cannot fall below 0 or exceed
    1 -- and a Gaussian of matching standard deviation would put mass where no real estimate can
    land. The Gaussian path is kept because it is the case the classical formula describes, so the
    two together separate "the coefficient moved" from "the coefficient moved *because the error
    is not classical*".

    Clipping to [0, 1] after perturbation is deliberate and is itself a small source of bias at
    the boundaries. It is applied because a prevalence outside [0, 1] is not a value any real
    pipeline could report, and simulating one would flatter the classical benchmark.
    """
    x = np.asarray(x, dtype="float64")
    if empirical_errors is not None and len(empirical_errors):
        noise = rng.choice(np.asarray(empirical_errors, dtype="float64"), size=len(x),
                           replace=True)
    else:
        noise = rng.normal(0.0, error_sd, size=len(x))
    out = x + noise
    return np.clip(out, *clip) if clip is not None else out


def coefficient_of(mean_equation, variable: str) -> dict[str, float]:
    """Pull one coefficient row out of a fitted mean equation.

    Returns NaNs rather than raising when the variable is absent, because a sensitivity sweep that
    drops a collinear regressor in one replicate should record the gap rather than abort the run
    -- and a silent zero would read as "no effect" instead of "not estimated".
    """
    coefs = mean_equation.coefficients
    row = coefs[coefs["variable"] == variable]
    if len(row) != 1:
        return {"estimate": float("nan"), "std_error": float("nan"),
                "t": float("nan"), "p_value": float("nan")}
    r = row.iloc[0]
    return {
        "estimate": float(r["estimate"]),
        "std_error": float(r["std_error"]),
        "t": float(r["t"]),
        "p_value": float(r["p_value"]),
    }


# Below this, an attenuation ratio is not reported as a number. Dividing by a baseline
# coefficient that is itself indistinguishable from zero produces ratios of 2.4 and -1.5 from
# nothing but noise, and they look like findings. The threshold is on the baseline's p-value
# because that is the question actually being asked: is there an effect here to attenuate?
BASELINE_P_FOR_RATIO = 0.10

# Floating-point floor. At zero error the perturbation is the identity, so the difference from
# baseline is exactly zero up to rounding and the Monte-Carlo standard error is exactly zero
# too. Without a floor, `|1e-17| > 0` reports the zero-error row as having moved.
_NOISE_FLOOR = 1e-9


def summarise_path(
    rows: pd.DataFrame, baseline_estimate: float, baseline_p: float = float("nan")
) -> pd.DataFrame:
    """Collapse replicate-level results into one row per error level.

    `attenuation_observed` is the mean perturbed coefficient divided by the unperturbed one -- the
    empirical counterpart of the `lambda` the formulas predict. It is reported with the
    Monte-Carlo standard error of the mean beside it, because the pre-registered null for Stage 9
    is that movement inside that standard error counts as no detectable propagation.

    **The ratio is suppressed when the baseline coefficient is not distinguishable from zero.**
    This is not a cosmetic guard. Attenuation is a statement about how much of a real effect
    survives mismeasurement; where there is no measurable effect, the ratio is a small number
    divided by a smaller one and swings wildly with the sign of the noise. Reporting it would
    manufacture a dramatic-looking result out of an absence, and the absence is the honest
    finding. `ratio_interpretable` carries that verdict into the output table so a reader does not
    have to infer it from the p-value column.
    """
    grouped = rows.groupby(["level_label", "kind", "magnitude"], dropna=False)
    out = grouped.agg(
        replicates=("estimate", "size"),
        estimate_mean=("estimate", "mean"),
        estimate_sd=("estimate", "std"),
        se_mean=("std_error", "mean"),
        p_median=("p_value", "median"),
        r_squared_mean=("r_squared", "mean"),
        bp_p_median=("breusch_pagan_p", "median"),
    ).reset_index()

    out["baseline_estimate"] = baseline_estimate
    out["baseline_p"] = baseline_p
    interpretable = bool(
        baseline_estimate and np.isfinite(baseline_p) and baseline_p <= BASELINE_P_FOR_RATIO
    )
    out["ratio_interpretable"] = interpretable
    out["attenuation_observed"] = (
        out["estimate_mean"] / baseline_estimate if interpretable else np.nan
    )

    out["mc_se"] = out["estimate_sd"] / np.sqrt(out["replicates"])
    difference = (out["estimate_mean"] - baseline_estimate).abs()
    out["moved_beyond_noise"] = (difference > out["mc_se"]) & (difference > _NOISE_FLOOR)
    return out
