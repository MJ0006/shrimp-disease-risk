"""Calibration -- making the classifier's confidence mean what it says.

The problem this module exists to fix
-------------------------------------
Every estimator in `quantification.py` consumes the classifier's *estimated* posteriors. Two of
them -- PCC and PACC -- average those numbers directly, so a model that says 0.90 when it is right
0.70 of the time does not merely make a noisy prevalence estimate: it makes a **systematically
inflated** one, in a direction the estimator cannot detect from its own inputs.

That is not a hypothetical here. `scripts/exp05_calibration.py` measured it over all 34 stored runs
(`D-028`):

    C_specimen_level   ECE 0.2046   mean confidence 0.892   accuracy 0.695   overconfident by +0.197
    B_image_level      ECE 0.1008   mean confidence 0.922   accuracy 0.829   overconfident by +0.093

and the second line is the awkward one. The project had been arguing that specimen leakage causes
overconfidence, so removing the leakage should improve calibration. Measured, the specimen-level
model is the **worse** calibrated of the two -- because memorising specimens makes the image-level
model accurate enough that its confidence looks justified. Specimen-level splitting is necessary for
an honest *estimate* and not sufficient for a trustworthy *posterior*, and this module is what
addresses the second half.

What temperature scaling does
-----------------------------
One scalar `T > 0`, applied to the logits before the softmax::

    p_calibrated = softmax(z / T)

`T > 1` softens the distribution and reduces confidence; `T < 1` sharpens it. It is fitted by
minimising negative log-likelihood on a held-out set. Because a single scalar cannot reorder logits,
**argmax is unchanged**, so accuracy, macro-F1, per-class recall and the confusion matrix are all
exactly what they were before calibration. Nothing that Stages 4-5 reported can move as a result of
this, which is the property that makes it safe to apply after the fact.

Why this works from stored probabilities, with no logits
--------------------------------------------------------
Every run in this project saved softmax outputs, not logits, so temperature scaling looks
impossible without retraining. It is not. Softmax is shift-invariant, and the stored probabilities
give back the logits up to exactly that shift::

    log(p) = log(softmax(z)) = z - logsumexp(z) = z - c

so `softmax(log(p) / T) = softmax((z - c) / T) = softmax(z / T)`, since the constant `c / T` shifts
every component equally and cancels. Working from `log(p)` is therefore not an approximation of
temperature scaling; it is the same computation.

The one practical caveat is float32 underflow: a stored probability that rounded to 0.0 gives
`log(0) = -inf`, so probabilities are clamped to a small epsilon first. The clamp is the only
inexactness in the path and it is bounded by that epsilon.

What still needs a retrain, and why
-----------------------------------
The *fitting set*. A temperature fitted on the test fold would tune a parameter on the data the
score is reported from -- the exact failure this project is about -- and one fitted on training data
would see examples the model has memorised. It has to be the validation fold, and the stored runs
saved test posteriors only. `scripts/exp06_calibrate_bridge.py` re-runs training with validation
posteriors saved alongside; `trainer.py` scores them on the *restored best* weights rather than
collecting them during training, so the temperature belongs to the model that is actually reported.

Where the metrics live
----------------------
`expected_calibration_error` and `reliability_bins` were written for Experiment 05 and are lifted
here unchanged so that the measurement and the correction share one implementation. A calibration
step evaluated by a second, subtly different ECE would be measuring itself against a moving target.
`scripts/exp05_calibration.py` now imports them from this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# 15 bins is the common default in the calibration literature. The count is reported in every
# output table so the choice is visible rather than inherited, and Experiment 05 confirmed the
# measured ECE is stable across 10, 15 and 20 bins -- so it is a reporting convention here, not a
# result that depends on it.
DEFAULT_BINS = 15

# Guards log(0) when a float32 softmax has rounded a component to zero. Small enough not to
# perturb any probability that is genuinely non-zero at float32 resolution.
_EPS = 1e-12

# Bounds for the temperature search. T -> 0 sharpens to a one-hot distribution and T -> inf
# flattens to uniform; both are degenerate, and a fitted temperature that lands on either bound is
# a signal that something is wrong rather than a calibration result, so the bound is reported.
_T_MIN, _T_MAX = 0.05, 20.0


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = DEFAULT_BINS
) -> dict[str, float]:
    """Confidence-vs-accuracy summary for one set of predictions.

    Returns ECE (population-weighted mean gap), MCE (worst single bin), the multi-class Brier
    score, and the mean confidence and accuracy that ECE is the gap between. Reporting the last
    two separately matters: ECE is an absolute value, so it cannot say whether the model is over-
    or under-confident, and for this project's models the *direction* is the finding.
    """
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == labels).astype("float64")
    n = len(labels)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Half-open on the left so every prediction lands in exactly one bin and the weights sum
        # to 1. A prediction at exactly 0.0 confidence cannot occur for a softmax over 4 classes.
        in_bin = (confidence > lo) & (confidence <= hi)
        if not in_bin.any():
            continue
        gap = abs(correct[in_bin].mean() - confidence[in_bin].mean())
        ece += in_bin.sum() / n * gap
        mce = max(mce, gap)

    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": brier,
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
        "overconfidence": float(confidence.mean() - correct.mean()),
        "n_test": int(n),
    }


def reliability_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = DEFAULT_BINS
) -> pd.DataFrame:
    """Per-bin accuracy against confidence -- the data behind a reliability diagram."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype("float64")
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        rows.append({
            "bin_lo": round(float(lo), 4),
            "bin_hi": round(float(hi), 4),
            "n": int(in_bin.sum()),
            "mean_confidence": float(confidence[in_bin].mean()) if in_bin.any() else float("nan"),
            "accuracy": float(correct[in_bin].mean()) if in_bin.any() else float("nan"),
        })
    return pd.DataFrame(rows)


def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    """Recover logits from stored softmax outputs, up to the shift softmax discards.

    See the module docstring: the shift cancels inside the softmax, so this is exact rather than
    approximate. The clamp is the only inexactness, and it only engages where float32 already
    rounded a probability to zero.
    """
    return np.log(np.clip(probs.astype("float64"), _EPS, 1.0))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Re-scale stored posteriors by a fitted temperature.

    `temperature == 1.0` returns the input unchanged to within floating point, which is worth
    knowing: it means a run whose fit lands at 1.0 can be reported as "already calibrated at this
    resolution" rather than as a failed correction.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive; got {temperature}")
    scaled = probs_to_logits(probs) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)  # stabilises exp() without changing the result
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean NLL of the true class -- the objective temperature scaling minimises.

    NLL rather than ECE, deliberately. ECE is the quantity being *reported*, and fitting directly
    on the reported metric would let the temperature exploit the binning: a model can shuffle mass
    across bin edges and improve ECE without its probabilities becoming more truthful. NLL is a
    strictly proper scoring rule -- it is minimised only by the true probabilities -- so optimising
    it and reporting ECE keeps the fit and the verdict independent.
    """
    p_true = np.clip(probs[np.arange(len(labels)), labels], _EPS, 1.0)
    return float(-np.log(p_true).mean())


def fit_temperature(
    probs: np.ndarray, labels: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Fit one temperature on a held-out set by minimising NLL.

    Bounded scalar minimisation rather than gradient descent. The objective is smooth and
    one-dimensional, so Brent's method converges in a few dozen evaluations with no learning rate,
    no step count and no seed -- which means the fitted temperature is a deterministic function of
    the inputs and two runs of this code cannot disagree.

    Returns the temperature and a small diagnostic record: NLL before and after, and whether the
    optimiser landed on a bound. A fit that hits a bound has not found an interior optimum and
    should be read as a failure of the model rather than a calibration.

    **The caller is responsible for passing held-out data.** This function cannot tell validation
    posteriors from test posteriors, and fitting on test would silently produce a better-looking
    ECE that means nothing.
    """
    labels = labels.astype("int64")

    def objective(t: float) -> float:
        return negative_log_likelihood(apply_temperature(probs, t), labels)

    result = minimize_scalar(objective, bounds=(_T_MIN, _T_MAX), method="bounded")
    temperature = float(result.x)

    return temperature, {
        "temperature": temperature,
        "nll_before": negative_log_likelihood(probs, labels),
        "nll_after": float(result.fun),
        "at_bound": bool(
            abs(temperature - _T_MIN) < 1e-6 or abs(temperature - _T_MAX) < 1e-6
        ),
        "n_fit": int(len(labels)),
    }
