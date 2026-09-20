"""Prevalence quantification -- turning per-image posteriors into a farm-level disease rate.

The estimator everyone reaches for first, and why it is wrong
------------------------------------------------------------
Count the images the classifier calls diseased, divide by the total. For a binary problem with
sensitivity `tpr` and specificity `tnr`, that estimator has expectation::

    E[p_naive] = p * tpr + (1 - p) * (1 - tnr)

which equals the true `p` only when `tpr + tnr == 2`, i.e. a perfect classifier. **The error is
systematic, not noise**: it does not shrink as more shrimp are photographed, and averaging more
images makes the wrong number more precise rather than more correct. At the specimen-level accuracy
this project actually achieves -- about 0.70 -- the naive estimator is wrong by an amount that
depends on the true prevalence, which is the quantity being estimated. That circularity is the
problem quantification solves.

With four classes the same statement is a matrix one. Writing `M[i, j] = P(predict i | true j)`,
the vector of *predicted* class proportions is::

    q = M p

so counting predictions reports `q` while meaning `p`. Recovering `p` means inverting `M`, and `M`
has to be estimated from held-out data -- which is where the validation fold earns its keep a second
time, after the calibration temperature.

Why this matters to the economics rather than just to the vision
----------------------------------------------------------------
Feeding an uncorrected prevalence into Track A puts measurement error into a *regressor*. Classical
errors-in-variables attenuates that coefficient toward zero by `var_true / (var_true + var_error)`,
so an uncorrected bridge does not merely report a wrong prevalence -- it systematically understates
how much disease matters to farm income. Stage 9 measures how far. This module is what makes the
input to that measurement defensible.

The estimators implemented here
-------------------------------
=================  =========================================================================
CC                 Classify and count. The biased baseline, kept because the comparison
                   against it *is* the evidence that correction was needed.
PCC                Probabilistic classify and count -- average the posteriors instead of the
                   hard labels. Lower variance than CC, and still biased: averaging
                   miscalibrated probabilities does not remove the miscalibration.
ACC                Adjusted count. Solves `M p = q` with `M` estimated from validation
                   argmax counts. Unbiased when `M` is right and the class-conditional
                   densities have not changed.
PACC              Probabilistic adjusted count. Same solve, but `M` holds mean *posteriors*
                   per true class rather than hard counts, so it uses the full probability
                   vector on both sides.
SLD-EM             Saerens-Latinne-Decaestecker (2002) expectation-maximisation for prior
                   shift. Iteratively re-weights the posteriors by the current prior estimate
                   and re-estimates the prior from the re-weighted posteriors.
=================  =========================================================================

Saerens et al. (2002) -- read in full on 2026-08-27, not cited second-hand -- compare their EM
procedure against the confusion-matrix method and report EM as superior for prior estimation, which
is why the module contract leads with it. Whether that holds *here* is an empirical question, and
Experiment 07 answers it on pools whose true prevalence we set, rather than assuming the published
ranking transfers to a four-class aquaculture problem with 416 specimens.

The assumption every one of them makes
--------------------------------------
Saerens et al. state it precisely: the **within-class densities `p(x | class)` are unchanged**
between training and deployment, and only the class proportions shift. A farm whose shrimp are
photographed under different light, or at a different growth stage, violates it, and no amount of
correction repairs that. This is a prior-shift correction, not a domain-adaptation method, and
nothing in this project's results should be read as claiming otherwise.

Prevalence is a property of shrimp, not of photographs
------------------------------------------------------
The corpus is 1,149 images of 416 specimens -- 2.76 photographs each, and unevenly: some specimens
appear twice, some five times. Quantifying over images would weight a five-photograph shrimp 2.5
times more heavily than a two-photograph one, so the estimate would drift toward whichever animals
the photographer lingered on. That is the same conflation of *image* with *individual* that
`D-007` found in the split protocol, reappearing one layer up in the estimator.

So every estimator here consumes **specimen-level** posteriors, produced by
`aggregate_to_specimen`, and every bootstrap resamples **specimens**. A farm's prevalence is the
fraction of its shrimp that are infected, and one shrimp is one observation however many times it
was photographed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Class order is fixed project-wide in `vision.datasets.CLASSES` = ["BG", "BG_WSSV", "Healthy",
# "WSSV"]. Duplicated as indices here rather than imported, because `integration` must not depend
# on `vision` -- the bridge consumes stored posterior arrays, not the training stack, and a farm
# running inference from an exported model has no torch import to spare.
CLASS_ORDER = ("BG", "BG_WSSV", "Healthy", "WSSV")

# White Spot positive = any class carrying WSSV. `BG_WSSV` is a co-infection of Black Gill and
# White Spot, so it counts as positive for White Spot prevalence. This matters because the Track A
# outcome it is being matched to is IFPRI's **White Spot Disease** variable, not "any disease" --
# defining the positive set as "not Healthy" would silently include pure Black Gill and inflate the
# prevalence that gets handed to the econometric model.
WSSV_POSITIVE = ("BG_WSSV", "WSSV")

_EPS = 1e-12


@dataclass
class PrevalenceEstimate:
    """One estimator's answer, with everything needed to judge it.

    `prior` is the full class-proportion vector; `prevalence` is the positive-class sum that the
    econometric model consumes. Both are kept because a four-class prior that looks wrong in a
    class nobody asked about is a warning sign about the estimate that was asked about.

    `left_simplex` records whether the unconstrained solve produced a negative or greater-than-one
    proportion before clipping. It is a diagnostic, not a failure: it means the estimated confusion
    matrix and the observed prediction proportions are mutually inconsistent, which happens when
    the matrix was estimated on too few validation samples. An estimate that needed clipping should
    be read more sceptically than one that did not, and silently clipping without saying so would
    hide exactly that.
    """

    method: str
    prior: np.ndarray
    prevalence: float
    n_specimens: int
    left_simplex: bool = False
    iterations: int | None = None
    converged: bool | None = None


def positive_indices(positive: tuple[str, ...] = WSSV_POSITIVE) -> list[int]:
    """Class indices for a named positive set, resolved against the fixed class order."""
    unknown = set(positive) - set(CLASS_ORDER)
    if unknown:
        raise ValueError(f"unknown class name(s) {sorted(unknown)}; expected {CLASS_ORDER}")
    return [CLASS_ORDER.index(c) for c in positive]


def aggregate_to_specimen(
    probs: np.ndarray, specimen_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse per-image posteriors to one posterior per specimen, by averaging.

    Averaging rather than voting. A majority vote over 2 or 3 photographs throws away the
    confidence that calibration was just fitted to correct -- and with 2.76 images per specimen,
    ties are common enough that the tie-break rule would materially affect the answer. Averaging
    keeps the probability information and degrades gracefully at one image, where it is the
    identity.

    Returns the specimen posteriors in the sorted order of their ids, so two calls on the same data
    always align row-for-row and a caller can safely index one result with another's ordering.
    """
    if len(probs) != len(specimen_ids):
        raise ValueError(
            f"{len(probs)} posteriors for {len(specimen_ids)} specimen ids -- these must be "
            "the same array length, or images are being attributed to the wrong shrimp"
        )
    ids = np.asarray(specimen_ids)
    unique = np.unique(ids)
    out = np.zeros((len(unique), probs.shape[1]), dtype="float64")
    for k, sid in enumerate(unique):
        out[k] = probs[ids == sid].mean(axis=0)
    return out, unique


def _as_prevalence(
    prior: np.ndarray, method: str, n: int, positive: list[int], **extra
) -> PrevalenceEstimate:
    return PrevalenceEstimate(
        method=method,
        prior=prior,
        prevalence=float(prior[positive].sum()),
        n_specimens=int(n),
        **extra,
    )


def _project_to_simplex(v: np.ndarray) -> tuple[np.ndarray, bool]:
    """Clip to non-negative and renormalise; report whether clipping was needed.

    This is the standard treatment of an ACC solve that leaves the simplex, and it is a genuine
    approximation: clipping biases the estimate away from the boundary, so an estimator that
    frequently clips is biased at extreme prevalences in a direction the correction cannot fix.
    Experiment 07 covers prevalences from near 0 to near 1 precisely so this shows up in the
    results rather than in a footnote.
    """
    left = bool((v < 0).any() or (v > 1).any())
    clipped = np.clip(v, 0.0, None)
    total = clipped.sum()
    if total <= _EPS:
        # Every component clipped to zero: the solve is uninformative. Falling back to a uniform
        # prior is the honest answer -- it says "this estimator has nothing to offer here" rather
        # than manufacturing a number out of the clipping rule.
        return np.full_like(clipped, 1.0 / len(clipped)), True
    return clipped / total, left


def classify_and_count(
    specimen_probs: np.ndarray, positive: list[int] | None = None
) -> PrevalenceEstimate:
    """CC -- the biased baseline. Hard-label counts over specimens."""
    positive = positive if positive is not None else positive_indices()
    preds = specimen_probs.argmax(axis=1)
    prior = np.bincount(preds, minlength=specimen_probs.shape[1]).astype("float64")
    prior /= max(prior.sum(), _EPS)
    return _as_prevalence(prior, "CC", len(specimen_probs), positive)


def probabilistic_classify_and_count(
    specimen_probs: np.ndarray, positive: list[int] | None = None
) -> PrevalenceEstimate:
    """PCC -- average the posteriors instead of the hard labels.

    Lower variance than CC because it does not discard the margin, and still biased for the same
    reason: it inherits whatever miscalibration the posteriors carry. It is the estimator that
    benefits most from the temperature fitted in Experiment 06, which is why both are reported.
    """
    positive = positive if positive is not None else positive_indices()
    prior = specimen_probs.mean(axis=0)
    return _as_prevalence(prior, "PCC", len(specimen_probs), positive)


def confusion_from_validation(
    val_probs: np.ndarray, val_labels: np.ndarray, soft: bool = False
) -> np.ndarray:
    """Estimate `M[i, j] = P(predict i | true j)` on held-out data.

    Columns are true classes and each column sums to 1, which is the orientation the `M p = q`
    solve needs. Getting this transposed produces a plausible-looking matrix and a silently wrong
    prevalence, so the normalisation is asserted by the tests rather than trusted.

    `soft=True` gives the PACC variant: entry `[i, j]` is the mean *posterior* mass on class `i`
    among validation samples whose true class is `j`, rather than the fraction predicted `i`. It
    uses the whole probability vector on both sides of the solve and is the better-conditioned of
    the two when a class is small -- which matters here, since Black Gill has 75 specimens.

    A true class absent from the validation fold gets a column of zeros, which would make the
    matrix singular. That column is set to the identity instead: with no evidence about how that
    class is confused, the least-assuming statement is that it is not.
    """
    n_classes = val_probs.shape[1]
    M = np.zeros((n_classes, n_classes), dtype="float64")
    preds = val_probs.argmax(axis=1)

    for j in range(n_classes):
        rows = val_labels == j
        if not rows.any():
            M[j, j] = 1.0
            continue
        if soft:
            M[:, j] = val_probs[rows].mean(axis=0)
        else:
            M[:, j] = np.bincount(preds[rows], minlength=n_classes) / rows.sum()
    return M


def confusion_at_specimen_grain(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    val_specimen_ids: np.ndarray,
    soft: bool = False,
) -> np.ndarray:
    """Estimate `M` from validation data at the grain the estimators actually observe.

    **Use this, not `confusion_from_validation`, whenever the estimate will be applied to
    specimen-level posteriors** -- which in this project is always.

    Why it needs its own function rather than a note in a docstring: ACC solves `M p = q`, where
    `q` is the vector of predicted proportions the estimator *observes*. Every estimator here
    observes specimen-level predictions, averaged over a shrimp's 2.76 photographs. A matrix
    estimated from individual images describes a different classifier -- a measurably worse one,
    because averaging photographs removes error. Applying it overstates confusion, so ACC
    over-corrects and the solve is pushed off the simplex.

    That mistake was made and caught in Experiment 07 on 2026-09-06. On fold 0 the two matrices
    differ by up to 0.11 in a single entry, and the wrong one raised the share of pools needing
    simplex clipping substantially. It was invisible because both matrices are valid-looking
    stochastic matrices and neither the types nor the shapes disagree. The fix is a function whose
    signature makes the grain impossible to get wrong: it demands the specimen ids, so an
    image-level call cannot be written by accident.

    The cost is real and is the honest trade. A fold has ~67 validation specimens against ~187
    validation images, so this matrix is estimated from a third of the rows and is noisier. A
    noisier estimate of the right quantity beats a tighter estimate of the wrong one; the noise
    surfaces in the results as clipping, which is reported rather than smoothed over.
    """
    specimen_probs, specimen_ids = aggregate_to_specimen(val_probs, val_specimen_ids)

    ids = np.asarray(val_specimen_ids)
    unique = np.unique(ids)
    truth = np.empty(len(unique), dtype="int64")
    for k, sid in enumerate(unique):
        vals = np.unique(np.asarray(val_labels)[ids == sid])
        if len(vals) != 1:
            raise ValueError(
                f"validation specimen {sid!r} carries {len(vals)} class labels {vals.tolist()} -- "
                "the manifest is inconsistent and the confusion matrix would be meaningless"
            )
        truth[k] = vals[0]

    assert (specimen_ids == unique).all(), "specimen ordering diverged between probs and labels"
    return confusion_from_validation(specimen_probs, truth, soft=soft)


def _solve_adjusted(
    observed: np.ndarray, M: np.ndarray, method: str, n: int, positive: list[int]
) -> PrevalenceEstimate:
    """Shared solve for ACC and PACC: `p = M^-1 q`, then back onto the simplex.

    `lstsq` rather than `inv`. A confusion matrix estimated from a fold where one class is nearly
    always confused with another is close to singular, and `inv` would return enormous components
    with no warning at all. Least squares degrades instead of exploding, and the simplex projection
    then reports that it had to intervene.
    """
    solution, *_ = np.linalg.lstsq(M, observed, rcond=None)
    prior, left = _project_to_simplex(solution)
    return _as_prevalence(prior, method, n, positive, left_simplex=left)


def adjusted_count(
    specimen_probs: np.ndarray, M: np.ndarray, positive: list[int] | None = None
) -> PrevalenceEstimate:
    """ACC -- invert the hard confusion matrix estimated on validation."""
    positive = positive if positive is not None else positive_indices()
    q = classify_and_count(specimen_probs, positive).prior
    return _solve_adjusted(q, M, "ACC", len(specimen_probs), positive)


def probabilistic_adjusted_count(
    specimen_probs: np.ndarray, M_soft: np.ndarray, positive: list[int] | None = None
) -> PrevalenceEstimate:
    """PACC -- the same solve with mean posteriors on both sides."""
    positive = positive if positive is not None else positive_indices()
    q = specimen_probs.mean(axis=0)
    return _solve_adjusted(q, M_soft, "PACC", len(specimen_probs), positive)


def saerens_em(
    specimen_probs: np.ndarray,
    train_prior: np.ndarray,
    positive: list[int] | None = None,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> PrevalenceEstimate:
    """SLD-EM -- Saerens, Latinne and Decaestecker (2002), Section 2.

    The classifier was trained under one class prior and is being applied where the prior is
    different. Its posteriors therefore encode the *training* prior, and the procedure removes it:

        E-step  reweight each posterior by (current prior / training prior), renormalise
        M-step  set the new prior to the mean of the reweighted posteriors

    iterated to a fixed point. It is ordinary EM on the mixture-proportion likelihood, and it
    increases that likelihood monotonically, so `converged=False` on the returned estimate means
    the iteration cap was hit -- worth knowing, not necessarily worth discarding.

    `train_prior` is the class distribution the model was **fitted** on. Passing the deployment
    prior by mistake makes this function a no-op that looks like it worked, which is exactly the
    failure `test_em_is_a_no_op_when_the_priors_match` pins in the other direction.
    """
    positive = positive if positive is not None else positive_indices()
    train_prior = np.asarray(train_prior, dtype="float64")
    train_prior = np.clip(train_prior, _EPS, None)
    train_prior = train_prior / train_prior.sum()

    prior = train_prior.copy()
    converged = False
    iterations = 0

    for iterations in range(1, max_iter + 1):
        # E-step. The ratio undoes the training prior baked into each posterior and imposes the
        # current estimate in its place.
        weighted = specimen_probs * (prior / train_prior)
        weighted /= np.clip(weighted.sum(axis=1, keepdims=True), _EPS, None)

        # M-step.
        new_prior = weighted.mean(axis=0)
        shift = np.abs(new_prior - prior).max()
        prior = new_prior
        if shift < tol:
            converged = True
            break

    return _as_prevalence(
        prior, "SLD-EM", len(specimen_probs), positive,
        iterations=iterations, converged=converged,
    )


def bootstrap_interval(
    estimate_fn,
    specimen_probs: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 2025,
) -> dict[str, float]:
    """Percentile bootstrap interval for any estimator, resampling **specimens**.

    The resampling unit is the row, and rows are specimens because `aggregate_to_specimen` was
    applied first. Resampling images would treat 2.76 photographs of one shrimp as 2.76
    independent draws and return an interval roughly `sqrt(2.76)` too narrow -- the project's own
    headline finding, committed inside its own uncertainty estimate.

    Percentile rather than normal-approximation, because a prevalence near 0 or 1 has a skewed
    sampling distribution and a symmetric interval would run off the end of [0, 1].

    The interval is only as good as its **coverage**, which is a property of the estimator and the
    sample size rather than of the formula. Experiment 07 measures it against pools of known
    prevalence and reports what it measures. Nothing here should be described as a 95% interval
    until that number says it is one.
    """
    rng = np.random.default_rng(seed)
    n = len(specimen_probs)
    draws = np.empty(n_boot, dtype="float64")
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = estimate_fn(specimen_probs[idx]).prevalence
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "lo": float(lo),
        "hi": float(hi),
        "width": float(hi - lo),
        "boot_sd": float(draws.std(ddof=1)),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
    }
