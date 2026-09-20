"""The bridge -- per-image posteriors to farm-level prevalence. **Built in Stage 8.**

The contract below was written in Stage 5, before any of this package existed, and is kept
unchanged apart from this header: it still describes what the package is for, and leaving it as
written makes it possible to check the implementation against the promise rather than against a
description edited afterwards to match. `econometrics` was declared the same way for the same
reason.

What now exists, and where
--------------------------
    calibration.py     temperature scaling, ECE/MCE/Brier, reliability bins
    quantification.py  CC, PCC, ACC, PACC and Saerens-Latinne-Decaestecker EM, with a
                       specimen-resampled bootstrap interval
    pools.py           pseudo-farm construction at a prevalence chosen in advance
    sensitivity.py     Stage 9 -- propagating the measured prevalence error into Track A

The experiments that exercise them are `scripts/exp06_calibrate_bridge.py` (the calibration step
the contract's final paragraph demands) and `scripts/exp07_prevalence_bridge.py` (the bias
comparison it promises). Both write to `reports/tables/`, and `tests/test_quantification.py`
pins the estimators against answers derived on paper rather than against their own output.

One thing the contract got backwards, settled by measurement
------------------------------------------------------------
The contract below says the bridge consumes **calibrated** posteriors, and `D-028` promoted that
from a design intent to a prerequisite. Experiment 07 measured it and the direction is wrong:
temperature scaling makes the best estimator **worse** -- SLD-EM's mean |bias| goes from 0.0391 on
raw posteriors to 0.0598 on calibrated ones, and PCC's from 0.0676 to 0.1022. The mechanism is
measured rather than guessed: the fitted temperature overshoots into *under*confidence (`D-034`),
and both estimators consume posterior magnitudes.

**The bridge runs on RAW posteriors** (`D-041`). The contract's wording is left as written because
it is the promise this package is checked against, and editing it to match the outcome would erase
the fact that the outcome was a surprise.

One thing the original contract did not anticipate
--------------------------------------------------
Prevalence had to become a property of **specimens**, not of images. The contract speaks of
"per-image posteriors" throughout, which is what the bridge receives -- but quantifying over
images would weight a shrimp photographed five times 2.5x more heavily than one photographed
twice, and the corpus is 1,149 photographs of 416 animals. Every estimator therefore consumes
specimen-level posteriors from `quantification.aggregate_to_specimen`, and every bootstrap
resamples specimens. It is the same conflation `D-007` found in the split protocol, one layer up.

Responsibility
--------------
Convert calibrated per-image class posteriors into an unbiased farm-level disease-prevalence
estimate with an uncertainty interval, then hand that estimate to Track A as a covariate.

Why this is not just averaging
------------------------------
Counting positive predictions and dividing is BIASED whenever the classifier is imperfect. For
a binary case with sensitivity `tpr` and specificity `tnr`:

    E[p_naive] = p * tpr + (1 - p) * (1 - tnr)

which equals the true `p` only when `tpr + tnr == 2`. The bias is systematic, not noise, so it
does not shrink as more images are collected. The corrections are the adjusted-count estimator
and the Saerens-Latinne-Decaestecker (2002) EM procedure for prior shift.

Saerens et al. (2002), read in full on 2026-08-27, state their condition as the **within-class
densities p(x|omega_i) being unchanged** from training to deployment -- only the class proportions
shift. They also compare their EM procedure against the confusion-matrix method (what this contract
calls the adjusted-count estimator) and report EM as superior for prior estimation, so EM is the
one to lead with here.

Calibration is a second requirement, and it is ours rather than theirs: both corrections consume
the classifier's *estimated* posteriors, so their quality propagates. A leakage-inflated model is
overconfident,
which is why Stage 4's specimen-level protocol is a prerequisite for this module rather than an
unrelated result.

Receives
--------
`experiments/runs/*/**_predictions.npz` -- per-image posteriors plus the specimen id of each
test image, written by every experiment run for exactly this purpose.

Produces
--------
Prevalence estimates (naive and corrected) with intervals, and the bias comparison between
them, measured on pseudo-farm pools constructed at KNOWN true prevalence.

The limitation this module is built around
------------------------------------------
The image dataset carries no farm identifier, so a real per-farm prevalence cannot be computed
from it. Pools of specimens at known prevalence are therefore constructed deliberately -- which
makes the experiment verifiable, because the answer that should be recovered is known in
advance. That is a designed experiment, but the absence of real farm IDs is a genuine
limitation and is declared in every result that depends on it.
"""
