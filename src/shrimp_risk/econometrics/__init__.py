"""Track A -- Just-Pope econometric risk modelling. **Implemented and run (Stage 6).**

Built out in Stage 6 as `data.py` (survey loading, cleaning, merging, provenance),
`just_pope.py` (the three-step estimator and its diagnostics), `ml_models.py` (the step-3
candidate comparison) and `pdp.py` (partial dependence, gated by `D-017`). The contract
below was written before any of it existed and is kept because it still describes what the
package is for.

This package is declared rather than empty-by-accident. Writing the contract first -- what this
module owns, what it receives, what it produces, what it must not depend on -- gives the code
something to satisfy and makes a later drift visible as a contradiction rather than a surprise.
The vision track was built this way and it is the reason `integration` can be specified before it
exists. (Master Prompt section 2 asks for the same legibility.)

Responsibility
--------------
Estimate income risk and disease-prevalence risk from farm survey covariates, following the
base paper's three-step Just-Pope procedure:

    step 1   y_i = f(x_i, beta) + e_i                     OLS on the level outcome
    step 2   d_i = (y_i - f(x_i, beta_hat))**2            the deviance; E[d|x] = Var(y|x)
    step 3   log(d_i) ~ RF / SVM / Cubist                 then importance + partial dependence

Receives
--------
Cleaned farm-level tables built from IFPRI 2024 (`data/raw/ifpri/`, n=1,221) and, as an
independent validation sample, Hasan & Haque 2020 (n=233).

Produces
--------
Coefficient tables with heteroskedasticity-consistent standard errors, ML performance under
identical folds, variable-importance rankings, partial dependence plots, and the risk estimates
that the Stage 9 sensitivity analysis perturbs.

Depends on
----------
`shrimp_risk.common` for paths and seeding. Nothing in `vision`: the two tracks meet only at
`integration`, which keeps either track testable without the other.

Known design questions, to be decided with evidence rather than assumed
-----------------------------------------------------------------------
Each of these is a fork where the obvious choice and the defensible choice may differ, and where
picking on preference would leave a claim nobody could check. They are listed at the top of the
package so they cannot be quietly settled in passing. (Master Prompt section 9.)
1. **Cubist has no first-class scikit-learn implementation.** The `cubist` PyPI package and
   gradient boosting are the candidate stand-ins; whichever is used must be named on the slide.
2. **Log-OLS on a bounded, zero-inflated prevalence outcome is questionable.** D2's
   `CurrentPrevalance(%)` is 40.3% exact zeros on [0, 100]. Fractional-response and two-part
   hurdle models are the textbook alternatives and must be *compared*, not asserted.
3. **Total farm income has to be constructed**, not read off a column, and reconciled against
   `c1004`. A derived variable must never be presented as an observed one.
"""
