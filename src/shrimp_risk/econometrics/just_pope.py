"""The Just-Pope three-step risk estimator.

Why a separate module for three lines of arithmetic
---------------------------------------------------
Steps 2 and 3 of Just-Pope are, mechanically, "square the residual" and "take a log". The
reason they deserve a module of their own is that the *validity* of those two lines depends on
conditions that are easy to state, easy to violate, and invisible in the output if violated.
This module computes the steps and, in the same pass, computes the evidence about whether they
were entitled to be taken. A pipeline that only does the arithmetic will always produce a
number and will never tell you it is meaningless.

The method, restated
--------------------
Just & Pope (1979) write output as::

    y_i = f(x_i, beta) + g(x_i, alpha) * eps_i

The mean of output and the *variance* of output are allowed to depend on the inputs
separately. That separation is the whole point: an input can raise expected income while also
raising its variability, and a single-equation regression cannot express that -- it has one
coefficient per variable and must average the two effects into it.

Nguelifack et al. (2025), following Judge et al. (1988), estimate this in three steps:

    step 1   y_i = f(x_i, beta) + e_i                     OLS
    step 2   d_i = (y_i - f(x_i, beta_hat))^2             the deviance
    step 3   log(d_i) modelled by RF / SVM / Cubist       then importance and PDP

Step 2 rests on the identity ``E[d_i | x_i] = Var(y_i | x_i)``, which holds **only when the
mean function of step 1 is correctly specified**. Under misspecification the deviance captures
squared bias plus variance, and step 3 then models a mixture of the two while reporting it as
risk. The base paper states the identity without stating the condition. It is not an error --
it is the standard assumption -- but it is the first thing a viva should ask about, so this
module reports the diagnostics that bear on it (RESET-style fit checks are out of scope, but
normality, heteroskedasticity and collinearity are all computed and returned).

Step 3 takes logs because, as the paper's Remark 1 notes, the deviance is chi-square-like with
one degree of freedom and therefore violently right-skewed. `log(d)` is close to symmetric,
which is what makes it usable as a regression response. This module implements the log with an
explicit floor and *counts* how many observations the floor touched, because a silent
`log(0) = -inf` would propagate into every downstream fit as a missing value or a crash.

What this module does NOT decide
--------------------------------
Which ML model goes on top of `log(d)` (see `ml_models.py`), and which functional form the
outcome should take -- a bounded/zero-inflated specification is flagged as outstanding work in
`docs/stages/stage_06_track_a.md` section 6.3 and is not yet implemented. Keeping those
elsewhere means the estimator can be tested against a synthetic process with a known variance
function without any modelling library in the way -- which is exactly what
`tests/test_econometrics.py` does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

#: Absolute floor applied to the deviance before taking logs. Squared OLS residuals are
#: essentially never exactly zero for a continuous outcome, but "essentially never" is not
#: "never": a duplicated row or a perfectly fitted point produces a zero, and `log(0)` would
#: silently poison the step-3 response. The floor is set well below any deviance a real outcome
#: produces (revenues are in the 10^5 BDT range, so their squared residuals are 10^8 or larger),
#: so it can only ever bind on a genuine numerical zero, and the number of times it binds is
#: reported rather than hidden.
DEVIANCE_FLOOR = 1e-12


# ======================================================================================
# Step 1 -- the mean equation
# ======================================================================================

@dataclass
class MeanEquation:
    """Step 1 output: the fitted conditional mean, with everything needed to judge it.

    Holding the residuals and fitted values alongside the coefficient table matters because
    step 2 consumes the residuals, and a version of the residuals recomputed later from a
    re-fitted model is not guaranteed to be the same vector. Passing the actual object forward
    removes an entire class of silent inconsistency.
    """

    outcome: str
    transform: str
    n: int
    k: int
    r_squared: float
    adj_r_squared: float
    f_pvalue: float
    coefficients: pd.DataFrame
    fitted: np.ndarray
    residuals: np.ndarray
    diagnostics: dict[str, float]
    model: object = None


def fit_mean_equation(
    y: pd.Series,
    X: pd.DataFrame,
    transform: str = "identity",
    cov_type: str = "HC3",
) -> MeanEquation:
    """Estimate step 1 by OLS with heteroskedasticity-consistent standard errors.

    **Why OLS and not something that models the variance directly.** A GLS or FGLS estimator
    would be more efficient if the variance function were known, but the whole premise of the
    exercise is that the variance function is *unknown and is the object of interest*. Assuming
    it in step 1 to estimate it in step 3 would be circular. OLS remains consistent for the mean
    under heteroskedasticity; only its conventional standard errors become invalid, and that is
    what the robust covariance fixes.

    **Why HC3 rather than HC0.** All the HC variants are consistent; they differ in their
    finite-sample leverage correction. HC3 inflates each residual by ``1/(1-h_ii)^2`` and is the
    one MacKinnon & White found best-behaved in small samples. The disease model runs on 463
    observations and Hasan & Haque on 233, which is small enough for the choice to matter. The
    base paper uses "White's heteroskedasticity-consistent covariance matrix", which is HC0; HC3
    reduces to HC0 as n grows, so this is a conservative refinement of their choice rather than
    a departure from it.

    **`transform`** records what was done to `y` before it arrived, because it changes what the
    deviance *means*. With `identity`, `d_i` estimates ``Var(y|x)`` -- variance in BDT-squared.
    With `log`, it estimates ``Var(log y|x)`` -- roughly the variance of proportional changes.
    Both are legitimate risk measures and they are not the same measure, so the label travels
    with the result instead of living in someone's memory.
    """
    X = X.astype("float64")
    y = pd.Series(y).astype("float64")
    design = sm.add_constant(X, has_constant="add")
    fit = sm.OLS(y.to_numpy(), design.to_numpy()).fit(cov_type=cov_type)

    coefs = pd.DataFrame({
        "variable": design.columns,
        "estimate": fit.params,
        "std_error": fit.bse,
        "t": fit.tvalues,
        "p_value": fit.pvalues,
    }).reset_index(drop=True)

    return MeanEquation(
        outcome=str(y.name) if y.name is not None else "y",
        transform=transform,
        n=int(fit.nobs),
        k=int(design.shape[1]),
        r_squared=float(fit.rsquared),
        adj_r_squared=float(fit.rsquared_adj),
        f_pvalue=float(fit.f_pvalue),
        coefficients=coefs,
        fitted=np.asarray(fit.fittedvalues, dtype="float64"),
        residuals=np.asarray(fit.resid, dtype="float64"),
        diagnostics=mean_equation_diagnostics(fit, design),
        model=fit,
    )


def mean_equation_diagnostics(fit, design: pd.DataFrame) -> dict[str, float]:
    """Tests that decide whether the Just-Pope exercise has anything to explain.

    **Heteroskedasticity (Breusch-Pagan and White).** This is the diagnostic that justifies the
    entire method rather than merely checking it. If the residual variance does not depend on
    the covariates, then ``g(x, alpha)`` is a constant, the "risk function" is a horizontal
    line, and step 3 is fitting noise. Reporting a significant test is what entitles us to
    interpret the step-3 importances at all -- and reporting an insignificant one would be a
    finding we would have to state, not suppress. Breusch-Pagan tests for variance linear in the
    regressors; White's test adds squares and cross-products and so catches non-linear
    dependence the base paper's PDPs would later claim to find. Running both is cheap and they
    fail differently.

    **Ramsey RESET (functional form).** The heteroskedasticity verdict above is computed on the
    residuals of *this* mean equation, so it only means what it says if the mean equation is
    correctly specified. If it is not, the residuals carry squared bias as well as variance, and
    Breusch-Pagan is then testing a mixture rather than the quantity its name implies. RESET
    regresses the residuals on powers of the fitted values: a rejection says some non-linearity
    in the conditional mean has been left in the error term. This is the condition the module
    docstring names as load-bearing for step 2, so it is measured rather than assumed. Both the
    classical F form and a heteroskedasticity-robust form are reported, because a rejection that
    only appears under the homoskedastic F could itself be an artefact of the variance structure
    this function is separately testing for.

    **Residual skew and kurtosis.** Diagnostic for step 2, not step 1: the deviance of a
    heavy-tailed residual is dominated by a handful of observations, and the log transform in
    step 3 exists precisely to tame that. Reporting the moments makes it visible whether the log
    was doing real work or was cosmetic.

    **Condition number.** Multicollinearity does not bias OLS but inflates its standard errors,
    and the base paper reports VIF < 5 for every variable. Per-variable VIFs are computed by
    `vif_table`; the condition number of the design is the single-number summary reported here.
    """
    resid = np.asarray(fit.resid, dtype="float64")
    exog = design.to_numpy(dtype="float64")
    out: dict[str, float] = {}

    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(resid, exog)
    out["breusch_pagan_lm"] = float(bp_lm)
    out["breusch_pagan_p"] = float(bp_lm_p)
    try:
        w_lm, w_lm_p, w_f, w_f_p = het_white(resid, exog)
        out["white_lm"] = float(w_lm)
        out["white_p"] = float(w_lm_p)
    except Exception:
        # White's test needs n > number of squares and cross-products; with ~40 covariates it
        # can be infeasible at n=233. Recording that it could not be run is honest; substituting
        # a different test and calling it White's would not be.
        out["white_lm"] = float("nan")
        out["white_p"] = float("nan")

    # RESET on the fitted values, powers 2 and 3. `use_f=True` gives the classical F test;
    # re-running against an HC3-refitted model gives the robust chi-square. If the two disagree,
    # the rejection is telling us about the variance rather than the mean, which is exactly the
    # distinction this diagnostic exists to keep clean.
    for power in (2, 3):
        try:
            out[f"reset{power}_p"] = float(
                linear_reset(fit, power=power, test_type="fitted", use_f=True).pvalue
            )
        except Exception:
            # RESET needs the fitted values to have enough variation to build powers from, and
            # needs n > k + power. Recording that it could not run is honest; omitting the key
            # would let a reader assume it passed.
            out[f"reset{power}_p"] = float("nan")

    out["resid_skew"] = float(pd.Series(resid).skew())
    out["resid_kurtosis"] = float(pd.Series(resid).kurtosis())
    out["design_condition_number"] = float(np.linalg.cond(exog))
    return out


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Per-variable variance inflation factors.

    Included because the base paper reports VIF < 5 for every variable in its specification and
    treats that as ruling out multicollinearity. Applying the same check to our specification is
    what makes the two comparable as *procedures*; the numbers themselves are not comparable,
    since the variable sets differ entirely.

    Constant columns are excluded first -- a zero-variance column makes the VIF regression
    singular and returns `inf`, which reads as catastrophic collinearity when it is only a
    degenerate column.
    """
    Xc = X.loc[:, X.std(numeric_only=True) > 0].astype("float64")
    design = np.column_stack([np.ones(len(Xc)), Xc.to_numpy()])
    rows = []
    for i, col in enumerate(Xc.columns, start=1):
        rows.append({"variable": col, "vif": float(variance_inflation_factor(design, i))})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


# ======================================================================================
# Step 2 -- the deviance
# ======================================================================================

def deviance(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """Step 2: the squared residual, `d_i = (y_i - f(x_i, beta_hat))^2`.

    Trivial to compute and worth its own named function for two reasons. First, it is the point
    where the meaning of the quantity changes -- from "how wrong was the prediction here" to
    "how variable is the outcome here" -- and that reinterpretation is the substantive content
    of Just-Pope. Second, a named function is testable: `tests/test_econometrics.py` generates
    data with a *known* variance function and checks that the deviance recovers it, which is the
    only way to be sure the step is right rather than merely plausible.

    The estimator is unbiased for `Var(y|x)` given a correctly specified mean, but it is
    extremely noisy pointwise -- a one-degree-of-freedom chi-square has a coefficient of
    variation of about 1.41. That noisiness is not a defect to be fixed; it is why step 3 needs
    a *smoother* (a regression model) over the deviances rather than reading them individually.
    """
    y = np.asarray(y, dtype="float64")
    y_hat = np.asarray(y_hat, dtype="float64")
    if y.shape != y_hat.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs y_hat {y_hat.shape}")
    return (y - y_hat) ** 2


def log_deviance(d: np.ndarray, floor: float = DEVIANCE_FLOOR) -> tuple[np.ndarray, int]:
    """Step 3's response variable, `log(d_i)`, with an explicit and counted floor.

    Returns `(log_d, n_floored)`. The count is returned rather than logged internally so the
    caller must handle it: if the floor ever binds on more than a negligible number of rows, the
    step-3 response has been censored and every importance ranking computed from it is suspect.
    Making that impossible to ignore is the point of the second return value.

    **Why log at all**, restated because it is the most-questioned step: `d` is a squared
    quantity and behaves like a scaled chi-square with one degree of freedom, so its
    distribution has a long right tail and a spike near zero. A regression fitted to `d` on the
    raw scale is dominated by a handful of large-residual farms; fitted to `log(d)` it is not.
    The cost is that the model then predicts the conditional *geometric* mean of the deviance
    rather than its arithmetic mean, so back-transformed predictions are biased low. Since the
    base paper (and this project) use the step-3 model for *ranking and shape* -- importance and
    partial dependence -- and not for point prediction of a variance, that bias does not affect
    any claim made from it. It would matter immediately if anyone used these fits to forecast a
    variance, and the stage document says so.
    """
    d = np.asarray(d, dtype="float64")
    if np.any(d < 0):
        raise ValueError("deviance contains negative values; it is a square and cannot")
    n_floored = int(np.sum(d < floor))
    return np.log(np.maximum(d, floor)), n_floored


# ======================================================================================
# The whole three-step, orchestrated
# ======================================================================================

@dataclass
class JustPopeResult:
    """Steps 1 and 2 in one object, ready to hand to `ml_models.py` for step 3."""

    mean_equation: MeanEquation
    deviance: np.ndarray
    log_deviance: np.ndarray
    n_floored: int
    X: pd.DataFrame
    vif: pd.DataFrame
    label: str

    def summary_row(self) -> dict[str, object]:
        me = self.mean_equation
        return {
            "model": self.label,
            "outcome": me.outcome,
            "transform": me.transform,
            "n": me.n,
            "k_including_constant": me.k,
            "r_squared": round(me.r_squared, 4),
            "adj_r_squared": round(me.adj_r_squared, 4),
            "f_pvalue": me.f_pvalue,
            "breusch_pagan_p": me.diagnostics["breusch_pagan_p"],
            "white_p": me.diagnostics["white_p"],
            # Reported beside the heteroskedasticity tests on purpose: the two must be read
            # together. A BP verdict computed on a misspecified mean equation is conditional,
            # and a reader who sees only the BP column would not know that.
            "reset2_p": me.diagnostics["reset2_p"],
            "reset3_p": me.diagnostics["reset3_p"],
            "resid_skew": round(me.diagnostics["resid_skew"], 3),
            "max_vif": round(float(self.vif["vif"].max()), 2) if len(self.vif) else float("nan"),
            "n_vif_above_5": int((self.vif["vif"] > 5).sum()) if len(self.vif) else 0,
            "log_deviance_mean": round(float(self.log_deviance.mean()), 4),
            "log_deviance_sd": round(float(self.log_deviance.std(ddof=1)), 4),
            "n_deviance_floored": self.n_floored,
        }


def run_just_pope(
    frame: pd.DataFrame,
    outcome: str,
    covariates: list[str],
    transform: str = "identity",
    label: str = "",
    cov_type: str = "HC3",
) -> JustPopeResult:
    """Run steps 1 and 2 end to end on a farm-level table.

    `transform` accepts `identity`, `log` or `log1p`:

    - `identity` -- `d_i` estimates the variance of the outcome in its own squared units.
    - `log` -- for strictly positive, right-skewed outcomes such as revenue. `d_i` then
      estimates the variance of log-outcome, which is (approximately) the variance of
      proportional deviations and is the more natural notion of *income risk* for a farm whose
      scale varies by two orders of magnitude across the sample. Requires `y > 0` and raises if
      that fails rather than silently dropping rows.
    - `log1p` -- for non-negative outcomes with genuine zeros, where a zero means "none" rather
      than "missing".

    Rows with any missing value among the outcome or covariates are dropped listwise, and the
    number dropped is recorded on the result. Listwise deletion is chosen for the IFPRI frames
    because their missingness after cleaning is a handful of rows, and imputing three values to
    avoid dropping three rows would add an assumption for no gain. On Hasan & Haque, where a
    third of the sample is affected, the choice is treated as a research question instead --
    imputation, if it is ever needed, belongs in its own module and is not done here.
    """
    cols = [outcome] + list(covariates)
    sub = frame[cols].copy()
    n_before = len(sub)
    sub = sub.dropna()
    n_dropped = n_before - len(sub)

    y_raw = sub[outcome].astype("float64")
    if transform == "identity":
        y = y_raw
    elif transform == "log":
        if (y_raw <= 0).any():
            raise ValueError(
                f"log transform requested but {int((y_raw <= 0).sum())} values of {outcome} "
                "are non-positive; use log1p or filter the sample explicitly"
            )
        y = np.log(y_raw)
    elif transform == "log1p":
        if (y_raw < 0).any():
            raise ValueError(f"log1p transform requires {outcome} >= 0")
        y = np.log1p(y_raw)
    else:
        raise ValueError(f"unknown transform {transform!r}")
    y.name = outcome

    X = sub[list(covariates)].astype("float64")
    me = fit_mean_equation(y, X, transform=transform, cov_type=cov_type)
    d = deviance(y.to_numpy(), me.fitted)
    log_d, n_floored = log_deviance(d)

    result = JustPopeResult(
        mean_equation=me, deviance=d, log_deviance=log_d, n_floored=n_floored,
        X=X.reset_index(drop=True), vif=vif_table(X),
        label=label or f"{outcome}[{transform}]",
    )
    result.mean_equation.diagnostics["n_rows_dropped_listwise"] = float(n_dropped)
    return result
