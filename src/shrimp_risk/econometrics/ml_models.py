"""Step 3 of Just-Pope: machine-learning models fitted to the log-deviance.

What step 3 is for, and what it is not for
------------------------------------------
After step 2 there is one number per farm -- `log(d_i)` -- that estimates how *variable* that
farm's outcome is, conditional on its covariates. Step 3 asks which covariates account for that
variability and in what shape. It is a **description of a risk surface**, not a forecast: none
of the claims this project makes require predicting an individual farm's variance, and the log
transform in step 2 would bias such a prediction anyway (see `just_pope.log_deviance`).

That distinction determines how the models are judged. Out-of-sample R-squared on `log(d)` is
reported because a model that cannot predict held-out log-deviance at all has not found any
structure worth interpreting -- but a *low* R-squared here is a substantive finding about how
predictable farm-level risk is, not a bug to be tuned away. The base paper prints no numeric
performance table for its ML stage at all (only the prose claim "low MSE and high R-squared"),
so there is nothing to hit and nothing to match; ours are reported because a claim without a
number is not evidence.

Why these four candidates
-------------------------
The base paper uses Random Forest, SVM and Cubist. All three are implemented here, plus
Gradient Boosting as a fourth. The reasoning is not "more models is better" -- each answers a
different objection:

- **Random Forest** -- the paper's best performer on both outcomes. Bagged trees, low variance,
  no tuning needed to be competitive, handles the binary-heavy design matrix without scaling.
- **SVM (RBF kernel)** -- the paper's second-best on disease risk. A completely different
  inductive bias (global kernel smoother rather than axis-aligned partitions), so agreement
  between it and RF is meaningful rather than a shared artefact of tree-based methods.
- **Cubist** -- Quinlan's rule-based model with linear models in the leaves. The paper's third,
  and the reason its results are interpretable as rules. Available here as a genuine
  implementation (see `D-016`), not a substitute.
- **Gradient Boosting** -- included as the pre-registered fallback in case Cubist proved
  unusable, and kept once it was not, because it is the obvious modern alternative a panel will
  ask about and because a boosted model that beat all three would itself be a finding.

Fairness of the comparison
--------------------------
Every candidate sees *identical folds* generated from one seeded splitter, identical features,
and identical preprocessing except where a model family genuinely requires otherwise (SVR needs
standardised inputs; trees do not and are not given them, since standardising a tree's inputs is
a monotone transform that changes nothing but wastes the reader's attention). This mirrors the
discipline the vision track applies to backbone comparison: a comparison in which conditions
differ in more than one factor cannot attribute a difference to any of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold, GroupShuffleSplit, RepeatedKFold, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

#: The base paper's resampling design: an 80/20 train-holdout split, then 10-fold
#: cross-validation repeated 4 times inside the training portion. Adopted unchanged so the
#: *procedure* is comparable even though the data is not. Repetition matters more here than in
#: the vision track: at n=463 for the disease model a single 10-fold estimate moves noticeably
#: with the fold assignment, and four repeats give a standard deviation across 40 fits that
#: makes the noise visible instead of hidden.
CV_FOLDS = 10
CV_REPEATS = 4
HOLDOUT_FRACTION = 0.20
DEFAULT_SEED = 2025


def make_candidates(seed: int = DEFAULT_SEED) -> dict[str, object]:
    """Build the four step-3 regressors.

    **Hyperparameters are near-library-defaults and deliberately so.** Tuning each model on this
    data would improve every number and destroy the comparison's meaning, because the winner
    would then be whichever model's search space happened to be explored best. The base paper
    reports no hyperparameters at all, so there is nothing to copy. What is defensible is a
    like-for-like comparison at sensible defaults, with the tuning limitation declared. The one
    place defaults are overridden is where a default is unsuitable at this sample size:

    - `RandomForestRegressor(n_estimators=500)` rather than 100: variance of the ensemble
      prediction falls with the number of trees at no cost in bias, and 500 is cheap on a
      1,200-row problem. `min_samples_leaf=5` because a leaf of one observation on a response as
      noisy as a log-deviance is fitting a single chi-square draw.
    - `SVR` inside a `StandardScaler` pipeline: an RBF kernel computes Euclidean distances, so
      an unscaled `stocking_cost_bdt` in the tens of thousands would dominate every binary
      management flag entirely and the model would effectively see one variable. The scaler is
      inside the pipeline, not applied beforehand, so it is refitted within each CV fold and
      cannot leak the fold's test statistics into its training.
    - `GradientBoostingRegressor(max_depth=3, n_estimators=300, learning_rate=0.05)`: a
      conventional slow-and-shallow configuration, which is the low-variance end of boosting and
      the fair counterpart to a large forest.
    - `Cubist(n_committees=5)`: committees are Cubist's own boosting-like averaging; a single
      rule set on 463 rows is high variance. Five is modest and keeps the rule output readable.

    Every model that has a random component is seeded from the same `seed`.
    """
    candidates: dict[str, object] = {
        "RandomForest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=5, random_state=seed, n_jobs=-1,
        ),
        "SVM_RBF": Pipeline([
            ("scale", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")),
        ]),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=seed,
        ),
    }
    try:
        from cubist import Cubist
        candidates["Cubist"] = Cubist(n_rules=100, n_committees=5, random_state=seed, verbose=0)
    except ImportError:
        # Recorded as an absence rather than silently substituted. If Cubist is unavailable the
        # comparison still runs on three models and the results table simply has no Cubist row,
        # which is honest; quietly renaming GradientBoosting to "Cubist substitute" would not be.
        pass
    return candidates


@dataclass
class CandidateResult:
    """Cross-validated and holdout performance for one step-3 model."""

    name: str
    cv_r2_mean: float
    cv_r2_sd: float
    cv_rmse_mean: float
    cv_rmse_sd: float
    cv_mae_mean: float
    holdout_r2: float
    holdout_rmse: float
    fit_seconds: float
    n_fits: int


@dataclass
class Step3Result:
    """Everything step 3 produces for one Just-Pope model."""

    label: str
    n_train: int
    n_holdout: int
    n_features: int
    table: pd.DataFrame
    fitted: dict[str, object] = field(default_factory=dict)
    X_train: pd.DataFrame | None = None
    X_holdout: pd.DataFrame | None = None
    y_train: np.ndarray | None = None
    y_holdout: np.ndarray | None = None
    baseline_rmse: float = float("nan")

    def best(self, metric: str = "cv_r2_mean") -> str:
        return str(self.table.sort_values(metric, ascending=False).iloc[0]["model"])


def evaluate_candidates(
    X: pd.DataFrame,
    y: np.ndarray,
    label: str,
    seed: int = DEFAULT_SEED,
    candidates: dict[str, object] | None = None,
    groups: np.ndarray | None = None,
) -> Step3Result:
    """Fit and score every candidate on identical folds, then on a common holdout.

    **Two levels of evaluation, because they answer different questions.** The repeated K-fold
    scores measure how stable each model is over resamplings of the training data and are what
    the model *ranking* is based on -- a ranking from a single holdout at n=463 would be a
    ranking of luck. The single holdout, never touched during cross-validation, is what the
    reported headline number should be, because it is the only estimate not used to choose
    anything.

    **The holdout split is made once and shared.** Every candidate is trained on the same rows
    and evaluated on the same rows. This is the same principle as the vision track's fixed folds
    and it is the difference between "RF beat SVM" and "RF's split beat SVM's split".

    **A no-skill baseline is computed and reported.** The RMSE of predicting the training mean
    for every holdout row is the number every model must beat to have found anything. Reporting
    R-squared alone hides this: a small positive R-squared and a barely-better-than-mean RMSE
    are the same result, and the second is harder to over-read.

    **`groups` makes both splits cluster-aware.** IFPRI drew its 1,221 farms from 68 clusters of
    roughly 18 (`D-029`), so two farms in one cluster are neighbours sharing water, weather,
    hatchery and extension advice. Splitting them at random puts a farm's neighbours in the
    training set and asks the model to predict the farm -- which is the same structure as
    photographing one shrimp twice and calling it two observations (`D-007`), one dataset over.
    Passing `groups` switches the holdout to `GroupShuffleSplit` and the folds to `GroupKFold`,
    so no cluster is ever split across a boundary.

    It is **off by default**, and that is deliberate rather than lazy: Experiment 03's reported
    result used random folds, and changing the default would silently restate a published number.
    Experiment 10 runs both and reports the difference, which is the only way to know whether the
    design effect mattered here.
    """
    import time

    candidates = candidates or make_candidates(seed)
    y = np.asarray(y, dtype="float64")

    if groups is None:
        X_tr, X_ho, y_tr, y_ho = train_test_split(
            X, y, test_size=HOLDOUT_FRACTION, random_state=seed, shuffle=True
        )
        cv = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=seed)
        fit_groups = None
        n_fits = CV_FOLDS * CV_REPEATS
    else:
        groups = np.asarray(groups)
        if len(groups) != len(X):
            raise ValueError(
                f"{len(groups)} group labels for {len(X)} rows -- a mismatch here would "
                "assign farms to the wrong cluster and make the grouping meaningless"
            )
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=HOLDOUT_FRACTION, random_state=seed)
        tr_idx, ho_idx = next(splitter.split(X, y, groups=groups))
        X_tr, X_ho = X.iloc[tr_idx], X.iloc[ho_idx]
        y_tr, y_ho = y[tr_idx], y[ho_idx]
        fit_groups = groups[tr_idx]
        # GroupKFold is deterministic and cannot be repeated with a different shuffle, so the
        # repeats that `RepeatedKFold` provides are simply unavailable. Reporting the true number
        # of fits keeps the two conditions comparable on their own terms rather than implying an
        # equivalence that is not there.
        cv = GroupKFold(n_splits=min(CV_FOLDS, len(np.unique(fit_groups))))
        n_fits = cv.get_n_splits()

    rows, fitted = [], {}
    for name, model in candidates.items():
        started = time.perf_counter()
        scores = cross_validate(
            clone(model), X_tr, y_tr, cv=cv, groups=fit_groups,
            scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"),
            n_jobs=1, error_score="raise",
        )
        final = clone(model).fit(X_tr, y_tr)
        pred = final.predict(X_ho)
        elapsed = time.perf_counter() - started

        rows.append(CandidateResult(
            name=name,
            cv_r2_mean=float(np.mean(scores["test_r2"])),
            cv_r2_sd=float(np.std(scores["test_r2"], ddof=1)),
            cv_rmse_mean=float(-np.mean(scores["test_neg_root_mean_squared_error"])),
            cv_rmse_sd=float(np.std(scores["test_neg_root_mean_squared_error"], ddof=1)),
            cv_mae_mean=float(-np.mean(scores["test_neg_mean_absolute_error"])),
            holdout_r2=float(r2_score(y_ho, pred)),
            holdout_rmse=float(np.sqrt(mean_squared_error(y_ho, pred))),
            fit_seconds=round(elapsed, 2),
            n_fits=n_fits,
        ))
        fitted[name] = final

    baseline_rmse = float(np.sqrt(mean_squared_error(y_ho, np.full_like(y_ho, y_tr.mean()))))
    table = pd.DataFrame([{"model": r.name, **{k: v for k, v in vars(r).items() if k != "name"}}
                          for r in rows])
    table.insert(1, "response", label)
    table["holdout_rmse_vs_mean_baseline"] = (
        table["holdout_rmse"] / baseline_rmse).round(4)
    table = table.sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)

    return Step3Result(
        label=label, n_train=len(X_tr), n_holdout=len(X_ho), n_features=X.shape[1],
        table=table, fitted=fitted, X_train=X_tr, X_holdout=X_ho,
        y_train=y_tr, y_holdout=y_ho, baseline_rmse=baseline_rmse,
    )


def importance_table(
    result: Step3Result,
    n_repeats: int = 20,
    seed: int = DEFAULT_SEED,
    labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Permutation importance for every fitted candidate, on the holdout set.

    **Why permutation importance rather than each model's built-in measure.** The base paper
    reads variable importance off Random Forest and Cubist directly. Those measures are not
    comparable across model families -- RF's impurity decrease, Cubist's rule-usage counts and
    an SVM's (absent) native importance are three different quantities, and ranking them against
    each other would be meaningless. Permutation importance asks the same question of every
    model: *how much worse does this model get when this column is shuffled?* That gives one
    scale on which all four can be compared, which is the comparison the project actually needs.

    **Why on the holdout and not on the training data.** Permuting a column on data the model
    has memorised measures how much the model *used* the column, not how much the column
    *matters*. On held-out data it measures the loss of genuine predictive value. For a
    500-tree forest on 1,200 rows the difference between the two is large.

    **A known limitation, declared rather than discovered later.** Permutation importance is
    unreliable when features are strongly correlated: shuffling one of a correlated pair leaves
    the model able to recover the information from the other, so both look unimportant. The VIF
    table from step 1 is the check on that, and any variable pair with high VIF must have its
    importance read with this in mind.
    """
    rows = []
    for name, model in result.fitted.items():
        pi = permutation_importance(
            model, result.X_holdout, result.y_holdout,
            n_repeats=n_repeats, random_state=seed, scoring="r2", n_jobs=1,
        )
        for j, col in enumerate(result.X_holdout.columns):
            rows.append({
                "response": result.label,
                "model": name,
                "variable": col,
                "label": (labels or {}).get(col, col),
                "importance_mean": float(pi.importances_mean[j]),
                "importance_sd": float(pi.importances_std[j]),
            })
    out = pd.DataFrame(rows)
    out["rank_within_model"] = (
        out.groupby("model")["importance_mean"].rank(ascending=False, method="min").astype(int)
    )
    return out.sort_values(["model", "rank_within_model"]).reset_index(drop=True)


def importance_overlap(importance: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Pairwise agreement between models on their top-N variables.

    The base paper's Table 5 reports exactly this kind of overlap between its econometric and ML
    rankings (8 of 13 income variables, 15 of 20 disease variables) and treats agreement as
    corroboration. The logic is sound and worth reproducing: two models with different inductive
    biases converging on the same variables is stronger evidence than either alone, and
    *disagreement* is a warning that at least one ranking is an artefact of its model family.
    """
    models = sorted(importance["model"].unique())
    tops = {
        m: set(importance[(importance.model == m) & (importance.rank_within_model <= top_n)]
               ["variable"])
        for m in models
    }
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            shared = tops[a] & tops[b]
            rows.append({
                "model_a": a, "model_b": b, "top_n": top_n,
                "n_shared": len(shared),
                "share": round(len(shared) / top_n, 3),
                "shared_variables": "; ".join(sorted(shared)),
            })
    return pd.DataFrame(rows)
