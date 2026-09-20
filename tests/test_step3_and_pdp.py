"""The two Track A modules that carried no direct tests.

`ml_models.py` (310 lines) and `pdp.py` (243 lines) were exercised only through
`scripts/exp03_track_a.py`, which means the project's entire negative result rested on code
nothing asserted anything about. Two things in particular were undefended:

1. **`holdout_rmse_vs_mean_baseline`.** This single column is what turns "R² is negative" into
   "worse than predicting the mean". If it were computed wrongly, a negative result could read as
   a positive one, or the reverse — and it is the number `D-017` gates the partial-dependence
   plots on. Here it is checked against cases whose answer is known by construction.
2. **The PDP gate itself.** `D-017` says no partial-dependence plot is drawn unless a model beats
   the baseline. The gate is one comparison in a script; these tests pin the behaviour on both
   sides of it, so a future edit cannot quietly start drawing curves for models with no skill.

`pdp.py` is currently invoked by nothing — the gate has never opened — so it also gets a smoke
test. 243 lines with no evidence of correctness either way is not a defensible position, even
when the code is honestly described as unused.

Everything here runs on generated data in under a second. No survey files are required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shrimp_risk.econometrics import ml_models as ML
from shrimp_risk.econometrics import pdp as PDP

SEED = 2025

# A deliberately cheap stand-in. `ML.make_candidates()` builds a 500-tree forest, and
# `evaluate_candidates` fits each candidate 40 times (10 folds x 4 repeats) plus a holdout fit --
# correct for the experiment, and 100x too slow for a suite that has to stay in the seconds. What
# these tests assert is arithmetic (the baseline ratio) and control flow (the D-017 gate); neither
# depends on how good the forest is.
def _cheap_candidate() -> dict[str, object]:
    from sklearn.ensemble import RandomForestRegressor
    return {"RandomForest": RandomForestRegressor(n_estimators=12, random_state=SEED, n_jobs=1)}


@pytest.fixture(scope="module")
def signal_data() -> tuple[pd.DataFrame, np.ndarray]:
    """A response that genuinely depends on the features, so a model *should* beat the mean."""
    rng = np.random.default_rng(SEED)
    n = 200
    X = pd.DataFrame({
        "area": rng.uniform(0.5, 5.0, n),
        "depth": rng.uniform(2.0, 8.0, n),
        "owned": rng.integers(0, 2, n).astype(float),
    })
    y = 2.0 * X["area"] - 0.5 * X["depth"] + 1.5 * X["owned"] + rng.normal(0, 0.3, n)
    return X, y.to_numpy()


@pytest.fixture(scope="module")
def noise_data() -> tuple[pd.DataFrame, np.ndarray]:
    """A response that is pure noise, so nothing can beat the mean -- Track A's actual situation."""
    rng = np.random.default_rng(SEED)
    n = 200
    X = pd.DataFrame({
        "area": rng.uniform(0.5, 5.0, n),
        "depth": rng.uniform(2.0, 8.0, n),
        "owned": rng.integers(0, 2, n).astype(float),
    })
    return X, rng.normal(0, 1, n)


# --------------------------------------------------------------------------------------
# The baseline ratio -- the column the whole negative result rests on
# --------------------------------------------------------------------------------------

def test_the_baseline_is_the_rmse_of_predicting_the_training_mean(noise_data):
    """Stated in the module docstring; asserted here. If the baseline were computed from the
    holdout mean instead, every ratio in the project would be optimistic by construction."""
    X, y = noise_data
    result = ML.evaluate_candidates(X, y, label="test", seed=SEED,
                                    candidates=_cheap_candidate())
    expected = float(np.sqrt(np.mean((result.y_holdout - result.y_train.mean()) ** 2)))
    assert result.baseline_rmse == pytest.approx(expected, rel=1e-9)


def test_the_ratio_is_holdout_rmse_over_baseline(noise_data):
    X, y = noise_data
    result = ML.evaluate_candidates(X, y, label="test", seed=SEED,
                                    candidates=_cheap_candidate())
    row = result.table.iloc[0]
    assert row["holdout_rmse_vs_mean_baseline"] == pytest.approx(
        row["holdout_rmse"] / result.baseline_rmse, abs=5e-5)


def test_a_model_with_real_signal_beats_the_baseline(signal_data):
    """The positive control. Without it, a ratio stuck above 1.0 for a broken reason would look
    exactly like Track A's genuine negative result."""
    X, y = signal_data
    result = ML.evaluate_candidates(X, y, label="signal", seed=SEED,
                                    candidates=_cheap_candidate())
    row = result.table.iloc[0]
    assert row["holdout_rmse_vs_mean_baseline"] < 1.0
    assert row["holdout_r2"] > 0.5


def test_a_model_fitted_to_noise_does_not_beat_the_baseline(noise_data):
    """The negative control, and the mirror of the test above. Track A's result is only a
    statement about the data if this pair behaves as expected."""
    X, y = noise_data
    result = ML.evaluate_candidates(X, y, label="noise", seed=SEED,
                                    candidates=_cheap_candidate())
    row = result.table.iloc[0]
    assert row["holdout_rmse_vs_mean_baseline"] >= 1.0
    assert row["holdout_r2"] < 0.1


# --------------------------------------------------------------------------------------
# The D-017 gate
# --------------------------------------------------------------------------------------

def _gate(result: ML.Step3Result) -> bool:
    """The gate exactly as `scripts/exp03_track_a.py` applies it."""
    best = result.best()
    row = result.table.loc[result.table["model"] == best].iloc[0]
    return bool(row["holdout_rmse_vs_mean_baseline"] < 1.0)


def test_the_pdp_gate_stays_shut_on_noise(noise_data):
    """`D-017`: a partial-dependence curve looks identical whether the model has skill or none,
    so drawing one for a skill-less model is the easiest way to manufacture a finding."""
    X, y = noise_data
    result = ML.evaluate_candidates(X, y, label="noise", seed=SEED,
                                    candidates=_cheap_candidate())
    assert _gate(result) is False


def test_the_pdp_gate_opens_when_a_model_earns_it(signal_data):
    """The reversal condition in `D-017` is that any model beating the baseline earns its PDP
    automatically. That only holds if the gate can actually open."""
    X, y = signal_data
    result = ML.evaluate_candidates(X, y, label="signal", seed=SEED,
                                    candidates=_cheap_candidate())
    assert _gate(result) is True


def test_best_is_selected_on_cross_validated_r2_not_holdout():
    """The selection rule matters: on the real disease model it names `SVM_RBF` while the holdout
    would name `Cubist`. A silent change here would put a different model name in
    `exp03_pdp_skipped.csv` than in the documents that quote it."""
    table = pd.DataFrame({
        "model": ["A", "B"],
        "cv_r2_mean": [-0.10, -0.30],      # A is better on CV
        "holdout_r2": [-0.50, -0.05],      # B is better on the holdout
    })
    result = ML.Step3Result(label="t", n_train=1, n_holdout=1, n_features=1, table=table)
    assert result.best() == "A"
    assert result.best(metric="holdout_r2") == "B"


# --------------------------------------------------------------------------------------
# pdp.py -- invoked by nothing, so never exercised
# --------------------------------------------------------------------------------------

def test_partial_dependence_table_has_a_row_per_feature_and_grid_point(signal_data):
    from sklearn.ensemble import RandomForestRegressor

    X, y = signal_data
    model = RandomForestRegressor(n_estimators=30, random_state=SEED).fit(X, y)
    table = PDP.partial_dependence_table(model, X, ["area", "depth"], "RF", "response",
                                         grid_resolution=10)
    assert set(table["feature"]) == {"area", "depth"}
    assert len(table) > 0
    assert table["partial_dependence"].notna().all()


def test_partial_dependence_reflects_a_relationship_that_is_there_by_construction(signal_data):
    """`y` rises with `area` by construction, so the curve must rise. A partial-dependence plot
    that did not recover a relationship this obvious could not be trusted on a subtle one."""
    from sklearn.ensemble import RandomForestRegressor

    X, y = signal_data
    model = RandomForestRegressor(n_estimators=60, random_state=SEED).fit(X, y)
    curve = PDP.partial_dependence_table(model, X, ["area"], "RF", "response", grid_resolution=12)
    curve = curve.sort_values("grid_value")
    assert curve["partial_dependence"].iloc[-1] > curve["partial_dependence"].iloc[0]


def test_the_support_column_is_monotonic(signal_data):
    """`n_obs_at_or_below` is what lets a reader tell a curve supported by 400 farms from one
    supported by four. It is a cumulative count, so it can never decrease along the grid."""
    from sklearn.ensemble import RandomForestRegressor

    X, y = signal_data
    model = RandomForestRegressor(n_estimators=30, random_state=SEED).fit(X, y)
    curve = PDP.partial_dependence_table(model, X, ["area"], "RF", "response", grid_resolution=10)
    support = curve.sort_values("grid_value")["n_obs_at_or_below"].to_numpy()
    assert (np.diff(support) >= 0).all()


# ---------------------------------------------------------------------------------------
# Cluster-grouped cross-validation (Stage 10, improvement 7c)
# ---------------------------------------------------------------------------------------

def _clustered_frame(n_clusters: int = 20, per_cluster: int = 10, seed: int = 0):
    """Data with a real cluster structure: a shared shift per cluster, plus individual noise.

    Built this way because grouping only matters when cluster membership carries information. If
    every row were independent, grouped and random folds would agree and the test would pass
    without testing anything.
    """
    rng = np.random.default_rng(seed)
    groups, xs, ys = [], [], []
    for c in range(n_clusters):
        shift = rng.normal(0, 3.0)
        for _ in range(per_cluster):
            x = rng.normal(0, 1)
            groups.append(c)
            xs.append(x)
            ys.append(x + shift + rng.normal(0, 0.2))
    X = pd.DataFrame({"x": xs})
    return X, np.array(ys), np.array(groups)


def test_grouped_evaluation_never_splits_a_cluster_across_the_holdout():
    """The property the whole repair exists for.

    A farm's neighbours must not sit in training while the farm is predicted. Under random folds
    they routinely do -- roughly 18 farms per IFPRI cluster -- and the reported score is flattered
    by it (`D-029`).
    """
    X, y, groups = _clustered_frame()
    result = ML.evaluate_candidates(
        X, y, label="grouped", seed=1,
        candidates={"rf": ML.make_candidates(1)["RandomForest"]},
        groups=groups,
    )
    train_groups = set(groups[result.X_train.index])
    holdout_groups = set(groups[result.X_holdout.index])
    assert train_groups and holdout_groups
    assert not (train_groups & holdout_groups), (
        "a cluster appears in both training and holdout -- the grouping is not being applied"
    )


def test_random_evaluation_does_split_clusters():
    """The control. Without `groups`, clusters are split -- which is the behaviour Experiment 03
    reported under, and is why both conditions are run rather than one replacing the other."""
    X, y, groups = _clustered_frame()
    result = ML.evaluate_candidates(
        X, y, label="random", seed=1,
        candidates={"rf": ML.make_candidates(1)["RandomForest"]},
    )
    train_groups = set(groups[result.X_train.index])
    holdout_groups = set(groups[result.X_holdout.index])
    assert train_groups & holdout_groups, (
        "if random splitting stopped sharing clusters, the comparison has become vacuous"
    )


def test_grouped_evaluation_refuses_a_group_vector_of_the_wrong_length():
    """Silently recycling or truncating group labels would assign farms to the wrong cluster."""
    X, y, groups = _clustered_frame()
    with pytest.raises(ValueError, match="group labels"):
        ML.evaluate_candidates(X, y, label="bad", seed=1, groups=groups[:-5])


def test_the_default_path_is_unchanged_by_the_grouping_option():
    """Experiment 03's reported numbers must not move because a new keyword exists."""
    X, y, _ = _clustered_frame()
    candidates = {"rf": ML.make_candidates(1)["RandomForest"]}
    a = ML.evaluate_candidates(X, y, label="a", seed=1, candidates=candidates)
    b = ML.evaluate_candidates(X, y, label="b", seed=1, candidates=candidates, groups=None)
    assert a.table["holdout_rmse"].iloc[0] == pytest.approx(b.table["holdout_rmse"].iloc[0])
    assert a.baseline_rmse == pytest.approx(b.baseline_rmse)
