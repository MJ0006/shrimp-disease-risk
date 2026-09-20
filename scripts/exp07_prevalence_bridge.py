"""Experiment 07 -- does correcting the count actually recover the prevalence?

The question
------------
`integration/quantification.py` argues that counting a classifier's positive predictions is a
biased estimator of prevalence, and that the adjusted-count and Saerens EM procedures remove that
bias. The argument is standard and the arithmetic is in the module docstring. Neither is evidence
about *this* corpus, *this* classifier, or a farm-sized sample of 20 to 40 shrimp.

This experiment measures it. Pools of specimens are constructed at a prevalence chosen in advance,
so the number each estimator should return is known before it runs, and the error is a measurement
rather than a comparison against another estimate.

Three things are varied, and each answers a separate question
--------------------------------------------------------------
1. **Estimator** -- CC, PCC, ACC, PACC, SLD-EM. Does correction help, and which correction?
2. **Calibration** -- raw posteriors against Experiment 06's temperature-scaled ones. Both
   corrections *assume* calibrated posteriors (`D-028`), so this says whether the assumption is
   load-bearing in practice or merely stated.
3. **Pool size** -- 20 and 40 specimens. A farm does not photograph thousands of shrimp. An
   estimator that is unbiased in the limit and useless at n = 20 is useless.

and prevalence itself is swept from 0.05 to 0.95, because the naive estimator's bias *depends on
the true prevalence* -- it is near zero where the classifier's errors happen to cancel, and largest
at the extremes. Reporting a single prevalence would let the naive estimator look accurate by
accident.

The pre-registered null, written down before this ran
-----------------------------------------------------
Master Plan Section 6, Stage 8:

    If the corrected estimators do not reduce absolute bias relative to CC by more than the
    Monte-Carlo standard error across the grid, the finding is "no measurable improvement", and
    contribution claim C1 is weakened to match.

The Monte-Carlo standard error is computed and reported next to every comparison, so that verdict
is read off the table rather than argued for.

What this experiment cannot do
------------------------------
There are no farm identifiers in `ShrimpDiseaseImageBD` (Master Plan limitation 2), so no pool here
is a real farm. Pools are constructed, deliberately, because that is what makes the truth known.
The estimator is therefore validated under the *statistical* conditions a farm would present --
finite sample, mixed classes, a classifier with real error rates -- and not under its *biological*
ones. Every table this writes carries that distinction, and no result from it may be described as
a measured farm prevalence.

Why each fold is analysed separately
------------------------------------
Every fold has its own trained model, its own calibration temperature and its own
validation-estimated confusion matrix. A pool mixing two folds' specimens would correct model A's
posteriors with model B's matrix. So pools are drawn within a fold and results are reported as
mean +/- SD across folds -- the same protocol, for the same reason, as Stage 5's backbone
comparison.

Outputs
-------
    reports/tables/exp07_bias.csv          bias/RMSE per (backbone, calibration, size, p, estimator)
    reports/tables/exp07_estimators.csv    the headline: mean |bias| per estimator, vs CC
    reports/tables/exp07_coverage.csv      measured coverage of the bootstrap interval
    reports/tables/exp07_feasibility.csv   grid cells the folds could not supply
    reports/tables/exp07_environment.csv   what produced all of the above
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shrimp_risk.common.paths import (  # noqa: E402
    RUNS, TABLES, resolve_image_manifest,
)
from shrimp_risk.common.seeding import environment_fingerprint, seed_everything  # noqa: E402
from shrimp_risk.integration import pools as P  # noqa: E402
from shrimp_risk.integration import quantification as Q  # noqa: E402
from shrimp_risk.integration.calibration import apply_temperature  # noqa: E402
from shrimp_risk.vision import splits as S  # noqa: E402
from shrimp_risk.vision.datasets import CLASSES  # noqa: E402

SEED = 2025
N_FOLDS = 5
POOL_SIZES = (20, 40)
PREVALENCE_GRID = np.round(np.arange(0.05, 0.96, 0.10), 2)
N_REPLICATES = 200

# Coverage costs a bootstrap per pool, so it is measured on a subset: fewer replicates, one pool
# size, and only the estimators a reader would actually deploy. Measuring it on everything would
# multiply the runtime by ~400 to refine a number that is already reported with its own standard
# error.
COVERAGE_ESTIMATORS = ("CC", "PACC", "SLD-EM")
COVERAGE_REPLICATES = 50
COVERAGE_BOOT = 300
COVERAGE_SIZE = 40
ALPHA = 0.05

# Coverage is measured on every second grid point. Each pool costs one bootstrap per estimator --
# 300 resamples x 3 estimators -- so the full grid would be roughly 14 million estimator calls to
# resolve a curve that is smooth in prevalence. Halving the grid keeps the shape and the endpoints
# and costs a third of the runtime. The points actually measured are in the output table, so no
# reader has to guess which they are.
COVERAGE_STRIDE = 2

# EM is capped well below the module default here. At 40 specimens it converges in tens of
# iterations; a pool that has not converged in 200 is degenerate (usually every specimen assigned
# to one class), and the cap keeps 700k estimator calls from waiting on it.
EM_MAX_ITER = 200


def load_run(path: Path) -> dict:
    """One Experiment 06 run: test and validation posteriors, ids, and the fitted temperature."""
    payload = np.load(path, allow_pickle=True)
    if "val_probs" not in payload.files:
        raise SystemExit(
            f"{path.name} has no validation posteriors. Experiment 07 needs runs written by "
            "scripts/exp06_calibrate_bridge.py -- the Experiment 04 artefacts predate the "
            "trainer change that saves them."
        )
    tag = path.name.replace("_predictions.npz", "")
    backbone, fold = tag.split("__fold")
    return {
        "backbone": backbone,
        "fold": int(fold),
        "test_probs": payload["probs"].astype("float64"),
        "test_labels": payload["labels"].astype("int64"),
        "test_ids": payload["specimen_ids"],
        "val_probs": payload["val_probs"].astype("float64"),
        "val_labels": payload["val_labels"].astype("int64"),
        "val_ids": payload["val_specimen_ids"],
        "temperature": float(payload["temperature"][0]),
    }


def training_priors(manifest: pd.DataFrame) -> dict[int, np.ndarray]:
    """Image-level class distribution of each fold's training partition.

    Image-level, not specimen-level, and this one really is deliberate rather than an oversight of
    the kind the confusion matrix suffered. SLD-EM removes the prior the classifier's posteriors
    were *fitted* under, and the loss saw images: a specimen contributing five photographs pushed
    the decision boundary five times. The specimen-level prior would remove a prior the model
    never had.

    The distinction from the confusion matrix is worth stating because the two look alike.
    `M` describes the *prediction process the estimator observes*, which is specimen-level.
    `train_prior` describes the *fitting process the model underwent*, which is image-level. They
    are estimated at different grains on purpose.

    The folds are rebuilt from the same seed rather than stored, which is only safe because
    `test_folds_are_reproducible_from_the_seed` asserts they come back identical.
    """
    folds = S.make_specimen_folds(manifest, n_splits=N_FOLDS, seed=SEED)
    out = {}
    for f in folds:
        counts = manifest.iloc[f.train]["class"].value_counts()
        prior = np.array([counts.get(c, 0) for c in CLASSES], dtype="float64")
        out[f.fold] = prior / prior.sum()
    return out


def estimators_for(
    specimen_probs: np.ndarray, M: np.ndarray, M_soft: np.ndarray,
    train_prior: np.ndarray, positive: list[int],
) -> dict[str, Q.PrevalenceEstimate]:
    """All five estimators on one pool. Kept in one place so the coverage pass cannot drift."""
    return {
        "CC": Q.classify_and_count(specimen_probs, positive),
        "PCC": Q.probabilistic_classify_and_count(specimen_probs, positive),
        "ACC": Q.adjusted_count(specimen_probs, M, positive),
        "PACC": Q.probabilistic_adjusted_count(specimen_probs, M_soft, positive),
        "SLD-EM": Q.saerens_em(specimen_probs, train_prior, positive, max_iter=EM_MAX_ITER),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=N_REPLICATES)
    ap.add_argument("--coverage-replicates", type=int, default=COVERAGE_REPLICATES)
    ap.add_argument("--skip-coverage", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="10 replicates, no coverage pass")
    args = ap.parse_args()
    if args.smoke:
        args.replicates, args.skip_coverage = 10, True

    seed_everything(SEED)
    positive = Q.positive_indices()
    print("environment:", environment_fingerprint())
    print(f"positive classes: {list(Q.WSSV_POSITIVE)} -> indices {positive}")

    run_files = sorted((RUNS / "exp06").glob("*_predictions.npz"))
    if not run_files:
        raise SystemExit(
            "no Experiment 06 runs found. Run scripts/exp06_calibrate_bridge.py first -- "
            "this experiment needs validation posteriors and a fitted temperature."
        )
    print(f"runs       : {len(run_files)} from experiments/runs/exp06/")

    manifest = S.load_raw_manifest(resolve_image_manifest(256))
    priors = training_priors(manifest)
    print(f"manifest   : {len(manifest):,} images / {manifest.specimen_id.nunique()} specimens")
    print(f"grid       : {list(PREVALENCE_GRID)} x sizes {POOL_SIZES} x "
          f"{args.replicates} replicates\n")

    bias_rows, coverage_rows, feasibility_rows = [], [], []

    for path in run_files:
        run = load_run(path)
        backbone, fold = run["backbone"], run["fold"]
        train_prior = priors[fold]

        for state in ("raw", "calibrated"):
            T = 1.0 if state == "raw" else run["temperature"]
            test_probs = apply_temperature(run["test_probs"], T)
            val_probs = apply_temperature(run["val_probs"], T)

            # The confusion matrices are estimated on the SAME calibration state the pools are
            # scored in. Correcting calibrated posteriors with a matrix built from raw ones would
            # mix two different classifiers, and the resulting bias would be attributable to
            # neither.
            #
            # They are also estimated at the SAME GRAIN. This is not a detail. ACC solves
            # `M p = q`, where `q` is the vector of predicted proportions the estimator observes --
            # and every estimator here observes *specimen-level* predictions, averaged over a
            # shrimp's 2.76 photographs. A matrix estimated from individual images describes a
            # different and measurably worse classifier: averaging photographs removes error, so
            # an image-level `M` overstates confusion, ACC over-corrects, and the solve is pushed
            # off the simplex. On fold 0 the two matrices differ by up to 0.11 in a single entry.
            #
            # The cost is that the matrix is now estimated from ~67 specimens rather than ~187
            # images, so it is noisier. That is the honest trade: a noisier estimate of the right
            # quantity beats a tighter estimate of the wrong one, and the noise shows up in the
            # results as clipping, which is reported.
            M = Q.confusion_at_specimen_grain(
                val_probs, run["val_labels"], run["val_ids"], soft=False)
            M_soft = Q.confusion_at_specimen_grain(
                val_probs, run["val_labels"], run["val_ids"], soft=True)

            specimen_probs, specimen_ids = Q.aggregate_to_specimen(test_probs, run["test_ids"])
            truth, truth_ids = P.specimen_truth(run["test_ids"], run["test_labels"])
            assert (specimen_ids == truth_ids).all(), (
                "specimen ordering diverged between posteriors and labels -- every estimate "
                "below would be scored against the wrong shrimp"
            )
            positive_mask = P.is_positive(truth, positive)

            rng = np.random.default_rng(SEED + 1000 * fold + (0 if state == "raw" else 1))

            for size in POOL_SIZES:
                feasible = P.feasible_prevalences(positive_mask, size, PREVALENCE_GRID)
                for p_target, ok in feasible.items():
                    if not ok:
                        feasibility_rows.append({
                            "backbone": backbone, "fold": fold, "calibration": state,
                            "pool_size": size, "target_prevalence": p_target,
                            "n_positive_available": int(positive_mask.sum()),
                            "n_negative_available": int((~positive_mask).sum()),
                            "reason": "fold cannot supply this mix without repeating a specimen",
                        })
                        continue

                    errors = {name: [] for name in
                              ("CC", "PCC", "ACC", "PACC", "SLD-EM")}
                    clipped = {name: 0 for name in errors}
                    realised = None

                    for _ in range(args.replicates):
                        pool = P.build_pool(rng, positive_mask, p_target, size)
                        realised = pool.true_prevalence
                        pooled = specimen_probs[pool.indices]
                        for name, est in estimators_for(
                                pooled, M, M_soft, train_prior, positive).items():
                            errors[name].append(est.prevalence - pool.true_prevalence)
                            clipped[name] += int(est.left_simplex)

                    for name, errs in errors.items():
                        e = np.asarray(errs)
                        bias_rows.append({
                            "backbone": backbone, "fold": fold, "calibration": state,
                            "pool_size": size,
                            "target_prevalence": p_target,
                            "true_prevalence": realised,
                            "estimator": name,
                            "bias": round(float(e.mean()), 5),
                            "abs_bias": round(float(abs(e.mean())), 5),
                            "rmse": round(float(np.sqrt((e ** 2).mean())), 5),
                            "mae": round(float(np.abs(e).mean()), 5),
                            "sd": round(float(e.std(ddof=1)), 5),
                            "mc_se": round(float(e.std(ddof=1) / np.sqrt(len(e))), 5),
                            "pct_clipped": round(100 * clipped[name] / len(e), 1),
                            "n_pools": len(e),
                        })

            if args.skip_coverage:
                continue

            # ---- interval coverage: does a "95%" interval contain the truth 95% of the time? ----
            cov_rng = np.random.default_rng(SEED + 500 + fold)
            coverage_grid = PREVALENCE_GRID[::COVERAGE_STRIDE]
            for p_target, ok in P.feasible_prevalences(
                    positive_mask, COVERAGE_SIZE, coverage_grid).items():
                if not ok:
                    continue
                hits = {name: 0 for name in COVERAGE_ESTIMATORS}
                widths = {name: [] for name in COVERAGE_ESTIMATORS}
                for _ in range(args.coverage_replicates):
                    pool = P.build_pool(cov_rng, positive_mask, p_target, COVERAGE_SIZE)
                    pooled = specimen_probs[pool.indices]
                    for name in COVERAGE_ESTIMATORS:
                        def one(sample, _n=name):
                            return estimators_for(
                                sample, M, M_soft, train_prior, positive)[_n]
                        ci = Q.bootstrap_interval(
                            one, pooled, n_boot=COVERAGE_BOOT, alpha=ALPHA, seed=SEED)
                        hits[name] += int(ci["lo"] <= pool.true_prevalence <= ci["hi"])
                        widths[name].append(ci["width"])
                for name in COVERAGE_ESTIMATORS:
                    coverage_rows.append({
                        "backbone": backbone, "fold": fold, "calibration": state,
                        "pool_size": COVERAGE_SIZE, "target_prevalence": p_target,
                        "estimator": name,
                        "nominal": 1 - ALPHA,
                        "coverage": round(hits[name] / args.coverage_replicates, 4),
                        "mean_width": round(float(np.mean(widths[name])), 4),
                        "n_pools": args.coverage_replicates,
                        "n_boot": COVERAGE_BOOT,
                    })

        print(f"  done {backbone} fold {fold}  (T = {run['temperature']:.3f})")

    TABLES.mkdir(parents=True, exist_ok=True)
    bias = pd.DataFrame(bias_rows)
    bias.to_csv(TABLES / "exp07_bias.csv", index=False)
    pd.DataFrame(feasibility_rows).to_csv(TABLES / "exp07_feasibility.csv", index=False)

    # ---- the headline table: mean absolute bias per estimator, against CC ----
    summary = (
        bias.groupby(["backbone", "calibration", "pool_size", "estimator"])
        .agg(mean_abs_bias=("abs_bias", "mean"),
             max_abs_bias=("abs_bias", "max"),
             mean_rmse=("rmse", "mean"),
             mean_mc_se=("mc_se", "mean"),
             pct_clipped=("pct_clipped", "mean"),
             cells=("abs_bias", "size"))
        .round(5)
        .reset_index()
    )
    baseline = (
        summary[summary.estimator == "CC"]
        .set_index(["backbone", "calibration", "pool_size"])["mean_abs_bias"]
    )
    summary["cc_mean_abs_bias"] = summary.set_index(
        ["backbone", "calibration", "pool_size"]).index.map(baseline)
    summary["improvement_vs_cc"] = (
        summary["cc_mean_abs_bias"] - summary["mean_abs_bias"]).round(5)

    # The pre-registered decision rule, evaluated in the table rather than in the prose that
    # describes it. `mean_mc_se` is the Monte-Carlo standard error of a single grid cell's bias;
    # an improvement smaller than that is not distinguishable from the pool draw.
    summary["beats_cc"] = np.where(
        summary["estimator"] == "CC", "-",
        np.where(summary["improvement_vs_cc"] > summary["mean_mc_se"], "YES", "no"),
    )
    summary = summary.sort_values(
        ["backbone", "calibration", "pool_size", "mean_abs_bias"])
    summary.to_csv(TABLES / "exp07_estimators.csv", index=False)

    if coverage_rows:
        cov = pd.DataFrame(coverage_rows)
        cov.to_csv(TABLES / "exp07_coverage.csv", index=False)

    pd.DataFrame([environment_fingerprint()]).to_csv(
        TABLES / "exp07_environment.csv", index=False)

    print("\n" + "=" * 86)
    print("EXPERIMENT 07 -- mean |bias| over the prevalence grid, by estimator")
    print("=" * 86)
    print(summary.to_string(index=False))

    if coverage_rows:
        print("\n" + "=" * 86)
        print("INTERVAL COVERAGE -- a '95%' interval is only 95% if it measures 95%")
        print("=" * 86)
        print(cov.groupby(["backbone", "calibration", "estimator"])
              .agg(coverage=("coverage", "mean"), mean_width=("mean_width", "mean"))
              .round(4).to_string())

    print(f"\ntables -> {TABLES}")
    print("\nPools are constructed, not observed: the image corpus has no farm identifier, so the")
    print("true prevalence is known only because it was chosen. No number here is a measured farm")
    print("prevalence, and nothing downstream may describe it as one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
