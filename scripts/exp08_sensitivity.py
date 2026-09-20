"""Experiment 08 -- how far does prevalence measurement error travel into the risk estimates?

The question, and why it is a contribution rather than a robustness check
--------------------------------------------------------------------------
The base paper regresses farm outcomes on disease prevalence and treats that prevalence as
observed. It is not observed. It is what a farmer remembered and told an enumerator. This project
proposes measuring it with a camera instead -- but a camera is also wrong, in its own way, and
swapping one mismeasured regressor for another is only progress if the error shrinks.

Classical errors-in-variables predicts the direction: a mismeasured regressor's coefficient is
biased **toward zero**, so measurement error makes disease look *less* costly than it is. This
experiment measures how much of that happens at the error magnitudes this pipeline actually
produces -- taken from Experiment 07, not invented for the occasion.

Two specifications, because the bridge changes the question
------------------------------------------------------------
**S1 -- the published specification.** The income equation exactly as Experiment 03 estimated it,
carrying `had_white_spot`: a self-reported 0/1. A vision system replacing that questionnaire item
would misclassify farms at rates its own confusion matrix gives. So S1 is perturbed by
**misclassification**, whose attenuation formula is not the classical one (see
`integration.sensitivity.binary_attenuation`) and depends on how common the condition is.

**S2 -- the specification the bridge is built to feed.** `had_white_spot` is replaced by
`wsd_prevalence`, a continuous share in [0, 1]. This is what the classifier estimates, and it is
perturbed by **continuous** error at the magnitudes Experiment 07 measured -- resampled from the
errors that experiment actually observed, not drawn from a Gaussian fitted to them, because a
classifier's error is skewed and bounded and a Gaussian would put mass where no estimate can land.

Reporting both matters. S1 says what happens to the model as published if the questionnaire is
replaced. S2 says what happens to the model the bridge is for. They are different questions and a
single number answering "the" sensitivity would be answering neither.

The pre-registered null, written before this ran
-------------------------------------------------
Master Plan Section 6, Stage 9:

    If coefficient movement across the grid stays within the bootstrap standard error of the
    unperturbed estimate, the finding is "no detectable propagation at the error magnitudes this
    pipeline actually produces", and it is reported as a result.

The Monte-Carlo standard error is computed per level and the comparison is a column in the output
table, so the verdict is read rather than argued.

Three things this experiment is not
------------------------------------
1. **It is not an end-to-end measurement.** The image corpus has no farm identifier and the IFPRI
   farms are a different sample from a different year. No shrimp in the photographs belongs to any
   farm in this regression. What is measured is the propagation of a *magnitude* -- real error,
   real covariates, no link between them.
2. **It is not unconditional.** `D-026`: RESET rejects both mean equations, so everything here is
   conditional on a functional form the project itself flags as doubtful.
3. **It does not treat the self-report as truth.** The perturbation is applied *on top of* the
   survey value, so what is measured is the error a camera would **add**. If the self-report is
   itself noisy -- and this project argues it is -- the real comparison favours the camera more
   than these numbers show. That is the conservative direction, deliberately.

Outputs
-------
    reports/tables/exp08_baseline.csv       the unperturbed fits, S1 and S2 side by side
    reports/tables/exp08_sensitivity.csv    coefficient path per error level, per specification
    reports/tables/exp08_attenuation.csv    observed attenuation against the formula's prediction
    reports/tables/exp08_step3.csv          what the error does to the step-3 comparison
    reports/tables/exp08_environment.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from shrimp_risk.common.paths import TABLES  # noqa: E402
from shrimp_risk.common.seeding import environment_fingerprint, seed_everything  # noqa: E402
from shrimp_risk.econometrics import data as D  # noqa: E402
from shrimp_risk.econometrics import just_pope as JP  # noqa: E402
from shrimp_risk.econometrics import ml_models as ML  # noqa: E402
from shrimp_risk.integration import sensitivity as SEN  # noqa: E402

# Reuse Experiment 03's covariate rule verbatim rather than restating it. A second copy would
# drift, and the whole point of S1 is that it is the published specification and not a lookalike.
from exp03_track_a import covariates_for  # noqa: E402

SEED = 2025
OUTCOME = "revenue_total_bdt"
TRANSFORM = "log"

# Misclassification grid for S1. The rates are symmetric for legibility; the measured
# specimen-level rates from Experiment 06's confusion matrices are located on this grid and
# annotated in the output rather than being the only point evaluated. A single point would show
# where this pipeline sits and not what the curve looks like around it.
MISCLASSIFICATION_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)

# Continuous error grid for S2, in units of prevalence (0-1). Experiment 07's measured RMSEs are
# added to this grid at runtime and flagged as measured.
CONTINUOUS_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)

REPLICATES = 200

# Step 3 is a repeated-K-fold comparison of four models and costs seconds per fit, so it runs at a
# few representative levels with fewer replicates. Running it at every level would multiply the
# runtime by two orders of magnitude to refine a secondary result: the headline of Stage 6 is that
# no step-3 model beats the mean, and the question here is only whether measurement error changes
# that verdict.
STEP3_LEVELS = (0.0, 0.10, 0.30)
STEP3_REPLICATES = 5


def measured_error_levels() -> tuple[list[dict], dict[str, float]]:
    """Read Experiment 07's measured error magnitudes, if it has run.

    Returns the levels to add to the grid and the specimen-level misclassification rates. Absent
    the table, returns empty -- the synthetic grid still runs and the output says plainly that no
    measured point could be located on it, rather than quietly presenting a synthetic level as a
    measured one.
    """
    path = TABLES / "exp07_estimators.csv"
    if not path.exists():
        return [], {}
    est = pd.read_csv(path)

    # RAW, not calibrated. `D-041` established that the bridge runs on raw posteriors: temperature
    # scaling was measured and makes the best estimator worse (SLD-EM mean |bias| 0.0391 raw
    # against 0.0598 calibrated). The error magnitude "this pipeline actually produces" is
    # therefore the raw one, and taking it from the calibrated rows would perturb the regressor by
    # an error the deployed configuration does not make.
    deployed = est[est.calibration == "raw"]
    if deployed.empty:
        return [], {}

    # Two of these are the comparison Stage 9 exists to make: CC is what an uncorrected bridge
    # would hand the econometric model, SLD-EM is what the corrected one hands it. The gap between
    # their attenuation curves is the economic value of the correction.
    roles = {
        "CC": "uncorrected bridge -- what counting would hand Track A",
        "SLD-EM": "DEPLOYED -- the configuration D-041 recommends",
        "PACC": "best of the adjusted-count family, for reference",
    }
    levels = []
    for name, role in roles.items():
        rows = deployed[deployed.estimator == name]
        if rows.empty:
            continue
        rmse = float(rows["mean_rmse"].mean())
        levels.append({
            "magnitude": round(rmse, 4),
            "label": f"measured: {name}",
            "source": (f"exp07_estimators.csv, RAW posteriors, mean RMSE over the grid = "
                       f"{rmse:.4f} -- {role}"),
        })
    return levels, {}


def specimen_misclassification_rates() -> dict[str, float]:
    """Binary White-Spot error rates from Experiment 06's stored validation posteriors.

    Collapses the 4-class problem to the binary question the survey item asks -- does this animal
    carry White Spot -- and reports the false-positive and false-negative rates at **specimen**
    level, averaged over folds.

    Read as a farm-level rate, this is deliberately pessimistic: it is what a farm-level verdict
    would cost if the farm were judged from a single shrimp. A farm sampled at 20 or 40 animals
    does far better, and Experiment 07 measures that case directly. Reporting the pessimistic
    bound here keeps the S1 curve from flattering the method.
    """
    from shrimp_risk.integration import pools as P
    from shrimp_risk.integration import quantification as Q

    files = sorted((ROOT / "experiments" / "runs" / "exp06").glob("*_predictions.npz"))
    if not files:
        return {}
    positive = Q.positive_indices()
    fps, fns = [], []
    for path in files:
        payload = np.load(path, allow_pickle=True)
        if "val_probs" not in payload.files:
            continue
        probs, labels = payload["probs"], payload["labels"]
        specimen_probs, ids = Q.aggregate_to_specimen(probs.astype("float64"),
                                                      payload["specimen_ids"])
        truth, _ = P.specimen_truth(payload["specimen_ids"], labels.astype("int64"))
        actual = P.is_positive(truth, positive)
        predicted = np.isin(specimen_probs.argmax(1), positive)
        if (~actual).any():
            fps.append(float(predicted[~actual].mean()))
        if actual.any():
            fns.append(float((~predicted[actual]).mean()))
    if not fps or not fns:
        return {}
    return {"fpr": float(np.mean(fps)), "fnr": float(np.mean(fns)), "n_runs": len(fps)}


def fit_once(frame: pd.DataFrame, covariates: list[str], target: str, label: str):
    """One Just-Pope step-1/2 fit, returning the row this experiment records."""
    res = JP.run_just_pope(frame, outcome=OUTCOME, covariates=covariates,
                           transform=TRANSFORM, label=label)
    coef = SEN.coefficient_of(res.mean_equation, target)
    return res, {
        **coef,
        "r_squared": res.mean_equation.r_squared,
        "breusch_pagan_p": res.mean_equation.diagnostics["breusch_pagan_p"],
        "reset2_p": res.mean_equation.diagnostics["reset2_p"],
        "n": res.mean_equation.n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=REPLICATES)
    ap.add_argument("--skip-step3", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="10 replicates, no step 3")
    args = ap.parse_args()
    if args.smoke:
        args.replicates, args.skip_step3 = 10, True

    seed_everything(SEED)
    print("environment:", environment_fingerprint())

    af = D.build_income_frame_with_prevalence()
    frame = af.frame
    covs_s1 = covariates_for(af, OUTCOME)

    # S2 swaps the binary self-report for the continuous share. Everything else is identical, so
    # the difference between the two baselines is attributable to the regressor and nothing else.
    if "had_white_spot" not in covs_s1:
        raise SystemExit(
            "had_white_spot is not in the Experiment 03 covariate set -- S1 has nothing to "
            "perturb, and the specification this experiment claims to be testing has changed."
        )
    covs_s2 = [c for c in covs_s1 if c != "had_white_spot"] + ["wsd_prevalence"]

    usable_s1 = frame.dropna(subset=[OUTCOME] + covs_s1).reset_index(drop=True)
    usable_s2 = frame.dropna(subset=[OUTCOME] + covs_s2).reset_index(drop=True)

    print(f"S1 covariates: {len(covs_s1)}, n = {len(usable_s1):,}  (target had_white_spot)")
    print(f"S2 covariates: {len(covs_s2)}, n = {len(usable_s2):,}  (target wsd_prevalence)")
    print(f"  {int((usable_s2.wsd_prevalence > 0).sum())} farms with prevalence > 0, "
          f"mean among those {usable_s2.loc[usable_s2.wsd_prevalence > 0, 'wsd_prevalence'].mean():.4f}")

    # ---- baselines ----
    res_s1, base_s1 = fit_once(usable_s1, covs_s1, "had_white_spot", "S1_published")
    res_s2, base_s2 = fit_once(usable_s2, covs_s2, "wsd_prevalence", "S2_bridge_facing")
    baseline = pd.DataFrame([
        {"specification": "S1_published", "target": "had_white_spot", **base_s1},
        {"specification": "S2_bridge_facing", "target": "wsd_prevalence", **base_s2},
    ])
    TABLES.mkdir(parents=True, exist_ok=True)
    baseline.round(6).to_csv(TABLES / "exp08_baseline.csv", index=False)
    print("\nUnperturbed baselines")
    print(baseline.round(4).to_string(index=False))

    measured_levels, _ = measured_error_levels()
    rates = specimen_misclassification_rates()
    if rates:
        print(f"\nspecimen-level binary error: FPR {rates['fpr']:.4f}, FNR {rates['fnr']:.4f} "
              f"(mean of {rates['n_runs']} Experiment 06 runs)")
    else:
        print("\nno Experiment 06 posteriors found -- the measured point cannot be located on "
              "the S1 grid, and only the synthetic grid is reported")

    rows = []

    # ---- S1: misclassification of the binary self-report ----
    p_hat = float(usable_s1["had_white_spot"].mean())
    for rate in MISCLASSIFICATION_GRID:
        rng = np.random.default_rng(SEED + int(rate * 1000))
        predicted_lambda = SEN.binary_attenuation(p_hat, rate, rate)
        for rep in range(args.replicates):
            perturbed = usable_s1.copy()
            perturbed["had_white_spot"] = SEN.misclassify_binary(
                usable_s1["had_white_spot"].to_numpy(), rng, fpr=rate, fnr=rate)
            if perturbed["had_white_spot"].nunique() < 2:
                continue
            _, row = fit_once(perturbed, covs_s1, "had_white_spot", "S1")
            rows.append({
                "specification": "S1_published", "target": "had_white_spot",
                "kind": "misclassification", "magnitude": rate,
                "level_label": f"fpr=fnr={rate:.2f}",
                "predicted_attenuation": predicted_lambda,
                "replicate": rep, **row,
            })
        print(f"  S1 fpr=fnr={rate:.2f}  predicted lambda {predicted_lambda:.4f}")

    # ---- S2: continuous error in the prevalence share ----
    s2_levels = [{"magnitude": m, "label": f"sd={m:.2f}", "source": "synthetic grid"}
                 for m in CONTINUOUS_GRID] + measured_levels
    x_true = usable_s2["wsd_prevalence"].to_numpy()
    for level in s2_levels:
        m = float(level["magnitude"])
        rng = np.random.default_rng(SEED + 7919 + int(m * 10000))
        predicted_lambda = SEN.classical_attenuation(x_true, m)
        for rep in range(args.replicates):
            perturbed = usable_s2.copy()
            perturbed["wsd_prevalence"] = SEN.perturb_continuous(x_true, rng, error_sd=m)
            _, row = fit_once(perturbed, covs_s2, "wsd_prevalence", "S2")
            rows.append({
                "specification": "S2_bridge_facing", "target": "wsd_prevalence",
                "kind": "classical", "magnitude": m,
                "level_label": level["label"],
                "source": level["source"],
                "predicted_attenuation": predicted_lambda,
                "replicate": rep, **row,
            })
        print(f"  S2 {level['label']:22s} predicted lambda {predicted_lambda:.4f}"
              f"   [{level['source']}]")

    paths = pd.DataFrame(rows)
    paths.to_csv(TABLES / "exp08_sensitivity.csv", index=False)

    # ---- attenuation summary, with the pre-registered verdict as a column ----
    summaries = []
    for spec, base in (("S1_published", base_s1), ("S2_bridge_facing", base_s2)):
        sub = paths[paths.specification == spec]
        if sub.empty:
            continue
        s = SEN.summarise_path(sub, base["estimate"], base["p_value"])
        s.insert(0, "specification", spec)
        s["predicted_attenuation"] = (
            sub.groupby(["level_label", "kind", "magnitude"], dropna=False)
            ["predicted_attenuation"].first().to_numpy()
        )
        s["formula_gap"] = (s["attenuation_observed"] - s["predicted_attenuation"]).round(4)
        summaries.append(s)
    attenuation = pd.concat(summaries, ignore_index=True).round(6)
    attenuation.to_csv(TABLES / "exp08_attenuation.csv", index=False)

    print("\n" + "=" * 100)
    print("EXPERIMENT 08 -- coefficient on the disease regressor as measurement error grows")
    print("=" * 100)
    print(attenuation[["specification", "level_label", "replicates", "estimate_mean",
                       "baseline_estimate", "baseline_p", "ratio_interpretable",
                       "attenuation_observed", "predicted_attenuation",
                       "formula_gap", "mc_se", "moved_beyond_noise", "p_median"]]
          .to_string(index=False))
    if not attenuation["ratio_interpretable"].all():
        blocked = sorted(attenuation.loc[~attenuation["ratio_interpretable"], "specification"]
                         .unique())
        print()
        for spec in blocked:
            row = baseline[baseline.specification == spec].iloc[0]
            print(f"  {spec}: attenuation ratio SUPPRESSED. The unperturbed coefficient on "
                  f"{row['target']} is {row['estimate']:.4f} at p = {row['p_value']:.3f} -- not "
                  "distinguishable from zero, so there is no effect for measurement error to")
            print("  attenuate, and a ratio computed from it would be noise divided by noise. "
                  "That absence is itself the finding; see the stage document.")

    # ---- step 3: does the error change Stage 6's negative result? ----
    if not args.skip_step3:
        step3_rows = []
        for m in STEP3_LEVELS:
            rng = np.random.default_rng(SEED + 31337 + int(m * 10000))
            for rep in range(STEP3_REPLICATES):
                perturbed = usable_s2.copy()
                perturbed["wsd_prevalence"] = SEN.perturb_continuous(x_true, rng, error_sd=m)
                res, _ = fit_once(perturbed, covs_s2, "wsd_prevalence", "S2")
                step3 = ML.evaluate_candidates(
                    res.X, np.asarray(res.log_deviance), label=f"S2_sd{m}", seed=SEED)
                table = step3.comparison if hasattr(step3, "comparison") else step3.table
                best = table.sort_values("holdout_rmse").iloc[0]
                step3_rows.append({
                    "error_sd": m, "replicate": rep,
                    "best_model": best.get("model", best.get("candidate", "")),
                    "holdout_rmse": float(best["holdout_rmse"]),
                    "baseline_rmse": float(step3.baseline_rmse),
                    # Project convention (`exp03_step3_candidates.csv`): above 1.0 is WORSE than
                    # predicting the mean. Stage 6's headline is 1.014 for the best model.
                    "holdout_rmse_vs_mean_baseline": float(
                        best["holdout_rmse"] / step3.baseline_rmse),
                })
            print(f"  step 3 at sd={m:.2f}: done")
        step3_df = pd.DataFrame(step3_rows)
        step3_df.to_csv(TABLES / "exp08_step3.csv", index=False)
        print("\nStep 3 -- does measurement error change the Stage 6 verdict?")
        print(step3_df.groupby("error_sd")[["holdout_rmse_vs_mean_baseline", "holdout_rmse"]]
              .agg(["mean", "std"]).round(4).to_string())
        print("Above 1.0 means WORSE than predicting the mean. Stage 6's finding was that every")
        print("model is, at 1.014 for the best. If that survives the perturbation, the negative")
        print("result is not an artefact of how prevalence was measured.")

    pd.DataFrame([environment_fingerprint()]).to_csv(
        TABLES / "exp08_environment.csv", index=False)

    print(f"\ntables -> {TABLES}")
    print("\nThe images and these farms are different samples. This measures the propagation of a")
    print("measured error MAGNITUDE through real covariates -- not the effect of this classifier")
    print("on these farms, which no dataset in existence would let us compute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
