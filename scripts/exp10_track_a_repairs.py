"""Experiment 10 -- the three Track A repairs that make Stage 6's verdicts unconditional.

What is wrong with Stage 6 as it stands
----------------------------------------
Stage 6 returned a negative result and reported it: no step-3 model beats predicting the mean, on
either outcome. Three things qualify that verdict, and all three were written down at the time
rather than discovered later.

**`D-026` -- RESET rejects both mean equations** (income p = 4.3e-5, disease p = 0.013). Step 2's
deviance is the squared residual of step 1, so if step 1 is misspecified those residuals carry
squared *bias* as well as variance. Every Breusch-Pagan verdict downstream is therefore conditional
on a functional form the project itself flags as doubtful. That is the single largest caveat on
Track A.

**`D-029` -- the sample is clustered.** IFPRI drew 1,221 farms from 68 clusters of roughly 18. The
design effect on step 1 was measured and is small, but step 3's cross-validation folds split those
clusters at random, so a farm's neighbours -- sharing water, weather, hatchery and extension advice
-- sit in the training set while the farm itself is predicted. That is the `D-007` structure again,
one dataset over, and it has never been checked.

**Improvement 7e -- "not demonstrated" is not "absent".** The disease model's Breusch-Pagan test
does not reject (p = 0.195) at n = 463, so the precondition for the whole Just-Pope exercise is not
demonstrated. `D-026` is careful to say *not demonstrated* rather than *fails*. What that sentence
is missing is a number: how large would the heteroskedasticity have to be for this test, at this
sample size, with this design matrix, to have found it?

The three parts
---------------
    A  cluster-grouped step-3 folds        does the design effect change the negative result?
    B  specification search                can any form pass RESET, and does the verdict move?
    C  Breusch-Pagan power                 what could n = 463 have detected?

Part B carries a caveat that must travel with it
-------------------------------------------------
Searching over specifications and then reporting inference on the winner is **post-selection
inference**, and the reported p-values of the winning specification are not valid as if it had been
chosen in advance. This script therefore reports **every** specification tried, in the order tried,
with its diagnostics -- not just the best one. The honest reading is "of the N forms we tried, M
passed RESET", not "we found the correct specification".

Nothing here is tuned toward a positive result. `D-016` governs: if the repairs leave the negative
result standing, that is a stronger version of the same finding, because it will then rest on a
specification that survives the diagnostic which previously undermined it.

Outputs
-------
    reports/tables/exp10_grouped_cv.csv     random vs cluster-grouped step 3
    reports/tables/exp10_specifications.csv every functional form tried, with diagnostics
    reports/tables/exp10_low_variance.csv   what a variance filter WOULD drop -- and does not
    reports/tables/exp10_bp_power.csv       detectable heteroskedasticity at n = 463
    reports/tables/exp10_environment.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from shrimp_risk.common.paths import TABLES  # noqa: E402
from shrimp_risk.common.seeding import environment_fingerprint, seed_everything  # noqa: E402
from shrimp_risk.econometrics import data as D  # noqa: E402
from shrimp_risk.econometrics import just_pope as JP  # noqa: E402
from shrimp_risk.econometrics import ml_models as ML  # noqa: E402
from statsmodels.stats.diagnostic import het_breuschpagan  # noqa: E402

from exp03_track_a import covariates_for  # noqa: E402

SEED = 2025
BP_POWER_REPLICATES = 400

# Heteroskedasticity strengths for Part C, as the log-variance slope on a standardised covariate.
# Expressed in the output as the ratio of residual SD between a farm one standard deviation above
# the mean of that covariate and one a standard deviation below, because "gamma = 0.4" means
# nothing to a reader and "the noisiest farms are 2.2x noisier" means something.
GAMMA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)


def _cluster_key(frame: pd.DataFrame) -> np.ndarray | None:
    """`cluster_uid` per row, or None if it cannot be joined.

    Returns None rather than a fabricated key: a grouped analysis on a key we could not join is
    worse than no grouped analysis, because it would look like one.
    """
    try:
        key = D.load_ifpri_module("000_hh_commonvar")[["farmer_uid", "cluster_uid"]]
    except Exception as exc:  # pragma: no cover - depends on the survey files being present
        print(f"  cluster key unavailable ({exc}); grouped analysis skipped")
        return None
    joined = frame[["farmer_uid"]].merge(key, on="farmer_uid", how="left")
    groups = joined["cluster_uid"].to_numpy()
    if pd.isna(groups).any():
        print(f"  {int(pd.isna(groups).sum())} farms have no cluster id; grouped analysis skipped")
        return None
    return groups


# ---------------------------------------------------------------------------------------
# Part A -- does grouping the step-3 folds on cluster change the negative result?
# ---------------------------------------------------------------------------------------

def part_a(specs: dict) -> pd.DataFrame:
    rows = []
    for name, spec in specs.items():
        print(f"\n[A] {name}")
        af = spec["builder"]()
        frame = af.frame
        covs = covariates_for(af, spec["outcome"])
        usable = frame.dropna(subset=[spec["outcome"]] + covs).reset_index(drop=True)
        res = JP.run_just_pope(usable, outcome=spec["outcome"], covariates=covs,
                               transform=spec["transform"], label=name)
        y = np.asarray(res.log_deviance)

        groups = _cluster_key(usable)
        conditions = {"random_folds": None}
        if groups is not None:
            conditions["cluster_grouped_folds"] = groups

        for condition, g in conditions.items():
            step3 = ML.evaluate_candidates(res.X, y, label=f"{name}:{condition}",
                                           seed=SEED, groups=g)
            table = step3.table
            best = table.sort_values("holdout_rmse").iloc[0]
            rows.append({
                "risk_model": name,
                "fold_scheme": condition,
                "n": len(usable),
                "n_clusters": int(len(np.unique(g))) if g is not None else None,
                "best_model": best["model"],
                "cv_r2_mean": round(float(best["cv_r2_mean"]), 4),
                "holdout_rmse": round(float(best["holdout_rmse"]), 4),
                "baseline_rmse": round(float(step3.baseline_rmse), 4),
                # `holdout_rmse / baseline_rmse`, matching `exp03_step3_candidates.csv` and every
                # document that quotes it: **above 1.0 means WORSE than predicting the mean.**
                # The reciprocal reads more naturally and is the wrong choice here -- Stage 6's
                # headline is "best ratio 1.014", and a table in this repository printing 0.986
                # for the same run would silently invert every comparison made against it.
                "holdout_rmse_vs_mean_baseline": round(
                    float(best["holdout_rmse"] / step3.baseline_rmse), 4),
                "beats_mean": bool(best["holdout_rmse"] < step3.baseline_rmse),
                "n_fits": int(best["n_fits"]),
            })
            beat = best["holdout_rmse"] < step3.baseline_rmse
            print(f"    {condition:22s} best {best['model']:20s} "
                  f"rmse/baseline {best['holdout_rmse'] / step3.baseline_rmse:.4f} "
                  f"({'beats' if beat else 'worse than'} the mean)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Part B -- can any functional form pass RESET?
# ---------------------------------------------------------------------------------------

def _augment(X: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Build one candidate design matrix from the baseline covariates.

    Each augmentation is a standard response to a RESET rejection -- unmodelled curvature in the
    conditional mean -- rather than a search over arbitrary transformations. Squares of the
    continuous covariates are the textbook first move; logs address multiplicative scale; the
    interaction tests whether the district effect differs with farm size, which is the one
    substantive interaction the survey design makes plausible.
    """
    out = X.copy()
    continuous = [c for c in ("pond_area_ha", "pond_depth_ft") if c in out.columns]
    if kind == "baseline":
        return out
    if kind == "squares":
        for c in continuous:
            out[f"{c}_sq"] = out[c] ** 2
        return out
    if kind == "logs":
        for c in continuous:
            # log1p, not log: a pond of zero recorded area exists in the survey and dropping it
            # would change the sample between specifications, which would make the comparison
            # meaningless.
            out[f"log_{c}"] = np.log1p(out[c].clip(lower=0))
        return out
    if kind == "squares_and_logs":
        for c in continuous:
            out[f"{c}_sq"] = out[c] ** 2
            out[f"log_{c}"] = np.log1p(out[c].clip(lower=0))
        return out
    if kind == "area_district_interaction":
        for d in ("district_khulna", "district_satkhira"):
            if d in out.columns and "pond_area_ha" in out.columns:
                out[f"area_x_{d}"] = out["pond_area_ha"] * out[d]
        return out
    raise ValueError(f"unknown augmentation {kind!r}")


def part_b(specs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    forms = ("baseline", "squares", "logs", "squares_and_logs", "area_district_interaction")
    rows, low_variance = [], []
    for name, spec in specs.items():
        print(f"\n[B] {name}")
        af = spec["builder"]()
        covs = covariates_for(af, spec["outcome"])
        usable = af.frame.dropna(subset=[spec["outcome"]] + covs).reset_index(drop=True)

        # `drop_low_variance_binaries` exists, is tested, and is applied to NOTHING this project
        # reports (open item 13). Running it here as a diagnostic turns "unapplied" from a claim
        # in a docstring into a table: a reader can see exactly which covariates a variance
        # filter would have removed, and therefore what including them costs. Nothing is
        # dropped -- the surviving column list is discarded deliberately.
        _kept, dropped = D.drop_low_variance_binaries(usable, covs)
        if len(dropped):
            dropped.insert(0, "risk_model", name)
            low_variance.append(dropped)
            print(f"    [diagnostic] a variance filter would drop {len(dropped)} covariate(s): "
                  + ", ".join(dropped["variable"]) + "  -- NOT applied")
        else:
            print("    [diagnostic] a variance filter would drop nothing -- NOT applied anyway")

        for form in forms:
            X = _augment(usable[covs].astype("float64"), form)
            try:
                res = JP.run_just_pope(
                    pd.concat([usable[[spec["outcome"]]], X], axis=1),
                    outcome=spec["outcome"], covariates=list(X.columns),
                    transform=spec["transform"], label=f"{name}:{form}",
                )
            except Exception as exc:
                rows.append({"risk_model": name, "form": form, "status": f"failed: {exc}"})
                print(f"    {form:28s} FAILED: {exc}")
                continue
            d = res.mean_equation.diagnostics
            passes = bool(d["reset2_p"] > 0.05 and d["reset3_p"] > 0.05)
            rows.append({
                "risk_model": name,
                "form": form,
                "status": "ok",
                "k": res.mean_equation.k,
                "n": res.mean_equation.n,
                "r_squared": round(res.mean_equation.r_squared, 4),
                "adj_r_squared": round(res.mean_equation.adj_r_squared, 4),
                "reset2_p": d["reset2_p"],
                "reset3_p": d["reset3_p"],
                "passes_reset_at_05": passes,
                "breusch_pagan_p": d["breusch_pagan_p"],
                "white_p": d["white_p"],
                "bp_rejects_at_05": bool(d["breusch_pagan_p"] < 0.05),
            })
            print(f"    {form:28s} R2 {res.mean_equation.r_squared:.4f}  "
                  f"RESET2 p={d['reset2_p']:.3g}  RESET3 p={d['reset3_p']:.3g}  "
                  f"BP p={d['breusch_pagan_p']:.3g}  "
                  f"{'RESET OK' if passes else 'RESET rejects'}")
    lv = pd.concat(low_variance, ignore_index=True) if low_variance else pd.DataFrame()
    return pd.DataFrame(rows), lv


# ---------------------------------------------------------------------------------------
# Part C -- what heteroskedasticity could n = 463 have detected?
# ---------------------------------------------------------------------------------------

def part_c(specs: dict, replicates: int) -> pd.DataFrame:
    """Monte-Carlo power of the Breusch-Pagan test on the real design matrices.

    The null being tested is homoskedasticity. Data are generated from the *actual* covariates
    with a variance function built in by construction:

        y = X b + e,   sd(e_i) = sigma * exp(gamma * z_i)

    where `z` is the standardised first continuous covariate. At `gamma = 0` the data are
    homoskedastic and the rejection rate must land near the nominal 0.05 -- that row is the
    experiment's own control, and if it comes out wrong nothing else in the table means anything.

    Power is what turns "not demonstrated" into a statement with a size attached: the smallest
    heteroskedasticity this sample could have found four times in five.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for name, spec in specs.items():
        print(f"\n[C] {name}")
        af = spec["builder"]()
        covs = covariates_for(af, spec["outcome"])
        usable = af.frame.dropna(subset=[spec["outcome"]] + covs).reset_index(drop=True)
        X = usable[covs].astype("float64")
        design = sm.add_constant(X, has_constant="add").to_numpy()
        n = len(X)

        driver = "pond_area_ha" if "pond_area_ha" in X.columns else covs[0]
        z = X[driver].to_numpy()
        z = (z - z.mean()) / (z.std(ddof=1) or 1.0)

        for gamma in GAMMA_GRID:
            rejections = 0
            for _ in range(replicates):
                sd = np.exp(gamma * z)
                y = design @ np.zeros(design.shape[1]) + rng.normal(0.0, sd, size=n)
                fit = sm.OLS(y, design).fit()
                _, p_value, _, _ = het_breuschpagan(fit.resid, design)
                rejections += int(p_value < 0.05)
            power = rejections / replicates
            rows.append({
                "risk_model": name,
                "n": n,
                "driver": driver,
                "gamma": gamma,
                # exp(2*gamma) is the SD ratio between +1 and -1 standard deviations of z.
                "sd_ratio_p90_to_p10": round(float(np.exp(2 * gamma)), 3),
                "replicates": replicates,
                "power_at_05": round(power, 4),
                "detectable_at_80pct": bool(power >= 0.80),
            })
            print(f"    gamma {gamma:4.2f}  SD ratio {np.exp(2 * gamma):5.2f}x  "
                  f"power {power:.3f}")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=BP_POWER_REPLICATES)
    ap.add_argument("--parts", nargs="*", default=["a", "b", "c"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.replicates = 40

    seed_everything(SEED)
    print("environment:", environment_fingerprint())

    specs = {
        "income": dict(builder=D.build_income_frame,
                       outcome="revenue_total_bdt", transform="log"),
        "disease": dict(builder=D.build_disease_frame,
                        outcome="wsd_loss_bdt", transform="log1p"),
    }
    TABLES.mkdir(parents=True, exist_ok=True)

    if "a" in args.parts:
        a = part_a(specs)
        a.to_csv(TABLES / "exp10_grouped_cv.csv", index=False)
        print("\n" + "=" * 88)
        print("PART A -- random vs cluster-grouped step-3 folds")
        print("=" * 88)
        print(a.to_string(index=False))
        print("\n`holdout_rmse_vs_mean_baseline` ABOVE 1.0 means WORSE than predicting the mean,")
        print("matching exp03_step3_candidates.csv. Stage 6 reported 1.014 for the best model")
        print("under random folds. If grouping RAISES it, the reported figure was flattered by")
        print("neighbouring farms sitting in the training set.")

    if "b" in args.parts:
        b, low_variance = part_b(specs)
        b.to_csv(TABLES / "exp10_specifications.csv", index=False)
        low_variance.to_csv(TABLES / "exp10_low_variance.csv", index=False)
        print("\n" + "=" * 88)
        print("PART B -- every functional form tried, in the order tried")
        print("=" * 88)
        print(b.to_string(index=False))
        ok = b[b.get("passes_reset_at_05", False) == True]  # noqa: E712
        print(f"\n{len(ok)} of {len(b)} specifications pass RESET at 0.05.")
        print("POST-SELECTION INFERENCE WARNING: these forms were chosen after seeing their own")
        print("diagnostics. The p-values of a winning specification are not valid as if it had")
        print("been specified in advance. Report the whole table, never only the survivor.")

    if "c" in args.parts:
        c = part_c(specs, args.replicates)
        c.to_csv(TABLES / "exp10_bp_power.csv", index=False)
        print("\n" + "=" * 88)
        print("PART C -- Breusch-Pagan power on the real design matrices")
        print("=" * 88)
        print(c.to_string(index=False))
        for name in c.risk_model.unique():
            sub = c[(c.risk_model == name) & c.detectable_at_80pct]
            if len(sub):
                r = sub.iloc[0]
                print(f"\n  {name}: 80% power reached at an SD ratio of "
                      f"{r['sd_ratio_p90_to_p10']:.2f}x across the covariate range.")
            else:
                print(f"\n  {name}: 80% power NOT reached anywhere on the grid -- this sample "
                      "could not have detected even the largest heteroskedasticity tested.")
        null_rows = c[c.gamma == 0.0]
        print("\n  Control (gamma = 0, homoskedastic by construction): rejection rates "
              f"{list(null_rows.power_at_05)} -- these must sit near the nominal 0.05, or the")
        print("  rest of this table is not measuring power.")

    pd.DataFrame([environment_fingerprint()]).to_csv(
        TABLES / "exp10_environment.csv", index=False)
    print(f"\ntables -> {TABLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
