"""Experiment 11 -- does Track A's negative result hold on an independent sample?

The question
------------
Stage 6 found that farm and management covariates predict neither income risk nor disease-loss risk
on the IFPRI survey: every step-3 model is worse than predicting the mean. Experiment 10 showed that
finding survives cluster-grouped folds and a mean equation that passes RESET. All of that is one
sample.

Hasan & Haque (2020) is a second, independent survey of **233 shrimp farms in the same three
Bangladeshi districts**, downloaded and SHA-256 verified in Stage 2 and **never used** (improvement
3). If the same negative result appears there, it stops being a property of one dataset. If it does
not, that is more interesting still, and the difference between the two samples becomes the finding.

Why this sample is worth more than a second opinion
----------------------------------------------------
Its outcome is `CurrentPrevalance(%)` -- **the share of stock currently infected with WSSV**. That is
the quantity the vision bridge estimates, measured directly, on real farms. IFPRI has no such
variable: its closest analogue is `c904`, mortality attributable to White Spot, which equals
infection prevalence only if every infected shrimp dies. So this is the only place in the project
where the econometric outcome and the classifier's target are the *same construct*.

It also carries water temperature, pH and salinity, which IFPRI does not -- at the cost of 28-30%
missingness, which is why the experiment runs two arms rather than quietly dropping a third of the
sample.

Two arms, because complete-case analysis is a decision
--------------------------------------------------------
    A  no water quality        n = 233, every farm, survey covariates only
    B  with water quality      complete-case, n ~ 163, adds temperature / pH / salinity

Arm B is not simply "arm A plus three variables": it is a **different sample**, because the farms
with missing water readings are dropped. Reporting only B would silently condition the result on
having been measured. Reporting only A would discard the variables most likely to explain disease
risk. Both are reported and the sample sizes are printed beside them.

What is NOT claimed
--------------------
This is not a transfer of coefficients. The two surveys share districts and a disease, not a
covariate set -- IFPRI has biosecurity flags and pond tenure, Hasan & Haque has water chemistry and
stocking practice. What transfers, or fails to, is the **finding**: whether farm covariates carry
information about risk at all. Comparing individual coefficients across the two would be comparing
different models fitted to different variables and calling the difference a result.

`VirusDetected` is excluded by the loader and must stay excluded: of the 89 farms where the virus
was not detected, **all 89** report zero prevalence. It is a laboratory confirmation of the same
infection the outcome measures, and including it would put the answer on the right-hand side --
the same leakage this project's vision half exists to expose, in tabular form.

Outputs
-------
    reports/tables/exp11_external_diagnostics.csv  step 1 + step 2, both arms
    reports/tables/exp11_external_step3.csv        step 3 candidates against the mean baseline
    reports/tables/exp11_comparison.csv            IFPRI vs Hasan & Haque, side by side
    reports/tables/exp11_environment.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shrimp_risk.common.paths import TABLES  # noqa: E402
from shrimp_risk.common.seeding import environment_fingerprint, seed_everything  # noqa: E402
from shrimp_risk.econometrics import data as D  # noqa: E402
from shrimp_risk.econometrics import just_pope as JP  # noqa: E402
from shrimp_risk.econometrics import ml_models as ML  # noqa: E402

SEED = 2025

# `CurrentPrevalance(%)` is a bounded proportion with 94 exact zeros of 233 (40.3%). `log1p` is
# used for comparability with the IFPRI disease model, which is also `log1p` on a zero-inflated
# outcome -- the point of an external validation is to run the same procedure, not a better one.
# Experiment 10's specification search is the place where the functional form is interrogated.
TRANSFORM = "log1p"


def run_arm(frame: pd.DataFrame, covariates: list[str], label: str) -> tuple[dict, pd.DataFrame]:
    outcome = D.D2_OUTCOME
    usable = frame.dropna(subset=[outcome] + covariates).reset_index(drop=True)
    res = JP.run_just_pope(usable, outcome=outcome, covariates=covariates,
                           transform=TRANSFORM, label=label)
    row = res.summary_row() if callable(res.summary_row) else res.summary_row
    row = dict(row)
    row["arm"] = label
    row["n_covariates"] = len(covariates)

    step3 = ML.evaluate_candidates(res.X, np.asarray(res.log_deviance), label=label, seed=SEED)
    table = step3.comparison if hasattr(step3, "comparison") else step3.table
    table = table.copy()
    table.insert(0, "arm", label)
    table["baseline_rmse"] = round(float(step3.baseline_rmse), 4)
    # `holdout_rmse / baseline_rmse`, the project convention: ABOVE 1.0 is worse than the mean.
    table["holdout_rmse_vs_mean_baseline"] = (
        table["holdout_rmse"] / step3.baseline_rmse).round(4)
    best = table.sort_values("holdout_rmse").iloc[0]
    row["best_model"] = best["model"]
    row["best_holdout_rmse_vs_mean"] = round(float(best["holdout_rmse_vs_mean_baseline"]), 4)
    row["best_cv_r2_mean"] = round(float(best["cv_r2_mean"]), 4)
    row["best_cv_r2_sd"] = round(float(best["cv_r2_sd"]), 4)
    row["n_holdout"] = int(step3.n_holdout)

    # Three separate questions, reported separately, because collapsing them is how a holdout
    # fluke becomes a finding.
    #
    #   beats_mean_on_holdout   did the best model beat the mean on the untouched holdout?
    #   cv_r2_exceeds_its_sd    is its cross-validated skill larger than its own spread?
    #   skill_supported         both -- the only combination that supports a positive claim
    #
    # The holdout here is ~47 farms (arm A) and ~30 (arm B). A single ratio on 30 observations is
    # exactly the "ranking of luck" that `evaluate_candidates` computes cross-validated scores to
    # avoid, and this project's standing rule -- a gap smaller than its own SD does not support a
    # claim (`D-027`, Stage 5) -- applies to skill as much as to a ranking.
    row["beats_mean_on_holdout"] = bool(best["holdout_rmse_vs_mean_baseline"] < 1.0)
    row["cv_r2_exceeds_its_sd"] = bool(best["cv_r2_mean"] > best["cv_r2_sd"])
    row["skill_supported"] = bool(
        row["beats_mean_on_holdout"] and row["cv_r2_exceeds_its_sd"])
    return row, table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    seed_everything(SEED)
    print("environment:", environment_fingerprint())

    af = D.load_hasan_haque()
    frame = af.frame
    print(f"Hasan & Haque: {len(frame)} farms")
    for key in ("n_zero_outcome", "share_zero_outcome", "outcome_min", "outcome_max",
                "n_complete_water_quality"):
        print(f"  {key:32s} {af.notes[key]}")
    print("  VirusDetected excluded: of "
          f"{af.notes['excluded_VirusDetected_n_undetected']} farms with no detection, "
          f"{af.notes['excluded_VirusDetected_zero_prev_when_undetected']} report zero prevalence")

    # `SlNo` is a row number and would be a perfect index for a tree to memorise. The outcome and
    # the water-quality block are held out of arm A by construction.
    excluded = {D.D2_OUTCOME, "SlNo", *D.D2_WATER_QUALITY}
    survey_covariates = [c for c in frame.columns
                         if c not in excluded and frame[c].nunique(dropna=True) > 1]
    with_water = survey_covariates + D.D2_WATER_QUALITY

    print(f"\narm A covariates: {len(survey_covariates)}")
    print(f"arm B covariates: {len(with_water)} (adds {D.D2_WATER_QUALITY})")

    rows, tables = [], []
    for label, covs in (("A_no_water_quality", survey_covariates),
                        ("B_with_water_quality", with_water)):
        print(f"\n[{label}]")
        row, table = run_arm(frame, covs, label)
        rows.append(row)
        tables.append(table)
        print(f"  n                     : {row['n']}")
        print(f"  step 1 R2             : {row['r_squared']:.4f}")
        print(f"  Breusch-Pagan p       : {row['breusch_pagan_p']:.4g}")
        print(f"  RESET2 p              : {row['reset2_p']:.4g}")
        print(f"  RESET3 p              : {row['reset3_p']:.4g}")
        print(f"  best step-3 model     : {row['best_model']} at "
              f"rmse/baseline {row['best_holdout_rmse_vs_mean']:.4f} "
              f"on {row['n_holdout']} holdout farms")
        print(f"  its cross-validated R2: {row['best_cv_r2_mean']:+.4f} "
              f"+/- {row['best_cv_r2_sd']:.4f}")
        print(f"  beats mean on holdout : {row['beats_mean_on_holdout']}")
        print(f"  cv R2 exceeds its SD  : {row['cv_r2_exceeds_its_sd']}")
        print(f"  SKILL SUPPORTED       : {row['skill_supported']}")

    diagnostics = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(TABLES / "exp11_external_diagnostics.csv", index=False)
    pd.concat(tables, ignore_index=True).to_csv(
        TABLES / "exp11_external_step3.csv", index=False)

    # ---- side by side with the IFPRI disease model ----
    comparison_rows = []
    ifpri_path = TABLES / "exp03_diagnostics.csv"
    if ifpri_path.exists():
        ifpri = pd.read_csv(ifpri_path)
        disease = ifpri[ifpri["model"].astype(str).str.contains("disease", case=False)]
        if len(disease):
            d = disease.iloc[0]
            comparison_rows.append({
                "sample": "IFPRI 2023 baseline (disease)",
                "n": int(d["n"]),
                "outcome": "wsd_loss_bdt (log1p) -- financial loss",
                "r_squared": round(float(d["r_squared"]), 4),
                "breusch_pagan_p": float(d["breusch_pagan_p"]),
                "reset2_p": float(d["reset2_p"]),
                "beats_mean_on_holdout": False,
                "skill_supported": False,
                "note": "Stage 6; best model 1.0794x the mean baseline",
            })
    for row in rows:
        comparison_rows.append({
            "sample": f"Hasan & Haque 2020 ({row['arm']})",
            "n": int(row["n"]),
            "outcome": "CurrentPrevalance(%) (log1p) -- share of stock infected",
            "r_squared": round(float(row["r_squared"]), 4),
            "breusch_pagan_p": float(row["breusch_pagan_p"]),
            "reset2_p": float(row["reset2_p"]),
            "beats_mean_on_holdout": bool(row["beats_mean_on_holdout"]),
            "skill_supported": bool(row["skill_supported"]),
            "note": (f"best model {row['best_holdout_rmse_vs_mean']}x the mean on "
                     f"{row['n_holdout']} holdout farms; cv R2 "
                     f"{row['best_cv_r2_mean']:+.3f} +/- {row['best_cv_r2_sd']:.3f}"),
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(TABLES / "exp11_comparison.csv", index=False)

    pd.DataFrame([environment_fingerprint()]).to_csv(
        TABLES / "exp11_environment.csv", index=False)

    print("\n" + "=" * 92)
    print("EXPERIMENT 11 -- external validation against an independent sample")
    print("=" * 92)
    print(comparison.to_string(index=False))
    print()
    print("`skill_supported` is the question, and it needs BOTH columns before it. Stage 6's")
    print("finding is that farm covariates carry no usable information about risk. A model that")
    print("beats the mean on a 30-to-47-farm holdout while its cross-validated R2 is smaller than")
    print("its own standard deviation has not shown skill -- it has shown one lucky split. That is")
    print("the same rule this project applies to backbone rankings (`D-027`).")
    print()
    print("The two samples share districts and a disease, NOT a covariate set. What is compared is")
    print("the finding, never the coefficients.")
    print(f"\ntables -> {TABLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
