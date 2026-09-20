"""Experiment 03 -- Track A: Just-Pope risk modelling on real Bangladeshi farm data.

Objective
---------
Apply the base paper's Just-Pope three-step to the two risk outcomes it studies -- income risk and
disease risk -- using real open survey data, and compare the step-3 regressors on evidence rather
than preference.

What this is and is not
-----------------------
This reproduces the base paper's **method**, not its results. Different country (Bangladesh, not
Vietnam), different sample, different variable coding, and an outcome we had to construct because
the survey records no single income column. Its R2 = 0.700 (income) and R2 = 0.486 (disease) are
context, never targets to hit. Any sentence implying we replicated the paper's findings is false.

The three steps
---------------
    1  OLS of the outcome on covariates          y = f(x, beta) + e
    2  deviance, the squared residual            d_i = (y_i - y_hat_i)^2   estimates Var(y|x)
    3  log(d_i) as the response for ML           RF / SVM / GBM / linear, then importance + PDP

Step 2 is the whole point: the *variance* of the outcome, not its level, is what "risk" means here.
A farm can have high revenue and high risk at once, and the mean equation alone cannot see that.

Design
------
Two outcomes, each run end to end:

  INCOME   log revenue_total_bdt   n ~ 1,219 farms   (revenue is DERIVED -- see provenance table)
  DISEASE  wsd_loss_bdt            n ~ 463 farms     (White Spot Disease subset, observed)

Step-3 candidates are compared on identical repeated-K-fold splits and then on a common holdout,
with a baseline RMSE for reference. Reporting the winner without the baseline would hide the case
where every model is useless.

Outputs
-------
    reports/tables/exp03_provenance.csv        observed vs derived vs imputed, per variable
    reports/tables/exp03_mean_equations.csv    step 1 coefficients, robust SEs
    reports/tables/exp03_diagnostics.csv       R2, heteroskedasticity tests, VIF summary
    reports/tables/exp03_step3_candidates.csv  the model comparison
    reports/tables/exp03_importance.csv        permutation importance per model
    reports/tables/exp03_importance_overlap.csv
    reports/tables/exp03_environment.csv       what this ran on
    reports/figures/fig10_pdp_income.png
    reports/figures/fig11_pdp_disease.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shrimp_risk.common.paths import FIGURES, TABLES  # noqa: E402
from shrimp_risk.common.seeding import environment_fingerprint, seed_everything  # noqa: E402
from shrimp_risk.econometrics import data as D  # noqa: E402
from shrimp_risk.econometrics import just_pope as JP  # noqa: E402
from shrimp_risk.econometrics import ml_models as ML  # noqa: E402
from shrimp_risk.econometrics import pdp as PDP  # noqa: E402

SEED = 2025

# Outcome per model. Income uses a log transform because revenue is strongly right-skewed and the
# base paper log-transforms its income response; disease loss is modelled on the level, with the
# log transform tested as an alternative below.
SPECS = {
    "income": dict(builder=D.build_income_frame, outcome="revenue_total_bdt",
                   transform="log", response_label="log income risk"),
    # log1p, not log: 11 farms in the White Spot subset report a loss of exactly 0 BDT.
    # Dropping them would condition the disease-risk model on having suffered a loss, which is a
    # different question from the one being asked.
    "disease": dict(builder=D.build_disease_frame, outcome="wsd_loss_bdt",
                    transform="log1p", response_label="log1p disease-loss risk"),
}


def covariates_for(af, outcome: str) -> list[str]:
    """Choose the design matrix: observed covariates only, minus the outcome's own components.

    Derived quantities that feed the outcome (revenue, quantities sold) are excluded from the
    right-hand side. Regressing revenue on its own components would be an accounting identity
    dressed up as a model -- it would fit beautifully and mean nothing.
    """
    kinds = af.kinds() if callable(af.kinds) else af.kinds
    banned = {
        outcome, "farmer_uid",
        "revenue_total_bdt", "revenue_shrimp_bdt", "qty_sold_kg", "qty_produced_kg",
        "wsd_loss_bdt", "wsd_mortality_pct",
    }
    cols = [c for c, kind in kinds.items() if kind == "observed" and c not in banned]
    frame = af.frame
    # Drop anything constant in this subset -- a column with one value carries no information
    # and makes the design matrix singular.
    return [c for c in cols if c in frame.columns and frame[c].nunique(dropna=True) > 1]


def _cluster_robust_comparison(usable: pd.DataFrame, res, name: str) -> pd.DataFrame:
    """Refit step 1 with cluster-robust errors and report how much the inference moves.

    Only the standard errors change -- clustering is a statement about the error covariance, not
    about the conditional mean -- so R-squared and every coefficient are identical by
    construction and are not re-reported here. What is reported is the ratio of cluster-robust to
    HC3 standard errors and the count of coefficients that cross p = 0.05, because those are the
    two things a reader would otherwise have to take on trust.

    Returns an empty frame if the cluster key cannot be joined, rather than silently omitting the
    check: a missing row here means "not measured", and that is different from "measured, fine".
    """
    out = {"risk_model": name}
    try:
        key = D.load_ifpri_module("000_hh_commonvar")[["farmer_uid", "cluster_uid"]]
        joined = usable[["farmer_uid"]].merge(key, on="farmer_uid", how="left")
        groups = joined["cluster_uid"].to_numpy()
        if pd.isna(groups).any():
            raise ValueError("cluster_uid missing for some farms")

        # Refit from the SAME exog/endog the HC3 fit used, rather than rebuilding the design.
        # A rebuilt design could differ by a column ordering or a dropped row and the comparison
        # would then be between two different models, which is not what is being asked.
        fit = res.mean_equation.model
        clustered = sm.OLS(fit.model.endog, fit.model.exog).fit(
            cov_type="cluster", cov_kwds={"groups": groups})

        hc3_se = np.asarray(fit.bse, dtype="float64")
        cl_se = np.asarray(clustered.bse, dtype="float64")
        ratio = cl_se / hc3_se
        hc3_sig = np.asarray(fit.pvalues, dtype="float64") < 0.05
        cl_sig = np.asarray(clustered.pvalues, dtype="float64") < 0.05
        names = list(res.mean_equation.coefficients["variable"])

        out.update({
            "n_clusters": int(pd.Series(groups).nunique()),
            "mean_cluster_size": round(len(groups) / pd.Series(groups).nunique(), 2),
            "se_ratio_median": round(float(np.median(ratio)), 4),
            "se_ratio_max": round(float(ratio.max()), 4),
            "se_ratio_min": round(float(ratio.min()), 4),
            "n_sig_hc3": int(hc3_sig.sum()),
            "n_sig_clustered": int(cl_sig.sum()),
            "n_significance_flips": int((hc3_sig != cl_sig).sum()),
            "flipped": ";".join(sorted(
                str(c) for c, f in zip(names, hc3_sig != cl_sig) if f)),
            "note": "sensitivity check; HC3 remains the reported specification",
        })
    except Exception as exc:                                    # noqa: BLE001
        out["note"] = f"not measured: {exc}"
    return pd.DataFrame([out])


def main() -> int:
    seed_everything(SEED)
    warnings.filterwarnings("ignore", category=FutureWarning)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Persisted for the same reason every training run persists it: this experiment writes
    # eight result tables and none of them would otherwise say what produced them.
    pd.DataFrame([environment_fingerprint()]).to_csv(
        TABLES / "exp03_environment.csv", index=False)
    print("environment:", environment_fingerprint())
    labels = D.variable_labels()

    prov, means, diags, cands, imps, overlaps = [], [], [], [], [], []
    clusters: list[pd.DataFrame] = []
    pdp_skipped: list[dict] = []

    for name, spec in SPECS.items():
        print("\n" + "=" * 74)
        print(f"{name.upper()} RISK MODEL")
        print("=" * 74)

        af = spec["builder"]()
        frame = af.frame
        outcome = spec["outcome"]

        # Provenance is recorded before anything is estimated, so the table cannot drift from
        # what was actually modelled.
        p = af.provenance_table() if callable(af.provenance_table) else af.provenance_table
        p = p.copy()
        p.insert(0, "risk_model", name)
        prov.append(p)

        covs = covariates_for(af, outcome)
        usable = frame.dropna(subset=[outcome] + covs)
        print(f"  farms in frame        : {len(frame):,}")
        print(f"  farms usable (complete): {len(usable):,}  "
              f"({len(frame) - len(usable)} dropped for missing outcome/covariates)")
        print(f"  covariates            : {len(covs)}")

        if outcome == "revenue_total_bdt":
            # Persisted, not just printed: revenue is a DERIVED variable and the
            # reconciliation is the evidence that the construction is sound. A number quoted in
            # a document must trace to a file, not to a console line nobody kept.
            rec = D.reconcile_revenue(usable, D.load_ifpri_module("021_hh_modc10"))
            pd.DataFrame([rec]).to_csv(TABLES / "exp03_revenue_reconciliation.csv", index=False)
            print(f"  revenue reconciliation vs c1004: {rec}")

        # ---- steps 1 and 2 ----
        res = JP.run_just_pope(usable, outcome=outcome, covariates=covs,
                               transform=spec["transform"], label=name)
        row = res.summary_row() if callable(res.summary_row) else res.summary_row
        diags.append(dict(row))
        print(f"  step 1 R2             : {row['r_squared']:.4f}  (adj {row['adj_r_squared']:.4f})")
        print(f"  heteroskedasticity    : Breusch-Pagan p={row['breusch_pagan_p']:.3g}, "
              f"White p={row['white_p']:.3g}")
        print(f"  multicollinearity     : max VIF {row['max_vif']}, "
              f"{row['n_vif_above_5']} above 5")
        print(f"  step 2 log-deviance   : n={row['n']}, floored={row['n_deviance_floored']}")

        coefs = res.mean_equation.coefficients if hasattr(res, "mean_equation") else None
        if coefs is None and hasattr(res, "coefficients"):
            coefs = res.coefficients
        if coefs is not None:
            c = coefs.copy(); c.insert(0, "risk_model", name); means.append(c)

        # ---- sensitivity of step 1 to the survey's sampling design ----
        # IFPRI drew 1,221 farms from 68 clusters, ~18 per cluster. HC3 corrects for
        # heteroskedasticity; it does NOT correct for correlation between farms in the same
        # cluster, and neighbours sharing water, suppliers and extension advice are not
        # independent draws. This is the same question the vision track asks about photographs
        # of one shrimp, one track over, so it is measured rather than waved away. The result
        # is reported as a sensitivity check, not adopted as the headline: the design effect
        # turns out to be small, and saying so with a number is stronger than not asking.
        clusters.append(_cluster_robust_comparison(usable, res, name))

        # ---- step 3 ----
        X = usable[covs].reset_index(drop=True)
        y = res.log_deviance if hasattr(res, "log_deviance") else res.y_step3
        step3 = ML.evaluate_candidates(X, np.asarray(y), label=name, seed=SEED)

        cmp_tbl = step3.comparison if hasattr(step3, "comparison") else step3.table
        cmp_tbl = cmp_tbl.copy(); cmp_tbl.insert(0, "risk_model", name)
        cands.append(cmp_tbl)
        print(f"\n  step 3 candidates (baseline RMSE {step3.baseline_rmse:.4f}):")
        print(cmp_tbl.to_string(index=False))

        imp = ML.importance_table(step3, seed=SEED, labels=labels)
        imp = imp.copy(); imp.insert(0, "risk_model", name); imps.append(imp)

        ov = ML.importance_overlap(imp)
        ov = ov.copy(); ov.insert(0, "risk_model", name); overlaps.append(ov)

        # ---- partial dependence on the best candidate ----
        best_name = step3.best()                      # `best` is a method, not an attribute
        best_model = step3.fitted[best_name]
        best_row = cmp_tbl.loc[cmp_tbl["model"] == best_name].iloc[0]
        beats_baseline = best_row["holdout_rmse_vs_mean_baseline"] < 1.0
        print(f"  best step-3 model     : {best_name} "
              f"(holdout R2 {best_row['holdout_r2']:.4f}, "
              f"RMSE/baseline {best_row['holdout_rmse_vs_mean_baseline']:.4f})")

        # A partial dependence plot draws the relationship a model believes in. If that model
        # cannot beat predicting the mean, its beliefs are not evidence about the world, and
        # plotting them would manufacture a finding out of noise. So the PDP is produced only
        # when the model has earned it, and the refusal is recorded rather than hidden.
        if not beats_baseline:
            print("  PDP SKIPPED           : no candidate beats the mean baseline "
                  "(all holdout RMSE ratios >= 1.0). Partial dependence of a model with no "
                  "predictive skill would show structure that is not there.")
            pdp_skipped.append({
                "risk_model": name, "best_model": best_name,
                "holdout_r2": round(float(best_row["holdout_r2"]), 4),
                "rmse_vs_baseline": round(float(best_row["holdout_rmse_vs_mean_baseline"]), 4),
                "reason": "no candidate beat the mean baseline; PDP would be uninterpretable",
            })
        else:
            top = (imp.sort_values("importance", ascending=False)
                      .drop_duplicates("feature")["feature"].head(6).tolist())
            print(f"  PDP features          : {top}")
            pdp_tbl = PDP.partial_dependence_table(
                best_model, step3.X_train, top, model_name=str(best_name),
                response_label=spec["response_label"],
            )
            fig_path = FIGURES / ("fig10_pdp_income.png" if name == "income"
                                  else "fig11_pdp_disease.png")
            PDP.plot_partial_dependence(
                pdp_tbl, step3.X_train, fig_path,
                title=f"Partial dependence — {name} risk ({best_name})",
                subtitle=f"response: {spec['response_label']} · n={len(usable):,} farms · "
                         f"real survey data",
                labels=labels,
            )
            print(f"  wrote {fig_path.name}")

    pd.concat(prov).to_csv(TABLES / "exp03_provenance.csv", index=False)
    if means:
        pd.concat(means).to_csv(TABLES / "exp03_mean_equations.csv", index=False)
    if clusters:
        pd.concat(clusters, ignore_index=True).to_csv(
            TABLES / "exp03_cluster_robust.csv", index=False)
    pd.DataFrame(diags).to_csv(TABLES / "exp03_diagnostics.csv", index=False)
    pd.concat(cands).to_csv(TABLES / "exp03_step3_candidates.csv", index=False)
    pd.concat(imps).to_csv(TABLES / "exp03_importance.csv", index=False)
    pd.concat(overlaps).to_csv(TABLES / "exp03_importance_overlap.csv", index=False)
    if pdp_skipped:
        pd.DataFrame(pdp_skipped).to_csv(TABLES / "exp03_pdp_skipped.csv", index=False)

    print("\n" + "=" * 74)
    print("DIAGNOSTICS")
    print("=" * 74)
    print(pd.DataFrame(diags).to_string(index=False))
    print(f"\ntables -> {TABLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
