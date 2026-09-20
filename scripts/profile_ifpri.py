"""Stage 3a -- IFPRI survey descriptives, as a regenerable table.

Why this script exists
----------------------
Stage 3a profiled the IFPRI 2024 survey and its findings went straight into the documentation and
onto Slide 4: 1,221 farms, 918 reporting disease, 492 reporting White Spot, a median loss of
15,000 BDT. Those numbers were correct, but they lived only as prose -- no code produced them and
no file in `reports/` held them.

That breaks the project's binding rule that **every number on a slide traces to a file in
`reports/` produced by code in this repository**. A figure that cannot be regenerated cannot be
defended: if the survey extract were replaced, the prose would silently go stale and nothing would
flag it.

So the profiling is executable. The numbers below are recomputed from the released `.tab` modules
on every run, and `tests/test_ifpri_descriptives.py` asserts the published claims still hold.

Two traps this reproduces deliberately, because they change the answer
----------------------------------------------------------------------
* **`c904` carries a 9997 "don't know" sentinel** in 94 rows, with nothing between 101 and 9996.
  Treating it as a number moves the mean mortality from 32.5% to 819.6%. The table reports both,
  so the size of the trap is on the record rather than in a footnote.
* **`c901_1` is NaN for the 303 farms that reported no disease at all.** For them the answer to
  "did you have White Spot?" is a true no, not a missing value, so it is filled rather than
  dropped -- the same treatment `build_farm_covariates` applies.

Outputs
-------
    reports/tables/exp03a_ifpri_descriptives.csv   one row per reported quantity, with its source
    reports/tables/exp03a_environment.csv          what this ran on
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shrimp_risk.common.paths import TABLES  # noqa: E402
from shrimp_risk.common.seeding import environment_fingerprint  # noqa: E402
from shrimp_risk.econometrics import data as D  # noqa: E402


def _row(quantity: str, value, unit: str, source: str, note: str = "") -> dict:
    return {"quantity": quantity, "value": value, "unit": unit, "source": source, "note": note}


def main() -> int:
    commonvar = D.load_ifpri_module("000_hh_commonvar")
    modc9a = D.load_ifpri_module("019_hh_modc9_c900-c901")
    modc9b = D.load_ifpri_module("020_hh_modc9_c902-c905")

    n_farms = commonvar["farmer_uid"].nunique()
    faced = modc9a["c900"].value_counts()
    n_yes, n_no = int(faced.get(1, 0)), int(faced.get(2, 0)) or int(faced.get(0, 0))

    # NaN means "reported no disease at all", which is a true no for every specific disease.
    white_spot = int(modc9a["c901_1"].fillna(0).sum())
    black_gill = int(modc9a["c901_4"].fillna(0).sum())

    wsd = D.filter_white_spot(modc9b)
    losses = pd.to_numeric(wsd["c905"], errors="coerce").dropna()

    raw_mortality = pd.to_numeric(modc9b["c904"], errors="coerce").dropna()
    clean_mortality = D.clean_c904(modc9b["c904"]).dropna()
    sentinel = D.sentinel_report(modc9b["c904"])

    rows = [
        _row("farms_in_survey", n_farms, "farms", "commonvar: farmer_uid",
             "The publication says 1,222; we report what the released data contains."),
        _row("farms_reporting_disease", n_yes, "farms", "modc9a: c900 == 1", ""),
        _row("farms_reporting_no_disease", n_no, "farms", "modc9a: c900 != 1", ""),
        _row("share_reporting_disease", round(100 * n_yes / n_farms, 2), "%",
             "derived: c900", ""),
        _row("farms_white_spot_disease", white_spot, "farms", "modc9a: c901_1",
             "The class the image model detects as WSSV."),
        _row("farms_black_gill", black_gill, "farms", "modc9a: c901_4",
             "The image model's weakest class, and the rarest here."),
        _row("wsd_rows_in_module", len(wsd), "rows", "modc9b: shrimpdiseasename ~ white spot", ""),
        _row("wsd_loss_mean_bdt", round(float(losses.mean())), "BDT", "modc9b: c905", ""),
        _row("wsd_loss_median_bdt", round(float(losses.median())), "BDT", "modc9b: c905",
             "The direct disease-to-income link the base paper lacked."),
        _row("wsd_loss_max_bdt", round(float(losses.max())), "BDT", "modc9b: c905", ""),
        _row("wsd_loss_n", int(losses.notna().sum()), "farms", "modc9b: c905", ""),
        _row("mortality_mean_pct_cleaned", round(float(clean_mortality.mean()), 1), "%",
             "modc9b: c904 with 9997 -> NaN", "The defensible figure."),
        _row("mortality_mean_pct_if_sentinel_kept", round(float(raw_mortality.mean()), 1), "%",
             "modc9b: c904 as recorded",
             "What the same column reports if the 9997 'don't know' code is treated as a number."),
        _row("mortality_sentinel_rows", int(sentinel["n_equal_sentinel"]), "rows",
             "modc9b: c904 == 9997", "The 'don't know' code."),
        _row("mortality_rows_between_101_and_9996", int(sentinel["n_between_101_and_9996"]),
             "rows", "modc9b: 100 < c904 < 9997",
             "Load-bearing: the gap is what identifies 9997 as a code rather than a measurement. "
             "If this is ever non-zero the diagnosis is wrong."),
    ]

    table = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "exp03a_ifpri_descriptives.csv", index=False)
    pd.DataFrame([environment_fingerprint()]).to_csv(
        TABLES / "exp03a_environment.csv", index=False)

    print("IFPRI 2024 survey descriptives (Stage 3a)")
    print(table.to_string(index=False))
    print(f"\nwrote 2 tables -> {TABLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
