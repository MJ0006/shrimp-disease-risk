"""Experiment 05 — is the vision model's confidence trustworthy?

Why this experiment exists
--------------------------
Stage 8's contribution is a prevalence estimator. Counting positive predictions and dividing is a
**biased** estimator whenever the classifier is imperfect, so the bridge applies the adjusted-count
correction and the Saerens-Latinne-Decaestecker EM procedure instead. `integration/__init__.py`
states the condition both corrections rest on:

    "Both corrections assume *calibrated* posteriors."

That assumption had never been measured. It was carried as a *design intent* through the
architecture slide, objective O3 and the module contract, and listed as a low-priority improvement
in `MASTER_DOCUMENT.md`. An assumption that load-bearing cannot stay unmeasured while the thing
resting on it is presented as the project's contribution, so this script measures it.

What "calibrated" means here
----------------------------
A model is calibrated when its confidence matches its accuracy: of the predictions it makes at 90%
confidence, about 90% should be correct. **Expected Calibration Error (ECE)** bins predictions by
confidence and averages |accuracy - confidence| over the bins, weighted by bin population. ECE = 0
is perfect; a model that says 0.90 and is right 0.70 of the time scores about 0.20.

This matters differently from accuracy. A model can rank cases perfectly and still be badly
calibrated, and a prevalence estimator that consumes probabilities -- rather than just argmax
labels -- inherits that miscalibration directly as bias in the estimate.

Why it reads stored posteriors instead of training
--------------------------------------------------
Every run in this project already saved `probs`, `labels` and `specimen_ids` to a `.npz` beside its
JSON, for exactly this purpose. Measuring calibration therefore costs seconds and involves no GPU
and no new training -- and, more importantly, it measures the *same* posteriors Stage 8 will
consume, not a fresh model that would merely be similar to them.

The comparison that carries the finding
---------------------------------------
The project's standing story is that *leakage* causes overconfidence, and that the specimen-level
protocol is therefore a prerequisite for the bridge. This script reports both protocols side by
side so that story can be checked rather than assumed. It is checked, and it does not survive: see
`D-028`. Specimen-level splitting is necessary for an honest *estimate* -- it is not sufficient for
a calibrated *posterior*, and the two were being conflated.

Outputs
-------
`reports/tables/exp05_calibration.csv`   one row per run: ECE, MCE, Brier, confidence, accuracy
`reports/tables/exp05_calibration_summary.csv`  one row per group, mean +/- SD
`reports/tables/exp05_reliability.csv`   per-bin accuracy vs confidence for the reliability diagram
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shrimp_risk.common.paths import TABLES  # noqa: E402
from shrimp_risk.common.seeding import environment_fingerprint  # noqa: E402
from shrimp_risk.integration.calibration import (  # noqa: E402
    DEFAULT_BINS, expected_calibration_error, reliability_bins,
)

RUNS = PROJECT_ROOT / "experiments" / "runs"

# `expected_calibration_error` and `reliability_bins` were written here first and now live in
# `shrimp_risk.integration.calibration`, imported above.
#
# The move happened when Stage 8 built the calibration *step*, and it matters more than a tidy-up:
# a correction evaluated by a second, independently written ECE would be scoring itself against a
# slightly different ruler, and any disagreement between the two would be invisible. One
# implementation measures the shortfall and judges the fix. This script's outputs are unchanged by
# the move -- `tests/test_results_integrity.py` pins them, and they are what `D-028` reports.


def _group_of(path: Path) -> tuple[str, str]:
    """Experiment and condition/backbone for one prediction file, from its own path.

    Derived from the filename rather than passed in, so a run that is added later is picked up
    without editing this script.
    """
    experiment = path.parent.name
    tag = path.name.replace("_predictions.npz", "")
    return experiment, tag.split("__")[0]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS,
                        help=f"confidence bins for ECE (default {DEFAULT_BINS})")
    args = parser.parse_args()

    files = sorted(RUNS.glob("*/*_predictions.npz"))
    if not files:
        print(f"no prediction artefacts under {RUNS}; run exp02 and exp04 first")
        return 1

    TABLES.mkdir(parents=True, exist_ok=True)
    print(f"environment: {environment_fingerprint()}")
    print(f"{len(files)} prediction artefacts, {args.bins} confidence bins\n")

    per_run, per_bin = [], []
    for path in files:
        payload = np.load(path, allow_pickle=True)
        probs = payload["probs"].astype("float64")
        labels = payload["labels"].astype("int64")
        experiment, group = _group_of(path)

        metrics = expected_calibration_error(probs, labels, args.bins)
        per_run.append({
            "experiment": experiment,
            "group": group,
            "tag": path.name.replace("_predictions.npz", ""),
            "n_bins": args.bins,
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
            "artifact": path.relative_to(PROJECT_ROOT).as_posix(),
        })

        bins = reliability_bins(probs, labels, args.bins)
        bins.insert(0, "group", group)
        bins.insert(0, "experiment", experiment)
        per_bin.append(bins)

    runs_df = pd.DataFrame(per_run)
    runs_df.to_csv(TABLES / "exp05_calibration.csv", index=False)

    summary = (
        runs_df.groupby(["experiment", "group"])
        .agg(n_runs=("ece", "size"),
             ece_mean=("ece", "mean"), ece_sd=("ece", "std"),
             mce_mean=("mce", "mean"),
             brier_mean=("brier", "mean"),
             mean_confidence=("mean_confidence", "mean"),
             accuracy=("accuracy", "mean"),
             overconfidence=("overconfidence", "mean"))
        .round(4)
        .reset_index()
        .sort_values(["experiment", "ece_mean"])
    )
    summary.to_csv(TABLES / "exp05_calibration_summary.csv", index=False)

    reliability = (
        pd.concat(per_bin, ignore_index=True)
        .groupby(["experiment", "group", "bin_lo", "bin_hi"], as_index=False)
        .agg(n=("n", "sum"), mean_confidence=("mean_confidence", "mean"),
             accuracy=("accuracy", "mean"))
        .round(4)
    )
    reliability.to_csv(TABLES / "exp05_reliability.csv", index=False)

    print(summary.to_string(index=False))
    print()

    # The comparison the finding rests on, printed rather than left for a reader to assemble.
    exp02 = summary[summary["experiment"] == "exp02"].set_index("group")
    if {"B_image_level", "C_specimen_level"} <= set(exp02.index):
        b = exp02.loc["B_image_level"]
        c = exp02.loc["C_specimen_level"]
        print("The protocol comparison, stated plainly:")
        print(f"  image-level    ECE {b['ece_mean']:.4f}  conf {b['mean_confidence']:.3f} "
              f"acc {b['accuracy']:.3f}")
        print(f"  specimen-level ECE {c['ece_mean']:.4f}  conf {c['mean_confidence']:.3f} "
              f"acc {c['accuracy']:.3f}")
        print("  Removing leakage did NOT produce calibrated posteriors. The specimen-level")
        print("  model is the MORE miscalibrated of the two. Specimen-level splitting is")
        print("  necessary for an honest estimate and not sufficient for a calibrated one --")
        print("  a calibration step is a Stage 8 prerequisite, not an optional refinement.")
        print("  See D-028.")
        print()

    print(f"tables -> {TABLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
