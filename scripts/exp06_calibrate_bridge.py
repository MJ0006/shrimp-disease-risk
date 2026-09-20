"""Experiment 06 -- fit a calibration temperature, and find out whether it is enough.

Why this experiment exists
--------------------------
Experiment 05 measured the assumption the whole bridge rests on and it failed: Expected Calibration
Error is about 0.20 at specimen level, with the models overconfident by +0.197 (`D-028`). Both
prevalence corrections consume estimated posteriors, so that error propagates into the estimate as
bias rather than noise. `D-028` therefore made a calibration step a **Stage 8 prerequisite** instead
of an optional refinement, and set the target that would retire the limitation:

    specimen-level ECE at or below 0.05 after calibration.

This script is the attempt, and the number it produces decides which of two things Stage 8 reports:
a corrected estimator whose stated precondition is met, or a corrected estimator that names the
calibration it actually achieved. Both are honest; only one is clean. The target was written down
before the run (Master Plan Section 6, Stage 8, pre-registered null iii), so whichever way it lands
is a result rather than a rationalisation.

Why it has to retrain, when Experiment 05 did not
--------------------------------------------------
A temperature must be fitted on data that is neither training nor test. Fitting on test would tune a
parameter on the set the score is reported from -- the exact failure this project exists to
document. Fitting on train would see images the model has already memorised, and an overconfident
model looks perfectly calibrated on its own training data.

That leaves the validation fold, and none of the 34 stored runs saved validation posteriors -- only
test. So the models are retrained with `trainer.py` now scoring validation on the **restored best
weights** after early stopping, rather than reusing the posteriors collected during the epoch that
happened to be running. Those are different networks whenever the best epoch was not the last, and a
temperature fitted on the wrong one corrects a model nobody is using.

What it does NOT do: overwrite Experiment 04
--------------------------------------------
Output goes to `experiments/runs/exp06/`. Experiment 04's artefacts are left alone. `D-020` cost
this project a set of per-epoch histories by regenerating in place, and they are still
unrecoverable; the rule since is that a new question gets a new directory.

A side effect worth having: because the folds, seeds, learning rates and budget are identical to
Experiment 04's, the test scores here are a **reproducibility check** on that experiment. The
comparison is reported per fold. A small difference is cuDNN and dataloader-worker nondeterminism
and is expected; a large one would mean something in the pipeline has moved since 2026-08-29 and
would need to be found before anything downstream is believed.

Which two backbones, and why not all five
------------------------------------------
`convnext_tiny` -- best under the corrected protocol at 0.7189 +/- 0.0453 -- and
`mobilenetv3_large_100` -- the edge pick at 4.21 M / 16.14 MB, 6.6x smaller. Those are the two
models any deployment story would actually use, and the bridge needs calibrated posteriors from the
model it will run on, not from all five. The other three would cost 15 more runs to answer a
question nothing downstream asks. If the two disagree about whether calibration is achievable, that
is itself the finding and the remaining three become worth running.

What calibration cannot change
------------------------------
A single positive scalar divides every logit equally, so it cannot reorder them. **Argmax is
invariant**, and therefore so are accuracy, macro-F1, per-class recall and the confusion matrix.
Nothing Stage 4 or Stage 5 reported can move because of this step, and the script asserts it rather
than asserting it in prose: if calibrated macro-F1 differs from uncalibrated macro-F1 by more than
floating-point noise, the run aborts, because the only way that happens is a bug.

Outputs
-------
    reports/tables/exp06_calibration.csv   one row per (backbone, fold): T, ECE before/after, NLL
    reports/tables/exp06_summary.csv       mean +/- SD per backbone, and the verdict against 0.05
    reports/tables/exp06_reliability.csv   per-bin accuracy vs confidence, before and after
    experiments/runs/exp06/*.json          history, per-class metrics, confusion, environment, T
    experiments/runs/exp06/*_predictions.npz  test AND validation posteriors, plus the temperature
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shrimp_risk.common.paths import (  # noqa: E402
    PROJECT_ROOT, RUNS, TABLES, resolve_image_manifest,
)
from shrimp_risk.common.seeding import environment_fingerprint, seed_everything  # noqa: E402
from shrimp_risk.integration.calibration import (  # noqa: E402
    DEFAULT_BINS, apply_temperature, expected_calibration_error, fit_temperature,
    negative_log_likelihood, reliability_bins,
)
from shrimp_risk.vision import splits as S  # noqa: E402
from shrimp_risk.vision.datasets import CLASSES, make_loaders  # noqa: E402
from shrimp_risk.vision.models import create_model, get_device  # noqa: E402
from shrimp_risk.vision.trainer import TrainConfig, train_and_evaluate  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

SEED = 2025
N_FOLDS = 5

# Identical to Experiment 04's, deliberately -- see the module docstring. A different learning rate
# here would make the reproducibility comparison meaningless and would change what is being
# calibrated.
BACKBONE_LR = {"convnext_tiny": 5e-5, "mobilenetv3_large_100": 3e-4}

# `D-028`'s reversal condition, stated as a constant so the verdict in the output is computed
# against the number that was written down rather than against one chosen after seeing the result.
ECE_TARGET = 0.05


def _verdict(ece: float) -> str:
    """The pre-registered reading of a post-calibration ECE."""
    if ece <= ECE_TARGET:
        return f"MET (<= {ECE_TARGET}) -- D-028's limitation can be retired"
    if ece <= 2 * ECE_TARGET:
        return f"CLOSE (> {ECE_TARGET}) -- improved, limitation stays declared"
    return f"NOT MET (> {2 * ECE_TARGET}) -- limitation stands, bridge reports at this ECE"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--backbones", nargs="*", default=list(BACKBONE_LR))
    ap.add_argument("--bins", type=int, default=DEFAULT_BINS)
    ap.add_argument("--smoke", action="store_true",
                    help="2 folds, 2 epochs, first fold only -- shape check")
    args = ap.parse_args()

    smoke_one_fold = args.smoke
    if args.smoke:
        args.folds, args.epochs = 2, 2

    device = get_device()
    print("device     :", device)
    print("environment:", environment_fingerprint())

    manifest_path = resolve_image_manifest(256)
    manifest = S.load_raw_manifest(manifest_path)
    print(f"manifest   : {manifest_path.name}")
    print(f"             {len(manifest):,} images / {manifest.specimen_id.nunique()} specimens")

    folds = S.make_specimen_folds(manifest, n_splits=args.folds, seed=SEED)
    for f in folds:
        leak = S.leakage_report(manifest, f)
        assert leak["test_images_with_train_sibling"] == 0, (
            f"fold {f.fold} leaks {leak['test_images_with_train_sibling']} images -- "
            "grouping is broken, refusing to run"
        )
        assert leak["shared_train_val_specimens"] == 0, (
            f"fold {f.fold} shares specimens between train and val -- the temperature would be "
            "fitted on memorised data, refusing to run"
        )
    if smoke_one_fold:
        folds = folds[:1]
    print(f"folds      : {len(folds)} x StratifiedGroupKFold on specimen_id, "
          "verified leak-free on BOTH test and val\n")

    # Experiment 04's per-fold scores, for the reproducibility comparison. Absent on a fresh
    # checkout, in which case the comparison columns are simply blank rather than fabricated.
    #
    # The comparison is only made at the fold geometry Experiment 04 actually used. Under
    # `--smoke` the splitter builds 2 folds, so "fold 0" holds 575 test images against Experiment
    # 04's 231 -- a different partition of a different size, and comparing the two scores would
    # produce a large delta that means nothing except that the folds differ. A number that looks
    # like a reproducibility failure and is not one is worse than no number, so the columns stay
    # empty instead.
    exp04_path = TABLES / "exp04_folds.csv"
    comparable = args.folds == N_FOLDS and not smoke_one_fold
    exp04 = pd.read_csv(exp04_path) if (exp04_path.exists() and comparable) else None
    if exp04_path.exists() and not comparable:
        print("note       : fold geometry differs from Experiment 04's 5 folds -- "
              "the reproducibility comparison is skipped, not estimated")

    out_dir = RUNS / "exp06"
    out_dir.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    rows, bin_frames = [], []
    for name in args.backbones:
        for fold in folds:
            tag = f"{name}__fold{fold.fold}"
            print("=" * 78)
            print(f"{tag}   ({len(fold.train)} train / {len(fold.val)} val / "
                  f"{len(fold.test)} test images)")
            print("=" * 78)

            seed_everything(SEED)
            cfg = TrainConfig(
                model_name=name, seed=SEED, protocol="specimen_level_kfold",
                fold=fold.fold, augment=True, epochs=args.epochs,
                patience=7, batch_size=32, lr=BACKBONE_LR.get(name, 3e-4),
            )
            loaders = make_loaders(
                manifest, fold, PROJECT_ROOT, batch_size=cfg.batch_size,
                img_size=cfg.img_size, augment=True, num_workers=4,
            )
            model = create_model(name, num_classes=len(CLASSES))
            res = train_and_evaluate(model, loaders, cfg, device, CLASSES)

            # ---- fit on validation, apply to test ----
            temperature, fit_info = fit_temperature(res.val_probs, res.val_labels)
            test_cal = apply_temperature(res.test_probs, temperature)
            val_cal = apply_temperature(res.val_probs, temperature)

            before = expected_calibration_error(
                res.test_probs.astype("float64"), res.test_labels, args.bins)
            after = expected_calibration_error(test_cal, res.test_labels, args.bins)

            # ---- the invariant, asserted rather than asserted-in-prose ----
            f1_before = f1_score(res.test_labels, res.test_probs.argmax(1),
                                 average="macro", zero_division=0)
            f1_after = f1_score(res.test_labels, test_cal.argmax(1),
                                average="macro", zero_division=0)
            assert abs(f1_before - f1_after) < 1e-9, (
                f"{tag}: temperature scaling changed macro-F1 from {f1_before} to {f1_after}. "
                "A positive scalar cannot reorder logits, so this is a bug, not a result."
            )

            exp04_f1 = None
            if exp04 is not None:
                match = exp04[(exp04.backbone == name) & (exp04.fold == fold.fold)]
                if len(match) == 1:
                    exp04_f1 = float(match.iloc[0]["test_f1_macro"])

            print(f"  temperature        {temperature:.4f}"
                  f"{'  [AT BOUND -- treat as a failed fit]' if fit_info['at_bound'] else ''}")
            print(f"  test ECE           {before['ece']:.4f}  ->  {after['ece']:.4f}")
            print(f"  test MCE           {before['mce']:.4f}  ->  {after['mce']:.4f}")
            print(f"  mean confidence    {before['mean_confidence']:.4f}  ->  "
                  f"{after['mean_confidence']:.4f}   (accuracy {after['accuracy']:.4f})")
            print(f"  val NLL            {fit_info['nll_before']:.4f}  ->  "
                  f"{fit_info['nll_after']:.4f}")
            print(f"  macro-F1           {f1_before:.4f}  (unchanged by construction)"
                  + (f"   exp04 was {exp04_f1:.4f}" if exp04_f1 is not None else ""))

            rows.append({
                "backbone": name,
                "fold": fold.fold,
                "lr": cfg.lr,
                "n_val_images": len(fold.val),
                "n_test_images": len(fold.test),
                "best_epoch": res.best_epoch,
                "temperature": round(temperature, 4),
                "at_bound": fit_info["at_bound"],
                "ece_before": round(before["ece"], 4),
                "ece_after": round(after["ece"], 4),
                "mce_before": round(before["mce"], 4),
                "mce_after": round(after["mce"], 4),
                "brier_before": round(before["brier"], 4),
                "brier_after": round(after["brier"], 4),
                "confidence_before": round(before["mean_confidence"], 4),
                "confidence_after": round(after["mean_confidence"], 4),
                "accuracy": round(after["accuracy"], 4),
                "overconfidence_before": round(before["overconfidence"], 4),
                "overconfidence_after": round(after["overconfidence"], 4),
                "val_nll_before": round(fit_info["nll_before"], 4),
                "val_nll_after": round(fit_info["nll_after"], 4),
                "val_ece_before": round(expected_calibration_error(
                    res.val_probs.astype("float64"), res.val_labels, args.bins)["ece"], 4),
                "val_ece_after": round(expected_calibration_error(
                    val_cal, res.val_labels, args.bins)["ece"], 4),
                "test_f1_macro": round(float(f1_before), 4),
                "exp04_f1_macro": round(exp04_f1, 4) if exp04_f1 is not None else None,
                "f1_delta_vs_exp04": (round(float(f1_before) - exp04_f1, 4)
                                      if exp04_f1 is not None else None),
                "n_bins": args.bins,
                "run_artifact": f"experiments/runs/exp06/{tag}.json",
            })

            for state, probs in (("before", res.test_probs.astype("float64")),
                                 ("after", test_cal)):
                b = reliability_bins(probs, res.test_labels, args.bins)
                b.insert(0, "state", state)
                b.insert(0, "fold", fold.fold)
                b.insert(0, "backbone", name)
                bin_frames.append(b)

            (out_dir / f"{tag}.json").write_text(json.dumps({
                "tag": tag,
                "config": asdict(res.config),
                "history": [asdict(h) for h in res.history],
                "per_class": res.per_class,
                "confusion": res.confusion,
                "calibration": {
                    "temperature": temperature,
                    "fitted_on": "validation fold, restored best weights",
                    **{k: v for k, v in fit_info.items() if k != "temperature"},
                    "ece_before": before["ece"],
                    "ece_after": after["ece"],
                    "target": ECE_TARGET,
                },
                "environment": environment_fingerprint(),
            }, indent=2), encoding="utf-8")

            # Test posteriors keep the key names every other run in this project uses, so
            # `exp05_calibration.py` picks these up unchanged and the "every stored run is
            # measured" property keeps holding. Validation posteriors and the temperature ride
            # along under distinct keys -- and the filename deliberately still ends in
            # `_predictions.npz` rather than hiding from that glob.
            np.savez_compressed(
                out_dir / f"{tag}_predictions.npz",
                probs=res.test_probs, labels=res.test_labels,
                specimen_ids=manifest.iloc[fold.test]["specimen_id"].to_numpy(),
                val_probs=res.val_probs, val_labels=res.val_labels,
                val_specimen_ids=manifest.iloc[fold.val]["specimen_id"].to_numpy(),
                temperature=np.array([temperature]),
            )
            print()

            # Written after every run so a crash does not lose the completed ones.
            pd.DataFrame(rows).to_csv(TABLES / "exp06_calibration.csv", index=False)

    df = pd.DataFrame(rows)
    pd.concat(bin_frames, ignore_index=True).to_csv(
        TABLES / "exp06_reliability.csv", index=False)

    summary = (
        df.groupby("backbone")
        .agg(folds=("fold", "count"),
             temperature_mean=("temperature", "mean"),
             temperature_sd=("temperature", "std"),
             ece_before_mean=("ece_before", "mean"),
             ece_after_mean=("ece_after", "mean"),
             ece_after_sd=("ece_after", "std"),
             mce_after_mean=("mce_after", "mean"),
             brier_before_mean=("brier_before", "mean"),
             brier_after_mean=("brier_after", "mean"),
             overconfidence_after=("overconfidence_after", "mean"),
             f1_macro_mean=("test_f1_macro", "mean"))
        .round(4)
        .reset_index()
    )
    summary["ece_reduction"] = (summary["ece_before_mean"] - summary["ece_after_mean"]).round(4)
    summary["target"] = ECE_TARGET
    summary["verdict"] = summary["ece_after_mean"].map(_verdict)
    summary.to_csv(TABLES / "exp06_summary.csv", index=False)

    print("=" * 78)
    print("EXPERIMENT 06 SUMMARY -- temperature fitted on validation, ECE measured on test")
    print("=" * 78)
    print(summary.drop(columns=["verdict"]).to_string(index=False))
    print()
    for _, r in summary.iterrows():
        print(f"  {r['backbone']:24s} ECE {r['ece_before_mean']:.4f} -> "
              f"{r['ece_after_mean']:.4f}   {r['verdict']}")
    print()
    print("Macro-F1 is unchanged by construction: a positive scalar divides every logit equally")
    print("and cannot reorder them, so accuracy, per-class recall and the confusion matrix all")
    print("stand exactly as Stage 5 reported them. Calibration changes what the probabilities")
    print("mean, not which class is predicted.")
    if df["exp04_f1_macro"].notna().any():
        delta = df["f1_delta_vs_exp04"].abs()
        print(f"\nReproducibility vs Experiment 04: max |delta macro-F1| = {delta.max():.4f}, "
              f"mean {delta.mean():.4f} over {int(delta.notna().sum())} matched folds.")
    print(f"\ntables -> {TABLES}\nruns   -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
