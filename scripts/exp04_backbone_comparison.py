"""Experiment 04 -- Backbone selection under the corrected protocol, with deployment cost.

Objective
---------
Two questions, answered by one experiment:

  Q1  Which backbone performs best on ShrimpDiseaseImageBD when the test set is NOT
      contaminated by specimen leakage?
  Q2  What does each backbone actually cost to run? The registered project title promises an
      **edge-deployable** framework, and "edge-deployable" is a measurable claim -- parameters,
      model size and per-image latency -- not an adjective.

Why this experiment exists
--------------------------
Experiment 02 tested one backbone (`rexnet_150`) under one split. That is enough to establish the
leakage effect -- one factor varied, everything else held -- but it cannot choose an architecture,
because a single model's score says nothing about whether a different one would do better under the
same protocol. It also produced no deployment numbers at all, so "edge-deployable" in the registered
title had no evidence behind it. Selecting a backbone on reputation instead would be a preference
dressed as a finding. (Master Prompt section 9.)

Design
------
5 backbones x 5 folds of **StratifiedGroupKFold on specimen_id**, so:

  * every specimen is tested exactly once across the folds;
  * no photograph of a test shrimp ever appears in that fold's training set;
  * all five backbones see the IDENTICAL folds, seeds and budget, so differences are
    attributable to architecture and nothing else.

Cross-validation matters more here than it would on a larger corpus. Grouping reduces the
effective sample size to 416 specimens, and a single specimen-level test split holds only ~62 of
them. A one-off number on 62 specimens cannot rank architectures, so every figure is reported as
mean +/- SD across folds.

Metrics
-------
**Macro-F1 is the headline, never accuracy.** Classes are imbalanced (403/328/220/198 images;
142/118/81/75 specimens) and accuracy lets a model coast on Healthy -- it is what would have
hidden the Black Gill collapse found in Experiment 02.

Outputs
-------
    reports/tables/exp04_folds.csv        one row per (backbone, fold)
    reports/tables/exp04_summary.csv      mean +/- SD per backbone, with deployment cost
    reports/tables/exp04_deployment.csv   params / size / latency, the 'edge-deployable' evidence
    experiments/runs/exp04/*.json         per-run history, per-class metrics, confusion matrices
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
from shrimp_risk.vision import splits as S  # noqa: E402
from shrimp_risk.vision.datasets import CLASSES, class_weights, make_loaders  # noqa: E402
from shrimp_risk.vision.models import (  # noqa: E402
    CANDIDATES, create_model, deployment_profile, get_device,
)
from shrimp_risk.vision.trainer import TrainConfig, train_and_evaluate  # noqa: E402

SEED = 2025
N_FOLDS = 5

# Per-backbone learning rate.
#
# The first run held lr = 3e-4 constant across all five backbones, on the principle that a fair
# comparison changes nothing but the architecture. That principle was wrong here. ConvNeXt is a
# transformer-style architecture with LayerNorm and no BatchNorm; it is well documented to need a
# smaller step size than ResNet-family CNNs. At 3e-4 it failed to train in 3 of 5 folds -- fold 0
# went from train F1 0.244 to 0.202, i.e. backwards -- and the resulting 0.397 +/- 0.198 would have
# been reported as an architecture result when it was a harness artefact.
#
# Holding a hyperparameter constant is only "fair" when it is equally appropriate for every
# candidate. Where it is not, the fair comparison is each architecture at a learning rate it can
# actually train at. The override is declared here so the departure is visible, not buried.
BACKBONE_LR = {
    "convnext_tiny": 5e-5,
}
DEFAULT_LR = 3e-4


def summarise(folds_table: pd.DataFrame, deploy: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-fold table into the reported summary.

    Split out of `main` so the summary can be rebuilt from `exp04_folds.csv` without retraining
    -- it is a pure derivation of that table, and 25 runs is 45 minutes of GPU time to reproduce
    a rounding change.

    **The learning rate is deliberately excluded from the rounding.** `round(5e-05, 4)` is
    `0.0001` under round-half-even and was written to the table as `0.0`, which contradicted
    every document that quotes ConvNeXt's tuned rate of 5e-5. Metrics are rounded for
    readability; a hyperparameter is a setting, not a measurement, and is written as it was used.
    """
    metrics = (
        folds_table.groupby("backbone")
        .agg(
            folds=("fold", "count"),
            f1_macro_mean=("test_f1_macro", "mean"),
            f1_macro_sd=("test_f1_macro", "std"),
            acc_mean=("test_acc", "mean"),
            f1_BG_mean=("f1_BG", "mean"),
            train_seconds=("train_seconds", "mean"),
        )
        .round(4)
        .reset_index()
    )
    learning_rates = folds_table.groupby("backbone")["lr"].first().reset_index()
    out = (
        metrics.merge(learning_rates, on="backbone", how="left")
        .merge(
            deploy[["model_name", "total_params", "size_mb",
                    "latency_ms_mean", "latency_ms_std"]],
            left_on="backbone", right_on="model_name", how="left",
        )
        .drop(columns=["model_name"])
    )
    out["M_params"] = (out["total_params"] / 1e6).round(2)
    columns = ["backbone", "folds", "lr", "f1_macro_mean", "f1_macro_sd", "acc_mean",
               "f1_BG_mean", "train_seconds", "total_params", "size_mb",
               "latency_ms_mean", "latency_ms_std", "M_params"]
    return out[columns].sort_values("f1_macro_mean", ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--backbones", nargs="*", default=list(CANDIDATES))
    ap.add_argument("--summary-only", action="store_true",
                    help="rebuild exp04_summary.csv from the folds and deployment tables")
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold, 2 epochs, all backbones -- shape check only")
    ap.add_argument("--class-weighted", action="store_true",
                    help="inverse-frequency class weighting in the loss. NOT used for the "
                         "reported comparison -- see the Stage 10 ablation, which writes to its "
                         "own run directory so these artefacts stay as reported")
    args = ap.parse_args()

    # Smoke mode still builds a real k-fold object (StratifiedGroupKFold needs n_splits >= 2)
    # but evaluates only the first fold, so it exercises the true code path cheaply.
    smoke_one_fold = args.smoke
    if args.smoke:
        args.folds, args.epochs = 2, 2

    if args.summary_only:
        folds_table = pd.read_csv(TABLES / "exp04_folds.csv")
        deploy = pd.read_csv(TABLES / "exp04_deployment.csv")
        summary = summarise(folds_table, deploy)
        summary.to_csv(TABLES / "exp04_summary.csv", index=False)
        print("rebuilt from artefacts -- nothing trained")
        print(summary.to_string(index=False))
        return 0

    device = get_device()
    print("device     :", device)
    print("environment:", environment_fingerprint())

    manifest_path = resolve_image_manifest(256)
    manifest = S.load_raw_manifest(manifest_path)
    print(f"manifest   : {manifest_path.name}")
    print(f"             {len(manifest):,} images / {manifest.specimen_id.nunique()} specimens")

    # Folds are built ONCE and reused by every backbone. Rebuilding them per backbone would
    # risk giving different architectures different data, which is the one thing that would
    # invalidate the comparison.
    folds = S.make_specimen_folds(manifest, n_splits=args.folds, seed=SEED)
    for f in folds:
        leak = S.leakage_report(manifest, f)
        assert leak["test_images_with_train_sibling"] == 0, (
            f"fold {f.fold} leaks {leak['test_images_with_train_sibling']} images -- "
            "grouping is broken, refusing to run"
        )
    print(f"folds      : {len(folds)} x StratifiedGroupKFold on specimen_id, all verified leak-free\n")

    out_dir = RUNS / "exp04"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- deployment cost: architecture-only, measured before any training ----
    print("=" * 74)
    print("DEPLOYMENT COST (batch=1, the way edge inference actually runs)")
    print("=" * 74)
    deploy_rows = []
    for name in args.backbones:
        prof = deployment_profile(name, device).as_dict()
        prof["role"] = CANDIDATES.get(name, "")
        deploy_rows.append(prof)
        print(f"  {name:24s} {prof['total_params']/1e6:6.2f} M params  "
              f"{prof['size_mb']:7.2f} MB  {prof['latency_ms_mean']:6.2f} ms")
    deploy = pd.DataFrame(deploy_rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    deploy.to_csv(TABLES / "exp04_deployment.csv", index=False)
    print()

    # ---- accuracy under the corrected protocol ----
    rows = []
    for name in args.backbones:
        for fold in folds:
            tag = f"{name}__fold{fold.fold}"
            print("=" * 74)
            print(f"{tag}   (specimen-level, {len(fold.train)} train / "
                  f"{len(fold.test)} test images)")
            print("=" * 74)

            seed_everything(SEED)
            cfg = TrainConfig(
                model_name=name, seed=SEED, protocol="specimen_level_kfold",
                fold=fold.fold, augment=True, epochs=args.epochs,
                patience=7, batch_size=32,
                lr=BACKBONE_LR.get(name, DEFAULT_LR),
                class_weighted_loss=args.class_weighted,
            )
            loaders = make_loaders(
                manifest, fold, PROJECT_ROOT, batch_size=cfg.batch_size,
                img_size=cfg.img_size, augment=True, num_workers=4,
            )
            model = create_model(name, num_classes=len(CLASSES))

            # `class_weights` was imported here and never called, so `class_weighted_loss=True`
            # silently trained an unweighted model -- a flag that appears to work and does
            # nothing (`D-030`, open item 11). It is wired now, exactly as Experiment 02 wires
            # it, so the setting is honest whether or not this run uses it.
            #
            # The 25 stored runs all have `class_weighted_loss=False`, and rerunning this script
            # with the default reproduces them unchanged. The Stage 10 ablation runs as its own
            # experiment rather than through this flag, so Experiment 04's artefacts are never
            # overwritten by a differently-configured run wearing the same tag.
            weights = class_weights(manifest, fold.train) if cfg.class_weighted_loss else None
            res = train_and_evaluate(
                model, loaders, cfg, device, CLASSES, loss_weights=weights
            )

            rows.append({
                "backbone": name,
                "fold": fold.fold,
                "lr": cfg.lr,
                "n_train_images": len(fold.train),
                "n_test_images": len(fold.test),
                "n_test_specimens": int(pd.Series(
                    manifest.iloc[fold.test]["specimen_id"]).nunique()),
                "best_epoch": res.best_epoch,
                "best_val_f1": round(res.best_val_f1, 4),
                "test_acc": round(res.test_acc, 4),
                "test_f1_macro": round(res.test_f1_macro, 4),
                "test_f1_weighted": round(res.test_f1_weighted, 4),
                "train_seconds": round(res.train_seconds, 1),
                **{f"f1_{c}": round(res.per_class[c]["f1-score"], 4) for c in CLASSES},
                **{f"recall_{c}": round(res.per_class[c]["recall"], 4) for c in CLASSES},
                "run_artifact": f"experiments/runs/exp04/{tag}.json",
            })

            (out_dir / f"{tag}.json").write_text(json.dumps({
                "tag": tag,
                "config": asdict(res.config),
                "history": [asdict(h) for h in res.history],
                "per_class": res.per_class,
                "confusion": res.confusion,
                "environment": environment_fingerprint(),
            }, indent=2), encoding="utf-8")
            np.savez_compressed(
                out_dir / f"{tag}_predictions.npz",
                probs=res.test_probs, labels=res.test_labels,
                specimen_ids=manifest.iloc[fold.test]["specimen_id"].to_numpy(),
            )
            print()

            # Written after every run so a crash 20 runs in does not lose the first 19.
            pd.DataFrame(rows).to_csv(TABLES / "exp04_folds.csv", index=False)

    df = pd.DataFrame(rows)
    summary = summarise(df, deploy)
    _unused = (
        df.groupby("backbone")
        .agg(
            folds=("fold", "count"),
            f1_macro_mean=("test_f1_macro", "mean"),
            f1_macro_sd=("test_f1_macro", "std"),
            acc_mean=("test_acc", "mean"),
            acc_sd=("test_acc", "std"),
            f1_BG_mean=("f1_BG", "mean"),
            train_seconds=("train_seconds", "mean"),
        )
        .round(4)
        .reset_index()
        .merge(
            deploy[["model_name", "total_params", "size_mb", "latency_ms_mean"]],
            left_on="backbone", right_on="model_name", how="left",
        )
        .drop(columns=["model_name"])
        .sort_values("f1_macro_mean", ascending=False)
    )
    summary.to_csv(TABLES / "exp04_summary.csv", index=False)

    print("=" * 74)
    print("EXPERIMENT 04 SUMMARY -- specimen-level k-fold, ranked by macro-F1")
    print("=" * 74)
    print(summary.to_string(index=False))
    print("\nFeatherNetX reference point for 'edge-deployable': 0.739 M params, 2.82 MB.")
    print("Interpret accuracy differences against fold-to-fold SD -- a gap smaller than the SD")
    print("does not support ranking one backbone above another.")
    print(f"\ntables -> {TABLES}\nruns   -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
