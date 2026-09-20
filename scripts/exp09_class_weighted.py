"""Experiment 09 -- does class-weighted loss rescue Black Gill?

The open item this closes
--------------------------
Stage 5 listed class-balanced loss among the repairs under test. It was never run: all 34 stored
runs carry `class_weighted_loss=False`, and the flag in `exp04_backbone_comparison.py` was **dead**
-- the helper was imported and never called, so setting it to True would have trained an unweighted
model and reported it as a weighted one. `D-030` recorded the omission rather than glossing it,
declared it as limitation C5, and scheduled it for Review 3. The dead flag is now wired (open item
11) and this is the experiment.

Why Black Gill is the target
-----------------------------
It is the weakest class for every backbone -- best F1 **0.6405** across the whole of Experiment 04 --
and it has the fewest specimens: **75** of 416, against Healthy's 142. It is also, of the four
classes, the one a farmer most needs an early warning for. A comparison that improves macro-F1 by
raising Healthy's recall would be worthless here, so per-class F1 is the reported quantity and
macro-F1 is secondary.

The design
----------
`convnext_tiny` at 5e-5 -- Experiment 04's best backbone at its tuned rate -- across the **same five
specimen folds**, with inverse-frequency class weights in the cross-entropy loss. Everything else is
identical to Experiment 04, so the comparison is against runs that already exist and no unweighted
control needs retraining. The folds are rebuilt from seed 2025, which is safe only because
`test_folds_are_reproducible_from_the_seed` asserts they come back identical, and because
Experiment 06 demonstrated this pipeline reproduces its own scores to four decimal places.

Output goes to `experiments/runs/exp09/`. Experiment 04's artefacts are not touched.

The pre-registered null, written before the run
------------------------------------------------
Master Plan Section 6, Stage 10: *"class weighting may not help Black Gill ... `D-016` governs -- a
negative result is the finding, and nothing is tuned until it looks positive."*

Concretely: **if Black Gill F1 does not improve by more than the fold-to-fold SD, the finding is
that class weighting does not rescue the weakest class on this corpus**, limitation C5 is restated
rather than retired, and no further weighting scheme is tried in search of a better number. The SD
is computed from the same five folds, so the threshold is a property of the data rather than a
choice made afterwards.

What weighting can and cannot do
---------------------------------
Inverse-frequency weighting changes what the loss *cares* about; it does not add information. If
Black Gill is weak because 75 specimens are too few to characterise it, or because it is genuinely
hard to distinguish from BG_WSSV in a photograph, weighting will move predictions toward the class
without making them more correct -- raising its recall while lowering its precision, leaving F1
roughly where it was. The per-class precision and recall are therefore both reported, because they
are what separates "the model learned the class" from "the model was pushed toward guessing it".

Outputs
-------
    reports/tables/exp09_folds.csv      one row per fold, weighted
    reports/tables/exp09_comparison.csv weighted vs Experiment 04's unweighted, per class
    experiments/runs/exp09/*.json       history, per-class metrics, confusion, environment
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
from shrimp_risk.vision.models import create_model, get_device  # noqa: E402
from shrimp_risk.vision.trainer import TrainConfig, train_and_evaluate  # noqa: E402

SEED = 2025
N_FOLDS = 5
BACKBONE = "convnext_tiny"
LR = 5e-5              # the tuned rate from `D-020`; changing it would confound the ablation
TARGET_CLASS = "BG"    # Black Gill -- the class this experiment exists for


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--smoke", action="store_true", help="2 folds, 2 epochs, first fold only")
    args = ap.parse_args()
    smoke_one_fold = args.smoke
    if args.smoke:
        args.folds, args.epochs = 2, 2

    device = get_device()
    print("device     :", device)
    print("environment:", environment_fingerprint())

    manifest = S.load_raw_manifest(resolve_image_manifest(256))
    folds = S.make_specimen_folds(manifest, n_splits=args.folds, seed=SEED)
    for f in folds:
        leak = S.leakage_report(manifest, f)
        assert leak["test_images_with_train_sibling"] == 0, (
            f"fold {f.fold} leaks -- grouping is broken, refusing to run")
    if smoke_one_fold:
        folds = folds[:1]

    print(f"manifest   : {len(manifest):,} images / {manifest.specimen_id.nunique()} specimens")
    print(f"folds      : {len(folds)}, all verified leak-free")

    # Printed before training so the weights are visible as a choice rather than a black box.
    example_weights = class_weights(manifest, folds[0].train)
    print(f"weights    : " + "  ".join(
        f"{c}={w:.3f}" for c, w in zip(CLASSES, example_weights.tolist())))
    print(f"             (fold 0; inverse frequency, so {TARGET_CLASS} is up-weighted)\n")

    out_dir = RUNS / "exp09"
    out_dir.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []

    for fold in folds:
        tag = f"{BACKBONE}_weighted__fold{fold.fold}"
        print("=" * 78)
        print(f"{tag}   class-weighted cross-entropy")
        print("=" * 78)

        seed_everything(SEED)
        cfg = TrainConfig(
            model_name=BACKBONE, seed=SEED, protocol="specimen_level_kfold",
            fold=fold.fold, augment=True, epochs=args.epochs, patience=7,
            batch_size=32, lr=LR, class_weighted_loss=True,
        )
        loaders = make_loaders(
            manifest, fold, PROJECT_ROOT, batch_size=cfg.batch_size,
            img_size=cfg.img_size, augment=True, num_workers=4,
        )
        model = create_model(BACKBONE, num_classes=len(CLASSES))
        weights = class_weights(manifest, fold.train)
        res = train_and_evaluate(model, loaders, cfg, device, CLASSES, loss_weights=weights)

        rows.append({
            "backbone": BACKBONE,
            "fold": fold.fold,
            "lr": LR,
            "class_weighted": True,
            "best_epoch": res.best_epoch,
            "test_acc": round(res.test_acc, 4),
            "test_f1_macro": round(res.test_f1_macro, 4),
            **{f"f1_{c}": round(res.per_class[c]["f1-score"], 4) for c in CLASSES},
            **{f"recall_{c}": round(res.per_class[c]["recall"], 4) for c in CLASSES},
            **{f"precision_{c}": round(res.per_class[c]["precision"], 4) for c in CLASSES},
            "train_seconds": round(res.train_seconds, 1),
            "run_artifact": f"experiments/runs/exp09/{tag}.json",
        })

        (out_dir / f"{tag}.json").write_text(json.dumps({
            "tag": tag,
            "config": asdict(res.config),
            "history": [asdict(h) for h in res.history],
            "per_class": res.per_class,
            "confusion": res.confusion,
            "class_weights": weights.tolist(),
            "environment": environment_fingerprint(),
        }, indent=2), encoding="utf-8")
        np.savez_compressed(
            out_dir / f"{tag}_predictions.npz",
            probs=res.test_probs, labels=res.test_labels,
            specimen_ids=manifest.iloc[fold.test]["specimen_id"].to_numpy(),
        )
        pd.DataFrame(rows).to_csv(TABLES / "exp09_folds.csv", index=False)
        print()

    weighted = pd.DataFrame(rows)

    # ---- comparison against Experiment 04's unweighted runs ----
    exp04_path = TABLES / "exp04_folds.csv"
    if not exp04_path.exists():
        print("exp04_folds.csv absent -- the unweighted control cannot be compared")
        return 0
    unweighted = pd.read_csv(exp04_path)
    unweighted = unweighted[unweighted.backbone == BACKBONE]

    comparison = []
    for metric in ["f1_macro"] + [f"f1_{c}" for c in CLASSES] + \
                  [f"recall_{c}" for c in CLASSES]:
        col_w = "test_f1_macro" if metric == "f1_macro" else metric
        col_u = "test_f1_macro" if metric == "f1_macro" else metric
        if col_u not in unweighted.columns:
            continue
        w_mean, w_sd = weighted[col_w].mean(), weighted[col_w].std(ddof=1)
        u_mean, u_sd = unweighted[col_u].mean(), unweighted[col_u].std(ddof=1)
        delta = w_mean - u_mean
        comparison.append({
            "metric": metric,
            "unweighted_mean": round(float(u_mean), 4),
            "unweighted_sd": round(float(u_sd), 4),
            "weighted_mean": round(float(w_mean), 4),
            "weighted_sd": round(float(w_sd), 4),
            "delta": round(float(delta), 4),
            # The pre-registered rule, evaluated in the table. The SD is the fold-to-fold spread
            # of the UNWEIGHTED runs -- the variation that already exists without any change --
            # so "beyond noise" means "larger than the variation the folds produce on their own".
            "beyond_fold_sd": bool(abs(delta) > u_sd),
        })
    comp = pd.DataFrame(comparison)
    comp.to_csv(TABLES / "exp09_comparison.csv", index=False)

    print("=" * 78)
    print("EXPERIMENT 09 -- class-weighted loss vs Experiment 04's unweighted runs")
    print("=" * 78)
    print(comp.to_string(index=False))

    bg = comp[comp.metric == f"f1_{TARGET_CLASS}"]
    if len(bg):
        r = bg.iloc[0]
        rescued = bool(r["beyond_fold_sd"]) and r["delta"] > 0
        print()
        print(f"  {TARGET_CLASS} F1: {r['unweighted_mean']:.4f} -> {r['weighted_mean']:.4f}  "
              f"(delta {r['delta']:+.4f}, fold SD {r['unweighted_sd']:.4f})")
        print(f"  VERDICT: {'IMPROVED beyond the fold SD' if rescued else 'NOT improved beyond the fold SD -- limitation C5 stands'}")
        print()
        print("  Read precision and recall together before concluding anything. Weighting pushes")
        print("  predictions toward the up-weighted class; recall rising while precision falls is")
        print("  the model guessing more often, not learning the class better.")

    print(f"\ntables -> {TABLES}\nruns   -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
