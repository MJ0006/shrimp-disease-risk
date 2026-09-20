"""Experiment 02 -- Does specimen leakage actually inflate reported accuracy?

Objective
---------
Experiment 01 established that image-level splitting puts a sibling photograph of ~88% of test
specimens into training. That is exposure, not yet harm. This experiment measures the harm:
train identical models under both protocols and compare test macro-F1.

Hypothesis
----------
H1: image-level splitting yields materially higher test macro-F1 than specimen-level
    splitting, with all other factors held constant. The difference is the optimism
    attributable to leakage.
H0: the difference is within seed-to-seed noise, meaning leakage exposure does not translate
    into inflated scores on this dataset.

Both outcomes are reportable. If H0 holds, that is the finding and the project says so.

Controlled design
-----------------
Three conditions. Conditions B and C differ in EXACTLY ONE factor -- the split protocol -- so
any difference between them is attributable to leakage rather than to augmentation, optimiser,
architecture or class balance.

    A  image_level_noaug  image-level    no augmentation   -- isolates the effect of augmentation
    B  image_level        image-level    augmentation      -- leaky protocol, modern training
    C  specimen_level     specimen-level augmentation      -- honest protocol, modern training

Condition A was originally named `A_notebook_repro`, on the assumption that dropping
augmentation from an image-level split reproduces the inherited notebook. It does not: A scores
0.74-0.79 against the notebook's 0.638, because it also differs in split fractions, optimiser,
schedule, F1 averaging, patience and input size. Calling it a reproduction would be exactly the
kind of unchecked claim the Stage 0 audit exists to catch, so it was renamed (`D-010`). Run
artefacts written before the rename keep their original filenames; the `condition` column in
the derived tables carries the corrected name, and `run_artifact` records where each row
came from.

Every condition uses the same backbone, learning rate, schedule, batch size, epoch budget and
patience. Each is repeated across several seeds because a single run on ~62 test specimens is
too noisy to support a claim.

Outputs
-------
    reports/tables/exp02_runs.csv        one row per run
    reports/tables/exp02_summary.csv     mean +/- SD per condition
    experiments/runs/exp02/*.json        full history, per-class metrics, confusion matrices
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shrimp_risk.common.paths import (  # noqa: E402
    PROJECT_ROOT, RUNS, TABLES, resolve_image_manifest,
)
from shrimp_risk.common.seeding import environment_fingerprint, seed_everything  # noqa: E402
from shrimp_risk.vision import splits as S  # noqa: E402
from shrimp_risk.vision.datasets import CLASSES, class_weights, make_loaders  # noqa: E402
from shrimp_risk.vision.models import create_model, get_device  # noqa: E402
from shrimp_risk.vision.trainer import TrainConfig, train_and_evaluate  # noqa: E402

SEEDS = (2025, 7, 42)
BACKBONE = "rexnet_150"  # the inherited notebook's architecture, held constant across conditions

CONDITIONS = {
    "A_image_level_noaug": dict(protocol="image_level", augment=False),
    "B_image_level": dict(protocol="image_level", augment=True),
    "C_specimen_level": dict(protocol="specimen_level", augment=True),
}

SPLITTERS = {
    "image_level": S.make_image_level_splits,
    "specimen_level": S.make_specimen_level_splits,
}


def main() -> int:
    device = get_device()
    print("device     :", device)
    print("environment:", environment_fingerprint())

    manifest_path = resolve_image_manifest(256)
    manifest = S.load_raw_manifest(manifest_path)
    print(f"manifest   : {manifest_path.name}")
    print(f"             {len(manifest):,} images / "
          f"{manifest.specimen_id.nunique()} specimens\n")

    out_dir = RUNS / "exp02"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for cond_name, cond in CONDITIONS.items():
        for seed in SEEDS:
            tag = f"{cond_name}__seed{seed}"
            print("=" * 74)
            print(f"{tag}   protocol={cond['protocol']}  augment={cond['augment']}")
            print("=" * 74)

            seed_everything(seed)
            split = SPLITTERS[cond["protocol"]](manifest, seed=seed)
            leak = S.leakage_report(manifest, split)

            cfg = TrainConfig(
                model_name=BACKBONE,
                seed=seed,
                protocol=cond["protocol"],
                augment=cond["augment"],
                epochs=30,
                patience=7,
                batch_size=32,
            )

            loaders = make_loaders(
                manifest, split, PROJECT_ROOT,
                batch_size=cfg.batch_size, img_size=cfg.img_size,
                augment=cfg.augment, num_workers=4,
            )
            model = create_model(cfg.model_name, num_classes=len(CLASSES))
            weights = class_weights(manifest, split.train) if cfg.class_weighted_loss else None

            res = train_and_evaluate(
                model, loaders, cfg, device, CLASSES, loss_weights=weights
            )

            rows.append({
                "condition": cond_name,
                "protocol": cond["protocol"],
                "augment": cond["augment"],
                "seed": seed,
                "backbone": BACKBONE,
                "pct_test_leaked": leak["pct_test_leaked"],
                "n_train_images": leak["n_train_images"],
                "n_test_images": leak["n_test_images"],
                "n_test_specimens": leak["n_test_specimens"],
                "best_epoch": res.best_epoch,
                "best_val_f1": round(res.best_val_f1, 4),
                "test_acc": round(res.test_acc, 4),
                "test_f1_macro": round(res.test_f1_macro, 4),
                "test_f1_weighted": round(res.test_f1_weighted, 4),
                "train_seconds": round(res.train_seconds, 1),
                **{f"f1_{c}": round(res.per_class[c]["f1-score"], 4) for c in CLASSES},
                # Every table row names the artefact it came from, so any number on a slide can
                # be traced back to the run that produced it without matching on filenames.
                "run_artifact": f"experiments/runs/exp02/{tag}.json",
            })

            payload = {
                "tag": tag,
                "config": asdict(res.config),
                "leakage": leak,
                "history": [asdict(h) for h in res.history],
                "per_class": res.per_class,
                "confusion": res.confusion,
                "environment": environment_fingerprint(),
            }
            (out_dir / f"{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            np.savez_compressed(
                out_dir / f"{tag}_predictions.npz",
                probs=res.test_probs, labels=res.test_labels,
                specimen_ids=manifest.iloc[split.test]["specimen_id"].to_numpy(),
            )
            print()

    df = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "exp02_runs.csv", index=False)

    summary = (
        df.groupby("condition")
        .agg(
            n=("seed", "count"),
            pct_test_leaked=("pct_test_leaked", "mean"),
            test_f1_macro_mean=("test_f1_macro", "mean"),
            test_f1_macro_sd=("test_f1_macro", "std"),
            test_acc_mean=("test_acc", "mean"),
            test_acc_sd=("test_acc", "std"),
            best_val_f1_mean=("best_val_f1", "mean"),
            train_seconds=("train_seconds", "mean"),
        )
        .round(4)
        .reset_index()
    )
    summary.to_csv(TABLES / "exp02_summary.csv", index=False)

    print("=" * 74)
    print("EXPERIMENT 02 SUMMARY")
    print("=" * 74)
    print(summary.to_string(index=False))

    if {"B_image_level", "C_specimen_level"}.issubset(set(summary.condition)):
        b = summary.loc[summary.condition == "B_image_level", "test_f1_macro_mean"].iloc[0]
        c = summary.loc[summary.condition == "C_specimen_level", "test_f1_macro_mean"].iloc[0]
        sd = summary.loc[summary.condition == "C_specimen_level", "test_f1_macro_sd"].iloc[0]
        print(f"\nLeakage-attributable optimism (B - C): {b - c:+.4f} macro-F1")
        print(f"  image-level    : {b:.4f}")
        print(f"  specimen-level : {c:.4f}   (SD across seeds {sd:.4f})")
        print("\nNOTE: interpret against seed-to-seed SD. A gap smaller than the SD does not")
        print("      support the inflation claim and must be reported as inconclusive.")

    print(f"\ntables -> {TABLES}\nruns   -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
