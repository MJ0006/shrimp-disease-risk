"""Experiment 10 (LR Sweep) -- Per-backbone learning rate sweep for untuned architectures.

Objective
---------
In Experiment 04, ConvNeXt Tiny failed to fit its training data at lr = 3e-4 and was tuned to
lr = 5e-5 (D-020), which lifted its macro-F1 from 0.3966 to 0.7189 (first place). The other four
backbones (rexnet_150, efficientnet_b0, mobilenetv3_large_100, resnet50) were left at the default
rate of 3e-4. As acknowledged in D-020 and limitation 7, this left the comparison defensible
("each at a rate it can train at") but incomplete.

This experiment answers two questions:
  Q1: Does a learning rate sweep over [1e-4, 3e-4, 1e-3] improve performance for any of the four
      untuned backbones?
  Q2: Does any backbone overtake convnext_tiny (macro-F1 0.7189 +/- 0.0453) when tuned?

Design
------
  * 4 backbones: rexnet_150, efficientnet_b0, mobilenetv3_large_100, resnet50
  * 3 candidate learning rates: [1e-4, 3e-4, 1e-3]
  * 5 folds of StratifiedGroupKFold on specimen_id (identical folds, seed 2025)
  * Total: 4 x 3 x 5 = 60 fold evaluations.
  * For lr = 3e-4, the runs are identical to Experiment 04; existing artifacts from
    experiments/runs/exp04/ are imported to avoid redundant compute unless --force is given.
  * Outputs are saved after every single fold to prevent data loss.

Outputs
-------
  experiments/runs/exp10_lr_sweep/{backbone}_lr{lr}__fold{fold}.json
  experiments/runs/exp10_lr_sweep/{backbone}_lr{lr}__fold{fold}_predictions.npz
  reports/tables/exp10_lr_sweep_folds.csv
  reports/tables/exp10_lr_sweep_summary.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import functools
import numpy as np
import pandas as pd

# Flush prints immediately for live logging
print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shrimp_risk.common.paths import (  # noqa: E402
    PROJECT_ROOT, RUNS, TABLES, resolve_image_manifest,
)
from shrimp_risk.common.seeding import environment_fingerprint, seed_everything  # noqa: E402
from shrimp_risk.vision import splits as S  # noqa: E402
from shrimp_risk.vision.datasets import CLASSES, make_loaders  # noqa: E402
from shrimp_risk.vision.models import create_model, get_device  # noqa: E402
from shrimp_risk.vision.trainer import TrainConfig, train_and_evaluate  # noqa: E402

SEED = 2025
N_FOLDS = 5

TARGET_BACKBONES = [
    "rexnet_150",
    "efficientnet_b0",
    "mobilenetv3_large_100",
    "resnet50",
]

CANDIDATE_LRS = [1e-4, 3e-4, 1e-3]


def summarise_lr_sweep(folds_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise macro-F1, accuracy, and BG performance across folds for each (backbone, lr)."""
    if folds_df.empty:
        return pd.DataFrame()
    summary = (
        folds_df.groupby(["backbone", "lr"])
        .agg(
            folds=("fold", "count"),
            f1_macro_mean=("test_f1_macro", "mean"),
            f1_macro_sd=("test_f1_macro", "std"),
            acc_mean=("test_acc", "mean"),
            acc_sd=("test_acc", "std"),
            f1_BG_mean=("f1_BG", "mean"),
            train_seconds_mean=("train_seconds", "mean"),
        )
        .round(4)
        .reset_index()
    )
    return summary.sort_values(["backbone", "f1_macro_mean"], ascending=[True, False])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--backbones", nargs="*", default=TARGET_BACKBONES)
    ap.add_argument("--lrs", nargs="*", type=float, default=CANDIDATE_LRS)
    ap.add_argument("--force", action="store_true", help="retrain even if artifact exists")
    ap.add_argument("--summary-only", action="store_true", help="rebuild summary table only")
    ap.add_argument("--smoke", action="store_true", help="1 fold, 2 epochs, smoke test only")
    args = ap.parse_args()

    out_dir = RUNS / "exp10_lr_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    folds_csv = TABLES / "exp10_lr_sweep_folds.csv"
    summary_csv = TABLES / "exp10_lr_sweep_summary.csv"

    if args.summary_only:
        if folds_csv.exists():
            df = pd.read_csv(folds_csv)
            summary = summarise_lr_sweep(df)
            summary.to_csv(summary_csv, index=False)
            print("Rebuilt summary table:")
            print(summary.to_string(index=False))
        else:
            print(f"Error: {folds_csv} does not exist.")
            return 1
        return 0

    if args.smoke:
        args.folds = 2
        args.epochs = 2

    device = get_device()
    print("device     :", device)
    print("environment:", environment_fingerprint())

    manifest_path = resolve_image_manifest(256)
    manifest = S.load_raw_manifest(manifest_path)
    print(f"manifest   : {manifest_path.name}")
    print(f"             {len(manifest):,} images / {manifest.specimen_id.nunique()} specimens")

    folds = S.make_specimen_folds(manifest, n_splits=args.folds, seed=SEED)
    if args.smoke:
        folds = folds[:1]

    for f in folds:
        leak = S.leakage_report(manifest, f)
        assert leak["test_images_with_train_sibling"] == 0, (
            f"fold {f.fold} leaks {leak['test_images_with_train_sibling']} images -- "
            "grouping is broken, refusing to run"
        )
    print(f"folds      : {len(folds)} verified leak-free\n")

    # Load existing progress if available
    existing_rows = []
    if folds_csv.exists() and not args.force:
        existing_df = pd.read_csv(folds_csv)
        existing_rows = existing_df.to_dict("records")
        print(f"Loaded {len(existing_rows)} existing fold results from {folds_csv.name}")

    existing_keys = {
        (r["backbone"], float(r["lr"]), int(r["fold"])) for r in existing_rows
    }

    rows = list(existing_rows)

    total_tasks = len(args.backbones) * len(args.lrs) * len(folds)
    current_task = 0

    for name in args.backbones:
        for lr in args.lrs:
            for fold in folds:
                current_task += 1
                key = (name, float(lr), int(fold.fold))
                tag = f"{name}_lr{lr}__fold{fold.fold}"
                run_json = out_dir / f"{tag}.json"
                run_npz = out_dir / f"{tag}_predictions.npz"

                if not args.force and key in existing_keys and run_json.exists():
                    print(f"[{current_task}/{total_tasks}] SKIP {tag} (already completed)")
                    continue

                # Check if this is lr=3e-4 and can be imported from exp04
                exp04_json = RUNS / "exp04" / f"{name}__fold{fold.fold}.json"
                exp04_npz = RUNS / "exp04" / f"{name}__fold{fold.fold}_predictions.npz"
                if not args.force and abs(lr - 3e-4) < 1e-7 and exp04_json.exists():
                    print(f"[{current_task}/{total_tasks}] IMPORT {tag} from exp04")
                    # Copy artifacts
                    shutil.copy2(exp04_json, run_json)
                    if exp04_npz.exists():
                        shutil.copy2(exp04_npz, run_npz)

                    # Extract record from exp04_folds.csv or json
                    data = json.loads(run_json.read_text(encoding="utf-8"))
                    per_class = data["per_class"]
                    history = data.get("history", [])
                    best_val_f1 = max((h.get("val_f1", h.get("val_f1_macro", 0.0)) for h in history), default=0.0)
                    best_epoch = data.get("config", {}).get("best_epoch", 0)
                    if not best_epoch and history:
                        best_epoch = max(range(len(history)), key=lambda i: history[i].get("val_f1", 0.0)) + 1
                    test_f1 = per_class.get("macro avg", {}).get("f1-score", 0.0)
                    test_f1_w = per_class.get("weighted avg", {}).get("f1-score", 0.0)
                    test_acc = data.get("accuracy", per_class.get("accuracy", 0.0))
                    train_secs = sum(h.get("seconds", h.get("epoch_seconds", 0.0)) for h in history)

                    record = {
                        "backbone": name,
                        "fold": fold.fold,
                        "lr": lr,
                        "n_train_images": len(fold.train),
                        "n_test_images": len(fold.test),
                        "n_test_specimens": int(pd.Series(
                            manifest.iloc[fold.test]["specimen_id"]).nunique()),
                        "best_epoch": best_epoch,
                        "best_val_f1": round(best_val_f1, 4),
                        "test_acc": round(test_acc, 4),
                        "test_f1_macro": round(test_f1, 4),
                        "test_f1_weighted": round(test_f1_w, 4),
                        "train_seconds": round(train_secs, 1),
                        **{f"f1_{c}": round(per_class[c]["f1-score"], 4) for c in CLASSES if c in per_class},
                        **{f"recall_{c}": round(per_class[c]["recall"], 4) for c in CLASSES if c in per_class},
                        "run_artifact": f"experiments/runs/exp10_lr_sweep/{tag}.json",
                    }
                    rows.append(record)
                    existing_keys.add(key)
                    pd.DataFrame(rows).to_csv(folds_csv, index=False)
                    summarise_lr_sweep(pd.DataFrame(rows)).to_csv(summary_csv, index=False)
                    continue

                # Otherwise train fresh
                print("=" * 74)
                print(f"[{current_task}/{total_tasks}] TRAIN {tag} (specimen-level, lr={lr})")
                print("=" * 74)

                seed_everything(SEED)
                cfg = TrainConfig(
                    model_name=name,
                    seed=SEED,
                    protocol="specimen_level_kfold",
                    fold=fold.fold,
                    augment=True,
                    epochs=args.epochs,
                    patience=7,
                    batch_size=32,
                    lr=lr,
                    class_weighted_loss=False,
                )
                loaders = make_loaders(
                    manifest,
                    fold,
                    PROJECT_ROOT,
                    batch_size=cfg.batch_size,
                    img_size=cfg.img_size,
                    augment=True,
                    num_workers=2,  # 2 workers safe against WinError 1455
                )
                model = create_model(name, num_classes=len(CLASSES))
                res = train_and_evaluate(model, loaders, cfg, device, CLASSES)

                rec = {
                    "backbone": name,
                    "fold": fold.fold,
                    "lr": lr,
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
                    "run_artifact": f"experiments/runs/exp10_lr_sweep/{tag}.json",
                }
                rows.append(rec)
                existing_keys.add(key)

                # Save run JSON and prediction NPZ
                run_json.write_text(json.dumps({
                    "tag": tag,
                    "backbone": name,
                    "lr": lr,
                    "fold": fold.fold,
                    "config": asdict(res.config),
                    "history": [asdict(h) for h in res.history],
                    "per_class": res.per_class,
                    "confusion": res.confusion,
                    "environment": environment_fingerprint(),
                }, indent=2), encoding="utf-8")

                np.savez_compressed(
                    run_npz,
                    probs=res.test_probs,
                    labels=res.test_labels,
                    specimen_ids=manifest.iloc[fold.test]["specimen_id"].to_numpy(),
                )

                # Persist tables incrementally after every fold
                df_current = pd.DataFrame(rows)
                df_current.to_csv(folds_csv, index=False)
                summary_df = summarise_lr_sweep(df_current)
                summary_df.to_csv(summary_csv, index=False)
                print(f"Completed {tag}: test macro-F1 = {res.test_f1_macro:.4f} (best val F1 = {res.best_val_f1:.4f})\n")

    final_df = pd.DataFrame(rows)
    final_summary = summarise_lr_sweep(final_df)
    final_summary.to_csv(summary_csv, index=False)

    print("=" * 74)
    print("STAGE 10 LR SWEEP COMPLETED")
    print("=" * 74)
    print(final_summary.to_string(index=False))
    print(f"\nTables written to:\n  {folds_csv}\n  {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
