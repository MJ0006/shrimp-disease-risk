"""Experiment 02b -- the faithful reproduction of the inherited notebook.

Why this exists, and why it took until Review 3
------------------------------------------------
The inherited notebook reports test macro-F1 **0.638**. Experiment 02's condition A was originally
called `A_notebook_repro` on the assumption that an image-level split without augmentation
reproduces it. It scored 0.74-0.79, and `D-010` established why: condition A matches the notebook in
two factors and differs in at least six others. It was renamed `A_image_level_noaug`, and the
project has since made **no claim whatsoever** about reproducing 0.638.

This script is the run that decides whether such a claim is available. It is the only run in this
project that may ever be described as reproducing the notebook.

The six factors, and what each is set to here
----------------------------------------------
========================  =========================  ==============================
Factor                    This project's default     Set here, to match the notebook
========================  =========================  ==============================
split fractions           70 / 15 / 15               **90 / 5 / 5** (57 test images)
stratification            stratified                 **unstratified** (`random_split`)
optimiser                 AdamW + cosine annealing   **plain Adam, no schedule**
model selection           macro-F1                   **micro-F1**, i.e. accuracy
patience / budget         7 of 30 epochs             **3 of 50 epochs**
augmentation              on                         **off**
========================  =========================  ==============================

Backbone, learning rate and batch size come from the notebook too: `rexnet_150`, 3e-4, batch 16.

The one departure that remains, stated rather than hidden
----------------------------------------------------------
Inputs are decoded from the 256px cache rather than from the 2048px originals, so the resampling
path is 2048 -> 256 -> 224 instead of 2048 -> 224. `--full-res` runs from the originals at roughly
four times the cost. The cache is the default because if this run lands near 0.638 through it, the
decode path was demonstrably not what produced the notebook's number; and if it does not, the flag
is the first thing to try. Whichever is used is recorded in every row.

What each outcome means -- decided before the run
--------------------------------------------------
* **Lands near 0.638 across seeds.** A reproduction claim becomes available, `D-010`'s reversal
  condition is met, and condition A can finally be described in relation to a real baseline.
* **Lands well above 0.638.** The gap is not explained by these six factors, and *that discrepancy
  becomes the finding*. It would mean something about the notebook's result is not recoverable from
  its stated configuration -- which is a stronger criticism of it than any this project currently
  makes, and must be reported as an open question rather than as a defect proven.
* **Lands well below.** Same treatment, opposite direction.

Nothing here is tuned toward 0.638. The configuration was read off `D-010` and the notebook, fixed
before the first run, and is not adjusted afterwards. `D-016` governs: the number that comes out is
the finding.

Outputs
-------
    reports/tables/exp02b_runs.csv       one row per seed
    reports/tables/exp02b_summary.csv    mean +/- SD, and the comparison against 0.638
    experiments/runs/exp02b/*.json       history, per-class metrics, confusion, environment
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
    IMAGE_MANIFEST, PROJECT_ROOT, RUNS, TABLES, resolve_image_manifest,
)
from shrimp_risk.common.seeding import environment_fingerprint, seed_everything  # noqa: E402
from shrimp_risk.vision import splits as S  # noqa: E402
from shrimp_risk.vision.datasets import CLASSES, make_loaders  # noqa: E402
from shrimp_risk.vision.models import create_model, get_device  # noqa: E402
from shrimp_risk.vision.trainer import TrainConfig, train_and_evaluate  # noqa: E402

SEEDS = (2025, 7, 42)

# The number this run exists to test itself against. Stated as a constant so the comparison in the
# output is made against the figure that was written down, not one recalled afterwards.
NOTEBOOK_F1 = 0.638

# "Near" is defined before the run rather than after seeing the result. 0.05 macro-F1 is roughly
# the fold-to-fold SD this project sees on partitions of this size, so a gap inside it is not
# distinguishable from the draw, and a gap outside it is.
NEAR_TOLERANCE = 0.05


def _merge_csv(path: Path, fresh: pd.DataFrame, source: str) -> None:
    """Write `fresh`, keeping any rows this run did not produce.

    The cache-256 and full-resolution conditions are two results of one experiment, not two
    attempts at one result: the second exists to close the last stated departure from the
    notebook, and what it shows is that closing it makes the gap *wider*. Overwriting would
    delete half the finding, and re-running one mode months later would silently delete the other.

    Rows are replaced by `image_source`, so re-running a mode refreshes exactly its own rows.
    """
    if path.exists():
        old = pd.read_csv(path)
        if "image_source" in old.columns:
            keep = old[old["image_source"] != source]
            fresh = pd.concat([keep, fresh], ignore_index=True)
    fresh.to_csv(path, index=False)


def verdict(mean_f1: float) -> str:
    gap = mean_f1 - NOTEBOOK_F1
    if abs(gap) <= NEAR_TOLERANCE:
        return (f"REPRODUCED within {NEAR_TOLERANCE} (gap {gap:+.4f}) -- "
                "D-010's reversal condition is met")
    direction = "ABOVE" if gap > 0 else "BELOW"
    return (f"NOT REPRODUCED: {direction} 0.638 by {abs(gap):.4f} -- "
            "the discrepancy is the finding, not a defect proven")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--full-res", action="store_true",
                    help="decode from the 2048px originals instead of the 256px cache "
                         "(~4x slower; removes the last departure from the notebook)")
    ap.add_argument("--smoke", action="store_true", help="one seed, 3 epochs")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs = [2025], 3

    device = get_device()
    print("device     :", device)
    print("environment:", environment_fingerprint())

    manifest_path = IMAGE_MANIFEST if args.full_res else resolve_image_manifest(256)
    manifest = S.load_raw_manifest(manifest_path)
    source = "full_resolution" if args.full_res else "cache_256"
    print(f"manifest   : {manifest_path.name}  ({source})")
    print(f"             {len(manifest):,} images / {manifest.specimen_id.nunique()} specimens")
    print(f"target     : notebook macro-F1 {NOTEBOOK_F1}, "
          f"'near' defined as +/- {NEAR_TOLERANCE}\n")

    out_dir = RUNS / "exp02b"
    out_dir.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in args.seeds:
        # The image source is IN THE TAG, not only in the JSON. The first version of this script
        # named both conditions `notebook_faithful__seed{seed}`, so running `--full-res` silently
        # overwrote the cache-256 run's history and posteriors -- the same way `D-020` lost a set
        # of per-epoch histories, in the same project, three weeks later. Two conditions are two
        # results; they do not share a filename.
        tag = f"notebook_faithful_{source}__seed{seed}"
        print("=" * 78)
        print(f"{tag}   90/5/5 unstratified, plain Adam, micro-F1 selection, patience 3")
        print("=" * 78)

        seed_everything(seed)
        split = S.make_image_level_splits(
            manifest, seed=seed, val_frac=0.05, test_frac=0.05, stratify=False)
        leak = S.leakage_report(manifest, split)

        # The notebook's validation set is 57 images. Printing the per-class counts makes the
        # instability visible before the score does: model selection over 50 epochs on a set this
        # small is the failure mode `D-010` identified, and a class with two members in it is how
        # that failure happens.
        val_counts = manifest.iloc[split.val]["class"].value_counts().to_dict()
        test_counts = manifest.iloc[split.test]["class"].value_counts().to_dict()
        print(f"  split      : {len(split.train)} train / {len(split.val)} val / "
              f"{len(split.test)} test images")
        print(f"  val classes: {val_counts}")
        print(f"  test classes: {test_counts}")
        print(f"  leakage    : {leak['pct_test_leaked']}% of test images have a train sibling")

        cfg = TrainConfig(
            model_name="rexnet_150",
            seed=seed,
            protocol="image_level_unstratified",
            augment=False,
            epochs=args.epochs,
            patience=3,
            batch_size=16,
            lr=3e-4,
            optimizer="adam",
            scheduler="none",
            selection_metric="micro_f1",
        )
        loaders = make_loaders(
            manifest, split, PROJECT_ROOT, batch_size=cfg.batch_size,
            img_size=cfg.img_size, augment=False, num_workers=4,
        )
        model = create_model(cfg.model_name, num_classes=len(CLASSES))
        res = train_and_evaluate(model, loaders, cfg, device, CLASSES)

        rows.append({
            "seed": seed,
            "image_source": source,
            "n_train_images": len(split.train),
            "n_val_images": len(split.val),
            "n_test_images": len(split.test),
            "n_test_specimens": leak["n_test_specimens"],
            "pct_test_leaked": leak["pct_test_leaked"],
            "best_epoch": res.best_epoch,
            "epochs_run": len(res.history),
            "best_val_score": round(res.best_val_f1, 4),
            "test_acc": round(res.test_acc, 4),
            "test_f1_macro": round(res.test_f1_macro, 4),
            "test_f1_weighted": round(res.test_f1_weighted, 4),
            "notebook_f1": NOTEBOOK_F1,
            "gap_vs_notebook": round(res.test_f1_macro - NOTEBOOK_F1, 4),
            "train_seconds": round(res.train_seconds, 1),
            **{f"f1_{c}": round(res.per_class[c]["f1-score"], 4) for c in CLASSES},
            "n_val_classes_present": len(val_counts),
            "smallest_val_class_n": min(val_counts.values()) if val_counts else 0,
            "run_artifact": f"experiments/runs/exp02b/{tag}.json",
        })

        (out_dir / f"{tag}.json").write_text(json.dumps({
            "tag": tag,
            "config": asdict(res.config),
            "history": [asdict(h) for h in res.history],
            "per_class": res.per_class,
            "confusion": res.confusion,
            "split_counts": {"val": val_counts, "test": test_counts},
            "image_source": source,
            "environment": environment_fingerprint(),
        }, indent=2), encoding="utf-8")
        np.savez_compressed(
            out_dir / f"{tag}_predictions.npz",
            probs=res.test_probs, labels=res.test_labels,
            specimen_ids=manifest.iloc[split.test]["specimen_id"].to_numpy(),
        )
        _merge_csv(TABLES / "exp02b_runs.csv", pd.DataFrame(rows), source)
        print()

    df = pd.DataFrame(rows)
    mean_f1 = float(df["test_f1_macro"].mean())
    summary = pd.DataFrame([{
        "condition": "notebook_faithful",
        "seeds": len(df),
        "image_source": source,
        "f1_macro_mean": round(mean_f1, 4),
        "f1_macro_sd": round(float(df["test_f1_macro"].std(ddof=1)), 4) if len(df) > 1 else None,
        "f1_macro_min": round(float(df["test_f1_macro"].min()), 4),
        "f1_macro_max": round(float(df["test_f1_macro"].max()), 4),
        "acc_mean": round(float(df["test_acc"].mean()), 4),
        "notebook_f1": NOTEBOOK_F1,
        "gap_vs_notebook": round(mean_f1 - NOTEBOOK_F1, 4),
        "tolerance": NEAR_TOLERANCE,
        "verdict": verdict(mean_f1),
        "mean_best_epoch": round(float(df["best_epoch"].mean()), 1),
        "mean_epochs_run": round(float(df["epochs_run"].mean()), 1),
        "pct_test_leaked_mean": round(float(df["pct_test_leaked"].mean()), 2),
    }])
    _merge_csv(TABLES / "exp02b_summary.csv", summary, source)

    print("=" * 78)
    print("EXPERIMENT 02b -- faithful notebook reproduction")
    print("=" * 78)
    print(df[["seed", "n_test_images", "best_epoch", "epochs_run", "test_acc",
              "test_f1_macro", "gap_vs_notebook", "smallest_val_class_n"]].to_string(index=False))
    print()
    print(f"  mean macro-F1 {mean_f1:.4f}   notebook {NOTEBOOK_F1}   "
          f"gap {mean_f1 - NOTEBOOK_F1:+.4f}")
    print(f"  {verdict(mean_f1)}")
    print()
    print("Read the `best_epoch` and `smallest_val_class_n` columns before the score. `D-010`'s")
    print("diagnosis is that the notebook stops early on a 57-image validation set, and if that is")
    print("right it should be visible here as selection at a very low epoch.")
    print(f"\ntables -> {TABLES}\nruns   -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
