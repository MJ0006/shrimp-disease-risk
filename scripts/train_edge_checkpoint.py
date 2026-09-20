"""Train the edge backbone and persist the weights the deployment pipeline exports.

Why this script exists
----------------------
Experiment 04 trains every backbone five times and reports mean +/- SD across folds. That is the
right shape for *choosing* an architecture: a single specimen-level split holds only ~62 specimens,
so one number cannot rank five models. But it is the wrong shape for *shipping* one, because it
leaves five different networks and no single artefact to deploy -- and, until 2026-09-20, it left
none at all: `trainer.py` kept the selected weights in memory and never wrote them to disk.

`export_edge_model.py` then built a fresh `mobilenetv3_large_100` with `pretrained=False` and
exported that. Size and latency are properties of the graph, not the weights, so every number in
`exp12_export.csv` was correct while the exported network returned ~0.25 on all four classes for
any input. This script closes that gap: it trains the deployment candidate on one fold and writes
the checkpoint the exporter loads.

What this is and is not
-----------------------
This produces a **deployable artefact**, not a new benchmark result. The reported backbone
comparison stays what Experiment 04 measured -- `mobilenetv3_large_100` at 0.6705 +/- 0.0415
macro-F1 across five specimen-grouped folds. The single fold trained here is one draw from that
distribution and its individual score must never be quoted as the model's performance: quoting the
best of five folds is the selection-on-test error this entire project exists to demonstrate.

Which fold, and why it is fixed
-------------------------------
Fold 0, always. Not the best-scoring fold. Picking the fold with the highest test score would
choose a model on the set its score is reported from -- the same failure as image-level splitting,
one level up. Fold 0 is an arbitrary but *pre-committed* choice, which is what keeps it honest.

The held-out fold is still a clean specimen-grouped test set for this checkpoint, so the score
printed at the end is an unbiased estimate for this one network. It is written into the checkpoint
so the exported model can always be traced back to the run that produced it.

Usage
-----
    .venv/Scripts/python.exe scripts/train_edge_checkpoint.py
    .venv/Scripts/python.exe scripts/train_edge_checkpoint.py --backbone convnext_tiny

Runtime is roughly 90 s per backbone on an RTX 5060. Run ONE torch script at a time -- three
concurrent torch processes exhaust the Windows paging file and a dataloader worker dies with
`WinError 1455`, whose traceback never mentions memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shrimp_risk.common.paths import (  # noqa: E402
    PROJECT_ROOT, REPORTS, RUNS, resolve_image_manifest,
)
from shrimp_risk.common.seeding import environment_fingerprint, seed_everything  # noqa: E402
from shrimp_risk.vision import splits as S  # noqa: E402
from shrimp_risk.vision.datasets import CLASSES, make_loaders  # noqa: E402
from shrimp_risk.vision.models import create_model, get_device  # noqa: E402
from shrimp_risk.vision.trainer import TrainConfig, train_and_evaluate  # noqa: E402

SEED = 2025
N_FOLDS = 5

# Mirrors `BACKBONE_LR` in exp04_backbone_comparison.py. ConvNeXt needs 5e-5: at the 3e-4 shared
# by the others it failed to fit its own *training* data in 3 of 5 folds and scored 0.3966
# (`D-020`). Keeping the two tables in step matters -- a checkpoint trained at a different
# learning rate from the benchmark is not the model the benchmark measured.
BACKBONE_LR = {"convnext_tiny": 5e-5}
DEFAULT_LR = 3e-4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=str, default="mobilenetv3_large_100")
    parser.add_argument("--fold", type=int, default=0,
                        help="pre-committed at 0; do not tune this")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    models_dir = REPORTS / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.out) if args.out else models_dir / f"{args.backbone}_best.pt"

    device = get_device()
    manifest = S.load_raw_manifest(resolve_image_manifest())
    folds = S.make_specimen_folds(manifest, n_splits=args.folds, seed=SEED)
    fold = folds[args.fold]

    print("=" * 74)
    print(f"Training deployment checkpoint: {args.backbone}  fold {fold.fold}/{args.folds}")
    print(f"  {len(fold.train)} train / {len(fold.test)} test images  "
          f"({pd.Series(manifest.iloc[fold.test]['specimen_id']).nunique()} test specimens)")
    print(f"  device {device}  lr {BACKBONE_LR.get(args.backbone, DEFAULT_LR)}")
    print("=" * 74)

    seed_everything(SEED)
    cfg = TrainConfig(
        model_name=args.backbone,
        seed=SEED,
        protocol="specimen_level_kfold",
        fold=fold.fold,
        augment=True,
        epochs=args.epochs,
        patience=7,
        batch_size=32,
        lr=BACKBONE_LR.get(args.backbone, DEFAULT_LR),
        checkpoint_path=str(ckpt_path),
    )
    loaders = make_loaders(
        manifest, fold, PROJECT_ROOT, batch_size=cfg.batch_size,
        img_size=cfg.img_size, augment=True, num_workers=4,
    )
    model = create_model(args.backbone, num_classes=len(CLASSES))
    res = train_and_evaluate(model, loaders, cfg, device, CLASSES)

    # Record the run beside the checkpoint. A weights file whose provenance is not written down
    # cannot be matched to the score that justified shipping it.
    run_dir = RUNS / "edge_checkpoint"
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "config": asdict(cfg),
        "fold": fold.fold,
        "best_epoch": res.best_epoch,
        "best_val_f1": round(res.best_val_f1, 4),
        "test_acc": round(res.test_acc, 4),
        "test_f1_macro": round(res.test_f1_macro, 4),
        "test_f1_weighted": round(res.test_f1_weighted, 4),
        **{f"f1_{c}": round(res.per_class[c]["f1-score"], 4) for c in CLASSES},
        "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
        "environment": environment_fingerprint(),
        "caveat": (
            "Single fold. This is a deployable artefact, not a benchmark result. The reported "
            "figure for this backbone is Experiment 04's 5-fold mean +/- SD; this fold's score "
            "is one draw from that distribution and must not be quoted as the model's "
            "performance."
        ),
    }
    (run_dir / f"{args.backbone}__fold{fold.fold}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")

    print()
    print("=" * 74)
    print(f"  checkpoint      {ckpt_path.relative_to(PROJECT_ROOT)}")
    print(f"  best epoch      {res.best_epoch}  (val macro-F1 {res.best_val_f1:.4f})")
    print(f"  held-out fold   macro-F1 {res.test_f1_macro:.4f}  acc {res.test_acc:.4f}")
    print()
    print("  This single-fold score is NOT the reported result. Experiment 04's 5-fold")
    print("  mean +/- SD remains the figure that goes on a slide.")
    print("=" * 74)
    print()
    print("Next:  .venv/Scripts/python.exe scripts/export_edge_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
