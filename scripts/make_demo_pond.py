"""Build the demonstration pond for the final review: held-out shrimp, labels hidden.

Why this script exists
----------------------
The final review asks for the project's outcomes to be demonstrated, and the outcome is a tool:
photographs of a pond sample go in, a pond-level White Spot prevalence and a farm risk figure come
out (`scripts/predict_farm.py`). Demonstrating it honestly needs two things the raw corpus does
not give.

1. **Shrimp the deployed model never trained on.** The deployment checkpoint was trained on fold 0
   of the specimen-grouped split (`train_edge_checkpoint.py`, "Fold 0, always"), so fold 0's
   held-out specimens are the only images it can be shown without leaking (`D-007`, `D-044`).
2. **Filenames that do not carry the answer.** ShrimpDiseaseBD names files `WSSV-12-img-1.jpg`.
   The model reads pixels only, but a panel looking at a folder of files named after their class
   will reasonably doubt that. Every specimen is renamed to a neutral `S001-img-1.jpg`, in a
   shuffled order, and the truth goes to a separate answer key opened after the prediction.
   `predict_farm.py` groups photographs on the `-img-` suffix, so the specimen structure survives.

The pond is every held-out specimen: 84 shrimp, 40 White Spot positive (47.6%). No selection,
so nothing was picked to make the demonstration look good.

Output goes to `data/demo/` (not published; `data/` is gitignored). Safe to re-run: it rebuilds
the same pond from scratch.

Usage:
    .venv/Scripts/python.exe scripts/make_demo_pond.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shrimp_risk.common.paths import resolve_image_manifest
from shrimp_risk.vision import splits as S

SEED = 2025
DEPLOYMENT_FOLD = 0          # must match train_edge_checkpoint.py
POSITIVE = ("WSSV", "BG_WSSV")   # both carry White Spot, as in quantification.positive_indices()
OUT = ROOT / "data" / "demo"


def write_pond(name: str, specimens: pd.DataFrame, images: pd.DataFrame,
               rng: np.random.Generator) -> None:
    """Copy the pond's photographs under neutral names and write its answer key beside it."""
    pond_dir = OUT / name
    if pond_dir.exists():
        shutil.rmtree(pond_dir)
    pond_dir.mkdir(parents=True)

    specimens = specimens.iloc[rng.permutation(len(specimens))].reset_index(drop=True)
    specimens["demo_id"] = [f"S{i + 1:03d}" for i in range(len(specimens))]

    for row in specimens.itertuples():
        photos = images[images["specimen_id"] == row.specimen_id]["path"]
        for k, src in enumerate(photos, start=1):
            src = Path(src) if Path(src).is_absolute() else ROOT / src
            shutil.copyfile(src, pond_dir / f"{row.demo_id}-img-{k}{src.suffix.lower()}")

    key = specimens[["demo_id", "specimen_id", "class", "white_spot"]]
    key.to_csv(OUT / f"{name}_answer_key.csv", index=False)
    n_pos = int(key["white_spot"].sum())
    print(f"{name}: {len(key)} specimens, {len(list(pond_dir.iterdir()))} photos, "
          f"{n_pos} White Spot positive -> true prevalence {n_pos / len(key):.4f}")


def main() -> int:
    manifest = S.load_raw_manifest(resolve_image_manifest())
    fold = S.make_specimen_folds(manifest, n_splits=5, seed=SEED)[DEPLOYMENT_FOLD]
    train, held_out = manifest.iloc[fold.train], manifest.iloc[fold.test]
    assert not set(train["specimen_id"]) & set(held_out["specimen_id"]), "fold leaks specimens"

    specimens = held_out.groupby("specimen_id", as_index=False)["class"].first()
    specimens["white_spot"] = specimens["class"].isin(POSITIVE).astype(int)

    write_pond("pond_A", specimens, held_out, np.random.default_rng(SEED))
    print(f"Written to {OUT}. Open the answer key only after predict_farm.py has run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
