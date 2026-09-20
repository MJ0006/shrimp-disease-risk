"""Experiment 01 -- Quantify specimen leakage in image-level vs specimen-level splitting.

Objective
---------
Establish, as a measured quantity rather than an argument, how much of a test set is
contaminated by the standard image-level random split on ShrimpDiseaseImageBD.

Hypothesis
----------
Because the 1,149 images are repeat photographs of only 416 individual shrimp (2.76 photos per
specimen), an image-level split places a sibling photograph of most test specimens into the
training set. Under specimen-level grouping this must fall to exactly zero.

Why it matters
--------------
A model evaluated on a leaked test set can score well by re-identifying an individual it
memorised rather than by learning disease morphology. Published results on this dataset report
93-96.88% accuracy; if those used image-level splits, the figures carry this optimism. This
experiment measures the exposure. It does NOT by itself prove the accuracy inflation -- that
requires training models under both protocols (Experiment 02).

Design
------
No model is trained here. Splits only, repeated over several seeds so the result is not a
property of one lucky partition. Both protocols are stratified by class and use identical
val/test fractions, so the only varying factor is whether grouping is applied.

Outputs
-------
    reports/tables/exp01_leakage_audit.csv      per-seed, per-protocol measurements
    reports/tables/exp01_leakage_summary.csv    mean +/- SD by protocol
    reports/tables/exp01_class_balance.csv      class counts per partition
    reports/tables/exp01_environment.csv        what this ran on
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shrimp_risk.common.paths import IMAGE_MANIFEST, TABLES  # noqa: E402
from shrimp_risk.common.seeding import environment_fingerprint  # noqa: E402
from shrimp_risk.vision import splits as S  # noqa: E402

# Several seeds because a single split's leakage figure is one draw from a distribution.
# Reporting one number would invite "you got unlucky/lucky with that split".
SEEDS = (2025, 7, 42, 123, 999)


def main() -> int:
    manifest = S.load_raw_manifest(IMAGE_MANIFEST)

    n_img = len(manifest)
    n_spec = manifest["specimen_id"].nunique()
    print(f"manifest : {n_img:,} images, {n_spec:,} specimens "
          f"({n_img / n_spec:.2f} images per specimen)")
    print(f"classes  : {dict(manifest['class'].value_counts())}\n")

    rows = []
    for seed in SEEDS:
        for protocol, fn in (
            ("image_level", S.make_image_level_splits),
            ("specimen_level", S.make_specimen_level_splits),
        ):
            split = fn(manifest, seed=seed)
            rec = S.leakage_report(manifest, split)
            rec["seed"] = seed
            rows.append(rec)

    df = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "exp01_leakage_audit.csv", index=False)

    summary = (
        df.groupby("protocol")
        .agg(
            n_seeds=("seed", "count"),
            train_images=("n_train_images", "mean"),
            test_images=("n_test_images", "mean"),
            train_specimens=("n_train_specimens", "mean"),
            test_specimens=("n_test_specimens", "mean"),
            shared_specimens=("shared_train_test_specimens", "mean"),
            pct_test_leaked_mean=("pct_test_leaked", "mean"),
            pct_test_leaked_sd=("pct_test_leaked", "std"),
            pct_val_leaked_mean=("pct_val_leaked", "mean"),
        )
        .round(2)
        .reset_index()
    )
    summary.to_csv(TABLES / "exp01_leakage_summary.csv", index=False)

    balance = S.class_balance(manifest, S.make_specimen_level_splits(manifest, seed=SEEDS[0]))
    balance.to_csv(TABLES / "exp01_class_balance.csv")

    print("=" * 72)
    print("RESULT -- % of TEST images whose specimen also appears in TRAIN")
    print("=" * 72)
    for _, r in summary.iterrows():
        sd = "" if pd.isna(r.pct_test_leaked_sd) else f" +/- {r.pct_test_leaked_sd:.2f}"
        print(f"  {r.protocol:16s} {r.pct_test_leaked_mean:6.2f}%{sd}"
              f"   (shared specimens: {r.shared_specimens:.1f})")
    print()
    print("class balance under specimen-level splitting:")
    print(balance.to_string())
    print()
    # Persisted, not merely printed. Master Plan section 8: a result without a recorded
    # environment is treated as not obtained, and a console line nobody kept is not a record.
    pd.DataFrame([environment_fingerprint()]).to_csv(
        TABLES / "exp01_environment.csv", index=False)

    print("environment:", environment_fingerprint())
    print(f"\nwrote 4 tables -> {TABLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
