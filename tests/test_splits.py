"""Property tests for the splitters.

These are the most important tests in the project. Stage 4's headline result -- that
image-level splitting inflates macro-F1 by 0.149 on this dataset -- is only meaningful if the
specimen-level splitter really does keep every photograph of one shrimp on one side of the
split. That is a property of the code, and properties are what tests are for.

Master Plan section 8 specifies two of these explicitly:
    "stratified split preserves class proportions within tolerance; no image in two splits".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shrimp_risk.vision import splits as S

SEEDS = [42, 7, 2025]


# --------------------------------------------------------------------------------------
# The guard inside Split itself
# --------------------------------------------------------------------------------------

def test_split_rejects_overlapping_partitions():
    """Split.__post_init__ must refuse to construct an invalid split.

    This guard is why no downstream experiment needs to re-check disjointness: an overlapping
    split cannot be built in the first place.
    """
    with pytest.raises(ValueError, match="overlap"):
        S.Split(
            train=np.array([0, 1, 2]),
            val=np.array([2, 3]),      # 2 is also in train
            test=np.array([4]),
            protocol="deliberately_broken",
            seed=0,
        )


def test_split_accepts_disjoint_partitions():
    split = S.Split(
        train=np.array([0, 1]), val=np.array([2]), test=np.array([3]),
        protocol="ok", seed=0,
    )
    assert split.sizes == {"train": 2, "val": 1, "test": 1}


# --------------------------------------------------------------------------------------
# Partition integrity -- true for both protocols
# --------------------------------------------------------------------------------------

@pytest.mark.dataset
@pytest.mark.parametrize("maker", [S.make_image_level_splits, S.make_specimen_level_splits])
@pytest.mark.parametrize("seed", SEEDS)
def test_no_image_appears_in_two_partitions(raw_manifest, maker, seed):
    split = maker(raw_manifest, seed=seed)
    all_idx = np.concatenate([split.train, split.val, split.test])
    assert len(all_idx) == len(set(all_idx.tolist())), "an image appears in more than one partition"


@pytest.mark.dataset
@pytest.mark.parametrize("maker", [S.make_image_level_splits, S.make_specimen_level_splits])
@pytest.mark.parametrize("seed", SEEDS)
def test_split_covers_every_row_exactly_once(raw_manifest, maker, seed):
    """No image may be silently dropped -- a shrinking corpus would bias results invisibly."""
    split = maker(raw_manifest, seed=seed)
    covered = np.sort(np.concatenate([split.train, split.val, split.test]))
    assert np.array_equal(covered, np.arange(len(raw_manifest)))


@pytest.mark.dataset
@pytest.mark.parametrize("maker", [S.make_image_level_splits, S.make_specimen_level_splits])
@pytest.mark.parametrize("seed", SEEDS)
def test_all_four_classes_present_in_every_partition(raw_manifest, maker, seed):
    """A partition missing a class makes macro-F1 undefined for it and the run uninterpretable."""
    split = maker(raw_manifest, seed=seed)
    for name in ("train", "val", "test"):
        present = set(raw_manifest.iloc[getattr(split, name)]["class"])
        assert present == {"Healthy", "BG", "WSSV", "BG_WSSV"}, f"{name} is missing a class"


# --------------------------------------------------------------------------------------
# The property the whole project rests on
# --------------------------------------------------------------------------------------

@pytest.mark.dataset
@pytest.mark.parametrize("seed", SEEDS)
def test_specimen_level_split_leaks_nothing(raw_manifest, seed):
    """ZERO test images may share a specimen with training. This is the load-bearing test.

    If this ever fails, every specimen-level number in the project is wrong and the Stage 4
    conclusion collapses -- so it is asserted exactly, not approximately.
    """
    split = S.make_specimen_level_splits(raw_manifest, seed=seed)
    report = S.leakage_report(raw_manifest, split)
    assert report["shared_train_test_specimens"] == 0
    assert report["shared_train_val_specimens"] == 0
    assert report["pct_test_leaked"] == 0.0
    assert report["pct_val_leaked"] == 0.0


@pytest.mark.dataset
@pytest.mark.parametrize("seed", SEEDS)
def test_specimen_stays_whole(raw_manifest, seed):
    """Stated the other way round: no specimen may be spread across two partitions."""
    split = S.make_specimen_level_splits(raw_manifest, seed=seed)
    groups = raw_manifest["specimen_id"].to_numpy()
    sets = {name: set(groups[getattr(split, name)]) for name in ("train", "val", "test")}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
    assert len(sets["train"] | sets["val"] | sets["test"]) == 416


@pytest.mark.dataset
@pytest.mark.parametrize("seed", SEEDS)
def test_image_level_split_leaks_heavily(raw_manifest, seed):
    """The counterpart: the conventional protocol must be shown to leak, not assumed to.

    Experiment 01 measured 87.98% +/- 1.32. The assertion is deliberately loose (>50%) so it
    tests the mechanism rather than re-asserting one measurement -- but a value this far above
    zero is the finding that motivates the entire corrected benchmark.
    """
    split = S.make_image_level_splits(raw_manifest, seed=seed)
    report = S.leakage_report(raw_manifest, split)
    assert report["pct_test_leaked"] > 50.0
    assert report["shared_train_test_specimens"] > 0


# --------------------------------------------------------------------------------------
# Stratification tolerance
# --------------------------------------------------------------------------------------

@pytest.mark.dataset
@pytest.mark.parametrize("seed", SEEDS)
def test_image_level_split_preserves_class_proportions(raw_manifest, seed):
    """Master Plan section 8: preserves class proportions within tolerance.

    Tolerance is 3 percentage points. The image-level splitter is stratified, so the only
    difference between it and the specimen-level splitter is grouping -- which is what makes
    the B-vs-C comparison attributable to leakage alone.
    """
    split = S.make_image_level_splits(raw_manifest, seed=seed)
    overall = raw_manifest["class"].value_counts(normalize=True)
    for name in ("train", "val", "test"):
        part = raw_manifest.iloc[getattr(split, name)]["class"].value_counts(normalize=True)
        for cls, p in overall.items():
            assert abs(float(part.get(cls, 0.0)) - float(p)) < 0.03, f"{name}/{cls} drifted"


@pytest.mark.dataset
@pytest.mark.parametrize("seed", SEEDS)
def test_partition_sizes_are_roughly_as_requested(raw_manifest, seed):
    """15/15 requested. Specimen-level grouping cannot hit it exactly -- groups are indivisible
    and vary in size -- so the tolerance is wider for it than the exact figure would suggest."""
    n = len(raw_manifest)
    for maker, tol in ((S.make_image_level_splits, 0.02), (S.make_specimen_level_splits, 0.05)):
        split = maker(raw_manifest, seed=seed)
        assert abs(split.sizes["val"] / n - 0.15) < tol
        assert abs(split.sizes["test"] / n - 0.15) < tol


# --------------------------------------------------------------------------------------
# k-fold -- the protocol Stage 5 uses for model comparison
# --------------------------------------------------------------------------------------

@pytest.mark.dataset
def test_kfold_tests_every_specimen_exactly_once(raw_manifest):
    """Regression guard from Master Plan section 8: model comparison uses identical folds.

    Across 5 folds every one of the 416 specimens must be tested exactly once. If a specimen
    were tested twice, or never, the fold means would silently weight specimens unequally.
    """
    folds = S.make_specimen_folds(raw_manifest, n_splits=5, seed=2025)
    assert len(folds) == 5
    groups = raw_manifest["specimen_id"].to_numpy()

    tested: list[str] = []
    for f in folds:
        tested.extend(sorted(set(groups[f.test])))
    assert len(tested) == len(set(tested)) == 416


@pytest.mark.dataset
def test_kfold_partitions_are_specimen_disjoint(raw_manifest):
    groups = raw_manifest["specimen_id"].to_numpy()
    for f in S.make_specimen_folds(raw_manifest, n_splits=5, seed=2025):
        tr, va, te = (set(groups[getattr(f, n)]) for n in ("train", "val", "test"))
        assert not tr & va and not tr & te and not va & te, f"fold {f.fold} is contaminated"
        assert S.leakage_report(raw_manifest, f)["pct_test_leaked"] == 0.0


@pytest.mark.dataset
def test_folds_are_reproducible_from_the_seed(raw_manifest):
    """Two calls with the same seed must give identical folds, or identical folds across
    candidates in the Stage 5 comparison is not actually guaranteed."""
    a = S.make_specimen_folds(raw_manifest, n_splits=5, seed=2025)
    b = S.make_specimen_folds(raw_manifest, n_splits=5, seed=2025)
    for fa, fb in zip(a, b):
        assert np.array_equal(fa.train, fb.train)
        assert np.array_equal(fa.val, fb.val)
        assert np.array_equal(fa.test, fb.test)


@pytest.mark.dataset
def test_different_seeds_give_different_splits(raw_manifest):
    """Guards against a seed that is accepted but ignored -- which would make the 3-seed
    stability result in Stage 4 meaningless."""
    a = S.make_specimen_level_splits(raw_manifest, seed=42)
    b = S.make_specimen_level_splits(raw_manifest, seed=7)
    assert not np.array_equal(np.sort(a.test), np.sort(b.test))


# --------------------------------------------------------------------------------------
# Manifest loading
# --------------------------------------------------------------------------------------

@pytest.mark.dataset
def test_load_raw_manifest_excludes_annotated_subset(tmp_path, manifest):
    """The 746 annotated images are re-published photographs of the SAME specimens. Including
    them would reintroduce the leakage this module exists to prevent, so the loader drops them.
    """
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)
    raw = S.load_raw_manifest(path)
    assert len(raw) == 1149
    assert set(raw["subset"]) == {"raw"}


def test_load_raw_manifest_rejects_missing_specimen_ids(tmp_path):
    """A blank specimen_id would silently degrade grouping into image-level splitting -- the
    exact failure this project exists to expose -- so it must be a hard error."""
    bad = pd.DataFrame(
        {
            "path": ["a.jpg", "b.jpg"],
            "subset": ["raw", "raw"],
            "class": ["BG", "BG"],
            "specimen_id": ["BG-1", None],
        }
    )
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="no specimen_id"):
        S.load_raw_manifest(path)
