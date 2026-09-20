"""Train/validation/test splitting for the shrimp image dataset.

This module carries the project's central methodological claim, so it is worth stating the
problem plainly before the code.

The problem
-----------
`data/interim/image_manifest.csv` holds 1,149 images. It does NOT hold 1,149 independent
observations. Filenames follow `<Class>-<specimen>-img-<n>`, so `Healthy-1-img-1`,
`Healthy-1-img-2` and `Healthy-1-img-3` are three photographs of the *same individual shrimp*.
Measured over the whole dataset:

    416 unique specimens across 1,149 images  ->  2.76 images per specimen
    341 specimens contribute 3 images, 51 contribute 2, 24 contribute 1

If images are split at random, photographs of one shrimp land on both sides of the split. A
model can then score highly by recognising an individual it memorised during training rather
than by learning what the disease looks like. The reported number goes up; the model does not
get better. This is textbook data leakage, and because no two files are byte-identical
(hash-duplicate count is exactly zero) it is invisible to any hash-based duplicate check.

What this module provides
-------------------------
Two splitters with a deliberately identical interface, so an experiment can swap one for the
other and change nothing else:

    make_image_level_splits    -- the leaky protocol, reproduced so its optimism is measurable
    make_specimen_level_splits -- the correct protocol, grouping on specimen

Keeping the leaky one is not an oversight. "Our method is better" is not a claim until the thing
it is better than has been measured under identical conditions -- and here that measurement *is*
the result: the 0.1494 macro-F1 gap only exists because both protocols are implemented behind one
interface and can be swapped with nothing else changing. Deleting the abandoned approach would
delete the evidence for replacing it. (Master Prompt section 12 asks for the same record.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedShuffleSplit,
)


@dataclass(frozen=True)
class Split:
    """One train/val/test partition, stored as manifest row indices.

    Indices rather than copies of the data: a split is metadata about the manifest, and
    keeping it that way makes it cheap to persist and impossible for the split to drift out
    of sync with the rows it describes.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    protocol: str
    seed: int
    fold: int = 0

    def __post_init__(self) -> None:
        # A silent overlap here would invalidate every downstream number, so it is checked
        # at construction rather than trusted.
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = set(getattr(self, a)) & set(getattr(self, b))
            if overlap:
                raise ValueError(f"{a}/{b} overlap by {len(overlap)} rows in {self.protocol}")

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def load_raw_manifest(path) -> pd.DataFrame:
    """Load the classification subset of the manifest.

    The manifest also contains the 746 annotated detection images, which are a *subset of the
    same specimens* re-published with bounding boxes. Including them in a classification split
    would reintroduce the exact leakage this module exists to prevent, so they are dropped here
    rather than left for each caller to remember.
    """
    df = pd.read_csv(path)
    raw = df[df["subset"] == "raw"].reset_index(drop=True)
    if raw["specimen_id"].eq("").any() or raw["specimen_id"].isna().any():
        n = int(raw["specimen_id"].isna().sum() + raw["specimen_id"].eq("").sum())
        raise ValueError(f"{n} manifest rows have no specimen_id; re-run scripts/extract_images.py")
    return raw


def make_image_level_splits(
    manifest: pd.DataFrame,
    seed: int = 2025,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    stratify: bool = True,
) -> Split:
    """The LEAKY protocol: split individual images, ignoring specimen identity.

    This is what the inherited notebook does (`random_split` at `[0.9, 0.05, 0.05]`) and what
    any straightforward implementation would do. Reproduced here so the size of its optimism
    can be measured rather than asserted.

    **`stratify=True` (the default) is the fair-comparison setting**, not the faithful one.
    Stratifying by class makes the *only* difference from the specimen-level splitter the
    grouping itself, which is what makes Experiment 02's B-versus-C comparison attributable to
    leakage rather than to class imbalance. Every run before 2026-09-06 used it.

    **`stratify=False` is what the notebook actually did.** `torch.utils.data.random_split`
    shuffles and cuts; it knows nothing about classes. At a 5% test fraction that is 57 images,
    and an unstratified draw of 57 from a 403/328/220/198 corpus can easily hold single-digit
    counts of the smallest class -- which is itself part of why the notebook's reported score is
    unstable. Experiment 02b sets this to reproduce that behaviour rather than a tidied version
    of it (`D-010`); nothing else in the project does.
    """
    y = manifest["class"].to_numpy()
    idx = np.arange(len(manifest))
    holdout = val_frac + test_frac

    if stratify:
        s1 = StratifiedShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
        train_idx, rest_idx = next(s1.split(idx, y))
        # Split the holdout into val and test, keeping their relative proportions.
        s2 = StratifiedShuffleSplit(
            n_splits=1, test_size=test_frac / holdout, random_state=seed
        )
        v_rel, t_rel = next(s2.split(rest_idx, y[rest_idx]))
        train, val, test = idx[train_idx], rest_idx[v_rel], rest_idx[t_rel]
    else:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(idx)
        n_test = int(round(test_frac * len(idx)))
        n_val = int(round(val_frac * len(idx)))
        test, val = shuffled[:n_test], shuffled[n_test:n_test + n_val]
        train = shuffled[n_test + n_val:]

    return Split(
        train=train,
        val=val,
        test=test,
        protocol="image_level" if stratify else "image_level_unstratified",
        seed=seed,
    )


def make_specimen_level_splits(
    manifest: pd.DataFrame,
    seed: int = 2025,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Split:
    """The CORRECT protocol: split specimens, so all photos of one shrimp stay together.

    `GroupShuffleSplit` guarantees group integrity but does not stratify, so class balance is
    checked afterwards by the caller. Perfect stratification and perfect grouping cannot both
    be guaranteed on a single split of 416 groups; group integrity is the non-negotiable one,
    because violating it invalidates the measurement, whereas mild class imbalance only makes
    it noisier. For the k-fold comparison we use `StratifiedGroupKFold`, which approximates
    both.
    """
    groups = manifest["specimen_id"].to_numpy()
    idx = np.arange(len(manifest))

    holdout = val_frac + test_frac
    g1 = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
    train_idx, rest_idx = next(g1.split(idx, groups=groups))

    g2 = GroupShuffleSplit(
        n_splits=1, test_size=test_frac / holdout, random_state=seed
    )
    v_rel, t_rel = next(g2.split(rest_idx, groups=groups[rest_idx]))

    return Split(
        train=idx[train_idx],
        val=rest_idx[v_rel],
        test=rest_idx[t_rel],
        protocol="specimen_level",
        seed=seed,
    )


def make_specimen_folds(
    manifest: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 2025,
    val_frac_of_fit: float = 0.20,
) -> list[Split]:
    """Stratified, group-aware k-fold -- the protocol used for model comparison.

    Cross-validation matters more here than it would on a larger corpus. Grouping reduces the
    effective sample size to 416, and a single specimen-level test split holds only ~62
    specimens. A one-off number on 62 specimens is too noisy to rank backbones, so every
    comparison reports mean +/- SD across folds instead of a single figure.

    The validation fold is carved out of the training portion of each fold, so val and test
    remain specimen-disjoint from train and from each other.

    `val_frac_of_fit` is an explicit fraction of the training portion rather than a quantity
    derived from `n_splits`. An earlier version used `1 / (n_splits - 1)`, which silently
    evaluates to 1.0 at `n_splits=2` and asks the splitter to put the entire fit set into
    validation. Tying one parameter to another like that hides a failure that only appears at
    one setting, so the fraction is now stated directly and validated.
    """
    if not 0.0 < val_frac_of_fit < 1.0:
        raise ValueError(f"val_frac_of_fit must lie in (0, 1); got {val_frac_of_fit}")
    if n_splits < 2:
        raise ValueError(f"k-fold needs n_splits >= 2; got {n_splits}")
    y = manifest["class"].to_numpy()
    groups = manifest["specimen_id"].to_numpy()
    idx = np.arange(len(manifest))

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[Split] = []

    for fold, (fit_idx, test_idx) in enumerate(sgkf.split(idx, y, groups)):
        inner = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_fit, random_state=seed)
        tr_rel, va_rel = next(inner.split(fit_idx, groups=groups[fit_idx]))
        folds.append(
            Split(
                train=fit_idx[tr_rel],
                val=fit_idx[va_rel],
                test=test_idx,
                protocol="specimen_level_kfold",
                seed=seed,
                fold=fold,
            )
        )
    return folds


def leakage_report(manifest: pd.DataFrame, split: Split) -> dict[str, float]:
    """Quantify how much specimen leakage a split actually contains.

    This is the measurement that turns "image-level splitting leaks" from an argument into a
    number, and it is what the Stage 4/5 results table reports alongside accuracy.

    `test_images_with_train_sibling` is the headline: the share of test images for which at
    least one photo of the same shrimp appears in training. Under specimen-level splitting it
    must be exactly 0 -- if it is not, the splitter is broken.
    """
    g = manifest["specimen_id"].to_numpy()
    train_specimens = set(g[split.train])
    test_specimens = set(g[split.test])
    val_specimens = set(g[split.val])

    leaked_test = [i for i in split.test if g[i] in train_specimens]
    leaked_val = [i for i in split.val if g[i] in train_specimens]

    return {
        "protocol": split.protocol,
        "n_train_images": len(split.train),
        "n_val_images": len(split.val),
        "n_test_images": len(split.test),
        "n_train_specimens": len(train_specimens),
        "n_test_specimens": len(test_specimens),
        "shared_train_test_specimens": len(train_specimens & test_specimens),
        "shared_train_val_specimens": len(train_specimens & val_specimens),
        "test_images_with_train_sibling": len(leaked_test),
        "pct_test_leaked": round(100 * len(leaked_test) / max(len(split.test), 1), 2),
        "pct_val_leaked": round(100 * len(leaked_val) / max(len(split.val), 1), 2),
    }


def class_balance(manifest: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Class counts per partition -- reported so imbalance is never an unexamined confounder."""
    rows = {}
    for name in ("train", "val", "test"):
        rows[name] = manifest.iloc[getattr(split, name)]["class"].value_counts()
    return pd.DataFrame(rows).fillna(0).astype(int)
