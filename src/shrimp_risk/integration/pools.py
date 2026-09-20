"""Pseudo-farm pools -- constructing a prevalence whose true value is known.

The limitation this module is built around
------------------------------------------
`ShrimpDiseaseImageBD` carries **no farm identifier**. There is no way to ask it "what fraction of
farm 17's shrimp had White Spot", because nothing in the dataset records which shrimp came from
which pond. That is limitation 2 in the Master Plan, and it is the single most important constraint
on the whole bridge.

The response is not to invent farm labels. It is to build pools *deliberately*, at a prevalence
chosen in advance, so that the answer the estimator ought to return is known before it runs. An
estimator evaluated against a truth we set is a stronger test than one evaluated against a
self-reported prevalence we could not verify -- which is precisely the weakness this project
identified in the base paper's data. The trade is that the pools are constructed rather than
observed, and every result that uses them says so.

What a pool is
--------------
A set of **distinct specimens** drawn from one fold's test partition, containing a specified
fraction of White-Spot-positive animals. Distinct, because a farm contains distinct shrimp: drawing
the same animal twice would put correlated images in the pool and make the estimator look more
precise than it is. Specimens rather than images, because prevalence is a property of animals --
see `quantification.aggregate_to_specimen`.

Why pools are drawn within a fold, never across folds
-----------------------------------------------------
Each fold's test specimens were scored by that fold's own model, which has that fold's own
calibration temperature and its own validation-estimated confusion matrix. Mixing specimens from
two folds into one pool would mean correcting posteriors from model A with a matrix estimated for
model B. The alternative -- pooling all five validation folds to estimate one shared matrix -- is
tempting because it would give 416 specimens to draw from instead of about 83, but the validation
partitions of different folds overlap (a specimen can be in validation for fold 0 and training for
fold 1), so the pooled matrix would be estimated from dependent samples and its standard error
would be unknowable.

So Experiment 07 runs **per fold and reports mean +/- SD across folds**, which is the same
protocol Stage 5 used for exactly the same reason: a single number from one partition of 83
specimens cannot be trusted, and five of them with a spread can.

What this costs: the reachable prevalence grid
----------------------------------------------
A fold's test partition holds roughly 83 specimens, of which about 40 are White-Spot-positive
(199 of 416 across the corpus -- 118 `WSSV` plus 81 `BG_WSSV`). A pool of 40 specimens at 95%
prevalence would need 38 positives out of the ~40 available; at some combinations of pool size and
prevalence there simply are not enough animals of one kind.

Those cells are **refused and recorded**, not filled by sampling the same shrimp twice. A silently
resampled pool would produce an estimate at a prevalence the data cannot actually support, and it
would look exactly like the ones that are sound. `with_replacement=True` exists for a documented
secondary analysis, and the experiment reports which cells needed it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Pool:
    """One pseudo-farm: which specimens are in it, and what its true prevalence is.

    `true_prevalence` is the *realised* fraction, not the requested one. They differ whenever
    `size * target` is not an integer -- a 20-shrimp pool cannot be 47% positive -- and reporting
    the request instead of the reality would put a small systematic error into the bias
    measurement, in the same direction for every pool of that size.
    """

    indices: np.ndarray
    target_prevalence: float
    true_prevalence: float
    size: int
    n_positive: int
    with_replacement: bool


def specimen_truth(
    specimen_ids: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One true class per specimen, aligned to sorted unique ids.

    Every image of a shrimp carries that shrimp's class, so this is a consistency check as much as
    a reduction: if two images of the same specimen disagree about its class, the manifest is wrong
    and nothing downstream of it is safe. The check is cheap and the failure it catches would
    otherwise surface as an inexplicable bias in the estimator.

    Returns the labels and the sorted unique ids, in the same order
    `quantification.aggregate_to_specimen` returns its rows, so the two can be indexed together.
    """
    ids = np.asarray(specimen_ids)
    unique = np.unique(ids)
    truth = np.empty(len(unique), dtype="int64")
    for k, sid in enumerate(unique):
        vals = np.unique(labels[ids == sid])
        if len(vals) != 1:
            raise ValueError(
                f"specimen {sid!r} carries {len(vals)} different class labels {vals.tolist()} -- "
                "the manifest is inconsistent and no result downstream of it is valid"
            )
        truth[k] = vals[0]
    return truth, unique


def is_positive(truth: np.ndarray, positive: list[int]) -> np.ndarray:
    """Boolean positive/negative per specimen, for a given positive class set."""
    return np.isin(truth, positive)


def build_pool(
    rng: np.random.Generator,
    positive_mask: np.ndarray,
    target_prevalence: float,
    size: int,
    with_replacement: bool = False,
) -> Pool | None:
    """Draw one pool at a target prevalence, or return None if the fold cannot supply it.

    `None` rather than an exception, because infeasibility is an ordinary and expected outcome
    across a grid -- the caller records the cell as unreachable and moves on. An exception would
    make the common case look like an error and would tempt whoever hits it into widening the
    sampling rule to get past it, which is the thing that must not happen quietly.
    """
    if not 0.0 <= target_prevalence <= 1.0:
        raise ValueError(f"prevalence must lie in [0, 1]; got {target_prevalence}")
    if size < 1:
        raise ValueError(f"pool size must be at least 1; got {size}")

    n_pos = int(round(target_prevalence * size))
    n_neg = size - n_pos

    pos_idx = np.flatnonzero(positive_mask)
    neg_idx = np.flatnonzero(~positive_mask)

    if not with_replacement and (len(pos_idx) < n_pos or len(neg_idx) < n_neg):
        return None
    if with_replacement and (len(pos_idx) == 0 and n_pos > 0):
        return None
    if with_replacement and (len(neg_idx) == 0 and n_neg > 0):
        return None

    drawn_pos = rng.choice(pos_idx, size=n_pos, replace=with_replacement) if n_pos else np.array([], dtype="int64")
    drawn_neg = rng.choice(neg_idx, size=n_neg, replace=with_replacement) if n_neg else np.array([], dtype="int64")
    indices = np.concatenate([drawn_pos, drawn_neg]).astype("int64")
    rng.shuffle(indices)

    return Pool(
        indices=indices,
        target_prevalence=float(target_prevalence),
        true_prevalence=float(n_pos / size),
        size=int(size),
        n_positive=int(n_pos),
        with_replacement=bool(with_replacement),
    )


def feasible_prevalences(
    positive_mask: np.ndarray, size: int, grid: np.ndarray
) -> dict[float, bool]:
    """Which grid points this fold can supply at this pool size, without replacement.

    Reported alongside the results so a gap in a bias curve reads as "the data could not reach
    here" rather than as a missing measurement.
    """
    n_pos_avail = int(positive_mask.sum())
    n_neg_avail = int((~positive_mask).sum())
    out = {}
    for p in grid:
        n_pos = int(round(p * size))
        out[float(p)] = n_pos <= n_pos_avail and (size - n_pos) <= n_neg_avail
    return out


def pool_images_mask(specimen_ids_all: np.ndarray, pool_specimen_ids: np.ndarray) -> np.ndarray:
    """Boolean mask over *images* for the specimens in a pool.

    Kept here rather than inlined at the call site because it is the one place where the
    specimen/image distinction is crossed deliberately, and it should be findable. Every image of a
    drawn specimen travels with it -- taking a subset of a shrimp's photographs would change what
    the pool is a sample of.
    """
    return np.isin(specimen_ids_all, pool_specimen_ids)
