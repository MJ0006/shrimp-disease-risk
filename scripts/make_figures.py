"""Generate every report figure from the manifest and the experiment artefacts.

Why this is a script
--------------------
A chart produced once in a throwaway cell and then hand-edited is a claim nobody can re-check --
and it goes stale silently, which is worse, because a stale chart still looks authoritative next to
a corrected table. Running this script rebuilds the whole figure set from
`data/interim/image_manifest.csv`, `reports/tables/exp0*.csv` and the per-run JSONs, so if an
experiment is ever re-run the figures move with it instead of quietly going stale.

    python scripts/make_figures.py

Outputs (reports/figures/):
    fig01_specimen_structure.png   1,149 images vs 416 shrimp; per class; group sizes
    fig02_leakage_exposure.png     how much of the test set has a training sibling
    fig03_headline_result.png      the result: macro-F1 under each protocol, per-seed points
    fig04_per_class_drop.png       which classes leakage flatters most, ordered by specimens
    fig05_generalisation_gap.png   train-vs-validation curves under both protocols
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write straight to file
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shrimp_risk.common.paths import FIGURES, IMAGE_MANIFEST, RUNS, TABLES  # noqa: E402

# Ordered by specimen count, descending -- the ordering that makes the per-class story legible.
CLASS_ORDER = ["Healthy", "WSSV", "BG_WSSV", "BG"]
SPECIMENS_PER_CLASS = {"Healthy": 142, "WSSV": 118, "BG_WSSV": 81, "BG": 75}

# One colour per role, held constant across every figure so the reader learns it once.
LEAKY = "#c44e52"    # image-level / contaminated
HONEST = "#4c72b0"   # specimen-level / corrected
NEUTRAL = "#7f8c8d"
ACCENT = "#dd8452"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"  wrote reports/figures/{name}")


def _label_bars(ax, xs, ys, fmt="{:,}", dy=0.02):
    span = max(ys) if len(ys) else 1
    for x, y in zip(xs, ys):
        ax.text(x, y + span * dy, fmt.format(y), ha="center", fontweight="bold", fontsize=10)


# --------------------------------------------------------------------------------------
# Figure 1 -- the structural fact the project rests on
# --------------------------------------------------------------------------------------

def fig_specimen_structure(raw: pd.DataFrame) -> None:
    images = raw["class"].value_counts().reindex(CLASS_ORDER)
    specimens = raw.groupby("class")["specimen_id"].nunique().reindex(CLASS_ORDER)
    groups = raw.groupby("specimen_id").size().value_counts().sort_index()

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax = axes[0]
    ax.bar(["Images", "Individual\nshrimp"], [len(raw), raw["specimen_id"].nunique()],
           color=[NEUTRAL, HONEST], width=0.55)
    _label_bars(ax, [0, 1], [len(raw), raw["specimen_id"].nunique()])
    ax.set_ylabel("count")
    ax.set_ylim(0, len(raw) * 1.18)
    ax.set_title(f"The dataset is {raw['specimen_id'].nunique()} shrimp,\nnot {len(raw):,} observations")

    ax = axes[1]
    x = range(len(CLASS_ORDER))
    ax.bar([i - 0.2 for i in x], images, width=0.4, label="images", color=NEUTRAL)
    ax.bar([i + 0.2 for i in x], specimens, width=0.4, label="specimens", color=HONEST)
    for i, (im, sp) in enumerate(zip(images, specimens)):
        ax.text(i - 0.2, im + 12, str(im), ha="center", fontsize=9)
        ax.text(i + 0.2, sp + 12, str(sp), ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(CLASS_ORDER, fontsize=9)
    ax.set_ylim(0, images.max() * 1.2)
    ax.set_title("Per class: imbalance is\nmoderate in images, sharper in shrimp")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    ax.bar(groups.index.astype(str), groups.to_numpy(), color=ACCENT, width=0.55)
    _label_bars(ax, range(len(groups)), groups.to_numpy(), fmt="{}")
    ax.set_xlabel("photographs of the same shrimp")
    ax.set_ylabel("number of shrimp")
    ax.set_ylim(0, groups.max() * 1.18)
    ax.set_title(f"{groups.get(3, 0)} of {raw['specimen_id'].nunique()} shrimp\nwere photographed 3 times")

    _save(fig, "fig01_specimen_structure.png")


# --------------------------------------------------------------------------------------
# Figure 2 -- exposure (Experiment 01)
# --------------------------------------------------------------------------------------

def fig_leakage_exposure(exp01: pd.DataFrame) -> None:
    e = exp01.set_index("protocol")
    means = [e.loc["image_level", "pct_test_leaked_mean"], e.loc["specimen_level", "pct_test_leaked_mean"]]
    sds = [e.loc["image_level", "pct_test_leaked_sd"], e.loc["specimen_level", "pct_test_leaked_sd"]]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(["Image-level\n(conventional)", "Specimen-level\n(corrected)"], means, yerr=sds,
           color=[LEAKY, HONEST], width=0.5, capsize=5)
    ax.text(0, means[0] + 4, f"{means[0]:.2f}%", ha="center", fontweight="bold")
    ax.text(1, 4, f"{means[1]:.2f}%", ha="center", fontweight="bold")
    ax.set_ylabel("% of test images with a training sibling")
    ax.set_ylim(0, 105)
    ax.set_title("Seven of every eight test images has a photo\nof the same shrimp in training")
    _save(fig, "fig02_leakage_exposure.png")


# --------------------------------------------------------------------------------------
# Figure 3 -- the headline result (Experiment 02)
# --------------------------------------------------------------------------------------

def fig_headline_result(summary: pd.DataFrame, runs: pd.DataFrame) -> None:
    order = ["A_image_level_noaug", "B_image_level", "C_specimen_level"]
    labels = ["Image-level\n(no augmentation)", "Image-level\n(augmented)", "Specimen-level\n(augmented)"]
    s = summary.set_index("condition")
    means = [s.loc[c, "test_f1_macro_mean"] for c in order]
    sds = [s.loc[c, "test_f1_macro_sd"] for c in order]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.bar(labels, means, yerr=sds, color=[NEUTRAL, LEAKY, HONEST], width=0.55, capsize=5)

    # Individual seeds as points: three runs are not a distribution, and drawing them keeps
    # the reader from reading the error bar as one.
    for i, cond in enumerate(order):
        pts = runs.loc[runs.condition == cond, "test_f1_macro"]
        ax.scatter([i] * len(pts), pts, color="black", zorder=3, s=18)

    for i, (m, sd) in enumerate(zip(means, sds)):
        ax.text(i, m + sd + 0.035, f"{m:.3f}", ha="center", fontweight="bold")

    gap = means[1] - means[2]
    ax.annotate("", xy=(2, means[2]), xytext=(1, means[1]),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="black"))
    ax.text(1.5, (means[1] + means[2]) / 2 + 0.06, f"−{gap:.3f}\nmacro-F1",
            ha="center", color=LEAKY, fontweight="bold", fontsize=10)

    ax.set_ylabel("test macro-F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Removing specimen leakage costs 15 points of macro-F1\n"
                 "(rexnet_150, 3 seeds, all else held constant)")
    _save(fig, "fig03_headline_result.png")


# --------------------------------------------------------------------------------------
# Figure 4 -- per class
# --------------------------------------------------------------------------------------

def fig_per_class_drop(runs: pd.DataFrame) -> None:
    b = runs[runs.condition == "B_image_level"]
    c = runs[runs.condition == "C_specimen_level"]
    bl = [b[f"f1_{k}"].mean() for k in CLASS_ORDER]
    cl = [c[f"f1_{k}"].mean() for k in CLASS_ORDER]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = range(len(CLASS_ORDER))
    ax.bar([i - 0.2 for i in x], bl, width=0.4, label="image-level (leaky)", color=LEAKY)
    ax.bar([i + 0.2 for i in x], cl, width=0.4, label="specimen-level (honest)", color=HONEST)
    for i, (hi, lo) in enumerate(zip(bl, cl)):
        ax.text(i + 0.2, lo + 0.03, f"{lo - hi:+.2f}", ha="center", fontweight="bold", fontsize=10)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{k}\n{SPECIMENS_PER_CLASS[k]} specimens" for k in CLASS_ORDER], fontsize=9)
    ax.set_ylabel("per-class F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fewest individuals, largest fall:\nBlack Gill drops 0.27 once the shrimp is unseen")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    _save(fig, "fig04_per_class_drop.png")


# --------------------------------------------------------------------------------------
# Figure 5 -- the generalisation gap
# --------------------------------------------------------------------------------------

def fig_generalisation_gap(seed: int = 2025) -> None:
    """Train and validation curves side by side under both protocols.

    This figure is the direct answer to ShrimpXNet's argument that its model "generalizes well
    and is not overfitted" because train and validation curves nearly overlap. Under an
    image-level split BOTH partitions contain the same specimens, so overlapping curves are the
    expected signature of leakage -- not evidence against overfitting. Once the split is
    specimen-level, the true gap appears.
    """
    pairs = [
        ("B_image_level", "Image-level split (leaky)", LEAKY),
        ("C_specimen_level", "Specimen-level split (honest)", HONEST),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), sharey=True)

    for ax, (tag, title, colour) in zip(axes, pairs):
        path = RUNS / "exp02" / f"{tag}__seed{seed}.json"
        if not path.exists():
            ax.set_visible(False)
            continue
        history = json.loads(path.read_text(encoding="utf-8"))["history"]
        epochs = [h["epoch"] for h in history]
        train = [h["train_f1"] for h in history]
        val = [h["val_f1"] for h in history]

        ax.plot(epochs, train, "-o", ms=3.5, color=colour, label="train macro-F1")
        ax.plot(epochs, val, "--s", ms=3.5, color=colour, alpha=0.6, label="validation macro-F1")
        ax.fill_between(epochs, val, train, color=colour, alpha=0.12)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, loc="lower right", fontsize=9)
        ax.set_title(f"{title}\nfinal train−val gap ≈ {train[-1] - val[-1]:.2f}")

    axes[0].set_ylabel("macro-F1")
    _save(fig, "fig05_generalisation_gap.png")


def main() -> int:
    manifest = pd.read_csv(IMAGE_MANIFEST)
    raw = manifest[manifest["subset"] == "raw"]

    print("Building figures...")
    fig_specimen_structure(raw)

    exp01, exp02s, exp02r = (
        TABLES / "exp01_leakage_summary.csv",
        TABLES / "exp02_summary.csv",
        TABLES / "exp02_runs.csv",
    )
    if not (exp01.exists() and exp02s.exists() and exp02r.exists()):
        print("  experiment tables missing; run scripts/exp01_*.py and exp02_*.py first")
        return 0

    runs, summary = pd.read_csv(exp02r), pd.read_csv(exp02s)
    fig_leakage_exposure(pd.read_csv(exp01))
    fig_headline_result(summary, runs)
    fig_per_class_drop(runs)
    fig_generalisation_gap()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
