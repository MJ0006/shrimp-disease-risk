"""Figures for Stage 5 (backbone comparison) and Stage 6 (Track A risk modelling).

Kept separate from `make_figures.py`, which owns the Stage 3-4 leakage figures. Both read only
from `reports/tables/` and `experiments/runs/`, so re-running an experiment moves its figures
with it and a slide can never disagree with the data behind it.

    python scripts/make_figures_stage56.py

Outputs (reports/figures/):
    fig06_backbone_comparison.png   ranked macro-F1 with per-fold points + significance band
    fig07_deployment_tradeoff.png   accuracy vs model size -- the 'edge-deployable' evidence
    fig08_convnext_correction.png   the learning-rate correction, and the curve that caught it
    fig09_track_a_diagnostics.png   the Just-Pope precondition, and every model vs the baseline
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shrimp_risk.common.paths import FIGURES, RUNS, TABLES  # noqa: E402

LEAKY = "#c44e52"
HONEST = "#4c72b0"
NEUTRAL = "#7f8c8d"
ACCENT = "#dd8452"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
})


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"  wrote reports/figures/{name}")


def fig_backbone_comparison(summary: pd.DataFrame, folds: pd.DataFrame) -> None:
    """Ranked macro-F1, per-fold points, and a band marking what cannot be ranked.

    The per-fold dots matter more than the bars. With ~83 test specimens per fold most gaps
    between backbones are smaller than the fold-to-fold spread, and bars alone would invite the
    reader to rank models this data cannot separate.
    """
    s = summary.sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    xs = list(range(len(s)))

    colours = [HONEST if i == 0 else NEUTRAL for i in xs]
    ax.bar(xs, s.f1_macro_mean, yerr=s.f1_macro_sd, color=colours, width=0.58, capsize=5)

    for i, b in enumerate(s.backbone):
        pts = folds.loc[folds.backbone == b, "test_f1_macro"]
        ax.scatter([i] * len(pts), pts, color="black", zorder=3, s=16, alpha=0.75)
        ax.text(i, s.f1_macro_mean[i] + s.f1_macro_sd[i] + 0.028,
                f"{s.f1_macro_mean[i]:.3f}", ha="center", fontweight="bold", fontsize=10)

    top, sd = s.f1_macro_mean.iloc[0], s.f1_macro_sd.iloc[0]
    ax.axhspan(top - sd, top, color=HONEST, alpha=0.10)
    ax.text(len(s) - 0.45, top - sd / 2,
            "within 1 SD of the leader:\nnot a supported ranking",
            ha="right", va="center", fontsize=9, style="italic", color=HONEST)

    ax.set_xticks(xs)
    ax.set_xticklabels([b.replace("_", "\n") for b in s.backbone], fontsize=8.5)
    ax.set_ylabel("test macro-F1")
    ax.set_ylim(0, 0.95)
    ax.set_title("Backbone comparison under specimen-level 5-fold CV\n"
                 "dots are individual folds; only EfficientNet-B0 is clearly behind")
    _save(fig, "fig06_backbone_comparison.png")


def fig_deployment_tradeoff(summary: pd.DataFrame) -> None:
    """Accuracy against model size -- the evidence behind the title's 'edge-deployable' claim.

    Size is the x-axis rather than latency because latency measured on a discrete laptop GPU
    varied 40% between runs and does not transfer to edge hardware. Size is exact and portable.
    """
    s = summary.dropna(subset=["size_mb"]).copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    ax.errorbar(s.size_mb, s.f1_macro_mean, yerr=s.f1_macro_sd, fmt="o", ms=10,
                color=NEUTRAL, capsize=4, zorder=3)

    for _, r in s.iterrows():
        pick = r.backbone == "mobilenetv3_large_100"
        ax.annotate(r.backbone.replace("_large_100", "").replace("_", " "),
                    (r.size_mb, r.f1_macro_mean),
                    textcoords="offset points", xytext=(9, 8), fontsize=9,
                    fontweight="bold" if pick else "normal",
                    color=HONEST if pick else "black")
        if pick:
            ax.scatter([r.size_mb], [r.f1_macro_mean], s=300, facecolors="none",
                       edgecolors=HONEST, linewidths=2.4, zorder=4)

    ax.axvline(2.82, color=ACCENT, linestyle=":", lw=1.8)
    ax.text(3.0, s.f1_macro_mean.min() - 0.03, "FeatherNetX\n2.82 MB (published)",
            fontsize=8.5, color=ACCENT)

    ax.set_xscale("log")
    ax.set_xlabel("model size (MB, log scale) — smaller is more deployable")
    ax.set_ylabel("test macro-F1")
    ax.set_ylim(s.f1_macro_mean.min() - 0.06, s.f1_macro_mean.max() + 0.06)
    ax.set_title("Accuracy vs model size\n"
                 "MobileNetV3 is 6.6x smaller than ConvNeXt for a gap barely above fold noise")
    _save(fig, "fig07_deployment_tradeoff.png")


def fig_convnext_correction(folds: pd.DataFrame, old_path: Path) -> None:
    """The correction, and the diagnostic that caught it.

    Left panel comes from the two fold tables. Right panel plots the CORRECTED run's training
    curves: the original run's per-epoch histories were overwritten when the re-run reused the
    same filenames, so they are no longer plottable. Their summary values survive in
    `exp04_folds_lr3e-4_all.csv` and in the Stage 5 document; the loss of the per-epoch detail
    is recorded there as a traceability lesson rather than reconstructed from memory.
    """
    old_c = pd.read_csv(old_path).query("backbone == 'convnext_tiny'")["test_f1_macro"]
    new_c = folds.query("backbone == 'convnext_tiny'")["test_f1_macro"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.7))

    means = [old_c.mean(), new_c.mean()]
    sds = [old_c.std(), new_c.std()]
    ax1.bar(["lr = 3e-4\n(original)", "lr = 5e-5\n(corrected)"], means, yerr=sds,
            color=[LEAKY, HONEST], width=0.5, capsize=6)
    for i, (m, sd, pts) in enumerate(zip(means, sds, [old_c, new_c])):
        ax1.scatter([i] * len(pts), pts, color="black", zorder=3, s=22)
        ax1.text(i, m + sd + 0.04, f"{m:.3f}\n± {sd:.3f}", ha="center", fontweight="bold")
    ax1.set_ylabel("test macro-F1")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("ConvNeXt-Tiny: last place to first\nafter one learning-rate change")

    for f in sorted(glob.glob(str(RUNS / "exp04" / "convnext_tiny__fold*.json"))):
        h = json.load(open(f, encoding="utf-8"))["history"]
        ax2.plot([e["epoch"] for e in h], [e["train_f1"] for e in h],
                 marker="o", ms=2.5, lw=1.3, alpha=0.85)
    ax2.axhline(0.25, color=LEAKY, ls="--", lw=1.3)
    ax2.text(1, 0.28, "chance for 4 classes", fontsize=8.5, color=LEAKY)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("TRAINING macro-F1")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Corrected run: all 5 folds now fit the training data\n"
                  "at lr 3e-4, three folds never rose above ~0.38")
    _save(fig, "fig08_convnext_correction.png")


def fig_track_a(diagnostics: pd.DataFrame, candidates: pd.DataFrame) -> None:
    """Two panels: the Just-Pope precondition, then every candidate against the mean baseline."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.7))

    d = diagnostics.set_index("model")
    models = list(d.index)
    bp = [float(d.loc[m, "breusch_pagan_p"]) for m in models]
    ax1.bar(models, bp, color=[HONEST if v < 0.05 else LEAKY for v in bp], width=0.45)
    ax1.axhline(0.05, color="black", ls="--", lw=1.3)
    # Anchored to the left edge: at the right edge it overran into the adjacent subplot.
    ax1.text(-0.42, 0.062, "p = 0.05", ha="left", fontsize=9)
    for i, v in enumerate(bp):
        ax1.text(i, v + 0.008, f"{v:.3g}", ha="center", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Breusch-Pagan p-value")
    ax1.set_ylim(0, 0.26)
    ax1.set_title("The Just-Pope precondition\n"
                  "income: variance depends on inputs · disease: it does not")

    piv = candidates.pivot_table(index="model", columns="risk_model",
                                 values="holdout_rmse_vs_mean_baseline")
    piv = piv.sort_values(piv.columns[0])
    xs = list(range(len(piv)))
    w = 0.38
    for k, col in enumerate(piv.columns):
        ax2.bar([i + (k - 0.5) * w for i in xs], piv[col], width=w, label=str(col),
                color=[LEAKY, NEUTRAL][k % 2])
    ax2.axhline(1.0, color="black", ls="--", lw=1.6)
    ax2.text(len(piv) - 0.5, 1.008, "= predicting the mean", ha="right", fontsize=9)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(list(piv.index), fontsize=9, rotation=12)
    ax2.set_ylabel("holdout RMSE ÷ mean baseline")
    ax2.set_ylim(0.92, max(1.30, float(piv.to_numpy().max()) + 0.04))
    ax2.legend(frameon=False, fontsize=9, title="risk model")
    ax2.set_title("Every candidate is worse than the mean\nbars above the line = no predictive skill")
    _save(fig, "fig09_track_a_diagnostics.png")


def main() -> int:
    print("Building Stage 5 and Stage 6 figures...")

    e4s, e4f = TABLES / "exp04_summary.csv", TABLES / "exp04_folds.csv"
    if e4s.exists() and e4f.exists():
        s4, f4 = pd.read_csv(e4s), pd.read_csv(e4f)
        fig_backbone_comparison(s4, f4)
        fig_deployment_tradeoff(s4)
        old = TABLES / "exp04_folds_lr3e-4_all.csv"
        if old.exists():
            fig_convnext_correction(f4, old)
        else:
            print("  superseded ConvNeXt run absent; skipping fig08")
    else:
        print("  exp04 tables missing; run scripts/exp04_backbone_comparison.py first")

    e3d, e3c = TABLES / "exp03_diagnostics.csv", TABLES / "exp03_step3_candidates.csv"
    if e3d.exists() and e3c.exists():
        fig_track_a(pd.read_csv(e3d), pd.read_csv(e3c))
    else:
        print("  exp03 tables missing; run scripts/exp03_track_a.py first")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
