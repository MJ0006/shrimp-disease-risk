"""Figures for Stage 8 (the prevalence bridge) and Stage 9 (sensitivity).

Kept separate from `make_figures.py` (Stages 3-4) and `make_figures_stage56.py` (Stages 5-6) for
the same reason those two are separate: each script owns the figures for the experiments it reads,
so re-running an experiment moves its figures with it and a slide cannot disagree with the table
behind it (`D-014`).

    python scripts/make_figures_stage89.py

Outputs (reports/figures/):
    fig10_calibration.png       reliability before and after temperature scaling, per fold
    fig11_bias_curves.png       estimator error against true prevalence -- the bridge's evidence
    fig12_estimator_ranking.png mean |bias| per estimator, raw against calibrated
    fig13_interval_coverage.png measured coverage of the '95%' bootstrap interval
    fig14_attenuation.png       coefficient decay as measurement error grows, vs the formula
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shrimp_risk.common.paths import FIGURES, TABLES  # noqa: E402

# Same palette as the Stage 5-6 figures: the deck reads as one document, and the colours already
# carry a meaning a reader has learned by the time these slides appear -- red is the flawed thing,
# blue is the corrected one.
LEAKY = "#c44e52"
HONEST = "#4c72b0"
NEUTRAL = "#7f8c8d"
ACCENT = "#dd8452"
GREEN = "#55a868"

ESTIMATOR_COLOURS = {
    "CC": LEAKY, "PCC": ACCENT, "ACC": GREEN, "PACC": HONEST, "SLD-EM": "#8172b3",
}

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


def _table(name: str) -> pd.DataFrame | None:
    path = TABLES / name
    if not path.exists():
        print(f"  skipped: {name} not found -- run the experiment that writes it")
        return None
    return pd.read_csv(path)


def fig_calibration(reliability: pd.DataFrame, calib: pd.DataFrame) -> None:
    """Reliability diagram before and after, plus the per-fold ECE the target is judged against.

    The diagonal is what calibration means: of the predictions made at 80% confidence, 80% should
    be correct. Points below it are overconfident, which is the direction `D-028` measured and the
    direction that biases a prevalence estimate built on these posteriors.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", label="perfect calibration")
    for state, colour, label in (("before", LEAKY, "raw posteriors"),
                                 ("after", HONEST, "temperature-scaled")):
        rows = reliability[reliability.state == state]
        # Bins are pooled across folds and backbones by population weight: a bin holding 4
        # predictions and one holding 300 must not count equally toward the curve.
        agg = rows.groupby("bin_lo").apply(
            lambda g: pd.Series({
                "conf": np.average(g["mean_confidence"].fillna(0), weights=g["n"].clip(lower=0))
                if g["n"].sum() else np.nan,
                "acc": np.average(g["accuracy"].fillna(0), weights=g["n"].clip(lower=0))
                if g["n"].sum() else np.nan,
                "n": g["n"].sum(),
            }), include_groups=False
        ).reset_index()
        agg = agg[agg["n"] > 0]
        ax.plot(agg["conf"], agg["acc"], marker="o", ms=4, color=colour, label=label)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability: does 0.9 confidence mean 90% correct?")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1]
    width = 0.38
    xs = np.arange(len(calib))
    ax.bar(xs - width / 2, calib["ece_before"], width, color=LEAKY, label="before")
    ax.bar(xs + width / 2, calib["ece_after"], width, color=HONEST, label="after")
    ax.axhline(0.05, color="black", lw=1.2, ls=":",
               label="D-028 target (ECE $\\leq$ 0.05)")
    # Short codes, not truncated backbone names. `b.split('_')[0]` gave "convnext" and
    # "mobilenetv3", which are long enough to overlap into an unreadable smear across ten bars.
    # The mapping goes in the axis label, so nothing is lost by shortening.
    short = {"convnext_tiny": "CN", "mobilenetv3_large_100": "MN"}
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{short.get(b, b[:3])}-{f}"
                        for b, f in zip(calib["backbone"], calib["fold"])], fontsize=9)
    ax.set_xlabel("CN = convnext_tiny,  MN = mobilenetv3_large_100,  suffix = fold", fontsize=9)
    ax.set_ylabel("expected calibration error")
    ax.set_title("Every run, against the pre-registered target")
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Stage 8a -- calibration is fitted on validation and judged on test",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig10_calibration.png")


def fig_bias_curves(bias: pd.DataFrame) -> None:
    """Error against true prevalence, per backbone and per calibration state.

    The shape is the point. A biased estimator's error is not a constant offset -- it depends on the
    true prevalence, crossing zero wherever the classifier's false positives happen to cancel its
    false negatives. That single crossing point near 0.55 is why reporting an estimator's accuracy
    at one prevalence proves nothing.

    **Both calibration states are drawn**, because the comparison between the rows is itself a
    finding: temperature scaling makes SLD-EM worse, not better, reversing what `D-028` expected.
    An earlier version of this figure showed only the calibrated row, in which SLD-EM is *not* the
    best estimator -- so the headline result was invisible in the figure meant to carry it.
    """
    size = bias.pool_size.max()
    sub = bias[bias.pool_size == size]
    if sub.empty:
        print("  skipped fig11: no rows at the largest pool size")
        return
    backbones = sorted(sub.backbone.unique())
    states = ["raw", "calibrated"]

    fig, axes = plt.subplots(len(states), len(backbones),
                             figsize=(6.0 * len(backbones), 4.4 * len(states)),
                             squeeze=False, sharex=True)

    for r, state in enumerate(states):
        for c, backbone in enumerate(backbones):
            ax = axes[r][c]
            rows = sub[(sub.backbone == backbone) & (sub.calibration == state)]
            for name, colour in ESTIMATOR_COLOURS.items():
                g = (rows[rows.estimator == name]
                     .groupby("true_prevalence")["bias"].agg(["mean", "std"]).reset_index())
                if g.empty:
                    continue
                ax.plot(g["true_prevalence"], g["mean"], marker="o", ms=4,
                        color=colour, label=name)
                ax.fill_between(g["true_prevalence"], g["mean"] - g["std"],
                                g["mean"] + g["std"], color=colour, alpha=0.10)
            ax.axhline(0, color="black", lw=1.2)
            ax.set_ylabel("estimated $-$ true")
            ax.set_title(f"{backbone} — {state} posteriors", fontsize=11)
            if r == len(states) - 1:
                ax.set_xlabel("true prevalence in the pool")
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=8.5, ncol=2)

    fig.suptitle(
        f"Stage 8 — counting is biased, and the bias depends on the prevalence being measured "
        f"(pools of {int(size)} specimens)",
        fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig11_bias_curves.png")


def fig_estimator_ranking(estimators: pd.DataFrame) -> None:
    """Mean absolute bias per estimator, raw against calibrated.

    Two questions on one axis: does correction help, and does calibration help the correction?
    `D-028` made the second a prerequisite rather than a refinement, so it has to be visible.
    """
    order = ["CC", "PCC", "ACC", "PACC", "SLD-EM"]
    sizes = sorted(estimators.pool_size.unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(6.0 * len(sizes), 4.7), squeeze=False)

    for ax, size in zip(axes[0], sizes):
        rows = estimators[estimators.pool_size == size]
        xs = np.arange(len(order))
        width = 0.38
        for offset, state, colour in ((-width / 2, "raw", LEAKY),
                                      (width / 2, "calibrated", HONEST)):
            vals = [rows.loc[(rows.estimator == e) & (rows.calibration == state),
                             "mean_abs_bias"].mean() for e in order]
            ax.bar(xs + offset, vals, width, color=colour, label=state)
        ax.set_xticks(xs)
        ax.set_xticklabels(order)
        ax.set_ylabel("mean |bias| over the prevalence grid")
        ax.set_title(f"pool = {int(size)} specimens")
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Stage 8 -- which correction, and does calibration matter to it?",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig12_estimator_ranking.png")


def fig_coverage(coverage: pd.DataFrame) -> None:
    """Measured coverage against the nominal 95%.

    An interval is only a 95% interval if it contains the truth 95% of the time. This project has
    no business printing "95% CI" next to a prevalence until this figure says it may.
    """
    fig, ax = plt.subplots(figsize=(9.0, 4.7))
    estimators = sorted(coverage.estimator.unique())
    xs = np.arange(len(estimators))
    offsets = {"raw": -0.10, "calibrated": 0.10}

    # Dots on a zoomed axis, not bars from zero. Every value here sits between 0.85 and 0.96, and
    # bars anchored at zero compress that range into visual noise -- the differences that matter
    # are 8 percentage points of coverage. Truncating a *bar* axis would be misleading, because a
    # bar's length encodes the value; a dot encodes only its position, so a non-zero baseline is
    # legitimate. The axis starts at 0.75 and the nominal line is drawn, so nothing is hidden.
    for state, colour in (("raw", LEAKY), ("calibrated", HONEST)):
        vals = [coverage.loc[(coverage.estimator == e) & (coverage.calibration == state),
                             "coverage"].mean() for e in estimators]
        ax.scatter(xs + offsets[state], vals, s=110, color=colour, zorder=3, label=state)
        for x, v in zip(xs + offsets[state], vals):
            ax.plot([x, x], [0.75, v], color=colour, lw=1.6, alpha=0.45, zorder=2)
            ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=9, color=colour)

    nominal = float(coverage["nominal"].iloc[0])
    ax.axhline(nominal, color="black", lw=1.3, ls=":", label=f"nominal {nominal:.0%}")
    ax.set_xticks(xs)
    ax.set_xticklabels(estimators)
    ax.set_xlim(-0.5, len(estimators) - 0.5)
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("share of pools whose interval contained the truth")
    ax.set_title("Stage 8 -- is the bootstrap interval worth its label?")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, "fig13_interval_coverage.png")


def fig_attenuation(attenuation: pd.DataFrame) -> None:
    """The Stage 9 result: how much of the disease coefficient survives measurement error.

    Plotted for the specification whose baseline coefficient is distinguishable from zero. Where
    it is not, there is no effect to attenuate and the ratio is suppressed in the table -- so it
    is not drawn here either, and the panel says why instead of showing a curve made of noise.
    """
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    drawn = False

    for spec, colour in (("S2_bridge_facing", HONEST), ("S1_published", NEUTRAL)):
        rows = attenuation[attenuation.specification == spec].sort_values("magnitude")
        if rows.empty:
            continue
        if not bool(rows["ratio_interpretable"].iloc[0]):
            base = rows["baseline_estimate"].iloc[0]
            p = rows["baseline_p"].iloc[0]
            ax.plot([], [], " ", label=f"{spec}: no effect to attenuate "
                                       f"(coef {base:.3f}, p = {p:.2f})")
            continue
        ax.plot(rows["magnitude"], rows["attenuation_observed"], marker="o", ms=5,
                color=colour, label=f"{spec}: measured")
        ax.plot(rows["magnitude"], rows["predicted_attenuation"], marker="s", ms=4,
                color=colour, ls="--", alpha=0.65,
                label=f"{spec}: classical prediction")

        # The deployed error magnitudes are the point of the whole figure and were previously
        # indistinguishable from the synthetic grid points around them. Without this band a reader
        # sees a curve that falls to 0.55 and concludes measurement error is devastating; the
        # finding is that this pipeline operates in the flat part, to the LEFT of the band's right
        # edge. The labels name which estimator each magnitude came from.
        measured = rows[rows["level_label"].astype(str).str.startswith("measured")]
        if len(measured):
            lo, hi = measured["magnitude"].min(), measured["magnitude"].max()
            ax.axvspan(lo, hi, color=GREEN, alpha=0.13, zorder=0)
            ax.annotate(
                "error this pipeline\nactually produces\n"
                + ", ".join(f"{r.level_label.replace('measured: ', '')} {r.magnitude:.3f}"
                            for r in measured.itertuples()),
                xy=((lo + hi) / 2, 0.62), ha="center", va="center", fontsize=8.5,
                color="#2f6b46",
            )
        drawn = True

    ax.axhline(1.0, color="black", lw=1.2)
    ax.set_xlabel("measurement error in the prevalence regressor (SD, prevalence units)")
    ax.set_ylabel("coefficient / unperturbed coefficient")
    # The earlier title asserted the conclusion the data does not support at the magnitudes that
    # matter. Attenuation is real above SD ~ 0.15 and absent where this pipeline operates, and the
    # title has to say both or it argues against the stage's own result.
    ax.set_title("Stage 9 — attenuation is real above SD $\\approx$ 0.15, and absent "
                 "where this pipeline operates")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    if not drawn:
        ax.text(0.5, 0.5, "no specification had an effect to attenuate",
                transform=ax.transAxes, ha="center", fontsize=11, color=NEUTRAL)
    fig.tight_layout()
    _save(fig, "fig14_attenuation.png")


def main() -> int:
    print("Stage 8-9 figures")

    reliability = _table("exp06_reliability.csv")
    calib = _table("exp06_calibration.csv")
    if reliability is not None and calib is not None:
        fig_calibration(reliability, calib)

    bias = _table("exp07_bias.csv")
    if bias is not None:
        fig_bias_curves(bias)

    estimators = _table("exp07_estimators.csv")
    if estimators is not None:
        fig_estimator_ranking(estimators)

    coverage = _table("exp07_coverage.csv")
    if coverage is not None:
        fig_coverage(coverage)

    attenuation = _table("exp08_attenuation.csv")
    if attenuation is not None:
        fig_attenuation(attenuation)

    print(f"\nfigures -> {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
