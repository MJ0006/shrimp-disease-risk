"""Partial dependence: turning a step-3 model back into something interpretable.

Why partial dependence is the point rather than a decoration
------------------------------------------------------------
The base paper's contribution is not that machine learning predicts risk. It is that a Just-Pope
econometric model gives you the *direction* of an effect with a standard error, a machine-learning
model gives you its *shape* without imposing one, and partial dependence plots are what let you
read the shape off the second. Several of the paper's substantive findings exist only in the PDPs
and are inexpressible as coefficients -- experience falling then rising against disease risk,
stocking density with an interior minimum near 75 pieces/m2, income risk rising with stocking
density then reversing above ~400. A linear term would average each of those to a single slope and
report the reversal as nothing at all.

What a PDP actually computes, and the assumption inside it
----------------------------------------------------------
For a feature `x_j` and grid value `v`, partial dependence is the model's average prediction when
every observation in the data has `x_j` set to `v` and all its other features left alone::

    PD_j(v) = (1/n) * sum_i  f_hat(v, x_i,-j)

That average is over the **marginal** distribution of the other features, not their distribution
conditional on `x_j = v`. Where features are correlated, the average therefore includes
combinations that do not occur in the data -- a 0.05 ha pond with a 20-species polyculture, say --
and the model is being extrapolated into a region it never saw. This is the standard and
well-known limitation of PDP (it is why ICE and ALE plots exist), and it is not fixed by anything
in this module. What this module does instead is make it *visible*: every plot carries a rug of
the actual observed values, and every returned table carries the number of observations near each
grid point. A PDP curve over a region with no data is a model artefact, and the reader is given
what they need to see that.

Design choices
--------------
- **Grid on observed quantiles, not on an equally spaced range.** An equally spaced grid over a
  right-skewed variable such as revenue or stocking cost spends most of its points in a tail that
  holds a handful of farms. Quantile grids put the resolution where the data is. The trade-off is
  that the x-axis is then unevenly spaced, which is stated on the axis label.
- **Values are returned as a tidy table before they are plotted.** Every figure in this project
  must be regenerable from a script (`D-014`), and a figure whose underlying numbers are not also
  written out cannot be checked against anything. The table is the artefact; the PNG is a view of
  it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

#: Number of grid points per feature. Twenty is enough to show a reversal or an interior optimum
#: -- the shapes the base paper reports -- while keeping each curve cheap: a PDP costs one model
#: prediction over the whole dataset per grid point, so 20 points on 1,200 rows is 24,000
#: predictions per feature per model.
GRID_RESOLUTION = 20


def partial_dependence_table(
    model,
    X: pd.DataFrame,
    features: list[str],
    model_name: str,
    response_label: str,
    grid_resolution: int = GRID_RESOLUTION,
) -> pd.DataFrame:
    """Compute one-way partial dependence for several features and return it tidy.

    One row per (model, response, feature, grid point). Alongside the partial dependence value
    each row carries `n_obs_at_or_below` -- the count of observations whose feature value is at
    or below that grid point -- which is what lets a reader tell a curve supported by 400 farms
    from one supported by four. Binary features are handled naturally: the quantile grid collapses
    to the two observed values and the "curve" is the two-point contrast.

    `partial_dependence` is called with `kind="average"`; the per-observation ICE curves are not
    retained because at 1,200 rows x 20 grid points x 4 models x 8 features they would dominate
    the artefact directory without being read. The averaging assumption is documented in the
    module docstring instead.
    """
    rows = []
    for feature in features:
        if feature not in X.columns:
            continue
        col = X[feature].to_numpy(dtype="float64")
        pdp = partial_dependence(
            model, X, [feature], kind="average", grid_resolution=grid_resolution,
            percentiles=(0.02, 0.98),
        )
        grid = np.asarray(pdp["grid_values"][0], dtype="float64")
        values = np.asarray(pdp["average"][0], dtype="float64")
        for g, v in zip(grid, values):
            rows.append({
                "response": response_label,
                "model": model_name,
                "feature": feature,
                "grid_value": float(g),
                "partial_dependence": float(v),
                "n_obs_at_or_below": int((col <= g).sum()),
                "feature_min": float(col.min()),
                "feature_max": float(col.max()),
            })
    out = pd.DataFrame(rows)
    if len(out):
        # Range of the curve: a flat PDP means the model found no dependence, and reporting the
        # range makes a visually dramatic but numerically trivial curve impossible to over-read.
        rng = out.groupby("feature")["partial_dependence"].transform(lambda s: s.max() - s.min())
        out["pd_range"] = rng.round(4)
    return out


def two_way_partial_dependence(
    model,
    X: pd.DataFrame,
    pair: tuple[str, str],
    model_name: str,
    response_label: str,
    grid_resolution: int = 12,
) -> pd.DataFrame:
    """Two-feature partial dependence, returned as a long table.

    The base paper's most-cited results are interaction surfaces (area x stocking density,
    experience x area, income x stocking density). Those cannot be read off two one-way curves,
    because a one-way PDP has already averaged over the other feature and so has averaged the
    interaction away.

    Grid resolution drops to 12 per axis because the cost is quadratic: 12x12 grid points on
    1,200 rows is ~173,000 predictions for a single surface, against 24,000 for a one-way curve.

    **STATUS: available and NOT CALLED anywhere.** `D-017` gates every partial-dependence plot on
    a model beating the mean baseline, and no Track A model ever has -- under random folds or
    cluster-grouped ones (`D-037`). So no one-way curve has been drawn either, and this
    two-way version is waiting on the same gate. It is kept, tested and declared unused rather
    than deleted, because the gate is a decision that could be met by a future model and
    deleting the code would hide what the project is prepared to draw when it is (open item 13).
    """
    pdp = partial_dependence(
        model, X, [tuple(pair)], kind="average", grid_resolution=grid_resolution,
        percentiles=(0.05, 0.95),
    )
    gx = np.asarray(pdp["grid_values"][0], dtype="float64")
    gy = np.asarray(pdp["grid_values"][1], dtype="float64")
    surface = np.asarray(pdp["average"][0], dtype="float64")
    rows = []
    for i, xv in enumerate(gx):
        for j, yv in enumerate(gy):
            rows.append({
                "response": response_label, "model": model_name,
                "feature_x": pair[0], "feature_y": pair[1],
                "grid_x": float(xv), "grid_y": float(yv),
                "partial_dependence": float(surface[i, j]),
            })
    return pd.DataFrame(rows)


def plot_partial_dependence(
    table: pd.DataFrame,
    X: pd.DataFrame,
    out_path,
    title: str,
    subtitle: str = "",
    labels: dict[str, str] | None = None,
    ncols: int = 3,
) -> None:
    """Render a one-way PDP table as a grid of small multiples.

    Deliberate choices in the rendering, each of which is about honesty rather than appearance:

    - **A rug of observed values under every panel.** Without it a PDP curve looks equally
      authoritative everywhere, including over ranges containing three farms. The rug is the
      cheapest available correction for the module docstring's extrapolation problem.
    - **Every panel shares the y-axis.** The response is log-deviance on one common scale, so a
      panel whose curve is nearly flat should *look* nearly flat. Free y-axes would make a
      variable with a 0.05 range look as consequential as one with a 3.0 range, which is the most
      common way a PDP grid misleads.
    - **The y-axis is labelled in log-deviance units and the caption says so.** A reader who
      thinks the axis is income, or a percentage, will draw a completely wrong conclusion about
      magnitude.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = list(dict.fromkeys(table["feature"]))
    if not features:
        return
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows), squeeze=False)

    lo = float(table["partial_dependence"].min())
    hi = float(table["partial_dependence"].max())
    pad = 0.06 * max(hi - lo, 1e-9)

    for ax, feature in zip(axes.ravel(), features):
        sub = table[table["feature"] == feature].sort_values("grid_value")
        ax.plot(sub["grid_value"], sub["partial_dependence"], lw=2, color="#1f4e79")
        obs = X[feature].to_numpy(dtype="float64")
        ax.plot(obs, np.full_like(obs, lo - pad * 0.5), "|", ms=5, alpha=0.15, color="#444444")
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title((labels or {}).get(feature, feature), fontsize=9)
        ax.set_xlabel("observed range (quantile grid)", fontsize=7)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)

    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    for row in axes:
        row[0].set_ylabel("partial dependence\nof log(deviance)", fontsize=8)

    fig.suptitle(title, fontsize=12, y=0.995)
    if subtitle:
        fig.text(0.5, 0.965, subtitle, ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_two_way(
    table: pd.DataFrame,
    out_path,
    title: str,
    subtitle: str = "",
    labels: dict[str, str] | None = None,
) -> None:
    """Render a two-feature partial dependence surface as a filled contour plot.

    A contour rather than a 3-D surface: a 3-D projection makes the height at any given point
    unreadable, and the quantity a reader needs from an interaction plot is *where* risk is high,
    not a decorative shape. Contour levels are linear in the response, so equal spacing between
    contour lines means an equal change in log-deviance.

    **STATUS: available and NOT CALLED anywhere**, for the same reason as
    `two_way_partial_dependence` -- `D-017`'s gate has never opened (open item 13).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fx = str(table["feature_x"].iloc[0])
    fy = str(table["feature_y"].iloc[0])
    piv = table.pivot_table(index="grid_y", columns="grid_x", values="partial_dependence")

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    cs = ax.contourf(piv.columns.to_numpy(), piv.index.to_numpy(), piv.to_numpy(),
                     levels=12, cmap="YlOrRd")
    ax.contour(piv.columns.to_numpy(), piv.index.to_numpy(), piv.to_numpy(),
               levels=12, colors="k", linewidths=0.3, alpha=0.4)
    fig.colorbar(cs, ax=ax, label="partial dependence of log(deviance)")
    ax.set_xlabel((labels or {}).get(fx, fx))
    ax.set_ylabel((labels or {}).get(fy, fy))
    ax.set_title(title, fontsize=11)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center",
                fontsize=8, color="#444444")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
