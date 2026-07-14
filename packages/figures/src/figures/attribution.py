"""The movement-attribution figures: where the classes move, and what carries the move (archived).

Archived. These figures render the refit-era ``attribute`` stage: the per-stratum re-estimated
membership churn and the mover-versus-stayer contrast. They are kept for the
:doc:`refit pilot </packages/analysis/archive/tracking-the-classes-across-strata>` page. The
single-fit category attribution ($H_0^F$) is now drawn by
:mod:`figures.category_decomposition` and :mod:`figures.dense_features`.

Both figures are built from an ``attribute`` run and are general renderers of its tables, not
tied to any particular result: the classes, strata, categories, and features are read from the
data, so the same code draws either axis and any run.

- :func:`attribution_figure` is the two-panel summary. Panel A is a heatmap of each class's
  churn (the fraction of its membership that changed, one minus the Jaccard overlap) across the
  strata of the axis, with a box around the cells where the membership reorganised. Panel B
  stacks each class's centroid shift by literature category, pooled across strata, so the panel
  shows which kinds of feature carry each class's movement.
- :func:`mover_contrast_figure` is the companion. One panel per class shows, at the stratum where
  the class churns most, the features that most distinguish the probands that changed class from
  the stable core (a signed standardised mean difference), so a movement reads down to the
  features that mark who moved.

The figures are descriptive: they open up an already-measured drift. They do not test it.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from figures import style

# Fixed category-to-colour map, so a category keeps its colour across panels and figures. The
# seven literature categories take the palette in a stable order; anything else is neutral grey.
_CATEGORY_ORDER: tuple[str, ...] = (
    "developmental",
    "social/communication",
    "restricted/repetitive",
    "disruptive behavior",
    "attention",
    "anxiety/mood",
    "self-injury",
)
_OTHER_COLOUR = "#BBBBBB"


def _natural_key(label: str) -> list:
    """Sort key that orders ``Q2`` before ``Q10`` by reading embedded integers."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(label))]


def _ordered_strata(frame: pd.DataFrame) -> list[str]:
    """Return the strata in natural order (so ``Q2`` precedes ``Q10``)."""
    return sorted(frame["stratum"].unique(), key=_natural_key)


def _class_order(summary: pd.DataFrame) -> list[int]:
    """Return the reference classes in ascending id order."""
    return sorted(int(c) for c in summary["ref_class"].unique())


def _class_names(summary: pd.DataFrame) -> dict[int, str]:
    """Map each reference class to its aligned name, falling back to the id."""
    if "class_name" in summary.columns:
        rows = summary.drop_duplicates("ref_class")
        return {int(c): str(n) for c, n in zip(rows["ref_class"], rows["class_name"], strict=True)}
    return {int(c): str(c) for c in _class_order(summary)}


def _category_colours() -> dict[str, str]:
    """Assign each literature category a stable palette colour."""
    return {cat: style.PALETTE[i % len(style.PALETTE)] for i, cat in enumerate(_CATEGORY_ORDER)}


def attribution_figure(summary: pd.DataFrame, category: pd.DataFrame, meta: dict) -> Figure:
    """Build the two-panel attribution summary from an ``attribute`` run.

    Parameters
    ----------
    summary : pandas.DataFrame
        The ``summary_<axis>`` table: one row per stratum and class, with ``churn``, ``jaccard``,
        ``ref_class``, and ``class_name``.
    category : pandas.DataFrame
        The ``category_<axis>`` table: one row per stratum, class, and category, with the signed
        ``contribution`` to the squared distance.
    meta : dict
        The run's manifest metrics, carrying ``axis``.

    Returns
    -------
    matplotlib.figure.Figure
        Panel A (churn heatmap) beside panel B (category composition).
    """
    import matplotlib.pyplot as plt

    strata = _ordered_strata(summary)
    classes = _class_order(summary)
    names = _class_names(summary)
    nice = "age at diagnosis" if meta.get("axis") == "age_at_diagnosis" else "diagnostic era"

    churn = summary.pivot(index="ref_class", columns="stratum", values="churn").reindex(
        index=classes, columns=strata
    )
    jaccard = summary.pivot(index="ref_class", columns="stratum", values="jaccard").reindex(
        index=classes, columns=strata
    )

    with style.house_style():
        fig, (ax_a, ax_b) = plt.subplots(
            1, 2, figsize=(9.4, 0.6 + 0.62 * len(classes) + 1.6), width_ratios=(1.35, 1.0)
        )

        grid = churn.to_numpy(dtype=float)
        image = ax_a.imshow(grid, cmap="cividis", vmin=0.0, vmax=1.0, aspect="auto")
        ax_a.set_xticks(range(len(strata)), strata, rotation=45, ha="right")
        ax_a.set_yticks(range(len(classes)), [names[c] for c in classes])
        ax_a.set_xlabel(f"{nice} stratum")
        ax_a.grid(False)
        for i in range(len(classes)):
            for j in range(len(strata)):
                value = grid[i, j]
                if not np.isfinite(value):
                    continue
                ink = "white" if value < 0.55 else "black"
                ax_a.text(j, i, f"{value:.2f}"[1:], ha="center", va="center", color=ink, fontsize=7)
                if float(jaccard.to_numpy()[i, j]) < 0.5:
                    ax_a.add_patch(
                        Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor=style.REFERENCE_COLOUR,
                            linewidth=1.5,
                        )
                    )
        bar = fig.colorbar(image, ax=ax_a, fraction=0.046, pad=0.03)
        bar.set_label("churn (1 minus Jaccard overlap)")
        style.panel_title(ax_a, "A", "membership churn across the axis")

        colours = _category_colours()
        pooled = category.groupby(["ref_class", "category"])["contribution"].sum().clip(lower=0.0)
        y = np.arange(len(classes))
        for idx, cls in enumerate(classes):
            shares = pooled.loc[cls] if cls in pooled.index.get_level_values(0) else pd.Series()
            total = float(shares.sum()) or 1.0
            left = 0.0
            for cat in _CATEGORY_ORDER + ("other",):
                if cat == "other":
                    value = float(shares.drop(list(_CATEGORY_ORDER), errors="ignore").sum())
                    colour = _OTHER_COLOUR
                else:
                    value = float(shares.get(cat, 0.0))
                    colour = colours[cat]
                if value <= 0:
                    continue
                ax_b.barh(y[idx], value / total, left=left, color=colour, height=0.72)
                left += value / total
        ax_b.set_yticks(y, [names[c] for c in classes])
        ax_b.set_xlim(0, 1)
        ax_b.set_xlabel("share of the class's centroid shift")
        ax_b.invert_yaxis()
        ax_b.grid(False)
        handles = [plt.Rectangle((0, 0), 1, 1, color=colours[c]) for c in _CATEGORY_ORDER] + [
            plt.Rectangle((0, 0), 1, 1, color=_OTHER_COLOUR)
        ]
        labels = [style.CATEGORY_LABELS.get(c, c) for c in _CATEGORY_ORDER] + ["other"]
        ax_b.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
        style.panel_title(ax_b, "B", "what carries each class's shift")

        fig.tight_layout()
    return fig


def mover_contrast_figure(
    summary: pd.DataFrame, movers: pd.DataFrame, meta: dict, top_k: int = 8
) -> Figure:
    """Build the per-class mover-contrast figure from an ``attribute`` run.

    For each class the panel is drawn at the stratum where the class churns most, and shows the
    ``top_k`` features whose standardised mean difference between the probands that changed class
    and the stayers is largest, signed so a positive bar is a feature higher in the movers.

    Parameters
    ----------
    summary : pandas.DataFrame
        The ``summary_<axis>`` table, used to pick each class's peak-churn stratum.
    movers : pandas.DataFrame
        The ``movers_<axis>`` table: one row per stratum, class, and feature, with the signed
        ``effect``, its ``magnitude``, and an ``fdr_significant`` flag.
    meta : dict
        The run's manifest metrics, carrying ``axis``.
    top_k : int, default 8
        Number of features to show per panel.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per class, arranged in a near-square grid.
    """
    import matplotlib.pyplot as plt

    classes = _class_order(summary)
    names = _class_names(summary)
    letters = "ABCDEFGH"
    n = len(classes)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))

    with style.house_style():
        fig, axes = plt.subplots(nrow, ncol, figsize=(9.4, 0.4 + 2.5 * nrow), squeeze=False)
        flat = axes.ravel()
        for k, cls in enumerate(classes):
            ax = flat[k]
            rows = summary[summary["ref_class"] == cls]
            peak = rows.loc[rows["churn"].idxmax(), "stratum"] if len(rows) else None
            sub = movers[(movers["ref_class"] == cls) & (movers["stratum"] == peak)]
            sub = sub.sort_values("magnitude", ascending=False).head(top_k).iloc[::-1]
            if sub.empty:
                ax.text(
                    0.5,
                    0.5,
                    "no members changed",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color=style.REFERENCE_COLOUR,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                pos = np.arange(len(sub))
                effects = sub["effect"].to_numpy(dtype=float)
                colours = ["#0072B2" if e >= 0 else "#D55E00" for e in effects]
                alphas = [1.0 if s else 0.4 for s in sub["fdr_significant"]]
                bars = ax.barh(pos, effects, color=colours, height=0.7)
                for patch, alpha in zip(bars, alphas, strict=True):
                    patch.set_alpha(alpha)
                ax.set_yticks(pos, [str(f) for f in sub["feature"]], fontsize=6)
                ax.axvline(0, color=style.REFERENCE_COLOUR, linewidth=0.8)
                ax.set_xlabel("movers minus stayers (SD units)")
                ax.grid(False)
            style.panel_title(ax, letters[k], f"{names[cls]} · {peak}")
        for extra in range(n, nrow * ncol):
            flat[extra].axis("off")
        fig.tight_layout()
    return fig
