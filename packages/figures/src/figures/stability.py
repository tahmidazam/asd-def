"""The stability figure: how the reference solution holds under refitting.

Built from a ``stability`` run (either mode), the figure shows three things. Panel (a)
contrasts the distribution of the seven-category profile correlation, which clusters high,
with the adjusted Rand index, which is more moderate: the class definitions reproduce, while
individual proband assignments are softer at the boundaries. Panel (b) gives the per-category
correlation, uniformly high. Panel (c) is the mean class-overlap matrix, whose diagonal is
each class's retention.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from figures import style


def _mode_label(aggregate: dict) -> str:
    """Return a short description of the stability run for the figure title."""
    if "n_fits" in aggregate:
        return (
            f"multi-initialisation ({int(aggregate['n_fits'])} fits, top {int(aggregate['top_k'])})"
        )
    if "n_reps" in aggregate:
        return f"{aggregate['frac']:.0%} subsampling ({int(aggregate['n_reps'])} refits)"
    return "refitting"


def stability_figure(
    comparisons: pd.DataFrame,
    aggregate: dict,
    overlap_mean: pd.DataFrame,
) -> Figure:
    """Build the stability figure from a ``stability`` run.

    Parameters
    ----------
    comparisons : pandas.DataFrame
        One row per compared fit, with ``overall_correlation`` and ``adjusted_rand_index``.
    aggregate : dict
        The aggregate metrics, with ``category_correlation_mean`` (one entry per category) and
        a run descriptor (``n_fits``/``top_k`` or ``n_reps``/``frac``).
    overlap_mean : pandas.DataFrame
        The mean class-overlap matrix (refit class on rows, reference class on columns).

    Returns
    -------
    matplotlib.figure.Figure
        A three-panel figure: the correlation and Rand-index distributions, the per-category
        correlation, and the mean overlap matrix.

    Raises
    ------
    ValueError
        When a required column or metric is missing.
    """
    for col in ("overall_correlation", "adjusted_rand_index"):
        if col not in comparisons.columns:
            raise ValueError(f"comparisons is missing column {col!r}")
    if "category_correlation_mean" not in aggregate:
        raise ValueError("aggregate is missing 'category_correlation_mean'")

    profile = comparisons["overall_correlation"].to_numpy(dtype=float)
    profile = profile[np.isfinite(profile)]
    rand = comparisons["adjusted_rand_index"].to_numpy(dtype=float)
    rand = rand[np.isfinite(rand)]
    per_category = aggregate["category_correlation_mean"]
    categories = list(per_category)

    with style.house_style():
        fig, (ax_dist, ax_cat, ax_overlap) = plt.subplots(1, 3, figsize=(11.0, 3.8))

        # Panel (a): profile correlation (tight, high) against the adjusted Rand index (more
        # moderate): stable class definitions, softer proband-level membership.
        bodies = ax_dist.violinplot([profile, rand], positions=[1, 2], showextrema=False)
        for body, colour in zip(bodies["bodies"], style.PALETTE[:2], strict=True):
            body.set_facecolor(colour)
            body.set_alpha(0.4)
        for position, sample, colour in (
            (1, profile, style.PALETTE[0]),
            (2, rand, style.PALETTE[1]),
        ):
            ax_dist.scatter(
                np.full(sample.shape, position), sample, color=colour, s=6, alpha=0.4, zorder=3
            )
            ax_dist.scatter(
                position, sample.mean(), color=colour, marker="_", s=400, linewidth=2.0, zorder=4
            )
        ax_dist.set_xticks([1, 2])
        ax_dist.set_xticklabels(["profile\ncorrelation", "adjusted\nRand index"])
        ax_dist.set_ylim(0.0, 1.05)
        ax_dist.set_ylabel("Value across refits")

        # Panel (b): per-category profile correlation, uniformly high.
        values = [float(per_category[cat]) for cat in categories]
        colours = [style.PALETTE[i % len(style.PALETTE)] for i in range(len(categories))]
        positions = np.arange(len(categories))
        ax_cat.bar(positions, values, color=colours)
        ax_cat.set_xticks(positions)
        ax_cat.set_xticklabels(
            [style.CATEGORY_LABELS.get(cat, cat) for cat in categories], rotation=45, ha="right"
        )
        ax_cat.set_ylim(0.0, 1.05)
        ax_cat.set_ylabel("Mean profile correlation")

        # Panel (c): the mean class-overlap matrix; the diagonal is each class's retention.
        matrix = overlap_mean.to_numpy(dtype=float)
        image = ax_overlap.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
        size = matrix.shape[0]
        for i in range(size):
            for j in range(size):
                value = matrix[i, j]
                if np.isfinite(value):
                    ax_overlap.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white" if value > 0.5 else "black",
                        fontsize=7,
                    )
        ax_overlap.set_xticks(range(size))
        ax_overlap.set_yticks(range(size))
        ax_overlap.set_xlabel("reference class")
        ax_overlap.set_ylabel("refit class")
        ax_overlap.grid(visible=False)
        fig.colorbar(image, ax=ax_overlap, fraction=0.046, pad=0.04)

        style.panel_title(ax_dist, "A", "Profile correlation and adjusted Rand index")
        style.panel_title(ax_cat, "B", "Per-category profile correlation")
        style.panel_title(ax_overlap, "C", "Mean class overlap")
        fig.suptitle(f"Reference-fit stability under {_mode_label(aggregate)}", y=1.0)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig
