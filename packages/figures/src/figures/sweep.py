"""The kernel-sweep trajectory figure: each class's drift as a smooth curve along the axis.

Built from a `sweep` run's decision table, the figure plots each reference class's drift,
expressed as a fraction of the between-class separation, against the focal point of the axis
(age at diagnosis or diagnostic era). A hard-bin sweep gives one point per bin; a kernel sweep
gives the smooth local-likelihood trajectory this figure is meant for. A focal point where the
class reorganised (Jaccard below 0.5) is ringed, since its position is a relabelling rather than
a move, and the dashed line at one marks the between-class separation, so a curve approaching it
is a class drifting as far as the gap between distinct classes.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from figures import style

_NICE_AXIS = {"age_at_diagnosis": "age at diagnosis (years)", "era": "diagnosis year"}


def sweep_trajectory_figure(decision: pd.DataFrame, names: dict[int, str], meta: dict) -> Figure:
    """Build the sweep-trajectory figure from a `sweep` decision table.

    Parameters
    ----------
    decision : pandas.DataFrame
        The decision table, with ``ref_class``, ``position``, ``drift_vs_separation``, and
        ``reorganised`` columns.
    names : dict of int to str
        Reference-class id to named class, for the legend. A missing id falls back to its number.
    meta : dict
        The run's manifest metrics, carrying ``axis`` and the scheme summary for the title.

    Returns
    -------
    matplotlib.figure.Figure
        The one-panel trajectory figure.
    """
    axis = str(meta.get("axis", ""))
    classes = sorted(int(c) for c in decision["ref_class"].unique())
    with style.house_style():
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for index, ref_class in enumerate(classes):
            sub = decision[decision["ref_class"] == ref_class].sort_values("position")
            colour = style.PALETTE[index % len(style.PALETTE)]
            ax.plot(
                sub["position"],
                sub["drift_vs_separation"],
                color=colour,
                lw=1.6,
                label=names.get(ref_class, f"Class {ref_class}"),
                zorder=3,
            )
            reorganised = sub[sub["reorganised"]]
            ax.scatter(
                reorganised["position"],
                reorganised["drift_vs_separation"],
                s=44,
                facecolor="none",
                edgecolor=colour,
                linewidths=1.2,
                zorder=4,
            )
        ax.axhline(1.0, color=style.REFERENCE_COLOUR, ls="--", lw=0.8)
        ax.text(
            0.99,
            1.0,
            "between-class separation",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7,
            color=style.REFERENCE_COLOUR,
        )
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(_NICE_AXIS.get(axis, axis))
        ax.set_ylabel("class drift / between-class separation")
        ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=len(classes))
        ax.margins(x=0.02)
    return fig
