"""The pairwise-trajectory figure: neighbour-to-neighbour class change along the axis.

Built from a ``drift --reference-scheme pairwise`` run, the figure shows how much each reference
class moves between adjacent strata of the axis (age at diagnosis or diagnostic era), rather than
how far it sits from the pooled reference. The upper panel plots each class's neighbour drift,
scaled by the pooled between-class separation, against the axis position of the earlier stratum in
each pair; the dashed line at one marks the separation, so a curve near it is a class changing as
much between neighbours as the gap between distinct classes. Adjacent strata share no probands, so
the classes are matched by centroid alignment, whose confidence the lower panel tracks: where it is
low, the neighbour match is uncertain and the drift above it should be read with care. The observed
trajectory is descriptive until the union-split null calibrates it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from figures import style

_NICE_AXIS = {"age_at_diagnosis": "age at diagnosis (years)", "era": "diagnosis year"}


def pairwise_trajectory_figure(
    trajectory: pd.DataFrame, names: dict[int, str], meta: dict
) -> Figure:
    """Build the pairwise-trajectory figure from a pairwise ``drift`` run's trajectory table.

    Parameters
    ----------
    trajectory : pandas.DataFrame
        The ``pairwise_<axis>`` table, with ``ref_class``, ``position``, ``drift_vs_separation``,
        ``centroid_quality``, and ``overall_quality`` columns.
    names : dict of int to str
        Reference-class id to named class, for the legend. A missing id falls back to its number.
    meta : dict
        The run's manifest metrics, carrying ``axis`` for the axis label.

    Returns
    -------
    matplotlib.figure.Figure
        A two-panel figure: the drift trajectory over the axis, and the centroid-alignment
        confidence beneath it.

    Raises
    ------
    ValueError
        When a required column is missing, or the table is an all-pairs run (which has more than
        one comparison per position and so is not a single trajectory).
    """
    required = {
        "ref_class",
        "position",
        "drift_vs_separation",
        "centroid_quality",
        "overall_quality",
    }
    missing = required - set(trajectory.columns)
    if missing:
        raise ValueError(f"trajectory is missing columns: {sorted(missing)}")
    per_position = trajectory.groupby(["ref_class", "position"]).size()
    if not trajectory.empty and bool((per_position > 1).any()):
        raise ValueError(
            "the pairwise-trajectory figure needs an adjacent-mode run (one comparison per "
            "position); this table has several, so it is an all-pairs run"
        )

    axis = str(meta.get("axis", ""))
    classes = sorted(int(c) for c in trajectory["ref_class"].unique())
    confidence = trajectory.groupby("position")["overall_quality"].first().sort_index()

    with style.house_style():
        fig, (ax, ax_conf) = plt.subplots(
            2, 1, figsize=(7.2, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        for index, ref_class in enumerate(classes):
            sub = trajectory[trajectory["ref_class"] == ref_class].sort_values("position")
            colour = style.PALETTE[index % len(style.PALETTE)]
            ax.plot(
                sub["position"],
                sub["drift_vs_separation"],
                color=colour,
                lw=1.6,
                marker="o",
                ms=4,
                label=names.get(ref_class, f"Class {ref_class}"),
                zorder=3,
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
        ax.set_ylabel("neighbour drift / separation")
        ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=max(1, len(classes)))
        ax.margins(x=0.02)

        ax_conf.plot(
            confidence.index, confidence.to_numpy(), color="#6f6f6f", lw=1.2, marker="o", ms=3
        )
        ax_conf.set_ylim(0.0, 1.0)
        ax_conf.set_ylabel("centroid\nconfidence", fontsize=8)
        ax_conf.set_xlabel(_NICE_AXIS.get(axis, axis))
    return fig
