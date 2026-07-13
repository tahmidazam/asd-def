r"""The score-based invariance figure: the empirical fluctuation process against its null band.

Built from an `invariance` run's stored process, the figure plots the squared norm of the
standardised fluctuation process $\lVert B(t) \rVert^2$ for the strongest-drifting focal block,
against the axis (age at diagnosis or diagnosis year), with the pointwise envelope of the
simulated Brownian-bridge null shaded beneath. Under stability the observed curve would sit
inside the null band; a curve that climbs far above it is a class profile drifting along the
axis, and the peak marks the estimated break. The y-axis is logarithmic because the observed
excursion dwarfs the null band by orders of magnitude at this sample size.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from figures import style

_NICE_AXIS = {"age_at_diagnosis": "age at diagnosis (years)", "era": "diagnosis year"}


def invariance_process_figure(process: pd.DataFrame, meta: dict) -> Figure:
    """Build the fluctuation-process figure from an `invariance` run's process table.

    Parameters
    ----------
    process : pandas.DataFrame
        The stored process, with ``position``, ``observed``, ``null_q50`` and ``null_q95``
        columns (one row per grid point).
    meta : dict
        The run's manifest metrics, carrying ``axis`` and ``top_block`` for the labels.

    Returns
    -------
    matplotlib.figure.Figure
        The one-panel figure.
    """
    axis = str(meta.get("axis", ""))
    block = str(meta.get("top_block", "focal block"))
    positions = process["position"].to_numpy()
    observed = process["observed"].to_numpy()
    q95 = process["null_q95"].to_numpy()
    q50 = process["null_q50"].to_numpy()

    # A log axis cannot show the endpoints, where the bridge is pinned to zero; floor the curves
    # at a small positive value so the shape reads without dropping to minus infinity.
    floor = max(1e-3, float(q50[q50 > 0].min()) / 10.0) if np.any(q50 > 0) else 1e-3
    obs_plot = np.clip(observed, floor, None)
    break_position = float(positions[int(np.argmax(observed))])

    with style.house_style():
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.fill_between(
            positions,
            floor,
            np.clip(q95, floor, None),
            color=style.REFERENCE_COLOUR,
            alpha=0.18,
            linewidth=0,
            label="bridge null, 95th percentile",
        )
        ax.plot(positions, np.clip(q50, floor, None), color=style.REFERENCE_COLOUR, lw=0.8, ls=":")
        ax.plot(positions, obs_plot, color=style.PALETTE[3], lw=1.8, zorder=3, label="observed")
        ax.axvline(break_position, color=style.PALETTE[0], ls="--", lw=1.0, zorder=2)
        ax.text(
            break_position,
            0.02,
            f" break {break_position:.1f}",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=7,
            color=style.PALETTE[0],
        )
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor)
        ax.set_xlabel(_NICE_AXIS.get(axis, axis))
        ax.set_ylabel(r"$\Vert B(t) \Vert^2$")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        style.panel_title(ax, "A", f"{block}: fluctuation process vs bridge null")
        ax.margins(x=0.02)
    return fig
