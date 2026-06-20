r"""The minimum-viable-stratum-size figure: recovery against sample size.

Built from an ``nmin`` run, panel (a) plots the profile correlation of each refit against its
sample size, with the per-size mean, the recovery benchmark, and the isotonic floor and its
bootstrap interval. The floor is drawn as an interval because the recovery metric is noisy
even at ten replicates. Panel (b) shows the smallest class proportion holding well clear of
zero, so the four classes survive at every size.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter, ScalarFormatter

from figures import style


def nmin_figure(
    per_fit: pd.DataFrame,
    summary: pd.DataFrame,
    metrics: dict,
) -> Figure:
    """Build the minimum-viable-stratum-size figure from an ``nmin`` run.

    Parameters
    ----------
    per_fit : pandas.DataFrame
        One row per (size, replicate) fit, with ``size``, ``overall_correlation``, and
        ``smallest_class_proportion``.
    summary : pandas.DataFrame
        The per-size summary, with the same three columns.
    metrics : dict
        The floor metrics: ``floor`` and ``floor_ci90`` (a two-element list, or ``None``), and
        the ``benchmark`` correlation.

    Returns
    -------
    matplotlib.figure.Figure
        A two-panel figure: recovery against size, and the smallest class proportion.

    Raises
    ------
    ValueError
        When a required column is missing.
    """
    required = {"size", "overall_correlation", "smallest_class_proportion"}
    missing = required - set(per_fit.columns)
    if missing:
        raise ValueError(f"per_fit is missing columns: {sorted(missing)}")

    benchmark = float(metrics.get("benchmark", 0.9))
    floor = metrics.get("floor")
    interval = metrics.get("floor_ci90")
    summary = summary.sort_values("size")
    sizes = summary["size"].to_numpy(dtype=float)

    with style.house_style():
        fig, (ax_recovery, ax_class) = plt.subplots(1, 2, figsize=(8.6, 3.9))

        # Panel (a): recovery against sample size.
        if interval is not None:
            ax_recovery.axvspan(
                float(interval[0]),
                float(interval[1]),
                color=style.PALETTE[0],
                alpha=0.12,
                label=f"90% CI [{int(interval[0])}, {int(interval[1])}]",
            )
        ax_recovery.axhline(
            benchmark,
            color=style.REFERENCE_COLOUR,
            linestyle="--",
            linewidth=1.0,
            label=f"benchmark $r = {benchmark:.2f}$",
        )
        ax_recovery.scatter(
            per_fit["size"].to_numpy(dtype=float),
            per_fit["overall_correlation"].to_numpy(dtype=float),
            color=style.PALETTE[0],
            s=14,
            alpha=0.35,
            zorder=2,
        )
        ax_recovery.plot(
            sizes,
            summary["overall_correlation"].to_numpy(dtype=float),
            color=style.PALETTE[0],
            marker="o",
            zorder=3,
            label="per-size mean",
        )
        if floor is not None:
            ax_recovery.axvline(
                float(floor),
                color=style.PALETTE[3],
                linewidth=1.5,
                label=rf"floor $\approx {int(floor)}$",
            )
        ax_recovery.set_xscale("log")
        ax_recovery.set_xticks(sizes)
        ax_recovery.xaxis.set_major_formatter(ScalarFormatter())
        ax_recovery.xaxis.set_minor_formatter(NullFormatter())
        ax_recovery.set_xlabel("Subsample size")
        ax_recovery.set_ylabel("Profile correlation to reference")
        ax_recovery.set_title("Recovery vs sample size")
        ax_recovery.legend(loc="lower right", fontsize=6)

        # Panel (b): the smallest class proportion stays clear of zero, so no class collapses.
        ax_class.plot(
            sizes,
            summary["smallest_class_proportion"].to_numpy(dtype=float),
            color=style.PALETTE[2],
            marker="s",
        )
        ax_class.set_xscale("log")
        ax_class.set_xticks(sizes)
        ax_class.xaxis.set_major_formatter(ScalarFormatter())
        ax_class.xaxis.set_minor_formatter(NullFormatter())
        ax_class.set_ylim(0.0, 0.30)
        ax_class.set_xlabel("Subsample size")
        ax_class.set_ylabel("Smallest class proportion")
        ax_class.set_title("Smallest class (no collapse)")

        for ax in (ax_recovery, ax_class):
            ax.tick_params(axis="x", labelrotation=45)
        for ax, letter in ((ax_recovery, "(a)"), (ax_class, "(b)")):
            style.panel_letter(ax, letter)
        fig.tight_layout()
    return fig
