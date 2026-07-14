r"""Figure for the prevalence-drift test (H0B, plan section 3 / 12b).

A ``prevalence`` run reads how each frozen class's mixing proportion trends along an axis, under
the maximum-likelihood three-step correction, with a family-clustered bootstrap band. The estimand
is the proportion as a function of the axis, so the figure draws that curve directly.

:func:`proportion_curve_figure` gives one panel per class: the corrected proportion curve with its
bootstrap band, the naive hard-label curve as a thin dashed cross-check, a dotted line at the
pooled (axis-free) proportion the class trends away from, and a title carrying the per-year log-odds
slope, its odds ratio, and whether the class's proportion trends under the false-discovery control.
:func:`stacked_area_figure` gives the compositional view: the four corrected proportions stacked to
one across the axis, so a class growing as another shrinks is read as a single shifting composition.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from figures import style

_NICE_AXIS = {"age_at_diagnosis": "age at diagnosis (years)", "era": "diagnostic era (year)"}
_LETTERS = ("A", "B", "C", "D")


def proportion_curve_figure(curve: pd.DataFrame, slopes: pd.DataFrame, meta: dict) -> Figure:
    """Build the H0B figure: each class's predicted proportion as a function of the axis.

    Parameters
    ----------
    curve : pandas.DataFrame
        The ``proportion_curve_<axis>`` table (``ref_class``, ``class_name``, ``position``,
        ``corrected``, ``naive``, ``band_lo``, ``band_hi``).
    slopes : pandas.DataFrame
        The ``slopes_<axis>`` table; the ``corrected`` rows supply each panel's slope, odds ratio,
        and false-discovery decision.
    meta : dict
        The run's manifest metrics, carrying ``axis``.

    Returns
    -------
    matplotlib.figure.Figure
        The four-panel figure, one class per panel, sharing the axis.
    """
    classes = sorted(int(c) for c in curve["ref_class"].unique())
    colours = {c: style.PALETTE[i % len(style.PALETTE)] for i, c in enumerate(classes)}
    corrected = slopes[slopes["kind"] == "corrected"].set_index("ref_class")
    axis_name = str(meta.get("axis"))
    nice = _NICE_AXIS.get(axis_name, axis_name)

    # A shared proportion range across panels, so the compositional shifts are comparable.
    y_max = float(np.nanmax(curve["band_hi"].to_numpy(dtype=float))) * 1.08

    with style.house_style():
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=True, sharey=True)
        flat = axes.flatten()
        for panel, c in enumerate(classes):
            ax = flat[panel]
            path = curve[curve["ref_class"] == c].sort_values("position")
            pos = path["position"].to_numpy(dtype=float)
            colour = colours[c]
            ax.fill_between(
                pos,
                path["band_lo"].to_numpy(dtype=float),
                path["band_hi"].to_numpy(dtype=float),
                color=colour,
                alpha=0.16,
                lw=0,
                zorder=2,
            )
            ax.plot(pos, path["corrected"].to_numpy(dtype=float), color=colour, lw=1.8, zorder=4)
            ax.plot(
                pos,
                path["naive"].to_numpy(dtype=float),
                color="#555",
                lw=1.0,
                ls=(0, (4, 2)),
                zorder=3,
            )
            # The pooled (axis-free) proportion the curve trends away from.
            if "pooled" in path:
                pooled = float(path["pooled"].iloc[0])
                ax.axhline(pooled, color=colour, ls=":", lw=1.0, alpha=0.8, zorder=2.5)
                ax.text(
                    pos[0],
                    pooled,
                    f" pooled {pooled:.2f}",
                    va="bottom",
                    ha="left",
                    fontsize=6.8,
                    color=colour,
                )
            name = str(path["class_name"].iloc[0])
            if c in corrected.index:
                row = corrected.loc[c]
                verdict = "trends" if bool(row["reject"]) else "flat"
                title = f"{name}: {row['slope']:+.3f}/unit (OR {row['odds_ratio']:.3f}), {verdict}"
            else:
                title = name
            style.panel_title(ax, _LETTERS[panel % len(_LETTERS)], title)
            ax.set_ylim(0.0, y_max)
            ax.margins(x=0.02)
            if panel % 2 == 0:
                ax.set_ylabel("Predicted proportion")
            if panel // 2 == 1:
                ax.set_xlabel(f"{nice[0].upper()}{nice[1:]}")
        # One shared legend explaining the corrected curve, band, and naive cross-check.
        handles = [
            plt.Line2D([], [], color="#333", lw=1.8, label="Corrected (3-step)"),
            plt.Line2D([], [], color="#555", lw=1.0, ls=(0, (4, 2)), label="Naive (hard label)"),
            plt.Line2D([], [], color="#333", lw=1.0, ls=":", label="Pooled proportion"),
            plt.Rectangle((0, 0), 1, 1, color="#333", alpha=0.16, label="Bootstrap band"),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.01),
        )
        fig.suptitle(
            f"Class prevalence along {_NICE_AXIS.get(axis_name, axis_name)}",
            x=0.02,
            ha="left",
            fontsize=10,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.98))
    return fig


def stacked_area_figure(curve: pd.DataFrame, meta: dict) -> Figure:
    """Build the stacked-area H0B figure: the class composition across the axis.

    The corrected proportions sum to one at every axis position, so they stack into a full
    composition. The classes are stacked largest-pooled at the bottom for a stable base, and each
    band is labelled with its class name, so the compositional shift (one class growing as another
    shrinks) is read directly.

    Parameters
    ----------
    curve : pandas.DataFrame
        The ``proportion_curve_<axis>`` table (``ref_class``, ``class_name``, ``position``,
        ``corrected``, ``pooled``).
    meta : dict
        The run's manifest metrics, carrying ``axis``.

    Returns
    -------
    matplotlib.figure.Figure
        The single-panel stacked-area figure.
    """
    classes = sorted(int(c) for c in curve["ref_class"].unique())
    colours = {c: style.PALETTE[i % len(style.PALETTE)] for i, c in enumerate(classes)}
    axis_name = str(meta.get("axis"))
    nice = _NICE_AXIS.get(axis_name, axis_name)

    positions = np.sort(curve["position"].unique())
    # Stack the largest pooled class at the bottom, so the base band is stable.
    pooled = {c: float(curve[curve["ref_class"] == c]["pooled"].iloc[0]) for c in classes}
    order = sorted(classes, key=lambda c: pooled[c], reverse=True)
    stacks = [
        curve[curve["ref_class"] == c].sort_values("position")["corrected"].to_numpy(dtype=float)
        for c in order
    ]
    names = [
        str(curve[curve["ref_class"] == c]["class_name"].iloc[0]).split(" (")[0] for c in order
    ]

    with style.house_style():
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        ax.stackplot(
            positions,
            *stacks,
            colors=[colours[c] for c in order],
            labels=names,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_xlim(float(positions.min()), float(positions.max()))
        ax.set_ylim(0.0, 1.0)
        ax.margins(x=0)
        ax.grid(False)
        ax.set_xlabel(f"{nice[0].upper()}{nice[1:]}")
        ax.set_ylabel("Class composition (proportion)")
        style.panel_title(ax, "A", f"Class composition along {nice}")
        # Legend ordered top-to-bottom to match the visual stack (reverse of the stack order).
        handles, legend_labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[::-1],
            legend_labels[::-1],
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
        )
        fig.tight_layout()
    return fig
