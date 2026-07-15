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
:func:`stacked_area_pair_figure` sets the diagnostic-era and age-at-diagnosis compositions side by
side in one figure, sharing the vertical scale and one legend, so the two axes read together.
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
    axis_name = str(meta.get("axis"))

    with style.house_style():
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        _draw_stack(ax, curve, axis_name, "A")
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


def stacked_area_pair_figure(curves: dict[str, pd.DataFrame], meta: dict) -> Figure:
    """Build the side-by-side stacked-area H0B figure: composition on both timing axes at once.

    The diagnostic-era and age-at-diagnosis compositions are drawn as two panels sharing the
    vertical scale, with a single legend beneath, so the two axes' shifts are read together: the
    gentler era trade on the left and the starker age-at-diagnosis trade on the right. The stacking
    order and colours are fixed from the pooled proportions, which are axis-free and so shared, so a
    class sits in the same band and colour in both panels.

    Parameters
    ----------
    curves : dict of str to pandas.DataFrame
        The ``proportion_curve_<axis>`` table (``ref_class``, ``class_name``, ``position``,
        ``corrected``, ``pooled``) for each timing axis, keyed ``"era"`` and
        ``"age_at_diagnosis"``. Whichever axes are present are drawn, era first.
    meta : dict
        The run metrics; unused beyond documenting provenance.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel stacked-area figure, one timing axis per panel.
    """
    axes_present = [a for a in ("era", "age_at_diagnosis") if a in curves]
    # Stacking order and colours from the pooled proportions, which are axis-free, so a class keeps
    # its band and colour across both panels.
    reference = curves[axes_present[0]]
    order, colours = _stack_order(reference)

    with style.house_style():
        fig, axes = plt.subplots(
            1, len(axes_present), figsize=(9.6, 4.6), sharey=True, squeeze=False
        )
        flat = axes[0]
        for panel, axis_name in enumerate(axes_present):
            ax = flat[panel]
            _draw_stack(
                ax, curves[axis_name], axis_name, _LETTERS[panel], order=order, colours=colours
            )
            if panel != 0:
                ax.set_ylabel("")
        # One shared legend, ordered top-to-bottom to match the visual stack.
        handles, legend_labels = flat[0].get_legend_handles_labels()
        fig.legend(
            handles[::-1],
            legend_labels[::-1],
            loc="lower center",
            ncol=len(order),
            bbox_to_anchor=(0.5, -0.02),
            fontsize=8,
        )
        fig.suptitle(
            "Class composition across diagnostic era and age at diagnosis",
            x=0.02,
            ha="left",
            fontsize=10,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
    return fig


def _stack_order(curve: pd.DataFrame) -> tuple[list[int], dict[int, str]]:
    """Return the class stacking order (largest pooled first) and each class's colour."""
    classes = sorted(int(c) for c in curve["ref_class"].unique())
    colours = {c: style.PALETTE[i % len(style.PALETTE)] for i, c in enumerate(classes)}
    pooled = {c: float(curve[curve["ref_class"] == c]["pooled"].iloc[0]) for c in classes}
    order = sorted(classes, key=lambda c: pooled[c], reverse=True)
    return order, colours


def _draw_stack(
    ax,
    curve: pd.DataFrame,
    axis_name: str,
    letter: str,
    *,
    order: list[int] | None = None,
    colours: dict[int, str] | None = None,
) -> None:
    """Draw one stacked-area composition on ``ax``: the corrected proportions stacked to one.

    The largest pooled class stacks at the bottom for a stable base. When ``order`` and ``colours``
    are given they fix the band order and palette across panels; otherwise they are taken from this
    axis's own pooled proportions.
    """
    if order is None or colours is None:
        order, colours = _stack_order(curve)
    nice = _NICE_AXIS.get(axis_name, axis_name)
    positions = np.sort(curve["position"].unique())
    stacks = [
        curve[curve["ref_class"] == c].sort_values("position")["corrected"].to_numpy(dtype=float)
        for c in order
    ]
    names = [
        str(curve[curve["ref_class"] == c]["class_name"].iloc[0]).split(" (")[0] for c in order
    ]
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
    style.panel_title(ax, letter, f"Class composition along {nice}")
