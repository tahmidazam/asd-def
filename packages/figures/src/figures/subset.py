"""The records-cutoff comparison figure: full release against the V9 subset and a control.

Built from three ``align`` runs and three ``select`` runs (the full ``2026-03-23`` release, the
``--as-of`` V9 subset, and a size-matched random subsample), this figure reads the class
proportions and the model-selection criteria across the cuts, against the values Litman et al.
(2025) report. Panel (a) groups the four named-class proportions by cut; panel (b) traces each
cut's information criterion across the number of classes, normalised within the cut so the
minima are comparable despite the cohorts differing in size. The point is to see which
divergences from the paper move when the cohort is cut back to the records present at V9.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from figures import style


def subset_comparison_figure(
    proportions: dict[str, dict[str, float]],
    selection: dict[str, pd.DataFrame],
    *,
    cut_order: list[str],
    class_order: list[str],
    criterion: str = "bic_mean",
) -> Figure:
    """Build the records-cutoff comparison figure.

    Parameters
    ----------
    proportions : dict
        Maps each cut label to its named-class proportions (class name to proportion). Every
        cut in ``cut_order`` must have an entry; the published values are one such cut.
    selection : dict
        Maps a cut label to its ``select`` summary frame (one row per number of components,
        with an ``n_components`` column and the ``criterion`` column). The published cut has no
        selection curve and is omitted here.
    cut_order : list of str
        The cuts, in the order they are drawn and coloured.
    class_order : list of str
        The named classes, in the order they appear on the proportion axis.
    criterion : str, default "bic_mean"
        The information-criterion column traced in panel (b).

    Returns
    -------
    matplotlib.figure.Figure
        A two-panel figure: the proportion comparison and the selection-criterion traces.

    Raises
    ------
    ValueError
        When a cut named in ``cut_order`` has no proportions, or a class is missing from one.
    """
    for cut in cut_order:
        if cut not in proportions:
            raise ValueError(f"no proportions for cut {cut!r}")
        missing = [c for c in class_order if c not in proportions[cut]]
        if missing:
            raise ValueError(f"cut {cut!r} is missing classes {missing}")

    colours = {cut: style.PALETTE[i % len(style.PALETTE)] for i, cut in enumerate(cut_order)}

    with style.house_style():
        fig, (ax_prop, ax_sel) = plt.subplots(1, 2, figsize=(9.2, 4.0))

        # Panel (a): the four named-class proportions, grouped by cut.
        positions = np.arange(len(class_order))
        width = 0.8 / len(cut_order)
        for i, cut in enumerate(cut_order):
            offset = (i - (len(cut_order) - 1) / 2) * width
            heights = [proportions[cut][c] for c in class_order]
            ax_prop.bar(positions + offset, heights, width=width, color=colours[cut], label=cut)
        ax_prop.set_xticks(positions)
        ax_prop.set_xticklabels(
            [style.CATEGORY_LABELS.get(c, c) for c in class_order], rotation=30, ha="right"
        )
        ax_prop.set_ylabel("Class proportion")
        ax_prop.set_ylim(0.0, max(max(p.values()) for p in proportions.values()) * 1.18)
        ax_prop.legend(loc="upper right", fontsize=6.5)

        # Panel (b): each cut's criterion across K, normalised within the cut to [0, 1] so the
        # minimum's location is comparable across cohorts of different size; a marker sits on
        # each minimum, and a reference line marks the four classes Litman et al. retained.
        for cut, summary in selection.items():
            frame = summary.sort_values("n_components")
            k = frame["n_components"].to_numpy()
            values = frame[criterion].to_numpy(dtype=float)
            spread = values.max() - values.min()
            scaled = (values - values.min()) / spread if spread > 0 else np.zeros_like(values)
            ax_sel.plot(k, scaled, marker="o", markersize=3, color=colours[cut], label=cut)
            argmin = int(k[int(np.argmin(values))])
            ax_sel.scatter(
                [argmin],
                [0.0],
                color=colours[cut],
                s=44,
                zorder=5,
                edgecolor="white",
                linewidth=0.6,
            )
        ax_sel.axvline(4, color=style.REFERENCE_COLOUR, linestyle="--", linewidth=1.0)
        ax_sel.text(
            4.15, 0.95, "K = 4 retained", color=style.REFERENCE_COLOUR, fontsize=6.5, va="top"
        )
        ax_sel.set_xlabel("Number of latent classes")
        ax_sel.set_ylabel("Bayesian information criterion (scaled within cut)")
        ax_sel.legend(loc="upper right", fontsize=6.5)

        style.panel_title(ax_prop, "A", "Class proportions across cuts")
        style.panel_title(ax_sel, "B", "Model selection across cuts")
        fig.tight_layout()
    return fig
