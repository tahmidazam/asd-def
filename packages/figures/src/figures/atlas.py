"""The displacement atlas: per-class endpoint drift along every non-modelling ordering axis.

Reads a ``displacement-atlas`` run (plan section 12b) and draws a heatmap of each class's
separation-scaled endpoint displacement along every continuous or ordered axis outside the 238
clustered features. The axes are grouped into stacked panels by kind (the timing axes, the covariate
pool, and the random floor), one panel each labelled A onward, sharing a single colour scale and the
class columns on the x-axis. Within each panel the rows run from the largest class-summed mover to
the smallest, so the map reads as a ranking. The random floor is the last panel: an axis whose row
sits above it carries drift beyond sampling noise. No covariate is assumed orthogonal to timing, so
the atlas reports every axis against that single random reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from figures import style

# The panels, top to bottom: the mechanism under test, then the external orderings, then the floor.
_GROUPS: tuple[tuple[str, str], ...] = (
    ("timing", "Timing (the mechanism under test)"),
    ("covariate", "Covariates"),
    ("random", "Random floor"),
)
_LETTERS = ("A", "B", "C", "D", "E", "F")

# Short axis labels for the narrow collaboration-brief figure, where the full names do not fit the
# column width. Keyed by axis name; an axis not listed keeps its full label.
_SHORT_LABELS: dict[str, str] = {
    "era": "Dx era",
    "age_at_diagnosis": "Age at dx",
    "lag": "Meas. lag",
    "age_at_eval": "Age at eval",
    "household_income": "Income",
    "area_deprivation": "Deprivation",
    "random": "Random",
}


def atlas_figure(
    atlas: pd.DataFrame,
    meta: dict,
    *,
    width_in: float = 6.4,
    height_in: float | None = None,
    label_pt: float = 9.0,
    value_pt: float = 7.0,
    compact: bool = False,
) -> Figure:
    """Build the displacement atlas: stacked per-kind heatmap panels, sorted by displacement.

    Parameters
    ----------
    atlas : pandas.DataFrame
        The ``displacement_atlas`` table: per axis and reference class, the separation-scaled
        endpoint displacement, the axis ``label`` and ``kind``, and the joined sample size.
    meta : dict
        The run metrics; unused beyond documenting provenance.
    width_in : float, optional
        Figure width in inches.
    height_in : float, optional
        Figure height in inches; derived from the row count when omitted, for dense rows.
    label_pt, value_pt : float, optional
        Point sizes for the axis and tick labels and for the in-cell values, so the same figure
        reads at both the documentation size and the smaller collaboration-brief size.
    compact : bool, optional
        Use the short axis labels, for the narrow collaboration-brief column where the full names
        do not fit.

    Returns
    -------
    matplotlib.figure.Figure
        The stacked atlas: one heatmap panel per axis kind, sharing a colour scale and the class
        columns, with the random floor as the last panel.
    """
    import matplotlib.pyplot as plt

    classes = sorted(atlas["ref_class"].unique())
    class_names = atlas.drop_duplicates("ref_class").set_index("ref_class")["class_name"].to_dict()
    labels = atlas.drop_duplicates("axis_name").set_index("axis_name")["label"].to_dict()
    if compact:
        labels = {name: _SHORT_LABELS.get(name, labels[name]) for name in labels}
    totals = atlas.groupby("axis_name")["endpoint_magnitude"].sum()
    ceiling = float(atlas["endpoint_magnitude"].max())

    # Group the axes by kind, dropping empty groups, and order each group's rows by the class-summed
    # displacement so the biggest mover sits on top of its panel.
    grouped: list[tuple[str, str, list[str]]] = []
    for kind, title in _GROUPS:
        names = atlas[atlas["kind"] == kind].drop_duplicates("axis_name")["axis_name"].tolist()
        if not names:
            continue
        names.sort(key=lambda a: totals[a], reverse=True)
        grouped.append((kind, title, names))

    row_counts = [len(names) for _, _, names in grouped]
    n_rows = sum(row_counts)
    if height_in is None:
        height_in = 0.30 * n_rows + 0.45 * len(grouped) + 0.9

    def matrix_for(names: list[str]) -> np.ndarray:
        return np.array(
            [
                [
                    float(
                        atlas[(atlas["axis_name"] == a) & (atlas["ref_class"] == c)][
                            "endpoint_magnitude"
                        ].iloc[0]
                    )
                    for c in classes
                ]
                for a in names
            ]
        )

    with style.house_style():
        fig, panel_axes = plt.subplots(
            len(grouped),
            1,
            figsize=(width_in, height_in),
            sharex=True,
            gridspec_kw={"height_ratios": row_counts, "hspace": 0.5},
            squeeze=False,
        )
        axes = [row[0] for row in panel_axes]
        image = None
        for panel, (ax, (_kind, title, names)) in enumerate(zip(axes, grouped, strict=True)):
            matrix = matrix_for(names)
            image = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=ceiling)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([labels.get(a, a) for a in names], fontsize=label_pt)
            ax.set_xticks(range(len(classes)))
            ax.grid(False)
            ax.tick_params(length=0)
            for i in range(len(names)):
                for j in range(len(classes)):
                    value = matrix[i, j]
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=value_pt,
                        color="white" if value > 0.5 * ceiling else "#333333",
                    )
            ax.set_title(rf"$\mathbf{{{_LETTERS[panel]}}}$  {title}", loc="left", fontsize=label_pt)
        # Only the bottom panel carries the shared class-column labels.
        axes[-1].set_xticklabels(
            [class_names.get(c, str(c)) for c in classes],
            rotation=30,
            ha="right",
            fontsize=label_pt,
        )

        assert image is not None  # at least one group is always present
        bar = fig.colorbar(image, ax=axes, fraction=0.04, pad=0.02)
        bar.set_label("Endpoint displacement (separation units)", fontsize=label_pt)
        bar.ax.tick_params(labelsize=value_pt)
    return fig
