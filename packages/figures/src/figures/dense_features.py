"""The dense feature matrix: every significant feature's signed drift, per class and axis.

The top-five lollipops of the {py:mod}`~figures.category_decomposition` figure name the leading
features; this figure shows them all. Each row is a feature that clears false-discovery-rate control
in at least one class on at least one axis (227 of the 238 do), each column is a class on an axis,
and the cell colour is the signed separation-standardised displacement, red for a rise and blue for
a fall, with the non-significant cells left faint so the significance pattern reads at a glance. The
rows are grouped by a chosen key, the author symptom category for $H_0^F$ or the instrument referent
for $H_0^G$, with a colour sidebar and a divider between groups, so the concentration the summary
figures report is visible feature by feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from figures import style
from figures.category_decomposition import _CATEGORY_ORDER, _category_of

_NICE_AXIS = {"era": "era", "age_at_diagnosis": "age"}


def _feature_groups(
    features: dict[str, pd.DataFrame], group_by: str
) -> tuple[dict[str, str], set[str]]:
    """Return the group of every feature and the set significant in at least one cell."""
    group_of: dict[str, str] = {}
    significant: set[str] = set()
    for frame in features.values():
        for feature, group, reject in zip(
            frame["feature"], frame[group_by], frame["reject"], strict=True
        ):
            group_of.setdefault(str(feature), _category_of(group))
            if bool(reject):
                significant.add(str(feature))
    return group_of, significant


def _order_features(
    features: list[str],
    group_of: dict[str, str],
    peak: dict[str, float],
    group_order: tuple[str, ...],
) -> list[str]:
    """Order features by group (in ``group_order``), then by descending peak magnitude within it."""
    rank = {name: i for i, name in enumerate(group_order)}
    return sorted(
        features,
        key=lambda f: (rank.get(group_of[f], len(rank)), group_of[f], -peak.get(f, 0.0)),
    )


def dense_feature_figure(
    features: dict[str, pd.DataFrame],
    meta: dict,
    *,
    group_by: str = "category",
    group_order: tuple[str, ...] = _CATEGORY_ORDER,
    group_label: str = "category",
) -> Figure:
    """Build the dense signed-displacement matrix over every significant feature.

    Parameters
    ----------
    features : dict of str to pandas.DataFrame
        The ``feature_displacement_<axis>.parquet`` frame per axis, each carrying ``feature``,
        ``ref_class``, ``class_name``, ``displacement``, ``reject``, and the grouping column.
    meta : dict
        Figure metadata; unused beyond documenting provenance.
    group_by : str, optional
        The column the rows are grouped by (``"category"`` for $H_0^F$, ``"referent"`` for $H_0^G$).
    group_order : tuple of str, optional
        The order the groups are shown in, top to bottom.
    group_label : str, optional
        The word for the grouping, used in the sidebar title.

    Returns
    -------
    matplotlib.figure.Figure
        The dense feature-by-class heatmap with a group colour sidebar.
    """
    import matplotlib.pyplot as plt

    axes_order = [a for a in ("era", "age_at_diagnosis") if a in features]
    disp = {
        a: features[a].pivot(index="feature", columns="ref_class", values="displacement")
        for a in axes_order
    }
    reject = {
        a: features[a].pivot(index="feature", columns="ref_class", values="reject")
        for a in axes_order
    }
    classes = sorted(int(c) for c in features[axes_order[0]]["ref_class"].unique())
    names = (
        features[axes_order[0]]
        .drop_duplicates("ref_class")
        .set_index("ref_class")["class_name"]
        .to_dict()
    )

    group_of, significant = _feature_groups(features, group_by)
    present_groups = [g for g in group_order if g in {group_of[f] for f in significant}]
    present_groups += sorted({group_of[f] for f in significant} - set(present_groups))
    colours = {g: style.PALETTE[i % len(style.PALETTE)] for i, g in enumerate(present_groups)}

    peak = {
        f: float(
            np.nanmax([np.nanmax(np.abs(disp[a].reindex([f]).to_numpy())) for a in axes_order])
        )
        for f in significant
    }
    ordered = _order_features(sorted(significant), group_of, peak, tuple(present_groups))

    # The signed-displacement matrix (features by axis-class), with non-significant cells masked so
    # only the significant drift shows in colour.
    columns: list[tuple[str, int]] = [(a, c) for a in axes_order for c in classes]
    matrix = np.full((len(ordered), len(columns)), np.nan)
    for j, (axis, cls) in enumerate(columns):
        col = disp[axis].reindex(ordered).get(cls)
        rej = reject[axis].reindex(ordered).get(cls)
        if col is not None:
            values = col.to_numpy(dtype=float)
            mask = np.asarray(rej.to_numpy() if rej is not None else np.zeros(len(ordered)), bool)
            matrix[mask, j] = values[mask]

    cap = float(np.nanpercentile(np.abs(matrix), 96)) if np.isfinite(matrix).any() else 1.0
    cap = max(cap, 0.1)

    with style.house_style():
        height = max(4.0, min(20.0, 0.045 * len(ordered) + 1.5))
        fig = plt.figure(figsize=(7.6, height))
        grid = GridSpec(1, 2, width_ratios=(0.05, 1.0), wspace=0.02, figure=fig)

        # The group colour sidebar.
        ax_side = fig.add_subplot(grid[0, 0])
        band = np.array([[present_groups.index(group_of[f])] for f in ordered])
        ax_side.imshow(
            band,
            aspect="auto",
            cmap=ListedColormap([colours[g] for g in present_groups]),
            vmin=0,
            vmax=max(len(present_groups) - 1, 1),
        )
        ax_side.set_xticks([])
        ax_side.set_yticks([])
        ax_side.set_title(f"n = {len(ordered)}", fontsize=7)
        boundaries = _group_boundaries([group_of[f] for f in ordered])
        for _, lo, _hi in boundaries[1:]:
            ax_side.axhline(lo - 0.5, color="white", linewidth=0.8)

        ax = fig.add_subplot(grid[0, 1])
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-cap, vmax=cap)
        image.cmap.set_bad("#f2f2f2")
        ax.set_yticks([])
        for _, lo, _hi in boundaries[1:]:
            ax.axhline(lo - 0.5, color="white", linewidth=0.8)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(
            [f"{names.get(c, c)}" for _a, c in columns], rotation=45, ha="right", fontsize=6.5
        )
        # A divider and axis super-labels between the era block and the age block.
        if len(axes_order) == 2:
            ax.axvline(len(classes) - 0.5, color=style.REFERENCE_COLOUR, linewidth=1.2)
            for i, axis in enumerate(axes_order):
                ax.text(
                    (i + 0.5) * len(classes) - 0.5,
                    -0.02 * len(ordered) - 1.0,
                    _NICE_AXIS[axis],
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )
        bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        bar.set_label("signed displacement (sep-standardised)", fontsize=7.5)
        bar.ax.tick_params(labelsize=7)

        handles = [
            plt.Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                markersize=7,
                color=colours[g],
                label=style.CATEGORY_LABELS.get(g, g),
            )
            for g in present_groups
        ]
        fig.legend(
            handles=handles,
            title=group_label,
            loc="lower center",
            ncol=min(len(handles), 5),
            fontsize=7,
            title_fontsize=7,
            bbox_to_anchor=(0.5, -0.04),
        )

    return fig


def _group_boundaries(sequence: list[str]) -> list[tuple[str, int, int]]:
    """Return each contiguous run in a sequence as ``(value, start, stop)`` (stop exclusive)."""
    runs: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(sequence) + 1):
        if i == len(sequence) or sequence[i] != sequence[start]:
            runs.append((sequence[start], start, i))
            start = i
    return runs
