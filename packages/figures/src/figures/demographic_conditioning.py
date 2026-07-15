r"""Do demographic differences explain the drift? The conditioning heatmap (plan section 7g).

Reads a ``demographic-conditioning`` run for each timing axis and draws, per demographic covariate,
how much of each class's along-axis drift the covariate accounts for. The main panel is the
shrinkage: the fraction of a class's separation-scaled endpoint drift removed by residualising the
238 clustered features on the covariate, read for every class on both the era and the age axis. A
near-zero value means the drift is untouched; a small negative value means it grew slightly, the
noise around no effect.
A narrow panel to its left carries each covariate's linear span of the timing axis (the axis
$R^2$), the ceiling on that shrinkage, because a covariate orthogonal to the axis cannot account for
an axis-ordered drift however much feature variance it explains. A colour sidebar groups the rows by
covariate family (socioeconomic, family structure, parental, individual), and the joined sample size
annotates each row, since the survey-version covariates join far fewer probands than the
registration-complete ones.

The figure is the demographic counterpart of the H0F category decomposition: where that asks which
symptom categories carry the drift, this asks whether any demographic does. On SPARK the answer
reads straight off the two panels: the axis $R^2$ column is near zero for every covariate, so the
shrinkage is near zero too, and the drift is not a demographic story.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from figures import style

# The covariate families, top to bottom, with their row-label headings.
_KINDS: tuple[tuple[str, str], ...] = (
    ("ses", "Socioeconomic"),
    ("family", "Family structure"),
    ("parental", "Parental"),
    ("individual", "Individual"),
)

# The timing axes, left to right, with their short headings.
_AXES: tuple[tuple[str, str], ...] = (
    ("era", "Diagnostic era"),
    ("age_at_diagnosis", "Age at diagnosis"),
)


def _human_n(value: float) -> str:
    """Return a compact sample-size label, for example ``11.6k`` or ``820``."""
    return f"{value / 1000:.1f}k" if value >= 1000 else f"{int(value)}"


def demographic_conditioning_figure(
    tables: dict[str, pd.DataFrame],
    meta: dict,
    *,
    width_in: float = 8.2,
) -> Figure:
    """Build the demographic conditioning heatmap: shrinkage per class, with the axis-span ceiling.

    Parameters
    ----------
    tables : dict of str to pandas.DataFrame
        The ``demographic_conditioning_<axis>`` frame per timing axis: per covariate and reference
        class, the ``shrinkage``, the ``axis_r2`` (constant across a covariate's classes), the
        covariate ``label``, ``kind``, and ``coding``, and the joined sample size ``n_joint``.
    meta : dict
        The run metrics; unused beyond documenting provenance.
    width_in : float, optional
        Figure width in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The row-aligned figure: a covariate-family colour strip, the axis-$R^2$ ceiling panel, and
        the shrinkage panel (four classes for each of the two axes, divided).
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    axes_present = [a for a, _ in _AXES if a in tables]
    first = tables[axes_present[0]]
    classes = sorted(int(c) for c in first["ref_class"].unique())
    class_names = first.drop_duplicates("ref_class").set_index("ref_class")["class_name"].to_dict()

    # One row per covariate, grouped by family and ordered within a family by the strongest
    # shrinkage it reaches on either axis, so the biggest mover in each block sits on top.
    meta_rows = pd.concat(tables.values()).drop_duplicates("name").set_index("name")
    peak = (
        pd.concat(tables.values())
        .assign(abs_shrink=lambda d: d["shrinkage"].abs())
        .groupby("name")["abs_shrink"]
        .max()
    )
    ordered: list[str] = []
    kinds_present: list[tuple[str, str]] = []
    for kind, heading in _KINDS:
        block = [n for n in meta_rows.index if meta_rows.loc[n, "kind"] == kind]
        if not block:
            continue
        block.sort(key=lambda n: float(peak.get(n, 0.0)), reverse=True)
        ordered += block
        kinds_present.append((kind, heading))

    colours = {kind: style.PALETTE[i % len(style.PALETTE)] for i, (kind, _) in enumerate(_KINDS)}

    def value_at(axis: str, name: str, cls: int) -> float:
        frame = tables.get(axis)
        if frame is None:
            return float("nan")
        cell = frame[(frame["name"] == name) & (frame["ref_class"] == cls)]["shrinkage"]
        return float(cell.iloc[0]) if not cell.empty else float("nan")

    def axis_r2_at(axis: str, name: str) -> float:
        frame = tables.get(axis)
        if frame is None:
            return float("nan")
        cell = frame[frame["name"] == name]["axis_r2"]
        return float(cell.iloc[0]) if not cell.empty else float("nan")

    def coverage(name: str) -> float:
        cells = pd.concat(tables.values())
        cell = cells[cells["name"] == name]["n_joint"]
        return float(cell.min()) if not cell.empty else float("nan")

    shrink = np.array([[value_at(a, n, c) for a in axes_present for c in classes] for n in ordered])
    r2 = np.array([[axis_r2_at(a, n) for a in axes_present] for n in ordered])
    shrink_cap = max(float(np.nanmax(shrink)) if np.isfinite(shrink).any() else 0.1, 0.02)
    r2_cap = max(float(np.nanmax(r2)) if np.isfinite(r2).any() else 0.01, 1e-3)

    row_labels = [f"{meta_rows.loc[n, 'label']}  ({_human_n(coverage(n))})" for n in ordered]
    boundaries = _kind_boundaries([meta_rows.loc[n, "kind"] for n in ordered])
    n_class = len(classes)

    with style.house_style():
        height_in = 0.34 * len(ordered) + 1.6
        fig = plt.figure(figsize=(width_in, height_in))
        grid = GridSpec(
            1,
            3,
            width_ratios=(0.05, 0.20 * len(axes_present), n_class * len(axes_present) * 0.16),
            wspace=0.04,
            figure=fig,
        )

        # The covariate-family colour strip.
        ax_side = fig.add_subplot(grid[0, 0])
        band = np.array([[list(dict(_KINDS)).index(meta_rows.loc[n, "kind"])] for n in ordered])
        ax_side.imshow(
            band,
            aspect="auto",
            cmap=ListedColormap([colours[k] for k, _ in _KINDS]),
            vmin=0,
            vmax=len(_KINDS) - 1,
        )
        ax_side.set_xticks([])
        ax_side.set_yticks(range(len(ordered)))
        ax_side.set_yticklabels(row_labels, fontsize=7.5)
        for _, lo, _hi in boundaries[1:]:
            ax_side.axhline(lo - 0.5, color="white", linewidth=1.0)

        # The axis-R^2 ceiling panel.
        ax_r2 = fig.add_subplot(grid[0, 1])
        ax_r2.imshow(r2, aspect="auto", cmap="Oranges", vmin=0.0, vmax=r2_cap)
        ax_r2.set_yticks([])
        ax_r2.set_xticks(range(len(axes_present)))
        ax_r2.set_xticklabels(
            [dict(_AXES)[a] for a in axes_present], rotation=45, ha="right", fontsize=7
        )
        ax_r2.grid(False)
        ax_r2.tick_params(length=0)
        for i in range(len(ordered)):
            for j in range(len(axes_present)):
                if np.isfinite(r2[i, j]):
                    ax_r2.text(
                        j,
                        i,
                        f"{r2[i, j]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if r2[i, j] > 0.5 * r2_cap else "#333333",
                    )
        for _, lo, _hi in boundaries[1:]:
            ax_r2.axhline(lo - 0.5, color="white", linewidth=1.0)
        ax_r2.set_title(r"Axis $R^2$" + "\n(ceiling)", fontsize=7.5)

        # The shrinkage panel: four classes for each axis, divided.
        ax = fig.add_subplot(grid[0, 2])
        image = ax.imshow(shrink, aspect="auto", cmap="Blues", vmin=0.0, vmax=shrink_cap)
        image.cmap.set_bad("#f2f2f2")
        ax.set_yticks([])
        ax.set_xticks(range(shrink.shape[1]))
        ax.set_xticklabels(
            [class_names.get(c, str(c)) for _a in axes_present for c in classes],
            rotation=45,
            ha="right",
            fontsize=6.5,
        )
        ax.grid(False)
        ax.tick_params(length=0)
        for i in range(len(ordered)):
            for j in range(shrink.shape[1]):
                if np.isfinite(shrink[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{shrink[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if shrink[i, j] > 0.5 * shrink_cap else "#333333",
                    )
        for _, lo, _hi in boundaries[1:]:
            ax.axhline(lo - 0.5, color="white", linewidth=1.0)
        for i in range(1, len(axes_present)):
            ax.axvline(i * n_class - 0.5, color=style.REFERENCE_COLOUR, linewidth=1.4)
        for i, axis in enumerate(axes_present):
            ax.text(
                (i + 0.5) * n_class - 0.5,
                -0.9,
                dict(_AXES)[axis],
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        ax.set_title("Shrinkage: fraction of class drift explained", fontsize=8, loc="left", pad=18)

        bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        bar.set_label("shrinkage", fontsize=7.5)
        bar.ax.tick_params(labelsize=7)

        handles = [
            plt.Line2D(
                [], [], marker="s", linestyle="none", markersize=7, color=colours[k], label=h
            )
            for k, h in kinds_present
        ]
        fig.legend(
            handles=handles,
            title="Covariate family",
            loc="lower center",
            ncol=min(len(handles), 4),
            fontsize=7,
            title_fontsize=7,
            bbox_to_anchor=(0.5, -0.02),
        )
    return fig


def _kind_boundaries(sequence: list[str]) -> list[tuple[str, int, int]]:
    """Return each contiguous run in a sequence as ``(value, start, stop)`` (stop exclusive)."""
    runs: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(sequence) + 1):
        if i == len(sequence) or sequence[i] != sequence[start]:
            runs.append((sequence[start], start, i))
            start = i
    return runs
