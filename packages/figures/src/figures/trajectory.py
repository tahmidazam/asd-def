"""The class-trajectory figure: each class's path through the strata in discriminant space.

Built from a ``trajectory`` run, the figure shows, in one panel per class, where the class
sits in the pooled four-class discriminant space and how its centroid moves across the strata
of the axis (age at diagnosis or diagnostic era). The focal class's members are drawn as nested
grey Gaussian coverage contours from 50 to 95 per cent, the tighter ones more opaque, so the
shading shows where its probands concentrate without plotting any individual; the other three
classes are marked by their centroid alone. The stratum centroids are coloured from the first
stratum to the last, with a red ring where membership reorganised (Jaccard below 0.5); an arrow
marks the net displacement from the first third of the strata to the last. The projection is a
linear discriminant embedding, so positions and distances are honest, but it is an
illustration: the drift claim rests on the full-dimensional statistics of the drift and
roughness stages, not on this picture.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, FancyArrowPatch

from figures import style

_LETTERS = ("A", "B", "C", "D")
# Coverage levels drawn as nested grey contours of the class's member density, from 50 to 95
# per cent. Each is a Gaussian coverage ellipse; the tighter (inner) contour is drawn more
# opaque, so the shading falls off outward the way a density does.
_COVERAGE_LEVELS = (0.50, 0.68, 0.80, 0.95)
# Mahalanobis radius of the widest (95 per cent) contour, for the axis-limit padding.
_COVERAGE_K = 2.4477


def _coverage_radius(coverage: float) -> float:
    """Return the Mahalanobis radius of a coverage level under a bivariate normal."""
    return float(np.sqrt(-2.0 * np.log(1.0 - coverage)))


def _coverage_alpha(coverage: float) -> float:
    """Return the contour opacity, higher for the tighter (inner) levels."""
    lo, hi = _COVERAGE_LEVELS[0], _COVERAGE_LEVELS[-1]
    return float(0.55 - (coverage - lo) / (hi - lo) * (0.55 - 0.12))


def _coverage_ellipse(
    mean: np.ndarray, cov: np.ndarray, k: float, colour: str, **kwargs: object
) -> Ellipse:
    """Return a Gaussian coverage ellipse of a class from its LD covariance, at radius ``k``."""
    values, vectors = np.linalg.eigh(cov)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
    width, height = 2.0 * k * np.sqrt(np.maximum(values, 0.0))
    return Ellipse(
        tuple(mean),
        float(width),
        float(height),
        angle=angle,
        facecolor="none",
        edgecolor=colour,
        **kwargs,
    )


def trajectory_figure(embedding: pd.DataFrame, meta: dict) -> Figure:
    """Build the class-trajectory figure from a ``trajectory`` run.

    Parameters
    ----------
    embedding : pandas.DataFrame
        The ``embedding_<axis>`` table: one row per anchor and per (class, stratum), with
        ``kind``, ``ref_class``, ``class_name``, ``order``, ``ld1``, ``ld2``, ``jaccard``, and
        ``reorganised``.
    meta : dict
        The run's manifest metrics, carrying ``axis`` and ``n_strata``.

    Returns
    -------
    matplotlib.figure.Figure
        A 2 by 2 figure, one panel per class.

    Raises
    ------
    ValueError
        When the embedding table is missing a required column.
    """
    required = {"kind", "ref_class", "class_name", "order", "ld1", "ld2", "reorganised"}
    missing = required - set(embedding.columns)
    if missing:
        raise ValueError(f"embedding is missing columns: {sorted(missing)}")

    anchors = embedding[embedding["kind"] == "anchor"].set_index("ref_class")
    strata = embedding[embedding["kind"] == "stratum"]
    has_cov = {"cov11", "cov12", "cov22"} <= set(embedding.columns)

    def _cov(cls: int) -> np.ndarray:
        row = anchors.loc[cls]
        return np.array([[row["cov11"], row["cov12"]], [row["cov12"], row["cov22"]]], dtype=float)

    classes = sorted(anchors.index)
    names = {int(c): str(anchors.loc[c, "class_name"]) for c in classes}
    colours = {int(c): style.PALETTE[i % len(style.PALETTE)] for i, c in enumerate(classes)}
    n_strata = int(meta.get("n_strata", strata["order"].max() + 1))

    both = embedding[["ld1", "ld2"]].to_numpy(dtype=float)
    xlo, xhi = both[:, 0].min(), both[:, 0].max()
    ylo, yhi = both[:, 1].min(), both[:, 1].max()
    if has_cov:
        for cls in classes:
            centre = anchors.loc[cls, ["ld1", "ld2"]].to_numpy(dtype=float)
            rx = _COVERAGE_K * float(np.sqrt(max(anchors.loc[cls, "cov11"], 0.0)))
            ry = _COVERAGE_K * float(np.sqrt(max(anchors.loc[cls, "cov22"], 0.0)))
            xlo, xhi = min(xlo, centre[0] - rx), max(xhi, centre[0] + rx)
            ylo, yhi = min(ylo, centre[1] - ry), max(yhi, centre[1] + ry)
    xlim = (xlo - 0.4, xhi + 0.4)
    ylim = (ylo - 0.4, yhi + 0.5)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(0, max(n_strata - 1, 1))
    nice = "age at diagnosis" if meta.get("axis") == "age_at_diagnosis" else "diagnostic era"

    with style.house_style():
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4), sharex=True, sharey=True)
        for panel, (letter, ax) in enumerate(zip(_LETTERS, axes.flat, strict=False)):
            if panel >= len(classes):
                ax.axis("off")
                continue
            focal = int(classes[panel])
            colour = colours[focal]
            for other in classes:
                if other == focal:
                    continue
                point = anchors.loc[other, ["ld1", "ld2"]].to_numpy(dtype=float)
                ax.scatter(*point, s=10, color="#9a9a9a", zorder=1.5)
                ax.annotate(
                    names[int(other)].split()[0],
                    point,
                    fontsize=6,
                    color="#9a9a9a",
                    ha="center",
                    va="center",
                    zorder=2,
                )
            path = strata[strata["ref_class"] == focal].sort_values("order")
            points = path[["ld1", "ld2"]].to_numpy(dtype=float)
            order = path["order"].to_numpy()
            reorganised = path["reorganised"].to_numpy(dtype=bool)
            ax.plot(points[:, 0], points[:, 1], color=colour, lw=0.7, alpha=0.30, zorder=3)
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=np.where(reorganised, 62, 38),
                c=order,
                cmap=cmap,
                norm=norm,
                edgecolor=np.where(reorganised, "#d62728", "white"),
                linewidth=np.where(reorganised, 1.4, 0.5),
                zorder=5,
            )
            third = max(2, len(points) // 3)
            ax.add_patch(
                FancyArrowPatch(
                    points[:third].mean(axis=0),
                    points[-third:].mean(axis=0),
                    arrowstyle="-|>",
                    mutation_scale=18,
                    color=colour,
                    lw=2.2,
                    alpha=0.9,
                    zorder=6,
                )
            )
            anchor = anchors.loc[focal, ["ld1", "ld2"]].to_numpy(dtype=float)
            if has_cov:
                for coverage in _COVERAGE_LEVELS:
                    ax.add_patch(
                        _coverage_ellipse(
                            anchor,
                            _cov(focal),
                            _coverage_radius(coverage),
                            "#6f6f6f",
                            lw=1.0,
                            zorder=1.1,
                            alpha=_coverage_alpha(coverage),
                        )
                    )
            ax.scatter(*anchor, s=22, color=colour, edgecolor="black", lw=0.5, zorder=1.4)
            style.panel_title(ax, letter, names[focal])
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            n_reorg = int(reorganised.sum())
            ax.text(
                0.98,
                0.03,
                f"{n_reorg} of {len(points)} strata reorganised",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.3,
                color="#555",
            )
        for ax in axes[1, :]:
            ax.set_xlabel("Linear discriminant 1")
        for ax in axes[:, 0]:
            ax.set_ylabel("Linear discriminant 2")
        smap = ScalarMappable(norm=norm, cmap=cmap)
        smap.set_array([])
        colourbar = fig.colorbar(smap, ax=axes, fraction=0.03, pad=0.02)
        colourbar.set_label(f"Stratum ({nice}, first to last)", fontsize=8)
        colourbar.set_ticks([0, max(n_strata - 1, 1)])
        colourbar.set_ticklabels(["first", "last"], fontsize=7)
    return fig
