"""The H0F category-decomposition figure: what symptom categories carry the class drift.

Reads the one-fit block-attribution decomposition (plan section 7f): the per-class, per-category
separation-scaled displacement magnitudes the ``invariance-trajectory`` stage writes to
``grain_magnitude_<axis>.parquet``, and the per-feature displacements it writes to
``feature_displacement_<axis>.parquet``. The figure has two parts. The heatmaps (panels A and B)
show each class's drift split into category shares along diagnostic era and age at diagnosis, so the
concentration is legible at a glance. The per-class lollipops (panel C) name the leading features
behind the age drift, coloured by category, so the developmental milestones carrying the
developmental class and the internalizing items carrying the others are visible feature by feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.image import AxesImage

from figures import style

_NICE_AXIS = {"era": "diagnostic era", "age_at_diagnosis": "age at diagnosis"}
_N_LEADING = 5  # leading features shown per class in the lollipop panels

# The seven author symptom categories Litman et al. use for their category signature (their
# ``features_to_visualize``). Our decomposition leads with these; the remaining categories the
# feature map defines are additional CBCL problem scales the paper excludes from the signature.
_LITMAN_CATEGORIES = (
    "developmental",
    "social/communication",
    "restricted/repetitive",
    "anxiety/mood",
    "disruptive behavior",
    "attention",
    "self-injury",
)
_ADDITIONAL_CATEGORIES = ("somatic", "thought problems", "other problems")

# The order the categories are shown in: the seven Litman categories first, then the additional
# scales, then any category present but not listed (so the figure never silently drops one).
_CATEGORY_ORDER = _LITMAN_CATEGORIES + _ADDITIONAL_CATEGORIES
# The label for a feature with no author category (an instrument total, for example).
_COMPOSITE = "composite"


def _endpoint_shares(grains: pd.DataFrame) -> pd.DataFrame:
    """Return the per-class category shares at the axis endpoint from a grain-magnitude frame."""
    endpoint = grains[grains["focal_index"] == grains["focal_index"].max()]
    cats = endpoint[endpoint["grain"].str.startswith("category:")].copy()
    cats["category"] = cats["grain"].str.removeprefix("category:")
    cats["sq"] = cats["magnitude"] ** 2
    cats["share"] = cats.groupby("ref_class")["sq"].transform(lambda s: s / s.sum())
    return cats


def _category_order(present: set[str]) -> list[str]:
    """Return the fixed category order restricted to those present, with any extras appended."""
    ordered = [c for c in _CATEGORY_ORDER if c in present]
    return ordered + sorted(present - set(ordered))


def _category_of(value: object) -> str:
    """Return the category name of a feature, mapping a blank to the composite bucket."""
    return value if isinstance(value, str) and value else _COMPOSITE


def _clean_feature(name: str) -> str:
    """Return a shorter, human-readable feature label for a tick."""
    trimmed = name.removesuffix("_age_mos").removesuffix("_mos")
    return trimmed.replace("_", " ")


def _draw_heatmap(ax, matrix, class_labels, categories, letter, title) -> AxesImage:
    """Draw one class-by-category share heatmap and return the image for the colour bar."""
    ceiling = float(np.nanmax(matrix))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=ceiling)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(
        [style.CATEGORY_LABELS.get(c, c) for c in categories], rotation=45, ha="right"
    )
    ax.set_yticks(range(len(class_labels)))
    ax.set_yticklabels(class_labels)
    ax.grid(False)
    # A divider after the seven Litman categories, so the additional CBCL scales read as a group.
    n_litman = sum(1 for c in categories if c in _LITMAN_CATEGORIES)
    if 0 < n_litman < len(categories):
        ax.axvline(
            n_litman - 0.5, color=style.REFERENCE_COLOUR, linewidth=1.0, linestyle=(0, (4, 2))
        )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value) and value >= 0.10:
                ax.text(
                    j,
                    i,
                    f"{value * 100:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if value > 0.5 * ceiling else "#333333",
                )
    style.panel_title(ax, letter, title)
    return image


def category_decomposition_figure(
    grains: dict[str, pd.DataFrame],
    features: dict[str, pd.DataFrame],
    meta: dict,
) -> Figure:
    """Build the H0F category-decomposition figure across era and age at diagnosis.

    Parameters
    ----------
    grains : dict of str to pandas.DataFrame
        The ``grain_magnitude_<axis>.parquet`` frame per axis (``"era"`` and
        ``"age_at_diagnosis"``), carrying the per-class category-grain magnitudes.
    features : dict of str to pandas.DataFrame
        The ``feature_displacement_<axis>.parquet`` frame per axis, carrying the per-feature
        signed displacement, category, and false-discovery-rate decision.
    meta : dict
        Figure metadata; unused beyond documenting provenance.

    Returns
    -------
    matplotlib.figure.Figure
        The two category heatmaps and the four per-class leading-feature panels.
    """
    import matplotlib.pyplot as plt

    axes_order = [a for a in ("era", "age_at_diagnosis") if a in grains]
    shares = {a: _endpoint_shares(grains[a]) for a in axes_order}
    present: set[str] = set()
    for cats in shares.values():
        present |= set(cats["category"])
    categories = _category_order(present)

    any_cats = next(iter(shares.values()))
    classes = sorted(int(c) for c in any_cats["ref_class"].unique())
    names = any_cats.drop_duplicates("ref_class").set_index("ref_class")["class_name"].to_dict()
    class_labels = [names.get(c, str(c)) for c in classes]

    with style.house_style():
        fig = plt.figure(figsize=(9.4, 10.4))
        grid = GridSpec(3, 2, height_ratios=(1.0, 0.95, 0.95), hspace=0.46, wspace=0.28, figure=fig)

        images = []
        heatmap_axes = []
        for i, axis in enumerate(axes_order):
            ax = fig.add_subplot(grid[0, i])
            heatmap_axes.append(ax)
            wide = shares[axis].pivot(index="ref_class", columns="category", values="share")
            matrix = wide.reindex(index=classes, columns=categories).to_numpy()
            images.append(
                _draw_heatmap(ax, matrix, class_labels, categories, "AB"[i], _NICE_AXIS[axis])
            )
        if images:
            bar = fig.colorbar(images[-1], ax=heatmap_axes, fraction=0.025, pad=0.02)
            bar.set_label("share of the class's drift", fontsize=8)

        # Colour the leading-feature markers by category, assigning distinct palette colours to
        # the categories that actually appear (so no two collide), with the composite bucket grey.
        lead = _leading_categories(features, classes)
        colours = {cat: style.PALETTE[i % len(style.PALETTE)] for i, cat in enumerate(lead)}
        colours[_COMPOSITE] = style.REFERENCE_COLOUR
        used = _draw_leading_panels(fig, grid, features, names, classes, colours)

        handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=colours[c],
                label=style.CATEGORY_LABELS.get(c, c),
            )
            for c in used
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 6),
            fontsize=7,
            bbox_to_anchor=(0.5, 0.005),
        )
    return fig


def _leading_categories(features: dict[str, pd.DataFrame], classes: list[int]) -> list[str]:
    """Return the categories (composite aside) that appear in the leading age-drift features."""
    axis = "age_at_diagnosis" if "age_at_diagnosis" in features else next(iter(features))
    frame = features[axis]
    present: set[str] = set()
    for c in classes:
        sub = frame[(frame["ref_class"] == c) & frame["reject"]].copy()
        sub["abs"] = sub["displacement"].abs()
        for r in sub.sort_values("abs", ascending=False).head(_N_LEADING).itertuples():
            present.add(_category_of(r.category))
    ordered = [c for c in _CATEGORY_ORDER if c in present]
    return ordered + sorted(present - set(ordered) - {_COMPOSITE})


def _draw_leading_panels(fig, grid, features, names, classes, colours) -> list[str]:
    """Draw panel C's per-class leading-feature lollipops; return the categories used."""
    axis = "age_at_diagnosis" if "age_at_diagnosis" in features else next(iter(features))
    frame = features[axis]
    used: list[str] = []
    for idx, c in enumerate(classes):
        ax = fig.add_subplot(grid[1 + idx // 2, idx % 2])
        sub = frame[(frame["ref_class"] == c) & frame["reject"]].copy()
        sub["abs"] = sub["displacement"].abs()
        top = sub.sort_values("abs", ascending=False).head(_N_LEADING)
        y = np.arange(len(top))[::-1]
        labels: list[str] = []
        for pos, r in zip(y, top.itertuples(), strict=True):
            category = _category_of(r.category)
            if category not in used:
                used.append(category)
            colour = colours.get(category, style.REFERENCE_COLOUR)
            ax.plot([0.0, r.displacement], [pos, pos], color=colour, linewidth=1.3, zorder=1)
            ax.scatter(r.displacement, pos, color=colour, s=26, zorder=2)
            labels.append(_clean_feature(str(r.feature)))
        peak = float(top["abs"].max()) if len(top) else 1.0
        ax.axvline(0.0, color=style.REFERENCE_COLOUR, linewidth=0.8)
        ax.set_xlim(-peak * 1.2, peak * 1.2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.6)
        ax.set_ylim(-0.6, _N_LEADING - 0.4)
        ax.grid(axis="x", alpha=0.3, linewidth=0.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7)
        if idx == 0:
            style.panel_title(ax, "C", names.get(c, str(c)))
        else:
            ax.set_title(names.get(c, str(c)), loc="left", fontsize=8)
        if idx // 2 == 1:
            ax.set_xlabel("signed displacement at the age endpoint", fontsize=7.5)
    ordered = [c for c in _CATEGORY_ORDER if c in used] + [c for c in used if c == _COMPOSITE]
    return ordered
