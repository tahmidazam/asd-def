r"""The reproduction figure: recovered class signatures against the published profile.

Built from an ``analysis align`` run, the figure puts each recovered class signature beside
the value read from figure 1b of Litman et al. (2025). One panel per named class shows the
seven-category profile two ways: the recovered signature (solid) and the published target
(dashed). The panels are ordered by published class size, and each title carries the class
proportion (recovered against published) and the per-class profile correlation, or a note
that the class is anchor-confirmed where its published profile is saturated and the
correlation is uninformative.

The figure is the visual form of the reproduction benchmark: a clean match in the
developmental-led and saturated classes, and the one real divergence (Social or behavioural
showing weaker social-communication and restricted-or-repetitive enrichment than the paper)
visible rather than hidden.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from figures import style

# The published class order, largest to smallest, so the panels read in the order the paper
# presents the classes.
_PANEL_ORDER: tuple[str, ...] = (
    "Social/behavioral",
    "Moderate challenges",
    "Mixed ASD with DD",
    "Broadly affected",
)


def _panel_subtitle(
    our_proportion: float | None,
    published_proportion: float | None,
    correlation: float | None,
    comparison_proportion: float | None = None,
) -> str:
    """Compose the per-panel annotation of class size and profile correlation.

    With a comparison proportion, the size line reads ``full / V9 vs published``; otherwise
    ``recovered vs published``.
    """
    parts: list[str] = []
    if our_proportion is not None and published_proportion is not None:
        if comparison_proportion is not None:
            parts.append(
                f"{our_proportion:.0%} / {comparison_proportion:.0%} vs {published_proportion:.0%}"
            )
        else:
            parts.append(f"{our_proportion:.0%} vs {published_proportion:.0%}")
    parts.append(f"$r = {correlation:.2f}$" if correlation is not None else "anchor-confirmed")
    return ", ".join(parts)


def reproduction_figure(
    our_signature: pd.DataFrame,
    published_signature: pd.DataFrame,
    alignment: dict,
    our_proportions: dict[int, float],
    published_proportions: dict[str, float],
    comparison: dict | None = None,
) -> Figure:
    """Build the reproduction figure from an ``align`` run.

    Parameters
    ----------
    our_signature : pandas.DataFrame
        The recovered class-by-category signature, indexed by class id, one column per
        category.
    published_signature : pandas.DataFrame
        The published figure-1b signature, indexed by named class, with the same columns.
    alignment : dict
        The alignment record, with ``mapping`` (class id to named class), ``correlations``
        (class id to a per-class profile correlation or ``None``), ``overall_correlation``,
        and ``anchors_hold``.
    our_proportions : dict of int to float
        The recovered class proportions, by class id.
    published_proportions : dict of str to float
        The published class proportions, by named class.
    comparison : dict, optional
        A second condition's reproduction (the V9 subset), with keys ``signature``,
        ``alignment``, and ``proportions`` mirroring the primary arguments. When given, each
        panel adds the subset's recovered signature and the size line reads
        ``full / V9 vs published``.

    Returns
    -------
    matplotlib.figure.Figure
        A two-by-two figure, one panel per named class, each overlaying the recovered and
        published seven-category profiles.

    Raises
    ------
    ValueError
        When the two signatures differ in their categories, or the alignment names a class
        the published signature does not carry.
    """
    if list(our_signature.columns) != list(published_signature.columns):
        raise ValueError("the recovered and published signatures must share their categories")

    categories = list(our_signature.columns)
    labels = [style.CATEGORY_LABELS.get(cat, cat) for cat in categories]
    positions = np.arange(len(categories))
    # Map each named class back to the recovered class id that was aligned to it.
    name_to_id = {name: int(cid) for cid, name in alignment["mapping"].items()}
    correlations = {int(cid): value for cid, value in alignment["correlations"].items()}
    missing = [name for name in _PANEL_ORDER if name not in published_signature.index]
    if missing:
        raise ValueError(f"published signature is missing classes: {missing}")

    comp_name_to_id: dict[str, int] = {}
    if comparison is not None:
        comp_name_to_id = {
            name: int(cid) for cid, name in comparison["alignment"]["mapping"].items()
        }

    full_colour, v9_colour = style.PALETTE[0], style.PALETTE[2]

    with style.house_style():
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), sharex=True, sharey=True)
        for ax, name, letter in zip(axes.flat, _PANEL_ORDER, ("A", "B", "C", "D"), strict=True):
            cid = name_to_id[name]
            published = published_signature.loc[name].to_numpy(dtype=float)
            recovered = our_signature.loc[cid].to_numpy(dtype=float)

            ax.axhline(0.0, color=style.REFERENCE_COLOUR, linewidth=0.6, zorder=0)
            ax.plot(
                positions,
                published,
                color=style.REFERENCE_COLOUR,
                linestyle="--",
                marker="o",
                markerfacecolor="white",
                label="published (figure 1b)",
                zorder=2,
            )
            ax.plot(
                positions,
                recovered,
                color=full_colour,
                marker="o",
                label="recovered (full 2026)",
                zorder=3,
            )
            comp_proportion = None
            if comparison is not None and name in comp_name_to_id:
                comp_cid = comp_name_to_id[name]
                comp_recovered = comparison["signature"].loc[comp_cid].to_numpy(dtype=float)
                ax.plot(
                    positions,
                    comp_recovered,
                    color=v9_colour,
                    marker="o",
                    markersize=4,
                    label="recovered (V9 subset)",
                    zorder=4,
                )
                comp_proportion = comparison["proportions"].get(comp_cid)
            # The class proportions and per-class correlation annotate each panel without
            # crowding the title, in the top-right corner kept readable by a light box.
            ax.text(
                0.97,
                0.95,
                _panel_subtitle(
                    our_proportions.get(cid),
                    published_proportions.get(name),
                    correlations.get(cid),
                    comp_proportion,
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            )
            style.panel_title(ax, letter, name)
            ax.set_ylim(-1.2, 1.2)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha="right")

        for ax in axes[:, 0]:
            ax.set_ylabel("Signed enrichment")
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", ncol=2, fontsize=7)

        overall = float(alignment["overall_correlation"])
        ci = alignment.get("overall_correlation_ci")
        ci_text = ""
        if isinstance(ci, dict) and ci.get("n_valid"):
            ci_text = f", 95% CI [{float(ci['ci_low']):.2f}, {float(ci['ci_high']):.2f}]"
        fig.suptitle(
            f"Recovered and published class signatures (overall $r = {overall:.2f}${ci_text})",
            y=1.0,
        )
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
    return fig
