"""The cross-cohort replication figure: SPARK class signatures projected onto the SSC.

Built from a ``replicate`` run, the figure shows how closely the seven-category class
signatures agree between the SPARK fit and its projection onto the SSC, against the values
Litman et al. (2025) report. Panel (a) scatters every class-by-category value, SSC against
SPARK, around the line of equality; panel (b) gives the per-category correlation with the
published per-category coefficients overlaid and a proband-bootstrap 95 per cent interval on
each. Each per-category coefficient is taken over the four classes alone, so it shifts easily
when one class moves and its interval is wide; the overall correlation, over all 28
class-by-category points, is the stabler quantity. The developmental category sits
below the rest, a gap the replication investigation examines rather than ascribes to a single
cause.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from figures import style

# Litman et al. (2025) reported SSC-replication values, overlaid as the reference our
# projection is read against. The overall r = 0.927 (P < 1e-4) is from the Results text and
# Extended Data Fig. 6, where their model was trained on 6,393 SPARK probands and applied to
# 861 in the SSC. The per-category coefficients are quoted in the Methods ("Phenotypic
# replication in the SSC"); the Methods gives the seven values without naming an order, so
# they are matched to the category order of figure 2c (the released features_to_visualize),
# which is SEVEN_CATEGORIES order.
LITMAN_OVERALL_R: float = 0.927
LITMAN_CATEGORY_R: dict[str, float] = {
    "anxiety/mood": 0.98,
    "attention": 0.92,
    "disruptive behavior": 0.94,
    "self-injury": 0.92,
    "social/communication": 0.89,
    "restricted/repetitive": 0.97,
    "developmental": 0.98,
}


def replication_figure(
    spark_signature: pd.DataFrame,
    ssc_signature: pd.DataFrame,
    metrics: dict,
    comparison: dict | None = None,
) -> Figure:
    """Build the cross-cohort replication figure from a ``replicate`` run.

    Parameters
    ----------
    spark_signature, ssc_signature : pandas.DataFrame
        The class-by-category signature matrices (one row per class, one column per category)
        for the SPARK fit and its SSC projection. They must share shape and columns.
    metrics : dict
        The replication metrics for the primary condition (the full release), with
        ``overall_correlation``, ``category_correlation`` (one entry per category), and
        optionally ``n_ssc``.
    comparison : dict, optional
        A second condition's metrics (the V9-subset projection), with the same keys. When
        given, the per-category panel groups the two conditions side by side against the
        published values, so the effect of cutting the training cohort back to V9 is visible.

    Returns
    -------
    matplotlib.figure.Figure
        A two-panel figure: the signature scatter and the per-category correlation.

    Raises
    ------
    ValueError
        When the two signatures differ in shape or columns, or a metric is missing.
    """
    same_columns = list(spark_signature.columns) == list(ssc_signature.columns)
    if not same_columns or spark_signature.shape != ssc_signature.shape:
        raise ValueError("the SPARK and SSC signatures must share shape and columns")
    for key in ("overall_correlation", "category_correlation"):
        if key not in metrics:
            raise ValueError(f"metrics is missing {key!r}")

    categories = list(spark_signature.columns)
    colours = {cat: style.PALETTE[i % len(style.PALETTE)] for i, cat in enumerate(categories)}
    overall = float(metrics["overall_correlation"])
    # Condition colours for the per-category panel: the full release and the V9 subset.
    full_colour, v9_colour = style.PALETTE[0], style.PALETTE[2]

    with style.house_style():
        fig, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(8.4, 3.9))

        # Panel (a): every class-by-category value, SSC against SPARK, around y = x.
        ax_scatter.axline(
            (0, 0), slope=1, color=style.REFERENCE_COLOUR, linestyle="--", linewidth=1.0, zorder=0
        )
        for cat in categories:
            ax_scatter.scatter(
                spark_signature[cat].to_numpy(),
                ssc_signature[cat].to_numpy(),
                color=colours[cat],
                label=style.CATEGORY_LABELS.get(cat, cat),
                s=30,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        ax_scatter.set_xlim(-1.08, 1.08)
        ax_scatter.set_ylim(-1.08, 1.08)
        ax_scatter.set_aspect("equal")
        ax_scatter.set_xlabel("SPARK signature")
        ax_scatter.set_ylabel("SSC signature")
        # The overall correlation, its interval, and the sample size annotate the panel
        # rather than the title, in the lower-right corner the scatter leaves clear.
        n_ssc = metrics.get("n_ssc")
        n_text = f", $n = {int(n_ssc)}$" if n_ssc is not None else ""
        ci = metrics.get("overall_correlation_ci")
        ci_text = ""
        if isinstance(ci, dict) and ci.get("n_valid"):
            ci_text = f" [{float(ci['ci_low']):.2f}, {float(ci['ci_high']):.2f}]"
        ax_scatter.text(
            0.97,
            0.03,
            f"$r = {overall:.2f}${ci_text}{n_text}\nLitman 2025: $r = {LITMAN_OVERALL_R:.3f}$",
            transform=ax_scatter.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
        ax_scatter.legend(loc="upper left", ncol=2, fontsize=6)

        # Panel (b): the per-category correlation for each training condition, with Litman et
        # al.'s published values overlaid as the reference each bar is read against. The
        # whiskers are the proband-bootstrap 95 per cent interval on each per-category
        # correlation; taken over four classes, these intervals are wide.
        def _values(source: dict) -> list[float]:
            per_cat = source["category_correlation"]
            return [
                float(per_cat[cat]) if per_cat.get(cat) is not None else np.nan
                for cat in categories
            ]

        def _errors(source: dict, values: list[float]) -> np.ndarray | None:
            # Asymmetric bar whiskers [[below], [above]] from the per-category bootstrap
            # interval, or None when the run predates it.
            ci = source.get("overall_correlation_ci")
            per_cat_ci = ci.get("category") if isinstance(ci, dict) else None
            if not isinstance(per_cat_ci, dict):
                return None
            below, above = [], []
            for value, cat in zip(values, categories, strict=True):
                entry = per_cat_ci.get(cat)
                if not isinstance(entry, dict) or not entry.get("n_valid") or np.isnan(value):
                    below.append(0.0)
                    above.append(0.0)
                    continue
                below.append(max(0.0, value - float(entry["ci_low"])))
                above.append(max(0.0, float(entry["ci_high"]) - value))
            return np.array([below, above])

        bar_kw = {"error_kw": {"elinewidth": 0.8, "capsize": 2, "ecolor": "#444444"}}
        positions = np.arange(len(categories))
        if comparison is None:
            values = _values(metrics)
            ax_bar.bar(
                positions,
                values,
                color=[colours[cat] for cat in categories],
                yerr=_errors(metrics, values),
                **bar_kw,
            )
        else:
            width = 0.4
            comp_overall = float(comparison["overall_correlation"])
            full_values, comp_values = _values(metrics), _values(comparison)
            ax_bar.bar(
                positions - width / 2,
                full_values,
                width,
                color=full_colour,
                yerr=_errors(metrics, full_values),
                label=f"Full 2026 ($r = {overall:.2f}$)",
                **bar_kw,
            )
            ax_bar.bar(
                positions + width / 2,
                comp_values,
                width,
                color=v9_colour,
                yerr=_errors(comparison, comp_values),
                label=f"V9 subset ($r = {comp_overall:.2f}$)",
                **bar_kw,
            )
        ax_bar.scatter(
            positions,
            [LITMAN_CATEGORY_R[cat] for cat in categories],
            marker="D",
            s=26,
            color=style.REFERENCE_COLOUR,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
            label=f"Litman 2025 ($r = {LITMAN_OVERALL_R:.3f}$)",
        )
        ax_bar.set_xticks(positions)
        ax_bar.set_xticklabels(
            [style.CATEGORY_LABELS.get(cat, cat) for cat in categories], rotation=45, ha="right"
        )
        ax_bar.set_ylim(0.0, 1.05)
        ax_bar.set_ylabel("Profile correlation, SPARK vs SSC")
        ax_bar.legend(loc="lower left", fontsize=6)

        style.panel_title(ax_scatter, "A", "Class signatures")
        style.panel_title(ax_bar, "B", "Per-category profile correlation")
        fig.tight_layout()
    return fig
