"""The cross-cohort replication figure: SPARK class signatures projected onto the SSC.

Built from a ``replicate`` run, the figure shows how closely the seven-category class
signatures agree between the SPARK fit and its projection onto the SSC. Panel (a) scatters
every class-by-category value, SSC against SPARK, around the line of equality; panel (b) gives
the per-category correlation, where the developmental category sits well below the rest. That
gap is the SSC milestone parsing rather than the classes, as the stability figure shows the
developmental category is as stable as the others within SPARK.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from figures import style


def replication_figure(
    spark_signature: pd.DataFrame,
    ssc_signature: pd.DataFrame,
    metrics: dict,
) -> Figure:
    """Build the cross-cohort replication figure from a ``replicate`` run.

    Parameters
    ----------
    spark_signature, ssc_signature : pandas.DataFrame
        The class-by-category signature matrices (one row per class, one column per category)
        for the SPARK fit and its SSC projection. They must share shape and columns.
    metrics : dict
        The replication metrics, with ``overall_correlation``, ``category_correlation`` (one
        entry per category), and optionally ``n_ssc``.

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
    per_category = metrics["category_correlation"]

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
            f"$r = {overall:.2f}${ci_text}{n_text}",
            transform=ax_scatter.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
        ax_scatter.legend(loc="upper left", ncol=2, fontsize=6)

        # Panel (b): the per-category correlation, with developmental the clear outlier.
        values = [
            float(per_category[cat]) if per_category.get(cat) is not None else np.nan
            for cat in categories
        ]
        positions = np.arange(len(categories))
        ax_bar.bar(positions, values, color=[colours[cat] for cat in categories])
        ax_bar.axhline(
            overall,
            color=style.REFERENCE_COLOUR,
            linestyle="--",
            linewidth=1.0,
            label=f"overall $r = {overall:.2f}$",
        )
        ax_bar.set_xticks(positions)
        ax_bar.set_xticklabels(
            [style.CATEGORY_LABELS.get(cat, cat) for cat in categories], rotation=45, ha="right"
        )
        ax_bar.set_ylim(0.0, 1.05)
        ax_bar.set_ylabel("Profile correlation, SPARK vs SSC")
        ax_bar.legend(loc="lower left")

        style.panel_title(ax_scatter, "A", "Class signatures")
        style.panel_title(ax_bar, "B", "Per-category profile correlation")
        fig.tight_layout()
    return fig
