"""The model-selection figure: information criteria across the number of classes.

Built from the summary table of an ``analysis select`` run, the figure tells the
number-of-classes story in three panels. The information criteria fall and formally minimise
far out along the grid (panel a), the cross-validated log-likelihood elbows at the chosen
four classes (panel b), and higher-class solutions degenerate into tiny, poorly separated
classes (panel c). A reference line marks the four classes chosen by Litman et al.

The naive Lo-Mendell-Rubin proxy that :mod:`analysis.selection` reports is left out, because
that module documents it as not the analytically correct test.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from figures import style

# Short labels for the information criteria a panel can draw.
_CRITERION_LABELS: dict[str, str] = {
    "bic": "BIC",
    "aic": "AIC",
    "caic": "CAIC",
    "sabic": "SABIC",
    "awe": "AWE",
}


def _required_columns(criteria: Sequence[str]) -> set[str]:
    """Return the summary columns the figure needs for the given criteria."""
    columns = {"n_components"}
    quantities = (*criteria, "val_log_likelihood", "smallest_class_proportion")
    for name in quantities:
        columns |= {f"{name}_mean", f"{name}_std"}
    return columns


def _plot_mean_band(
    ax: Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    colour: str,
    label: str,
    marker: str = "o",
    alpha: float = 1.0,
) -> None:
    """Draw a mean line with a plus-or-minus one standard deviation band on an axis."""
    ax.plot(x, mean, color=colour, marker=marker, label=label, alpha=alpha)
    ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.15, linewidth=0.0)


def _add_reference_line(ax: Axes, k: int, *, label: str | None = None) -> None:
    """Draw a dashed vertical reference line at ``k`` classes, optionally labelled."""
    ax.axvline(k, color=style.REFERENCE_COLOUR, linestyle="--", linewidth=1.0, zorder=0)
    if label is not None:
        ax.text(
            k,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            color=style.REFERENCE_COLOUR,
        )


def selection_figure(
    summary: pd.DataFrame,
    *,
    reference_k: int = 4,
    criteria: Sequence[str] = ("bic", "aic", "caic"),
    comparison: pd.DataFrame | None = None,
) -> Figure:
    """Build the model-selection figure from a ``select`` summary table.

    Parameters
    ----------
    summary : pandas.DataFrame
        The per-component summary from :mod:`analysis.selection`, with an ``n_components``
        column and ``<name>_mean`` / ``<name>_std`` columns for each criterion drawn.
    reference_k : int, default 4
        The number of classes to mark with a reference line (the Litman choice).
    criteria : sequence of str, optional
        The information criteria to draw in the first panel; the first is emphasised.
        Defaults to ``("bic", "aic", "caic")``. Ignored when ``comparison`` is given, where the
        first panel shows the leading criterion alone for both conditions.
    comparison : pandas.DataFrame, optional
        A second condition's summary (the V9 subset). When given, each panel overlays the two
        conditions, the full release solid and the subset dashed, so the effect of cutting the
        cohort back to V9 on model selection is visible. The first panel then shows the leading
        criterion (BIC) for both conditions rather than the three criteria of one.

    Returns
    -------
    matplotlib.figure.Figure
        A three-panel figure: the information criteria, the cross-validated log-likelihood,
        and the smallest-class proportion.

    Raises
    ------
    ValueError
        When ``summary`` is missing a column the figure needs.
    """
    missing = _required_columns(criteria) - set(summary.columns)
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")

    summary = summary.sort_values("n_components")
    k = summary["n_components"].to_numpy()
    k_int = k.astype(int)
    comp = comparison.sort_values("n_components") if comparison is not None else None
    full_colour, v9_colour = style.PALETTE[0], style.PALETTE[2]

    def _mark_min(ax: Axes, x: np.ndarray, mean: np.ndarray, colour: str) -> None:
        if np.isfinite(mean).any():
            argmin = int(np.nanargmin(mean))
            ax.scatter(x[argmin], mean[argmin], color=colour, marker="v", s=28, zorder=5)

    with style.house_style():
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
        ax_ic, ax_ll, ax_quality = axes

        # Panel (a): the information criteria, which keep falling and formally minimise far
        # out along the grid in a sample this large. With a comparison, the leading criterion
        # is shown for both conditions instead of three criteria of one.
        if comp is None:
            for index, name in enumerate(criteria):
                colour = style.PALETTE[index % len(style.PALETTE)]
                mean = summary[f"{name}_mean"].to_numpy()
                std = summary[f"{name}_std"].to_numpy()
                _plot_mean_band(
                    ax_ic,
                    k,
                    mean,
                    std,
                    colour=colour,
                    label=_CRITERION_LABELS.get(name, name.upper()),
                    alpha=1.0 if index == 0 else 0.6,
                )
                _mark_min(ax_ic, k, mean, colour)
        else:
            # The criterion scales with sample size, so the two cohorts sit at very different
            # absolute levels; each is normalised within its own curve to [0, 1] so that where
            # it minimises is comparable.
            lead = criteria[0]
            comp_k = comp["n_components"].to_numpy()

            def _scaled(values: np.ndarray) -> np.ndarray:
                spread = np.nanmax(values) - np.nanmin(values)
                return (
                    (values - np.nanmin(values)) / spread if spread > 0 else np.zeros_like(values)
                )

            full_mean = _scaled(summary[f"{lead}_mean"].to_numpy())
            comp_mean = _scaled(comp[f"{lead}_mean"].to_numpy())
            ax_ic.plot(k, full_mean, color=full_colour, marker="o", label="Full 2026")
            ax_ic.plot(
                comp_k, comp_mean, color=v9_colour, marker="o", linestyle="--", label="V9 subset"
            )
            _mark_min(ax_ic, k, full_mean, full_colour)
            _mark_min(ax_ic, comp_k, comp_mean, v9_colour)
        ax_ic.set_ylabel(
            f"{_CRITERION_LABELS.get(criteria[0], criteria[0].upper())} (scaled within cut)"
            if comp is not None
            else "Criterion (lower is better)"
        )
        ax_ic.legend(loc="upper right")
        _add_reference_line(ax_ic, reference_k, label=f"Litman $K={reference_k}$")

        # Panel (b): the cross-validated log-likelihood, the out-of-sample fit, which gains
        # little past the reference choice.
        ll_mean = summary["val_log_likelihood_mean"].to_numpy()
        ll_std = summary["val_log_likelihood_std"].to_numpy()
        _plot_mean_band(
            ax_ll,
            k,
            ll_mean,
            ll_std,
            colour=full_colour,
            label="Full 2026" if comp is not None else "Validation log-likelihood",
        )
        if comp is not None:
            ax_ll.plot(
                comp["n_components"].to_numpy(),
                comp["val_log_likelihood_mean"].to_numpy(),
                color=v9_colour,
                marker="o",
                linestyle="--",
                label="V9 subset",
            )
        ax_ll.set_ylabel("Validation log-likelihood")
        if reference_k in k_int.tolist() and comp is None:
            position = int(np.where(k_int == reference_k)[0][0])
            ax_ll.scatter(
                k[position],
                ll_mean[position],
                color=style.PALETTE[3],
                s=30,
                zorder=5,
                label="elbow",
            )
        ax_ll.legend(loc="lower right")
        _add_reference_line(ax_ll, reference_k)

        # Panel (c): the smallest class proportion, which collapses towards zero as classes are
        # added; it is class size, not classification certainty, that marks the higher-class
        # solutions as not interpretable.
        prop_mean = summary["smallest_class_proportion_mean"].to_numpy()
        prop_std = summary["smallest_class_proportion_std"].to_numpy()
        _plot_mean_band(
            ax_quality,
            k,
            prop_mean,
            prop_std,
            colour=full_colour,
            label="Full 2026" if comp is not None else "smallest class",
        )
        if comp is not None:
            ax_quality.plot(
                comp["n_components"].to_numpy(),
                comp["smallest_class_proportion_mean"].to_numpy(),
                color=v9_colour,
                marker="o",
                linestyle="--",
                label="V9 subset",
            )
            ax_quality.legend(loc="upper right")
        ax_quality.set_ylabel("Smallest class proportion")
        ax_quality.set_ylim(0.0, 1.05)
        _add_reference_line(ax_quality, reference_k)

        for ax in (ax_ic, ax_ll, ax_quality):
            ax.set_xlabel("Number of latent classes $K$")
            ax.set_xticks(k_int.tolist())
        style.panel_title(ax_ic, "A", "Information criteria")
        style.panel_title(ax_ll, "B", "Cross-validated log-likelihood")
        style.panel_title(ax_quality, "C", "Smallest class proportion")

        fig.tight_layout()
    return fig
