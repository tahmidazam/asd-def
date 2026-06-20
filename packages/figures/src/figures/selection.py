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
    quantities = (*criteria, "val_log_likelihood", "smallest_class_proportion", "relative_entropy")
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
        Defaults to ``("bic", "aic", "caic")``.

    Returns
    -------
    matplotlib.figure.Figure
        A three-panel figure: the information criteria, the cross-validated log-likelihood,
        and the smallest-class proportion with the relative entropy.

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

    with style.house_style():
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
        ax_ic, ax_ll, ax_quality = axes

        # Panel (a): the information criteria, which keep falling and formally minimise far
        # out along the grid in a sample this large.
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
            if np.isfinite(mean).any():
                argmin = int(np.nanargmin(mean))
                ax_ic.scatter(k[argmin], mean[argmin], color=colour, marker="v", s=25, zorder=5)
        primary = summary[f"{criteria[0]}_mean"].to_numpy()
        if np.isfinite(primary).any():
            ax_ic.set_title(
                f"Information criteria (minimum at $K={int(k[np.nanargmin(primary)])}$)"
            )
        else:
            ax_ic.set_title("Information criteria")
        ax_ic.set_ylabel("Criterion (lower is better)")
        ax_ic.legend(loc="upper right")
        _add_reference_line(ax_ic, reference_k, label=f"Litman $K={reference_k}$")

        # Panel (b): the cross-validated log-likelihood, the honest out-of-sample fit, which
        # gains little past the reference choice.
        ll_mean = summary["val_log_likelihood_mean"].to_numpy()
        ll_std = summary["val_log_likelihood_std"].to_numpy()
        _plot_mean_band(
            ax_ll, k, ll_mean, ll_std, colour=style.PALETTE[2], label="Validation log-likelihood"
        )
        ax_ll.set_title("Cross-validated fit")
        ax_ll.set_ylabel("Validation log-likelihood")
        if reference_k in k_int.tolist():
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

        # Panel (c): solution quality. The smallest class collapses towards zero as classes
        # are added, while the relative entropy stays high, so class size, not classification
        # certainty, is what marks the far-out solutions as not interpretable. Both are
        # dimensionless quantities on the unit interval, so they share one axis (a twin axis
        # would auto-scale the near-constant entropy and turn its noise into false structure).
        prop_mean = summary["smallest_class_proportion_mean"].to_numpy()
        prop_std = summary["smallest_class_proportion_std"].to_numpy()
        _plot_mean_band(
            ax_quality,
            k,
            prop_mean,
            prop_std,
            colour=style.PALETTE[0],
            label="Smallest class proportion",
        )
        ent_mean = summary["relative_entropy_mean"].to_numpy()
        ent_std = summary["relative_entropy_std"].to_numpy()
        _plot_mean_band(
            ax_quality,
            k,
            ent_mean,
            ent_std,
            colour=style.PALETTE[1],
            label="Relative entropy",
            marker="s",
        )
        ax_quality.set_ylabel("Class proportion / relative entropy")
        ax_quality.set_ylim(0.0, 1.05)
        ax_quality.set_title("Solution quality")
        ax_quality.legend(loc="center right")
        _add_reference_line(ax_quality, reference_k)

        for ax in (ax_ic, ax_ll, ax_quality):
            ax.set_xlabel("Number of latent classes $K$")
            ax.set_xticks(k_int.tolist())
        for ax, letter in ((ax_ic, "(a)"), (ax_ll, "(b)"), (ax_quality, "(c)")):
            style.panel_letter(ax, letter)

        fig.tight_layout()
    return fig
