"""The trajectory-roughness figure: step size against noise, and directional movement.

Built from the ``trajectory`` runs of both axes, the figure reads each class's path in two
ways. Panel A compares the mean step between adjacent strata with the step that independent
sampling of a class of that size would produce, so a jagged path can be read as sampling noise
rather than movement. Panel B compares each class's net young-to-old displacement with an
ordering-shuffle null, so a large displacement tied to the axis reads as directional drift
rather than scatter. The directional test is a pilot on the observed centroids; the
confirmatory test is the continuous-trend regression against the refit permutation null.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from figures import style


def _class_names(frame: pd.DataFrame) -> list[str]:
    """Return the class names in reference-class order."""
    ordered = frame.sort_values("ref_class")
    return [str(name) for name in ordered["class_name"]]


def roughness_figure(
    roughness_by_axis: dict[str, pd.DataFrame], directional_by_axis: dict[str, pd.DataFrame]
) -> Figure:
    """Build the trajectory-roughness figure from both axes' ``trajectory`` runs.

    Parameters
    ----------
    roughness_by_axis : dict of str to pandas.DataFrame
        Per axis label, the ``roughness_<axis>`` table (``ref_class``, ``class_name``,
        ``step``, ``sampling_noise``).
    directional_by_axis : dict of str to pandas.DataFrame
        Per axis label, the ``directional_<axis>`` table (``ref_class``, ``class_name``,
        ``net``, ``null95``, ``significant``).

    Returns
    -------
    matplotlib.figure.Figure
        A two-panel figure: step magnitude versus sampling noise, and net displacement against
        the ordering-shuffle null.

    Raises
    ------
    ValueError
        When the two mappings do not share their axis labels, or a table is empty.
    """
    axes_labels = list(roughness_by_axis)
    if list(directional_by_axis) != axes_labels or not axes_labels:
        raise ValueError("roughness and directional mappings must share the same axis labels")

    first = next(iter(roughness_by_axis.values()))
    names = _class_names(first)
    n_classes = len(names)
    positions = np.arange(n_classes)
    width = 0.8 / max(len(axes_labels), 1)

    def ordered(frame: pd.DataFrame, column: str) -> np.ndarray:
        return frame.sort_values("ref_class")[column].to_numpy(dtype=float)

    with style.house_style():
        fig, (ax_step, ax_net) = plt.subplots(1, 2, figsize=(9.6, 4.0))
        for k, axis in enumerate(axes_labels):
            offset = (k - (len(axes_labels) - 1) / 2) * width
            rough = roughness_by_axis[axis]
            ax_step.bar(
                positions + offset,
                ordered(rough, "step"),
                width,
                color=[style.PALETTE[4 + k]] * n_classes,
                label=f"observed, {axis}",
            )
            ax_step.scatter(
                positions + offset,
                ordered(rough, "sampling_noise"),
                marker="D",
                s=28,
                color="black",
                zorder=5,
                label="sampling-noise expectation" if k == 0 else None,
            )
        ax_step.set_xticks(positions)
        ax_step.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
        ax_step.set_ylabel("Mean centroid step between\nadjacent strata (SD units)")
        ax_step.legend(fontsize=6.8, loc="upper left")
        style.panel_title(ax_step, "A", "Step magnitude versus sampling noise")

        top = max(ordered(directional_by_axis[axis], "net").max() for axis in axes_labels)
        for k, axis in enumerate(axes_labels):
            offset = (k - (len(axes_labels) - 1) / 2) * width
            direction = directional_by_axis[axis].sort_values("ref_class")
            nets = direction["net"].to_numpy(dtype=float)
            ax_net.bar(
                positions + offset,
                nets,
                width,
                color=[style.PALETTE[4 + k]] * n_classes,
                label=f"observed, {axis}",
            )
            ax_net.scatter(
                positions + offset,
                direction["null95"].to_numpy(dtype=float),
                marker="_",
                s=190,
                color="black",
                linewidth=1.6,
                zorder=5,
                label="ordering-shuffle 95th pct" if k == 0 else None,
            )
            for pos, net, significant in zip(
                positions + offset,
                nets,
                direction["significant"].to_numpy(dtype=bool),
                strict=False,
            ):
                if significant:
                    ax_net.text(pos, net + 0.06 * top, "*", ha="center", va="bottom", fontsize=11)
        ax_net.set_ylim(0, top * 1.16)
        ax_net.set_xticks(positions)
        ax_net.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
        ax_net.set_ylabel("Net young-to-old displacement\n(SD units)")
        ax_net.legend(fontsize=6.8, loc="upper center")
        ax_net.text(
            0.02,
            0.02,
            "* p < 0.05, ordering-shuffle on observed centroids (pilot);\n"
            "confirmatory: refit permutation null and continuous trend",
            transform=ax_net.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.0,
            color="#777",
            style="italic",
        )
        style.panel_title(ax_net, "B", "Net displacement against the ordering-shuffle null")
        fig.tight_layout()
    return fig
