"""The H0G referent figure: is the era drift measurement timing or a population change.

Reads the H0G referent split of the era drift (plan section 7f; the block engine's size-fair
grain contrast). Panel A is the test: the per-class current-minus-retrospective root-mean-square
contrast, negative when the drift sits in the retrospective and lifetime instruments (the
diagnosed-population signature) and positive when it sits in the current-state instruments (the
measurement-timing signature). Panel B opens the contrast up by instrument, so the reader sees which
questionnaires carry each referent: the lifetime SCQ and the developmental history against the
current-state CBCL and RBS-R.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from figures import style

# The two referents get a fixed colour: retrospective (the population signature) blue, current-state
# (the timing signature) vermillion, from the house palette.
_RETROSPECTIVE = style.PALETTE[0]
_CURRENT = style.PALETTE[3]
_REFERENT_COLOUR = {"retrospective": _RETROSPECTIVE, "current_state": _CURRENT}
_REFERENT_LABEL = {"retrospective": "retrospective / lifetime", "current_state": "current-state"}
_INSTRUMENT_LABEL = {
    "scq": "SCQ (lifetime)",
    "background_history_child": "developmental history",
    "cbcl_6_18": "CBCL 6-18",
    "rbsr": "RBS-R",
}


def referent_decomposition_figure(
    grains: pd.DataFrame, contrast: pd.DataFrame, meta: dict
) -> Figure:
    """Build the H0G referent figure: the current-minus-retrospective contrast and its instruments.

    Parameters
    ----------
    grains : pandas.DataFrame
        The ``referent_era`` table: per class and grain (referent and instrument), the size-fair
        root-mean-square intensity, additive share, and FDR-surviving feature count.
    contrast : pandas.DataFrame
        The ``referent_contrast_era`` table: per class, the current-minus-retrospective contrast
        with its interval, ``p``-value, FDR decision, and mechanism.
    meta : dict
        Figure metadata; unused beyond documenting provenance.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel referent figure.
    """
    import matplotlib.pyplot as plt

    contrast = contrast.sort_values("ref_class")
    classes = contrast["ref_class"].tolist()
    names = contrast.set_index("ref_class")["class_name"].to_dict()
    y = np.arange(len(classes))[::-1]

    instruments = grains[grains["grain_kind"] == "instrument"]
    instrument_names = sorted(
        instruments["grain"].unique(),
        key=lambda n: str(instruments.loc[instruments["grain"] == n, "referent"].iloc[0]),
    )

    with style.house_style():
        fig = plt.figure(figsize=(9.4, 3.8))
        grid = GridSpec(1, 2, width_ratios=(1.0, 1.2), wspace=0.3, figure=fig)

        # Panel A: the per-class current-minus-retrospective contrast.
        ax_a = fig.add_subplot(grid[0, 0])
        for pos, (_, row) in zip(y, contrast.iterrows(), strict=True):
            colour = _CURRENT if row["contrast"] > 0 else _RETROSPECTIVE
            ax_a.barh(pos, row["contrast"], color=colour, height=0.62, zorder=2)
            ax_a.plot(
                [row["ci_low"], row["ci_high"]],
                [pos, pos],
                color="#333333",
                linewidth=1.2,
                zorder=3,
            )
            if bool(row["reject"]):
                ax_a.text(
                    row["contrast"],
                    pos + 0.34,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#333333",
                )
        ax_a.axvline(0.0, color=style.REFERENCE_COLOUR, linewidth=0.8)
        ax_a.set_yticks(y)
        ax_a.set_yticklabels([names[c] for c in classes])
        ax_a.set_xlabel("current − retrospective RMS contrast")
        style.panel_title(ax_a, "A", "referent contrast (all retrospective-dominant)")

        # Panel B: the per-instrument RMS intensity, grouped by class and coloured by referent.
        ax_b = fig.add_subplot(grid[0, 1])
        n_inst = len(instrument_names)
        width = 0.8 / max(n_inst, 1)
        seen: set[str] = set()
        for k, name in enumerate(instrument_names):
            sub = instruments[instruments["grain"] == name].set_index("ref_class")
            referent = str(sub["referent"].iloc[0])
            colour = _REFERENT_COLOUR.get(referent, style.REFERENCE_COLOUR)
            xs = np.arange(len(classes)) + (k - (n_inst - 1) / 2) * width
            heights = [float(sub["rms"].get(c, 0.0)) for c in classes]
            label = _REFERENT_LABEL.get(referent, referent) if referent not in seen else None
            ax_b.bar(xs, heights, width=width, color=colour, label=label, zorder=2)
            for x, height in zip(xs, heights, strict=True):
                ax_b.text(
                    x,
                    height + 0.005,
                    str(k + 1),
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    color="#555555",
                )
            seen.add(referent)
        ax_b.set_xticks(np.arange(len(classes)))
        ax_b.set_xticklabels([names[c] for c in classes], rotation=20, ha="right")
        ax_b.set_ylabel("size-fair RMS intensity")
        ax_b.legend(loc="upper left", fontsize=7, title="referent", title_fontsize=7)
        style.panel_title(ax_b, "B", "which instruments carry the drift")
        key = ",   ".join(
            f"{i + 1} {_INSTRUMENT_LABEL.get(n, n)}" for i, n in enumerate(instrument_names)
        )
        fig.text(
            0.5,
            -0.14,
            f"instruments per class: {key}",
            ha="center",
            fontsize=6.4,
            color="#555555",
        )

    return fig
