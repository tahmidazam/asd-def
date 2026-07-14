"""House style for the figures: Matplotlib settings, a palette, and a save helper.

The settings are deliberately small and explicit, so a figure looks the same whoever builds
it. :func:`house_style` returns a context manager that applies them around figure
construction without leaking into the global state, and :func:`save_figure` writes one file
per format alongside a JSON provenance sidecar.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

from figures import __version__

# Colourblind-safe qualitative palette (Wong, Nature Methods 2011), used in order for the
# criterion lines.
PALETTE: tuple[str, ...] = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
)

# Colour of the reference lines that mark the chosen number of classes.
REFERENCE_COLOUR = "#444444"

# Matplotlib pgf-backend settings for the ``pgf`` export format. The figure text is left to the
# document's own LaTeX run (``rcfonts`` off), so an included ``.pgf`` matches the surrounding
# prose; the caller passes the variant whose preamble mirrors the target document. The default is
# serif (Latin Modern); PGF_RC_SANS mirrors reports/brief/main.tex (Helvetica via ``helvet``).
_PGF_RC: dict[str, object] = {
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "pgf.preamble": "\n".join((r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern}")),
}

PGF_RC_SANS: dict[str, object] = {
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family": "sans-serif",
    "pgf.preamble": "\n".join(
        (
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{helvet}",
            r"\usepackage[helvet]{sfmath}",
            r"\renewcommand{\familydefault}{\sfdefault}",
        )
    ),
}

# Short display labels for the seven enrichment categories, so axis ticks stay legible. Keyed
# by the names the analysis writes (American spelling and slashes preserved to match the data).
CATEGORY_LABELS: dict[str, str] = {
    "anxiety/mood": "anxiety/mood",
    "attention": "attention",
    "disruptive behavior": "disruptive",
    "self-injury": "self-injury",
    "social/communication": "social/comm",
    "restricted/repetitive": "restricted/rep",
    "developmental": "developmental",
}


def panel_title(ax: Axes, letter: str, text: str) -> None:
    """Set a left-aligned panel title prefixed with a bold letter label.

    The letter is rendered in bold and the scientific title follows in the regular weight, so
    each panel reads as, for example, "A  Information criteria", left-aligned over the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to title.
    letter : str
        The panel letter, shown in bold (for example ``"A"``).
    text : str
        The scientific title that follows the letter.
    """
    ax.set_title(rf"$\mathbf{{{letter}}}$  {text}", loc="left")


# LaTeX font sizes in the 11pt collaboration brief: ``\small`` is 10pt and ``\footnotesize`` is
# 9pt. The brief inputs these figures as pgf set in the document's own fonts, so the axis and tick
# labels are sized in points to read at the same sizes as the surrounding prose.
_BRIEF_LABEL_PT = 10.0
_BRIEF_TICK_PT = 9.0


def brief_axis_style(ax: Axes, *, minor_x: bool = True) -> None:
    r"""Match a brief figure's axis fonts to the document and add major and minor gridlines.

    The axis labels are sized to the brief's ``\small`` (10pt) and the tick labels to
    ``\footnotesize`` (9pt), so an included ``.pgf`` reads at the same sizes as the surrounding
    LaTeX. Major and minor gridlines are drawn (the minor lines fainter). Minor ticks and their
    gridlines are added on the y-axis always and on the x-axis only when it carries a continuous
    scale.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to restyle.
    minor_x : bool, optional
        Whether to add minor ticks and gridlines on the x-axis, and to resize its tick labels.
        Set false for a categorical x-axis (bar groups), whose labels are kept at their own size.
    """
    ax.xaxis.label.set_fontsize(_BRIEF_LABEL_PT)
    ax.yaxis.label.set_fontsize(_BRIEF_LABEL_PT)
    ax.tick_params(axis="y", which="major", labelsize=_BRIEF_TICK_PT)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if minor_x:
        ax.tick_params(axis="x", which="major", labelsize=_BRIEF_TICK_PT)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which="major", alpha=0.3, linewidth=0.5)
    ax.grid(which="minor", alpha=0.15, linewidth=0.4)


_RC_PARAMS: dict[str, object] = {
    "figure.dpi": 120,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}


@contextmanager
def house_style() -> Iterator[None]:
    """Apply the package Matplotlib settings for the duration of the context.

    Yields
    ------
    None
        Control returns to the caller with the package ``rcParams`` in effect; the previous
        settings are restored on exit, so the styling does not leak into other figures.
    """
    with mpl.rc_context(_RC_PARAMS):
        yield


def save_figure(
    fig: Figure,
    stem: Path,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    provenance: dict[str, object] | None = None,
    pgf_rc: dict[str, object] | None = None,
) -> list[Path]:
    """Write a figure to one file per format and a JSON provenance sidecar.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    stem : pathlib.Path
        Output path without a suffix; each format appends its own. Parent directories are
        created if absent.
    formats : sequence of str, optional
        File extensions to write; defaults to ``("pdf", "png")``. PDF is the vector format
        for the manuscript; PNG is the raster preview the docs embed. ``"pgf"`` writes a
        LaTeX picture the document inputs, its text rendered with the document's fonts; it
        needs a working TeX install on the path.
    provenance : dict, optional
        Extra fields to record in the ``<stem>.json`` sidecar, alongside the package version
        and the time of writing.
    pgf_rc : dict, optional
        Matplotlib settings for the ``"pgf"`` format, so the figure text matches the target
        document's fonts; defaults to the serif preamble. Pass :data:`PGF_RC_SANS` for a
        sans-serif document. Ignored when ``"pgf"`` is not in ``formats``.

    Returns
    -------
    list of pathlib.Path
        The written image paths, in ``formats`` order.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        if fmt == "pgf":
            with mpl.rc_context(pgf_rc or _PGF_RC):
                fig.savefig(path, bbox_inches="tight")
        else:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    sidecar: dict[str, object] = {
        "figures_version": __version__,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if provenance:
        sidecar.update(provenance)
    stem.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return written
