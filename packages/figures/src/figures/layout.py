r"""Page geometry for a figure placed in a specific document.

A figure is built at the width it will occupy, so the document inputs it at natural size.
The pgf export renders text with the document's own LaTeX fonts at compile time, so a figure
must not be rescaled by the ``\input``: scaling desynchronises the font sizes from the box.
The widths therefore live here, in inches, and the builder sizes the figure to one of them.

The collaboration brief (``reports/brief/main.tex``) is a two-column letterpaper article, so a
figure is placed either in a single column or spanning both columns (a ``figure*`` float). Both
measured widths are recorded below, with their TeX-point source, so a geometry change is easy to
follow (1 TeX point is 1/72.27 in).
"""

from __future__ import annotations

# reports/brief/main.tex, from \the\textwidth (469.755 pt) and \the\columnwidth (229.878 pt) under
# 11pt twocolumn, letterpaper, margin=1in. Re-probe with \showthe if the class or margins change.
BRIEF_TEXTWIDTH_IN: float = 6.500
BRIEF_COLUMNWIDTH_IN: float = 3.181


def cols(fraction: float, *, width_in: float = BRIEF_COLUMNWIDTH_IN) -> float:
    """Return a width in inches as a fraction of a reference block.

    Parameters
    ----------
    fraction : float
        The share of the reference width to occupy, for example ``1.0`` for the full block.
    width_in : float, optional
        The reference width in inches; defaults to the brief's single-column width. Pass
        :data:`BRIEF_TEXTWIDTH_IN` for a figure that spans both columns.

    Returns
    -------
    float
        The width in inches.
    """
    return fraction * width_in
