"""Output paths for generated figures, under the gitignored artefacts tree."""

from __future__ import annotations

from pathlib import Path

from analysis.paths import artefacts_dir


def figures_dir(root: Path) -> Path:
    """Return the figures output directory, ``<root>/artefacts/figures``."""
    return artefacts_dir(root) / "figures"


def docs_figures_dir(root: Path) -> Path:
    """Return the committed documentation figures directory.

    This is ``<root>/docs/source/_figures``, the one place under version control that holds
    rendered figures. The ``figures publish`` command copies the chosen PNGs here from the
    gitignored artefacts tree, and the documentation pages embed them from here.
    """
    return root / "docs" / "source" / "_figures"


def figure_stem(root: Path, source_stage: str, source_hash: str, name: str) -> Path:
    """Return the output-path stem for a figure built from one source run.

    Parameters
    ----------
    root : pathlib.Path
        The monorepo root.
    source_stage : str
        The analysis stage whose run the figure visualises, for example ``"select"``.
    source_hash : str
        The short hash of that source run, so a figure is traceable to its inputs.
    name : str
        The figure's base name, without a suffix.

    Returns
    -------
    pathlib.Path
        ``<root>/artefacts/figures/<source_stage>/<source_hash>/<name>`` with no suffix; the
        save helper appends one suffix per format.
    """
    return figures_dir(root) / source_stage / source_hash / name
