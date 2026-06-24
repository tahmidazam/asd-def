"""Promoting generated figures into the committed documentation tree.

Figures are rendered under the gitignored ``artefacts/`` directory, so they do not reach the
published documentation on their own. ``figures publish`` copies the chosen PNGs into
``docs/source/_figures/``, a committed directory the documentation pages embed, and writes a
provenance sidecar beside each so a published figure traces back to the analysis run and the
commit it was built from.

Only the rendered PNGs cross into the committed tree. They are aggregate, non-disclosive
summaries (class profiles, correlations, drift), so committing them is within the data
governance that keeps the rest of the artefacts tree out of the history.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from analysis import cache

from figures import __version__, data, paths


@dataclass(frozen=True)
class FigureSpec:
    """One publishable figure.

    Attributes
    ----------
    name : str
        The figure's key: its ``figures <name>`` build command and its documentation handle.
    source_stage : str
        The analysis stage whose cached run the figure visualises.
    file_name : str
        The figure's base file name, without a suffix.
    """

    name: str
    source_stage: str
    file_name: str


# Every figure the package can publish, in the order the documentation presents them.
FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec("reproduce", "align", "reproduction"),
    FigureSpec("select", "select", "selection_criteria"),
    FigureSpec("stability", "stability", "stability"),
    FigureSpec("nmin", "nmin", "stratum_size"),
    FigureSpec("replicate", "replicate", "replication"),
)

FIGURES_BY_NAME: dict[str, FigureSpec] = {spec.name: spec for spec in FIGURES}


def publish_figure(root: Path, spec: FigureSpec, run: str | None = None) -> Path:
    """Copy one rendered figure into the documentation tree and record its provenance.

    Parameters
    ----------
    root : pathlib.Path
        The monorepo root.
    spec : FigureSpec
        The figure to publish.
    run : str, optional
        The source run's short hash. When omitted, the latest completed run of the figure's
        source stage is used, the same run :func:`figures.data.resolve_run` would pick.

    Returns
    -------
    pathlib.Path
        The written PNG in the documentation tree.

    Raises
    ------
    FileNotFoundError
        When no PNG has been rendered for the figure yet, so the caller is told to build it
        first.
    """
    source_run = data.resolve_run(root, spec.source_stage, run)
    source_hash = source_run.name
    png = paths.figure_stem(root, spec.source_stage, source_hash, spec.file_name).with_suffix(
        ".png"
    )
    if not png.is_file():
        msg = f"no rendered figure at {png}; run `figures {spec.name}` first"
        raise FileNotFoundError(msg)

    destination_dir = paths.docs_figures_dir(root)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{spec.file_name}.png"
    shutil.copyfile(png, destination)

    # A timestamp is deliberately left out: the source run and commit already pin the figure,
    # so the sidecar changes only when the figure does, and re-publishing an unchanged figure
    # leaves no spurious diff.
    sidecar = {
        "figure": spec.name,
        "source_stage": spec.source_stage,
        "source_run": source_hash,
        "source_git_commit": (cache.read_manifest(source_run) or {}).get("git_commit"),
        "figures_version": __version__,
    }
    destination.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )
    return destination
