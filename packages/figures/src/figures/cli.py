"""figures command-line interface (Typer).

One subcommand per figure. Each resolves a cached analysis run, builds the figure, and writes
it under ``artefacts/figures/`` with a JSON provenance sidecar.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import typer
from analysis import cache
from analysis.paths import find_repo_root
from matplotlib.figure import Figure

from figures import data, paths, style
from figures.nmin import nmin_figure
from figures.publish import FIGURES, FIGURES_BY_NAME, publish_figure
from figures.replication import replication_figure
from figures.reproduction import reproduction_figure
from figures.selection import selection_figure
from figures.stability import stability_figure
from figures.subset import subset_comparison_figure

app = typer.Typer(
    name="figures",
    help="Generate figures from the analysis artefacts.",
    no_args_is_help=True,
    add_completion=False,
)

_RUN = typer.Option(None, "--run", help="Run short hash; defaults to the latest completed run.")
_FMT = typer.Option("pdf,png", "--format", help="Comma-separated output formats.")


def _name(default: str) -> str:
    """Return the standard ``--name`` option with a per-figure default."""
    return typer.Option(default, help="Output file name, without a suffix.")


@app.callback()
def main() -> None:
    """Generate figures from the analysis artefacts."""


def _write(
    root: Path, stage: str, run_directory: Path, figure: Figure, name: str, fmt: str
) -> None:
    """Save a built figure under the figures tree and report the written paths."""
    source_hash = run_directory.name
    formats = tuple(part.strip() for part in fmt.split(",") if part.strip())
    stem = paths.figure_stem(root, stage, source_hash, name)
    manifest = cache.read_manifest(run_directory) or {}
    written = style.save_figure(
        figure,
        stem,
        formats=formats,
        provenance={
            "source_stage": stage,
            "source_run": source_hash,
            "source_git_commit": manifest.get("git_commit"),
        },
    )
    plt.close(figure)
    typer.echo(f"figures {stage}: from {stage} run {source_hash}")
    for path in written:
        typer.echo(f"  wrote {path.relative_to(root)}")


@app.command()
def reproduce(run: str | None = _RUN, name: str = _name("reproduction"), fmt: str = _FMT) -> None:
    """Plot the recovered class signatures against the published figure-1b profile."""
    root = find_repo_root()
    run_directory = data.resolve_run(root, "align", run)
    our, published, alignment, our_props, published_props = data.load_alignment(run_directory, root)
    figure = reproduction_figure(our, published, alignment, our_props, published_props)
    _write(root, "align", run_directory, figure, name, fmt)


@app.command()
def select(
    run: str | None = _RUN, name: str = _name("selection_criteria"), fmt: str = _FMT
) -> None:
    """Plot the model-selection criteria across the number of latent classes."""
    root = find_repo_root()
    run_directory = data.resolve_run(root, "select", run)
    summary = data.load_selection_summary(run_directory)
    _write(root, "select", run_directory, selection_figure(summary), name, fmt)


@app.command()
def replicate(run: str | None = _RUN, name: str = _name("replication"), fmt: str = _FMT) -> None:
    """Plot the SPARK-to-SSC class signatures and the per-category replication."""
    root = find_repo_root()
    run_directory = data.resolve_run(root, "replicate", run)
    metrics, spark_signature, ssc_signature = data.load_replication(run_directory)
    figure = replication_figure(spark_signature, ssc_signature, metrics)
    _write(root, "replicate", run_directory, figure, name, fmt)


@app.command()
def stability(run: str | None = _RUN, name: str = _name("stability"), fmt: str = _FMT) -> None:
    """Plot the profile and membership stability of the reference fit under refitting."""
    root = find_repo_root()
    run_directory = data.resolve_run(root, "stability", run)
    comparisons, aggregate, overlap_mean = data.load_stability(run_directory)
    figure = stability_figure(comparisons, aggregate, overlap_mean)
    _write(root, "stability", run_directory, figure, name, fmt)


@app.command()
def nmin(run: str | None = _RUN, name: str = _name("stratum_size"), fmt: str = _FMT) -> None:
    """Plot recovery against subsample size and the minimum viable stratum size."""
    root = find_repo_root()
    run_directory = data.resolve_run(root, "nmin", run)
    per_fit, summary, metrics = data.load_nmin(run_directory)
    figure = nmin_figure(per_fit, summary, metrics)
    _write(root, "nmin", run_directory, figure, name, fmt)


def _named_proportions(root: Path, align_hash: str) -> dict[str, float]:
    """Return one align run's class proportions keyed by named class."""
    run_directory = data.resolve_run(root, "align", align_hash)
    _, _, alignment, our_props, _ = data.load_alignment(run_directory, root)
    mapping = alignment["mapping"]
    return {mapping[str(cid)]: fraction for cid, fraction in our_props.items()}


@app.command()
def subset(
    full_align: str = typer.Option("a5e4220612cc3564", help="Full-release align run hash."),
    subset_align: str = typer.Option("f925f49f27d51bd7", help="V9-subset align run hash."),
    control_align: str = typer.Option("46ede877a8aca499", help="Size-matched align run hash."),
    full_select: str = typer.Option("ab88110bae17a09a", help="Full-release select run hash."),
    subset_select: str = typer.Option("63a12d5534bddbeb", help="V9-subset select run hash."),
    control_select: str = typer.Option("b2218e6d55dd3205", help="Size-matched select run hash."),
    name: str = _name("subset_comparison"),
    fmt: str = _FMT,
) -> None:
    """Compare class proportions and model selection across the full, subset, and control cuts."""
    from analysis import reference

    root = find_repo_root()
    proportions = {
        "Litman 2025": dict(reference.PUBLISHED_PROPORTIONS),
        "Full release": _named_proportions(root, full_align),
        "V9 subset": _named_proportions(root, subset_align),
        "Size-matched": _named_proportions(root, control_align),
    }
    selection = {
        "Full release": data.load_selection_summary(data.resolve_run(root, "select", full_select)),
        "V9 subset": data.load_selection_summary(data.resolve_run(root, "select", subset_select)),
        "Size-matched": data.load_selection_summary(
            data.resolve_run(root, "select", control_select)
        ),
    }
    figure = subset_comparison_figure(
        proportions,
        selection,
        cut_order=["Litman 2025", "Full release", "V9 subset", "Size-matched"],
        class_order=list(reference.PUBLISHED_PROPORTIONS),
    )
    # The figure spans several runs; the V9 subset is its primary source, so it is written and
    # published under that align run.
    run_directory = data.resolve_run(root, "align", subset_align)
    _write(root, "align", run_directory, figure, name, fmt)


@app.command()
def publish(
    figure: str | None = typer.Argument(None, help="Figure to publish; every figure when omitted."),
    run: str | None = _RUN,
) -> None:
    """Copy rendered figures into the committed documentation tree, with provenance.

    Each figure is taken from the latest completed run of its source stage (or ``--run``)
    and copied to ``docs/source/_figures/`` beside a JSON sidecar recording its source. A
    figure that has not been rendered yet is skipped with a note, so publishing the whole set
    surfaces what still needs building.
    """
    root = find_repo_root()
    if figure is None:
        specs = FIGURES
        run = None  # each stage resolves its own latest run when publishing the whole set
    elif figure in FIGURES_BY_NAME:
        specs = (FIGURES_BY_NAME[figure],)
    else:
        choices = ", ".join(FIGURES_BY_NAME)
        raise typer.BadParameter(f"unknown figure {figure!r}; choose from {choices}")

    for spec in specs:
        try:
            destination = publish_figure(root, spec, run)
        except FileNotFoundError as error:
            typer.echo(f"figures publish: skipping {spec.name} ({error})")
            continue
        typer.echo(f"figures publish: {spec.name} -> {destination.relative_to(root)}")
