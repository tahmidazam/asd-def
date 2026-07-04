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
from figures.roughness import roughness_figure
from figures.selection import selection_figure
from figures.stability import stability_figure
from figures.trajectory import trajectory_figure

_NICE_AXIS = {"age_at_diagnosis": "age at diagnosis", "era": "diagnostic era"}

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
def reproduce(
    run: str = typer.Option("a5e4220612cc3564", "--run", help="Full-release align run hash."),
    as_of_run: str = typer.Option(
        "f925f49f27d51bd7", "--as-of-run", help="V9-subset align run hash ('' to omit)."
    ),
    name: str = _name("reproduction"),
    fmt: str = _FMT,
) -> None:
    """Plot the recovered class signatures against the published profile, across conditions."""
    root = find_repo_root()
    run_directory = data.resolve_run(root, "align", run)
    our, published, alignment, our_props, published_props = data.load_alignment(run_directory, root)
    comparison = None
    if as_of_run:
        comp_dir = data.resolve_run(root, "align", as_of_run)
        comp_sig, _, comp_align, comp_props, _ = data.load_alignment(comp_dir, root)
        comparison = {"signature": comp_sig, "alignment": comp_align, "proportions": comp_props}
    figure = reproduction_figure(our, published, alignment, our_props, published_props, comparison)
    _write(root, "align", run_directory, figure, name, fmt)


@app.command()
def select(
    run: str = typer.Option("ab88110bae17a09a", "--run", help="Full-release select run hash."),
    as_of_run: str = typer.Option(
        "63a12d5534bddbeb", "--as-of-run", help="V9-subset select run hash ('' to omit)."
    ),
    name: str = _name("selection_criteria"),
    fmt: str = _FMT,
) -> None:
    """Plot the model-selection criteria across the number of latent classes, across conditions."""
    root = find_repo_root()
    run_directory = data.resolve_run(root, "select", run)
    summary = data.load_selection_summary(run_directory)
    comparison = None
    if as_of_run:
        comparison = data.load_selection_summary(data.resolve_run(root, "select", as_of_run))
    figure = selection_figure(summary, comparison=comparison)
    _write(root, "select", run_directory, figure, name, fmt)


@app.command()
def replicate(
    run: str = typer.Option("10906a3bbea3c8ab", "--run", help="Full-release replicate run hash."),
    as_of_run: str = typer.Option(
        "861b6f101a10e3ae", "--as-of-run", help="V9-subset replicate run hash ('' to omit)."
    ),
    name: str = _name("replication"),
    fmt: str = _FMT,
) -> None:
    """Plot the SPARK-to-SSC class signatures and the per-category replication across conditions."""
    root = find_repo_root()
    run_directory = data.resolve_run(root, "replicate", run)
    metrics, spark_signature, ssc_signature = data.load_replication(run_directory)
    comparison = None
    if as_of_run:
        comparison, _, _ = data.load_replication(data.resolve_run(root, "replicate", as_of_run))
    figure = replication_figure(spark_signature, ssc_signature, metrics, comparison)
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


@app.command()
def trajectory(
    axis: str = typer.Option("age_at_diagnosis", "--axis", help="Axis: age_at_diagnosis or era."),
    run: str | None = _RUN,
    name: str | None = typer.Option(None, help="Output file name, without a suffix."),
    fmt: str = _FMT,
) -> None:
    """Plot each class's trajectory through the strata in the pooled discriminant space."""
    root = find_repo_root()
    run_directory = data.resolve_run(root, "trajectory", run, axis=axis)
    embedding, meta = data.load_trajectory(run_directory)
    figure = trajectory_figure(embedding, meta)
    _write(root, "trajectory", run_directory, figure, name or f"trajectory_{axis}", fmt)


@app.command()
def roughness(
    age_run: str | None = typer.Option(None, "--age-run", help="Age trajectory run hash."),
    era_run: str | None = typer.Option(None, "--era-run", help="Era trajectory run hash."),
    name: str = _name("roughness"),
    fmt: str = _FMT,
) -> None:
    """Plot trajectory roughness and directional movement across both axes."""
    root = find_repo_root()
    age_dir = data.resolve_run(root, "trajectory", age_run, axis="age_at_diagnosis")
    era_dir = data.resolve_run(root, "trajectory", era_run, axis="era")
    roughness_by_axis = {}
    directional_by_axis = {}
    for directory in (age_dir, era_dir):
        axis, rough, direction = data.load_roughness(directory)
        label = _NICE_AXIS.get(axis, axis)
        roughness_by_axis[label] = rough
        directional_by_axis[label] = direction
    figure = roughness_figure(roughness_by_axis, directional_by_axis)
    _write(root, "trajectory", age_dir, figure, name, fmt)


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
