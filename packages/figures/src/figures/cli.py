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

from figures import data, layout, paths, style
from figures.attribution import attribution_figure, mover_contrast_figure
from figures.category_decomposition import category_decomposition_figure
from figures.dense_features import dense_feature_figure
from figures.invariance import invariance_process_figure
from figures.nmin import nmin_figure
from figures.pairwise import pairwise_trajectory_figure
from figures.prevalence import proportion_curve_figure, stacked_area_figure
from figures.publish import FIGURES, FIGURES_BY_NAME, publish_figure
from figures.referent_decomposition import referent_decomposition_figure
from figures.replication import replication_figure
from figures.reproduction import reproduction_figure
from figures.roughness import roughness_figure
from figures.selection import selection_figure
from figures.stability import stability_figure
from figures.sweep import sweep_trajectory_figure
from figures.trajectory import trajectory_figure
from figures.trajectory_local import (
    directional_figure,
    panels_figure,
    plane_figure,
    plane_overlay_figure,
    referent_figure,
    specificity_figure,
    specificity_panels_figure,
)

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
def sweep(
    axis: str = typer.Option("age_at_diagnosis", "--axis", help="Axis: age_at_diagnosis or era."),
    run: str | None = _RUN,
    name: str | None = typer.Option(None, help="Output file name, without a suffix."),
    fmt: str = _FMT,
) -> None:
    """Plot each class's drift as a curve along the axis, from a `sweep` run's decision table."""
    root = find_repo_root()
    run_directory = data.resolve_run(root, "sweep", run, axis=axis)
    decision, manifest = data.load_sweep(run_directory)
    meta = {**manifest.get("metrics", {}), "axis": axis}
    figure = sweep_trajectory_figure(decision, data.class_names(root, axis), meta)
    _write(root, "sweep", run_directory, figure, name or f"sweep_trajectory_{axis}", fmt)


@app.command(name="local-trajectory")
def local_trajectory(
    axis: str = typer.Option("era", "--axis", help="Axis: era or age_at_diagnosis."),
    run: str | None = _RUN,
    name: str | None = typer.Option(None, help="Output file name, without a suffix."),
    fmt: str = _FMT,
) -> None:
    """Plot the combined four-class local trajectory in the discriminant plane, with the tube."""
    root = find_repo_root()
    run_directory = data.resolve_run(root, "invariance-trajectory", run, axis=axis)
    plane, capture, meta = data.load_local_trajectory(run_directory)
    figure = plane_figure(plane, capture, {**meta, "axis": axis})
    _write(root, "invariance-trajectory", run_directory, figure, name or f"local_plane_{axis}", fmt)


@app.command(name="local-panels")
def local_panels(
    axis: str = typer.Option("era", "--axis", help="Axis: era or age_at_diagnosis."),
    run: str | None = _RUN,
    name: str | None = typer.Option(None, help="Output file name, without a suffix."),
    fmt: str = _FMT,
) -> None:
    """Plot the per-class local-trajectory panels, each tube over its faint member ellipse."""
    root = find_repo_root()
    run_directory = data.resolve_run(root, "invariance-trajectory", run, axis=axis)
    plane, capture, meta = data.load_local_trajectory(run_directory)
    figure = panels_figure(plane, capture, {**meta, "axis": axis})
    _write(
        root, "invariance-trajectory", run_directory, figure, name or f"local_panels_{axis}", fmt
    )


@app.command(name="local-specificity")
def local_specificity(
    name: str | None = typer.Option(None, help="Output file name, without a suffix."),
    fmt: str = _FMT,
) -> None:
    """Plot the specificity small-multiple: timing-axis drift against the control panel.

    Reads the latest ``invariance-trajectory`` run for each timing axis and pools their endpoint
    displacements, so the era and age effects are shown together against household income, area
    deprivation, and the random-ordering floor.
    """
    import pandas as pd

    root = find_repo_root()
    frames = []
    source_dir = None
    for timing_axis in ("era", "age_at_diagnosis"):
        try:
            run_directory = data.resolve_run(root, "invariance-trajectory", axis=timing_axis)
        except FileNotFoundError:
            continue
        source_dir = source_dir or run_directory
        table = data.load_local_specificity(run_directory)
        # Keep the timing row from its own run; the control rows are pooled across the runs.
        frames.append(table)
    if source_dir is None:
        raise typer.BadParameter("no completed invariance-trajectory run for either timing axis")
    merged = pd.concat(frames, ignore_index=True)
    figure = specificity_figure(merged, {"timing_axes": ["era", "age_at_diagnosis"]})
    _write(root, "invariance-trajectory", source_dir, figure, name or "local_specificity", fmt)


@app.command(name="category-decomposition")
def category_decomposition(
    name: str | None = typer.Option(None, help="Output file name, without a suffix."),
    fmt: str = _FMT,
) -> None:
    """Plot the H0F category decomposition: what symptom categories carry each class's drift.

    Reads the latest ``invariance-trajectory`` run for each timing axis and pools their category
    grains and per-feature displacements, so the era and age concentrations are shown together with
    the leading features behind the age drift.
    """
    import pandas as pd

    root = find_repo_root()
    grains: dict[str, pd.DataFrame] = {}
    features: dict[str, pd.DataFrame] = {}
    source_dir = None
    for timing_axis in ("era", "age_at_diagnosis"):
        try:
            run_directory = data.resolve_run(root, "invariance-trajectory", axis=timing_axis)
        except FileNotFoundError:
            continue
        source_dir = source_dir or run_directory
        grains[timing_axis] = data.load_grain_magnitude(run_directory)
        features[timing_axis] = data.load_feature_displacement(run_directory)
    if source_dir is None:
        raise typer.BadParameter("no completed invariance-trajectory run for either timing axis")
    figure = category_decomposition_figure(grains, features, {"axes": list(grains)})
    _write(root, "invariance-trajectory", source_dir, figure, name or "category_decomposition", fmt)


@app.command(name="dense-features")
def dense_features(
    name: str | None = typer.Option(None, help="Output file name, without a suffix."),
    fmt: str = _FMT,
) -> None:
    """Plot every significant feature's signed drift, per class and axis, grouped by category."""
    import pandas as pd

    root = find_repo_root()
    features: dict[str, pd.DataFrame] = {}
    source_dir = None
    for timing_axis in ("era", "age_at_diagnosis"):
        try:
            run_directory = data.resolve_run(root, "invariance-trajectory", axis=timing_axis)
        except FileNotFoundError:
            continue
        source_dir = source_dir or run_directory
        features[timing_axis] = data.load_feature_displacement(run_directory)
    if source_dir is None:
        raise typer.BadParameter("no completed invariance-trajectory run for either timing axis")
    figure = dense_feature_figure(features, {"axes": list(features)})
    _write(root, "invariance-trajectory", source_dir, figure, name or "dense_features", fmt)


@app.command(name="referent-decomposition")
def referent_decomposition(
    run: str | None = _RUN,
    name: str = _name("referent_decomposition"),
    fmt: str = _FMT,
) -> None:
    """Plot the H0G referent split of the era drift: the contrast and its instruments."""
    root = find_repo_root()
    run_directory = data.resolve_run(root, "invariance-trajectory", run, axis="era")
    grains, contrast, meta = data.load_referent(run_directory)
    figure = referent_decomposition_figure(grains, contrast, meta)
    _write(root, "invariance-trajectory", run_directory, figure, name, fmt)


@app.command()
def brief(
    fmt: str = typer.Option("pgf,pdf", "--format", help="Comma-separated output formats."),
) -> None:
    """Build the collaboration-brief figures: the single-fit trajectory and the specificity panels.

    Writes each figure to ``reports/brief/figures`` beside ``main.tex``, so the brief inputs the
    ``.pgf`` at natural size, its text set in the brief's sans-serif font. The trajectory shows the
    age-at-diagnosis drift in one column; the specificity figure is the wider two-panel view (timing
    axes and controls), built at the text width. Needs a working TeX install on the path.
    """
    import pandas as pd

    root = find_repo_root()
    planes: dict[str, pd.DataFrame] = {}
    captures: dict[str, pd.DataFrame] = {}
    spec_frames: list[pd.DataFrame] = []
    source_dir: Path | None = None
    for timing_axis in ("age_at_diagnosis", "era"):
        try:
            run_directory = data.resolve_run(root, "invariance-trajectory", axis=timing_axis)
        except FileNotFoundError:
            continue
        source_dir = source_dir or run_directory
        plane, capture, _ = data.load_local_trajectory(run_directory)
        planes[timing_axis] = plane
        captures[timing_axis] = capture
        spec_frames.append(data.load_local_specificity(run_directory))
    if source_dir is None:
        raise typer.BadParameter("no completed invariance-trajectory run for either timing axis")
    if "age_at_diagnosis" not in planes:
        raise typer.BadParameter("the trajectory needs a completed age_at_diagnosis run")

    # The trajectory is a single-column float showing age at diagnosis only (input at natural size,
    # never rescaled). The specificity figure carries per-class bars in two panels and needs the
    # room, so it spans the text width.
    age = {"age_at_diagnosis": planes["age_at_diagnosis"]}
    age_capture = {"age_at_diagnosis": captures["age_at_diagnosis"]}
    trajectory = plane_overlay_figure(
        age, age_capture, width_in=layout.BRIEF_COLUMNWIDTH_IN, brief=True
    )
    specificity = specificity_panels_figure(
        pd.concat(spec_frames, ignore_index=True),
        {"timing_axes": ["era", "age_at_diagnosis"]},
        width_in=layout.BRIEF_COLUMNWIDTH_IN,
        height_in=2.8,
    )

    manifest = cache.read_manifest(source_dir) or {}
    out_dir = paths.brief_figures_dir(root)
    formats = tuple(part.strip() for part in fmt.split(",") if part.strip())
    provenance = {
        "source_stage": "invariance-trajectory",
        "source_run": source_dir.name,
        "source_git_commit": manifest.get("git_commit"),
    }
    for figure, name in ((trajectory, "trajectory_overlay"), (specificity, "specificity")):
        written = style.save_figure(
            figure, out_dir / name, formats=formats, provenance=provenance, pgf_rc=style.PGF_RC_SANS
        )
        plt.close(figure)
        for path in written:
            typer.echo(f"  wrote {path.relative_to(root)}")


@app.command(name="local-directional")
def local_directional(
    axis: str = typer.Option("era", "--axis", help="Axis: era or age_at_diagnosis."),
    run: str | None = _RUN,
    name: str | None = typer.Option(None, help="Output file name, without a suffix."),
    fmt: str = _FMT,
) -> None:
    """Plot the H0E figure: each class's signed trajectory along the axis, with its break."""
    root = find_repo_root()
    run_directory = data.resolve_run(root, "invariance-trajectory", run, axis=axis)
    signed, directional, meta = data.load_local_directional(run_directory)
    figure = directional_figure(signed, directional, {**meta, "axis": axis})
    _write(
        root,
        "invariance-trajectory",
        run_directory,
        figure,
        name or f"local_directional_{axis}",
        fmt,
    )


@app.command(name="local-referent")
def local_referent(
    axis: str = typer.Option("era", "--axis", help="Axis: era only (H0G is era-only)."),
    run: str | None = _RUN,
    name: str | None = typer.Option(None, help="Output file name, without a suffix."),
    fmt: str = _FMT,
) -> None:
    """Plot the H0G figure: per-class current-versus-retrospective drift with the underlay."""
    root = find_repo_root()
    run_directory = data.resolve_run(root, "invariance-trajectory", run, axis=axis)
    grains, contrast, meta = data.load_referent(run_directory)
    figure = referent_figure(grains, contrast, {**meta, "axis": axis})
    _write(
        root,
        "invariance-trajectory",
        run_directory,
        figure,
        name or f"local_referent_{axis}",
        fmt,
    )


@app.command()
def prevalence(
    axis: str = typer.Option("era", "--axis", help="Axis: era or age_at_diagnosis."),
    layout: str = typer.Option(
        "panels", "--layout", help="Figure layout: panels (per class) or stacked (composition)."
    ),
    run: str | None = _RUN,
    name: str | None = typer.Option(None, help="Output file name, without a suffix."),
    fmt: str = _FMT,
) -> None:
    """Plot the H0B figure: per-class proportion curves, or the stacked class composition.

    ``--layout panels`` draws one panel per class, the corrected proportion curve with its
    bootstrap band, the naive cross-check, and the pooled proportion line. ``--layout stacked``
    draws the four corrected proportions stacked to one across the axis, the compositional view.
    """
    if layout not in ("panels", "stacked"):
        raise typer.BadParameter("layout must be 'panels' or 'stacked'")
    root = find_repo_root()
    run_directory = data.resolve_run(root, "prevalence", run, axis=axis)
    curve, slopes, meta = data.load_prevalence(run_directory)
    if layout == "stacked":
        figure = stacked_area_figure(curve, {**meta, "axis": axis})
        default_name = f"prevalence_stacked_{axis}"
    else:
        figure = proportion_curve_figure(curve, slopes, {**meta, "axis": axis})
        default_name = f"prevalence_{axis}"
    _write(root, "prevalence", run_directory, figure, name or default_name, fmt)


@app.command()
def invariance(
    axis: str = typer.Option("age_at_diagnosis", "--axis", help="Axis: age_at_diagnosis or era."),
    run: str | None = _RUN,
    name: str | None = typer.Option(None, help="Output file name, without a suffix."),
    fmt: str = _FMT,
) -> None:
    """Plot the strongest-drifting block's fluctuation process against its bridge null."""
    root = find_repo_root()
    run_directory = data.resolve_run(root, "invariance", run, axis=axis)
    process, manifest = data.load_invariance(run_directory)
    meta = {**manifest.get("metrics", {}), "axis": axis}
    figure = invariance_process_figure(process, meta)
    _write(root, "invariance", run_directory, figure, name or f"invariance_process_{axis}", fmt)


@app.command()
def pairwise(
    axis: str = typer.Option("age_at_diagnosis", "--axis", help="Axis: age_at_diagnosis or era."),
    run: str | None = _RUN,
    name: str | None = typer.Option(None, help="Output file name, without a suffix."),
    fmt: str = _FMT,
) -> None:
    """Plot each class's neighbour-to-neighbour drift along the axis, from a pairwise run."""
    root = find_repo_root()
    run_directory = data.resolve_run(
        root, "drift", run, axis=axis, require={"reference_scheme": "pairwise"}
    )
    trajectory, metrics = data.load_pairwise(run_directory)
    meta = {**metrics, "axis": axis}
    figure = pairwise_trajectory_figure(trajectory, data.class_names(root, axis), meta)
    _write(root, "drift", run_directory, figure, name or f"pairwise_trajectory_{axis}", fmt)


@app.command()
def attribute(
    axis: str = typer.Option("age_at_diagnosis", "--axis", help="Axis: age_at_diagnosis or era."),
    run: str | None = _RUN,
    name: str | None = typer.Option(None, help="Output file name, without a suffix."),
    fmt: str = _FMT,
) -> None:
    """Plot each class's membership churn across the strata and what carries each shift (archived).

    The refit-era attribution figure, kept for the refit-pilot archive page; the single-fit $H_0^F$
    category attribution is drawn by ``category-decomposition`` and ``dense-features``.
    """
    root = find_repo_root()
    run_directory = data.resolve_run(root, "attribute", run, axis=axis)
    summary, category, _movers, meta = data.load_attribution(run_directory)
    figure = attribution_figure(summary, category, meta)
    _write(root, "attribute", run_directory, figure, name or f"attribution_{axis}", fmt)


@app.command()
def attribute_contrast(
    axis: str = typer.Option("age_at_diagnosis", "--axis", help="Axis: age_at_diagnosis or era."),
    run: str | None = _RUN,
    name: str | None = typer.Option(None, help="Output file name, without a suffix."),
    fmt: str = _FMT,
) -> None:
    """Plot, per class, the features marking the probands that changed class at its peak churn.

    The refit-era mover contrast, kept for the refit-pilot archive page (a single fit relabels no
    proband, so it has no single-fit counterpart).
    """
    root = find_repo_root()
    run_directory = data.resolve_run(root, "attribute", run, axis=axis)
    summary, _category, movers, meta = data.load_attribution(run_directory)
    figure = mover_contrast_figure(summary, movers, meta)
    _write(root, "attribute", run_directory, figure, name or f"attribution_movers_{axis}", fmt)


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
