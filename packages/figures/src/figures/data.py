"""Locating and loading the cached analysis run a figure visualises."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from analysis import cache
from analysis.paths import run_dir, stage_dir


def resolve_run(root: Path, stage: str, run: str | None = None, *, axis: str | None = None) -> Path:
    """Return the run directory to visualise for an analysis stage.

    Parameters
    ----------
    root : pathlib.Path
        The monorepo root.
    stage : str
        The analysis stage, for example ``"select"``.
    run : str, optional
        A run's short hash. When given, that run's directory is returned. When omitted, the
        most recently finished run for the stage is chosen (the latest manifest whose status
        is ``"ok"``).
    axis : str, optional
        Restrict the latest-run search to runs whose manifest records this axis (for the
        per-axis stages, ``"age_at_diagnosis"`` or ``"era"``). Ignored when ``run`` is given.

    Returns
    -------
    pathlib.Path
        The resolved run directory.

    Raises
    ------
    FileNotFoundError
        When the named run has no manifest, or when no completed run exists for the stage.
    """
    if run is not None:
        rdir = run_dir(root, stage, run)
        if cache.read_manifest(rdir) is None:
            msg = f"no run {run!r} for stage {stage!r} under {stage_dir(root, stage)}"
            raise FileNotFoundError(msg)
        return rdir

    sdir = stage_dir(root, stage)
    completed: list[tuple[str, Path]] = []
    if sdir.is_dir():
        for child in sorted(sdir.iterdir()):
            manifest = cache.read_manifest(child) if child.is_dir() else None
            if manifest is None or manifest.get("status") != "ok":
                continue
            if axis is not None and manifest.get("params", {}).get("axis") != axis:
                continue
            completed.append((str(manifest.get("finished_at", "")), child))
    if not completed:
        where = f"{stage!r}" if axis is None else f"{stage!r} (axis {axis!r})"
        msg = f"no completed {where} run under {sdir}; run `analysis {stage}` first"
        raise FileNotFoundError(msg)
    completed.sort(key=lambda item: item[0])
    return completed[-1][1]


def _recovered_proportions(align_run_directory: Path, root: Path) -> dict[int, float]:
    """Return the recovered class proportions for an ``align`` run, by class id.

    The proportions come from the upstream ``fit`` run's labels, whose hash the align
    manifest records. An empty dictionary is returned when that fit run is not on disk, so a
    figure can still be drawn without the per-class size annotation.
    """
    manifest = cache.read_manifest(align_run_directory) or {}
    fit_hash = manifest.get("params", {}).get("fit")
    if not fit_hash:
        return {}
    fit_directory = run_dir(root, "fit", cache.short_hash(fit_hash))
    labels_path = fit_directory / "labels.parquet"
    if not labels_path.is_file():
        return {}
    fractions = cache.load_frame(labels_path)["class"].value_counts(normalize=True)
    return {int(cid): float(fraction) for cid, fraction in fractions.items()}


def load_alignment(
    run_directory: Path, root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[int, float], dict[str, float]]:
    """Load an ``align`` run's reproduction inputs.

    Parameters
    ----------
    run_directory : pathlib.Path
        A completed ``align`` run directory.
    root : pathlib.Path
        The monorepo root, used to resolve the upstream ``fit`` run for the class
        proportions.

    Returns
    -------
    tuple
        ``(our_signature, published_signature, alignment, our_proportions,
        published_proportions)``: the recovered class-by-category signature, the published
        figure-1b signature, the alignment record (``alignment.json``), the recovered class
        proportions by class id, and the published proportions by named class.
    """
    from analysis import reference  # local: pulls scipy and statsmodels, only needed here

    our_signature = cache.load_frame(run_directory / "signature.parquet")
    alignment = cache.load_json(run_directory / "alignment.json")
    published_signature = reference.published_signature()
    published_proportions = dict(reference.PUBLISHED_PROPORTIONS)
    our_proportions = _recovered_proportions(run_directory, root)
    return our_signature, published_signature, alignment, our_proportions, published_proportions


def load_selection_summary(run_directory: Path) -> pd.DataFrame:
    """Load the per-component selection summary from a ``select`` run directory.

    Parameters
    ----------
    run_directory : pathlib.Path
        A completed ``select`` run directory.

    Returns
    -------
    pandas.DataFrame
        The summary table, one row per number of components, sorted by component count.
    """
    summary = cache.load_frame(run_directory / "summary.parquet")
    return summary.sort_values("n_components").reset_index(drop=True)


def load_replication(run_directory: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load a ``replicate`` run's metrics and its two category signatures.

    Returns
    -------
    tuple
        ``(metrics, spark_signature, ssc_signature)``: the metrics dictionary
        (``replication.json``) and the two class-by-category signature matrices.
    """
    metrics = cache.load_json(run_directory / "replication.json")
    spark_signature = cache.load_frame(run_directory / "spark_signature.parquet")
    ssc_signature = cache.load_frame(run_directory / "ssc_signature.parquet")
    return metrics, spark_signature, ssc_signature


def load_stability(run_directory: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Load a ``stability`` run's per-fit comparisons, aggregate, and mean overlap.

    Returns
    -------
    tuple
        ``(comparisons, aggregate, overlap_mean)``: one row per compared fit, the aggregate
        dictionary (``aggregate.json``), and the mean class-overlap matrix.
    """
    comparisons = cache.load_frame(run_directory / "comparisons.parquet")
    aggregate = cache.load_json(run_directory / "aggregate.json")
    overlap_mean = cache.load_frame(run_directory / "overlap_mean.parquet")
    return comparisons, aggregate, overlap_mean


def load_trajectory(run_directory: Path) -> tuple[pd.DataFrame, dict]:
    """Load a ``trajectory`` run's embedding table and its manifest metrics.

    Returns
    -------
    tuple
        ``(embedding, meta)``: the ``embedding_<axis>`` table (anchors and stratum centroids
        in discriminant coordinates) and the manifest metrics (``axis``, ``n_strata``).
    """
    manifest = cache.read_manifest(run_directory) or {}
    axis = manifest.get("params", {}).get("axis")
    embedding = cache.load_frame(run_directory / f"embedding_{axis}.parquet")
    return embedding, manifest.get("metrics", {})


def load_roughness(run_directory: Path) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Load a ``trajectory`` run's roughness and directional tables.

    Returns
    -------
    tuple
        ``(axis, roughness, directional)``: the axis id and the per-class ``roughness_<axis>``
        and ``directional_<axis>`` tables.
    """
    manifest = cache.read_manifest(run_directory) or {}
    axis = manifest.get("params", {}).get("axis")
    roughness = cache.load_frame(run_directory / f"roughness_{axis}.parquet")
    directional = cache.load_frame(run_directory / f"directional_{axis}.parquet")
    return str(axis), roughness, directional


def load_nmin(run_directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load an ``nmin`` run's per-fit metrics, per-size summary, and floor metrics.

    Returns
    -------
    tuple
        ``(per_fit, summary, metrics)``: one row per (size, replicate) fit, the per-size
        summary, and the manifest metrics (the floor and its bootstrap interval).
    """
    per_fit = cache.load_frame(run_directory / "per_fit.parquet")
    summary = cache.load_frame(run_directory / "summary.parquet")
    manifest = cache.read_manifest(run_directory) or {}
    metrics = manifest.get("metrics", {})
    return per_fit, summary, metrics
