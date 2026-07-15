"""Locating and loading the cached analysis run a figure visualises."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from analysis import cache
from analysis.paths import run_dir, stage_dir


def resolve_run(
    root: Path,
    stage: str,
    run: str | None = None,
    *,
    axis: str | None = None,
    require: dict[str, object] | None = None,
) -> Path:
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
    require : dict, optional
        Further manifest-parameter equalities the latest run must satisfy, so one stage that
        writes several run flavours (for example the ``drift`` stage's pooled and pairwise runs)
        can be told apart by ``{"reference_scheme": "pairwise"}``. Ignored when ``run`` is given.

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
            params = manifest.get("params", {})
            if axis is not None and params.get("axis") != axis:
                continue
            if require is not None and any(
                params.get(key) != value for key, value in require.items()
            ):
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


def load_attribution(run_directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load an ``attribute`` run's summary, category, and mover tables and its metrics.

    Returns
    -------
    tuple
        ``(summary, category, movers, meta)``: the per-class ``summary_<axis>`` headline, the
        per-category ``category_<axis>`` contributions, the per-feature ``movers_<axis>``
        contrast, and the manifest metrics (``axis``).
    """
    manifest = cache.read_manifest(run_directory) or {}
    axis = manifest.get("params", {}).get("axis")
    summary = cache.load_frame(run_directory / f"summary_{axis}.parquet")
    category = cache.load_frame(run_directory / f"category_{axis}.parquet")
    movers = cache.load_frame(run_directory / f"movers_{axis}.parquet")
    return summary, category, movers, manifest.get("metrics", {})


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


def load_sweep(run_directory: Path) -> tuple[pd.DataFrame, dict]:
    """Load a `sweep` run's decision table and its manifest.

    Parameters
    ----------
    run_directory : pathlib.Path
        The `sweep` run directory.

    Returns
    -------
    tuple
        The decision table (``decision_<axis>.parquet``) and the run manifest.
    """
    decisions = sorted(run_directory.glob("decision_*.parquet"))
    if not decisions:
        raise FileNotFoundError(f"no decision table in {run_directory}")
    return cache.load_frame(decisions[0]), cache.read_manifest(run_directory) or {}


def load_invariance(run_directory: Path) -> tuple[pd.DataFrame, dict]:
    """Load an `invariance` run's stored fluctuation process and its manifest.

    Parameters
    ----------
    run_directory : pathlib.Path
        The `invariance` run directory.

    Returns
    -------
    tuple
        The process table (``process_<axis>.parquet``) and the run manifest.
    """
    processes = sorted(run_directory.glob("process_*.parquet"))
    if not processes:
        raise FileNotFoundError(f"no process table in {run_directory}")
    return cache.load_frame(processes[0]), cache.read_manifest(run_directory) or {}


def load_pairwise(run_directory: Path) -> tuple[pd.DataFrame, dict]:
    """Load a pairwise ``drift`` run's trajectory table and its manifest metrics.

    Parameters
    ----------
    run_directory : pathlib.Path
        The ``drift`` run directory of a ``--reference-scheme pairwise`` run.

    Returns
    -------
    tuple
        The trajectory table (``pairwise_<axis>.parquet``: one row per neighbour comparison and
        reference class) and the manifest metrics (``axis``, ``mode``, ``class_separation``).
    """
    tables = sorted(run_directory.glob("pairwise_*.parquet"))
    if not tables:
        raise FileNotFoundError(f"no pairwise trajectory table in {run_directory}")
    manifest = cache.read_manifest(run_directory) or {}
    return cache.load_frame(tables[0]), manifest.get("metrics", {})


def load_local_trajectory(run_directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load an ``invariance-trajectory`` run's plane and capture tables and its metrics.

    Returns
    -------
    tuple
        ``(plane, capture, meta)``: the ``trajectory_<axis>`` discriminant-plane table (anchors
        and per-focal centroids with the bootstrap tube), the per-class ``capture_<axis>`` table,
        and the manifest metrics (``axis``).
    """
    manifest = cache.read_manifest(run_directory) or {}
    axis = manifest.get("params", {}).get("axis")
    plane = cache.load_frame(run_directory / f"trajectory_{axis}.parquet")
    capture = cache.load_frame(run_directory / f"capture_{axis}.parquet")
    return plane, capture, manifest.get("metrics", {})


def load_local_specificity(run_directory: Path) -> pd.DataFrame:
    """Load an ``invariance-trajectory`` run's specificity table (endpoint magnitude by axis)."""
    manifest = cache.read_manifest(run_directory) or {}
    axis = manifest.get("params", {}).get("axis")
    return cache.load_frame(run_directory / f"specificity_{axis}.parquet")


def load_local_directional(run_directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load an ``invariance-trajectory`` run's H0E tables and its metrics.

    Returns
    -------
    tuple
        ``(signed, directional, meta)``: the ``signed_trajectory_<axis>`` table (per class per
        focal point, the one-dimensional signed trajectory with its bootstrap band), the per-class
        ``directional_<axis>`` summary (net trend, interval, ``p``, FDR decision, break), and the
        manifest metrics (``axis``).
    """
    manifest = cache.read_manifest(run_directory) or {}
    axis = manifest.get("params", {}).get("axis")
    signed = cache.load_frame(run_directory / f"signed_trajectory_{axis}.parquet")
    directional = cache.load_frame(run_directory / f"directional_{axis}.parquet")
    return signed, directional, manifest.get("metrics", {})


def load_grain_magnitude(run_directory: Path) -> pd.DataFrame:
    """Load an ``invariance-trajectory`` run's per-grain magnitude table (H0F category grains).

    Returns
    -------
    pandas.DataFrame
        The ``grain_magnitude_<axis>`` frame: per grain (``"class"`` and ``"category:<name>"``),
        class, and focal point, the separation-scaled magnitude with its bootstrap band.
    """
    manifest = cache.read_manifest(run_directory) or {}
    axis = manifest.get("params", {}).get("axis")
    return cache.load_frame(run_directory / f"grain_magnitude_{axis}.parquet")


def load_feature_displacement(run_directory: Path) -> pd.DataFrame:
    """Load an ``invariance-trajectory`` run's per-feature displacement table.

    Returns
    -------
    pandas.DataFrame
        The ``feature_displacement_<axis>`` frame: per class and feature, the signed
        separation-standardised endpoint displacement, its category, and the FDR decision.
    """
    manifest = cache.read_manifest(run_directory) or {}
    axis = manifest.get("params", {}).get("axis")
    return cache.load_frame(run_directory / f"feature_displacement_{axis}.parquet")


def load_demographic_conditioning(run_directory: Path) -> pd.DataFrame:
    """Load a ``demographic-conditioning`` run's per-covariate, per-class shrinkage table.

    Returns
    -------
    pandas.DataFrame
        The ``demographic_conditioning_<axis>`` frame: per covariate and reference class, the
        shrinkage, the raw and conditioned magnitude, the covariate's axis $R^2$, the covariate
        ``label``, ``kind``, and ``coding``, and the joined sample size.
    """
    manifest = cache.read_manifest(run_directory) or {}
    axis = manifest.get("params", {}).get("axis")
    return cache.load_frame(run_directory / f"demographic_conditioning_{axis}.parquet")


def load_referent(run_directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load an era ``invariance-trajectory`` run's H0G tables and its metrics.

    Returns
    -------
    tuple
        ``(grains, contrast, meta)``: the ``referent_<axis>`` table (per class per grain, the
        per-referent and per-instrument root-mean-square intensity, additive share, and FDR count),
        the per-class ``referent_contrast_<axis>`` table (the current-minus-retrospective contrast
        with its interval, ``p``, FDR decision, and mechanism), and the manifest metrics (``axis``).
    """
    manifest = cache.read_manifest(run_directory) or {}
    axis = manifest.get("params", {}).get("axis")
    grains = cache.load_frame(run_directory / f"referent_{axis}.parquet")
    contrast = cache.load_frame(run_directory / f"referent_contrast_{axis}.parquet")
    return grains, contrast, manifest.get("metrics", {})


def load_prevalence(run_directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load a ``prevalence`` run's proportion-curve and slope tables and its metrics.

    Returns
    -------
    tuple
        ``(curve, slopes, meta)``: the ``proportion_curve_<axis>`` table (per class per focal
        point, the corrected and naive predicted proportion with the corrected bootstrap band),
        the ``slopes_<axis>`` table (the corrected, naive, adjusted, and DSM-5 per-class contrasts),
        and the manifest metrics (``axis``).
    """
    manifest = cache.read_manifest(run_directory) or {}
    axis = manifest.get("params", {}).get("axis")
    curve = cache.load_frame(run_directory / f"proportion_curve_{axis}.parquet")
    slopes = cache.load_frame(run_directory / f"slopes_{axis}.parquet")
    return curve, slopes, manifest.get("metrics", {})


def load_atlas(run_directory: Path) -> tuple[pd.DataFrame, dict]:
    """Load a ``displacement-atlas`` run's per-axis, per-class endpoint table and its metrics.

    Returns
    -------
    tuple
        ``(atlas, meta)``: the ``displacement_atlas`` frame (per axis and reference class, the
        separation-scaled endpoint displacement, the axis ``label`` and ``kind``, and the joined
        sample size), and the manifest metrics (the class-summed displacement per axis, the random
        floor, and the axes dropped below the coverage floor).
    """
    manifest = cache.read_manifest(run_directory) or {}
    atlas = cache.load_frame(run_directory / "displacement_atlas.parquet")
    return atlas, manifest.get("metrics", {})


def class_names(root: Path, axis: str) -> dict[int, str]:
    """Return the reference-class id to name map for an axis, from its `trajectory` run.

    The `sweep` decision table carries only class ids, so the names come from the latest
    `trajectory` run for the axis, whose directional table pairs each id with its name. An empty
    map is returned when no such run exists, and the figure falls back to the numeric ids.
    """
    try:
        directory = resolve_run(root, "trajectory", axis=axis)
    except FileNotFoundError:
        return {}
    directionals = sorted(directory.glob("directional_*.parquet"))
    if not directionals:
        return {}
    frame = cache.load_frame(directionals[0])
    return {
        int(ref_class): str(class_name)
        for ref_class, class_name in zip(frame["ref_class"], frame["class_name"], strict=True)
    }
