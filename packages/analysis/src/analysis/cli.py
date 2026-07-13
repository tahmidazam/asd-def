"""analysis command-line interface (Typer).

One subcommand per pipeline stage. Each implemented stage reads named inputs and writes
named outputs plus a manifest under ``artefacts/<stage>/<run-hash>/``, so a later run
recomputes only what changed (plan section 11). Stages not yet written are grouped under a
"planned" panel in ``analysis --help`` and exit non-zero; a command leaves that panel when
its stage is implemented.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from analysis import cache, checkpoint, config, features, model
from analysis.cohort import build_matrix, get_cohort
from analysis.cohort.schema import load_feature_list
from analysis.features import Typing
from analysis.paths import find_repo_root
from analysis.run import run_context

if TYPE_CHECKING:
    import pandas as pd

    from analysis.cohort import CohortMatrix

app = typer.Typer(
    name="analysis",
    help="Reproduce the Litman autism classes and test their stability across age at "
    "diagnosis and diagnostic era.",
    no_args_is_help=True,
    add_completion=False,
)

_PLANNED = "Planned (not yet implemented)"


def _todo(stage: str, phase: int) -> None:
    """Report that ``stage`` is not implemented yet and exit non-zero."""
    typer.echo(f"the {stage!r} stage is planned for phase {phase}, not implemented yet", err=True)
    raise typer.Exit(1)


def _cohort_params(
    root: Path,
    dataset: str,
    version: str,
    *,
    as_of: str | None = None,
    sample_n: int | None = None,
    sample_seed: int = 0,
) -> dict[str, object]:
    """Return the hashing parameters for the cohort (and typing) stage.

    The records cutoff and the size-matched subsample enter the hash only when set, so a
    default (full-cohort) run keeps the same hash as before they existed and its cache stays
    valid. A cutoff or subsample run lands at its own hash, cached separately.
    """
    typing_dir = config.litman_typing_dir(root)
    params: dict[str, object] = {
        "dataset": dataset,
        "version": version,
        "feature_list": cache.file_digest(config.author_feature_list(root)),
        "covariates": list(config.COVARIATES),
        "instruments": list(config.COHORT_INSTRUMENTS),
        "typing_pickles": {
            name: cache.file_digest(typing_dir / f"{name}_columns.pkl")
            for name in ("binary", "categorical", "continuous")
        },
    }
    if as_of is not None:
        params["as_of"] = as_of
    if sample_n is not None:
        params["sample_n"] = sample_n
        params["sample_seed"] = sample_seed
    return params


def _run_cohort(
    root: Path,
    dataset: str,
    version: str,
    *,
    force: bool,
    as_of: str | None = None,
    sample_n: int | None = None,
    sample_seed: int = 0,
) -> tuple[str, dict]:
    """Build (or load) the cohort matrix and typing, returning the run hash and metrics."""
    params = _cohort_params(
        root, dataset, version, as_of=as_of, sample_n=sample_n, sample_seed=sample_seed
    )
    with run_context("cohort", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            manifest = cache.read_manifest(ctx.run_dir) or {}
            return ctx.run_hash, manifest.get("metrics", {})
        feature_names = load_feature_list(config.author_feature_list(root))
        cohort = get_cohort(dataset, version, root, as_of=as_of)
        integrated = cohort.integrate()
        if sample_n is not None:
            # The size-matched control: a random draw of the full cohort, separating "fewer
            # records" (this draw) from "different records" (the cutoff subset) when the two
            # are read side by side.
            integrated = integrated.sample(n=sample_n, random_state=sample_seed)
        matrix = build_matrix(integrated, feature_names, dataset, version)
        typing, report = features.build_typing(
            root, dataset, version, feature_names, frame=integrated
        )

        cache.save_frame(matrix.features, ctx.path("features.parquet"))
        cache.save_frame(matrix.covariates, ctx.path("covariates.parquet"))
        cache.save_frame(report, ctx.path("typing_report.parquet"))
        cache.save_json(
            {
                "continuous": typing.continuous,
                "binary": typing.binary,
                "categorical": typing.categorical,
            },
            ctx.path("typing.json"),
        )

        n_conflicts = int((report["dictionary_pickle_agree"] == False).sum())  # noqa: E712
        ctx.metrics = {
            "n_probands": matrix.n_probands,
            "n_features": len(feature_names),
            "typing_counts": typing.counts,
            "typing_conflicts": n_conflicts,
            "supports_timing": cohort.supports_timing(),
            "as_of": as_of,
            "sample_n": sample_n,
        }
        ctx.log.info(
            "cohort %s/%s: %d probands, typing %s, %d typing conflict(s)",
            dataset,
            version,
            matrix.n_probands,
            typing.counts,
            n_conflicts,
        )
        return ctx.run_hash, ctx.metrics


def _load_cohort_matrix(root: Path, cohort_hash: str, dataset: str, version: str):
    """Load the cached cohort matrix and typing for a cohort run hash."""
    from analysis.cohort import CohortMatrix
    from analysis.paths import run_dir

    rdir = run_dir(root, "cohort", cache.short_hash(cohort_hash))
    features_df = cache.load_frame(rdir / "features.parquet")
    covariates_df = cache.load_frame(rdir / "covariates.parquet")
    typing_json = cache.load_json(rdir / "typing.json")
    matrix = CohortMatrix(features_df, covariates_df, dataset, version)
    typing = Typing(**typing_json)
    return matrix, typing


def _build_shared_cohort(
    root: Path, dataset: str, version: str, shared_with: str, *, force: bool
) -> tuple[CohortMatrix, Typing, str]:
    """Build ``dataset``'s matrix on the features it shares with ``shared_with``.

    Cross-cohort stratification (plan section 8) re-estimates the mixture within strata on the
    intersection of the two cohorts' harmonised features, so a cohort that provides only a
    subset of the 238 author features (the SSC) can still be fitted and compared against the
    reference cohort. This reuses the shared-feature construction of the replicate stage
    (:func:`analysis.replicate.shared_feature_set`). The reference cohort (``shared_with``, at
    :data:`config.REFERENCE_VERSION`) is built on the full feature set; this cohort is
    restricted to the shared columns. The reference typing is returned unchanged and filtered
    to the shared columns at fit time (as in the replicate stage). The returned hash is a
    content digest over the integrated frame and the shared feature set, so a change to either
    cohort's harmonisation invalidates the cache.
    """
    from analysis.cohort import CohortMatrix
    from analysis.replicate import shared_feature_set

    other_hash, _ = _run_cohort(root, shared_with, config.REFERENCE_VERSION, force=force)
    other_matrix, typing = _load_cohort_matrix(
        root, other_hash, shared_with, config.REFERENCE_VERSION
    )

    integrated = get_cohort(dataset, version, root).integrate()
    feature_set = set(load_feature_list(config.author_feature_list(root)))
    available = [c for c in integrated.columns if c in feature_set and c not in config.COVARIATES]
    matrix = build_matrix(integrated, available, dataset, version)

    shared = shared_feature_set(other_matrix, matrix)
    shared_matrix = CohortMatrix(matrix.features[shared], matrix.covariates, dataset, version)
    cohort_hash = cache.compute_hash(
        {
            "dataset": dataset,
            "version": version,
            "shared_with": shared_with,
            "reference_version": config.REFERENCE_VERSION,
            "content": cache.frame_digest(integrated),
            "shared_features": shared,
        }
    )
    return shared_matrix, typing, cohort_hash


_DATASET = typer.Option(config.REFERENCE_DATASET, "--dataset", "-d", help="Cohort id.")
_VERSION = typer.Option(config.REFERENCE_VERSION, "--version", "-v", help="Dataset version.")
_FORCE = typer.Option(False, "--force", help="Recompute even on a cache hit.")
_AS_OF = typer.Option(
    None,
    "--as-of",
    help="Restrict to records present at a SPARK freeze (e.g. 2022-12-12, the V9 cut).",
)
_SAMPLE_N = typer.Option(
    None, "--sample-n", help="Draw a random subsample of this many probands (size-matched control)."
)
_SAMPLE_SEED = typer.Option(0, "--sample-seed", help="Seed for --sample-n.")


@app.command()
def cohort(
    dataset: str = _DATASET,
    version: str = _VERSION,
    as_of: str | None = _AS_OF,
    sample_n: int | None = _SAMPLE_N,
    sample_seed: int = _SAMPLE_SEED,
    force: bool = _FORCE,
) -> None:
    """Build the harmonised proband-by-feature matrix and its typing manifest."""
    root = find_repo_root()
    run_hash, metrics = _run_cohort(
        root, dataset, version, force=force, as_of=as_of, sample_n=sample_n, sample_seed=sample_seed
    )
    typer.echo(f"cohort {dataset}/{version}: run {cache.short_hash(run_hash)}")
    typer.echo(f"  probands={metrics['n_probands']} features={metrics['n_features']}")
    typer.echo(f"  typing={metrics['typing_counts']} conflicts={metrics['typing_conflicts']}")
    if as_of is not None or sample_n is not None:
        typer.echo(f"  as_of={as_of} sample_n={sample_n}")


def _fit_params(cohort_hash: str, n_components: int, n_init: int, seed: int) -> dict[str, object]:
    """Return the hashing parameters for the fit stage."""
    return {
        "cohort": cohort_hash,
        "n_components": n_components,
        "n_init": n_init,
        "n_steps": config.DEFAULT_N_STEPS,
        "seed": seed,
        "structural": "covariate",
        "round": True,
    }


def _measurement_fit_params(cohort_hash: str, n_init: int, seed: int) -> dict[str, object]:
    """Return the fit-stage hash parameters for the measurement-only reference.

    The score-based invariance test reads the marginal (measurement-only) reference, the fit
    produced by ``analysis fit --no-covariates``. It differs from the covariate reference only in
    the ``structural`` key, so the hash recomputes the exact directory the measurement-only fit
    wrote (``41ab0e38...`` on SPARK 2026-03-23 at ``n_init=50``).
    """
    params = _fit_params(cohort_hash, config.DEFAULT_N_COMPONENTS, n_init, seed)
    return {**params, "structural": "measurement"}


def _resolve_measurement_reference(root: Path, cohort_hash: str, n_init: int, seed: int) -> str:
    """Return the run hash of the measurement-only reference fit for a cohort.

    Recomputes the expected hash first (the ``n_init`` default matches the cached reference), and
    falls back to scanning the fit artefacts for a completed measurement-only four-class fit on
    the same cohort, so a reference produced at a different ``n_init`` is still found. Raises a
    clear instruction when none exists.
    """
    from analysis.paths import run_dir

    expected = cache.compute_hash(_measurement_fit_params(cohort_hash, n_init, seed))
    if _completed(run_dir(root, "fit", cache.short_hash(expected))):
        return expected

    candidates: list[tuple[int, str]] = []
    for manifest_path in sorted((root / "artefacts" / "fit").glob("*/manifest.json")):
        manifest = cache.read_manifest(manifest_path.parent) or {}
        params = manifest.get("params", {})
        if (
            manifest.get("status") == "ok"
            and params.get("structural") == "measurement"
            and params.get("n_components") == config.DEFAULT_N_COMPONENTS
            and params.get("cohort") == cohort_hash
        ):
            candidates.append((int(params.get("n_init", 0)), manifest["hash"]))
    if candidates:
        # Prefer the most thorough fit (largest n_init); ties broken by hash for determinism.
        return max(candidates, key=lambda c: (c[0], c[1]))[1]

    raise typer.BadParameter(
        "no measurement-only reference fit found; run `analysis fit --no-covariates` first"
    )


def _align_params(root: Path, fit_hash: str) -> dict[str, object]:
    """Return the hashing parameters for the align stage."""
    return {"fit": fit_hash, "category_map": cache.file_digest(config.author_category_map(root))}


def _completed(run_dir: Path) -> bool:
    """Return whether a run finished cleanly (a manifest exists with status ``ok``).

    A run directory can hold a manifest with status ``running`` or ``failed`` (the lifecycle
    writes it before the body runs and only flips it in the finally block), alongside missing
    or half-written artefacts. A reference must have completed, so presence alone is not
    enough; this mirrors the cache-hit gate in :mod:`analysis.run`.
    """
    manifest = cache.read_manifest(run_dir)
    return manifest is not None and manifest.get("status") == "ok"


def _load_reference(
    root: Path,
    dataset: str,
    version: str,
    *,
    n_components: int,
    n_init: int,
    seed: int,
    force: bool,
    as_of: str | None = None,
    sample_n: int | None = None,
    sample_seed: int = 0,
):
    """Load the cached reference cohort, typing, labels, and enrichment.

    The reference is the canonical fit (``analysis fit``) and its alignment
    (``analysis align``); the stability, nmin, and report stages compare against it. Exits
    non-zero with guidance when either stage has not completed cleanly for these settings.
    The records cutoff and size-matched subsample select which cohort (and so which reference
    fit) to compare against, so a cutoff run resolves its own subset reference.
    """
    from analysis.paths import run_dir

    cohort_hash, _ = _run_cohort(
        root, dataset, version, force=force, as_of=as_of, sample_n=sample_n, sample_seed=sample_seed
    )
    matrix, typing = _load_cohort_matrix(root, cohort_hash, dataset, version)

    fit_hash = cache.compute_hash(_fit_params(cohort_hash, n_components, n_init, seed))
    fit_dir = run_dir(root, "fit", cache.short_hash(fit_hash))
    if not _completed(fit_dir):
        typer.echo(
            f"no completed reference fit for these settings (n_init={n_init}, seed={seed}); "
            "run `analysis fit` first",
            err=True,
        )
        raise typer.Exit(1)
    reference_labels = cache.load_frame(fit_dir / "labels.parquet")["class"]

    align_hash = cache.compute_hash(_align_params(root, fit_hash))
    align_dir = run_dir(root, "align", cache.short_hash(align_hash))
    if not _completed(align_dir):
        typer.echo(
            "no completed reference alignment for these settings; run `analysis align` first",
            err=True,
        )
        raise typer.Exit(1)
    reference_enrichment = cache.load_frame(align_dir / "enrichment.parquet")

    return matrix, typing, reference_labels, reference_enrichment, align_hash


@app.command()
def fit(
    dataset: str = _DATASET,
    version: str = _VERSION,
    n_components: int = typer.Option(config.DEFAULT_N_COMPONENTS, help="Number of latent classes."),
    n_init: int = typer.Option(config.DEFAULT_N_INIT, help="Random restarts (StepMix n_init)."),
    seed: int = typer.Option(0, help="Random seed for reproducible restarts."),
    as_of: str | None = _AS_OF,
    sample_n: int | None = _SAMPLE_N,
    sample_seed: int = _SAMPLE_SEED,
    no_covariates: bool = typer.Option(
        False,
        "--no-covariates",
        help="Fit the measurement model alone (no covariate structural model), the "
        "measurement-only reference the kernel sweep is compared against.",
    ),
    force: bool = _FORCE,
) -> None:
    """Fit the reference general finite mixture model and predict class labels."""
    root = find_repo_root()
    cohort_hash, _ = _run_cohort(
        root, dataset, version, force=force, as_of=as_of, sample_n=sample_n, sample_seed=sample_seed
    )
    structural = None if no_covariates else "covariate"
    params = _fit_params(cohort_hash, n_components, n_init, seed)
    if no_covariates:
        params = {**params, "structural": "measurement"}
    with run_context("fit", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            metrics = (cache.read_manifest(ctx.run_dir) or {}).get("metrics", {})
            typer.echo(
                f"fit: cache hit {cache.short_hash(ctx.run_hash)}; "
                f"proportions {metrics['class_proportions']}"
            )
            return
        matrix, typing = _load_cohort_matrix(root, cohort_hash, dataset, version)
        ctx.log.info("fitting GFMM: n_components=%d n_init=%d seed=%d", n_components, n_init, seed)
        result = model.fit_gfmm(
            matrix,
            typing,
            n_components=n_components,
            n_init=n_init,
            random_state=seed,
            structural=structural,
        )
        centroids = model.class_centroids(result.measurement_data, result.labels)
        cache.save_model(result.model, ctx.path("model.joblib"))
        cache.save_frame(result.labels.to_frame(), ctx.path("labels.parquet"))
        cache.save_frame(centroids, ctx.path("centroids.parquet"))
        ctx.metrics = result.metrics
        ctx.log.info(
            "fit done: proportions=%s bic=%.0f",
            result.metrics["class_proportions"],
            result.metrics["bic"],
        )
    typer.echo(f"fit {dataset}/{version}: run {cache.short_hash(ctx.run_hash)}")
    typer.echo(
        f"  proportions={ctx.metrics['class_proportions']} "
        f"smallest={ctx.metrics['smallest_class_proportion']}"
    )


@app.command()
def align(
    dataset: str = _DATASET,
    version: str = _VERSION,
    n_components: int = typer.Option(config.DEFAULT_N_COMPONENTS, help="Number of latent classes."),
    n_init: int = typer.Option(config.DEFAULT_N_INIT, help="Random restarts (StepMix n_init)."),
    seed: int = typer.Option(0, help="Random seed for reproducible restarts."),
    as_of: str | None = _AS_OF,
    sample_n: int | None = _SAMPLE_N,
    sample_seed: int = _SAMPLE_SEED,
    force: bool = _FORCE,
) -> None:
    """Compute the seven-category signature and align our classes to Litman's named classes."""
    from analysis import enrich, reference
    from analysis.paths import run_dir

    root = find_repo_root()
    cohort_hash, _ = _run_cohort(
        root, dataset, version, force=force, as_of=as_of, sample_n=sample_n, sample_seed=sample_seed
    )
    fit_hash = cache.compute_hash(_fit_params(cohort_hash, n_components, n_init, seed))
    fit_dir = run_dir(root, "fit", cache.short_hash(fit_hash))
    if not _completed(fit_dir):
        typer.echo("no completed fit for these settings; run `analysis fit` first", err=True)
        raise typer.Exit(1)

    params = {"fit": fit_hash, "category_map": cache.file_digest(config.author_category_map(root))}
    with run_context("align", params, root=root, force=force) as ctx:
        matrix, typing = _load_cohort_matrix(root, cohort_hash, dataset, version)
        labels = cache.load_frame(fit_dir / "labels.parquet")["class"]
        proportions = (cache.read_manifest(fit_dir) or {})["metrics"]["class_proportions"]
        proportions = {int(k): float(v) for k, v in proportions.items()}

        measurement_data, _, _ = model.prepare_inputs(matrix, typing)
        ctx.log.info("computing per-class enrichment over %d features", measurement_data.shape[1])
        enrichment = enrich.feature_enrichment(measurement_data, labels, n_classes=n_components)
        category_map = features.load_category_map(config.author_category_map(root))
        signature = enrich.category_signature(enrichment, category_map, n_classes=n_components)
        named = reference.align_to_named(signature, proportions)

        # Put a confidence interval on the overall reproduction correlation by resampling
        # probands with the fitted labels held fixed, so the headline r carries its sampling
        # uncertainty rather than standing as a single number (plan section 4).
        order = [named.mapping[c] for c in range(n_components)]
        target = reference.published_signature().loc[order].set_axis(range(n_components))
        ctx.log.info(
            "bootstrapping the overall correlation (%d resamples)", config.DEFAULT_N_BOOTSTRAP
        )
        correlation_ci = enrich.bootstrap_overall_correlation(
            measurement_data,
            labels,
            target,
            category_map,
            n_boot=config.DEFAULT_N_BOOTSTRAP,
            seed=config.DEFAULT_BOOTSTRAP_SEED,
            n_classes=n_components,
        )

        cache.save_frame(enrichment, ctx.path("enrichment.parquet"))
        cache.save_frame(signature, ctx.path("signature.parquet"))
        cache.save_json(
            {
                "mapping": {str(k): v for k, v in named.mapping.items()},
                "correlations": {str(k): v for k, v in named.correlations.items()},
                "overall_correlation": named.overall_correlation,
                "overall_correlation_ci": correlation_ci,
                "anchors": named.anchors,
                "anchors_hold": named.anchors_hold,
            },
            ctx.path("alignment.json"),
        )
        ctx.metrics = {
            "mapping": {str(k): v for k, v in named.mapping.items()},
            "anchors_hold": named.anchors_hold,
            "overall_correlation": round(named.overall_correlation, 4),
            "overall_correlation_ci": correlation_ci,
        }
        ctx.log.info(
            "named-class mapping: %s (anchors hold: %s, overall r=%.3f)",
            named.mapping,
            named.anchors_hold,
            named.overall_correlation,
        )

    typer.echo(f"align {dataset}/{version}: run {cache.short_hash(ctx.run_hash)}")
    for cid, name in named.mapping.items():
        r = named.correlations[cid]
        r_text = "n/a (saturated)" if r is None else f"{r:.2f}"
        typer.echo(f"  class {cid} -> {name} (profile r={r_text})")
    typer.echo(f"  overall profile correlation: {named.overall_correlation:.3f}")
    typer.echo(
        f"  95% bootstrap CI: [{correlation_ci['ci_low']:.3f}, {correlation_ci['ci_high']:.3f}] "
        f"over {correlation_ci['n_valid']} resamples"
    )
    typer.echo(f"  anchors hold: {named.anchors_hold} {named.anchors}")


@app.command()
def select(
    dataset: str = _DATASET,
    version: str = _VERSION,
    k_min: int = typer.Option(1, help="Smallest number of components in the grid."),
    k_max: int = typer.Option(10, help="Largest number of components in the grid."),
    n_iterations: int = typer.Option(20, help="Seeded repetitions (Litman use 200)."),
    n_init: int = typer.Option(1, help="Random restarts per fit (Litman validation use 1)."),
    cv: int = typer.Option(3, help="Cross-validation folds for the validation log-likelihood."),
    seed: int = typer.Option(0, help="Base seed; iteration i uses seed+i."),
    as_of: str | None = _AS_OF,
    sample_n: int | None = _SAMPLE_N,
    sample_seed: int = _SAMPLE_SEED,
    workers: int = typer.Option(0, help="Parallel fit workers (0 = logical cores minus one)."),
    force: bool = _FORCE,
) -> None:
    """Grid over the number of components and report the information criteria."""
    from analysis import selection

    root = find_repo_root()
    cohort_hash, _ = _run_cohort(
        root, dataset, version, force=force, as_of=as_of, sample_n=sample_n, sample_seed=sample_seed
    )
    k_values = list(range(k_min, k_max + 1))
    params = {
        "cohort": cohort_hash,
        "k_values": k_values,
        "n_iterations": n_iterations,
        "n_init": n_init,
        "cv": cv,
        "seed": seed,
    }
    with run_context("select", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            metrics = (cache.read_manifest(ctx.run_dir) or {}).get("metrics", {})
            typer.echo(f"select: cache hit {cache.short_hash(ctx.run_hash)}")
            typer.echo(f"  {metrics}")
            return
        if force:
            checkpoint.clear_checkpoints(ctx.run_dir)
        matrix, typing = _load_cohort_matrix(root, cohort_hash, dataset, version)
        ctx.log.info("model selection over K=%s, %d iterations", k_values, n_iterations)
        result = selection.run_selection(
            matrix,
            typing,
            k_values=k_values,
            n_iterations=n_iterations,
            n_init=n_init,
            base_seed=seed,
            cv=cv,
            checkpoint_dir=ctx.run_dir,
            workers=workers,
        )
        cache.save_frame(result.per_iteration, ctx.path("per_iteration.parquet"))
        cache.save_frame(result.summary, ctx.path("summary.parquet"))
        checkpoint.clear_checkpoints(ctx.run_dir)
        best = {
            "bic": int(result.summary.loc[result.summary["bic_mean"].idxmin(), "n_components"]),
            "aic": int(result.summary.loc[result.summary["aic_mean"].idxmin(), "n_components"]),
            "caic": int(result.summary.loc[result.summary["caic_mean"].idxmin(), "n_components"]),
        }
        ctx.metrics = {"k_values": k_values, "best_by_criterion": best}
        ctx.log.info("criteria minimised at: %s (reference choice is K=4)", best)
    typer.echo(f"select {dataset}/{version}: run {cache.short_hash(ctx.run_hash)}")
    typer.echo(f"  criteria minimised at {ctx.metrics['best_by_criterion']} (reference choice K=4)")


_STABILITY_MODES = ("multi-init", "subsample")


@app.command()
def stability(
    dataset: str = _DATASET,
    version: str = _VERSION,
    mode: str = typer.Option(
        "multi-init", help="'multi-init' (rank single-init fits) or 'subsample' (refit halves)."
    ),
    n_fits: int = typer.Option(200, help="multi-init: single-init fits (Litman use 2,000)."),
    top_k: int = typer.Option(100, help="multi-init: best fits compared to the reference."),
    n_reps: int = typer.Option(50, help="subsample: replicates (Litman use 100)."),
    frac: float = typer.Option(0.5, help="subsample: fraction without replacement."),
    sub_n_init: int = typer.Option(20, help="subsample: restarts per replicate (Litman use 20)."),
    n_components: int = typer.Option(config.DEFAULT_N_COMPONENTS, help="Number of latent classes."),
    ref_n_init: int = typer.Option(config.DEFAULT_N_INIT, help="Reference fit n_init to load."),
    ref_seed: int = typer.Option(0, help="Seed of the reference fit to compare against."),
    seed: int = typer.Option(0, help="Base seed for the stability fits."),
    as_of: str | None = _AS_OF,
    sample_n: int | None = _SAMPLE_N,
    sample_seed: int = _SAMPLE_SEED,
    workers: int = typer.Option(0, help="Parallel fit workers (0 = logical cores minus one)."),
    force: bool = _FORCE,
) -> None:
    """Summarise multi-initialisation or subsampling stability of the reference fit."""
    from analysis import stability as stability_mod

    if mode not in _STABILITY_MODES:
        typer.echo(f"--mode must be one of {_STABILITY_MODES}", err=True)
        raise typer.Exit(1)

    root = find_repo_root()
    matrix, typing, ref_labels, ref_enrichment, align_hash = _load_reference(
        root,
        dataset,
        version,
        n_components=n_components,
        n_init=ref_n_init,
        seed=ref_seed,
        force=force,
        as_of=as_of,
        sample_n=sample_n,
        sample_seed=sample_seed,
    )
    category_map = features.load_category_map(config.author_category_map(root))

    if mode == "multi-init":
        params = {
            "reference": align_hash,
            "mode": mode,
            "n_fits": n_fits,
            "top_k": top_k,
            "n_components": n_components,
            "seed": seed,
        }
    else:
        params = {
            "reference": align_hash,
            "mode": mode,
            "n_reps": n_reps,
            "frac": frac,
            "n_init": sub_n_init,
            "n_components": n_components,
            "seed": seed,
        }

    with run_context("stability", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            metrics = (cache.read_manifest(ctx.run_dir) or {}).get("metrics", {})
            typer.echo(f"stability ({mode}): cache hit {cache.short_hash(ctx.run_hash)}")
            typer.echo(f"  {metrics}")
            return
        if force:
            checkpoint.clear_checkpoints(ctx.run_dir)
        if mode == "multi-init":
            ctx.log.info("multi-init stability: %d fits, comparing top %d", n_fits, top_k)
            summary = stability_mod.run_multi_init_stability(
                matrix,
                typing,
                ref_labels,
                ref_enrichment,
                category_map,
                n_fits=n_fits,
                top_k=top_k,
                n_components=n_components,
                base_seed=seed,
                checkpoint_dir=ctx.run_dir,
                workers=workers,
            )
        else:
            ctx.log.info("subsampling stability: %d reps at frac=%.2f", n_reps, frac)
            summary = stability_mod.run_subsampling_stability(
                matrix,
                typing,
                ref_labels,
                ref_enrichment,
                category_map,
                n_reps=n_reps,
                frac=frac,
                n_init=sub_n_init,
                n_components=n_components,
                base_seed=seed,
                checkpoint_dir=ctx.run_dir,
                workers=workers,
            )
        cache.save_frame(summary.fits, ctx.path("fits.parquet"))
        cache.save_frame(summary.comparisons, ctx.path("comparisons.parquet"))
        cache.save_frame(summary.overlap_mean, ctx.path("overlap_mean.parquet"))
        cache.save_json(summary.aggregate, ctx.path("aggregate.json"))
        checkpoint.clear_checkpoints(ctx.run_dir)
        ctx.metrics = {
            k: summary.aggregate[k]
            for k in ("overall_correlation_mean", "adjusted_rand_index_mean", "n_compared")
            if k in summary.aggregate
        }
        ctx.log.info("stability aggregate: %s", ctx.metrics)
    typer.echo(f"stability ({mode}) {dataset}/{version}: run {cache.short_hash(ctx.run_hash)}")
    typer.echo(f"  {ctx.metrics}")


@app.command()
def nmin(
    dataset: str = _DATASET,
    version: str = _VERSION,
    sizes: str = typer.Option(
        "", help="Comma-separated target sizes; empty derives a default sweep."
    ),
    n_reps: int = typer.Option(3, help="Replicates per target size."),
    benchmark: float = typer.Option(0.9, help="Profile-correlation threshold defining recovery."),
    sweep_n_init: int = typer.Option(20, help="Restarts per fit."),
    n_components: int = typer.Option(config.DEFAULT_N_COMPONENTS, help="Number of latent classes."),
    ref_n_init: int = typer.Option(config.DEFAULT_N_INIT, help="Reference fit n_init to load."),
    seed: int = typer.Option(0, help="Base seed."),
    as_of: str | None = _AS_OF,
    sample_n: int | None = _SAMPLE_N,
    sample_seed: int = _SAMPLE_SEED,
    workers: int = typer.Option(0, help="Parallel fit workers (0 = logical cores minus one)."),
    force: bool = _FORCE,
) -> None:
    """Find the minimum viable stratum size by refitting at descending sample sizes."""
    from analysis import stability as stability_mod

    root = find_repo_root()
    matrix, typing, ref_labels, ref_enrichment, align_hash = _load_reference(
        root,
        dataset,
        version,
        n_components=n_components,
        n_init=ref_n_init,
        seed=0,
        force=force,
        as_of=as_of,
        sample_n=sample_n,
        sample_seed=sample_seed,
    )
    category_map = features.load_category_map(config.author_category_map(root))

    total = matrix.n_probands
    if sizes.strip():
        size_list = [int(s) for s in sizes.split(",") if s.strip()]
    else:
        fractions = (0.9, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05)
        size_list = [int(total * f) for f in fractions]

    params = {
        "reference": align_hash,
        "sizes": size_list,
        "n_reps": n_reps,
        "benchmark": benchmark,
        "n_init": sweep_n_init,
        "n_components": n_components,
        "seed": seed,
    }
    with run_context("nmin", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            metrics = (cache.read_manifest(ctx.run_dir) or {}).get("metrics", {})
            typer.echo(f"nmin: cache hit {cache.short_hash(ctx.run_hash)}; {metrics}")
            return
        if force:
            checkpoint.clear_checkpoints(ctx.run_dir)
        ctx.log.info("nmin sweep over sizes %s, benchmark r>=%.2f", size_list, benchmark)
        result = stability_mod.run_nmin_sweep(
            matrix,
            typing,
            ref_enrichment,
            ref_labels,
            category_map,
            sizes=size_list,
            n_reps=n_reps,
            benchmark=benchmark,
            n_init=sweep_n_init,
            n_components=n_components,
            base_seed=seed,
            checkpoint_dir=ctx.run_dir,
            workers=workers,
        )
        cache.save_frame(result.per_fit, ctx.path("per_fit.parquet"))
        cache.save_frame(result.summary, ctx.path("summary.parquet"))
        checkpoint.clear_checkpoints(ctx.run_dir)
        ctx.metrics = {
            "n_min": result.n_min,
            "floor": result.floor,
            "floor_ci90": list(result.floor_ci) if result.floor_ci else None,
            "benchmark": benchmark,
            "sizes": size_list,
        }
        ctx.log.info(
            "recovery floor (isotonic): %s, 90%% CI %s; smallest clearing size: %s",
            result.floor,
            result.floor_ci,
            result.n_min,
        )
    typer.echo(f"nmin {dataset}/{version}: run {cache.short_hash(ctx.run_hash)}")
    typer.echo(
        f"  recovery floor (isotonic, r>={benchmark}): {result.floor}  90% CI {result.floor_ci}"
    )
    typer.echo(f"  smallest clearing size (cleared.min): {result.n_min}")


@app.command()
def replicate(
    version: str = _VERSION,
    ssc_version: str = typer.Option("15.3", help="SSC dataset version to project onto."),
    n_components: int = typer.Option(config.DEFAULT_N_COMPONENTS, help="Number of latent classes."),
    n_init: int = typer.Option(config.DEFAULT_N_INIT, help="Random restarts for the SPARK fit."),
    n_permutations: int = typer.Option(200, help="SSC label permutations for the null (0 skips)."),
    seed: int = typer.Option(0, help="Random seed."),
    as_of: str | None = _AS_OF,
    sample_n: int | None = _SAMPLE_N,
    sample_seed: int = _SAMPLE_SEED,
    force: bool = _FORCE,
) -> None:
    """Project the SPARK model onto the SSC and correlate the category profiles."""
    from analysis import replicate as replicate_mod
    from analysis.cohort import build_matrix, get_cohort
    from analysis.cohort.schema import load_feature_list

    root = find_repo_root()
    # The cutoff and subsample apply to the SPARK training cohort only; the SSC is projected
    # onto in full, since the records cutoff is a SPARK-side timing field.
    spark_hash, _ = _run_cohort(
        root, "spark", version, force=force, as_of=as_of, sample_n=sample_n, sample_seed=sample_seed
    )
    # Build the SSC cohort before the run hash so its content enters the hash, as the SPARK
    # cohort does through its run hash. The cohort-stage hash covers the input files and
    # settings but not the harmonisation code, so an SSC-side code change (the milestone
    # parser, a rename map, the milestone priors) leaves it unmoved; digesting the integrated
    # frame makes any such change invalidate the cache without --force. The SSC build then runs
    # on a cache hit too, which is the cost of hashing what was previously computed inline.
    ssc_integrated = get_cohort("ssc", ssc_version, root).integrate()
    params = {
        "spark_cohort": spark_hash,
        "ssc_version": ssc_version,
        "ssc_cohort": cache.frame_digest(ssc_integrated),
        "category_map": cache.file_digest(config.author_category_map(root)),
        "n_components": n_components,
        "n_init": n_init,
        "n_permutations": n_permutations,
        "seed": seed,
    }
    with run_context("replicate", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            metrics = (cache.read_manifest(ctx.run_dir) or {}).get("metrics", {})
            typer.echo(f"replicate: cache hit {cache.short_hash(ctx.run_hash)}; {metrics}")
            return
        spark_matrix, typing = _load_cohort_matrix(root, spark_hash, "spark", version)
        feature_names = load_feature_list(config.author_feature_list(root))
        feature_set = set(feature_names)
        ssc_available = [
            c for c in ssc_integrated.columns if c in feature_set and c not in config.COVARIATES
        ]
        ssc_matrix = build_matrix(ssc_integrated, ssc_available, "ssc", ssc_version)
        category_map = features.load_category_map(config.author_category_map(root))

        ctx.log.info(
            "replication: %d SPARK x %d SSC probands, %d SSC features available",
            spark_matrix.n_probands,
            ssc_matrix.n_probands,
            len(ssc_available),
        )
        result = replicate_mod.run_replication(
            spark_matrix,
            ssc_matrix,
            typing,
            category_map,
            n_components=n_components,
            n_init=n_init,
            n_permutations=n_permutations,
            seed=seed,
        )
        cache.save_frame(result.spark_signature, ctx.path("spark_signature.parquet"))
        cache.save_frame(result.ssc_signature, ctx.path("ssc_signature.parquet"))
        cache.save_json(result.metrics, ctx.path("replication.json"))
        ctx.metrics = result.metrics
        ctx.log.info(
            "replication overall r=%s p=%s", result.metrics["overall_correlation"], result.p_value
        )
    typer.echo(
        f"replicate spark/{version} -> ssc/{ssc_version}: run {cache.short_hash(ctx.run_hash)}"
    )
    typer.echo(
        f"  shared features={ctx.metrics['n_shared_features']} "
        f"n_ssc={ctx.metrics['n_ssc']} overall r={ctx.metrics['overall_correlation']} "
        f"p={ctx.metrics['p_value']}"
    )


@app.command(name="strata-describe")
def strata_describe(
    dataset: str = _DATASET,
    version: str = _VERSION,
    quantile_bins: int = typer.Option(4, help="Bins for the quantile sensitivity policy."),
    force: bool = _FORCE,
) -> None:
    """Characterise the stratification axes and test the candidate binning policies.

    A phase-3 (pre-registration) stage. It builds age at diagnosis, the derived diagnostic
    era, and the measurement-to-diagnosis lag for the modelling cohort, then evaluates each
    binning policy on both axes against the acceptance requirements
    (:mod:`analysis.requirements`): the substantive fixed bands, an equal-frequency quantile
    split, and the max-equal split that is the chosen primary scheme. No model is fitted; the
    output feeds the frozen bin choice (plan sections 7 and 12).
    """
    import pandas as pd

    from analysis import requirements, strata_data
    from analysis import strata as strata_mod

    root = find_repo_root()
    cohort_hash, _ = _run_cohort(root, dataset, version, force=force)
    matrix, _typing = _load_cohort_matrix(root, cohort_hash, dataset, version)

    thresholds = requirements.DEFAULT_THRESHOLDS
    quantile = strata_mod.QuantileBins(q=quantile_bins)
    max_equal = strata_mod.MaxEqualBins(min_bin_size=thresholds.min_bin_size)
    policies = {
        "age_at_diagnosis": {
            "bands": strata_mod.PROVISIONAL_AGE_BANDS,
            "quantile": quantile,
            "max_equal": max_equal,
        },
        "era": {
            "bands": strata_mod.PROVISIONAL_ERA_BANDS,
            "quantile": quantile,
            "max_equal": max_equal,
        },
    }
    params = {
        "cohort": cohort_hash,
        "policies": {a: {n: p.spec() for n, p in d.items()} for a, d in policies.items()},
        "thresholds": asdict(thresholds),
    }
    with run_context("strata-describe", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            metrics = (cache.read_manifest(ctx.run_dir) or {}).get("metrics", {})
            typer.echo(f"strata-describe: cache hit {cache.short_hash(ctx.run_hash)}; {metrics}")
            return

        data = strata_data.build_strata_data(
            root,
            version,
            matrix.features.index,
            matrix.covariates["age_at_eval_years"],
            matrix.covariates["sex"],
        )
        axis_series = {
            "age_at_diagnosis": data.axes["age_at_diagnosis_years"],
            "era": data.axes["diagnosis_year"],
        }

        req_rows: list[dict[str, object]] = []
        count_rows: list[dict[str, object]] = []
        demo_frames: list[pd.DataFrame] = []
        summary: dict[str, dict[str, object]] = {}
        for axis, axis_policies in policies.items():
            for pname, policy in axis_policies.items():
                report = requirements.evaluate_policy(
                    policy,
                    axis_series[axis],
                    lag=data.lag,
                    covariates=data.demographics,
                    thresholds=thresholds,
                )
                for r in report.results:
                    req_rows.append(
                        {
                            "axis": axis,
                            "policy": pname,
                            "key": r.key,
                            "tier": r.tier,
                            "status": r.status,
                            "observed": r.observed,
                            "threshold": r.threshold,
                            "detail": r.detail,
                        }
                    )
                for bin_label, n in report.counts.items():
                    count_rows.append({"axis": axis, "policy": pname, "bin": bin_label, "n": n})
                if report.demographics is not None:
                    demo = report.demographics.reset_index(names="covariate")
                    demo.insert(0, "policy", pname)
                    demo.insert(0, "axis", axis)
                    demo_frames.append(demo)
                summary[f"{axis}/{pname}"] = {
                    "eligible": report.eligible,
                    "flags": report.flags,
                    "n_assigned": report.n_assigned,
                    "counts": report.counts,
                }

        cache.save_frame(pd.DataFrame(req_rows), ctx.path("requirements.parquet"))
        cache.save_frame(pd.DataFrame(count_rows), ctx.path("bin_counts.parquet"))
        cache.save_frame(
            pd.concat(demo_frames, ignore_index=True), ctx.path("demographics.parquet")
        )
        cache.save_frame(data.axes.reset_index(), ctx.path("axes.parquet"))
        cache.save_frame(data.instrument_years.reset_index(), ctx.path("instrument_years.parquet"))
        cache.save_json(data.diagnostics, ctx.path("distributions.json"))
        cache.save_json(summary, ctx.path("policy_summary.json"))

        ctx.metrics = {
            "eligibility": {k: v["eligible"] for k, v in summary.items()},
            "flags": {k: v["flags"] for k, v in summary.items()},
            "contemporaneity": data.diagnostics["contemporaneity"],
        }
        ctx.log.info("strata-describe eligibility: %s", ctx.metrics["eligibility"])
    typer.echo(f"strata-describe {dataset}/{version}: run {cache.short_hash(ctx.run_hash)}")
    for key, eligible in ctx.metrics["eligibility"].items():
        verdict = "eligible" if eligible else "INELIGIBLE"
        typer.echo(f"  {key}: {verdict}  flags={ctx.metrics['flags'][key]}")


@app.command()
def strata(
    dataset: str = _DATASET,
    version: str = _VERSION,
    min_bin_size: int = typer.Option(
        1000, help="Floor every bin must clear; sets the MaxEqualBins resolution (plan 12a)."
    ),
    force: bool = _FORCE,
) -> None:
    """Assign each proband to an age-at-diagnosis and a diagnostic-era stratum.

    A phase-4 stage. It builds the two stratifying axes for the modelling cohort and assigns
    every proband to a stratum on each axis with the frozen primary policy,
    ``MaxEqualBins(min_bin_size)`` (plan section 12a): the finest equal-frequency split that
    keeps every bin at or above the floor. The per-proband assignments and the realised bin
    edges are cached for the ``stratify`` stage to consume.
    """
    import pandas as pd

    from analysis import strata as strata_mod
    from analysis import strata_data

    root = find_repo_root()
    cohort_hash, _ = _run_cohort(root, dataset, version, force=force)
    matrix, _typing = _load_cohort_matrix(root, cohort_hash, dataset, version)

    policy = strata_mod.MaxEqualBins(min_bin_size=min_bin_size)
    params = {"cohort": cohort_hash, "policy": policy.spec(), "axes": ["age_at_diagnosis", "era"]}
    with run_context("strata", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            cached = (cache.read_manifest(ctx.run_dir) or {}).get("metrics", {})
            typer.echo(f"strata: cache hit {cache.short_hash(ctx.run_hash)}; {cached}")
            return

        data = strata_data.build_strata_data(
            root,
            version,
            matrix.features.index,
            matrix.covariates["age_at_eval_years"],
            matrix.covariates["sex"],
        )
        axis_series = {
            "age_at_diagnosis": data.axes["age_at_diagnosis_years"],
            "era": data.axes["diagnosis_year"],
        }

        assignments: dict[str, object] = {}
        strata_spec: dict[str, object] = {}
        metrics: dict[str, dict[str, object]] = {}
        for axis, series in axis_series.items():
            assignment = policy.assign(series)
            assignments[axis] = assignment.codes
            strata_spec[axis] = {
                "labels": assignment.labels,
                "edges": assignment.edges,
                "counts": assignment.counts,
                "n_missing": assignment.n_missing,
                "spec": assignment.spec,
            }
            metrics[axis] = {"n_bins": len(assignment.labels), "counts": assignment.counts}

        cache.save_frame(pd.DataFrame(assignments).reset_index(), ctx.path("assignments.parquet"))
        cache.save_json(strata_spec, ctx.path("strata.json"))
        cache.save_json(data.diagnostics, ctx.path("axis_distributions.json"))

        ctx.metrics = metrics
        ctx.log.info("strata bins: %s", {a: m["n_bins"] for a, m in metrics.items()})
    typer.echo(f"strata {dataset}/{version}: run {cache.short_hash(ctx.run_hash)}")
    for axis, m in ctx.metrics.items():
        typer.echo(f"  {axis}: {m['n_bins']} bins  {m['counts']}")


def _fit_and_save_stratum(
    run_dir: Path,
    axis: str,
    label: str,
    features: pd.DataFrame,
    covariates: pd.DataFrame,
    dataset: str,
    version: str,
    typing: Typing,
    n_init: int,
    random_state: int,
) -> dict:
    """Fit one stratum's GFMM, save its artefacts, and return the checkpoint record.

    A top-level function so it pickles for a process pool (mirrors
    :func:`analysis.drift.summarise_pseudo_stratum`). Runs inside a worker process, so its
    own :func:`analysis.profiling.measure` call reads that process's CPU and memory, giving
    a true per-fit reading even when every stratum fits at once.
    """
    from analysis import profiling
    from analysis.cohort import CohortMatrix

    sub = CohortMatrix(features, covariates, dataset, version)
    with profiling.measure() as unit:
        fit = model.fit_gfmm(
            sub, typing, n_init=n_init, random_state=random_state, progress_bar=0, verbose=0
        )
        centroids = model.class_centroids(fit.measurement_data, fit.labels)
        cache.save_model(fit.model, run_dir / f"fit_{axis}_{label}.joblib")
        cache.save_frame(fit.labels.reset_index(), run_dir / f"labels_{axis}_{label}.parquet")
        cache.save_frame(centroids.reset_index(), run_dir / f"centroids_{axis}_{label}.parquet")
        unit.output_bytes = sum(
            profiling.path_bytes(run_dir / f"{kind}_{axis}_{label}.{ext}")
            for kind, ext in (("fit", "joblib"), ("labels", "parquet"), ("centroids", "parquet"))
        )
    metrics = unit.metrics
    assert metrics is not None  # measure() always sets metrics on exit
    return {
        "stratum": label,
        "n": int(len(features)),
        "seed": random_state,
        "fit": fit.metrics,
        "resources": metrics.to_dict(),
    }


@app.command()
def stratify(
    axis: str = typer.Option(
        "age_at_diagnosis",
        help="Axis: age_at_diagnosis or era (SPARK), or cognitive_impairment or iq (both cohorts).",
    ),
    dataset: str = _DATASET,
    version: str = _VERSION,
    n_init: int = typer.Option(config.DEFAULT_N_INIT, help="StepMix restarts per stratum fit."),
    seed: int = typer.Option(
        config.DEFAULT_STRATIFY_SEED, help="Base seed; each stratum's seed is derived from it."
    ),
    limit: int = typer.Option(
        0, help="Fit only the first N strata (0 = all), for a fast pilot or debugging."
    ),
    min_bin_size: int = typer.Option(1000, help="Strata floor; must match the strata stage."),
    workers: int = typer.Option(0, help="Parallel fit workers (0 = logical cores minus one)."),
    shared_with: str | None = typer.Option(
        None,
        "--shared-with",
        help="Fit on the features shared with this cohort (e.g. spark), for a cohort that "
        "provides only a subset of the 238 features (the SSC). Enables SSC-native stratify.",
    ),
    force: bool = _FORCE,
) -> None:
    """Re-estimate the GFMM independently within each stratum of an axis.

    A phase-4 stage. For the chosen axis it assigns the modelling cohort to the frozen
    ``MaxEqualBins(min_bin_size)`` strata, then fits the four-class covariate GFMM within each
    stratum on the same features, typing, and hyperparameters as the reference. Each stratum's
    fitted model, hard labels, and class-by-feature centroids are stored so the drift analysis
    is a pure consumer that never refits. The strata are independent fits, so they run
    concurrently over a :class:`~concurrent.futures.ProcessPoolExecutor` (measured: a StepMix
    fit is single-core, so throughput comes from running many at once, not from one fit using
    many cores), each pinned to a single BLAS thread
    (:func:`analysis.profiling.single_threaded_blas`) so concurrent workers do not
    oversubscribe the machine. Every fit is measured (:mod:`analysis.profiling`) and streamed
    to a resumable checkpoint, so an interrupt continues from the first unfitted stratum.
    ``--limit`` fits only the first few strata, to debug the instrumented pipeline or pilot it
    before the full run. ``--shared-with`` fits on the cross-cohort shared feature set (plan
    section 8), so the SSC, which provides only a subset of the 238 features, can be
    re-estimated within its cognitive strata and compared against the reference cohort.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import pandas as pd

    from analysis import checkpoint, profiling
    from analysis.cohort import get_cohort
    from analysis.progress import task_bar

    root = find_repo_root()
    if shared_with:
        matrix, typing, cohort_hash = _build_shared_cohort(
            root, dataset, version, shared_with, force=force
        )
    else:
        cohort_hash, _ = _run_cohort(root, dataset, version, force=force)
        matrix, typing = _load_cohort_matrix(root, cohort_hash, dataset, version)

    resolved = get_cohort(dataset, version, root).axis(
        axis, matrix.features.index, matrix.covariates, min_bin_size
    )
    if resolved is None:
        raise typer.BadParameter(f"cohort {dataset!r} does not provide axis {axis!r}")
    axis_values, policy = resolved
    assignment = policy.assign(axis_values)
    order = assignment.labels

    # The shared-feature cohort hash (``_build_shared_cohort``) already differs from the full
    # cohort hash, so the shared and full runs cache separately without a redundant key here;
    # the age/era run hashes are therefore unchanged from before this option existed.
    params = {
        "cohort": cohort_hash,
        "axis": axis,
        "policy": policy.spec(),
        "n_components": config.DEFAULT_N_COMPONENTS,
        "n_init": n_init,
        "seed": seed,
        "limit": limit,
    }

    def unit_from_dict(d: dict) -> profiling.UnitMetrics:
        return profiling.UnitMetrics(
            wall_s=d["wall_s"],
            cpu_s=d["cpu_s"],
            peak_rss_bytes=d["peak_rss_bytes"],
            start_rss_bytes=d["start_rss_bytes"],
            n_samples=d["n_samples"],
            output_bytes=d.get("output_bytes"),
        )

    with run_context("stratify", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            cached = (cache.read_manifest(ctx.run_dir) or {}).get("metrics", {})
            typer.echo(f"stratify: cache hit {cache.short_hash(ctx.run_hash)}; {cached}")
            return
        if force:
            checkpoint.clear_checkpoints(ctx.run_dir)

        todo = order if limit <= 0 else order[:limit]
        log = checkpoint.CheckpointLog(ctx.path(f"{axis}.checkpoint.jsonl"))
        records: list[dict] = list(log.load())
        done = {r["stratum"] for r in records}

        n_workers = workers or max(1, (os.cpu_count() or 2) - 1)
        pending = [
            (i, label) for i, label in enumerate(order) if label in todo and label not in done
        ]

        with (
            task_bar(len(todo), f"stratify {axis}") as bar,
            profiling.single_threaded_blas(),
            ProcessPoolExecutor(max_workers=n_workers) as pool,
        ):
            bar.update(sum(1 for label in todo if label in done))
            futures = {}
            for i, label in pending:
                members = assignment.codes[assignment.codes == label].index
                future = pool.submit(
                    _fit_and_save_stratum,
                    ctx.run_dir,
                    axis,
                    label,
                    matrix.features.loc[members],
                    matrix.covariates.loc[members],
                    dataset,
                    version,
                    typing,
                    n_init,
                    seed + i,
                )
                futures[future] = label
            for future in as_completed(futures):
                record = future.result()
                log.append(record)
                records.append(record)
                resources = record["resources"]
                props = record["fit"].get("class_proportions")
                smallest = (
                    min(props.values()) if isinstance(props, dict) and props else float("nan")
                )
                bar.set_postfix(
                    {
                        "stratum": record["stratum"],
                        "n": record["n"],
                        "wall": f"{resources['wall_s']:.0f}s",
                        "rss": f"{resources['peak_rss_bytes'] / 1024**2:.0f}M",
                        "min_class": f"{smallest:.2f}",
                    }
                )
                bar.update(1)

        summary = pd.DataFrame(
            [
                {
                    "stratum": r["stratum"],
                    "n": r["n"],
                    "avg_log_likelihood": r["fit"].get("avg_log_likelihood"),
                    "bic": r["fit"].get("bic"),
                    "smallest_class": min(
                        r["fit"].get("class_proportions", {1: float("nan")}).values()
                    ),
                    "wall_s": r["resources"]["wall_s"],
                    "peak_rss_bytes": r["resources"]["peak_rss_bytes"],
                }
                for r in records
            ]
        )
        cache.save_frame(summary, ctx.path(f"summary_{axis}.parquet"))
        units = [unit_from_dict(r["resources"]) for r in records]
        res = profiling.summarise(units)
        ctx.metrics = {"axis": axis, "n_strata_fitted": len(records), "resources": res}
        peak_mib = max((u.peak_rss_bytes for u in units), default=0) / 1024**2
        n_fitted = len(records)
        checkpoint.clear_checkpoints(ctx.run_dir)
    typer.echo(f"stratify {axis} {dataset}/{version}: run {cache.short_hash(ctx.run_hash)}")
    typer.echo(
        f"  fitted {n_fitted} strata; median {res.get('wall_s_median')} s/fit, "
        f"peak RSS {peak_mib:.0f} MiB"
    )


def _run_drift_null(
    root: Path,
    null_params: dict,
    *,
    features,
    covariates,
    typing,
    reference_labels,
    assigned,
    sizes: list[int],
    n_labels: int,
    n_init: int,
    n_permutations: int,
    seed: int,
    workers: int,
    force: bool,
):
    """Fit the permutation null and store each pseudo-stratum's summary.

    The heavy, method-independent half of the drift stage: it refits within size-shuffled
    pseudo-strata and stores the centroids and reference contingency (not a drift value), so
    the alignment and distance can be chosen afterwards without re-fitting. Keyed only by the
    fitting parameters, concurrent across the single-core fits (each pinned to a single BLAS
    thread, :func:`analysis.profiling.single_threaded_blas`, so concurrent workers do not
    oversubscribe the machine), and resumable through the same append-only store it writes.

    Work is dispatched from one flat, permutation-major queue kept at ``n_workers`` futures in
    flight, rather than one :func:`~concurrent.futures.as_completed` barrier per permutation: on
    a machine with heterogeneous cores (a laptop's efficiency cores run a fit markedly slower
    than its performance cores), a per-permutation barrier leaves the fast workers idle every
    round, waiting on whichever fit landed on a slow core. Returns the run directory holding
    ``null_summaries``.
    """
    import os
    from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait

    from analysis import checkpoint, profiling
    from analysis import drift as drift_mod
    from analysis.progress import task_bar

    n_workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with run_context("drift-null", null_params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            return ctx.run_dir
        if force:
            checkpoint.clear_checkpoints(ctx.run_dir)
        store = checkpoint.CheckpointLog(ctx.path("null_summaries.checkpoint.jsonl"))
        done = {(r["perm"], r["s_idx"]) for r in store.load()}
        queue = iter(
            (perm, s)
            for perm in range(n_permutations)
            for s in range(n_labels)
            if (perm, s) not in done
        )
        # Memoised per permutation so its several strata share one shuffle. Never evicted, so
        # a full run caches all n_permutations entries, but each is a handful of index arrays
        # over the cohort (about 90 KiB; under 100 MiB even at the frozen N_perm=1000), well
        # under what the concurrent worker processes themselves hold.
        chunks_cache: dict[int, list] = {}

        def chunks_for(perm: int) -> list:
            if perm not in chunks_cache:
                chunks_cache[perm] = drift_mod.null_partition(assigned, sizes, seed=perm)
            return chunks_cache[perm]

        with (
            task_bar(n_permutations * n_labels, "drift null") as bar,
            profiling.single_threaded_blas(),
            ProcessPoolExecutor(max_workers=n_workers) as pool,
        ):
            bar.update(len(done))

            def submit_next() -> tuple[Future, tuple[int, int]] | None:
                perm_s = next(queue, None)
                if perm_s is None:
                    return None
                perm, s = perm_s
                chunk = chunks_for(perm)[s]
                future = pool.submit(
                    drift_mod.summarise_pseudo_stratum,
                    features.loc[chunk],
                    covariates.loc[chunk],
                    typing,
                    reference_labels,
                    n_init,
                    seed + perm * n_labels + s,
                )
                return future, perm_s

            in_flight: dict[Future, tuple[int, int]] = {}
            for _ in range(n_workers):
                submitted = submit_next()
                if submitted is None:
                    break
                future, key = submitted
                in_flight[future] = key
            while in_flight:
                finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in finished:
                    perm, s = in_flight.pop(future)
                    result = future.result()
                    if result is None:
                        # A degenerate refit (singular covariate GLM); recorded so the perm is
                        # marked done and resumable, dropped from the null at read time.
                        store.append({"perm": int(perm), "s_idx": int(s), "degenerate": True})
                    else:
                        store.append(drift_mod.serialise_summary(result, perm, s))
                    bar.update(1)
                    bar.set_postfix({"perm": perm})
                    submitted = submit_next()
                    if submitted is not None:
                        next_future, key = submitted
                        in_flight[next_future] = key
        ctx.metrics = {"n_permutations": n_permutations, "n_units": n_permutations * n_labels}
    return ctx.run_dir


@app.command()
def drift(
    axis: str = typer.Option(
        "age_at_diagnosis",
        help="Axis: age_at_diagnosis or era (SPARK), or cognitive_impairment or iq (both cohorts).",
    ),
    dataset: str = _DATASET,
    version: str = _VERSION,
    n_init: int = typer.Option(config.DEFAULT_N_INIT, help="StepMix restarts per fit."),
    n_permutations: int = typer.Option(
        config.DEFAULT_N_PERMUTATIONS,
        "--n-permutations",
        help="Permutations for the matched-size null. The frozen confirmatory value is 1000; "
        "pass 1 to smoke-test the pipeline or 100 for an overnight pilot.",
    ),
    alignment: str = typer.Option(
        "membership", help="Alignment method: membership (default) or centroid."
    ),
    distance: str = typer.Option(
        "mahalanobis",
        help="Distance method: mahalanobis (default), euclidean, mean-abs, or jsd.",
    ),
    seed: int = typer.Option(config.DEFAULT_STRATIFY_SEED, help="Base seed."),
    workers: int = typer.Option(0, help="Parallel fit workers (0 = logical cores minus one)."),
    min_bin_size: int = typer.Option(1000, help="Strata floor; must match strata/stratify."),
    reference_scheme: str = typer.Option(
        "pooled",
        "--reference-scheme",
        help="What each stratum's drift is measured against: 'pooled' (default, the frozen "
        "primary, calibrated against the permutation null) or 'pairwise' (each stratum against a "
        "neighbouring stratum along the axis; observed-only over the cached fits, no refit, the "
        "union-split null is a later stage).",
    ),
    pairwise_mode: str = typer.Option(
        "adjacent",
        "--pairwise-mode",
        help="For --reference-scheme pairwise: 'adjacent' (default, each stratum against its "
        "successor) or 'all-pairs'.",
    ),
    force: bool = _FORCE,
) -> None:
    """Align stratum classes to the reference and measure drift against the permutation null.

    A phase-4 stage (the confirmatory core, frozen in plan section 12a), split so the analysis
    is decoupled from the fitting. The heavy half (the ``drift-null`` sub-stage) refits within
    size-shuffled pseudo-strata and stores each fit's centroids and reference contingency,
    keyed only by the fitting parameters. This cheap half then aligns and measures: it reads
    the observed per-stratum fits (the stratify stage) and the stored null fits, applies the
    chosen ``--alignment`` and ``--distance`` over them, and reads each aligned class's drift
    against its size-matched null (beyond the 95th percentile, Benjamini-Hochberg controlled).
    Because the fits are stored, changing the alignment or distance re-measures without
    re-fitting. The alignment confidence (per-class Jaccard, overall adjusted Rand index) is
    reported alongside, so a large shift with low overlap reads as reorganisation, not drift.
    """
    from collections import defaultdict

    import pandas as pd

    from analysis import checkpoint, model
    from analysis import drift as drift_mod
    from analysis import reference_scheme as reference_scheme_mod
    from analysis.cohort import CohortMatrix, get_cohort
    from analysis.paths import run_dir

    if alignment not in drift_mod.ALIGNMENTS:
        raise typer.BadParameter(f"alignment must be one of {sorted(drift_mod.ALIGNMENTS)}")
    if distance not in drift_mod.DISTANCES:
        raise typer.BadParameter(f"distance must be one of {sorted(drift_mod.DISTANCES)}")
    if reference_scheme not in reference_scheme_mod.REFERENCE_SCHEMES:
        raise typer.BadParameter(
            f"reference-scheme must be one of {sorted(reference_scheme_mod.REFERENCE_SCHEMES)}"
        )
    if reference_scheme == "pairwise" and pairwise_mode not in ("adjacent", "all-pairs"):
        raise typer.BadParameter("pairwise-mode must be 'adjacent' or 'all-pairs'")

    def labels_series(frame: pd.DataFrame) -> pd.Series:
        others = [c for c in frame.columns if c != "class"]
        return frame.set_index(others[0])["class"] if others else frame["class"]

    root = find_repo_root()
    cohort_hash, _ = _run_cohort(root, dataset, version, force=force)
    matrix, typing = _load_cohort_matrix(root, cohort_hash, dataset, version)

    ref_fit_hash = cache.compute_hash(
        _fit_params(cohort_hash, config.DEFAULT_N_COMPONENTS, config.DEFAULT_N_INIT, seed)
    )
    ref_dir = run_dir(root, "fit", cache.short_hash(ref_fit_hash))
    if not (ref_dir / "labels.parquet").is_file():
        raise typer.BadParameter(f"no reference fit at {ref_dir}; run `analysis fit`")
    ref_labels = labels_series(cache.load_frame(ref_dir / "labels.parquet"))
    measurement_data, _descriptor, _covariates = model.prepare_inputs(matrix, typing)
    # The reference carries the per-class centroids and dispersions plus the Ledoit-Wolf-shrunk
    # within-class precision, recomputed from the pooled fit's measurement data and labels (the
    # recomputed centroids match the stored fit centroids; this also gives the precision and
    # dispersions the Mahalanobis and Jensen-Shannon distances need).
    reference = drift_mod.build_reference(measurement_data, ref_labels)
    aligner = drift_mod.ALIGNMENTS[alignment]
    distancer = drift_mod.DISTANCES[distance]

    resolved = get_cohort(dataset, version, root).axis(
        axis, matrix.features.index, matrix.covariates, min_bin_size
    )
    if resolved is None:
        raise typer.BadParameter(f"cohort {dataset!r} does not provide axis {axis!r}")
    axis_values, policy = resolved
    assignment = policy.assign(axis_values)
    labels = assignment.labels
    sizes = [int((assignment.codes == label).sum()) for label in labels]
    assigned = assignment.codes[assignment.codes.notna()].index

    stratify_params = {
        "cohort": cohort_hash,
        "axis": axis,
        "policy": policy.spec(),
        "n_components": config.DEFAULT_N_COMPONENTS,
        "n_init": n_init,
        "seed": seed,
        "limit": 0,
    }
    stratify_dir = run_dir(root, "stratify", cache.short_hash(cache.compute_hash(stratify_params)))

    def stratum_query(
        label: object,
    ) -> tuple[reference_scheme_mod.QueryFit, drift_mod.ReferenceModel]:
        """Load one cached stratum fit as a query summary and a promoted reference (no refit)."""
        lpath = stratify_dir / f"labels_{axis}_{label}.parquet"
        if not lpath.is_file():
            raise typer.BadParameter(
                f"no stratify fit for {axis}/{label}; run `analysis stratify --axis {axis}`"
            )
        members = assignment.codes[assignment.codes == label].index
        sub = CohortMatrix(
            matrix.features.loc[members], matrix.covariates.loc[members], dataset, version
        )
        stratum_md = model.prepare_inputs(sub, typing)[0]
        stratum_labels = labels_series(cache.load_frame(lpath)).reindex(stratum_md.index)
        summary = drift_mod.summarise(stratum_md, stratum_labels, ref_labels)
        promoted = drift_mod.build_reference(stratum_md, stratum_labels)
        position = float(axis_values.loc[members].median())
        return reference_scheme_mod.QueryFit(str(label), position, summary), promoted

    if reference_scheme == "pairwise":
        # Observed-only trajectory of change along the axis, reusing the cached stratify fits: each
        # stratum is aligned by centroid (the strata are disjoint) to a neighbour promoted to a
        # reference, so no refit and no permutation null run here. The union-split null is a later
        # stage; drift is still scaled by the pooled between-class separation for a common scale.
        if alignment != "centroid":
            typer.echo(
                "note: pairwise uses centroid alignment (disjoint strata); --alignment ignored"
            )
        scheme = reference_scheme_mod.PairwiseReference(mode=pairwise_mode)
        separation = drift_mod.class_separation(reference, distancer)
        measure_params = {
            "stratify": cache.compute_hash(stratify_params),
            "axis": axis,
            "distance": distance,
            "reference_scheme": "pairwise",
            "mode": pairwise_mode,
        }
        with run_context("drift", measure_params, root=root, force=force) as ctx:
            if ctx.cache_hit:
                cached = (cache.read_manifest(ctx.run_dir) or {}).get("metrics", {})
                typer.echo(f"drift pairwise: cache hit {cache.short_hash(ctx.run_hash)}; {cached}")
                return
            queries: list[reference_scheme_mod.QueryFit] = []
            promoted_refs: dict[str, drift_mod.ReferenceModel] = {}
            for label in labels:
                query, promoted = stratum_query(label)
                queries.append(query)
                promoted_refs[str(label)] = promoted
            resolver = reference_scheme_mod.MappingResolver(reference, promoted_refs)
            comparisons = reference_scheme_mod.resolve_comparisons(scheme, queries, resolver)
            measured = reference_scheme_mod.measure(comparisons, distance)
            rows: list[dict] = []
            for comparison in comparisons:
                result = measured[comparison.query_label]
                for ref_class, value in result.distances.items():
                    rows.append(
                        {
                            "query_stratum": comparison.query_label,
                            "reference_stratum": comparison.reference_label,
                            "position": comparison.position,
                            "ref_class": int(ref_class),
                            "drift": float(value),
                            "drift_vs_separation": (
                                float(value) / separation if separation else float("nan")
                            ),
                            "centroid_quality": float(
                                result.alignment.quality.get(int(ref_class), float("nan"))
                            ),
                            "overall_quality": float(result.alignment.overall),
                        }
                    )
            trajectory = pd.DataFrame(rows)
            cache.save_frame(trajectory, ctx.path(f"pairwise_{axis}.parquet"))
            ctx.metrics = {
                "axis": axis,
                "reference_scheme": "pairwise",
                "mode": pairwise_mode,
                "distance": distance,
                "n_pairs": len(comparisons),
                "n_tests": len(trajectory),
                "class_separation": separation,
                "mean_drift_vs_separation": (
                    float(trajectory["drift_vs_separation"].mean())
                    if len(trajectory)
                    else float("nan")
                ),
            }
        typer.echo(
            f"drift {axis} pairwise/{pairwise_mode} ({distance}): "
            f"run {cache.short_hash(ctx.run_hash)}"
        )
        typer.echo(
            f"  {ctx.metrics['n_pairs']} neighbour comparisons, mean drift/separation "
            f"{ctx.metrics['mean_drift_vs_separation']:.3f}; observed-only (the union-split null "
            f"is a later stage)"
        )
        return

    null_params = {
        "cohort": cohort_hash,
        "axis": axis,
        "ref_fit": ref_fit_hash,
        "policy": policy.spec(),
        "n_init": n_init,
        "n_permutations": n_permutations,
        "seed": seed,
    }
    null_dir = _run_drift_null(
        root,
        null_params,
        features=matrix.features,
        covariates=matrix.covariates,
        typing=typing,
        reference_labels=ref_labels,
        assigned=assigned,
        sizes=sizes,
        n_labels=len(labels),
        n_init=n_init,
        n_permutations=n_permutations,
        seed=seed,
        workers=workers,
        force=force,
    )

    measure_params = {
        "null": cache.compute_hash(null_params),
        "stratify": cache.compute_hash(stratify_params),
        "axis": axis,
        "alignment": alignment,
        "distance": distance,
    }
    with run_context("drift", measure_params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            cached = (cache.read_manifest(ctx.run_dir) or {}).get("metrics", {})
            typer.echo(f"drift: cache hit {cache.short_hash(ctx.run_hash)}; {cached}")
            return

        observed: dict[str, drift_mod.DriftResult] = {}
        for label in labels:
            lpath = stratify_dir / f"labels_{axis}_{label}.parquet"
            if not lpath.is_file():
                raise typer.BadParameter(
                    f"no stratify fit for {axis}/{label}; run `analysis stratify --axis {axis}`"
                )
            # Rebuild the stratum's measurement data (so the per-class dispersions the
            # distributional distance needs are available) and re-summarise under the stored
            # stratum labels; the recomputed centroids match the stratify fit's.
            members = assignment.codes[assignment.codes == label].index
            sub = CohortMatrix(
                matrix.features.loc[members], matrix.covariates.loc[members], dataset, version
            )
            stratum_md = model.prepare_inputs(sub, typing)[0]
            stratum_labels = labels_series(cache.load_frame(lpath)).reindex(stratum_md.index)
            summary = drift_mod.summarise(stratum_md, stratum_labels, ref_labels)
            observed[label] = drift_mod.compute_drift(summary, reference, aligner, distancer)

        null_store = checkpoint.CheckpointLog(null_dir / "null_summaries.checkpoint.jsonl")
        null_records = null_store.load()
        n_null_degenerate = sum(1 for rec in null_records if rec.get("degenerate"))
        null_drift: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for rec in null_records:
            if rec.get("degenerate"):
                continue
            result = drift_mod.compute_drift(
                drift_mod.deserialise_summary(rec), reference, aligner, distancer
            )
            for ref_class, value in result.distances.items():
                null_drift[int(rec["s_idx"])][int(ref_class)].append(value)

        separation = drift_mod.class_separation(reference, distancer)
        rows: list[dict] = []
        for s_idx, label in enumerate(labels):
            result = observed[label]
            for ref_class, obs in result.distances.items():
                read = drift_mod.read_against_null(obs, null_drift[s_idx][int(ref_class)])
                rows.append(
                    {
                        **read,
                        "stratum": label,
                        "ref_class": int(ref_class),
                        "jaccard": float(
                            result.alignment.quality.get(int(ref_class), float("nan"))
                        ),
                        "ari": float(result.alignment.overall),
                        "drift_vs_separation": obs / separation if separation else float("nan"),
                    }
                )
        decision = pd.DataFrame(rows)
        decision["fdr_significant"] = drift_mod.benjamini_hochberg(
            decision["p_value"].to_numpy(), q=0.05
        )
        decision["reorganised"] = decision["jaccard"] < 0.5
        cache.save_frame(decision, ctx.path(f"decision_{axis}.parquet"))
        n_drift = int((decision["fdr_significant"] & ~decision["reorganised"]).sum())
        n_reorg = int(decision["reorganised"].sum())
        ctx.metrics = {
            "axis": axis,
            "alignment": alignment,
            "distance": distance,
            "n_permutations": n_permutations,
            "class_separation": separation,
            "n_tests": len(decision),
            "n_drift": n_drift,
            "n_reorganised": n_reorg,
            "n_null_degenerate": n_null_degenerate,
        }
    typer.echo(f"drift {axis} ({alignment}/{distance}): run {cache.short_hash(ctx.run_hash)}")
    typer.echo(
        f"  {ctx.metrics['n_drift']}/{ctx.metrics['n_tests']} classes drift beyond the "
        f"N={n_permutations} null (BH q=0.05); {ctx.metrics['n_reorganised']} reorganised "
        f"(low overlap); separation {separation:.3f}"
    )


def _make_order_dispatch(pool, log, done, *, scope_id, descriptor, columns, recipe, base_seed):
    """Build a bootstrap-draw dispatcher backed by a process pool and a resumable checkpoint.

    The dispatcher :func:`analysis.order.sequential_search` calls: given the null parameters and
    a half-open block of draw indices, it returns that block's null likelihood-ratio statistics.
    Draws already in ``done`` (loaded from the checkpoint) are returned without recomputation;
    the rest are submitted to ``pool`` and each result is appended to ``log`` as it lands, so an
    interrupt resumes from the first missing draw. Each draw's seed is deterministic in its
    scope, null class count, and index, so a resumed run reproduces an uninterrupted one.
    """
    from concurrent.futures import as_completed

    from analysis import order as order_mod

    columns_list = [str(c) for c in columns]

    def dispatch(k_null, null_params, n, start, count):
        results: list[float | None] = [None] * count
        futures = {}
        for offset in range(count):
            idx = start + offset
            key = (scope_id, int(k_null), int(idx))
            if key in done:
                results[offset] = done[key]
                continue
            draw_seed = base_seed + scope_id * 10_000_000 + int(k_null) * 100_000 + int(idx)
            future = pool.submit(
                order_mod.bootstrap_lr,
                null_params,
                descriptor,
                columns_list,
                int(n),
                int(k_null),
                n_init=recipe.n_init,
                n_random=recipe.n_random,
                seed=draw_seed,
                jitter=recipe.jitter,
            )
            futures[future] = (offset, key)
        for future in as_completed(futures):
            offset, key = futures[future]
            value = future.result()
            results[offset] = value
            log.append({"scope": key[0], "k_null": key[1], "idx": key[2], "lr": value})
            done[key] = value
        return results

    return dispatch


def _run_order_search(
    measurement_data,
    covariates,
    descriptor,
    *,
    scope_id,
    pool,
    log,
    done,
    recipe,
    schedule,
    anchor,
    cap,
    floor,
    seed,
    cv_n_init,
):
    """Run the elbow corroborator and the anchored BLRT search for one dataset.

    The cross-validated log-likelihood elbow (a corroborator, so its failure is non-fatal) is
    computed over the ``floor`` to ``cap`` grid, then the sequential bootstrap likelihood-ratio
    search runs anchored at ``anchor``. Returns the :class:`~analysis.order.SearchResult`.
    """
    from analysis import order as order_mod

    k_values = list(range(floor, cap + 1))
    try:
        cv_scores = order_mod.cross_validated_log_likelihood(
            measurement_data, covariates, descriptor, k_values, seed=seed, n_init=cv_n_init
        )
    except Exception:  # noqa: BLE001 - the elbow only corroborates, so a failed grid is not fatal
        cv_scores = {}
    dispatch = _make_order_dispatch(
        pool,
        log,
        done,
        scope_id=scope_id,
        descriptor=descriptor,
        columns=measurement_data.columns,
        recipe=recipe,
        base_seed=seed + 500,
    )
    return order_mod.sequential_search(
        measurement_data,
        descriptor,
        dispatch=dispatch,
        recipe=recipe,
        schedule=schedule,
        anchor=anchor,
        cap=cap,
        floor=floor,
        seed=seed,
        cv_scores=cv_scores,
    )


def _order_supported_label(result, cap: int) -> str:
    """Render a supported order, marking the cap as a lower bound (``">=cap"``)."""
    return f">={cap}" if result.capped else str(result.supported_k)


def _order_step_rows(scope: str, result) -> list[dict]:
    """Flatten a search's BLRT steps into per-step records for the steps table."""
    return [
        {
            "scope": scope,
            "k_null": s.k_null,
            "k_alt": s.k_alt,
            "direction": s.direction,
            "observed_lr": s.observed_lr,
            "p_value": s.p_value,
            "b_used": s.b_used,
            "escalated": s.escalated,
            "n_dropped": s.n_dropped,
            "rejected": s.rejected,
        }
        for s in result.steps
    ]


def _order_params(
    cohort_hash: str,
    *,
    recipe,
    schedule,
    anchor: int,
    cap: int,
    floor: int,
    seed: int,
) -> dict:
    """Return the recipe, anchor, caps, and schedule folded into every order run hash."""
    return {
        "recipe": recipe.spec(),
        "schedule": schedule.spec(),
        "anchor": anchor,
        "cap": cap,
        "floor": floor,
        "seed": seed,
    }


def _run_pooled_order(
    root,
    cohort_hash,
    matrix,
    typing,
    *,
    recipe,
    schedule,
    anchor,
    cap,
    floor,
    seed,
    cv_n_init,
    workers,
    force,
):
    """Compute (or load) the pooled cohort's supported order, the K*-reference both axes share.

    Cached as its own stage (``order-pooled``), keyed only by the cohort and the search recipe,
    so the age and era axis runs reuse one pooled search rather than recomputing it. Returns the
    pooled run hash and its result dict (supported order, elbow knee, VLMR, steps).
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    from analysis import checkpoint as checkpoint_mod
    from analysis import order as order_mod
    from analysis import profiling

    params = {
        "cohort": cohort_hash,
        **_order_params(
            cohort_hash,
            recipe=recipe,
            schedule=schedule,
            anchor=anchor,
            cap=cap,
            floor=floor,
            seed=seed,
        ),
    }
    with run_context("order-pooled", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            return ctx.run_hash, cache.load_json(ctx.path("pooled.json"))
        if force:
            checkpoint_mod.clear_checkpoints(ctx.run_dir)
        measurement_data, descriptor = order_mod.measurement_inputs(matrix, typing)
        log = checkpoint_mod.CheckpointLog(ctx.path("draws.checkpoint.jsonl"))
        done = {(r["scope"], r["k_null"], r["idx"]): r["lr"] for r in log.load()}
        n_workers = workers or max(1, (os.cpu_count() or 2) - 1)
        ctx.log.info(
            "pooled order search: n=%d, B up to %d", len(measurement_data), schedule.b_escalate
        )
        with (
            profiling.single_threaded_blas(),
            ProcessPoolExecutor(max_workers=n_workers) as pool,
        ):
            result = _run_order_search(
                measurement_data,
                matrix.covariates,
                descriptor,
                scope_id=0,
                pool=pool,
                log=log,
                done=done,
                recipe=recipe,
                schedule=schedule,
                anchor=anchor,
                cap=cap,
                floor=floor,
                seed=seed,
                cv_n_init=cv_n_init,
            )
        payload = {
            "supported_k": result.supported_k,
            "supported_label": _order_supported_label(result, cap),
            "capped": result.capped,
            "direction": result.direction,
            "elbow_knee": result.elbow_knee,
            "vlmr_p": result.vlmr.get("p", float("nan")),
            "cv_log_likelihood": {str(k): v for k, v in result.cv_log_likelihood.items()},
            "n_dropped": result.n_dropped,
            "steps": _order_step_rows("__pooled__", result),
        }
        cache.save_json(payload, ctx.path("pooled.json"))
        checkpoint_mod.clear_checkpoints(ctx.run_dir)
        ctx.metrics = {
            "supported_k": result.supported_k,
            "supported_label": payload["supported_label"],
            "elbow_knee": result.elbow_knee,
            "n_dropped": result.n_dropped,
        }
    return ctx.run_hash, payload


@app.command()
def order(
    axis: str = typer.Option(
        "age_at_diagnosis", help="Axis: age_at_diagnosis or era (SPARK diagnosis-timing axes)."
    ),
    dataset: str = _DATASET,
    version: str = _VERSION,
    n_init: int = typer.Option(
        config.DEFAULT_ORDER_N_INIT, "--n-init", help="Random restarts for each K-class fit."
    ),
    k_anchor: int = typer.Option(
        config.DEFAULT_ORDER_K_ANCHOR, "--k-anchor", help="Anchor number of classes (four)."
    ),
    k_cap: int = typer.Option(
        config.DEFAULT_ORDER_K_CAP, "--k-cap", help="Upper cap; the search reports '>=cap' there."
    ),
    b_screen: int = typer.Option(
        config.DEFAULT_ORDER_B_SCREEN,
        "--b-screen",
        help="Bootstrap draws every step is screened at.",
    ),
    b_escalate: int = typer.Option(
        config.DEFAULT_ORDER_B_ESCALATE,
        "--b-escalate",
        help="Bootstrap draws a step escalates to when its screen p is below the threshold.",
    ),
    escalate_threshold: float = typer.Option(
        config.DEFAULT_ORDER_ESCALATE_THRESHOLD,
        "--escalate-threshold",
        help="Screen-p threshold that triggers escalation to --b-escalate.",
    ),
    cv_n_init: int = typer.Option(
        1, "--cv-n-init", help="Restarts for the cross-validated elbow corroborator (kept cheap)."
    ),
    seed: int = typer.Option(config.DEFAULT_ORDER_SEED, "--seed", help="Base seed."),
    workers: int = typer.Option(
        0, help="Parallel bootstrap workers (0 = logical cores minus one)."
    ),
    force: bool = _FORCE,
) -> None:
    """Test whether the supported number of latent classes is stable across strata of an axis.

    The ORDER hypothesis (plan section 7, ``hypotheses.md`` ORDER). Per stratum, the supported
    number of classes is found by a warm-started parametric bootstrap likelihood-ratio search
    anchored at four (:mod:`analysis.order`), and read relative to the pooled cohort put through
    the identical procedure, so the shared over-extraction cancels: an order change is a stratum
    whose supported order differs from the pooled cohort's. The strata are the four
    equal-frequency quantile bins of the axis (a fifth class in a thousand-proband bin is not
    estimable, so the fine ``MaxEqualBins`` scheme is deliberately not used here), plus, for the
    era axis, the DSM-IV against DSM-5 (2013) split as one targeted secondary pair. The pooled
    search is cached as its own stage so both axes reuse it. Bootstrap draws run concurrently and
    stream to a resumable checkpoint. A positive order-change claim needs agreement: the BLRT
    order differs from the pooled order, the cross-validated elbow knee moves off it, and the
    adjusted Lo-Mendell-Rubin test agrees.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    from analysis import checkpoint as checkpoint_mod
    from analysis import order as order_mod
    from analysis import profiling
    from analysis import strata as strata_mod
    from analysis.cohort import CohortMatrix, get_cohort

    if axis not in ("age_at_diagnosis", "era"):
        raise typer.BadParameter(
            "order supports the diagnosis-timing axes: age_at_diagnosis or era"
        )

    recipe = order_mod.Recipe(n_init=n_init)
    schedule = order_mod.Schedule(
        b_screen=b_screen,
        b_escalate=b_escalate,
        escalate_threshold=escalate_threshold,
        alpha=config.DEFAULT_ORDER_ALPHA,
    )

    root = find_repo_root()
    cohort_hash, _ = _run_cohort(root, dataset, version, force=force)
    matrix, typing = _load_cohort_matrix(root, cohort_hash, dataset, version)

    pooled_hash, pooled = _run_pooled_order(
        root,
        cohort_hash,
        matrix,
        typing,
        recipe=recipe,
        schedule=schedule,
        anchor=k_anchor,
        cap=k_cap,
        floor=config.DEFAULT_ORDER_K_FLOOR,
        seed=seed,
        cv_n_init=cv_n_init,
        workers=workers,
        force=force,
    )
    pooled_k = int(pooled["supported_k"])

    resolved = get_cohort(dataset, version, root).axis(
        axis, matrix.features.index, matrix.covariates
    )
    if resolved is None:
        raise typer.BadParameter(f"cohort {dataset!r} does not provide axis {axis!r}")
    axis_values, _policy = resolved

    # The ORDER strata are the coarse four-way quantile split (plan section 7), not the fine
    # frozen MaxEqualBins primary; the DSM-5 boundary split is an era-only secondary pair.
    quantile = strata_mod.QuantileBins(q=4)
    quantile_assignment = quantile.assign(axis_values)
    stratum_plan: list[tuple[str, str, pd.Index]] = [
        ("quantile", label, quantile_assignment.codes[quantile_assignment.codes == label].index)
        for label in quantile_assignment.labels
    ]
    dsm_assignment = None
    if axis == "era":
        dsm = strata_mod.FixedBands(edges=(2013.0,), labels=("pre_dsm5", "dsm5"), name="dsm5")
        dsm_assignment = dsm.assign(axis_values)
        stratum_plan += [
            ("dsm5", label, dsm_assignment.codes[dsm_assignment.codes == label].index)
            for label in dsm_assignment.labels
        ]

    params = {
        "cohort": cohort_hash,
        "axis": axis,
        "pooled": pooled_hash,
        "quantile_policy": quantile.spec(),
        "dsm_split": dsm_assignment is not None,
        **_order_params(
            cohort_hash,
            recipe=recipe,
            schedule=schedule,
            anchor=k_anchor,
            cap=k_cap,
            floor=config.DEFAULT_ORDER_K_FLOOR,
            seed=seed,
        ),
    }

    with run_context("order", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            cached = (cache.read_manifest(ctx.run_dir) or {}).get("metrics", {})
            typer.echo(f"order: cache hit {cache.short_hash(ctx.run_hash)}; {cached}")
            return
        if force:
            checkpoint_mod.clear_checkpoints(ctx.run_dir)

        log = checkpoint_mod.CheckpointLog(ctx.path(f"draws_{axis}.checkpoint.jsonl"))
        done = {(r["scope"], r["k_null"], r["idx"]): r["lr"] for r in log.load()}
        n_workers = workers or max(1, (os.cpu_count() or 2) - 1)

        rows: list[dict] = []
        step_rows: list[dict] = _order_step_rows("__pooled__", _PooledSteps(pooled["steps"]))
        with (
            profiling.single_threaded_blas(),
            ProcessPoolExecutor(max_workers=n_workers) as pool,
        ):
            for scope_id, (scheme, label, members) in enumerate(stratum_plan, start=1):
                sub = CohortMatrix(
                    matrix.features.loc[members], matrix.covariates.loc[members], dataset, version
                )
                measurement_data, descriptor = order_mod.measurement_inputs(sub, typing)
                ctx.log.info(
                    "order %s stratum %s (%s): n=%d", axis, label, scheme, len(measurement_data)
                )
                result = _run_order_search(
                    measurement_data,
                    sub.covariates,
                    descriptor,
                    scope_id=scope_id,
                    pool=pool,
                    log=log,
                    done=done,
                    recipe=recipe,
                    schedule=schedule,
                    anchor=k_anchor,
                    cap=k_cap,
                    floor=config.DEFAULT_ORDER_K_FLOOR,
                    seed=seed,
                    cv_n_init=cv_n_init,
                )
                step_rows.extend(_order_step_rows(f"{scheme}:{label}", result))
                total_draws = sum(s.b_used + s.n_dropped for s in result.steps)
                drop_fraction = result.n_dropped / total_draws if total_draws else 0.0
                primary = next(
                    (s for s in result.steps if s.k_null == k_anchor and s.direction == "split"),
                    None,
                )
                order_changed = result.supported_k != pooled_k
                rows.append(
                    {
                        "stratum": label,
                        "scheme": scheme,
                        "n": int(len(measurement_data)),
                        "supported_k": result.supported_k,
                        "supported_label": _order_supported_label(result, k_cap),
                        "capped": result.capped,
                        "direction": result.direction,
                        "primary_p": primary.p_value if primary else float("nan"),
                        "primary_escalated": bool(primary.escalated) if primary else False,
                        "elbow_knee": result.elbow_knee,
                        "vlmr_p": result.vlmr.get("p", float("nan")),
                        "vlmr_lmr": result.vlmr.get("lmr", float("nan")),
                        "order_changed_vs_pooled": order_changed,
                        "agreement": order_mod.agreement_flag(result, pooled_k),
                        "n_dropped": result.n_dropped,
                        "drop_fraction": drop_fraction,
                        "flagged_degenerate": drop_fraction > 0.10,
                    }
                )

        decision = pd.DataFrame(rows)
        cache.save_frame(decision, ctx.path(f"decision_{axis}.parquet"))
        cache.save_frame(pd.DataFrame(step_rows), ctx.path(f"steps_{axis}.parquet"))
        cache.save_json(
            {
                "pooled_hash": cache.short_hash(pooled_hash),
                "pooled_supported_k": pooled_k,
                "pooled_supported_label": pooled["supported_label"],
                "pooled_elbow_knee": pooled["elbow_knee"],
                "pooled_vlmr_p": pooled["vlmr_p"],
            },
            ctx.path(f"pooled_reference_{axis}.json"),
        )
        checkpoint_mod.clear_checkpoints(ctx.run_dir)

        n_changed = int(decision["order_changed_vs_pooled"].sum())
        n_agree = int(decision["agreement"].sum())
        n_flagged = int(decision["flagged_degenerate"].sum())
        ctx.metrics = {
            "axis": axis,
            "pooled_supported_k": pooled_k,
            "pooled_supported_label": pooled["supported_label"],
            "n_strata": len(decision),
            "supported_k": {r["stratum"]: r["supported_label"] for r in rows},
            "n_order_changed": n_changed,
            "n_agreement": n_agree,
            "n_flagged_degenerate": n_flagged,
        }
    typer.echo(f"order {axis} {dataset}/{version}: run {cache.short_hash(ctx.run_hash)}")
    typer.echo(
        f"  pooled K*={pooled['supported_label']}; per-stratum "
        f"{ctx.metrics['supported_k']}; {n_agree} corroborated order change(s)"
    )


class _PooledSteps:
    """Adapt the cached pooled step records to the ``.steps`` attribute the flattener reads."""

    def __init__(self, step_dicts: list[dict]) -> None:
        from types import SimpleNamespace

        self.steps = [SimpleNamespace(**d) for d in step_dicts]


@app.command()
def sweep(
    axis: str = typer.Option("age_at_diagnosis", help="Axis: age_at_diagnosis or era."),
    dataset: str = _DATASET,
    version: str = _VERSION,
    scheme: list[str] = typer.Option(  # noqa: B006 (typer reads the default list)
        ["hardbins:max-equal:1000"],
        "--scheme",
        help="Localisation scheme, repeatable. 'hardbins:max-equal:<floor>', "
        "'hardbins:quantile:<q>', or 'kernel:<bandwidth>:<n_points>'. The kernel arm is heavy "
        "(each focal fit spans the cohort under a wide bandwidth) and its bandwidth needs "
        "choosing, so it is opt-in: add e.g. '--scheme kernel:1.0:12'.",
    ),
    n_init: int = typer.Option(config.DEFAULT_N_INIT, help="StepMix restarts per local fit."),
    n_permutations: int = typer.Option(
        config.DEFAULT_N_PERMUTATIONS,
        "--n-permutations",
        help="Permutations for the shared axis-permutation null (1 to smoke, 100 pilot, 1000 "
        "confirmatory).",
    ),
    alignment: str = typer.Option("membership", help="Alignment: membership or centroid."),
    distance: str = typer.Option(
        "mahalanobis", help="Distance: mahalanobis, euclidean, mean-abs, or jsd."
    ),
    seed: int = typer.Option(config.DEFAULT_STRATIFY_SEED, help="Base seed."),
    workers: int = typer.Option(0, help="Parallel fit workers (0 = logical cores minus one)."),
    no_covariates: bool = typer.Option(
        False,
        "--no-covariates",
        help="Fit the local models measurement-only (drop the covariate structural model), which "
        "the kernel arm needs because the covariate GLM diverges under fractional weights. This "
        "changes the estimand, so compare it only against a measurement-only reference.",
    ),
    force: bool = _FORCE,
) -> None:
    """Run every localisation scheme end to end and read them against one shared null.

    The phase-4 conductor. For each ``--scheme`` (hard bins, the frozen primary, and the kernel
    LSEM trajectory, by default) it fits the observed sweep and the axis-permutation null, aligns
    each local fit to the pooled reference, measures its drift with the chosen ``--distance``, and
    reads it against the null, writing one combined decision table with a ``scheme`` and
    ``position`` column so the arms plot as trajectories side by side. Nothing here is frozen in
    code: the primary scheme is a choice of which row of the table to privilege, so the pipeline
    stays flexible while the whole battery is computed on every run. This is the heavy stage: the
    null is ``n_permutations`` full sweeps, so the confirmatory both-scheme run is a cluster job;
    pass ``--n-permutations 1`` to smoke the chain or ``100`` for an overnight pilot.
    """
    import os
    from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait

    import pandas as pd

    from analysis import checkpoint, model, profiling, strata_data
    from analysis import drift as drift_mod
    from analysis import sweep as sweep_mod
    from analysis.localise import DEFAULT_WEIGHT_FLOOR, LocalFit, permute_axis
    from analysis.paths import run_dir
    from analysis.progress import task_bar

    if axis not in ("age_at_diagnosis", "era"):
        raise typer.BadParameter("axis must be 'age_at_diagnosis' or 'era'")
    if alignment not in drift_mod.ALIGNMENTS:
        raise typer.BadParameter(f"alignment must be one of {sorted(drift_mod.ALIGNMENTS)}")
    if distance not in drift_mod.DISTANCES:
        raise typer.BadParameter(f"distance must be one of {sorted(drift_mod.DISTANCES)}")
    try:
        schemes = [(spec, sweep_mod.parse_scheme(spec)) for spec in scheme]
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    def labels_series(frame: pd.DataFrame) -> pd.Series:
        others = [c for c in frame.columns if c != "class"]
        return frame.set_index(others[0])["class"] if others else frame["class"]

    root = find_repo_root()
    cohort_hash, _ = _run_cohort(root, dataset, version, force=force)
    matrix, typing = _load_cohort_matrix(root, cohort_hash, dataset, version)

    # Align to a like-specification reference: a measurement-only sweep matches the
    # measurement-only reference (`fit --no-covariates`), not the covariate one, so the estimand
    # is consistent on both sides.
    ref_params = _fit_params(cohort_hash, config.DEFAULT_N_COMPONENTS, config.DEFAULT_N_INIT, seed)
    if no_covariates:
        ref_params = {**ref_params, "structural": "measurement"}
    ref_fit_hash = cache.compute_hash(ref_params)
    ref_dir = run_dir(root, "fit", cache.short_hash(ref_fit_hash))
    if not (ref_dir / "labels.parquet").is_file():
        hint = " --no-covariates" if no_covariates else ""
        raise typer.BadParameter(f"no reference fit at {ref_dir}; run `analysis fit{hint}`")
    ref_labels = labels_series(cache.load_frame(ref_dir / "labels.parquet"))
    measurement_data, _descriptor, _covariates = model.prepare_inputs(matrix, typing)
    reference = drift_mod.build_reference(measurement_data, ref_labels)
    aligner = drift_mod.ALIGNMENTS[alignment]
    distancer = drift_mod.DISTANCES[distance]

    data = strata_data.build_strata_data(
        root,
        version,
        matrix.features.index,
        matrix.covariates["age_at_eval_years"],
        matrix.covariates["sex"],
    )
    column = "age_at_diagnosis_years" if axis == "age_at_diagnosis" else "diagnosis_year"
    axis_values = data.axes[column].reindex(matrix.features.index).dropna()

    n_workers = workers or max(1, (os.cpu_count() or 2) - 1)
    structural = None if no_covariates else "covariate"
    params = {
        "cohort": cohort_hash,
        "axis": axis,
        "ref_fit": ref_fit_hash,
        "schemes": [s.spec() for _, s in schemes],
        "structural": "measurement" if no_covariates else "covariate",
        "n_init": n_init,
        "n_permutations": n_permutations,
        "seed": seed,
    }

    # The local fits, keyed by (scheme, perm, offset) where perm = -1 is the observed sweep. Each
    # (scheme, perm) shares one axis (the observed axis, or a per-perm shuffle), memoised so its
    # several local fits reuse the same weights.
    def locales_for(scheme_index: int, perm: int) -> list[LocalFit]:
        _, sch = schemes[scheme_index]
        values = axis_values if perm < 0 else permute_axis(axis_values, seed=perm)
        return sch.locales(values)

    locale_cache: dict[tuple[int, int], list[LocalFit]] = {}

    def locales_cached(scheme_index: int, perm: int) -> list[LocalFit]:
        key = (scheme_index, perm)
        if key not in locale_cache:
            locale_cache[key] = locales_for(scheme_index, perm)
        return locale_cache[key]

    def n_locales(scheme_index: int) -> int:
        return len(locales_cached(scheme_index, -1))

    with run_context("sweep", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            cached = (cache.read_manifest(ctx.run_dir) or {}).get("metrics", {})
            typer.echo(f"sweep: cache hit {cache.short_hash(ctx.run_hash)}; {cached}")
            return
        if force:
            checkpoint.clear_checkpoints(ctx.run_dir)

        store = checkpoint.CheckpointLog(ctx.path("summaries.checkpoint.jsonl"))
        done = {(r["scheme_idx"], r["perm"], r["s_idx"]) for r in store.load()}
        perms = [-1, *range(n_permutations)]
        queue = iter(
            (k, perm, offset)
            for k in range(len(schemes))
            for perm in perms
            for offset in range(n_locales(k))
            if (k, perm, offset) not in done
        )
        total = sum(len(perms) * n_locales(k) for k in range(len(schemes)))

        with (
            task_bar(total, f"sweep {axis}") as bar,
            profiling.single_threaded_blas(),
            ProcessPoolExecutor(max_workers=n_workers) as pool,
        ):
            bar.update(len(done))

            def submit_next() -> tuple[Future, tuple[int, int, int], LocalFit] | None:
                task = next(queue, None)
                if task is None:
                    return None
                k, perm, offset = task
                locale = locales_cached(k, perm)[offset]
                weights = locale.weights.reindex(matrix.features.index).fillna(0.0)
                keep = weights.to_numpy() > DEFAULT_WEIGHT_FLOOR
                retained = weights[keep]
                sub_weights = None if bool((retained == 1.0).all()) else retained
                future = pool.submit(
                    sweep_mod.summarise_local_worker,
                    matrix.features.loc[keep],
                    matrix.covariates.loc[keep],
                    typing,
                    dataset,
                    version,
                    sub_weights,
                    ref_labels,
                    n_init,
                    seed + (perm + 1) * 100000 + k * 1000 + offset,
                    structural,
                )
                return future, task, locale

            in_flight: dict[Future, tuple[tuple[int, int, int], LocalFit]] = {}
            for _ in range(n_workers):
                submitted = submit_next()
                if submitted is None:
                    break
                future, key, locale = submitted
                in_flight[future] = (key, locale)
            while in_flight:
                finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in finished:
                    (k, perm, offset), locale = in_flight.pop(future)
                    result = future.result()
                    meta = {
                        "scheme_idx": int(k),
                        "label": locale.label,
                        "position": locale.position,
                    }
                    if result is None:
                        # A degenerate local refit; recorded (resumable) and dropped at read time.
                        record = {
                            "perm": int(perm),
                            "s_idx": int(offset),
                            "degenerate": True,
                            **meta,
                        }
                    else:
                        record = drift_mod.serialise_summary(result, perm, offset)
                        record.update(meta)
                    store.append(record)
                    bar.update(1)
                    bar.set_postfix({"scheme": k, "perm": perm})
                    submitted = submit_next()
                    if submitted is not None:
                        next_future, key, next_locale = submitted
                        in_flight[next_future] = (key, next_locale)

        # Cheap half: rebuild summaries from the store and read each scheme against the null.
        records = store.load()
        n_degenerate = sum(1 for r in records if r.get("degenerate"))
        decisions: list[pd.DataFrame] = []
        summary_metrics: dict[str, dict] = {}
        for k, (spec, _sch) in enumerate(schemes):
            scheme_records = [
                r for r in records if r["scheme_idx"] == k and not r.get("degenerate")
            ]
            observed = [
                sweep_mod.LocaleSummary(
                    r["label"], float(r["position"]), drift_mod.deserialise_summary(r)
                )
                for r in sorted(
                    (r for r in scheme_records if r["perm"] == -1), key=lambda r: r["s_idx"]
                )
            ]
            null = [
                (int(r["s_idx"]), drift_mod.deserialise_summary(r))
                for r in scheme_records
                if r["perm"] >= 0
            ]
            decision = sweep_mod.sweep_decision(spec, observed, null, reference, aligner, distancer)
            decisions.append(decision)
            n_drift = int((decision["fdr_significant"] & ~decision["reorganised"]).sum())
            summary_metrics[spec] = {
                "n_locales": len(observed),
                "n_tests": int(len(decision)),
                "n_drift": n_drift,
                "n_reorganised": int(decision["reorganised"].sum()),
            }
        combined = pd.concat(decisions, ignore_index=True)
        cache.save_frame(combined, ctx.path(f"decision_{axis}.parquet"))
        ctx.metrics = {
            "axis": axis,
            "alignment": alignment,
            "distance": distance,
            "n_permutations": n_permutations,
            "class_separation": drift_mod.class_separation(reference, distancer),
            "n_degenerate": n_degenerate,
            "schemes": summary_metrics,
        }
    typer.echo(f"sweep {axis} ({alignment}/{distance}): run {cache.short_hash(ctx.run_hash)}")
    for spec, m in summary_metrics.items():
        typer.echo(
            f"  {spec}: {m['n_drift']}/{m['n_tests']} classes drift beyond the "
            f"N={n_permutations} null; {m['n_reorganised']} reorganised"
        )


@app.command()
def bandwidth(
    axis: str = typer.Option("age_at_diagnosis", help="Axis: age_at_diagnosis or era."),
    dataset: str = _DATASET,
    version: str = _VERSION,
    n_points: int = typer.Option(20, help="Focal grid size the bandwidth is chosen for."),
    targets: str = typer.Option(
        "500,1000,2000,4000", help="Comma-separated target effective sample sizes."
    ),
    reduce: str = typer.Option(
        "min", help="Hold the target at the 'min' focal point (every fit clears it) or 'median'."
    ),
    force: bool = _FORCE,
) -> None:
    """Choose a kernel bandwidth by the effective sample size of its focal fits.

    A kernel focal fit is worth the sum of its weights in whole probands, its effective sample
    size, which sets its power the way a bin's count sets a hard-bin fit's. This stage inverts
    that: for each target effective size it reports the bandwidth whose focal fits reach it, so a
    kernel window can be set to carry the same power as a hard bin at the recovery floor. With
    ``--reduce min`` (the default) every focal fit clears the target, the same guarantee the
    hard-bin floor gives every bin; ``--reduce median`` fixes the typical focal point instead and
    lets the edges thin. The bandwidth is in the axis units (years for age at diagnosis, years for
    the diagnosis year). No model is fitted, so the stage is fast; the reported bandwidths feed a
    kernel sweep, for example ``--scheme kernel:<bandwidth>:<n_points>``.
    """
    import pandas as pd

    from analysis import localise, strata_data

    if axis not in ("age_at_diagnosis", "era"):
        raise typer.BadParameter("axis must be 'age_at_diagnosis' or 'era'")
    if reduce not in ("min", "median"):
        raise typer.BadParameter("reduce must be 'min' or 'median'")
    try:
        target_list = [float(t) for t in targets.split(",")]
    except ValueError as exc:
        raise typer.BadParameter(f"targets must be comma-separated numbers: {exc}") from exc

    root = find_repo_root()
    cohort_hash, _ = _run_cohort(root, dataset, version, force=force)
    matrix, _typing = _load_cohort_matrix(root, cohort_hash, dataset, version)
    data = strata_data.build_strata_data(
        root,
        version,
        matrix.features.index,
        matrix.covariates["age_at_eval_years"],
        matrix.covariates["sex"],
    )
    column = "age_at_diagnosis_years" if axis == "age_at_diagnosis" else "diagnosis_year"
    axis_values = data.axes[column].reindex(matrix.features.index).dropna()
    focal = localise.focal_grid(axis_values, n_points, (0.025, 0.975))

    params = {
        "cohort": cohort_hash,
        "axis": axis,
        "n_points": n_points,
        "targets": target_list,
        "reduce": reduce,
    }
    with run_context("bandwidth", params, root=root, force=force) as ctx:
        rows: list[dict] = []
        for target in target_list:
            h = localise.bandwidth_for_effective_n(axis_values, focal, target, reduce=reduce)
            eff = localise.effective_sample_sizes(axis_values, focal, h)
            rows.append(
                {
                    "target_n": target,
                    "bandwidth": round(h, 4),
                    "eff_min": round(float(eff.min()), 1),
                    "eff_median": round(float(pd.Series(eff).median()), 1),
                    "eff_max": round(float(eff.max()), 1),
                }
            )
        table = pd.DataFrame(rows)
        cache.save_frame(table, ctx.path(f"bandwidth_{axis}.parquet"))
        ctx.metrics = {"axis": axis, "n_points": n_points, "reduce": reduce, "rows": rows}
    typer.echo(
        f"bandwidth {axis} (reduce={reduce}, {n_points} focal points): "
        f"run {cache.short_hash(ctx.run_hash)}"
    )
    for row in rows:
        typer.echo(
            f"  effective N {int(row['target_n']):>5} -> bandwidth {row['bandwidth']:.3f}  "
            f"(focal N min {row['eff_min']:.0f} / median {row['eff_median']:.0f} / "
            f"max {row['eff_max']:.0f})"
        )


def _embedding_row(
    kind: str,
    ref_class: int,
    class_name: str,
    stratum: str,
    order: int,
    ld: Sequence[float],
    jaccard: float,
    *,
    reorganised: bool,
    cov: Sequence[float] = (float("nan"), float("nan"), float("nan")),
) -> dict[str, object]:
    """Return one row of the embedding table.

    Pads the discriminant coordinates to three, and carries the class members' covariance in
    the first two discriminant axes (``cov11``, ``cov12``, ``cov22``) on the anchor rows, so the
    figure can draw each class's coverage ellipse; the stratum rows leave it missing.
    """
    coords = [float(x) for x in ld]
    coords += [float("nan")] * (3 - len(coords))
    covariance = [float(x) for x in cov]
    return {
        "kind": kind,
        "ref_class": int(ref_class),
        "class_name": class_name,
        "stratum": stratum,
        "order": int(order),
        "ld1": coords[0],
        "ld2": coords[1],
        "ld3": coords[2],
        "jaccard": float(jaccard),
        "reorganised": bool(reorganised),
        "cov11": covariance[0],
        "cov12": covariance[1],
        "cov22": covariance[2],
    }


@app.command()
def trajectory(
    axis: str = typer.Option("age_at_diagnosis", help="Axis: age_at_diagnosis or era."),
    dataset: str = _DATASET,
    version: str = _VERSION,
    n_init: int = typer.Option(
        config.DEFAULT_N_INIT, help="StepMix restarts of the reference fit."
    ),
    n_shuffle: int = typer.Option(
        config.DEFAULT_TRAJECTORY_SHUFFLES,
        "--n-shuffle",
        help="Ordering-shuffle permutations for the directional test.",
    ),
    seed: int = typer.Option(config.DEFAULT_TRAJECTORY_SEED, help="Base seed."),
    min_bin_size: int = typer.Option(1000, help="Strata floor; must match strata/stratify."),
    from_sweep: str | None = typer.Option(
        None,
        "--from-sweep",
        help="Build the trajectory from a sweep run's focal centroids (its short hash) instead "
        "of the hard-bin strata, so a kernel sweep gets the same discriminant-space figure.",
    ),
    force: bool = _FORCE,
) -> None:
    """Embed the classes and measure how each one's centroid moves across the strata.

    A phase-4 presentation stage that consumes the stratified fits (``analysis stratify``) and
    writes only aggregate, class-level outputs. It aligns each stratum's classes to the named
    reference by membership (reusing :mod:`analysis.drift`), fits a linear-discriminant
    embedding of the four pooled classes, projects the reference anchors and the aligned
    stratum centroids into it, and quantifies each class's trajectory: a directional test (net
    young-to-old displacement against an ordering-shuffle null) and a roughness measure (step
    size against the sampling-noise expectation). The directional test is a pilot on the
    observed centroids; the confirmatory test is the continuous-trend regression against the
    refit permutation null (plan section 12a). Nothing per-proband is written, so the outputs
    may be promoted to the manuscript and the docs.
    """
    from collections import defaultdict

    import numpy as np
    import pandas as pd

    from analysis import checkpoint, model, strata_data, trajectory
    from analysis import drift as drift_mod
    from analysis import strata as strata_mod
    from analysis.cohort import CohortMatrix
    from analysis.paths import run_dir

    if axis not in ("age_at_diagnosis", "era"):
        raise typer.BadParameter("axis must be 'age_at_diagnosis' or 'era'")

    def labels_series(frame: pd.DataFrame) -> pd.Series:
        others = [c for c in frame.columns if c != "class"]
        return frame.set_index(others[0])["class"] if others else frame["class"]

    root = find_repo_root()
    cohort_hash, _ = _run_cohort(root, dataset, version, force=force)
    matrix, typing = _load_cohort_matrix(root, cohort_hash, dataset, version)

    ref_fit_hash = cache.compute_hash(
        _fit_params(cohort_hash, config.DEFAULT_N_COMPONENTS, config.DEFAULT_N_INIT, seed)
    )
    ref_dir = run_dir(root, "fit", cache.short_hash(ref_fit_hash))
    if not _completed(ref_dir):
        raise typer.BadParameter(f"no reference fit at {ref_dir}; run `analysis fit`")
    ref_labels = labels_series(cache.load_frame(ref_dir / "labels.parquet"))

    align_hash = cache.compute_hash(_align_params(root, ref_fit_hash))
    align_dir = run_dir(root, "align", cache.short_hash(align_hash))
    if not _completed(align_dir):
        raise typer.BadParameter(f"no reference alignment at {align_dir}; run `analysis align`")
    name_of = {
        int(k): v for k, v in cache.load_json(align_dir / "alignment.json")["mapping"].items()
    }

    measurement_data, _descriptor, _covariates = model.prepare_inputs(matrix, typing)
    reference = drift_mod.build_reference(measurement_data, ref_labels)
    columns = list(reference.centroids.columns)

    if from_sweep is not None:
        # Build the trajectory from a sweep run's observed focal centroids, so a kernel (LSEM)
        # sweep gets the same discriminant-space picture as the hard-bin strata.
        sweep_dir = run_dir(root, "sweep", from_sweep)
        if not _completed(sweep_dir):
            raise typer.BadParameter(f"no completed sweep run at {sweep_dir}; run `analysis sweep`")
        observed = [
            record
            for record in checkpoint.CheckpointLog(sweep_dir / "summaries.checkpoint.jsonl").load()
            if record.get("perm") == -1 and not record.get("degenerate")
        ]
        if not observed:
            raise typer.BadParameter(f"sweep run {from_sweep} has no observed focal fits")
        # A sweep may hold several schemes; take the one with the most focal points (the kernel
        # grid), ordered along the axis.
        by_scheme: dict[int, list] = defaultdict(list)
        for record in observed:
            by_scheme[int(record.get("scheme_idx", 0))].append(record)
        focal_records = max(by_scheme.values(), key=len)
        focal_records.sort(key=lambda record: float(record["position"]))
        labels = [str(record["label"]) for record in focal_records]
        params = {
            "cohort": cohort_hash,
            "ref_fit": ref_fit_hash,
            "align": align_hash,
            "sweep": from_sweep,
            "axis": axis,
            "embedding": "lda+ellipse",
            "n_shuffle": n_shuffle,
            "seed": seed,
        }
    else:
        focal_records = None
        strata = strata_data.build_strata_data(
            root,
            version,
            matrix.features.index,
            matrix.covariates["age_at_eval_years"],
            matrix.covariates["sex"],
        )
        column = "age_at_diagnosis_years" if axis == "age_at_diagnosis" else "diagnosis_year"
        policy = strata_mod.MaxEqualBins(min_bin_size=min_bin_size)
        assignment = policy.assign(strata.axes[column])
        labels = list(assignment.labels)

        stratify_params = {
            "cohort": cohort_hash,
            "axis": axis,
            "policy": policy.spec(),
            "n_components": config.DEFAULT_N_COMPONENTS,
            "n_init": n_init,
            "seed": seed,
            "limit": 0,
        }
        stratify_dir = run_dir(
            root, "stratify", cache.short_hash(cache.compute_hash(stratify_params))
        )
        params = {
            "cohort": cohort_hash,
            "ref_fit": ref_fit_hash,
            "align": align_hash,
            "stratify": cache.compute_hash(stratify_params),
            "axis": axis,
            "embedding": "lda+ellipse",
            "n_shuffle": n_shuffle,
            "seed": seed,
        }
    with run_context("trajectory", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            cached = (cache.read_manifest(ctx.run_dir) or {}).get("metrics", {})
            typer.echo(f"trajectory: cache hit {cache.short_hash(ctx.run_hash)}; {cached}")
            return

        embedding = trajectory.fit_embedding(measurement_data, ref_labels)
        n_classes = int(reference.centroids.shape[0])

        # Per stratum, align the fit's classes to the reference and pull the aligned centroid,
        # size, and Jaccard for each reference class.
        aligned: dict[int, list[pd.Series]] = {c: [] for c in range(n_classes)}
        sizes: dict[int, list[int]] = {c: [] for c in range(n_classes)}
        jaccard: dict[int, list[float]] = {c: [] for c in range(n_classes)}

        def _collect(summary: drift_mod.StratumSummary, label: str) -> None:
            alignment = drift_mod.ALIGNMENTS["membership"].align(summary, reference)
            inverse = {ref: fit for fit, ref in alignment.mapping.items()}
            if len(inverse) != n_classes:
                raise typer.BadParameter(
                    f"{label} did not recover {n_classes} classes; cannot build a trajectory for it"
                )
            for c in range(n_classes):
                fit_class = inverse[c]
                aligned[c].append(summary.centroids.loc[fit_class])
                sizes[c].append(int(round(float(summary.contingency.loc[fit_class].sum()))))
                jaccard[c].append(float(alignment.quality.get(c, float("nan"))))

        if focal_records is not None:
            for record in focal_records:
                _collect(drift_mod.deserialise_summary(record), str(record["label"]))
        else:
            for label in labels:
                lpath = stratify_dir / f"labels_{axis}_{label}.parquet"
                if not lpath.is_file():
                    raise typer.BadParameter(
                        f"no stratify fit for {axis}/{label}; run `analysis stratify --axis {axis}`"
                    )
                members = assignment.codes[assignment.codes == label].index
                sub = CohortMatrix(
                    matrix.features.loc[members], matrix.covariates.loc[members], dataset, version
                )
                stratum_md = model.prepare_inputs(sub, typing)[0]
                stratum_labels = labels_series(cache.load_frame(lpath)).reindex(stratum_md.index)
                _collect(drift_mod.summarise(stratum_md, stratum_labels, ref_labels), str(label))

        pooled_sd = embedding.sd
        embed_rows: list[dict] = []
        directional_rows: list[dict] = []
        roughness_rows: list[dict] = []
        anchor_ld = trajectory.project(embedding, reference.centroids.reindex(range(n_classes)))
        # Each pooled class's member covariance in the first two discriminant axes, the shape
        # the coverage ellipse is drawn from. Only the 2x2 covariance leaves the stage, never a
        # per-proband coordinate.
        ref_by_index = ref_labels.reindex(measurement_data.index).to_numpy()
        z_all = (
            measurement_data.loc[:, columns].to_numpy(dtype=float) - embedding.mean
        ) / pooled_sd
        member_ld = embedding.transformer.transform(z_all)[:, :2]
        for c in range(n_classes):
            covariance = np.cov(member_ld[ref_by_index == c], rowvar=False)
            embed_rows.append(
                _embedding_row(
                    "anchor",
                    c,
                    name_of.get(c, str(c)),
                    "",
                    -1,
                    anchor_ld[c].tolist(),
                    float("nan"),
                    reorganised=False,
                    cov=(covariance[0, 0], covariance[0, 1], covariance[1, 1]),
                )
            )
            raw = np.vstack(
                [aligned[c][s].loc[columns].to_numpy(dtype=float) for s in range(len(labels))]
            )
            standardised = (raw - embedding.mean) / pooled_sd
            stratum_ld = embedding.transformer.transform(standardised)
            within = reference.dispersions.loc[c, columns].to_numpy(dtype=float) / pooled_sd
            direction = trajectory.directional_test(
                standardised, seed=seed + c, n_shuffle=n_shuffle
            )
            rough = trajectory.roughness_metrics(
                standardised, np.asarray(sizes[c], dtype=float), within
            )
            for s, label in enumerate(labels):
                embed_rows.append(
                    _embedding_row(
                        "stratum",
                        c,
                        name_of.get(c, str(c)),
                        label,
                        s,
                        stratum_ld[s].tolist(),
                        jaccard[c][s],
                        reorganised=jaccard[c][s] < 0.5,
                    )
                )
            directional_rows.append(
                {"ref_class": c, "class_name": name_of.get(c, str(c)), **direction}
            )
            roughness_rows.append({"ref_class": c, "class_name": name_of.get(c, str(c)), **rough})

        cache.save_frame(pd.DataFrame(embed_rows), ctx.path(f"embedding_{axis}.parquet"))
        cache.save_frame(pd.DataFrame(directional_rows), ctx.path(f"directional_{axis}.parquet"))
        cache.save_frame(pd.DataFrame(roughness_rows), ctx.path(f"roughness_{axis}.parquet"))
        n_directional = sum(1 for row in directional_rows if row["significant"])
        ctx.metrics = {
            "axis": axis,
            "n_strata": len(labels),
            "strata": labels,
            "n_shuffle": n_shuffle,
            "lda_explained_variance": [float(v) for v in embedding.explained_variance_ratio],
            "n_directional": n_directional,
        }
    typer.echo(f"trajectory {axis} (lda): run {cache.short_hash(ctx.run_hash)}")
    typer.echo(
        f"  {ctx.metrics['n_directional']}/{n_classes} classes move with the axis "
        f"(ordering-shuffle p<0.05, pilot); {len(labels)} strata"
    )


@app.command()
def attribute(
    axis: str = typer.Option("age_at_diagnosis", help="Axis: age_at_diagnosis or era."),
    dataset: str = _DATASET,
    version: str = _VERSION,
    n_init: int = typer.Option(
        config.DEFAULT_N_INIT, help="Stratify restarts; must match the stratify run."
    ),
    alignment: str = typer.Option(
        "membership", help="Alignment: membership (default) or centroid."
    ),
    decomposition: str = typer.Option(
        "mahalanobis",
        help="Centroid-shift split: mahalanobis (default) or standardised.",
    ),
    contrast: str = typer.Option(
        "univariate", help="Mover/stayer contrast: univariate (default) or logistic."
    ),
    feature_space: str = typer.Option(
        "clustering",
        help="Features the contrast runs over. Only 'clustering' is built; the held-out SPARK "
        "and genetic frames are added later.",
    ),
    seed: int = typer.Option(
        config.DEFAULT_STRATIFY_SEED, help="Base seed; match the stratify run."
    ),
    min_bin_size: int = typer.Option(1000, help="Strata floor; must match strata/stratify."),
    force: bool = _FORCE,
) -> None:
    """Attribute each stratum class's movement to features and to the probands that moved.

    A phase-4 interpretation stage and a pure consumer of the reference fit and the stratify
    fits (no re-fitting). For every stratum it aligns the fit to the reference, splits each
    aligned class's centroid shift into per-feature contributions (:mod:`analysis.attribution`),
    and contrasts the probands that left the class against those that stayed. It writes the
    per-feature decomposition, the per-category totals, the mover-versus-stayer contrast, and a
    per-class headline table, so a movement reads as which features and which people carry it.
    The decomposition and contrast are descriptive readouts of an already-measured drift, so
    they sit outside the section-12a confirmatory freeze.
    """
    import pandas as pd

    from analysis import attribution as attr
    from analysis import drift as drift_mod
    from analysis import model, strata_data
    from analysis import strata as strata_mod
    from analysis.cohort import CohortMatrix
    from analysis.paths import run_dir

    if axis not in ("age_at_diagnosis", "era"):
        raise typer.BadParameter("axis must be 'age_at_diagnosis' or 'era'")
    if alignment not in drift_mod.ALIGNMENTS:
        raise typer.BadParameter(f"alignment must be one of {sorted(drift_mod.ALIGNMENTS)}")
    if decomposition not in attr.DECOMPOSITIONS:
        raise typer.BadParameter(f"decomposition must be one of {sorted(attr.DECOMPOSITIONS)}")
    if contrast not in attr.CONTRASTS:
        raise typer.BadParameter(f"contrast must be one of {sorted(attr.CONTRASTS)}")
    if feature_space != "clustering":
        raise typer.BadParameter("feature_space 'clustering' is the only space built yet")

    def labels_series(frame: pd.DataFrame) -> pd.Series:
        others = [c for c in frame.columns if c != "class"]
        return frame.set_index(others[0])["class"] if others else frame["class"]

    root = find_repo_root()
    cohort_hash, _ = _run_cohort(root, dataset, version, force=force)
    matrix, typing = _load_cohort_matrix(root, cohort_hash, dataset, version)

    ref_fit_hash = cache.compute_hash(
        _fit_params(cohort_hash, config.DEFAULT_N_COMPONENTS, config.DEFAULT_N_INIT, seed)
    )
    ref_dir = run_dir(root, "fit", cache.short_hash(ref_fit_hash))
    if not (ref_dir / "labels.parquet").is_file():
        raise typer.BadParameter(f"no reference fit at {ref_dir}; run `analysis fit`")
    ref_labels = labels_series(cache.load_frame(ref_dir / "labels.parquet"))
    measurement_data, _descriptor, _covariates = model.prepare_inputs(matrix, typing)
    reference = drift_mod.build_reference(measurement_data, ref_labels)

    align_dir = run_dir(
        root, "align", cache.short_hash(cache.compute_hash(_align_params(root, ref_fit_hash)))
    )
    name_of = (
        {int(k): v for k, v in cache.load_json(align_dir / "alignment.json")["mapping"].items()}
        if _completed(align_dir)
        else {}
    )

    aligner = drift_mod.ALIGNMENTS[alignment]
    decomposer = attr.DECOMPOSITIONS[decomposition]
    contraster = attr.CONTRASTS[contrast]
    category_map = features.load_category_map(config.author_category_map(root))

    data = strata_data.build_strata_data(
        root,
        version,
        matrix.features.index,
        matrix.covariates["age_at_eval_years"],
        matrix.covariates["sex"],
    )
    column = "age_at_diagnosis_years" if axis == "age_at_diagnosis" else "diagnosis_year"
    policy = strata_mod.MaxEqualBins(min_bin_size=min_bin_size)
    assignment = policy.assign(data.axes[column])
    labels = assignment.labels

    stratify_params = {
        "cohort": cohort_hash,
        "axis": axis,
        "policy": policy.spec(),
        "n_components": config.DEFAULT_N_COMPONENTS,
        "n_init": n_init,
        "seed": seed,
        "limit": 0,
    }
    stratify_hash = cache.compute_hash(stratify_params)
    stratify_dir = run_dir(root, "stratify", cache.short_hash(stratify_hash))

    params = {
        "stratify": stratify_hash,
        "ref_fit": ref_fit_hash,
        "axis": axis,
        "alignment": alignment,
        "decomposition": decomposition,
        "contrast": contrast,
        "feature_space": feature_space,
    }
    with run_context("attribute", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            cached = (cache.read_manifest(ctx.run_dir) or {}).get("metrics", {})
            typer.echo(f"attribute: cache hit {cache.short_hash(ctx.run_hash)}; {cached}")
            return

        decomp_rows: list[dict] = []
        category_rows: list[dict] = []
        mover_rows: list[dict] = []
        summary_rows: list[dict] = []

        for label in labels:
            lpath = stratify_dir / f"labels_{axis}_{label}.parquet"
            if not lpath.is_file():
                raise typer.BadParameter(
                    f"no stratify fit for {axis}/{label}; run `analysis stratify --axis {axis}`"
                )
            members = assignment.codes[assignment.codes == label].index
            sub = CohortMatrix(
                matrix.features.loc[members], matrix.covariates.loc[members], dataset, version
            )
            stratum_md = model.prepare_inputs(sub, typing)[0]
            stratum_labels = labels_series(cache.load_frame(lpath)).reindex(stratum_md.index)
            summary = drift_mod.summarise(stratum_md, stratum_labels, ref_labels)
            aligned = aligner.align(summary, reference)
            comparison = attr.Comparison(
                reference=reference,
                stratum=summary,
                ref_labels=ref_labels.loc[stratum_md.index],
                fit_labels=stratum_labels,
                alignment=aligned,
            )
            for movement in comparison.movements():
                contributions = decomposer.contributions(movement)
                shift = attr.signed_shift(movement)
                for feature, value in contributions.items():
                    decomp_rows.append(
                        {
                            "stratum": label,
                            "ref_class": movement.ref_class,
                            "feature": str(feature),
                            "contribution": float(value),
                            "signed_shift": float(shift.get(feature, float("nan"))),
                            "category": attr.category_of(feature, category_map),
                        }
                    )
                for category, total in attr.category_totals(contributions, category_map).items():
                    category_rows.append(
                        {
                            "stratum": label,
                            "ref_class": movement.ref_class,
                            "category": str(category),
                            "contribution": float(total),
                        }
                    )
                moved = attr.movers(movement, kind="either")
                result = contraster.contrast(moved, stratum_md.loc[moved.index])
                for record in result.importances.to_dict("records"):
                    mover_rows.append({"stratum": label, "ref_class": movement.ref_class, **record})
                top_shift = contributions.abs().idxmax() if len(contributions) else None
                top_mover = (
                    result.importances.iloc[0]["feature"] if len(result.importances) else None
                )
                counts = attr.membership_counts(movement)
                churned = counts["n_leavers"] + counts["n_joiners"]
                union = churned + counts["n_stayers"]
                summary_rows.append(
                    {
                        "stratum": label,
                        "ref_class": movement.ref_class,
                        "class_name": name_of.get(movement.ref_class, str(movement.ref_class)),
                        "n_stayers": counts["n_stayers"],
                        "n_leavers": counts["n_leavers"],
                        "n_joiners": counts["n_joiners"],
                        "churn": churned / union if union else float("nan"),
                        "jaccard": float(aligned.quality.get(movement.ref_class, float("nan"))),
                        "ari": float(aligned.overall),
                        "top_shift_feature": str(top_shift) if top_shift is not None else "",
                        "top_shift_category": (
                            attr.category_of(top_shift, category_map)
                            if top_shift is not None
                            else ""
                        ),
                        "top_mover_feature": str(top_mover) if top_mover is not None else "",
                    }
                )

        cache.save_frame(pd.DataFrame(decomp_rows), ctx.path(f"decomposition_{axis}.parquet"))
        cache.save_frame(pd.DataFrame(category_rows), ctx.path(f"category_{axis}.parquet"))
        cache.save_frame(pd.DataFrame(mover_rows), ctx.path(f"movers_{axis}.parquet"))
        summary_frame = pd.DataFrame(summary_rows)
        cache.save_frame(summary_frame, ctx.path(f"summary_{axis}.parquet"))
        ctx.metrics = {
            "axis": axis,
            "alignment": alignment,
            "decomposition": decomposition,
            "contrast": contrast,
            "feature_space": feature_space,
            "n_strata": len(labels),
            "n_classes": int(summary_frame["ref_class"].nunique()) if len(summary_frame) else 0,
            "mean_churn": (
                float(summary_frame["churn"].mean()) if len(summary_frame) else float("nan")
            ),
        }
    typer.echo(
        f"attribute {axis} ({decomposition}/{contrast}): run {cache.short_hash(ctx.run_hash)}"
    )
    typer.echo(
        f"  {ctx.metrics['n_strata']} strata x {ctx.metrics['n_classes']} classes; "
        f"mean churn {ctx.metrics['mean_churn']:.2f}"
    )


@app.command()
def invariance(
    axis: str = typer.Option(
        "age_at_diagnosis", help="Ordering variable: age_at_diagnosis or era (SPARK timing)."
    ),
    dataset: str = _DATASET,
    version: str = _VERSION,
    n_simulations: int = typer.Option(
        2000, "--n-simulations", help="Simulated Brownian bridges for the analytic null."
    ),
    seed: int = typer.Option(config.DEFAULT_STRATIFY_SEED, help="Base seed for the null draws."),
    max_grid: int = typer.Option(
        512, help="Largest evaluation grid (tied axis values collapsed, then thinned to this)."
    ),
    n_init: int = typer.Option(
        50, help="The n_init of the measurement-only reference fit to resolve."
    ),
    q: float = typer.Option(0.05, help="Benjamini-Hochberg FDR level across focal blocks."),
    min_bin_size: int = typer.Option(
        1000, help="Axis-policy floor for Cohort.axis; does not affect the continuous score test."
    ),
    force: bool = _FORCE,
) -> None:
    """Test the reference class profiles for stability along an axis, from the single cached fit.

    The score-based measurement-invariance test (plan section 7e): for the measurement-only
    reference, each proband's casewise score on every class-conditional location parameter is
    cumulated in axis order into an empirical fluctuation process, standardised to a Brownian
    bridge under stability. The maxLM and Cramer-von Mises functionals are read against an
    analytic (simulated-bridge) null, per focal block (each class, and each class crossed with a
    feature category), with Benjamini-Hochberg control across blocks. No mixture is refitted.

    Unlike the drift stage, this consumes the fitted model itself (its parameters and
    responsibilities), not only the labels, and it reads the marginal (measurement-only)
    reference, so its estimand matches the kernel and pairwise arms rather than the covariate
    fit.
    """
    import pandas as pd

    from analysis import invariance as invariance_mod
    from analysis import model, profiling
    from analysis.cohort import get_cohort
    from analysis.paths import run_dir

    root = find_repo_root()
    cohort_hash, _ = _run_cohort(root, dataset, version, force=force)
    matrix, typing = _load_cohort_matrix(root, cohort_hash, dataset, version)

    ref_fit_hash = _resolve_measurement_reference(root, cohort_hash, n_init, seed)
    ref_dir = run_dir(root, "fit", cache.short_hash(ref_fit_hash))
    reference_model = cache.load_model(ref_dir / "model.joblib")
    measurement_data, _descriptor, _covariates = model.prepare_inputs(matrix, typing)

    resolved = get_cohort(dataset, version, root).axis(
        axis, matrix.features.index, matrix.covariates, min_bin_size
    )
    if resolved is None:
        raise typer.BadParameter(f"cohort {dataset!r} does not provide axis {axis!r}")
    axis_values, _policy = resolved

    category_map = features.load_category_map(config.author_category_map(root))
    params = {
        "fit": ref_fit_hash,
        "axis": axis,
        "focal": "means+logit;class,class-x-category",
        "n_simulations": n_simulations,
        "seed": seed,
        "max_grid": max_grid,
        "q": q,
        "category_map": cache.file_digest(config.author_category_map(root)),
    }
    with run_context("invariance", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            cached = (cache.read_manifest(ctx.run_dir) or {}).get("metrics", {})
            typer.echo(f"invariance: cache hit {cache.short_hash(ctx.run_hash)}; {cached}")
            return
        with profiling.measure() as unit:
            result = invariance_mod.run_invariance(
                reference_model,
                measurement_data,
                typing,
                axis_values,
                axis=axis,
                category_map=category_map,
                by_category=True,
                n_sim=n_simulations,
                seed=seed,
                max_grid=max_grid,
                q=q,
            )
            blocks_frame = pd.DataFrame([asdict(b) for b in result.blocks])
            cache.save_frame(blocks_frame, ctx.path(f"blocks_{axis}.parquet"))
            if result.top_process is not None:
                top = result.top_process
                process_frame = pd.DataFrame(
                    {
                        "t": top.t,
                        "position": top.positions,
                        "observed": top.observed,
                        "null_q50": top.null_q50,
                        "null_q95": top.null_q95,
                    }
                )
                cache.save_frame(process_frame, ctx.path(f"process_{axis}.parquet"))
            unit.output_bytes = profiling.path_bytes(ctx.path(f"blocks_{axis}.parquet"))
        resources = unit.metrics
        assert resources is not None  # measure() always sets metrics on exit
        n_reject_max = int(blocks_frame["reject_max_lm"].sum())
        n_reject_cvm = int(blocks_frame["reject_cvm"].sum())
        ctx.metrics = {
            "axis": axis,
            "reference_fit": cache.short_hash(ref_fit_hash),
            "n_reference": result.n_reference,
            "n_covered": result.n_covered,
            "coverage": round(result.coverage, 4),
            "n_blocks": len(result.blocks),
            "n_simulations": n_simulations,
            "n_reject_max_lm": n_reject_max,
            "n_reject_cvm": n_reject_cvm,
            "min_p_max_lm": float(blocks_frame["p_max_lm"].min()),
            "top_block": result.top_process.label if result.top_process else None,
            "resources": resources.to_dict(),
        }
    typer.echo(f"invariance {axis}: run {cache.short_hash(ctx.run_hash)}")
    typer.echo(
        f"  coverage {result.n_covered}/{result.n_reference} ({result.coverage:.1%}); "
        f"{n_reject_max}/{len(result.blocks)} blocks drift beyond the bridge null "
        f"(maxLM, FDR q={q})"
    )


# The nine ordered household-income bands of ``background_history_child.annual_household_income``,
# mapped to a monotone rank so the band order (not the raw label) drives the control ordering.
_INCOME_BANDS = {
    "less_than_20000": 1,
    "21000_35000": 2,
    "36000_50000": 3,
    "51000_65000": 4,
    "66000_80000": 5,
    "81000_100000": 6,
    "101000_130000": 7,
    "131000_160000": 8,
    "over_161000": 9,
}


def _control_axis_series(
    name: str,
    root: Path,
    dataset: str,
    version: str,
    index: pd.Index,
    covariates: pd.DataFrame,
    seed: int,
) -> pd.Series:
    """Return a control-panel ordering variable on the cohort index.

    The specificity check reads the same separation-scaled displacement along variables that are
    not the mechanism under test, so that the era and age magnitudes can be read as larger than a
    control rather than merely non-zero. A control is chosen for two screened properties: it is a
    real proband-level covariate that is not the phenotype (so ordering by it is not the circular
    self-drift that a clustered feature or a symptom total would give), and it is empirically
    orthogonal to both timing axes (its rank correlation with year of diagnosis and with age at
    diagnosis is near zero on the modelling cohort). ``household_income`` (a nine-band ordinal) and
    ``area_deprivation`` (the 2019 ADI national-rank percentile) meet both and are graded, so the
    displacement trajectory has an axis to walk. ``random`` is a seeded uniform ordering, the floor
    a meaningless axis produces. ``sex`` is retained for reproducing the earlier panel, but it is
    binary (so the trajectory is degenerate) and it is the least timing-orthogonal of the
    candidates, so it is no longer in the default panel.
    """
    import numpy as np
    import pandas as pd

    from analysis.cohort import open_catalogue, read_columns, source_csv

    if name == "sex":
        return covariates["sex"].astype(float).reindex(index)
    if name == "random":
        rng = np.random.default_rng(seed)
        return pd.Series(rng.uniform(size=len(index)), index=index, name="random")
    if name == "household_income":
        cat = open_catalogue(root)
        path = source_csv(cat, root, dataset, version, "background_history_child")
        frame = read_columns(path, ["subject_sp_id", "annual_household_income"]).set_index(
            "subject_sp_id"
        )
        frame = frame[~frame.index.duplicated(keep="first")]
        return frame["annual_household_income"].map(_INCOME_BANDS).reindex(index)
    if name == "area_deprivation":
        cat = open_catalogue(root)
        path = source_csv(cat, root, dataset, version, "area_deprivation_index")
        # The release ships the column with an R-style ``X`` prefix (it starts with a digit),
        # which the dscat catalogue records without; read the on-disk name.
        column = "X2019_adi_national_rank_percentile"
        frame = read_columns(path, ["subject_sp_id", column]).set_index("subject_sp_id")
        frame = frame[~frame.index.duplicated(keep="first")]
        return pd.to_numeric(frame[column], errors="coerce").reindex(index)
    raise typer.BadParameter(f"unknown control axis {name!r}")


def _endpoint_magnitudes(
    x_values,
    responsibilities,
    axis_values: pd.Series,
    *,
    pooled_sd,
    separation_scale: float,
    grains,
    embedding,
    precision,
    plane,
    n_points: int,
) -> tuple[list[float], float, float]:
    """Return each class's separation-scaled endpoint magnitude, the bandwidth, and the endpoint.

    A light read used for the control panel: it derives the axis's own bandwidth at the recovery
    floor, builds the focal grid, and takes the whole-class magnitude at the endpoint focal point.
    The endpoint focal value is returned too, so a paired specificity test can re-read the same
    control variable at the same kernel centre and width.
    """
    import numpy as np

    from analysis import localise
    from analysis import trajectory_local as tl

    finite = axis_values.dropna()
    grid = localise.focal_grid(finite, min(n_points, max(2, finite.nunique())), (0.025, 0.975))
    bandwidth = localise.bandwidth_for_effective_n(finite, grid, 1000.0, reduce="min")
    observed = tl.observed_trajectory(
        x_values,
        responsibilities,
        axis_values.to_numpy(dtype=float),
        np.asarray(grid, dtype=float),
        bandwidth,
        pooled_sd=pooled_sd,
        separation_scale=separation_scale,
        grains={"class": grains["class"]},
        embedding=embedding,
        precision=precision,
        plane=plane,
    )
    endpoint = observed.grain_magnitude["class"][:, observed.focal_ref]
    return [float(v) for v in endpoint], float(bandwidth), float(grid[observed.focal_ref])


def _measurement_class_names(
    root: Path,
    dataset: str,
    version: str,
    cohort_hash: str,
    measurement_reference,
    measurement_data,
    seed: int,
) -> tuple[dict[int, str], str]:
    """Name the measurement-only classes by centroid-matching them to the covariate reference.

    The ``align`` stage names the covariate reference's classes (plan section 6a) but not the
    measurement-only fit the score test reads. Rather than leave the classes as bare ids, this
    aligns the measurement-only pooled centroids to the named covariate reference on the same
    cohort by the Hungarian centroid match (:class:`analysis.drift.CentroidHungarian`), so the
    figures carry Litman's class names without any refit. Falls back to numeric names when the
    covariate reference or its alignment is not cached.
    """
    from analysis import drift as drift_mod
    from analysis.paths import run_dir

    n_classes = int(measurement_reference.centroids.shape[0])
    default = {c: f"class {c}" for c in range(n_classes)}

    cov_fit_hash = cache.compute_hash(
        _fit_params(cohort_hash, config.DEFAULT_N_COMPONENTS, config.DEFAULT_N_INIT, seed)
    )
    cov_fit_dir = run_dir(root, "fit", cache.short_hash(cov_fit_hash))
    cov_align_hash = cache.compute_hash(_align_params(root, cov_fit_hash))
    cov_align_dir = run_dir(root, "align", cache.short_hash(cov_align_hash))
    if not (_completed(cov_fit_dir) and _completed(cov_align_dir)):
        return default, "class-ids"

    cov_labels = cache.load_frame(cov_fit_dir / "labels.parquet")["class"]
    cov_labels.index = measurement_data.index
    cov_reference = drift_mod.build_reference(measurement_data, cov_labels)
    cov_names = {
        int(k): v for k, v in cache.load_json(cov_align_dir / "alignment.json")["mapping"].items()
    }
    alignment = drift_mod.CentroidHungarian().align(
        measurement_reference.as_stratum(), cov_reference
    )
    names = {meas: cov_names.get(cov, f"class {meas}") for meas, cov in alignment.mapping.items()}
    resolved = {c: names.get(c, f"class {c}") for c in range(n_classes)}
    return resolved, cache.short_hash(cov_align_hash)


@app.command(name="invariance-trajectory")
def invariance_trajectory(
    axis: str = typer.Option(
        "era", help="Ordering variable: era or age_at_diagnosis (SPARK timing)."
    ),
    dataset: str = _DATASET,
    version: str = _VERSION,
    n_points: int = typer.Option(25, help="Focal grid points the local centroids are read at."),
    n_boot: int = typer.Option(
        500, "--n-boot", help="Clustered-bootstrap replicates for the tube and per-feature CIs."
    ),
    controls: bool = typer.Option(
        True, "--controls/--no-controls", help="Also read the control-panel specificity axes."
    ),
    seed: int = typer.Option(config.DEFAULT_STRATIFY_SEED, help="Base seed for the bootstrap."),
    n_init: int = typer.Option(
        50, help="The n_init of the measurement-only reference fit to resolve."
    ),
    q: float = typer.Option(0.05, help="Benjamini-Hochberg FDR level across per-feature tests."),
    min_bin_size: int = typer.Option(
        1000, help="Recovery floor the bandwidth is derived against (does not bin the axis)."
    ),
    force: bool = _FORCE,
) -> None:
    """Read how each class profile drifts along an axis as a null-free, separation-scaled effect.

    The recast of the score-based invariance test (plan section 7e). Freezing the measurement-only
    reference's responsibilities, it reads each class's local centroid as a smooth function of the
    axis, forms the per-feature displacement from the pooled centroid, and scales magnitudes by the
    between-class separation. Uncertainty is a family-clustered bootstrap (the tube and per-feature
    intervals), replacing the saturated bridge null. It reports the in-plane capture fraction so
    the 2D figure cannot hide out-of-plane drift, a covariance-aware Mahalanobis corroboration, and
    a control-panel specificity comparison (era and age against household income, area deprivation,
    and a random ordering). No mixture is refitted; every artefact is class or feature level.
    """
    import numpy as np
    import pandas as pd

    from analysis import drift as drift_mod
    from analysis import localise, profiling
    from analysis import trajectory as trajectory_mod
    from analysis import trajectory_local as tl
    from analysis.cohort import get_cohort
    from analysis.paths import run_dir

    if axis not in ("age_at_diagnosis", "era"):
        raise typer.BadParameter("axis must be 'age_at_diagnosis' or 'era'")

    def labels_series(frame: pd.DataFrame) -> pd.Series:
        others = [c for c in frame.columns if c != "class"]
        return frame.set_index(others[0])["class"] if others else frame["class"]

    root = find_repo_root()
    cohort_hash, _ = _run_cohort(root, dataset, version, force=force)
    matrix, typing = _load_cohort_matrix(root, cohort_hash, dataset, version)

    ref_fit_hash = _resolve_measurement_reference(root, cohort_hash, n_init, seed)
    ref_dir = run_dir(root, "fit", cache.short_hash(ref_fit_hash))
    reference_model = cache.load_model(ref_dir / "model.joblib")
    ref_labels = labels_series(cache.load_frame(ref_dir / "labels.parquet"))

    measurement_data, _descriptor, _covariates = model.prepare_inputs(matrix, typing)
    columns = list(measurement_data.columns)
    x_values = measurement_data.to_numpy(dtype=float)
    responsibilities = invariance_responsibilities(reference_model, x_values)

    reference = drift_mod.build_reference(measurement_data, ref_labels)
    pooled_sd = reference.pooled_sd.reindex(columns).to_numpy(dtype=float)
    precision = reference.precision
    separation_scale = tl.separation(reference)
    embedding = trajectory_mod.fit_embedding(measurement_data, ref_labels)
    plane = tl.discriminant_plane(embedding)
    name_of, name_source = _measurement_class_names(
        root, dataset, version, cohort_hash, reference, measurement_data, seed
    )

    category_map = features.load_category_map(config.author_category_map(root))
    grains = tl.category_grains(columns, category_map)

    cohort = get_cohort(dataset, version, root)
    resolved = cohort.axis(axis, matrix.features.index, matrix.covariates, min_bin_size)
    if resolved is None:
        raise typer.BadParameter(f"cohort {dataset!r} does not provide axis {axis!r}")
    axis_values, _policy = resolved
    families = cohort.family_ids(matrix.features.index)
    if families is None:
        raise typer.BadParameter(
            f"cohort {dataset!r} exposes no family identifier for the clustered bootstrap"
        )

    covered = axis_values.reindex(measurement_data.index).notna().to_numpy()
    n_reference = int(x_values.shape[0])
    n_covered = int(covered.sum())

    x_cov = x_values[covered]
    resp_cov = responsibilities[covered]
    axis_cov = axis_values.reindex(measurement_data.index)[covered]
    fam_cov = families.reindex(measurement_data.index)[covered].to_numpy()

    finite = axis_cov.dropna()
    focal_points = np.asarray(localise.focal_grid(finite, n_points, (0.025, 0.975)), dtype=float)
    band_grid = localise.focal_grid(finite, 20, (0.025, 0.975))
    bandwidth = round(
        localise.bandwidth_for_effective_n(finite, band_grid, float(min_bin_size), reduce="min"), 4
    )

    n_classes = int(reference.centroids.shape[0])
    control_names = ["household_income", "area_deprivation", "random"] if controls else []
    params = {
        "fit": ref_fit_hash,
        "names": name_source,
        "axis": axis,
        "quantity": "local-centroid-displacement;frozen-responsibilities",
        "directional": "signed-net-projected-slope;single-break;v1",
        "bandwidth": bandwidth,
        "n_points": n_points,
        "n_boot": n_boot,
        "seed": seed,
        "q": q,
        "controls": control_names,
        "category_map": cache.file_digest(config.author_category_map(root)),
    }
    with run_context("invariance-trajectory", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            cached = (cache.read_manifest(ctx.run_dir) or {}).get("metrics", {})
            typer.echo(
                f"invariance-trajectory: cache hit {cache.short_hash(ctx.run_hash)}; {cached}"
            )
            return
        with profiling.measure() as unit:
            observed = tl.observed_trajectory(
                x_cov,
                resp_cov,
                axis_cov.to_numpy(dtype=float),
                focal_points,
                bandwidth,
                pooled_sd=pooled_sd,
                separation_scale=separation_scale,
                grains=grains,
                embedding=embedding,
                precision=precision,
                plane=plane,
            )
            # DIREC: the observed directional statistic (signed net-projected slope). Its net
            # directions are frozen here and handed to the bootstrap, so the projected slope stays
            # a fixed linear functional and its clustered-bootstrap interval can cover zero.
            directional = tl.directional_statistic(
                observed.displacement, pooled_sd, focal_points, separation_scale
            )
            tube = tl.clustered_bootstrap_tube(
                x_cov,
                resp_cov,
                axis_cov.to_numpy(dtype=float),
                fam_cov,
                focal_points,
                bandwidth,
                pooled_sd=pooled_sd,
                separation_scale=separation_scale,
                grains=grains,
                embedding=embedding,
                precision=precision,
                focal_ref=observed.focal_ref,
                n_boot=n_boot,
                seed=seed,
                net_directions=directional.net_direction,
            )
            observed_endpoint = observed.displacement[:, observed.focal_ref] / pooled_sd
            inference = tl.per_feature_inference(observed_endpoint, tube.feature_displacement, q=q)
            directional_inf = tl.directional_inference(directional, tube, q=q)

            # The fixed anchors and member ellipses of the pooled classes, in the embedding.
            anchor_ld = trajectory_mod.project(
                embedding, reference.centroids.reindex(range(n_classes))
            )[:, :2]
            member_cov = _member_covariance(embedding, ref_labels, measurement_data, n_classes)

            # Aggregate-only artefacts (class or feature level, never per proband).
            _write_local_trajectory_artefacts(
                ctx,
                axis=axis,
                columns=columns,
                category_map=category_map,
                name_of=name_of,
                observed=observed,
                tube=tube,
                inference=inference,
                anchor_ld=anchor_ld,
                member_cov=member_cov,
            )
            _write_directional_artefacts(
                ctx,
                axis=axis,
                name_of=name_of,
                observed=observed,
                directional=directional,
                directional_inf=directional_inf,
                tube=tube,
            )

            specificity_rows: list[dict] = []
            primary_endpoint = observed.grain_magnitude["class"][:, observed.focal_ref]
            for c in range(n_classes):
                specificity_rows.append(
                    {
                        "axis_name": axis,
                        "ref_class": c,
                        "class_name": name_of.get(c, str(c)),
                        "endpoint_magnitude": float(primary_endpoint[c]),
                    }
                )
            control_bandwidths: dict[str, float] = {}
            specificity_tests: list[dict] = []
            axis_endpoint_focal = float(focal_points[-1])
            fam_full = families.reindex(measurement_data.index)
            for control in control_names:
                control_axis = _control_axis_series(
                    control, root, dataset, version, matrix.features.index, matrix.covariates, seed
                )
                control_cov = control_axis.reindex(measurement_data.index)
                mask = control_cov.notna().to_numpy()
                mags, cband, cfocal = _endpoint_magnitudes(
                    x_values[mask],
                    responsibilities[mask],
                    control_cov[mask],
                    pooled_sd=pooled_sd,
                    separation_scale=separation_scale,
                    grains=grains,
                    embedding=embedding,
                    precision=precision,
                    plane=plane,
                    n_points=n_points,
                )
                control_bandwidths[control] = cband
                for c in range(n_classes):
                    specificity_rows.append(
                        {
                            "axis_name": control,
                            "ref_class": c,
                            "class_name": name_of.get(c, str(c)),
                            "endpoint_magnitude": float(mags[c]),
                        }
                    )
                # The paired specificity test (plan section 12b): resample the same families,
                # restricted to probands finite on both the timing axis and this control, for
                # both quantities in every replicate, so the difference is a genuine paired
                # statistic rather than a comparison of two separately noisy magnitudes.
                joint = covered & mask
                comparison = tl.control_specificity_bootstrap(
                    x_values[joint],
                    responsibilities[joint],
                    fam_full[joint].to_numpy(),
                    axis_values.reindex(measurement_data.index)[joint].to_numpy(dtype=float),
                    bandwidth,
                    axis_endpoint_focal,
                    control_cov[joint].to_numpy(dtype=float),
                    cband,
                    cfocal,
                    pooled_sd=pooled_sd,
                    separation_scale=separation_scale,
                    n_boot=n_boot,
                    seed=seed,
                )
                specificity_tests.append(
                    {
                        "axis": axis,
                        "control": control,
                        "n_joint": int(joint.sum()),
                        "axis_magnitude": comparison.axis_magnitude,
                        "control_magnitude": comparison.control_magnitude,
                        "difference": comparison.difference,
                        "p_value": comparison.p_value,
                        "p_value_greater": comparison.p_value_greater,
                    }
                )
            if specificity_tests:
                raw_p = np.array([row["p_value"] for row in specificity_tests])
                reject = drift_mod.benjamini_hochberg(raw_p, q)
                for row, rej in zip(specificity_tests, reject, strict=True):
                    row["reject"] = bool(rej)
                cache.save_frame(
                    pd.DataFrame(specificity_tests), ctx.path(f"specificity_test_{axis}.parquet")
                )
            cache.save_frame(
                pd.DataFrame(specificity_rows), ctx.path(f"specificity_{axis}.parquet")
            )
            unit.output_bytes = profiling.path_bytes(ctx.path(f"capture_{axis}.parquet"))

        resources = unit.metrics
        assert resources is not None
        corroboration = _score_test_corroboration(root, ref_fit_hash, axis)
        n_reject = int(inference.reject.sum())
        mean_endpoint = float(np.mean(primary_endpoint))
        ctx.metrics = {
            "axis": axis,
            "reference_fit": cache.short_hash(ref_fit_hash),
            "n_reference": n_reference,
            "n_covered": n_covered,
            "coverage": round(n_covered / n_reference, 4),
            "bandwidth": bandwidth,
            "n_points": n_points,
            "n_boot": n_boot,
            "separation": round(separation_scale, 4),
            "endpoint_magnitude_mean": round(mean_endpoint, 4),
            "endpoint_magnitude_by_class": [round(float(v), 4) for v in primary_endpoint],
            "capture_by_class": [round(float(v), 4) for v in observed.capture],
            "n_feature_reject": n_reject,
            "n_features_tested": int(inference.reject.size),
            "control_bandwidths": control_bandwidths,
            "specificity_tests": [
                {
                    "control": row["control"],
                    "n_joint": row["n_joint"],
                    "axis_magnitude": round(row["axis_magnitude"], 4),
                    "control_magnitude": round(row["control_magnitude"], 4),
                    "difference": round(row["difference"], 4),
                    "p_value": round(row["p_value"], 4),
                    "reject": row["reject"],
                }
                for row in specificity_tests
            ],
            "score_test_corroboration": corroboration,
            "directional": {
                "net_trend_by_class": [round(float(v), 4) for v in directional_inf.net_trend],
                "net_trend_ci": [
                    [round(float(lo), 4), round(float(hi), 4)]
                    for lo, hi in zip(
                        directional_inf.net_trend_lo, directional_inf.net_trend_hi, strict=True
                    )
                ],
                "p_value_by_class": [round(float(v), 4) for v in directional_inf.p_value],
                "reject_by_class": [bool(v) for v in directional_inf.reject],
                "n_directional": int(directional_inf.reject.sum()),
                "break_by_class": [round(float(v), 4) for v in directional_inf.break_position],
            },
            "resources": resources.to_dict(),
        }
    typer.echo(f"invariance-trajectory {axis}: run {cache.short_hash(ctx.run_hash)}")
    typer.echo(
        f"  coverage {n_covered}/{n_reference} ({n_covered / n_reference:.1%}); "
        f"bandwidth {bandwidth}; endpoint displacement (sep units) mean {mean_endpoint:.3f}"
    )
    typer.echo(
        "  per-class endpoint "
        + ", ".join(
            f"{name_of.get(c, c)} {primary_endpoint[c]:.2f} (capture {observed.capture[c]:.2f})"
            for c in range(n_classes)
        )
    )
    typer.echo(
        f"  {n_reject}/{inference.reject.size} per-feature displacements survive FDR (q={q})"
    )
    typer.echo(
        "  directional (net trend, sep units): "
        + ", ".join(
            f"{name_of.get(c, c)} {directional_inf.net_trend[c]:+.2f} "
            f"[{directional_inf.net_trend_lo[c]:+.2f},{directional_inf.net_trend_hi[c]:+.2f}] "
            f"{'DIRECTIONAL' if directional_inf.reject[c] else 'ns'}"
            for c in range(n_classes)
        )
    )
    if specificity_tests:
        typer.echo(
            f"  specificity (paired bootstrap, mean endpoint vs. control, BH q={q:.2f}): "
            + ", ".join(
                f"{row['control']} d={row['difference']:+.2f} p={row['p_value']:.4f} "
                f"{'SPECIFIC' if row['reject'] else 'ns'}"
                for row in specificity_tests
            )
        )


def _write_local_trajectory_artefacts(
    ctx,
    *,
    axis: str,
    columns: list[str],
    category_map,
    name_of: dict,
    observed,
    tube,
    inference,
    anchor_ld,
    member_cov,
) -> None:
    """Write the class and feature-level tables of a local-trajectory run.

    Four aggregate artefacts: the per-feature endpoint displacement with its interval and FDR
    decision; the separation-scaled grain magnitude trajectory with its tube; the discriminant-
    plane trajectory with the centroid tube and the anchor member ellipses; and the per-class
    summary (capture fraction, endpoint magnitude, Mahalanobis).
    """
    import numpy as np
    import pandas as pd

    n_classes = observed.displacement.shape[0]
    focal_ref = observed.focal_ref

    feature_rows: list[dict] = []
    for c in range(n_classes):
        for j, feature in enumerate(columns):
            feature_rows.append(
                {
                    "ref_class": c,
                    "class_name": name_of.get(c, str(c)),
                    "feature": feature,
                    "category": category_map.get(str(feature), "unmapped"),
                    "displacement": float(inference.displacement[c, j]),
                    "ci_low": float(inference.ci_low[c, j]),
                    "ci_high": float(inference.ci_high[c, j]),
                    "p_value": float(inference.p_value[c, j]),
                    "reject": bool(inference.reject[c, j]),
                    "covers_zero": bool(inference.covers_zero[c, j]),
                }
            )
    cache.save_frame(pd.DataFrame(feature_rows), ctx.path(f"feature_displacement_{axis}.parquet"))

    magnitude_rows: list[dict] = []
    for grain, bands in tube.grain_bands.items():
        observed_grain = observed.grain_magnitude[grain]
        for c in range(n_classes):
            for j, position in enumerate(observed.focal_points):
                magnitude_rows.append(
                    {
                        "grain": grain,
                        "ref_class": c,
                        "class_name": name_of.get(c, str(c)),
                        "focal_index": j,
                        "position": float(position),
                        "magnitude": float(observed_grain[c, j]),
                        "band_lo": float(bands[0, c, j]),
                        "band_hi": float(bands[2, c, j]),
                    }
                )
    cache.save_frame(pd.DataFrame(magnitude_rows), ctx.path(f"grain_magnitude_{axis}.parquet"))

    # The plane trajectory: anchors (with the member ellipse) plus per-focal centroids with the
    # clustered-bootstrap tube box (the 2.5 and 97.5 percentiles of the resampled position).
    ld_lo = np.nanpercentile(tube.ld, 2.5, axis=0)
    ld_hi = np.nanpercentile(tube.ld, 97.5, axis=0)
    plane_rows: list[dict] = []
    for c in range(n_classes):
        plane_rows.append(
            {
                "kind": "anchor",
                "ref_class": c,
                "class_name": name_of.get(c, str(c)),
                "focal_index": -1,
                "position": float("nan"),
                "ld1": float(anchor_ld[c, 0]),
                "ld2": float(anchor_ld[c, 1]),
                "ld1_lo": float("nan"),
                "ld1_hi": float("nan"),
                "ld2_lo": float("nan"),
                "ld2_hi": float("nan"),
                "cov11": float(member_cov[c][0, 0]),
                "cov12": float(member_cov[c][0, 1]),
                "cov22": float(member_cov[c][1, 1]),
                "capture": float(observed.capture[c]),
            }
        )
        for j, position in enumerate(observed.focal_points):
            plane_rows.append(
                {
                    "kind": "focal",
                    "ref_class": c,
                    "class_name": name_of.get(c, str(c)),
                    "focal_index": j,
                    "position": float(position),
                    "ld1": float(observed.ld[c, j, 0]),
                    "ld2": float(observed.ld[c, j, 1]),
                    "ld1_lo": float(ld_lo[c, j, 0]),
                    "ld1_hi": float(ld_hi[c, j, 0]),
                    "ld2_lo": float(ld_lo[c, j, 1]),
                    "ld2_hi": float(ld_hi[c, j, 1]),
                    "cov11": float("nan"),
                    "cov12": float("nan"),
                    "cov22": float("nan"),
                    "capture": float("nan"),
                }
            )
    cache.save_frame(pd.DataFrame(plane_rows), ctx.path(f"trajectory_{axis}.parquet"))

    capture_rows: list[dict] = []
    for c in range(n_classes):
        capture_rows.append(
            {
                "ref_class": c,
                "class_name": name_of.get(c, str(c)),
                "capture": float(observed.capture[c]),
                "peak_position": float(observed.focal_points[observed.peak_focal[c]]),
                "endpoint_magnitude": float(observed.grain_magnitude["class"][c, focal_ref]),
                "endpoint_magnitude_lo": float(tube.grain_bands["class"][0, c, focal_ref]),
                "endpoint_magnitude_hi": float(tube.grain_bands["class"][2, c, focal_ref]),
                "mahalanobis": float(observed.mahalanobis[c, focal_ref]),
                "mahalanobis_lo": float(tube.mahalanobis_bands[0, c, focal_ref]),
                "mahalanobis_hi": float(tube.mahalanobis_bands[2, c, focal_ref]),
                "n_feature_reject": int(inference.reject[c].sum()),
                "n_feature_covers_zero": int(inference.covers_zero[c].sum()),
            }
        )
    cache.save_frame(pd.DataFrame(capture_rows), ctx.path(f"capture_{axis}.parquet"))


def _write_directional_artefacts(
    ctx,
    *,
    axis: str,
    name_of: dict,
    observed,
    directional,
    directional_inf,
    tube,
) -> None:
    """Write the DIREC tables of a local-trajectory run (plan sections 7e, 12b).

    Two aggregate artefacts: the per-class directional summary (the separation-scaled net-trend
    effect size with its clustered-bootstrap interval, the signed-slope interval, the two-sided
    bootstrap ``p``, the Benjamini-Hochberg decision across the four classes, the biased slope
    norm for context, and the descriptive single-break location with its bootstrap spread); and
    the per-class one-dimensional signed trajectory with its bootstrap band, the figure's series.
    """
    import numpy as np
    import pandas as pd

    n_classes = observed.displacement.shape[0]

    directional_rows: list[dict] = []
    for c in range(n_classes):
        directional_rows.append(
            {
                "ref_class": c,
                "class_name": name_of.get(c, str(c)),
                "net_trend": float(directional_inf.net_trend[c]),
                "net_trend_lo": float(directional_inf.net_trend_lo[c]),
                "net_trend_hi": float(directional_inf.net_trend_hi[c]),
                "signed_slope": float(directional_inf.signed_slope[c]),
                "signed_slope_lo": float(directional_inf.signed_slope_lo[c]),
                "signed_slope_hi": float(directional_inf.signed_slope_hi[c]),
                "slope_norm": float(directional.slope_norm[c]),
                "p_value": float(directional_inf.p_value[c]),
                "reject": bool(directional_inf.reject[c]),
                "break_position": float(directional_inf.break_position[c]),
                "break_lo": float(directional_inf.break_lo[c]),
                "break_hi": float(directional_inf.break_hi[c]),
            }
        )
    cache.save_frame(pd.DataFrame(directional_rows), ctx.path(f"directional_{axis}.parquet"))

    # The one-dimensional signed trajectory per class: the observed projection onto the frozen net
    # direction, with the clustered-bootstrap band (2.5 and 97.5 percentiles of the resampled
    # projection). This is what the DIREC figure draws.
    if tube.signed_trajectory is not None:
        band_lo = np.nanpercentile(tube.signed_trajectory, 2.5, axis=0)
        band_hi = np.nanpercentile(tube.signed_trajectory, 97.5, axis=0)
    else:  # pragma: no cover - the CLI always passes net directions
        band_lo = np.full_like(observed.grain_magnitude["class"], np.nan)
        band_hi = band_lo
    signed_rows: list[dict] = []
    for c in range(n_classes):
        for j, position in enumerate(observed.focal_points):
            signed_rows.append(
                {
                    "ref_class": c,
                    "class_name": name_of.get(c, str(c)),
                    "focal_index": j,
                    "position": float(position),
                    "signed": float(directional.signed_trajectory[c, j]),
                    "band_lo": float(band_lo[c, j]),
                    "band_hi": float(band_hi[c, j]),
                }
            )
    cache.save_frame(pd.DataFrame(signed_rows), ctx.path(f"signed_trajectory_{axis}.parquet"))


def _member_covariance(embedding, ref_labels, measurement_data, n_classes):
    """Return each class's member covariance in the first two discriminant axes, for the ellipse."""
    import numpy as np

    z_all = (
        measurement_data.loc[:, embedding.columns].to_numpy(dtype=float) - embedding.mean
    ) / embedding.sd
    member_ld = embedding.transformer.transform(z_all)[:, :2]
    labels = ref_labels.reindex(measurement_data.index).to_numpy()
    return [np.cov(member_ld[labels == c], rowvar=False) for c in range(n_classes)]


def _score_test_corroboration(root: Path, ref_fit_hash: str, axis: str) -> dict | None:
    """Return the cached score-based invariance run's headline for this fit and axis, if present.

    The bridge test is demoted to corroboration under this recast, so its minimum $p$-value and
    rejection count are carried alongside the effect size when a completed run exists.
    """
    for manifest_path in sorted((root / "artefacts" / "invariance").glob("*/manifest.json")):
        manifest = cache.read_manifest(manifest_path.parent) or {}
        params = manifest.get("params", {})
        if (
            manifest.get("status") == "ok"
            and params.get("fit") == ref_fit_hash
            and params.get("axis") == axis
        ):
            metrics = manifest.get("metrics", {})
            return {
                "run": cache.short_hash(manifest["hash"]),
                "min_p_max_lm": metrics.get("min_p_max_lm"),
                "n_reject_max_lm": metrics.get("n_reject_max_lm"),
                "n_blocks": metrics.get("n_blocks"),
            }
    return None


def invariance_responsibilities(model_obj, x_values):
    """Return the frozen posterior responsibilities of the measurement-only reference."""
    from analysis import invariance

    return invariance.responsibilities(model_obj, x_values)


@app.command()
def prevalence(
    axis: str = typer.Option(
        "era", help="Ordering variable: era or age_at_diagnosis (SPARK timing)."
    ),
    dataset: str = _DATASET,
    version: str = _VERSION,
    n_points: int = typer.Option(25, help="Grid points the predicted proportion curve is read at."),
    n_boot: int = typer.Option(
        500, "--n-boot", help="Family-clustered bootstrap replicates for the CIs, bands, and p."
    ),
    adjusted: bool = typer.Option(
        True,
        "--adjusted/--no-adjusted",
        help="Also read the axis slope net of sex, the lag, and age at evaluation.",
    ),
    seed: int = typer.Option(config.DEFAULT_STRATIFY_SEED, help="Base seed for the bootstrap."),
    n_init: int = typer.Option(
        50, help="The n_init of the measurement-only reference fit to resolve."
    ),
    q: float = typer.Option(0.05, help="Benjamini-Hochberg FDR level across the class contrasts."),
    min_bin_size: int = typer.Option(
        1000, help="Recovery floor the axis policy is derived against (does not bin the axis)."
    ),
    force: bool = _FORCE,
) -> None:
    """Test whether the frozen class proportions trend along an axis (PREV).

    The prevalence-drift test of ``.context/hypotheses.md``. The four classes are held fixed at
    the measurement-only reference fit (no mixture is refitted); their mixing proportions are
    regressed on the axis. The rigorous read is a maximum-likelihood three-step correction that
    removes the classify-analyse bias of a hard-label regression by fixing the classification-error
    matrix of the frozen posteriors; a naive hard-label multinomial logit is reported beside it as
    an uncorrected cross-check. It reports, per class, the one-versus-rest axis log-odds slope and
    odds ratio with a family-clustered bootstrap interval and $p$, the naive Wald and
    likelihood-ratio $p$-values, the joint likelihood-ratio test of ``class ~ axis`` against
    ``class ~ 1``, the predicted proportion curve with its band, an adjusted axis slope net of sex,
    the lag, and age at evaluation, and (for era) the DSM-5 pre/post-2013 contrast. Every artefact
    is class or coefficient level; no per-proband quantity is written.
    """
    import numpy as np
    import pandas as pd

    from analysis import drift as drift_mod
    from analysis import localise, profiling, strata_data
    from analysis import prevalence as prev
    from analysis.cohort import get_cohort
    from analysis.paths import run_dir

    if axis not in ("age_at_diagnosis", "era"):
        raise typer.BadParameter("axis must be 'age_at_diagnosis' or 'era'")

    def labels_series(frame: pd.DataFrame) -> pd.Series:
        others = [c for c in frame.columns if c != "class"]
        return frame.set_index(others[0])["class"] if others else frame["class"]

    root = find_repo_root()
    cohort_hash, _ = _run_cohort(root, dataset, version, force=force)
    matrix, typing = _load_cohort_matrix(root, cohort_hash, dataset, version)

    ref_fit_hash = _resolve_measurement_reference(root, cohort_hash, n_init, seed)
    ref_dir = run_dir(root, "fit", cache.short_hash(ref_fit_hash))
    reference_model = cache.load_model(ref_dir / "model.joblib")
    ref_labels = labels_series(cache.load_frame(ref_dir / "labels.parquet"))

    measurement_data, _descriptor, _covariates = model.prepare_inputs(matrix, typing)
    x_values = measurement_data.to_numpy(dtype=float)
    responsibilities = invariance_responsibilities(reference_model, x_values)
    hard_labels = pd.Series(responsibilities.argmax(axis=1), index=measurement_data.index)

    reference = drift_mod.build_reference(measurement_data, ref_labels)
    name_of, name_source = _measurement_class_names(
        root, dataset, version, cohort_hash, reference, measurement_data, seed
    )
    n_classes = int(reference.centroids.shape[0])

    cohort = get_cohort(dataset, version, root)
    resolved = cohort.axis(axis, matrix.features.index, matrix.covariates, min_bin_size)
    if resolved is None:
        raise typer.BadParameter(f"cohort {dataset!r} does not provide axis {axis!r}")
    axis_values, _policy = resolved
    families = cohort.family_ids(matrix.features.index)
    if families is None:
        raise typer.BadParameter(
            f"cohort {dataset!r} exposes no family identifier for the clustered bootstrap"
        )

    # The adjustment covariates: sex and age at evaluation from the cohort, the
    # diagnosis-to-measurement lag from the strata-describe machinery (never re-derived here).
    strata = strata_data.build_strata_data(
        root,
        version,
        matrix.features.index,
        matrix.covariates["age_at_eval_years"],
        matrix.covariates["sex"],
    )
    axis_series = axis_values.reindex(measurement_data.index)
    sex = matrix.covariates["sex"].reindex(measurement_data.index).astype(float)
    age_at_eval = (
        matrix.covariates["age_at_eval_years"].reindex(measurement_data.index).astype(float)
    )
    lag = strata.lag.reindex(measurement_data.index).astype(float)

    # Keep probands with a finite axis value and (for the adjusted model) finite covariates.
    finite_axis = axis_series.notna().to_numpy()
    finite_cov = (sex.notna() & age_at_eval.notna() & lag.notna()).to_numpy()
    covered = finite_axis & finite_cov
    n_reference = int(x_values.shape[0])
    n_covered = int(covered.sum())

    resp_cov = responsibilities[covered]
    labels_cov = hard_labels.to_numpy()[covered]
    axis_cov = axis_series[covered].to_numpy(dtype=float)
    fam_cov = families.reindex(measurement_data.index)[covered].to_numpy()
    covariate_arrays = {
        "sex": sex[covered].to_numpy(dtype=float),
        "lag": lag[covered].to_numpy(dtype=float),
        "age_at_eval": age_at_eval[covered].to_numpy(dtype=float),
    }

    grid = np.asarray(
        localise.focal_grid(pd.Series(axis_cov), n_points, (0.025, 0.975)), dtype=float
    )

    params = {
        "fit": ref_fit_hash,
        "names": name_source,
        "axis": axis,
        "estimand": "mixing-proportions;frozen-responsibilities",
        "method": "ml-3step-correction;naive-hardlabel-crosscheck;v1",
        "covariates": ["sex", "lag", "age_at_eval"] if adjusted else [],
        "dsm5_contrast": axis == "era",
        "n_points": n_points,
        "n_boot": n_boot,
        "seed": seed,
        "q": q,
    }
    with run_context("prevalence", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            cached = (cache.read_manifest(ctx.run_dir) or {}).get("metrics", {})
            typer.echo(f"prevalence: cache hit {cache.short_hash(ctx.run_hash)}; {cached}")
            return
        with profiling.measure() as unit:
            result = prev.prevalence_analysis(
                resp_cov,
                labels_cov,
                axis_cov,
                fam_cov,
                axis=axis,
                covariates=covariate_arrays if adjusted else None,
                grid=grid,
                n_boot=n_boot,
                seed=seed,
                q=q,
            )
            _write_prevalence_artefacts(ctx, axis=axis, name_of=name_of, result=result)
            unit.output_bytes = profiling.path_bytes(ctx.path(f"slopes_{axis}.parquet"))

        resources = unit.metrics
        assert resources is not None
        ctx.metrics = {
            "axis": axis,
            "reference_fit": cache.short_hash(ref_fit_hash),
            "n_reference": n_reference,
            "n_covered": n_covered,
            "coverage": round(n_covered / n_reference, 4),
            "n_boot": n_boot,
            "pooled_proportion_by_class": [round(float(v), 4) for v in result.curve.pooled],
            "corrected_slope_by_class": [round(s.slope, 4) for s in result.corrected_slopes],
            "corrected_boot_p_by_class": [round(s.boot_p, 4) for s in result.corrected_slopes],
            "corrected_reject_by_class": [bool(s.reject) for s in result.corrected_slopes],
            "n_corrected_reject": int(sum(s.reject for s in result.corrected_slopes)),
            "naive_slope_by_class": [round(s.slope, 4) for s in result.naive_slopes],
            "naive_wald_p_by_class": [round(s.wald_p, 4) for s in result.naive_slopes],
            "joint_lrt": {
                t.estimator: {"lr_stat": round(t.lr_stat, 3), "df": t.df, "p": round(t.p_value, 6)}
                for t in result.joint_tests
            },
            "resources": resources.to_dict(),
        }
    typer.echo(f"prevalence {axis}: run {cache.short_hash(ctx.run_hash)}")
    typer.echo(
        f"  coverage {n_covered}/{n_reference} ({n_covered / n_reference:.1%}); n_boot {n_boot}"
    )
    typer.echo(
        "  per-class corrected slope (log-odds/axis-unit): "
        + ", ".join(
            f"{name_of.get(c, c)} {result.corrected_slopes[c].slope:+.3f} "
            f"(OR {result.corrected_slopes[c].odds_ratio:.3f}) "
            f"[{result.corrected_slopes[c].ci_low:+.3f},{result.corrected_slopes[c].ci_high:+.3f}] "
            f"p={result.corrected_slopes[c].boot_p:.3f} "
            f"{'PREV' if result.corrected_slopes[c].reject else 'ns'}"
            for c in range(n_classes)
        )
    )
    joint = {t.estimator: t for t in result.joint_tests}
    typer.echo(
        f"  joint LRT class~axis vs class~1: corrected chi2={joint['corrected'].lr_stat:.1f} "
        f"(df {joint['corrected'].df}) p={joint['corrected'].p_value:.3g}; "
        f"naive p={joint['naive'].p_value:.3g}"
    )


def _write_prevalence_artefacts(ctx, *, axis: str, name_of: dict, result) -> None:
    """Write the class and coefficient-level tables of a prevalence run.

    Three aggregate artefacts: the per-class slope table (corrected, naive, adjusted, and, for
    era, the DSM-5 contrast, each with its interval and test); the predicted proportion curves
    with the corrected band; and the joint likelihood-ratio tests. Nothing is per proband.
    """
    import pandas as pd

    def slope_rows(slopes, kind: str) -> list[dict]:
        rows = []
        for s in slopes:
            rows.append(
                {
                    "axis": axis,
                    "kind": kind,
                    "ref_class": s.ref_class,
                    "class_name": name_of.get(s.ref_class, str(s.ref_class)),
                    "slope": float(s.slope),
                    "odds_ratio": float(s.odds_ratio),
                    "ci_low": float(s.ci_low),
                    "ci_high": float(s.ci_high),
                    "wald_p": float(s.wald_p),
                    "lrt_p": float(s.lrt_p),
                    "boot_p": float(s.boot_p),
                    "reject": bool(s.reject),
                }
            )
        return rows

    rows: list[dict] = []
    rows += slope_rows(result.corrected_slopes, "corrected")
    rows += slope_rows(result.naive_slopes, "naive")
    rows += slope_rows(result.adjusted_slopes, "adjusted")
    rows += slope_rows(result.dsm_contrasts, "dsm5_contrast")
    cache.save_frame(pd.DataFrame(rows), ctx.path(f"slopes_{axis}.parquet"))

    curve = result.curve
    curve_rows: list[dict] = []
    n_classes = curve.corrected.shape[0]
    for c in range(n_classes):
        for j, position in enumerate(curve.positions):
            curve_rows.append(
                {
                    "axis": axis,
                    "ref_class": c,
                    "class_name": name_of.get(c, str(c)),
                    "position": float(position),
                    "corrected": float(curve.corrected[c, j]),
                    "naive": float(curve.naive[c, j]),
                    "band_lo": float(curve.band_lo[c, j]),
                    "band_hi": float(curve.band_hi[c, j]),
                    "pooled": float(curve.pooled[c]),
                }
            )
    cache.save_frame(pd.DataFrame(curve_rows), ctx.path(f"proportion_curve_{axis}.parquet"))

    joint_rows = [
        {
            "axis": axis,
            "estimator": t.estimator,
            "lr_stat": float(t.lr_stat),
            "df": int(t.df),
            "p_value": float(t.p_value),
        }
        for t in result.joint_tests
    ]
    cache.save_frame(pd.DataFrame(joint_rows), ctx.path(f"joint_test_{axis}.parquet"))


@app.command(rich_help_panel=_PLANNED)
def sensitivity() -> None:
    """Re-fit under alternative feature sets and within cognitive-level strata."""
    _todo("sensitivity", 5)


@app.command(rich_help_panel=_PLANNED)
def report() -> None:
    """Assemble the non-disclosive tables and figures for the manuscript."""
    _todo("report", 7)


if __name__ == "__main__":
    app()
