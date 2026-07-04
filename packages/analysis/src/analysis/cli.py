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
    force: bool = _FORCE,
) -> None:
    """Fit the reference general finite mixture model and predict class labels."""
    root = find_repo_root()
    cohort_hash, _ = _run_cohort(
        root, dataset, version, force=force, as_of=as_of, sample_n=sample_n, sample_seed=sample_seed
    )
    params = _fit_params(cohort_hash, n_components, n_init, seed)
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
            matrix, typing, n_components=n_components, n_init=n_init, random_state=seed
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
    axis: str = typer.Option("age_at_diagnosis", help="Axis: age_at_diagnosis or era."),
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
    before the full run.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import pandas as pd

    from analysis import checkpoint, profiling, strata_data
    from analysis import strata as strata_mod
    from analysis.progress import task_bar

    if axis not in ("age_at_diagnosis", "era"):
        raise typer.BadParameter("axis must be 'age_at_diagnosis' or 'era'")

    root = find_repo_root()
    cohort_hash, _ = _run_cohort(root, dataset, version, force=force)
    matrix, typing = _load_cohort_matrix(root, cohort_hash, dataset, version)

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
    order = assignment.labels

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
                    store.append(drift_mod.serialise_summary(future.result(), perm, s))
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
    axis: str = typer.Option("age_at_diagnosis", help="Axis: age_at_diagnosis or era."),
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

    from analysis import checkpoint, model, strata_data
    from analysis import drift as drift_mod
    from analysis import strata as strata_mod
    from analysis.cohort import CohortMatrix
    from analysis.paths import run_dir

    if axis not in ("age_at_diagnosis", "era"):
        raise typer.BadParameter("axis must be 'age_at_diagnosis' or 'era'")
    if alignment not in drift_mod.ALIGNMENTS:
        raise typer.BadParameter(f"alignment must be one of {sorted(drift_mod.ALIGNMENTS)}")
    if distance not in drift_mod.DISTANCES:
        raise typer.BadParameter(f"distance must be one of {sorted(drift_mod.DISTANCES)}")

    def labels_series(frame: pd.DataFrame) -> pd.Series:
        others = [c for c in frame.columns if c != "class"]
        return frame.set_index(others[0])["class"] if others else frame["class"]

    root = find_repo_root()
    cohort_hash, _ = _run_cohort(root, dataset, version, force=force)
    matrix, typing = _load_cohort_matrix(root, cohort_hash, dataset, version)

    ref_fit_hash = cache.compute_hash(
        _fit_params(cohort_hash, config.DEFAULT_N_COMPONENTS, n_init, seed)
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
        null_drift: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for rec in null_store.load():
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
        }
    typer.echo(f"drift {axis} ({alignment}/{distance}): run {cache.short_hash(ctx.run_hash)}")
    typer.echo(
        f"  {ctx.metrics['n_drift']}/{ctx.metrics['n_tests']} classes drift beyond the "
        f"N={n_permutations} null (BH q=0.05); {ctx.metrics['n_reorganised']} reorganised "
        f"(low overlap); separation {separation:.3f}"
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
    import numpy as np
    import pandas as pd

    from analysis import drift as drift_mod
    from analysis import model, strata_data, trajectory
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
        _fit_params(cohort_hash, config.DEFAULT_N_COMPONENTS, n_init, seed)
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
    stratify_dir = run_dir(root, "stratify", cache.short_hash(cache.compute_hash(stratify_params)))

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
            alignment = drift_mod.ALIGNMENTS["membership"].align(summary, reference)
            inverse = {ref: fit for fit, ref in alignment.mapping.items()}
            if len(inverse) != n_classes:
                raise typer.BadParameter(
                    f"stratum {label} did not recover {n_classes} classes; cannot build a "
                    "trajectory for it"
                )
            for c in range(n_classes):
                fit_class = inverse[c]
                aligned[c].append(summary.centroids.loc[fit_class])
                sizes[c].append(int(summary.contingency.loc[fit_class].sum()))
                jaccard[c].append(float(alignment.quality.get(c, float("nan"))))

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
        _fit_params(cohort_hash, config.DEFAULT_N_COMPONENTS, n_init, seed)
    )
    ref_dir = run_dir(root, "fit", cache.short_hash(ref_fit_hash))
    if not (ref_dir / "labels.parquet").is_file():
        raise typer.BadParameter(f"no reference fit at {ref_dir}; run `analysis fit`")
    ref_labels = labels_series(cache.load_frame(ref_dir / "labels.parquet"))
    measurement_data, _descriptor, _covariates = model.prepare_inputs(matrix, typing)
    reference = drift_mod.build_reference(measurement_data, ref_labels)

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
                moved = attr.movers(movement)
                result = contraster.contrast(moved, stratum_md.loc[moved.index])
                for record in result.importances.to_dict("records"):
                    mover_rows.append({"stratum": label, "ref_class": movement.ref_class, **record})
                top_shift = contributions.abs().idxmax() if len(contributions) else None
                top_mover = (
                    result.importances.iloc[0]["feature"] if len(result.importances) else None
                )
                total = result.n_movers + result.n_stayers
                summary_rows.append(
                    {
                        "stratum": label,
                        "ref_class": movement.ref_class,
                        "n_movers": result.n_movers,
                        "n_stayers": result.n_stayers,
                        "mover_fraction": result.n_movers / total if total else float("nan"),
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
            "mean_mover_fraction": (
                float(summary_frame["mover_fraction"].mean())
                if len(summary_frame)
                else float("nan")
            ),
        }
    typer.echo(
        f"attribute {axis} ({decomposition}/{contrast}): run {cache.short_hash(ctx.run_hash)}"
    )
    typer.echo(
        f"  {ctx.metrics['n_strata']} strata x {ctx.metrics['n_classes']} classes; "
        f"mean mover fraction {ctx.metrics['mean_mover_fraction']:.2f}"
    )


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
