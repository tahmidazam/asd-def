"""analysis command-line interface (Typer).

One subcommand per pipeline stage. Each implemented stage reads named inputs and writes
named outputs plus a manifest under ``artefacts/<stage>/<run-hash>/``, so a later run
recomputes only what changed (plan section 11). Stages not yet written are grouped under a
"planned" panel in ``analysis --help`` and exit non-zero; a command leaves that panel when
its stage is implemented.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import typer

from analysis import cache, checkpoint, config, features, model
from analysis.cohort import build_matrix, get_cohort
from analysis.cohort.schema import load_feature_list
from analysis.features import Typing
from analysis.paths import find_repo_root
from analysis.run import run_context

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
    params = {
        "spark_cohort": spark_hash,
        "ssc_version": ssc_version,
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
        ssc = get_cohort("ssc", ssc_version, root)
        ssc_integrated = ssc.integrate()
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


@app.command(rich_help_panel=_PLANNED)
def strata() -> None:
    """Assign each proband to an age-at-diagnosis and a diagnostic-era stratum."""
    _todo("strata", 4)


@app.command(rich_help_panel=_PLANNED)
def stratify() -> None:
    """Re-estimate the model independently within each stratum of an axis."""
    _todo("stratify", 4)


@app.command(rich_help_panel=_PLANNED)
def drift() -> None:
    """Align stratum classes to the reference and measure drift against the null."""
    _todo("drift", 4)


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
