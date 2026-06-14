"""analysis command-line interface (Typer).

One subcommand per pipeline stage. Each implemented stage reads named inputs and writes
named outputs plus a manifest under ``artefacts/<stage>/<run-hash>/``, so a later run
recomputes only what changed (plan section 11). Stages not yet written are grouped under a
"planned" panel in ``analysis --help`` and exit non-zero; a command leaves that panel when
its stage is implemented.
"""

from __future__ import annotations

from pathlib import Path

import typer

from analysis import cache, config, features, model
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


def _cohort_params(root: Path, dataset: str, version: str) -> dict[str, object]:
    """Return the hashing parameters for the cohort (and typing) stage."""
    typing_dir = config.litman_typing_dir(root)
    return {
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


def _run_cohort(root: Path, dataset: str, version: str, *, force: bool) -> tuple[str, dict]:
    """Build (or load) the cohort matrix and typing, returning the run hash and metrics."""
    params = _cohort_params(root, dataset, version)
    with run_context("cohort", params, root=root, force=force) as ctx:
        if ctx.cache_hit:
            manifest = cache.read_manifest(ctx.run_dir) or {}
            return ctx.run_hash, manifest.get("metrics", {})
        feature_names = load_feature_list(config.author_feature_list(root))
        cohort = get_cohort(dataset, version, root)
        integrated = cohort.integrate()
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


@app.command()
def cohort(
    dataset: str = _DATASET,
    version: str = _VERSION,
    force: bool = _FORCE,
) -> None:
    """Build the harmonised proband-by-feature matrix and its typing manifest."""
    root = find_repo_root()
    run_hash, metrics = _run_cohort(root, dataset, version, force=force)
    typer.echo(f"cohort {dataset}/{version}: run {cache.short_hash(run_hash)}")
    typer.echo(f"  probands={metrics['n_probands']} features={metrics['n_features']}")
    typer.echo(f"  typing={metrics['typing_counts']} conflicts={metrics['typing_conflicts']}")


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


@app.command()
def fit(
    dataset: str = _DATASET,
    version: str = _VERSION,
    n_components: int = typer.Option(config.DEFAULT_N_COMPONENTS, help="Number of latent classes."),
    n_init: int = typer.Option(config.DEFAULT_N_INIT, help="Random restarts (StepMix n_init)."),
    seed: int = typer.Option(0, help="Random seed for reproducible restarts."),
    force: bool = _FORCE,
) -> None:
    """Fit the reference general finite mixture model and predict class labels."""
    root = find_repo_root()
    cohort_hash, _ = _run_cohort(root, dataset, version, force=force)
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
    force: bool = _FORCE,
) -> None:
    """Compute the seven-category signature and align our classes to Litman's named classes."""
    from analysis import enrich, reference
    from analysis.paths import run_dir

    root = find_repo_root()
    cohort_hash, _ = _run_cohort(root, dataset, version, force=force)
    fit_hash = cache.compute_hash(_fit_params(cohort_hash, n_components, n_init, seed))
    fit_dir = run_dir(root, "fit", cache.short_hash(fit_hash))
    if cache.read_manifest(fit_dir) is None:
        typer.echo("no fit found for these settings; run `analysis fit` first", err=True)
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

        cache.save_frame(enrichment, ctx.path("enrichment.parquet"))
        cache.save_frame(signature, ctx.path("signature.parquet"))
        cache.save_json(
            {
                "mapping": {str(k): v for k, v in named.mapping.items()},
                "correlations": {str(k): v for k, v in named.correlations.items()},
                "overall_correlation": named.overall_correlation,
                "anchors": named.anchors,
                "anchors_hold": named.anchors_hold,
            },
            ctx.path("alignment.json"),
        )
        ctx.metrics = {
            "mapping": {str(k): v for k, v in named.mapping.items()},
            "anchors_hold": named.anchors_hold,
            "overall_correlation": round(named.overall_correlation, 4),
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
    typer.echo(f"  anchors hold: {named.anchors_hold} {named.anchors}")


@app.command(rich_help_panel=_PLANNED)
def select() -> None:
    """Grid over the number of components and report the information criteria."""
    _todo("select", 2)


@app.command(rich_help_panel=_PLANNED)
def stability() -> None:
    """Summarise multi-initialisation and subsampling stability of the reference fit."""
    _todo("stability", 2)


@app.command(rich_help_panel=_PLANNED)
def replicate() -> None:
    """Project the reference model onto the SSC and correlate the category profiles."""
    _todo("replicate", 2)


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
