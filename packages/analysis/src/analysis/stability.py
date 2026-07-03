"""Multi-initialisation and subsampling stability, and the minimum viable stratum size.

Reproduces the released stability and subsampling analyses (plan section 6, step 5) and
reuses the same machinery to fix the minimum stratum size for the stratified work (plan
section 7b).

Multi-initialisation stability runs many single-initialisation fits from different random
starts, ranks them by log-likelihood, and compares the best of them to the reference
solution. Subsampling stability refits on random halves of the cohort and compares each
back to the reference. The comparison is threefold: the seven-category profile correlation
(the authors' own measure), the class-overlap matrix, and the adjusted Rand index, which the
plan adds to the released overlap because it is label-invariant and chance-corrected (plan
section 6, deviations). Same-sample comparisons align class labels with the released greedy
overlap rule (:func:`analysis.align.greedy_overlap_align`); the Rand index needs no
alignment.

The released code runs 2,000 single-init fits and reports the best 100, and refits on 100
halves; both counts are configurable here. Litman et al. do not seed these fits; we seed
them for reproducibility (plan section 11), the only deliberate divergence.

The minimum viable stratum size is found by refitting at descending sample sizes and
recording where four-class recovery degrades: the smallest class proportion, the scaled
relative entropy, the average latent-class posterior probability, and the profile
correlation to the full-sample reference. The size below which the profile correlation falls
past the reproduction benchmark is the floor for the stratification bins.

A fit can fail to converge when the structural covariate M-step meets a near-singular design,
most often at higher class counts on a small subsample. Rather than abort the stage, a failed
fit is recorded as missing (``_try_fit``): the multi-initialisation run drops it before
ranking, and the subsampling and minimum-size sweeps mark its replicate degenerate, so it
falls out of the aggregate means.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import adjusted_rand_score
from stepmix.stepmix import StepMix

from analysis import config, profiling
from analysis.align import greedy_overlap_align
from analysis.checkpoint import SUFFIX, CheckpointLog
from analysis.cohort import CohortMatrix
from analysis.enrich import (
    SEVEN_CATEGORIES,
    category_signature,
    contributory_features,
    feature_enrichment,
    profile_correlation,
)
from analysis.features import REVERSE_CODED_SCQ, Typing
from analysis.model import prepare_inputs
from analysis.progress import task_bar


def _n_workers(workers: int) -> int:
    """Return the worker-process count: ``workers`` if positive, else logical cores minus one."""
    return workers or max(1, (os.cpu_count() or 2) - 1)


@dataclass
class Comparison:
    """One fit's comparison to the reference solution.

    Attributes
    ----------
    overall_correlation : float or None
        Pearson correlation over the flattened class-by-category profiles, or ``None`` when
        a profile is near-constant.
    category_correlation : dict of str to float or None
        Per-category Pearson correlation across classes.
    adjusted_rand_index : float
        Chance-corrected agreement between the fit and reference labellings.
    smallest_class_proportion : float
        The smallest class proportion in the aligned fit, counting a collapsed class as
        zero (so a four-to-three collapse reads as a proportion near zero, not the smallest
        surviving class).
    overlap : pandas.DataFrame
        The source-by-reference class-overlap matrix.
    degenerate : bool
        Whether the fit collapsed a class (recovered fewer than ``n_components`` classes), in
        which case the profile correlation is undefined and the fit is dropped from the
        aggregate means, as in the released code.
    """

    overall_correlation: float | None
    category_correlation: dict[str, float | None]
    adjusted_rand_index: float
    smallest_class_proportion: float
    overlap: pd.DataFrame
    degenerate: bool


@dataclass
class StabilitySummary:
    """The result of a multi-initialisation or subsampling stability run.

    Attributes
    ----------
    fits : pandas.DataFrame
        One row per fit with its seed, average log-likelihood, and convergence flag.
    comparisons : pandas.DataFrame
        One row per compared fit with the overall and per-category profile correlations, the
        adjusted Rand index, and the smallest class proportion.
    overlap_mean : pandas.DataFrame
        The mean class-overlap matrix over the compared fits (source class on rows, reference
        class on columns).
    aggregate : dict
        Mean and standard deviation of the overall correlation and the adjusted Rand index,
        and the mean per-category correlation.
    """

    fits: pd.DataFrame
    comparisons: pd.DataFrame
    overlap_mean: pd.DataFrame
    aggregate: dict[str, object] = field(default_factory=dict)


@dataclass
class NminResult:
    """The minimum-viable-stratum-size sweep.

    Attributes
    ----------
    per_fit : pandas.DataFrame
        One row per (target size, replicate) with the recovery metrics.
    summary : pandas.DataFrame
        Mean recovery metrics per target size.
    n_min : int or None
        The smallest target size whose mean profile correlation holds at or above the
        benchmark, or ``None`` when no swept size clears it. Kept for continuity; it reads one
        clearing size off a possibly non-monotone curve, so prefer ``floor`` below.
    benchmark : float
        The profile-correlation threshold used.
    floor : int or None
        The recovery floor from a monotone (isotonic) fit of correlation against log-size: the
        smallest size at which the fitted recovery reaches the benchmark. ``None`` when the
        fitted curve does not reach the benchmark anywhere in the swept range. Pools every fit,
        so it is robust to the scatter a small replicate count produces.
    floor_ci : tuple of int or None
        The 90 per cent bootstrap confidence interval (lower, upper) for ``floor``, or ``None``
        when too few resamples cross to form one. The upper bound is the conservative bin floor.
    """

    per_fit: pd.DataFrame
    summary: pd.DataFrame
    n_min: int | None
    benchmark: float
    floor: int | None
    floor_ci: tuple[int, int] | None


def _fit(
    measurement: pd.DataFrame,
    covariates: pd.DataFrame,
    descriptor: dict,
    *,
    n_components: int,
    n_init: int,
    seed: int,
) -> StepMix:
    """Fit one StepMix GFMM on a prepared measurement and covariate matrix."""
    model = StepMix(
        n_components=n_components,
        measurement=descriptor,
        structural="covariate",
        n_steps=config.DEFAULT_N_STEPS,
        n_init=n_init,
        random_state=seed,
        progress_bar=0,
        verbose=0,
    )
    model.fit(measurement, covariates)
    return model


def _try_fit(
    measurement: pd.DataFrame,
    covariates: pd.DataFrame,
    descriptor: dict,
    *,
    n_components: int,
    n_init: int,
    seed: int,
) -> StepMix | None:
    """Fit a model, returning ``None`` when the structural M-step fails to converge.

    The covariate emission inverts a Hessian through the pseudo-inverse, which raises
    :class:`numpy.linalg.LinAlgError` (SVD non-convergence) when the design is near-singular,
    most often at higher class counts on a small subsample. A fit with few or no restarts has
    nothing to fall back on, so the caller records that fit as missing rather than letting one
    failure crash the whole stage, mirroring how the cross-validated selection pass already
    degrades a failed fold to ``nan``.
    """
    try:
        return _fit(
            measurement, covariates, descriptor, n_components=n_components, n_init=n_init, seed=seed
        )
    except LinAlgError:
        return None


def _labels(model: StepMix, measurement: pd.DataFrame) -> pd.Series:
    """Return the hard labels from the measurement posterior (no covariates)."""
    return pd.Series(model.predict(measurement), index=measurement.index, name="class")


def class_overlap_matrix(source: pd.Series, target: pd.Series, n_components: int) -> pd.DataFrame:
    """Return the class-overlap matrix between two labellings of shared probands.

    Cell ``(k, j)`` is the proportion of reference class ``j`` whose probands fall in source
    class ``k``. After the source labels are aligned to the reference, the diagonal is the
    retention of each class.

    Parameters
    ----------
    source : pandas.Series
        Source class label per proband (aligned to the reference class ids).
    target : pandas.Series
        Reference class label per proband, on the same index.
    n_components : int
        Number of classes.

    Returns
    -------
    pandas.DataFrame
        Source-by-reference overlap proportions.
    """
    matrix = np.full((n_components, n_components), np.nan)
    for j in range(n_components):
        target_index = target.index[target == j]
        denominator = len(target_index)
        if not denominator:
            continue
        for k in range(n_components):
            shared = len(source.index[source == k].intersection(target_index))
            matrix[k, j] = shared / denominator
    return pd.DataFrame(
        matrix, index=pd.Index(range(n_components), name="source"), columns=range(n_components)
    )


def compare_to_reference(
    measurement: pd.DataFrame,
    fit_labels: pd.Series,
    reference_labels: pd.Series,
    reference_enrichment: pd.DataFrame,
    category_map: dict[str, str],
    *,
    n_components: int,
    reverse_coded: tuple[str, ...] = REVERSE_CODED_SCQ,
) -> Comparison:
    """Compare one fit's labelling to the reference on shared probands.

    The fit labels are aligned to the reference by greedy overlap, the seven-category
    signature is recomputed on the aligned labels, and three statistics are returned: the
    overall and per-category profile correlations against the reference signature, the
    adjusted Rand index (on the raw labels, which is label-invariant), and the class-overlap
    matrix. The contributory feature set is taken from the reference enrichment and applied to
    both signatures, so the correlation is computed over the same feature universe the authors
    used (plan section 6, step 7).

    Parameters
    ----------
    measurement : pandas.DataFrame
        The measurement matrix the fit was made on (full cohort or subsample).
    fit_labels : pandas.Series
        The fit's hard labels.
    reference_labels : pandas.Series
        The reference solution's hard labels.
    reference_enrichment : pandas.DataFrame
        The reference solution's per-feature enrichment, from which the reference signature
        and the contributory feature set are derived.
    category_map : dict of str to str
        Feature-to-category map for the signature.
    n_components : int
        Number of classes.
    reverse_coded : tuple of str, optional
        SCQ items whose enrichment direction is flipped before the signature.

    Returns
    -------
    Comparison
        The overall and per-category profile correlations, the adjusted Rand index, the
        smallest class proportion, the class-overlap matrix, and a degenerate-fit flag.
    """
    shared = fit_labels.index.intersection(reference_labels.index)
    fit_shared = fit_labels.loc[shared]
    reference_shared = reference_labels.loc[shared]

    mapping = greedy_overlap_align(fit_shared, reference_shared)
    aligned = fit_shared.map(mapping).astype(int)

    keep = set(contributory_features(reference_enrichment, n_components))
    reference_signature = category_signature(
        reference_enrichment,
        category_map,
        n_classes=n_components,
        reverse_coded=reverse_coded,
        keep=keep,
    )

    degenerate = bool(aligned.nunique() < n_components)
    if degenerate:
        # A degenerate fit collapsed a class, so the seven-category profile cannot be
        # computed (a class has no probands to enrich). The correlation is recorded as
        # undefined; the Rand index and overlap still describe the partition, and this is
        # exactly the recovery breakdown the nmin sweep is meant to detect.
        overall: float | None = None
        per_category: dict[str, float | None] = dict.fromkeys(SEVEN_CATEGORIES, None)
    else:
        enrichment = feature_enrichment(measurement.loc[shared], aligned, n_classes=n_components)
        signature = category_signature(
            enrichment, category_map, n_classes=n_components, reverse_coded=reverse_coded, keep=keep
        )
        overall, per_category = profile_correlation(signature, reference_signature)

    counts = aligned.value_counts().reindex(range(n_components), fill_value=0)
    smallest = float(counts.min() / counts.sum())
    return Comparison(
        overall_correlation=overall,
        category_correlation=per_category,
        adjusted_rand_index=float(
            adjusted_rand_score(reference_shared.to_numpy(), fit_shared.to_numpy())
        ),
        smallest_class_proportion=smallest,
        overlap=class_overlap_matrix(aligned, reference_shared, n_components),
        degenerate=degenerate,
    )


def _aggregate(
    comparisons: pd.DataFrame, overlaps: list[pd.DataFrame]
) -> tuple[dict[str, object], pd.DataFrame]:
    """Summarise the per-fit comparisons into means, standard deviations, and a mean overlap.

    Degenerate fits (a collapsed class) are dropped from every aggregate, as in the released
    code, so the correlation, the adjusted Rand index, and the overlap all describe the same
    set of well-formed fits. The degenerate fits stay in the per-fit table for transparency,
    and their count is reported.
    """
    valid = comparisons[~comparisons["degenerate"]]
    category_means = {cat: float(valid[f"{cat}_r"].mean(skipna=True)) for cat in SEVEN_CATEGORIES}
    aggregate: dict[str, object] = {
        "n_compared": int(len(comparisons)),
        "n_valid": int(len(valid)),
        "n_degenerate": int(comparisons["degenerate"].sum()),
        "overall_correlation_mean": float(valid["overall_correlation"].mean(skipna=True)),
        "overall_correlation_std": float(valid["overall_correlation"].std(skipna=True)),
        "adjusted_rand_index_mean": float(valid["adjusted_rand_index"].mean(skipna=True)),
        "adjusted_rand_index_std": float(valid["adjusted_rand_index"].std(skipna=True)),
        "category_correlation_mean": category_means,
    }
    overlap_mean = pd.concat(overlaps).groupby(level=0).mean() if overlaps else pd.DataFrame()
    return aggregate, overlap_mean


def _comparison_row(seed: int, avg_ll: float, comparison: Comparison) -> dict[str, object]:
    """Flatten a comparison into one summary row."""
    row: dict[str, object] = {
        "seed": seed,
        "avg_log_likelihood": avg_ll,
        "overall_correlation": comparison.overall_correlation,
        "adjusted_rand_index": comparison.adjusted_rand_index,
        "smallest_class_proportion": comparison.smallest_class_proportion,
        "degenerate": comparison.degenerate,
    }
    for cat in SEVEN_CATEGORIES:
        row[f"{cat}_r"] = comparison.category_correlation[cat]
    return row


def _overlap_payload(comparison: Comparison) -> list[list[float]] | None:
    """Serialise a comparison's overlap matrix for a checkpoint, or ``None`` if degenerate.

    A degenerate fit contributes no overlap to the aggregate (a collapsed class has no
    probands to overlap), matching the in-memory path that skips it.
    """
    if comparison.degenerate:
        return None
    return comparison.overlap.to_numpy().tolist()


def _overlap_from_payload(data: list[list[float]], n_components: int) -> pd.DataFrame:
    """Rebuild an overlap matrix from its checkpointed nested list.

    The frame matches :func:`class_overlap_matrix`: a source-class index named ``source`` and
    integer reference-class columns, so the rebuilt matrices aggregate identically.
    """
    return pd.DataFrame(
        np.array(data, dtype=float),
        index=pd.Index(range(n_components), name="source"),
        columns=range(n_components),
    )


def _failed_comparison_row(seed: int) -> dict[str, object]:
    """Return a stand-in comparison row for a fit that did not converge.

    Marked degenerate so it is dropped from the aggregate means, the same as a fit that
    collapsed a class, with its correlations and Rand index left undefined.
    """
    row: dict[str, object] = {
        "seed": seed,
        "avg_log_likelihood": float("nan"),
        "overall_correlation": None,
        "adjusted_rand_index": float("nan"),
        "smallest_class_proportion": float("nan"),
        "degenerate": True,
    }
    for cat in SEVEN_CATEGORIES:
        row[f"{cat}_r"] = None
    return row


def _run_multi_init_fit(
    measurement: pd.DataFrame,
    covariates: pd.DataFrame,
    descriptor: dict,
    *,
    n_components: int,
    seed: int,
) -> tuple[dict[str, object], pd.Series | None]:
    """Fit one single-initialisation stability fit and return its record and labels.

    A top-level function so it pickles for a process pool. Labels are returned (not just the
    scalar record) so the caller can still skip refitting a seed reused in the comparison
    phase, matching the in-memory cache a serial run builds for free.
    """
    model = _try_fit(
        measurement, covariates, descriptor, n_components=n_components, n_init=1, seed=seed
    )
    if model is None:
        return {"seed": seed, "avg_log_likelihood": float("nan"), "converged": False}, None
    avg_ll = float(model.score(measurement, covariates))
    labels = _labels(model, measurement)
    record: dict[str, object] = {
        "seed": seed,
        "avg_log_likelihood": avg_ll,
        "converged": bool(model.converged_),
    }
    return record, labels


def _run_comparison(
    measurement: pd.DataFrame,
    covariates: pd.DataFrame,
    descriptor: dict,
    reference_labels: pd.Series,
    reference_enrichment: pd.DataFrame,
    category_map: dict[str, str],
    *,
    n_components: int,
    seed: int,
    avg_ll: float,
    labels: pd.Series | None,
) -> dict[str, object]:
    """Compare one fit's labels to the reference, refitting first if not already known.

    A top-level function so it pickles for a process pool. ``labels`` is the labelling from
    the fitting phase when this seed was computed in the same run; when ``None`` (a seed
    restored from a prior run's checkpoint), it is refit here on demand.
    """
    if labels is None:
        labels = _labels(
            _fit(
                measurement, covariates, descriptor, n_components=n_components, n_init=1, seed=seed
            ),
            measurement,
        )
    comparison = compare_to_reference(
        measurement,
        labels,
        reference_labels,
        reference_enrichment,
        category_map,
        n_components=n_components,
    )
    row = _comparison_row(seed, avg_ll, comparison)
    return {"row": row, "overlap": _overlap_payload(comparison)}


def run_multi_init_stability(
    matrix: CohortMatrix,
    typing: Typing,
    reference_labels: pd.Series,
    reference_enrichment: pd.DataFrame,
    category_map: dict[str, str],
    *,
    n_fits: int,
    top_k: int,
    n_components: int = config.DEFAULT_N_COMPONENTS,
    base_seed: int = 0,
    checkpoint_dir: Path | None = None,
    workers: int = 0,
) -> StabilitySummary:
    """Run many single-init fits, rank by log-likelihood, and compare the best to the reference.

    Both phases parallelise over independent single-init fits, each pinned to a single BLAS
    thread (:func:`analysis.profiling.single_threaded_blas`) so concurrent workers do not
    oversubscribe the machine. The run has two resumable phases when ``checkpoint_dir`` is
    given. Each fit appends its seed and log-likelihood to a checkpoint as it completes, and
    each top-``top_k`` comparison appends its result to a second checkpoint; a re-run over the
    same directory recomputes only the fits and comparisons missing from them (tracked by seed,
    not by checkpoint length, since concurrent units do not finish in submission order). The
    per-fit labels are not stored: a comparison whose fit was restored from a prior run refits
    that seed on demand (the same seed and single initialisation reproduce it), which keeps the
    checkpoint to scalars while still resuming exactly. Both phases reassemble their rows in a
    deterministic order regardless of completion order (fits by seed before the final rank,
    comparisons in ``best_seeds`` order), so a resumed run's tables match an uninterrupted run's
    exactly.

    A single-initialisation fit that fails to converge (see ``_try_fit``) is kept in the
    ranked table with a ``nan`` log-likelihood and dropped before the top-``top_k`` selection,
    so one bad fit never crashes the run and only well-formed fits are compared.

    Parameters
    ----------
    matrix : analysis.cohort.CohortMatrix
        The cohort feature and covariate matrices.
    typing : analysis.features.Typing
        The reconciled feature typing.
    reference_labels : pandas.Series
        The reference solution's labels.
    reference_enrichment : pandas.DataFrame
        The reference solution's per-feature enrichment (for the signature and the
        contributory feature set).
    category_map : dict of str to str
        Feature-to-category map.
    n_fits : int
        Number of single-initialisation fits (the released code uses 2,000).
    top_k : int
        Number of best fits (by log-likelihood) to compare to the reference (released: 100).
    n_components : int, optional
        Number of classes.
    base_seed : int, optional
        Seeds are ``base_seed`` to ``base_seed + n_fits - 1``.
    checkpoint_dir : Path, optional
        Directory for the resumable checkpoints. When ``None`` the run is held in memory and
        an interrupt loses the work. The directory must be specific to these parameters.
    workers : int, default 0
        Concurrent worker processes per phase. ``0`` uses the logical core count minus one.
        ``1`` runs in-process instead of through a pool, which a test that monkeypatches a
        fitting dependency relies on (a spawned worker process would not see the patch).

    Returns
    -------
    StabilitySummary
        The ranked fits, the per-fit comparisons of the top ``top_k``, the mean overlap, and
        the aggregate statistics.
    """
    measurement, descriptor, covariates = prepare_inputs(matrix, typing)
    n_workers = _n_workers(workers)

    fit_log = CheckpointLog(checkpoint_dir / f"fits{SUFFIX}") if checkpoint_dir else None
    done_fits = fit_log.load() if fit_log else []
    fits_by_seed: dict[int, dict[str, object]] = {int(r["seed"]): r for r in done_fits}
    # Labels are kept only for fits computed in this run; a fit restored from the checkpoint
    # is refit on demand in the comparison phase, so the checkpoint holds scalars, not vectors.
    labels_by_seed: dict[int, pd.Series] = {}
    best_ll = float("-inf")
    pending_fits = [i for i in range(n_fits) if base_seed + i not in fits_by_seed]
    with task_bar(n_fits, "stability:multi-init", initial=len(fits_by_seed)) as bar:

        def handle_fit(record: dict[str, object], labels: pd.Series | None) -> None:
            nonlocal best_ll
            seed = cast(int, record["seed"])
            if labels is not None:
                best_ll = max(best_ll, cast(float, record["avg_log_likelihood"]))
                labels_by_seed[seed] = labels
            if fit_log:
                fit_log.append(record)
            fits_by_seed[seed] = record
            if best_ll != float("-inf"):
                bar.set_postfix(best_ll=f"{best_ll:.1f}")
            bar.update(1)

        if n_workers <= 1:
            # In-process: no pool overhead for a single worker, and a monkeypatched
            # dependency (as the non-convergence-guard test uses) still takes effect, which a
            # spawned process would not see (it re-imports the module fresh).
            for i in pending_fits:
                handle_fit(
                    *_run_multi_init_fit(
                        measurement,
                        covariates,
                        descriptor,
                        n_components=n_components,
                        seed=base_seed + i,
                    )
                )
        else:
            with (
                profiling.single_threaded_blas(),
                ProcessPoolExecutor(max_workers=n_workers) as pool,
            ):
                futures = {
                    pool.submit(
                        _run_multi_init_fit,
                        measurement,
                        covariates,
                        descriptor,
                        n_components=n_components,
                        seed=base_seed + i,
                    )
                    for i in pending_fits
                }
                for future in as_completed(futures):
                    handle_fit(*future.result())

    fit_records = [fits_by_seed[base_seed + i] for i in range(n_fits)]
    fits = pd.DataFrame.from_records(fit_records).sort_values(
        "avg_log_likelihood", ascending=False, ignore_index=True
    )
    # A fit that failed to converge has a nan log-likelihood; drop it before ranking, so only
    # well-formed fits are compared to the reference.
    best_seeds = [
        int(seed)
        for seed in fits.dropna(subset=["avg_log_likelihood"]).head(top_k)["seed"].tolist()
    ]

    cmp_log = CheckpointLog(checkpoint_dir / f"compare{SUFFIX}") if checkpoint_dir else None
    done_cmps = cmp_log.load() if cmp_log else []
    cmp_by_seed: dict[int, dict[str, object]] = {int(e["row"]["seed"]): e for e in done_cmps}
    remaining = [seed for seed in best_seeds if seed not in cmp_by_seed]
    with task_bar(len(best_seeds), "stability:compare", initial=len(done_cmps)) as bar:

        def handle_comparison(seed: int, entry: dict[str, object]) -> None:
            if cmp_log:
                cmp_log.append(entry)
            cmp_by_seed[seed] = entry
            bar.update(1)

        if n_workers <= 1:
            for seed in remaining:
                avg_ll = float(fits.loc[fits["seed"] == seed, "avg_log_likelihood"].iloc[0])
                handle_comparison(
                    seed,
                    _run_comparison(
                        measurement,
                        covariates,
                        descriptor,
                        reference_labels,
                        reference_enrichment,
                        category_map,
                        n_components=n_components,
                        seed=seed,
                        avg_ll=avg_ll,
                        labels=labels_by_seed.get(seed),
                    ),
                )
        else:
            with (
                profiling.single_threaded_blas(),
                ProcessPoolExecutor(max_workers=n_workers) as pool,
            ):
                futures = {
                    pool.submit(
                        _run_comparison,
                        measurement,
                        covariates,
                        descriptor,
                        reference_labels,
                        reference_enrichment,
                        category_map,
                        n_components=n_components,
                        seed=seed,
                        avg_ll=float(fits.loc[fits["seed"] == seed, "avg_log_likelihood"].iloc[0]),
                        labels=labels_by_seed.get(seed),
                    ): seed
                    for seed in remaining
                }
                for future in as_completed(futures):
                    handle_comparison(futures[future], future.result())

    rows: list[dict[str, object]] = [cast(dict, cmp_by_seed[seed]["row"]) for seed in best_seeds]
    overlaps: list[pd.DataFrame] = [
        _overlap_from_payload(cast(list, cmp_by_seed[seed]["overlap"]), n_components)
        for seed in best_seeds
        if cmp_by_seed[seed]["overlap"] is not None
    ]

    comparisons = pd.DataFrame.from_records(rows)
    aggregate, overlap_mean = _aggregate(comparisons, overlaps)
    aggregate["n_fits"] = int(n_fits)
    aggregate["top_k"] = int(top_k)
    return StabilitySummary(fits, comparisons, overlap_mean, aggregate)


def _run_subsample_fit(
    measurement: pd.DataFrame,
    covariates: pd.DataFrame,
    descriptor: dict,
    reference_labels: pd.Series,
    reference_enrichment: pd.DataFrame,
    category_map: dict[str, str],
    *,
    n_components: int,
    n_init: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, object], list[list[float]] | None]:
    """Fit one subsample and compare it to the reference.

    A top-level function so it pickles for a process pool. ``measurement`` and ``covariates``
    are already sliced to the subsample. Returns the fit record, the comparison row, and the
    serialised overlap payload (``None`` when the fit failed to converge).
    """
    model = _try_fit(
        measurement, covariates, descriptor, n_components=n_components, n_init=n_init, seed=seed
    )
    fit_record: dict[str, object]
    if model is None:
        fit_record = {"seed": seed, "avg_log_likelihood": float("nan"), "converged": False}
        return fit_record, _failed_comparison_row(seed), None
    labels = _labels(model, measurement)
    avg_ll = float(model.score(measurement, covariates))
    comparison = compare_to_reference(
        measurement,
        labels,
        reference_labels,
        reference_enrichment,
        category_map,
        n_components=n_components,
    )
    fit_record = {"seed": seed, "avg_log_likelihood": avg_ll, "converged": bool(model.converged_)}
    row = _comparison_row(seed, avg_ll, comparison)
    return fit_record, row, _overlap_payload(comparison)


def run_subsampling_stability(
    matrix: CohortMatrix,
    typing: Typing,
    reference_labels: pd.Series,
    reference_enrichment: pd.DataFrame,
    category_map: dict[str, str],
    *,
    n_reps: int,
    frac: float = 0.5,
    n_init: int = 20,
    n_components: int = config.DEFAULT_N_COMPONENTS,
    base_seed: int = 0,
    checkpoint_dir: Path | None = None,
    workers: int = 0,
) -> StabilitySummary:
    """Refit on random subsamples and compare each back to the reference.

    Replicates are independent, so they run concurrently over a
    :class:`~concurrent.futures.ProcessPoolExecutor`, each pinned to a single BLAS thread
    (:func:`analysis.profiling.single_threaded_blas`) so concurrent workers do not
    oversubscribe the machine. Each replicate is one resumable unit: a re-run over the same
    ``checkpoint_dir`` recomputes only the replicates missing from it (tracked by replicate
    index, not by checkpoint length, since concurrent replicates do not finish in submission
    order), and the results are reassembled in replicate order regardless of completion order,
    so a resumed run's tables match an uninterrupted run's exactly.

    Parameters
    ----------
    matrix : analysis.cohort.CohortMatrix
        The cohort feature and covariate matrices.
    typing : analysis.features.Typing
        The reconciled feature typing.
    reference_labels : pandas.Series
        The reference solution's labels.
    reference_enrichment : pandas.DataFrame
        The reference solution's per-feature enrichment (for the signature and the
        contributory feature set).
    category_map : dict of str to str
        Feature-to-category map.
    n_reps : int
        Number of subsample replicates (the released code uses 100).
    frac : float, default 0.5
        Subsample fraction without replacement.
    n_init : int, default 20
        Random restarts per subsample fit (the released code uses 20).
    n_components : int, optional
        Number of classes.
    base_seed : int, optional
        Seeds are ``base_seed`` to ``base_seed + n_reps - 1``.
    checkpoint_dir : Path, optional
        Directory for the resumable checkpoint. When ``None`` the run is held in memory and an
        interrupt loses the work. The directory must be specific to these parameters.
    workers : int, default 0
        Concurrent worker processes. ``0`` uses the logical core count minus one. ``1`` runs
        in-process instead of through a pool.

    Returns
    -------
    StabilitySummary
        The per-replicate fits, comparisons, mean overlap, and aggregate statistics.
    """
    measurement, descriptor, covariates = prepare_inputs(matrix, typing)
    n_workers = _n_workers(workers)

    log = CheckpointLog(checkpoint_dir / f"subsample{SUFFIX}") if checkpoint_dir else None
    done = log.load() if log else []
    entries_by_index: dict[int, dict[str, object]] = {
        cast(int, cast(dict, e["fit"])["seed"]) - base_seed: e for e in done
    }
    pending = [i for i in range(n_reps) if i not in entries_by_index]

    with task_bar(n_reps, "stability:subsample", initial=len(entries_by_index)) as bar:

        def handle(
            i: int,
            fit_record: dict[str, object],
            row: dict[str, object],
            overlap_payload: list[list[float]] | None,
        ) -> None:
            entry: dict[str, object] = {"fit": fit_record, "row": row, "overlap": overlap_payload}
            if log:
                log.append(entry)
            entries_by_index[i] = entry
            bar.update(1)

        if n_workers <= 1:
            # In-process: no pool overhead for a single worker, and preserves the ability to
            # monkeypatch a fitting dependency in a test (a spawned process would not see it).
            for i in pending:
                seed = base_seed + i
                index = measurement.sample(frac=frac, random_state=seed).index
                handle(
                    i,
                    *_run_subsample_fit(
                        measurement.loc[index],
                        covariates.loc[index],
                        descriptor,
                        reference_labels,
                        reference_enrichment,
                        category_map,
                        n_components=n_components,
                        n_init=n_init,
                        seed=seed,
                    ),
                )
        else:
            with (
                profiling.single_threaded_blas(),
                ProcessPoolExecutor(max_workers=n_workers) as pool,
            ):
                futures = {}
                for i in pending:
                    seed = base_seed + i
                    index = measurement.sample(frac=frac, random_state=seed).index
                    future = pool.submit(
                        _run_subsample_fit,
                        measurement.loc[index],
                        covariates.loc[index],
                        descriptor,
                        reference_labels,
                        reference_enrichment,
                        category_map,
                        n_components=n_components,
                        n_init=n_init,
                        seed=seed,
                    )
                    futures[future] = i
                for future in as_completed(futures):
                    handle(futures[future], *future.result())

    fit_records = [cast(dict, entries_by_index[i]["fit"]) for i in range(n_reps)]
    rows = [cast(dict, entries_by_index[i]["row"]) for i in range(n_reps)]
    overlaps: list[pd.DataFrame] = [
        _overlap_from_payload(cast(list, entries_by_index[i]["overlap"]), n_components)
        for i in range(n_reps)
        if entries_by_index[i]["overlap"] is not None
    ]

    fits = pd.DataFrame.from_records(fit_records)
    comparisons = pd.DataFrame.from_records(rows)
    aggregate, overlap_mean = _aggregate(comparisons, overlaps)
    aggregate["n_reps"] = int(n_reps)
    aggregate["frac"] = float(frac)
    aggregate["n_init"] = int(n_init)
    return StabilitySummary(fits, comparisons, overlap_mean, aggregate)


def estimate_floor(
    per_fit: pd.DataFrame,
    benchmark: float,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> tuple[int | None, tuple[int, int] | None]:
    r"""Estimate the recovery floor by isotonic regression with a bootstrap interval.

    Fits a monotone (non-decreasing) regression of the per-fit profile correlation on
    :math:`\log_{10}` size, then reads the smallest size at which the fitted recovery reaches
    ``benchmark``. Because recovery improves with sample size in expectation, the monotone fit
    irons out the scatter a small replicate count produces, so the estimate is stable where the
    smallest-clearing-size rule is not. A fit-level bootstrap gives a percentile interval; its
    upper bound is the conservative bin floor.

    Parameters
    ----------
    per_fit : pandas.DataFrame
        One row per fit, with ``size`` and ``overall_correlation``. Rows whose correlation is
        missing (a fit that collapsed a class) are dropped.
    benchmark : float
        The profile-correlation threshold that defines recovery.
    n_bootstrap : int, default 1000
        Fit-level bootstrap resamples for the interval.
    seed : int, default 0
        Seed for the bootstrap resampling.

    Returns
    -------
    tuple
        ``(floor, (lower, upper))``. ``floor`` is the crossing size, or ``None`` when the
        fitted curve does not reach the benchmark in the swept range. The interval is ``None``
        when fewer than half the resamples cross.
    """
    data = per_fit.dropna(subset=["overall_correlation"])
    sizes = data["size"].to_numpy(dtype=float)
    correlation = data["overall_correlation"].to_numpy(dtype=float)
    if len(data) < 3 or len(np.unique(sizes)) < 2:
        return None, None
    log_sizes = np.log10(sizes)
    grid = np.linspace(log_sizes.min(), log_sizes.max(), 200)

    def crossing(x: np.ndarray, y: np.ndarray) -> float | None:
        fitted = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y).predict(grid)
        above = grid[fitted >= benchmark]
        return float(above.min()) if above.size else None

    point = crossing(log_sizes, correlation)
    floor = int(round(10**point)) if point is not None else None

    rng = np.random.default_rng(seed)
    n = len(data)
    resampled: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        crossed = crossing(log_sizes[idx], correlation[idx])
        if crossed is not None:
            resampled.append(crossed)
    floor_ci: tuple[int, int] | None = None
    if len(resampled) >= n_bootstrap // 2:
        lower, upper = np.percentile(resampled, [5, 95])
        floor_ci = (int(round(10**lower)), int(round(10**upper)))
    return floor, floor_ci


def _run_nmin_fit(
    measurement: pd.DataFrame,
    covariates: pd.DataFrame,
    descriptor: dict,
    reference_labels: pd.Series,
    reference_enrichment: pd.DataFrame,
    category_map: dict[str, str],
    *,
    n_components: int,
    n_init: int,
    seed: int,
    size: int,
    rep: int,
) -> dict[str, object]:
    """Fit one (size, replicate) subsample and score its recovery against the reference.

    A top-level function so it pickles for a process pool. ``measurement`` and ``covariates``
    are already sliced to the subsample.
    """
    model = _try_fit(
        measurement, covariates, descriptor, n_components=n_components, n_init=n_init, seed=seed
    )
    if model is None:
        return {
            "size": size,
            "replicate": rep,
            "overall_correlation": None,
            "smallest_class_proportion": float("nan"),
            "degenerate": True,
            "relative_entropy": float("nan"),
            "alcpp": float("nan"),
            "converged": False,
        }
    labels = _labels(model, measurement)
    comparison = compare_to_reference(
        measurement,
        labels,
        reference_labels,
        reference_enrichment,
        category_map,
        n_components=n_components,
    )
    relative_entropy = float(model.relative_entropy(measurement, covariates))
    alcpp = float(model.predict_proba(measurement).max(axis=1).mean())
    return {
        "size": size,
        "replicate": rep,
        "overall_correlation": comparison.overall_correlation,
        "smallest_class_proportion": comparison.smallest_class_proportion,
        "degenerate": comparison.degenerate,
        "relative_entropy": relative_entropy,
        "alcpp": alcpp,
        "converged": bool(model.converged_),
    }


def run_nmin_sweep(
    matrix: CohortMatrix,
    typing: Typing,
    reference_enrichment: pd.DataFrame,
    reference_labels: pd.Series,
    category_map: dict[str, str],
    *,
    sizes: Sequence[int],
    n_reps: int,
    benchmark: float,
    n_init: int = 20,
    n_components: int = config.DEFAULT_N_COMPONENTS,
    base_seed: int = 0,
    checkpoint_dir: Path | None = None,
    workers: int = 0,
) -> NminResult:
    """Refit at descending sample sizes to fix the minimum viable stratum size.

    Each target size is fitted ``n_reps`` times on a random subsample of that size; the
    recovery metrics recorded are the smallest class proportion, the scaled relative entropy,
    the average latent-class posterior probability, and the profile correlation to the
    full-sample reference. The minimum viable size is the smallest swept size whose mean
    profile correlation holds at or above ``benchmark`` (plan section 7b).

    Every ``(size, replicate)`` cell is an independent fit, so the sweep runs concurrently over
    a :class:`~concurrent.futures.ProcessPoolExecutor`, each worker pinned to a single BLAS
    thread (:func:`analysis.profiling.single_threaded_blas`) so concurrent workers do not
    oversubscribe the machine. Each cell is one resumable unit: a re-run over the same
    ``checkpoint_dir`` recomputes only the cells missing from it (tracked by ``(size,
    replicate)``, not by checkpoint length, since concurrent cells do not finish in submission
    order), and the results are reassembled in the sweep's own grid order regardless of
    completion order, so a resumed run's tables match an uninterrupted run's exactly.

    Parameters
    ----------
    matrix : analysis.cohort.CohortMatrix
        The cohort feature and covariate matrices.
    typing : analysis.features.Typing
        The reconciled feature typing.
    reference_enrichment : pandas.DataFrame
        The full-sample reference per-feature enrichment (for the signature and the
        contributory feature set).
    reference_labels : pandas.Series
        The reference solution's labels (for the profile-correlation alignment).
    category_map : dict of str to str
        Feature-to-category map.
    sizes : sequence of int
        Target subsample sizes to sweep, largest first.
    n_reps : int
        Replicates per size.
    benchmark : float
        Profile-correlation threshold that defines recovery.
    n_init : int, default 20
        Random restarts per fit.
    n_components : int, optional
        Number of classes.
    base_seed : int, optional
        Base seed; each (size, replicate) gets a distinct derived seed.
    checkpoint_dir : Path, optional
        Directory for the resumable checkpoint. When ``None`` the run is held in memory and an
        interrupt loses the work. The directory must be specific to these parameters.
    workers : int, default 0
        Concurrent worker processes. ``0`` uses the logical core count minus one. ``1`` runs
        in-process instead of through a pool, which a test that monkeypatches a fitting
        dependency relies on (a spawned worker process would not see the patch).

    Returns
    -------
    NminResult
        The per-fit metrics, the per-size summary, the smallest-clearing-size ``n_min``, and
        the isotonic recovery floor with its bootstrap interval.
    """
    measurement, descriptor, covariates = prepare_inputs(matrix, typing)
    total_rows = len(measurement)
    sizes = [int(s) for s in sizes if int(s) <= total_rows]
    n_workers = _n_workers(workers)

    log = CheckpointLog(checkpoint_dir / f"nmin{SUFFIX}") if checkpoint_dir else None
    done = list(log.load()) if log else []
    records_by_pair: dict[tuple[int, int], dict[str, object]] = {
        (cast(int, r["size"]), cast(int, r["replicate"])): r for r in done
    }
    # The (size, replicate) grid in a fixed order, so a resumed run's table matches an
    # uninterrupted run's regardless of which cells happened to already be on disk.
    units = [
        (size_index, size, rep) for size_index, size in enumerate(sizes) for rep in range(n_reps)
    ]
    pending = [
        (size_index, size, rep)
        for size_index, size, rep in units
        if (size, rep) not in records_by_pair
    ]

    with task_bar(len(units), "nmin", initial=len(records_by_pair)) as bar:

        def handle(size: int, rep: int, record: dict[str, object]) -> None:
            if log:
                log.append(record)
            records_by_pair[size, rep] = record
            bar.set_postfix(size=size)
            bar.update(1)

        if n_workers <= 1:
            # In-process: no pool overhead for a single worker, and preserves the ability to
            # monkeypatch a fitting dependency in a test (a spawned process would not see it).
            for size_index, size, rep in pending:
                seed = base_seed + size_index * 1000 + rep
                index = measurement.sample(n=size, random_state=seed).index
                handle(
                    size,
                    rep,
                    _run_nmin_fit(
                        measurement.loc[index],
                        covariates.loc[index],
                        descriptor,
                        reference_labels,
                        reference_enrichment,
                        category_map,
                        n_components=n_components,
                        n_init=n_init,
                        seed=seed,
                        size=size,
                        rep=rep,
                    ),
                )
        else:
            with (
                profiling.single_threaded_blas(),
                ProcessPoolExecutor(max_workers=n_workers) as pool,
            ):
                futures = {}
                for size_index, size, rep in pending:
                    seed = base_seed + size_index * 1000 + rep
                    index = measurement.sample(n=size, random_state=seed).index
                    future = pool.submit(
                        _run_nmin_fit,
                        measurement.loc[index],
                        covariates.loc[index],
                        descriptor,
                        reference_labels,
                        reference_enrichment,
                        category_map,
                        n_components=n_components,
                        n_init=n_init,
                        seed=seed,
                        size=size,
                        rep=rep,
                    )
                    futures[future] = (size, rep)
                for future in as_completed(futures):
                    size, rep = futures[future]
                    handle(size, rep, future.result())

    records = [records_by_pair[size, rep] for _, size, rep in units]
    per_fit = pd.DataFrame.from_records(records)
    summary = (
        per_fit.groupby("size")[
            ["overall_correlation", "smallest_class_proportion", "relative_entropy", "alcpp"]
        ]
        .mean()
        .reset_index()
        .sort_values("size", ascending=False, ignore_index=True)
    )
    cleared = summary[summary["overall_correlation"] >= benchmark]["size"]
    n_min = int(cleared.min()) if not cleared.empty else None
    floor, floor_ci = estimate_floor(per_fit, benchmark, seed=base_seed)
    return NminResult(
        per_fit=per_fit,
        summary=summary,
        n_min=n_min,
        benchmark=benchmark,
        floor=floor,
        floor_ci=floor_ci,
    )
