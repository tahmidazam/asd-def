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

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import adjusted_rand_score
from stepmix.stepmix import StepMix

from analysis import config
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
) -> StabilitySummary:
    """Run many single-init fits, rank by log-likelihood, and compare the best to the reference.

    The run has two resumable phases when ``checkpoint_dir`` is given. Each fit appends its
    seed and log-likelihood to a checkpoint as it completes, and each top-``top_k`` comparison
    appends its result to a second checkpoint; a re-run over the same directory continues from
    the first missing fit and the first missing comparison. The per-fit labels are not stored:
    a comparison whose fit was restored from a prior run refits that seed on demand (the same
    seed and single initialisation reproduce it), which keeps the checkpoint to scalars while
    still resuming exactly.

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

    Returns
    -------
    StabilitySummary
        The ranked fits, the per-fit comparisons of the top ``top_k``, the mean overlap, and
        the aggregate statistics.
    """
    measurement, descriptor, covariates = prepare_inputs(matrix, typing)

    fit_log = CheckpointLog(checkpoint_dir / f"fits{SUFFIX}") if checkpoint_dir else None
    done_fits = fit_log.load() if fit_log else []
    fit_records: list[dict[str, object]] = list(done_fits)
    # Labels are kept only for fits computed in this run; a fit restored from the checkpoint
    # is refit on demand in the comparison phase, so the checkpoint holds scalars, not vectors.
    labels_by_seed: dict[int, pd.Series] = {}
    best_ll = float("-inf")
    with task_bar(n_fits, "stability:multi-init", initial=len(done_fits)) as bar:
        for i in range(len(done_fits), n_fits):
            seed = base_seed + i
            model = _try_fit(
                measurement, covariates, descriptor, n_components=n_components, n_init=1, seed=seed
            )
            if model is None:
                avg_ll = float("nan")
                converged = False
            else:
                avg_ll = float(model.score(measurement, covariates))
                best_ll = max(best_ll, avg_ll)
                labels_by_seed[seed] = _labels(model, measurement)
                converged = bool(model.converged_)
            record: dict[str, object] = {
                "seed": seed,
                "avg_log_likelihood": avg_ll,
                "converged": converged,
            }
            if fit_log:
                fit_log.append(record)
            fit_records.append(record)
            if best_ll != float("-inf"):
                bar.set_postfix(best_ll=f"{best_ll:.1f}")
            bar.update(1)

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
    done_seeds = {int(entry["row"]["seed"]) for entry in done_cmps}
    rows: list[dict[str, object]] = [entry["row"] for entry in done_cmps]
    overlaps: list[pd.DataFrame] = [
        _overlap_from_payload(entry["overlap"], n_components)
        for entry in done_cmps
        if entry["overlap"] is not None
    ]
    remaining = [seed for seed in best_seeds if seed not in done_seeds]
    with task_bar(len(best_seeds), "stability:compare", initial=len(done_cmps)) as bar:
        for seed in remaining:
            avg_ll = float(fits.loc[fits["seed"] == seed, "avg_log_likelihood"].iloc[0])
            labels = labels_by_seed.get(seed)
            if labels is None:
                labels = _labels(
                    _fit(
                        measurement,
                        covariates,
                        descriptor,
                        n_components=n_components,
                        n_init=1,
                        seed=seed,
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
            if cmp_log:
                cmp_log.append({"row": row, "overlap": _overlap_payload(comparison)})
            rows.append(row)
            if not comparison.degenerate:
                overlaps.append(comparison.overlap)
            bar.update(1)

    comparisons = pd.DataFrame.from_records(rows)
    aggregate, overlap_mean = _aggregate(comparisons, overlaps)
    aggregate["n_fits"] = int(n_fits)
    aggregate["top_k"] = int(top_k)
    return StabilitySummary(fits, comparisons, overlap_mean, aggregate)


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
) -> StabilitySummary:
    """Refit on random subsamples and compare each back to the reference.

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

    Returns
    -------
    StabilitySummary
        The per-replicate fits, comparisons, mean overlap, and aggregate statistics.
    """
    measurement, descriptor, covariates = prepare_inputs(matrix, typing)

    log = CheckpointLog(checkpoint_dir / f"subsample{SUFFIX}") if checkpoint_dir else None
    done = log.load() if log else []
    fit_records: list[dict[str, object]] = [entry["fit"] for entry in done]
    rows: list[dict[str, object]] = [entry["row"] for entry in done]
    overlaps: list[pd.DataFrame] = [
        _overlap_from_payload(entry["overlap"], n_components)
        for entry in done
        if entry["overlap"] is not None
    ]
    with task_bar(n_reps, "stability:subsample", initial=len(done)) as bar:
        for i in range(len(done), n_reps):
            seed = base_seed + i
            index = measurement.sample(frac=frac, random_state=seed).index
            model = _try_fit(
                measurement.loc[index],
                covariates.loc[index],
                descriptor,
                n_components=n_components,
                n_init=n_init,
                seed=seed,
            )
            overlap_payload: list[list[float]] | None
            if model is None:
                avg_ll = float("nan")
                converged = False
                row = _failed_comparison_row(seed)
                overlap_payload = None
            else:
                labels = _labels(model, measurement.loc[index])
                avg_ll = float(model.score(measurement.loc[index], covariates.loc[index]))
                converged = bool(model.converged_)
                comparison = compare_to_reference(
                    measurement.loc[index],
                    labels,
                    reference_labels,
                    reference_enrichment,
                    category_map,
                    n_components=n_components,
                )
                row = _comparison_row(seed, avg_ll, comparison)
                overlap_payload = _overlap_payload(comparison)
                if not comparison.degenerate:
                    overlaps.append(comparison.overlap)
            fit_record: dict[str, object] = {
                "seed": seed,
                "avg_log_likelihood": avg_ll,
                "converged": converged,
            }
            if log:
                log.append({"fit": fit_record, "row": row, "overlap": overlap_payload})
            fit_records.append(fit_record)
            rows.append(row)
            bar.update(1)

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
) -> NminResult:
    """Refit at descending sample sizes to fix the minimum viable stratum size.

    Each target size is fitted ``n_reps`` times on a random subsample of that size; the
    recovery metrics recorded are the smallest class proportion, the scaled relative entropy,
    the average latent-class posterior probability, and the profile correlation to the
    full-sample reference. The minimum viable size is the smallest swept size whose mean
    profile correlation holds at or above ``benchmark`` (plan section 7b).

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

    Returns
    -------
    NminResult
        The per-fit metrics, the per-size summary, the smallest-clearing-size ``n_min``, and
        the isotonic recovery floor with its bootstrap interval.
    """
    measurement, descriptor, covariates = prepare_inputs(matrix, typing)
    total_rows = len(measurement)
    sizes = [int(s) for s in sizes if int(s) <= total_rows]

    log = CheckpointLog(checkpoint_dir / f"nmin{SUFFIX}") if checkpoint_dir else None
    records: list[dict[str, object]] = list(log.load()) if log else []
    # The (size, replicate) grid in a fixed order, so a resumed run skips the leading
    # records already on disk and continues with the same per-unit seeds.
    units = [
        (size_index, size, rep) for size_index, size in enumerate(sizes) for rep in range(n_reps)
    ]
    with task_bar(len(units), "nmin", initial=len(records)) as bar:
        for size_index, size, rep in units[len(records) :]:
            seed = base_seed + size_index * 1000 + rep
            index = measurement.sample(n=size, random_state=seed).index
            model = _try_fit(
                measurement.loc[index],
                covariates.loc[index],
                descriptor,
                n_components=n_components,
                n_init=n_init,
                seed=seed,
            )
            if model is None:
                overall_correlation: float | None = None
                smallest = float("nan")
                degenerate = True
                relative_entropy = float("nan")
                alcpp = float("nan")
                converged = False
            else:
                labels = _labels(model, measurement.loc[index])
                comparison = compare_to_reference(
                    measurement.loc[index],
                    labels,
                    reference_labels,
                    reference_enrichment,
                    category_map,
                    n_components=n_components,
                )
                overall_correlation = comparison.overall_correlation
                smallest = comparison.smallest_class_proportion
                degenerate = comparison.degenerate
                relative_entropy = float(
                    model.relative_entropy(measurement.loc[index], covariates.loc[index])
                )
                alcpp = float(model.predict_proba(measurement.loc[index]).max(axis=1).mean())
                converged = bool(model.converged_)
            record: dict[str, object] = {
                "size": size,
                "replicate": rep,
                "overall_correlation": overall_correlation,
                "smallest_class_proportion": smallest,
                "degenerate": degenerate,
                "relative_entropy": relative_entropy,
                "alcpp": alcpp,
                "converged": converged,
            }
            if log:
                log.append(record)
            records.append(record)
            bar.set_postfix(size=size)
            bar.update(1)

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
