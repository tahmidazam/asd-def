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
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from stepmix.stepmix import StepMix

from analysis import config
from analysis.align import greedy_overlap_align
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
        benchmark, or ``None`` when no swept size clears it.
    benchmark : float
        The profile-correlation threshold used.
    """

    per_fit: pd.DataFrame
    summary: pd.DataFrame
    n_min: int | None
    benchmark: float


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
) -> StabilitySummary:
    """Run many single-init fits, rank by log-likelihood, and compare the best to the reference.

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

    Returns
    -------
    StabilitySummary
        The ranked fits, the per-fit comparisons of the top ``top_k``, the mean overlap, and
        the aggregate statistics.
    """
    measurement, descriptor, covariates = prepare_inputs(matrix, typing)

    fit_records: list[dict[str, object]] = []
    labels_by_seed: dict[int, pd.Series] = {}
    with task_bar(n_fits, "stability:multi-init") as bar:
        for i in range(n_fits):
            seed = base_seed + i
            model = _fit(
                measurement, covariates, descriptor, n_components=n_components, n_init=1, seed=seed
            )
            labels = _labels(model, measurement)
            avg_ll = float(model.score(measurement, covariates))
            labels_by_seed[seed] = labels
            fit_records.append(
                {"seed": seed, "avg_log_likelihood": avg_ll, "converged": bool(model.converged_)}
            )
            bar.set_postfix(best_ll=f"{max(r['avg_log_likelihood'] for r in fit_records):.1f}")
            bar.update(1)

    fits = pd.DataFrame.from_records(fit_records).sort_values(
        "avg_log_likelihood", ascending=False, ignore_index=True
    )
    best_seeds = fits.head(top_k)["seed"].tolist()

    rows: list[dict[str, object]] = []
    overlaps: list[pd.DataFrame] = []
    with task_bar(len(best_seeds), "stability:compare") as bar:
        for seed in best_seeds:
            avg_ll = float(fits.loc[fits["seed"] == seed, "avg_log_likelihood"].iloc[0])
            comparison = compare_to_reference(
                measurement,
                labels_by_seed[seed],
                reference_labels,
                reference_enrichment,
                category_map,
                n_components=n_components,
            )
            rows.append(_comparison_row(seed, avg_ll, comparison))
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

    Returns
    -------
    StabilitySummary
        The per-replicate fits, comparisons, mean overlap, and aggregate statistics.
    """
    measurement, descriptor, covariates = prepare_inputs(matrix, typing)

    fit_records: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    overlaps: list[pd.DataFrame] = []
    with task_bar(n_reps, "stability:subsample") as bar:
        for i in range(n_reps):
            seed = base_seed + i
            index = measurement.sample(frac=frac, random_state=seed).index
            model = _fit(
                measurement.loc[index],
                covariates.loc[index],
                descriptor,
                n_components=n_components,
                n_init=n_init,
                seed=seed,
            )
            labels = _labels(model, measurement.loc[index])
            avg_ll = float(model.score(measurement.loc[index], covariates.loc[index]))
            fit_records.append(
                {"seed": seed, "avg_log_likelihood": avg_ll, "converged": bool(model.converged_)}
            )
            comparison = compare_to_reference(
                measurement.loc[index],
                labels,
                reference_labels,
                reference_enrichment,
                category_map,
                n_components=n_components,
            )
            rows.append(_comparison_row(seed, avg_ll, comparison))
            if not comparison.degenerate:
                overlaps.append(comparison.overlap)
            bar.update(1)

    fits = pd.DataFrame.from_records(fit_records)
    comparisons = pd.DataFrame.from_records(rows)
    aggregate, overlap_mean = _aggregate(comparisons, overlaps)
    aggregate["n_reps"] = int(n_reps)
    aggregate["frac"] = float(frac)
    aggregate["n_init"] = int(n_init)
    return StabilitySummary(fits, comparisons, overlap_mean, aggregate)


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

    Returns
    -------
    NminResult
        The per-fit metrics, the per-size summary, and the minimum viable size.
    """
    measurement, descriptor, covariates = prepare_inputs(matrix, typing)
    total_rows = len(measurement)
    sizes = [int(s) for s in sizes if int(s) <= total_rows]

    records: list[dict[str, object]] = []
    with task_bar(len(sizes) * n_reps, "nmin") as bar:
        for size_index, size in enumerate(sizes):
            for rep in range(n_reps):
                seed = base_seed + size_index * 1000 + rep
                index = measurement.sample(n=size, random_state=seed).index
                model = _fit(
                    measurement.loc[index],
                    covariates.loc[index],
                    descriptor,
                    n_components=n_components,
                    n_init=n_init,
                    seed=seed,
                )
                labels = _labels(model, measurement.loc[index])
                comparison = compare_to_reference(
                    measurement.loc[index],
                    labels,
                    reference_labels,
                    reference_enrichment,
                    category_map,
                    n_components=n_components,
                )
                records.append(
                    {
                        "size": size,
                        "replicate": rep,
                        "overall_correlation": comparison.overall_correlation,
                        "smallest_class_proportion": comparison.smallest_class_proportion,
                        "degenerate": comparison.degenerate,
                        "relative_entropy": float(
                            model.relative_entropy(measurement.loc[index], covariates.loc[index])
                        ),
                        "alcpp": float(
                            model.predict_proba(measurement.loc[index]).max(axis=1).mean()
                        ),
                        "converged": bool(model.converged_),
                    }
                )
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
    return NminResult(per_fit=per_fit, summary=summary, n_min=n_min, benchmark=benchmark)
