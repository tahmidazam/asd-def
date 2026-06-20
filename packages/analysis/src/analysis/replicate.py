"""Cross-cohort replication: train on SPARK, project onto the SSC.

Reproduces the released ``replication_on_SSC`` procedure (plan section 6, step 6). The
shared feature set is the intersection of the harmonised SPARK and SSC matrices. A fresh
GFMM is fitted on SPARK restricted to those shared features, then the fitted model predicts
class labels on the SSC. Because both cohorts pass through the one model, the class ids
already correspond, so no cross-cohort label alignment is needed (unlike the disjoint-strata
case in phase 4). The replication measure is the correlation of the seven-category
enrichment profiles between the two cohorts, the same currency Litman et al. use to declare
replication.

The plan adds a calibration the released code lacks: a permutation null that breaks the SSC
class-to-profile association, so the observed correlation is read against chance rather than
asserted (plan section 6, deviations; section 12).

Two faithfulness points. StepMix validates prediction inputs by feature count, not by name,
so the SSC measurement matrix is reindexed to the exact SPARK column order before
prediction. The SSC harmonisation relies on our own milestone mapping (the authors used a
hand-cleaned background-history file that was not released), and the locally held SSC
release is small, so the replication is reported with its sample size and these caveats
rather than as a clean reproduction of the published value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from stepmix.stepmix import StepMix

from analysis import config
from analysis.cohort import CohortMatrix
from analysis.enrich import (
    SEVEN_CATEGORIES,
    bootstrap_overall_correlation,
    category_signature,
    contributory_features,
    feature_enrichment,
    profile_correlation,
)
from analysis.features import REVERSE_CODED_SCQ, Typing
from analysis.model import prepare_inputs
from analysis.progress import task_bar


@dataclass
class ReplicationResult:
    """The cross-cohort replication and its calibration.

    Attributes
    ----------
    shared_features : list of str
        The features shared by the two cohorts and used for the fit.
    spark_signature, ssc_signature : pandas.DataFrame
        The seven-category signatures of the SPARK fit and the SSC projection.
    overall_correlation : float or None
        Pearson correlation over the flattened class-by-category profiles.
    category_correlation : dict
        Per-category Pearson correlation across classes.
    null_overall : list of float
        The overall correlation under each label permutation.
    p_value : float or None
        The proportion of null correlations at or above the observed one, or ``None`` when
        the observed correlation is undefined or no permutations were run.
    correlation_ci : dict or None
        The proband-bootstrap percentile interval on the overall correlation, or ``None``
        when the bootstrap was skipped.
    metrics : dict
        Sample sizes, class proportions, and the convergence flag.
    """

    shared_features: list[str]
    spark_signature: pd.DataFrame
    ssc_signature: pd.DataFrame
    overall_correlation: float | None
    category_correlation: dict[str, float | None]
    null_overall: list[float]
    p_value: float | None
    correlation_ci: dict[str, float | int] | None
    metrics: dict[str, object]


def shared_feature_set(spark: CohortMatrix, ssc: CohortMatrix) -> list[str]:
    """Return the features shared by both cohorts, in SPARK feature order.

    A stable order (SPARK order) is used rather than a set's arbitrary order, so the fit and
    projection see identical column positions (StepMix checks feature count, not name).

    Parameters
    ----------
    spark, ssc : analysis.cohort.CohortMatrix
        The harmonised cohort matrices.

    Returns
    -------
    list of str
        The shared feature names.
    """
    ssc_columns = set(ssc.features.columns)
    return [f for f in spark.features.columns if f in ssc_columns]


def _signature(
    measurement: pd.DataFrame,
    labels: pd.Series,
    category_map: dict[str, str],
    n_components: int,
    reverse_coded: tuple[str, ...],
    keep: set[str] | None = None,
) -> pd.DataFrame:
    """Compute the seven-category signature for a cohort's labels."""
    enrichment = feature_enrichment(measurement, labels, n_classes=n_components)
    return category_signature(
        enrichment, category_map, n_classes=n_components, reverse_coded=reverse_coded, keep=keep
    )


def _class_proportions(labels: pd.Series) -> dict[int, float]:
    """Return the class proportions keyed by integer class id."""
    counts = labels.value_counts(normalize=True)
    ids = counts.index.to_numpy()
    values = counts.to_numpy()
    return {int(ids[i]): round(float(values[i]), 4) for i in range(len(ids))}


def _rounded(value: float | None) -> float | None:
    """Round a correlation to four places, preserving ``None`` for undefined values."""
    return None if value is None else round(value, 4)


def run_replication(
    spark: CohortMatrix,
    ssc: CohortMatrix,
    typing: Typing,
    category_map: dict[str, str],
    *,
    n_components: int = config.DEFAULT_N_COMPONENTS,
    n_init: int = config.DEFAULT_N_INIT,
    n_permutations: int = 200,
    n_bootstrap: int = config.DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
    reverse_coded: tuple[str, ...] = REVERSE_CODED_SCQ,
) -> ReplicationResult:
    """Fit a GFMM on the SPARK shared features, project onto the SSC, and correlate profiles.

    Parameters
    ----------
    spark, ssc : analysis.cohort.CohortMatrix
        The harmonised SPARK and SSC matrices.
    typing : analysis.features.Typing
        The reconciled feature typing (restricted internally to the shared features).
    category_map : dict of str to str
        Feature-to-category map for the signatures.
    n_components : int, optional
        Number of classes.
    n_init : int, optional
        Random restarts for the SPARK fit.
    n_permutations : int, optional
        Number of SSC label permutations for the null. Zero skips the null.
    seed : int, optional
        Random seed for the fit and the permutations.
    reverse_coded : tuple of str, optional
        SCQ items whose enrichment direction is flipped before the signature.

    Returns
    -------
    ReplicationResult
        The shared features, both signatures, the observed and null correlations, the
        permutation :math:`p`-value, and the sample-size metrics.
    """
    shared = shared_feature_set(spark, ssc)
    spark_shared = CohortMatrix(
        spark.features[shared], spark.covariates, spark.dataset, spark.version
    )
    ssc_shared = CohortMatrix(ssc.features[shared], ssc.covariates, ssc.dataset, ssc.version)

    spark_measurement, descriptor, spark_covariates = prepare_inputs(spark_shared, typing)
    ssc_measurement, _, _ = prepare_inputs(ssc_shared, typing)
    # StepMix validates by feature count, not name; pin the SSC columns to the fit order.
    ssc_measurement = ssc_measurement.reindex(columns=spark_measurement.columns)

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
    model.fit(spark_measurement, spark_covariates)

    spark_labels = pd.Series(
        model.predict(spark_measurement), index=spark_measurement.index, name="class"
    )
    ssc_labels = pd.Series(
        model.predict(ssc_measurement), index=ssc_measurement.index, name="class"
    )

    # The contributory feature set is fixed on the SPARK fit and applied to both cohorts, as
    # in the released code, so the profile correlation is computed over one feature universe.
    spark_enrichment = feature_enrichment(spark_measurement, spark_labels, n_classes=n_components)
    keep = set(contributory_features(spark_enrichment, n_components))
    spark_signature = category_signature(
        spark_enrichment,
        category_map,
        n_classes=n_components,
        reverse_coded=reverse_coded,
        keep=keep,
    )
    ssc_signature = _signature(
        ssc_measurement, ssc_labels, category_map, n_components, reverse_coded, keep=keep
    )
    overall, per_category = profile_correlation(ssc_signature, spark_signature)

    null_overall: list[float] = []
    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        with task_bar(n_permutations, "replicate:null") as bar:
            for _ in range(n_permutations):
                permuted = pd.Series(
                    rng.permutation(ssc_labels.to_numpy()),
                    index=ssc_labels.index,
                    name="class",
                )
                permuted_signature = _signature(
                    ssc_measurement, permuted, category_map, n_components, reverse_coded, keep=keep
                )
                null_r, _ = profile_correlation(permuted_signature, spark_signature)
                if null_r is not None:
                    null_overall.append(null_r)
                bar.update(1)

    p_value: float | None = None
    if overall is not None and null_overall:
        at_least = sum(1 for r in null_overall if r >= overall)
        p_value = (at_least + 1) / (len(null_overall) + 1)

    # Resample the SSC probands (the labels held fixed) for a confidence interval on the
    # overall correlation, so the replication r carries its sampling uncertainty. The
    # interval is wider here than for the reproduction because the SSC sample is small and
    # one class holds very few probands.
    correlation_ci: dict[str, float | int] | None = None
    if n_bootstrap > 0 and overall is not None:
        correlation_ci = bootstrap_overall_correlation(
            ssc_measurement,
            ssc_labels,
            spark_signature,
            category_map,
            n_boot=n_bootstrap,
            seed=seed,
            n_classes=n_components,
            keep=keep,
        )

    metrics: dict[str, object] = {
        "n_shared_features": len(shared),
        "n_spark": int(len(spark_measurement)),
        "n_ssc": int(len(ssc_measurement)),
        "spark_proportions": _class_proportions(spark_labels),
        "ssc_proportions": _class_proportions(ssc_labels),
        "converged": bool(model.converged_),
        "overall_correlation": _rounded(overall),
        "category_correlation": {cat: _rounded(per_category[cat]) for cat in SEVEN_CATEGORIES},
        "p_value": p_value,
        "overall_correlation_ci": correlation_ci,
    }
    return ReplicationResult(
        shared_features=shared,
        spark_signature=spark_signature,
        ssc_signature=ssc_signature,
        overall_correlation=overall,
        category_correlation=per_category,
        null_overall=null_overall,
        p_value=p_value,
        correlation_ci=correlation_ci,
        metrics=metrics,
    )
