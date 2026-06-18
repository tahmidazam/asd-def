"""Unit tests for the phase-2 stages, on synthetic data only.

No participant data is read here. The pure comparison helpers (label alignment, profile
correlation, the overlap matrix, the LMR proxy) are tested directly, and the model-bearing
stages (selection, stability, replication) are exercised end to end on a small synthetic
cohort with a planted class structure, with the smallest workable fitting budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analysis import replicate, selection, stability
from analysis.align import greedy_overlap_align
from analysis.cohort import CohortMatrix
from analysis.enrich import SEVEN_CATEGORIES, profile_correlation
from analysis.features import Typing

_CATEGORY_MAP = {
    "c1": "anxiety/mood",
    "c2": "attention",
    "b1": "self-injury",
    "b2": "social/communication",
    "cat1": "restricted/repetitive",
}


def _synthetic_matrix(n: int = 80, seed: int = 0) -> tuple[CohortMatrix, Typing]:
    """Build a synthetic cohort with four well-separated planted classes.

    The cohort carries all three feature types (two continuous, two binary, one
    three-level categorical) so the StepMix mixed descriptor exercises every emission model.
    """
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1, 2, 3], n // 4)
    c1 = rng.normal(0, 1, n) + (labels == 1) * 5
    c2 = rng.normal(0, 1, n) + (labels == 3) * 5
    b1 = ((labels == 0) | (rng.random(n) < 0.05)).astype(int)
    b2 = ((labels == 2) | (rng.random(n) < 0.05)).astype(int)
    cat1 = np.clip(labels, 0, 2)  # three levels {0, 1, 2}, informative about class
    index = pd.Index([f"p{i}" for i in range(n)], name="proband")
    features = pd.DataFrame({"c1": c1, "c2": c2, "b1": b1, "b2": b2, "cat1": cat1}, index=index)
    covariates = pd.DataFrame(
        {"sex": rng.integers(0, 2, n), "age_at_eval_years": rng.integers(4, 18, n)}, index=index
    )
    matrix = CohortMatrix(features, covariates, "synthetic", "v0")
    typing = Typing(continuous=["c1", "c2"], binary=["b1", "b2"], categorical=["cat1"])
    return matrix, typing


# ---- label alignment ---------------------------------------------------------
def test_greedy_overlap_align_recovers_permutation() -> None:
    index = pd.Index(range(12))
    reference = pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], index=index)
    # source classes are a relabelling: 0->2, 1->3, 2->0, 3->1.
    source = pd.Series([2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1], index=index)
    mapping = greedy_overlap_align(source, reference)
    assert mapping == {2: 0, 3: 1, 0: 2, 1: 3}


def test_greedy_overlap_align_uses_shared_index_only() -> None:
    reference = pd.Series([0, 0, 1, 1], index=[0, 1, 2, 3])
    source = pd.Series([1, 1, 0, 0, 0], index=[0, 1, 2, 3, 99])
    mapping = greedy_overlap_align(source, reference)
    assert mapping == {1: 0, 0: 1}


# ---- profile correlation -----------------------------------------------------
def _signature(values: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        values, index=pd.Index(range(len(values)), name="class"), columns=list(SEVEN_CATEGORIES)
    )


def test_profile_correlation_identity() -> None:
    rng = np.random.default_rng(1)
    sig = _signature(rng.normal(size=(4, 7)).tolist())
    overall, per_category = profile_correlation(sig, sig)
    assert overall == pytest.approx(1.0)
    assert all(r == pytest.approx(1.0) for r in per_category.values())


def test_profile_correlation_constant_returns_none() -> None:
    constant = _signature([[1.0] * 7 for _ in range(4)])
    varied = _signature(np.random.default_rng(2).normal(size=(4, 7)).tolist())
    overall, _ = profile_correlation(constant, varied)
    assert overall is None


def test_profile_correlation_requires_shared_index() -> None:
    a = _signature(np.zeros((4, 7)).tolist())
    b = a.copy()
    b.index = pd.Index([10, 11, 12, 13], name="class")
    with pytest.raises(ValueError, match="share the same class index"):
        profile_correlation(a, b)


# ---- overlap matrix ----------------------------------------------------------
def test_class_overlap_matrix_perfect_recovery() -> None:
    labels = pd.Series([0, 0, 1, 1, 2, 2, 3, 3], index=range(8))
    overlap = stability.class_overlap_matrix(labels, labels, n_components=4)
    assert np.allclose(np.diag(overlap.to_numpy()), 1.0)
    assert overlap.to_numpy().sum() == pytest.approx(4.0)  # off-diagonal all zero


# ---- contributory features + empty-class robustness --------------------------
def test_contributory_features_filters_noise() -> None:
    from analysis.enrich import contributory_features

    # 'strong' is significant with a large continuous effect; 'noise' never reaches
    # significance and stays below the magnitude threshold, so it is dropped.
    enr = pd.DataFrame(
        {
            "is_binary": [0.0, 0.0],
            "class0_dir": [1.0, 0.0],
            "class0_effect": [0.9, 0.05],
            "class1_dir": [0.0, 0.0],
            "class1_effect": [0.1, 0.05],
            "class2_dir": [0.0, 0.0],
            "class2_effect": [0.0, 0.0],
            "class3_dir": [0.0, 0.0],
            "class3_effect": [0.0, 0.0],
        },
        index=pd.Index(["strong", "noise"], name="feature"),
    )
    assert contributory_features(enr, n_classes=4) == ["strong"]


def test_feature_enrichment_tolerates_empty_class() -> None:
    from analysis.enrich import feature_enrichment

    # Class 3 has no probands; the binary and continuous branches must not raise.
    data = pd.DataFrame({"b": [1, 0, 1, 0, 1, 0], "c": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    labels = pd.Series([0, 0, 1, 1, 2, 2])  # no class 3
    enr = feature_enrichment(data, labels, n_classes=4)
    assert np.isnan(enr.loc["b", "class3_effect"])
    assert enr.loc["b", "class3_dir"] == 0.0


# ---- selection ---------------------------------------------------------------
def test_lmr_lrt_proxy_matches_chi_square() -> None:
    from scipy.stats import chi2

    val_ll = {1: -10.0, 2: -8.0, 3: -7.5}
    lmr = selection.lmr_lrt_proxy(val_ll)
    assert set(lmr) == {2, 3}
    assert lmr[2] == pytest.approx(chi2.sf(-2.0 * (-10.0 - -8.0), df=1))
    assert lmr[3] == pytest.approx(chi2.sf(-2.0 * (-8.0 - -7.5), df=1))


def test_awe_matches_manual_formula() -> None:
    from analysis.model import prepare_inputs

    matrix, typing = _synthetic_matrix()
    measurement, descriptor, covariates = prepare_inputs(matrix, typing)
    model = stability._fit(measurement, covariates, descriptor, n_components=2, n_init=1, seed=0)
    n = measurement.shape[0]
    expected = -2.0 * model.score(measurement, covariates) * n + model.n_parameters * (
        np.log(n) + 1.5
    )
    assert selection.awe(model, measurement, covariates) == pytest.approx(expected)


def test_run_selection_small() -> None:
    matrix, typing = _synthetic_matrix()
    result = selection.run_selection(
        matrix, typing, k_values=[1, 2], n_iterations=1, n_init=1, cv=3
    )
    assert list(result.summary["n_components"]) == [1, 2]
    for column in ("bic_mean", "aic_mean", "val_log_likelihood_mean", "alcpp_mean"):
        assert column in result.summary.columns
    assert len(result.per_iteration) == 2


# ---- stability ---------------------------------------------------------------
def test_compare_to_reference_identity() -> None:
    from analysis.enrich import feature_enrichment
    from analysis.model import prepare_inputs

    matrix, typing = _synthetic_matrix()
    measurement, _, _ = prepare_inputs(matrix, typing)
    labels = pd.Series(np.repeat([0, 1, 2, 3], len(measurement) // 4), index=measurement.index)
    reference_enrichment = feature_enrichment(measurement, labels, n_classes=4)

    comparison = stability.compare_to_reference(
        measurement,
        labels,
        labels,
        reference_enrichment,
        _CATEGORY_MAP,
        n_components=4,
        reverse_coded=(),
    )
    assert comparison.adjusted_rand_index == pytest.approx(1.0)
    assert comparison.overall_correlation == pytest.approx(1.0)
    assert not comparison.degenerate
    assert np.allclose(np.diag(comparison.overlap.to_numpy()), 1.0)


def test_compare_to_reference_flags_collapsed_fit() -> None:
    from analysis.enrich import feature_enrichment
    from analysis.model import prepare_inputs

    matrix, typing = _synthetic_matrix()
    measurement, _, _ = prepare_inputs(matrix, typing)
    reference = pd.Series(np.repeat([0, 1, 2, 3], len(measurement) // 4), index=measurement.index)
    reference_enrichment = feature_enrichment(measurement, reference, n_classes=4)
    # A collapsed fit recovering only three of four classes.
    collapsed = pd.Series(np.repeat([0, 1, 2, 2], len(measurement) // 4), index=measurement.index)

    comparison = stability.compare_to_reference(
        measurement,
        collapsed,
        reference,
        reference_enrichment,
        _CATEGORY_MAP,
        n_components=4,
        reverse_coded=(),
    )
    assert comparison.degenerate
    assert comparison.overall_correlation is None
    # The collapsed (empty) class drives the smallest proportion to zero.
    assert comparison.smallest_class_proportion == pytest.approx(0.0)


def test_run_multi_init_stability_small() -> None:
    from analysis.enrich import feature_enrichment
    from analysis.model import prepare_inputs

    matrix, typing = _synthetic_matrix()
    measurement, _, _ = prepare_inputs(matrix, typing)
    reference_labels = pd.Series(
        np.repeat([0, 1, 2, 3], len(measurement) // 4), index=measurement.index, name="class"
    )
    reference_enrichment = feature_enrichment(measurement, reference_labels, n_classes=4)
    summary = stability.run_multi_init_stability(
        matrix,
        typing,
        reference_labels,
        reference_enrichment,
        _CATEGORY_MAP,
        n_fits=3,
        top_k=2,
        base_seed=0,
    )
    assert len(summary.fits) == 3
    assert len(summary.comparisons) == 2
    # The fits are sorted by descending log-likelihood.
    lls = summary.fits["avg_log_likelihood"].tolist()
    assert lls == sorted(lls, reverse=True)
    assert "adjusted_rand_index_mean" in summary.aggregate


def test_run_nmin_sweep_small() -> None:
    from analysis.enrich import feature_enrichment
    from analysis.model import prepare_inputs

    matrix, typing = _synthetic_matrix(n=120)
    measurement, _, _ = prepare_inputs(matrix, typing)
    reference_labels = pd.Series(
        np.repeat([0, 1, 2, 3], len(measurement) // 4), index=measurement.index, name="class"
    )
    reference_enrichment = feature_enrichment(measurement, reference_labels, n_classes=4)
    result = stability.run_nmin_sweep(
        matrix,
        typing,
        reference_enrichment,
        reference_labels,
        _CATEGORY_MAP,
        sizes=[120, 80],
        n_reps=1,
        benchmark=0.5,
        n_init=1,
    )
    assert list(result.summary["size"]) == [120, 80]
    assert {"overall_correlation", "relative_entropy", "alcpp"} <= set(result.summary.columns)


# ---- replication -------------------------------------------------------------
def test_shared_feature_set_preserves_spark_order() -> None:
    spark, _ = _synthetic_matrix()
    ssc_features = spark.features[["c1", "b1"]]
    ssc = CohortMatrix(ssc_features, spark.covariates, "ssc", "v0")
    assert replicate.shared_feature_set(spark, ssc) == ["c1", "b1"]


def test_class_proportions_sum_to_one() -> None:
    labels = pd.Series([0, 0, 0, 1, 2, 2, 3], name="class")
    proportions = replicate._class_proportions(labels)
    assert set(proportions) == {0, 1, 2, 3}
    assert sum(proportions.values()) == pytest.approx(1.0, abs=1e-3)


def test_run_replication_small() -> None:
    spark, typing = _synthetic_matrix(seed=0)
    ssc_matrix, _ = _synthetic_matrix(n=40, seed=1)
    ssc = CohortMatrix(ssc_matrix.features[["c1", "c2", "b1"]], ssc_matrix.covariates, "ssc", "v0")
    result = replicate.run_replication(
        spark, ssc, typing, _CATEGORY_MAP, n_init=1, n_permutations=3, seed=0
    )
    assert result.shared_features == ["c1", "c2", "b1"]
    assert result.metrics["n_ssc"] == 40
    assert len(result.null_overall) <= 3
