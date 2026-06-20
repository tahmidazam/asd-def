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
from analysis import checkpoint, replicate, selection, stability
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


def test_bootstrap_overall_correlation_structure() -> None:
    from analysis.enrich import (
        bootstrap_overall_correlation,
        category_signature,
        feature_enrichment,
    )

    matrix, _ = _synthetic_matrix(n=120, seed=0)
    labels = pd.Series(np.repeat([0, 1, 2, 3], 120 // 4), index=matrix.features.index, name="class")
    enrichment = feature_enrichment(matrix.features, labels, n_classes=4)
    target = category_signature(enrichment, _CATEGORY_MAP, n_classes=4, reverse_coded=())
    ci = bootstrap_overall_correlation(
        matrix.features, labels, target, _CATEGORY_MAP, n_boot=15, seed=0, reverse_coded=()
    )
    assert 0 < ci["n_valid"] <= 15
    assert ci["ci_low"] <= ci["median"] <= ci["ci_high"]
    assert ci["level"] == 0.95


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
        spark, ssc, typing, _CATEGORY_MAP, n_init=1, n_permutations=3, n_bootstrap=3, seed=0
    )
    assert result.shared_features == ["c1", "c2", "b1"]
    assert result.metrics["n_ssc"] == 40
    assert len(result.null_overall) <= 3
    assert result.correlation_ci is not None
    assert result.correlation_ci["n_valid"] <= 3


# ---- checkpointing -----------------------------------------------------------
def _reference(
    matrix: CohortMatrix, typing: Typing
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return the measurement matrix, planted reference labels, and reference enrichment."""
    from analysis.enrich import feature_enrichment
    from analysis.model import prepare_inputs

    measurement, _, _ = prepare_inputs(matrix, typing)
    labels = pd.Series(
        np.repeat([0, 1, 2, 3], len(measurement) // 4), index=measurement.index, name="class"
    )
    return measurement, labels, feature_enrichment(measurement, labels, n_classes=4)


def test_checkpoint_log_roundtrip(tmp_path) -> None:
    log = checkpoint.CheckpointLog(tmp_path / f"x{checkpoint.SUFFIX}")
    assert log.load() == []
    log.append({"a": 1})
    log.append([{"b": 2.0}, {"c": None}])  # one line may hold several records
    assert log.load() == [{"a": 1}, [{"b": 2.0}, {"c": None}]]


def test_checkpoint_log_skips_torn_final_line(tmp_path) -> None:
    path = tmp_path / f"x{checkpoint.SUFFIX}"
    log = checkpoint.CheckpointLog(path)
    log.append({"a": 1})
    # A process killed mid-write leaves a partial JSON line; it must be dropped, not read.
    with path.open("a", encoding="utf-8") as f:
        f.write('{"a": 2, "b":')
    assert log.load() == [{"a": 1}]


def test_checkpoint_log_preserves_nan(tmp_path) -> None:
    import math

    log = checkpoint.CheckpointLog(tmp_path / f"x{checkpoint.SUFFIX}")
    log.append({"p": float("nan")})
    (loaded,) = log.load()
    assert math.isnan(loaded["p"])


def test_clear_checkpoints_removes_only_checkpoints(tmp_path) -> None:
    (tmp_path / f"select{checkpoint.SUFFIX}").write_text("{}\n", encoding="utf-8")
    (tmp_path / "summary.parquet").write_text("keep", encoding="utf-8")
    checkpoint.clear_checkpoints(tmp_path)
    assert not (tmp_path / f"select{checkpoint.SUFFIX}").exists()
    assert (tmp_path / "summary.parquet").exists()


def test_run_selection_resumes_from_checkpoint(tmp_path) -> None:
    matrix, typing = _synthetic_matrix()
    full = selection.run_selection(matrix, typing, k_values=[1, 2], n_iterations=2, n_init=1, cv=3)
    # First leg writes the checkpoint for iteration 0 only.
    selection.run_selection(
        matrix, typing, k_values=[1, 2], n_iterations=1, n_init=1, cv=3, checkpoint_dir=tmp_path
    )
    # Resuming over the same directory continues from iteration 1 and matches the whole run.
    resumed = selection.run_selection(
        matrix, typing, k_values=[1, 2], n_iterations=2, n_init=1, cv=3, checkpoint_dir=tmp_path
    )
    pd.testing.assert_frame_equal(resumed.per_iteration, full.per_iteration)
    pd.testing.assert_frame_equal(resumed.summary, full.summary)


def test_run_subsampling_stability_resumes_from_checkpoint(tmp_path) -> None:
    matrix, typing = _synthetic_matrix(n=120)
    _, ref_labels, ref_enrichment = _reference(matrix, typing)
    common = (matrix, typing, ref_labels, ref_enrichment, _CATEGORY_MAP)
    full = stability.run_subsampling_stability(*common, n_reps=2, frac=0.75, n_init=1)
    stability.run_subsampling_stability(
        *common, n_reps=1, frac=0.75, n_init=1, checkpoint_dir=tmp_path
    )
    resumed = stability.run_subsampling_stability(
        *common, n_reps=2, frac=0.75, n_init=1, checkpoint_dir=tmp_path
    )
    pd.testing.assert_frame_equal(resumed.fits, full.fits)
    pd.testing.assert_frame_equal(resumed.comparisons, full.comparisons)


def test_run_nmin_sweep_resumes_from_checkpoint(tmp_path) -> None:
    matrix, typing = _synthetic_matrix(n=120)
    _, ref_labels, ref_enrichment = _reference(matrix, typing)
    common = (matrix, typing, ref_enrichment, ref_labels, _CATEGORY_MAP)
    full = stability.run_nmin_sweep(*common, sizes=[120, 80], n_reps=1, benchmark=0.5, n_init=1)
    stability.run_nmin_sweep(
        *common, sizes=[120], n_reps=1, benchmark=0.5, n_init=1, checkpoint_dir=tmp_path
    )
    resumed = stability.run_nmin_sweep(
        *common, sizes=[120, 80], n_reps=1, benchmark=0.5, n_init=1, checkpoint_dir=tmp_path
    )
    pd.testing.assert_frame_equal(resumed.per_fit, full.per_fit)
    pd.testing.assert_frame_equal(resumed.summary, full.summary)
    assert resumed.n_min == full.n_min


def test_run_multi_init_stability_resumes_after_fits_complete(tmp_path) -> None:
    # The hard case: the fit phase finished, then the run was interrupted during the
    # comparison phase. On resume no labels are in memory, so every remaining comparison
    # refits its seed on demand; the result must still equal the uninterrupted run.
    matrix, typing = _synthetic_matrix()
    _, ref_labels, ref_enrichment = _reference(matrix, typing)
    common = (matrix, typing, ref_labels, ref_enrichment, _CATEGORY_MAP)
    full = stability.run_multi_init_stability(*common, n_fits=4, top_k=3)
    stability.run_multi_init_stability(*common, n_fits=4, top_k=3, checkpoint_dir=tmp_path)
    # Keep every fit but drop all comparisons except the first.
    compare = tmp_path / f"compare{checkpoint.SUFFIX}"
    compare.write_text(compare.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    resumed = stability.run_multi_init_stability(
        *common, n_fits=4, top_k=3, checkpoint_dir=tmp_path
    )
    pd.testing.assert_frame_equal(resumed.fits, full.fits)
    pd.testing.assert_frame_equal(resumed.comparisons, full.comparisons)
    pd.testing.assert_frame_equal(resumed.overlap_mean, full.overlap_mean)


# ---- non-convergence guard ---------------------------------------------------
def test_run_selection_records_nan_for_nonconvergent_fit(monkeypatch) -> None:
    matrix, typing = _synthetic_matrix()
    real = selection._fit_model

    def flaky(measurement, covariates, descriptor, *, n_components, n_init, seed):
        if n_components == 2:
            raise np.linalg.LinAlgError("SVD did not converge")
        return real(
            measurement, covariates, descriptor, n_components=n_components, n_init=n_init, seed=seed
        )

    monkeypatch.setattr(selection, "_fit_model", flaky)
    result = selection.run_selection(
        matrix, typing, k_values=[1, 2], n_iterations=1, n_init=1, cv=3
    )
    by_k = result.per_iteration.set_index("n_components")
    assert np.isnan(by_k.loc[2, "bic"])  # the failed direct fit is recorded as missing
    assert not np.isnan(by_k.loc[1, "bic"])  # the converged fit is unaffected


def test_run_nmin_sweep_records_nan_for_nonconvergent_fit(monkeypatch) -> None:
    matrix, typing = _synthetic_matrix(n=120)
    _, ref_labels, ref_enrichment = _reference(matrix, typing)
    real = stability._fit

    def flaky(measurement, covariates, descriptor, *, n_components, n_init, seed):
        if len(measurement) == 120:  # fail the full-size fit, succeed on the smaller subsample
            raise np.linalg.LinAlgError("SVD did not converge")
        return real(
            measurement, covariates, descriptor, n_components=n_components, n_init=n_init, seed=seed
        )

    monkeypatch.setattr(stability, "_fit", flaky)
    result = stability.run_nmin_sweep(
        matrix,
        typing,
        ref_enrichment,
        ref_labels,
        _CATEGORY_MAP,
        sizes=[120, 80],
        n_reps=1,
        benchmark=0.5,
        n_init=1,
    )
    by_size = result.per_fit.set_index("size")
    assert bool(by_size.loc[120, "degenerate"]) and np.isnan(by_size.loc[120, "relative_entropy"])
    assert not np.isnan(by_size.loc[80, "relative_entropy"])


def test_run_multi_init_stability_drops_nonconvergent_fit(monkeypatch) -> None:
    matrix, typing = _synthetic_matrix()
    _, ref_labels, ref_enrichment = _reference(matrix, typing)
    real = stability._fit
    failed_seed = 1

    def flaky(measurement, covariates, descriptor, *, n_components, n_init, seed):
        if seed == failed_seed:
            raise np.linalg.LinAlgError("SVD did not converge")
        return real(
            measurement, covariates, descriptor, n_components=n_components, n_init=n_init, seed=seed
        )

    monkeypatch.setattr(stability, "_fit", flaky)
    summary = stability.run_multi_init_stability(
        matrix, typing, ref_labels, ref_enrichment, _CATEGORY_MAP, n_fits=4, top_k=3
    )
    failed = summary.fits.loc[summary.fits["seed"] == failed_seed, "avg_log_likelihood"]
    assert failed.isna().all()  # kept in the ranked table as nan
    assert failed_seed not in set(summary.comparisons["seed"])  # excluded from the compared top-k


# ---- nmin floor estimator ----------------------------------------------------
def test_estimate_floor_finds_monotone_crossing() -> None:
    sizes = [100, 200, 400, 800, 1600, 3200]

    # Correlation rises monotonically through the 0.90 benchmark between sizes 400 and 800.
    def corr(s: int) -> float:
        return min(0.5 + 0.15 * float(np.log2(s / 100)), 1.0)

    per_fit = pd.DataFrame(
        [{"size": s, "overall_correlation": corr(s)} for s in sizes for _ in range(3)]
    )
    floor, ci = stability.estimate_floor(per_fit, benchmark=0.90, n_bootstrap=200, seed=0)
    assert floor is not None and 400 <= floor <= 800
    assert ci is not None and ci[0] <= floor <= ci[1]


def test_estimate_floor_returns_none_without_crossing() -> None:
    # Recovery never reaches the benchmark anywhere in the swept range.
    per_fit = pd.DataFrame(
        {"size": [100, 400, 1600] * 3, "overall_correlation": [0.5, 0.6, 0.7] * 3}
    )
    floor, ci = stability.estimate_floor(per_fit, benchmark=0.90, n_bootstrap=100, seed=0)
    assert floor is None
    assert ci is None


def test_estimate_floor_drops_degenerate_fits() -> None:
    # A collapsed fit (nan correlation) is dropped rather than breaking the regression.
    per_fit = pd.DataFrame(
        {
            "size": [400, 400, 800, 800, 1600, 1600],
            "overall_correlation": [0.80, float("nan"), 0.95, 0.93, 0.98, float("nan")],
        }
    )
    floor, _ = stability.estimate_floor(per_fit, benchmark=0.90, n_bootstrap=100, seed=0)
    assert floor is not None and 400 <= floor <= 800
