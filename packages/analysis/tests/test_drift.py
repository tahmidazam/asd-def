"""Tests for the drift abstractions, distances, and permutation-null helpers (synthetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analysis.drift import (
    ALIGNMENTS,
    DEFAULT_ALIGNMENT,
    DEFAULT_DISTANCE,
    DISTANCES,
    CentroidHungarian,
    JensenShannon,
    Mahalanobis,
    MeanAbsolute,
    MembershipJaccard,
    ReferenceModel,
    StandardisedEuclidean,
    StratumSummary,
    adjusted_rand_index,
    benjamini_hochberg,
    build_reference,
    class_separation,
    common_columns,
    compute_drift,
    contingency_table,
    null_partition,
    read_against_null,
)
from analysis.features import Typing


def _reference(precision: np.ndarray | None = None) -> ReferenceModel:
    centroids = pd.DataFrame(
        {"f0": [0.0, 10.0, 0.0, 10.0], "f1": [0.0, 0.0, 10.0, 10.0]},
        index=pd.Index([0, 1, 2, 3], name="class"),
    )
    dispersions = pd.DataFrame(
        {"f0": [1.0, 1.0, 1.0, 1.0], "f1": [1.0, 1.0, 1.0, 1.0]},
        index=pd.Index([0, 1, 2, 3], name="class"),
    )
    return ReferenceModel(
        centroids=centroids,
        dispersions=dispersions,
        pooled_sd=pd.Series({"f0": 2.0, "f1": 4.0}),
        precision=np.eye(2) if precision is None else precision,
        labels=pd.Series([0, 1, 2, 3], index=["a", "b", "c", "d"]),
    )


def _identity_contingency() -> pd.DataFrame:
    return pd.DataFrame(np.diag([100] * 4), index=[0, 1, 2, 3], columns=[0, 1, 2, 3])


def _permuted_contingency() -> pd.DataFrame:
    rows = {0: {2: 100}, 1: {3: 100}, 2: {0: 100}, 3: {1: 100}}
    return pd.DataFrame(rows).T.reindex(columns=[0, 1, 2, 3]).fillna(0).astype(int)


def _summary(centroids: pd.DataFrame, contingency: pd.DataFrame) -> StratumSummary:
    dispersions = pd.DataFrame(1.0, index=centroids.index, columns=centroids.columns)
    return StratumSummary(
        centroids=centroids, dispersions=dispersions, contingency=contingency, n=400
    )


def test_common_columns_intersect_in_reference_order() -> None:
    ref = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    src = pd.DataFrame({"c": [1], "a": [2]})
    assert common_columns(src, ref) == ["a", "c"]


def test_contingency_table_crosstabs_shared_probands() -> None:
    fit = pd.Series([0, 0, 1, 1], index=["a", "b", "c", "d"])
    ref = pd.Series([2, 2, 3, 0], index=["a", "b", "c", "d"])
    table = contingency_table(fit, ref)
    assert table.loc[0, 2] == 2
    assert table.loc[1, 3] == 1


def test_adjusted_rand_index_is_one_for_identical_partitions() -> None:
    assert adjusted_rand_index(np.diag([50, 30, 20, 10]).astype(float)) == pytest.approx(1.0)


def test_adjusted_rand_index_is_near_zero_for_independent_partitions() -> None:
    assert adjusted_rand_index(np.full((4, 4), 25.0)) == pytest.approx(0.0, abs=0.02)


def test_build_reference_computes_centroids_dispersions_precision() -> None:
    rng = np.random.default_rng(0)
    blocks = []
    labels = []
    for cls, (m0, m1) in enumerate([(0, 0), (10, 0), (0, 10), (10, 10)]):
        blocks.append(rng.normal([m0, m1], [1.0, 2.0], size=(60, 2)))
        labels += [cls] * 60
    md = pd.DataFrame(np.vstack(blocks), columns=["f0", "f1"])
    md.index = [f"p{i}" for i in range(len(md))]
    ref = build_reference(md, pd.Series(labels, index=md.index))
    assert list(ref.centroids.index) == [0, 1, 2, 3]
    assert ref.precision.shape == (2, 2)
    assert ref.dispersions.loc[0, "f1"] == pytest.approx(2.0, abs=0.5)


def test_membership_alignment_recovers_the_permutation() -> None:
    aligned = MembershipJaccard().align(
        _summary(_reference().centroids, _permuted_contingency()), _reference()
    )
    assert aligned.mapping == {0: 2, 1: 3, 2: 0, 3: 1}
    assert aligned.overall == pytest.approx(1.0)


def test_membership_alignment_flags_low_overlap() -> None:
    contingency = pd.DataFrame([[25, 25, 25, 25]] * 4, index=[0, 1, 2, 3], columns=[0, 1, 2, 3])
    aligned = MembershipJaccard().align(_summary(_reference().centroids, contingency), _reference())
    assert max(aligned.quality.values()) < 0.5
    assert aligned.overall < 0.1


def test_centroid_alignment_recovers_a_permutation() -> None:
    ref = _reference()
    src = ref.centroids.iloc[[2, 3, 0, 1]].reset_index(drop=True)
    src.index = pd.Index([0, 1, 2, 3], name="class")
    aligned = CentroidHungarian().align(_summary(src, _permuted_contingency()), ref)
    assert aligned.mapping == {0: 2, 1: 3, 2: 0, 3: 1}


def test_mahalanobis_with_identity_precision_is_the_centroid_distance() -> None:
    ref = _reference()
    src = ref.centroids.copy()
    src.loc[0] = [3.0, 4.0]  # shifted (3, 4) from (0, 0); identity precision -> 5
    assert Mahalanobis().class_distance(
        _summary(src, _identity_contingency()), 0, ref, 0
    ) == pytest.approx(5.0)


def test_diagonal_distances_count_in_pooled_sd_units() -> None:
    ref = _reference()
    src = ref.centroids.copy()
    src.loc[0] = [2.0, 4.0]  # +1 SD on each feature (pooled sd 2 and 4)
    stratum = _summary(src, _identity_contingency())
    assert StandardisedEuclidean().class_distance(stratum, 0, ref, 0) == pytest.approx(1.0)
    assert MeanAbsolute().class_distance(stratum, 0, ref, 0) == pytest.approx(1.0)


def test_jensen_shannon_is_zero_for_identical_and_grows_with_separation() -> None:
    ref = _reference()
    same = _summary(ref.centroids.copy(), _identity_contingency())
    assert JensenShannon().class_distance(same, 0, ref, 0) == pytest.approx(0.0, abs=1e-6)
    shifted = ref.centroids.copy()
    shifted.loc[0] = [8.0, 8.0]  # far apart relative to dispersion 1 -> near-disjoint Gaussians
    assert (
        JensenShannon().class_distance(_summary(shifted, _identity_contingency()), 0, ref, 0) > 0.8
    )


def test_class_separation_uses_the_chosen_distance() -> None:
    # two classes 2 apart on f0 (sd 1); identity precision -> Mahalanobis distance 2
    ref = ReferenceModel(
        centroids=pd.DataFrame({"f0": [0.0, 2.0]}, index=pd.Index([0, 1], name="class")),
        dispersions=pd.DataFrame({"f0": [1.0, 1.0]}, index=pd.Index([0, 1], name="class")),
        pooled_sd=pd.Series({"f0": 1.0}),
        precision=np.eye(1),
        labels=pd.Series([0, 1], index=["a", "b"]),
    )
    assert class_separation(ref, Mahalanobis()) == pytest.approx(2.0)
    assert class_separation(ref, StandardisedEuclidean()) == pytest.approx(2.0)


def test_compute_drift_aligns_then_measures() -> None:
    ref = _reference()
    src = ref.centroids.iloc[[2, 3, 0, 1]].reset_index(drop=True)
    src.index = pd.Index([0, 1, 2, 3], name="class")
    result = compute_drift(
        _summary(src, _permuted_contingency()), ref, MembershipJaccard(), Mahalanobis()
    )
    assert all(v == pytest.approx(0.0) for v in result.distances.values())
    assert result.alignment.overall == pytest.approx(1.0)


def test_registries_expose_the_defaults() -> None:
    assert ALIGNMENTS[DEFAULT_ALIGNMENT].name == "membership"
    assert DISTANCES[DEFAULT_DISTANCE].name == "mahalanobis"
    assert set(DISTANCES) == {"mahalanobis", "euclidean", "mean-abs", "jsd"}


def test_null_partition_is_a_disjoint_cover() -> None:
    index = pd.Index([f"p{i}" for i in range(10)])
    chunks = null_partition(index, [3, 3, 4], seed=0)
    assert [len(c) for c in chunks] == [3, 3, 4]
    combined = np.concatenate([c.to_numpy() for c in chunks])
    assert set(combined) == set(index)
    assert len(combined) == len(set(combined))


def test_null_partition_is_seed_deterministic() -> None:
    a = null_partition(pd.Index(range(20)), [10, 10], seed=7)
    b = null_partition(pd.Index(range(20)), [10, 10], seed=7)
    assert [list(x) for x in a] == [list(x) for x in b]


def test_read_against_null_reports_percentile_and_corrected_p() -> None:
    out = read_against_null(5.0, [1.0, 2.0, 3.0, 4.0])
    assert out["exceeds_p95"] == 1.0
    assert out["p_value"] == pytest.approx(1 / 5)


def test_pseudo_stratum_returns_none_on_degenerate_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    # A singular covariate GLM raises LinAlgError mid-null; the worker drops that pseudo-stratum
    # instead of letting the exception abort the whole permutation run.
    import analysis.drift as drift_mod

    def boom(*args: object, **kwargs: object) -> None:
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(drift_mod, "fit_gfmm", boom)
    features = pd.DataFrame({"f0": [0.0, 1.0], "f1": [1.0, 0.0]}, index=["a", "b"])
    covariates = pd.DataFrame({"sex": [0, 1], "age_at_eval_years": [5, 9]}, index=["a", "b"])
    typing = Typing(continuous=["f0", "f1"], binary=[], categorical=[])
    ref_labels = pd.Series([0, 1], index=["a", "b"])
    out = drift_mod.summarise_pseudo_stratum(features, covariates, typing, ref_labels, 1, 0)
    assert out is None


def test_is_degenerate_fit_flags_non_finite_loglik() -> None:
    from typing import Any, cast

    from analysis.drift import is_degenerate_fit
    from analysis.model import FitResult

    md = pd.DataFrame({"f0": [0.0, 1.0]}, index=["a", "b"])
    labels = pd.Series([0, 1], index=["a", "b"])
    model = cast(Any, None)
    finite = FitResult(model, labels, md, {"avg_log_likelihood": -3.2})
    diverged = FitResult(model, labels, md, {"avg_log_likelihood": float("nan")})
    assert is_degenerate_fit(finite) is False
    assert is_degenerate_fit(diverged) is True


def test_benjamini_hochberg_rejects_only_the_small_p() -> None:
    reject = benjamini_hochberg(np.array([0.001, 0.2, 0.3, 0.4]), q=0.05)
    assert list(reject) == [True, False, False, False]


def test_benjamini_hochberg_ignores_nan() -> None:
    assert not benjamini_hochberg(np.array([np.nan, np.nan]), q=0.05).any()
