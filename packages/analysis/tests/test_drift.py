"""Tests for the drift abstractions, metrics, and permutation-null helpers (synthetic)."""

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
    MeanAbsolute,
    MembershipJaccard,
    ReferenceModel,
    StandardisedEuclidean,
    StratumSummary,
    adjusted_rand_index,
    benjamini_hochberg,
    class_separation,
    common_columns,
    compute_drift,
    contingency_table,
    null_partition,
    read_against_null,
)


def _reference() -> ReferenceModel:
    centroids = pd.DataFrame(
        {"f0": [0.0, 10.0, 0.0, 10.0], "f1": [0.0, 0.0, 10.0, 10.0]},
        index=pd.Index([0, 1, 2, 3], name="class"),
    )
    pooled_sd = pd.Series({"f0": 2.0, "f1": 4.0})
    labels = pd.Series([0, 1, 2, 3], index=["a", "b", "c", "d"])
    return ReferenceModel(centroids=centroids, pooled_sd=pooled_sd, labels=labels)


def _permuted_contingency() -> pd.DataFrame:
    # fit class i overlaps reference class (i + 2) mod 4 perfectly
    rows = {0: {2: 100}, 1: {3: 100}, 2: {0: 100}, 3: {1: 100}}
    return pd.DataFrame(rows).T.reindex(columns=[0, 1, 2, 3]).fillna(0).astype(int)


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
    assert table.loc[1, 0] == 1


def test_adjusted_rand_index_is_one_for_identical_partitions() -> None:
    table = np.diag([50, 30, 20, 10]).astype(float)
    assert adjusted_rand_index(table) == pytest.approx(1.0)


def test_adjusted_rand_index_is_near_zero_for_independent_partitions() -> None:
    table = np.full((4, 4), 25.0)  # every cell equal -> no association
    assert adjusted_rand_index(table) == pytest.approx(0.0, abs=0.02)


def test_membership_alignment_recovers_the_permutation_with_full_confidence() -> None:
    stratum = StratumSummary(
        centroids=_reference().centroids, contingency=_permuted_contingency(), n=400
    )
    aligned = MembershipJaccard().align(stratum, _reference())
    assert aligned.mapping == {0: 2, 1: 3, 2: 0, 3: 1}
    assert all(q == pytest.approx(1.0) for q in aligned.quality.values())
    assert aligned.overall == pytest.approx(1.0)


def test_membership_alignment_flags_low_overlap() -> None:
    # a fit class that splits evenly across two reference classes -> low Jaccard, low ARI
    contingency = pd.DataFrame([[25, 25, 25, 25]] * 4, index=[0, 1, 2, 3], columns=[0, 1, 2, 3])
    stratum = StratumSummary(centroids=_reference().centroids, contingency=contingency, n=400)
    aligned = MembershipJaccard().align(stratum, _reference())
    assert max(aligned.quality.values()) < 0.5  # no clean correspondence
    assert aligned.overall < 0.1  # near chance


def test_centroid_alignment_recovers_a_permutation() -> None:
    ref = _reference()
    order = [2, 3, 0, 1]
    src = ref.centroids.iloc[order].reset_index(drop=True)
    src.index = pd.Index([0, 1, 2, 3], name="class")
    stratum = StratumSummary(centroids=src, contingency=_permuted_contingency(), n=400)
    aligned = CentroidHungarian().align(stratum, ref)
    assert aligned.mapping == {0: 2, 1: 3, 2: 0, 3: 1}


def test_distance_methods_count_in_pooled_sd_units() -> None:
    a = np.array([2.0, 4.0])
    b = np.array([0.0, 0.0])
    sd = np.array([2.0, 4.0])  # each feature shifts by exactly 1 SD
    assert StandardisedEuclidean().pairwise(a, b, sd) == pytest.approx(1.0)
    assert MeanAbsolute().pairwise(a, b, sd) == pytest.approx(1.0)


def test_class_separation_uses_the_chosen_distance() -> None:
    ref = ReferenceModel(
        centroids=pd.DataFrame({"f0": [0.0, 2.0]}, index=pd.Index([0, 1], name="class")),
        pooled_sd=pd.Series({"f0": 1.0}),
        labels=pd.Series([0, 1], index=["a", "b"]),
    )
    assert class_separation(ref, StandardisedEuclidean()) == pytest.approx(2.0)


def test_compute_drift_aligns_then_measures() -> None:
    ref = _reference()
    # stratum centroids equal the reference but with the same class permutation as the overlap
    order = [2, 3, 0, 1]
    src = ref.centroids.iloc[order].reset_index(drop=True)
    src.index = pd.Index([0, 1, 2, 3], name="class")
    stratum = StratumSummary(centroids=src, contingency=_permuted_contingency(), n=400)
    result = compute_drift(stratum, ref, MembershipJaccard(), StandardisedEuclidean())
    # membership pairs fit i to ref (i+2)%4, whose centroid is identical -> zero drift
    assert all(v == pytest.approx(0.0) for v in result.distances.values())
    assert result.alignment.overall == pytest.approx(1.0)


def test_registries_expose_the_defaults() -> None:
    assert DEFAULT_ALIGNMENT in ALIGNMENTS
    assert DEFAULT_DISTANCE in DISTANCES
    assert ALIGNMENTS[DEFAULT_ALIGNMENT].name == "membership"
    assert DISTANCES[DEFAULT_DISTANCE].name == "euclidean"


def test_null_partition_is_a_disjoint_cover() -> None:
    index = pd.Index([f"p{i}" for i in range(10)])
    chunks = null_partition(index, [3, 3, 4], seed=0)
    assert [len(c) for c in chunks] == [3, 3, 4]
    combined = np.concatenate([c.to_numpy() for c in chunks])
    assert set(combined) == set(index)
    assert len(combined) == len(set(combined))


def test_null_partition_is_seed_deterministic() -> None:
    index = pd.Index(range(20))
    a = null_partition(index, [10, 10], seed=7)
    b = null_partition(index, [10, 10], seed=7)
    assert [list(x) for x in a] == [list(x) for x in b]


def test_read_against_null_reports_percentile_and_corrected_p() -> None:
    out = read_against_null(5.0, [1.0, 2.0, 3.0, 4.0])
    assert out["exceeds_p95"] == 1.0
    assert out["p_value"] == pytest.approx(1 / 5)
    assert out["n_null"] == 4.0


def test_benjamini_hochberg_rejects_only_the_small_p() -> None:
    reject = benjamini_hochberg(np.array([0.001, 0.2, 0.3, 0.4]), q=0.05)
    assert list(reject) == [True, False, False, False]


def test_benjamini_hochberg_ignores_nan() -> None:
    assert not benjamini_hochberg(np.array([np.nan, np.nan]), q=0.05).any()
