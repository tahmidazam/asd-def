"""Tests for the drift metrics and the permutation-null helpers (synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analysis.drift import (
    align_centroids,
    benjamini_hochberg,
    class_separation,
    common_columns,
    normalised_centroid_shift,
    null_partition,
    read_against_null,
)


def _reference() -> pd.DataFrame:
    # Four well-separated classes at the corners of a 10x10 square.
    return pd.DataFrame(
        {"f0": [0.0, 10.0, 0.0, 10.0], "f1": [0.0, 0.0, 10.0, 10.0]},
        index=pd.Index([0, 1, 2, 3], name="class"),
    )


def test_common_columns_intersect_in_reference_order() -> None:
    ref = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    src = pd.DataFrame({"c": [1], "a": [2]})
    assert common_columns(src, ref) == ["a", "c"]


def test_align_centroids_recovers_a_permutation() -> None:
    ref = _reference()
    # source class i carries reference class (i + 2) mod 4's centroid
    order = [2, 3, 0, 1]
    source = ref.iloc[order].reset_index(drop=True)
    source.index = pd.Index([0, 1, 2, 3], name="class")
    assert align_centroids(source, ref) == {0: 2, 1: 3, 2: 0, 3: 1}


def test_normalised_shift_is_zero_when_centroids_match() -> None:
    ref = _reference()
    sd = pd.Series({"f0": 2.0, "f1": 4.0})
    mapping = {0: 0, 1: 1, 2: 2, 3: 3}
    shifts = normalised_centroid_shift(ref, ref, sd, mapping)
    assert all(v == pytest.approx(0.0) for v in shifts.values())


def test_normalised_shift_counts_in_pooled_sd_units() -> None:
    ref = _reference()
    sd = pd.Series({"f0": 2.0, "f1": 4.0})
    source = ref.copy()
    source.loc[0] = [ref.loc[0, "f0"] + 2.0, ref.loc[0, "f1"] + 4.0]  # +1 SD on each feature
    shifts = normalised_centroid_shift(source, ref, sd, {0: 0, 1: 1, 2: 2, 3: 3})
    assert shifts[0] == pytest.approx(1.0)
    assert shifts[1] == pytest.approx(0.0)


def test_normalised_shift_drops_zero_spread_features() -> None:
    ref = pd.DataFrame({"f0": [0.0, 1.0], "flat": [5.0, 5.0]}, index=pd.Index([0, 1], name="class"))
    sd = pd.Series({"f0": 1.0, "flat": 0.0})
    source = ref.copy()
    source.loc[0, "f0"] = 1.0  # one informative feature shifts by 1 SD
    shifts = normalised_centroid_shift(source, ref, sd, {0: 0, 1: 1})
    assert shifts[0] == pytest.approx(1.0)  # the flat feature is excluded, not div-by-zero


def test_class_separation_is_the_mean_normalised_gap() -> None:
    ref = pd.DataFrame({"f0": [0.0, 2.0]}, index=pd.Index([0, 1], name="class"))
    sd = pd.Series({"f0": 1.0})
    assert class_separation(ref, sd) == pytest.approx(2.0)


def test_null_partition_is_a_disjoint_cover_of_the_given_sizes() -> None:
    index = pd.Index([f"p{i}" for i in range(10)])
    chunks = null_partition(index, [3, 3, 4], seed=0)
    assert [len(c) for c in chunks] == [3, 3, 4]
    combined = pd.Index(np.concatenate([c.to_numpy() for c in chunks]))
    assert set(combined) == set(index)
    assert len(combined) == len(set(combined))  # disjoint


def test_null_partition_is_seed_deterministic() -> None:
    index = pd.Index(range(20))
    a = null_partition(index, [10, 10], seed=7)
    b = null_partition(index, [10, 10], seed=7)
    assert [list(x) for x in a] == [list(x) for x in b]


def test_read_against_null_reports_percentile_and_corrected_p() -> None:
    out = read_against_null(5.0, [1.0, 2.0, 3.0, 4.0])
    assert out["exceeds_p95"] == 1.0
    assert out["p_value"] == pytest.approx(1 / 5)  # (1 + 0 exceedances) / (1 + 4)
    assert out["n_null"] == 4.0


def test_read_against_null_counts_exceedances() -> None:
    out = read_against_null(2.0, [1.0, 2.0, 3.0, 4.0])
    # draws >= 2.0 are {2, 3, 4} -> 3 exceedances
    assert out["p_value"] == pytest.approx(4 / 5)


def test_benjamini_hochberg_rejects_only_the_small_p() -> None:
    # thresholds q*i/m = [0.0125, 0.025, 0.0375, 0.05]; only 0.001 clears its threshold
    reject = benjamini_hochberg(np.array([0.001, 0.2, 0.3, 0.4]), q=0.05)
    assert list(reject) == [True, False, False, False]


def test_benjamini_hochberg_steps_up() -> None:
    # all four small enough that the step-up rejects every one
    reject = benjamini_hochberg(np.array([0.001, 0.002, 0.003, 0.004]), q=0.05)
    assert reject.all()


def test_benjamini_hochberg_ignores_nan() -> None:
    reject = benjamini_hochberg(np.array([np.nan, np.nan]), q=0.05)
    assert not reject.any()
