"""Phase 3 (pre-registration): the binning policies the stratification axes share.

These tests pin the policy contract that keeps the stratified analysis independent of the
binning choice: both policies return a :class:`~analysis.strata.StratumAssignment` with an
ordered categorical, label-ordered counts, a missing count, and a serialisable spec. The
data here is synthetic, so the tests need no cohort and no model fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analysis.strata import (
    PROVISIONAL_ERA_BANDS,
    BinningPolicy,
    FixedBands,
    MaxEqualBins,
    QuantileBins,
    StratumAssignment,
)


def test_fixed_bands_assigns_left_closed_with_open_outer_bins() -> None:
    values = pd.Series([2.0, 4.0, 6.0, 7.0, 12.0, np.nan])
    result = FixedBands(edges=(4, 7, 11), labels=("<4", "4-6", "7-10", ">=11")).assign(values)
    assert result.labels == ["<4", "4-6", "7-10", ">=11"]
    # A value on an interior edge falls in the upper band (4 -> "4-6", 7 -> "7-10").
    assert list(result.codes.iloc[:5]) == ["<4", "4-6", "4-6", "7-10", ">=11"]
    assert result.counts == {"<4": 1, "4-6": 2, "7-10": 1, ">=11": 1}
    assert result.n_missing == 1
    assert pd.isna(result.codes.iloc[5])


def test_fixed_bands_default_labels_are_half_open_ranges() -> None:
    result = FixedBands(edges=(3.0, 6.0)).assign(pd.Series([1.0, 4.0, 9.0]))
    assert result.labels == ["<3", "3-6", ">=6"]


def test_fixed_bands_codes_are_ordered_categorical() -> None:
    result = FixedBands(edges=(4, 7, 11)).assign(pd.Series([2.0, 8.0]))
    assert isinstance(result.codes.dtype, pd.CategoricalDtype)
    assert result.codes.cat.ordered
    assert list(result.codes.cat.categories) == result.labels


def test_fixed_bands_spec_round_trips_edges_and_labels() -> None:
    spec = PROVISIONAL_ERA_BANDS.spec()
    assert spec["policy"] == "fixed"
    assert spec["interval"] == "left-closed"
    assert spec["edges"] == [2013.0, 2017.0, 2021.0]
    assert spec["labels"] == ["<=2012", "2013-2016", "2017-2020", ">=2021"]


def test_era_boundary_year_falls_on_the_dsm5_side() -> None:
    result = PROVISIONAL_ERA_BANDS.assign(pd.Series([2012.0, 2013.0, 2021.0]))
    assert list(result.codes) == ["<=2012", "2013-2016", ">=2021"]


@pytest.mark.parametrize("bad_edges", [(7, 4), (4, 4)])
def test_fixed_bands_rejects_non_ascending_edges(bad_edges: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="ascending"):
        FixedBands(edges=bad_edges)


def test_fixed_bands_rejects_wrong_label_count() -> None:
    with pytest.raises(ValueError, match="labels"):
        FixedBands(edges=(4, 7), labels=("a", "b"))


def test_quantile_bins_give_roughly_equal_counts() -> None:
    values = pd.Series(np.arange(100, dtype=float))
    result = QuantileBins(q=4).assign(values)
    assert result.labels == ["Q1", "Q2", "Q3", "Q4"]
    assert all(count == 25 for count in result.counts.values())
    assert result.spec["q"] == 4


def test_quantile_bins_record_realised_edges() -> None:
    values = pd.Series(np.arange(100, dtype=float))
    result = QuantileBins(q=4).assign(values)
    # The interior quartile edges of 0..99 sit near 24.75, 49.5, 74.25.
    assert len(result.edges) == 3
    assert result.edges == sorted(result.edges)
    assert 24.0 < result.edges[0] < 26.0


def test_quantile_bins_collapse_under_heavy_ties() -> None:
    # Almost every value is identical, so the interior quantiles coincide and bins collapse.
    values = pd.Series([0.0] * 95 + [1.0, 2.0, 3.0, 4.0, 5.0])
    result = QuantileBins(q=4).assign(values)
    assert len(result.labels) < 4
    assert sum(result.counts.values()) + result.n_missing == len(values)


def test_quantile_bins_reject_q_below_two() -> None:
    with pytest.raises(ValueError, match="q >= 2"):
        QuantileBins(q=1)


def test_max_equal_bins_picks_count_from_size_and_floor() -> None:
    values = pd.Series(np.arange(11000, dtype=float))
    result = MaxEqualBins(min_bin_size=1000).assign(values)
    # floor(11000 / 1000) = 11 equal bins, each about 1000.
    assert len(result.labels) == 11
    assert min(result.counts.values()) >= 1000
    assert result.spec["q_realised"] == 11
    assert result.spec["min_bin_size"] == 1000


def test_max_equal_bins_guarantee_holds_under_skew() -> None:
    # A right-skewed variable still yields bins that all clear the floor (the step-down).
    rng = np.random.default_rng(7)
    values = pd.Series(rng.exponential(scale=3.0, size=8000))
    result = MaxEqualBins(min_bin_size=1000).assign(values)
    assert min(result.counts.values()) >= 1000
    assert 2 <= len(result.labels) <= 8


def test_max_equal_bins_floor_drives_resolution() -> None:
    values = pd.Series(np.arange(10000, dtype=float))
    coarse = MaxEqualBins(min_bin_size=2000).assign(values)
    fine = MaxEqualBins(min_bin_size=1000).assign(values)
    assert len(coarse.labels) == 5
    assert len(fine.labels) == 10


def test_max_equal_bins_reject_nonpositive_floor() -> None:
    with pytest.raises(ValueError, match="min_bin_size"):
        MaxEqualBins(min_bin_size=0)


def test_all_policies_satisfy_the_binning_protocol() -> None:
    assert isinstance(FixedBands(edges=(4,)), BinningPolicy)
    assert isinstance(QuantileBins(q=4), BinningPolicy)
    assert isinstance(MaxEqualBins(min_bin_size=1000), BinningPolicy)


def test_assignment_accounts_for_every_row() -> None:
    values = pd.Series([1.0, 5.0, 9.0, np.nan, 13.0])
    for policy in (FixedBands(edges=(4, 7, 11)), QuantileBins(q=2)):
        result: StratumAssignment = policy.assign(values)
        assert sum(result.counts.values()) + result.n_missing == len(values)
