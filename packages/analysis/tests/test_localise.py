"""Tests for the localisation schemes and the weighted-fit path (synthetic).

The load-bearing guarantees: a :class:`HardBins` fit equals the subset fit the current
stratified analysis runs, and the weighted :func:`analysis.drift.summarise` reduces to the
unweighted one when every weight is one. Both are what let the kernel scheme sit under the
existing pipeline without changing the hard-bin results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from analysis.cohort import CohortMatrix
from analysis.drift import summarise
from analysis.features import Typing
from analysis.localise import (
    HardBins,
    KernelWindows,
    LocalisationScheme,
    fit_locale,
    focal_grid,
    gaussian_weights,
    permute_axis,
)
from analysis.model import fit_gfmm
from analysis.strata import MaxEqualBins, QuantileBins


def _cohort(n: int = 160, seed: int = 0) -> tuple[CohortMatrix, Typing]:
    """A synthetic cohort with four planted classes across the three feature types."""
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1, 2, 3], n // 4)
    c1 = rng.normal(0, 1, n) + (labels == 1) * 5
    c2 = rng.normal(0, 1, n) + (labels == 3) * 5
    b1 = ((labels == 0) | (rng.random(n) < 0.05)).astype(int)
    b2 = ((labels == 2) | (rng.random(n) < 0.05)).astype(int)
    cat1 = np.clip(labels, 0, 2)
    index = pd.Index([f"p{i}" for i in range(n)], name="proband")
    features = pd.DataFrame({"c1": c1, "c2": c2, "b1": b1, "b2": b2, "cat1": cat1}, index=index)
    covariates = pd.DataFrame(
        {"sex": rng.integers(0, 2, n), "age_at_eval_years": rng.integers(4, 18, n)}, index=index
    )
    return CohortMatrix(features, covariates, "synthetic", "v0"), Typing(
        continuous=["c1", "c2"], binary=["b1", "b2"], categorical=["cat1"]
    )


def _axis(matrix: CohortMatrix, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.uniform(2, 18, len(matrix.features)), index=matrix.features.index, name="axis"
    )


def test_hardbins_is_a_localisation_scheme() -> None:
    assert isinstance(HardBins(QuantileBins(q=2)), LocalisationScheme)
    assert isinstance(KernelWindows(bandwidth=1.0, grid=5), LocalisationScheme)


def test_hardbins_locales_reproduce_the_binning() -> None:
    matrix, _ = _cohort()
    axis = _axis(matrix)
    policy = QuantileBins(q=3)
    assignment = policy.assign(axis)
    fits = HardBins(policy).locales(axis)

    assert [f.label for f in fits] == assignment.labels
    for fit in fits:
        # Indicator weight: exactly the bin members carry weight one, everyone else zero.
        member = (assignment.codes == fit.label).to_numpy()
        assert set(np.unique(fit.weights.to_numpy())) <= {0.0, 1.0}
        assert fit.weights.sum() == assignment.counts[fit.label]
        np.testing.assert_array_equal(fit.weights.to_numpy() > 0, member)


def test_fit_locale_hard_bin_equals_subset_fit() -> None:
    # The regression guarantee: a hard-bin local fit is the current subset fit, bit for bit.
    matrix, typing = _cohort()
    axis = _axis(matrix)
    fits = HardBins(MaxEqualBins(min_bin_size=40)).locales(axis)
    locale = fits[0]

    local = fit_locale(matrix, typing, locale, n_init=3, random_state=0)

    keep = locale.weights.reindex(matrix.features.index).fillna(0.0).to_numpy() > 0
    subset = CohortMatrix(
        matrix.features.loc[keep], matrix.covariates.loc[keep], matrix.dataset, matrix.version
    )
    direct = fit_gfmm(subset, typing, n_init=3, random_state=0, progress_bar=0, verbose=0)

    pd.testing.assert_series_equal(local.labels, direct.labels)
    pd.testing.assert_frame_equal(local.measurement_data, direct.measurement_data)


def test_all_ones_sample_weight_equals_no_weight() -> None:
    matrix, typing = _cohort()
    ones = np.ones(len(matrix.features))
    weighted = fit_gfmm(matrix, typing, n_init=3, random_state=0, sample_weight=ones, verbose=0)
    plain = fit_gfmm(matrix, typing, n_init=3, random_state=0, verbose=0)
    pd.testing.assert_series_equal(weighted.labels, plain.labels)


def test_weighted_summarise_reduces_to_unweighted() -> None:
    matrix, typing = _cohort()
    fit = fit_gfmm(matrix, typing, n_init=3, random_state=0, verbose=0)
    ref_labels = fit.labels
    ones = pd.Series(1.0, index=fit.measurement_data.index)

    weighted = summarise(fit.measurement_data, fit.labels, ref_labels, weights=ones)
    unweighted = summarise(fit.measurement_data, fit.labels, ref_labels)

    # Weighted means and the weighted contingency match the unweighted counts under unit weights
    # (dispersion differs only by the ddof convention, so it is not asserted equal here).
    pd.testing.assert_frame_equal(weighted.centroids, unweighted.centroids)
    pd.testing.assert_frame_equal(
        weighted.contingency.astype(int), unweighted.contingency, check_dtype=False
    )


def test_gaussian_weights_peak_and_falloff() -> None:
    values = pd.Series([5.0, 6.0, 7.0, np.nan], index=list("abcd"))
    w = gaussian_weights(values, focal=5.0, bandwidth=1.0)
    assert w["a"] == 1.0
    assert np.isclose(w["b"], np.exp(-0.5))
    assert w["d"] == 0.0  # missing value gets no weight
    assert w["a"] > w["b"] > w["c"]


def test_focal_grid_spans_inner_quantiles_sorted() -> None:
    values = pd.Series(np.arange(100.0))
    grid = focal_grid(values, n_points=5, quantile_span=(0.1, 0.9))
    assert len(grid) == 5
    assert grid == sorted(grid)
    assert grid[0] >= np.quantile(values, 0.1) - 1e-9
    assert grid[-1] <= np.quantile(values, 0.9) + 1e-9


def test_kernel_windows_one_locale_per_focal_point() -> None:
    matrix, _ = _cohort()
    axis = _axis(matrix)
    fits = KernelWindows(bandwidth=2.0, grid=6).locales(axis)
    assert len(fits) == 6
    assert [f.position for f in fits] == sorted(f.position for f in fits)
    for fit in fits:
        w = fit.weights.to_numpy()
        assert np.all((w >= 0.0) & (w <= 1.0))
        assert fit.weights.index.equals(axis.index)


def test_permute_axis_preserves_multiset_and_is_seeded() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=list("abcde"))
    a = permute_axis(values, seed=7)
    b = permute_axis(values, seed=7)
    pd.testing.assert_series_equal(a, b)
    assert sorted(a.to_numpy()) == sorted(values.to_numpy())
    assert a.index.equals(values.index)
