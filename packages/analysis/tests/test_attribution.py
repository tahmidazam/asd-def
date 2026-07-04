"""Unit tests for the movement-attribution module, on synthetic data only.

No participant data is read here (the governance rule): the reference and stratum summaries,
the labellings, and the mover/stayer feature blocks are all constructed in memory, so the
tests exercise the decomposition algebra and the contrast ranking against known answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from analysis import attribution as attr
from analysis import drift


def _reference(
    centroids: pd.DataFrame, precision: np.ndarray, labels: pd.Series
) -> drift.ReferenceModel:
    """Build a reference model with an explicit precision, for the decomposition tests."""
    dispersions = pd.DataFrame(
        np.ones_like(centroids.to_numpy()), index=centroids.index, columns=centroids.columns
    )
    pooled_sd = pd.Series(np.ones(centroids.shape[1]), index=centroids.columns)
    return drift.ReferenceModel(
        centroids=centroids,
        dispersions=dispersions,
        pooled_sd=pooled_sd,
        precision=precision,
        labels=labels,
    )


def _stratum(centroids: pd.DataFrame) -> drift.StratumSummary:
    """A stratum summary with unit dispersions and an identity contingency."""
    dispersions = pd.DataFrame(
        np.ones_like(centroids.to_numpy()), index=centroids.index, columns=centroids.columns
    )
    contingency = pd.DataFrame(
        np.eye(len(centroids), dtype=int), index=centroids.index, columns=centroids.index
    )
    return drift.StratumSummary(centroids, dispersions, contingency, n=len(centroids))


def _movement(
    reference: drift.ReferenceModel, stratum: drift.StratumSummary, ref_class: int, fit_class: int
) -> attr.Movement:
    """Wrap a reference and stratum in a minimal Comparison and return one Movement."""
    alignment = drift.ClassAlignment(
        mapping={fit_class: ref_class}, quality={ref_class: 1.0}, overall=1.0
    )
    comparison = attr.Comparison(
        reference=reference,
        stratum=stratum,
        ref_labels=reference.labels,
        fit_labels=pd.Series(dtype=int),
        alignment=alignment,
    )
    return attr.Movement(comparison, ref_class=ref_class, fit_class=fit_class)


def test_mahalanobis_contributions_sum_to_squared_distance() -> None:
    """The per-feature Mahalanobis contributions sum to the squared Mahalanobis distance."""
    cols = ["f0", "f1", "f2"]
    ref_centroids = pd.DataFrame([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], columns=cols)
    ref_centroids.index = pd.Index([0, 1], name="class")
    # A symmetric positive-definite precision (not diagonal), so the cross terms are exercised.
    a = np.array([[2.0, 0.3, -0.1], [0.3, 1.5, 0.2], [-0.1, 0.2, 1.8]])
    precision = a @ a.T
    reference = _reference(ref_centroids, precision, pd.Series([0, 1], index=["p0", "p1"]))

    stratum_centroids = pd.DataFrame([[0.2, -0.1, 0.3], [1.4, 0.9, 1.1]], columns=cols)
    stratum_centroids.index = pd.Index([0, 1], name="class")
    stratum = _stratum(stratum_centroids)

    movement = _movement(reference, stratum, ref_class=0, fit_class=0)
    contributions = attr.MahalanobisContribution().contributions(movement)
    distance = drift.Mahalanobis().class_distance(stratum, 0, reference, 0)

    assert contributions.index.tolist() == cols
    np.testing.assert_allclose(contributions.sum(), distance**2, rtol=1e-10)


def test_standardised_contributions_are_squared_standardised_shift() -> None:
    """The diagonal split is the squared per-feature shift in pooled-SD units, non-negative."""
    cols = ["f0", "f1"]
    ref_centroids = pd.DataFrame([[0.0, 0.0]], columns=cols)
    ref_centroids.index = pd.Index([0], name="class")
    reference = _reference(ref_centroids, np.eye(2), pd.Series([0], index=["p0"]))
    stratum_centroids = pd.DataFrame([[0.5, -2.0]], columns=cols)
    stratum_centroids.index = pd.Index([0], name="class")
    stratum = _stratum(stratum_centroids)

    contributions = attr.StandardisedContribution().contributions(
        _movement(reference, stratum, 0, 0)
    )
    assert (contributions >= 0).all()
    np.testing.assert_allclose(contributions.to_numpy(), [0.25, 4.0])


def test_category_totals_sum_within_category() -> None:
    """Per-feature contributions aggregate to signed category totals, sorted descending."""
    contributions = pd.Series({"a": 1.0, "b": 2.0, "c": -0.5}, name="contribution")
    category_map = {"a": "social", "b": "social", "c": "developmental"}
    totals = attr.category_totals(contributions, category_map)
    assert totals.loc["social"] == 3.0
    assert totals.loc["developmental"] == -0.5
    assert totals.index[0] == "social"  # sorted by descending contribution


def test_category_totals_keep_blank_and_absent_features() -> None:
    """A blank (NaN) or absent category resolves to 'unmapped' and its contribution is kept.

    The category totals must sum to the same value as the per-feature contributions, so a blank
    category cannot silently drop a feature (which a NaN group in a pandas ``groupby`` would).
    """
    contributions = pd.Series({"a": 1.0, "b": 2.0, "c": -0.5}, name="contribution")
    category_map = {"a": "social", "b": float("nan")}  # b blank, c absent
    assert attr.category_of("b", category_map) == "unmapped"
    assert attr.category_of("c", category_map) == "unmapped"
    totals = attr.category_totals(contributions, category_map)
    assert totals.loc["unmapped"] == 1.5  # b (2.0) + c (-0.5), not dropped
    np.testing.assert_allclose(totals.sum(), contributions.sum())


def test_movers_flags_probands_that_left_the_class() -> None:
    """A reference-class-k proband is a mover when its aligned second-fit class is not k."""
    ref_labels = pd.Series([0, 0, 0, 1, 1, 1], index=[f"p{i}" for i in range(6)])
    fit_labels = pd.Series([10, 11, 10, 11, 10, 11], index=[f"p{i}" for i in range(6)])
    alignment = drift.ClassAlignment(mapping={10: 0, 11: 1}, quality={0: 1.0, 1: 1.0}, overall=1.0)
    cols = ["f0"]
    centroids = pd.DataFrame([[0.0], [1.0]], columns=cols)
    centroids.index = pd.Index([0, 1], name="class")
    reference = _reference(centroids, np.eye(1), ref_labels)
    comparison = attr.Comparison(
        reference=reference,
        stratum=_stratum(centroids),
        ref_labels=ref_labels,
        fit_labels=fit_labels,
        alignment=alignment,
    )
    movement = attr.Movement(comparison, ref_class=0, fit_class=10)

    leavers = attr.movers(movement, kind="leavers")
    assert leavers.index.tolist() == ["p0", "p1", "p2"]
    assert leavers.tolist() == [False, True, False]

    joiners = attr.movers(movement, kind="joiners")
    assert joiners.index.tolist() == ["p0", "p2", "p4"]
    assert joiners.tolist() == [False, False, True]

    either = attr.movers(movement, kind="either")  # the default: churn against the stable core
    assert either.index.tolist() == ["p0", "p1", "p2", "p4"]
    assert either.tolist() == [False, True, False, True]

    assert attr.membership_counts(movement) == {"n_stayers": 2, "n_leavers": 1, "n_joiners": 1}


def test_comparison_movements_one_per_reference_class() -> None:
    """A comparison yields one movement per reference class, in class order."""
    cols = ["f0"]
    centroids = pd.DataFrame([[0.0], [1.0]], columns=cols)
    centroids.index = pd.Index([0, 1], name="class")
    reference = _reference(centroids, np.eye(1), pd.Series([0, 1], index=["p0", "p1"]))
    alignment = drift.ClassAlignment(mapping={7: 1, 8: 0}, quality={0: 1.0, 1: 1.0}, overall=1.0)
    comparison = attr.Comparison(
        reference=reference,
        stratum=_stratum(centroids),
        ref_labels=reference.labels,
        fit_labels=pd.Series(dtype=int),
        alignment=alignment,
    )
    movements = comparison.movements()
    assert [(m.ref_class, m.fit_class) for m in movements] == [(0, 8), (1, 7)]


def _planted_contrast_frame(seed: int) -> tuple[pd.Series, pd.DataFrame]:
    """Movers and stayers where one feature separates them and two are noise."""
    rng = np.random.default_rng(seed)
    n = 120
    moved = pd.Series([True] * (n // 2) + [False] * (n // 2), index=[f"p{i}" for i in range(n)])
    signal = np.where(moved.to_numpy(), 1.0, 0.0) + rng.normal(0, 0.3, n)
    features = pd.DataFrame(
        {
            "signal": signal,
            "noise_a": rng.normal(0, 1, n),
            "noise_b": rng.normal(0, 1, n),
        },
        index=moved.index,
    )
    return moved, features


def test_univariate_contrast_recovers_planted_signal() -> None:
    """The separating feature ranks first, with a positive effect and FDR significance."""
    moved, features = _planted_contrast_frame(seed=0)
    result = attr.UnivariateContrast().contrast(moved, features)
    assert result.n_movers == 60
    assert result.n_stayers == 60
    top = result.importances.iloc[0]
    assert top["feature"] == "signal"
    assert top["effect"] > 0
    assert bool(top["fdr_significant"])


def test_logistic_contrast_recovers_planted_signal() -> None:
    """The penalised model gives the separating feature the largest signed coefficient."""
    moved, features = _planted_contrast_frame(seed=1)
    result = attr.LogisticContrast().contrast(moved, features)
    top = result.importances.iloc[0]
    assert top["feature"] == "signal"
    assert top["effect"] > 0


def test_contrast_degenerate_when_one_group_empty() -> None:
    """With no stayers the contrast returns an empty ranking but reports the counts."""
    moved = pd.Series([True, True, True], index=["p0", "p1", "p2"])
    features = pd.DataFrame({"f0": [1.0, 2.0, 3.0]}, index=moved.index)
    result = attr.UnivariateContrast().contrast(moved, features)
    assert result.n_movers == 3
    assert result.n_stayers == 0
    assert result.importances.empty
