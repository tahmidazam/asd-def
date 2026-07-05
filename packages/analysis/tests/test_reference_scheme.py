"""Unit tests for the reference-scheme abstraction, on synthetic data only.

No participant data is read here (the governance rule): the summaries, reference models, and
labellings are constructed in memory. The tests fix the comparison topology (which fit pairs with
which), confirm the pooled scheme reproduces :func:`analysis.drift.compute_drift` exactly (the
regression anchor for the abstraction), and check the pairwise scheme promotes the right neighbour
and aligns by centroid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analysis import drift
from analysis import reference_scheme as rs


def _reference(centroids: pd.DataFrame, precision: np.ndarray) -> drift.ReferenceModel:
    """Build a reference model with unit dispersions and pooled spread from given centroids."""
    dispersions = pd.DataFrame(
        np.ones_like(centroids.to_numpy()), index=centroids.index, columns=centroids.columns
    )
    pooled_sd = pd.Series(np.ones(centroids.shape[1]), index=centroids.columns)
    labels = pd.Series(list(centroids.index), index=[f"p{i}" for i in range(len(centroids))])
    return drift.ReferenceModel(
        centroids=centroids,
        dispersions=dispersions,
        pooled_sd=pooled_sd,
        precision=precision,
        labels=labels,
    )


def _stratum(centroids: pd.DataFrame, contingency: pd.DataFrame) -> drift.StratumSummary:
    """A stratum summary with unit dispersions and an explicit contingency."""
    dispersions = pd.DataFrame(
        np.ones_like(centroids.to_numpy()), index=centroids.index, columns=centroids.columns
    )
    return drift.StratumSummary(
        centroids, dispersions, contingency, n=int(contingency.to_numpy().sum())
    )


def _two_class_centroids(rows: list[list[float]]) -> pd.DataFrame:
    """Two-class centroids over three features, indexed as a class axis."""
    frame = pd.DataFrame(rows, columns=["f0", "f1", "f2"])
    frame.index = pd.Index(range(len(rows)), name="class")
    return frame


class _Resolver:
    """A resolver backed by in-memory reference models, for the resolution tests."""

    def __init__(
        self, pooled: drift.ReferenceModel, promoted: dict[str, drift.ReferenceModel]
    ) -> None:
        self._pooled = pooled
        self._promoted = promoted

    def pooled(self) -> drift.ReferenceModel:
        return self._pooled

    def promote(self, label: str) -> drift.ReferenceModel:
        return self._promoted[label]

    def complement(self, label: str) -> drift.ReferenceModel:  # pragma: no cover - not built yet
        raise NotImplementedError


def test_pooled_pairings_one_per_fit_in_order() -> None:
    """The pooled scheme emits one pooled-reference pairing per fit, order preserved."""
    ordered = [("a", 1.0), ("b", 2.0), ("c", 3.0)]
    pairings = rs.PooledReference().pairings(ordered)
    assert [p.query_label for p in pairings] == ["a", "b", "c"]
    assert all(p.reference_kind == "pooled" and p.reference_label is None for p in pairings)


def test_pairwise_adjacent_pairings_order_by_position() -> None:
    """Adjacent pairwise pairs each fit with its successor, whatever the input order."""
    # Deliberately out of position order to check the scheme sorts before pairing.
    ordered = [("c", 3.0), ("a", 1.0), ("b", 2.0)]
    pairings = rs.PairwiseReference(mode="adjacent").pairings(ordered)
    assert [(p.query_label, p.reference_label) for p in pairings] == [("a", "b"), ("b", "c")]
    assert all(p.reference_kind == "promote" for p in pairings)


def test_pairwise_all_pairs_pairings_count() -> None:
    """All-pairs pairwise compares every earlier fit to every later one."""
    ordered = [("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)]
    pairings = rs.PairwiseReference(mode="all-pairs").pairings(ordered)
    assert len(pairings) == 4 * 3 // 2
    assert ("a", "d") in [(p.query_label, p.reference_label) for p in pairings]
    assert all(p.query_label != p.reference_label for p in pairings)


def test_pairwise_rejects_unknown_mode() -> None:
    """An unknown pairing mode is rejected at construction."""
    with pytest.raises(ValueError, match="mode must be"):
        rs.PairwiseReference(mode="triples")


def test_resolve_pooled_uses_pooled_reference_and_membership() -> None:
    """A pooled comparison targets the pooled reference and aligns by membership."""
    centroids = _two_class_centroids([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    reference = _reference(centroids, np.eye(3))
    contingency = pd.DataFrame([[10, 2], [1, 8]], index=[0, 1], columns=[0, 1])
    stratum = _stratum(centroids, contingency)
    resolver = _Resolver(reference, {})

    comparisons = rs.resolve_comparisons(
        rs.PooledReference(), [rs.QueryFit("q", 5.0, stratum)], resolver
    )
    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison.reference is reference
    assert comparison.alignment == "membership"
    assert comparison.reference_label is None


def test_resolve_pairwise_promotes_neighbour_and_uses_centroid() -> None:
    """A pairwise comparison targets the promoted neighbour and aligns by centroid."""
    centroids = _two_class_centroids([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    pooled = _reference(centroids, np.eye(3))
    neighbour = _reference(centroids, np.eye(3))
    identity = pd.DataFrame(np.eye(2, dtype=int), index=[0, 1], columns=[0, 1])
    queries = [
        rs.QueryFit("a", 1.0, _stratum(centroids, identity)),
        rs.QueryFit("b", 2.0, _stratum(centroids, identity)),
    ]
    resolver = _Resolver(pooled, {"b": neighbour})

    comparisons = rs.resolve_comparisons(rs.PairwiseReference(), queries, resolver)
    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison.query_label == "a"
    assert comparison.reference_label == "b"
    assert comparison.reference is neighbour
    assert comparison.alignment == "centroid"


def test_resolve_promote_pairing_without_label_raises() -> None:
    """A malformed promote pairing (no reference label) is rejected during resolution."""

    class _BadScheme:
        name = "bad"
        default_alignment = "centroid"

        def pairings(self, ordered: list[tuple[str, float]]) -> list[rs.Pairing]:
            return [rs.Pairing("a", "promote", None)]

        def spec(self) -> dict[str, object]:
            return {}

    identity = pd.DataFrame(np.eye(2, dtype=int), index=[0, 1], columns=[0, 1])
    centroids = _two_class_centroids([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    resolver = _Resolver(_reference(centroids, np.eye(3)), {})
    with pytest.raises(ValueError, match="needs a reference label"):
        rs.resolve_comparisons(
            _BadScheme(), [rs.QueryFit("a", 1.0, _stratum(centroids, identity))], resolver
        )


def test_measure_pooled_matches_compute_drift() -> None:
    """The pooled scheme reproduces ``compute_drift`` exactly: the regression anchor.

    Measuring through the abstraction must equal calling the drift stage directly with the same
    membership alignment and Mahalanobis distance, so the abstraction changes nothing on the frozen
    primary path.
    """
    ref_centroids = _two_class_centroids([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    a = np.array([[2.0, 0.3, -0.1], [0.3, 1.5, 0.2], [-0.1, 0.2, 1.8]])
    reference = _reference(ref_centroids, a @ a.T)
    stratum_centroids = _two_class_centroids([[0.2, -0.1, 0.3], [1.4, 0.9, 1.1]])
    contingency = pd.DataFrame([[9, 1], [2, 7]], index=[0, 1], columns=[0, 1])
    stratum = _stratum(stratum_centroids, contingency)
    resolver = _Resolver(reference, {})

    measured = rs.measure(
        rs.resolve_comparisons(rs.PooledReference(), [rs.QueryFit("q", 0.0, stratum)], resolver)
    )
    expected = drift.compute_drift(
        stratum,
        reference,
        drift.ALIGNMENTS[rs.PooledReference().default_alignment],
        drift.DISTANCES[rs.DEFAULT_DISTANCE],
    )

    assert measured["q"].distances == expected.distances
    assert measured["q"].alignment.mapping == expected.alignment.mapping


def test_measure_pairwise_against_self_is_zero() -> None:
    """A stratum promoted as its own pairwise reference has no drift (identical centroids)."""
    centroids = _two_class_centroids([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    identity = pd.DataFrame(np.eye(2, dtype=int), index=[0, 1], columns=[0, 1])
    stratum = _stratum(centroids, identity)
    self_reference = _reference(centroids, np.eye(3))
    queries = [rs.QueryFit("a", 1.0, stratum), rs.QueryFit("b", 2.0, stratum)]
    resolver = _Resolver(self_reference, {"b": self_reference})

    measured = rs.measure(rs.resolve_comparisons(rs.PairwiseReference(), queries, resolver))
    assert set(measured) == {"a"}
    assert all(abs(distance) < 1e-9 for distance in measured["a"].distances.values())
