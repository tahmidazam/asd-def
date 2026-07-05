r"""Reference schemes: what each local fit's drift is measured against.

The drift stage (:mod:`analysis.drift`) measures how far the reference classes move when the
mixture is re-estimated within a stratum, and it compares every stratum to one fixed target, the
pooled reference (section 6a). That target is itself a design choice with consequences for what
the drift means, so this module turns it into a pluggable axis, orthogonal to the localisation
scheme (:mod:`analysis.localise`) that forms the fits. A run picks a reference scheme the same way
it picks an alignment or a distance.

A stratum is a proper subset of the pooled cohort, so comparing it to the pooled reference has
the stratum contribute to the target it is judged against. The size-matched permutation null
(plan section 12a) calibrates that overlap away rather than removing it: a null pseudo-stratum is
a subset of the same size, so it carries the same overlap, and the residual pull is conservative,
because a stratum's own members draw the pooled centroids toward it and so understate its drift.
Two alternative targets remove the overlap by construction instead:

- :class:`PooledReference` (the frozen primary) compares each stratum to the pooled reference. The
  stratum and the reference share probands, so membership alignment applies and a class that moved
  can be told from one that reorganised.
- :class:`PairwiseReference` compares each stratum to a neighbouring stratum. The two are disjoint,
  so nothing is shared and the comparison is independent, at the cost of centroid alignment (the
  move-versus-reorganise distinction needs shared members). It reads change along the axis rather
  than distance from a single pooled partition, the honest test when the existence of one reference
  structure is itself in question.

The work splits in two. Topology (which fit pairs with which) is a :class:`Pairing`, a pure
function of the fit labels and positions, so it is tested without any fit. Resolution turns a
pairing into a :class:`DriftComparison` by building the concrete
:class:`~analysis.drift.ReferenceModel` for it through a :class:`ReferenceResolver`, which the
caller backs with the cached fits. The measurement itself is unchanged:
:func:`~analysis.drift.compute_drift` runs over the query summary and the resolved reference, so no
distance or alignment code is duplicated here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from analysis.drift import (
    ALIGNMENTS,
    DEFAULT_DISTANCE,
    DISTANCES,
    DriftResult,
    ReferenceModel,
    StratumSummary,
    compute_drift,
)


@dataclass(frozen=True)
class QueryFit:
    """The query side of a comparison: one local fit and where it sits on the axis.

    Attributes
    ----------
    label : str
        The fit's name, unique within a run (a stratum name, or ``focal=6.5``).
    position : float
        The fit's location on the axis, in the axis units, used to order neighbours for the
        pairwise topology.
    summary : analysis.drift.StratumSummary
        The fit's method-independent summary (centroids, dispersions, contingency).
    """

    label: str
    position: float
    summary: StratumSummary


@dataclass(frozen=True)
class Pairing:
    """A pure-topology comparison: which fit is the query and where its reference comes from.

    Carries no fit data, so a scheme's topology is decided and tested without building any
    reference. :func:`resolve_comparisons` turns a pairing into a concrete
    :class:`DriftComparison`.

    Attributes
    ----------
    query_label : str
        The fit whose drift is measured.
    reference_kind : str
        How the reference is obtained: ``"pooled"`` (the fixed pooled reference) or ``"promote"``
        (build a reference from the ``reference_label`` fit).
    reference_label : str or None
        The other fit a ``"promote"`` pairing draws its reference from; ``None`` otherwise.
    """

    query_label: str
    reference_kind: str
    reference_label: str | None = None


@dataclass(frozen=True)
class DriftComparison:
    """A resolved comparison: a query summary against a concrete reference, with its alignment.

    Attributes
    ----------
    query_label : str
        The query fit's name.
    position : float
        The query fit's position on the axis.
    query : analysis.drift.StratumSummary
        The query fit's summary.
    reference : analysis.drift.ReferenceModel
        The target the query is aligned and measured against.
    alignment : str
        The alignment method's registry name (``"membership"`` when the two share probands,
        ``"centroid"`` when they are disjoint).
    reference_label : str or None
        The fit the reference was built from, for a pairwise or leave-one-out comparison; ``None``
        for the pooled reference.
    """

    query_label: str
    position: float
    query: StratumSummary
    reference: ReferenceModel
    alignment: str
    reference_label: str | None = None


@runtime_checkable
class ReferenceResolver(Protocol):
    """Build the concrete reference model a pairing names, backed by the caller's fits.

    A scheme names its references abstractly (the pooled reference, or a named neighbour); the
    resolver turns each name into a :class:`~analysis.drift.ReferenceModel`. The caller implements
    it over the cached fits, so a pairwise reference is built from a neighbour stratum with no
    re-fit.
    """

    def pooled(self) -> ReferenceModel:
        """Return the fixed pooled reference."""
        ...

    def promote(self, label: str) -> ReferenceModel:
        """Return the fit named ``label`` promoted to a reference (its centroids as a target)."""
        ...


@dataclass
class MappingResolver:
    """Resolve references from a fixed pooled reference and a per-label promoted map.

    The concrete resolver the drift stage uses. ``pooled`` returns the pooled reference; ``promote``
    looks a fit's own promoted reference up in ``promoted``, which the caller has built once from
    each cached fit, so a pairwise comparison reuses a neighbour fit with no re-fit.

    Attributes
    ----------
    pooled_reference : analysis.drift.ReferenceModel
        The fixed pooled reference.
    promoted : Mapping of str to analysis.drift.ReferenceModel
        Each fit's own promoted reference, keyed by the fit label.
    """

    pooled_reference: ReferenceModel
    promoted: Mapping[str, ReferenceModel]

    def pooled(self) -> ReferenceModel:
        """Return the fixed pooled reference."""
        return self.pooled_reference

    def promote(self, label: str) -> ReferenceModel:
        """Return the promoted reference for the fit named ``label``."""
        return self.promoted[label]


@runtime_checkable
class ReferenceScheme(Protocol):
    """Decide which fit each drift is measured against."""

    name: str
    default_alignment: str

    def pairings(self, ordered: list[tuple[str, float]]) -> list[Pairing]:
        """Return the comparison topology for fits given as ``(label, position)`` pairs."""
        ...

    def spec(self) -> dict[str, object]:
        """Return the serialisable scheme specification for the run manifest."""
        ...


@dataclass(frozen=True)
class PooledReference:
    """Compare every stratum to the pooled reference: the frozen confirmatory primary.

    The stratum is a subset of the pooled cohort, so both labellings exist on its probands and
    membership alignment applies. This reproduces the current drift stage exactly, so it is the
    regression anchor for the abstraction.
    """

    name: str = "pooled"
    default_alignment: str = "membership"

    def pairings(self, ordered: list[tuple[str, float]]) -> list[Pairing]:
        """One pooled-reference pairing per fit, in the given order."""
        return [Pairing(label, "pooled") for label, _ in ordered]

    def spec(self) -> dict[str, object]:
        """Return the specification (the scheme name and its alignment default)."""
        return {"scheme": self.name, "alignment": self.default_alignment}


@dataclass(frozen=True)
class PairwiseReference:
    """Compare each stratum to another stratum, so the comparison carries no self-overlap.

    The two strata are disjoint, so alignment is by centroid only. ``mode`` sets which pairs are
    formed: ``"adjacent"`` (the default) pairs each fit with its successor along the axis, a
    sequence of local comparisons that reads change along the axis; ``"all-pairs"`` compares every
    earlier fit to every later one. Fits are ordered by position, and each query's reference is the
    later fit, so the direction of every comparison runs along the axis.

    Attributes
    ----------
    mode : str
        ``"adjacent"`` or ``"all-pairs"``.
    """

    mode: str = "adjacent"
    name: str = "pairwise"
    default_alignment: str = "centroid"

    def __post_init__(self) -> None:
        """Validate the pairing mode."""
        if self.mode not in ("adjacent", "all-pairs"):
            raise ValueError(f"mode must be 'adjacent' or 'all-pairs', not {self.mode!r}")

    def pairings(self, ordered: list[tuple[str, float]]) -> list[Pairing]:
        """Pair each fit with a later fit, ``adjacent`` by default, ``all-pairs`` otherwise."""
        labels = [label for label, _ in sorted(ordered, key=lambda item: item[1])]
        pairs: list[Pairing] = []
        if self.mode == "adjacent":
            for query, reference in zip(labels[:-1], labels[1:], strict=True):
                pairs.append(Pairing(query, "promote", reference))
        else:
            for i, query in enumerate(labels):
                for reference in labels[i + 1 :]:
                    pairs.append(Pairing(query, "promote", reference))
        return pairs

    def spec(self) -> dict[str, object]:
        """Return the specification (the scheme name, its mode, and its alignment default)."""
        return {"scheme": self.name, "mode": self.mode, "alignment": self.default_alignment}


REFERENCE_SCHEMES: dict[str, ReferenceScheme] = {
    "pooled": PooledReference(),
    "pairwise": PairwiseReference(),
}
DEFAULT_REFERENCE_SCHEME = "pooled"


def resolve_comparisons(
    scheme: ReferenceScheme, queries: list[QueryFit], resolver: ReferenceResolver
) -> list[DriftComparison]:
    """Turn a scheme's topology into resolved comparisons, ready to measure.

    Reads the topology from ``scheme.pairings`` over the query labels and positions, then builds
    each pairing's reference through ``resolver``: the pooled reference for a ``"pooled"`` pairing,
    or the named neighbour promoted to a reference for a ``"promote"`` pairing. The alignment is the
    scheme's default, which the caller may override per comparison.
    """
    by_label = {q.label: q for q in queries}
    ordered = [(q.label, q.position) for q in queries]
    comparisons: list[DriftComparison] = []
    for pairing in scheme.pairings(ordered):
        query = by_label[pairing.query_label]
        if pairing.reference_kind == "pooled":
            reference = resolver.pooled()
        elif pairing.reference_kind == "promote":
            if pairing.reference_label is None:
                raise ValueError("a 'promote' pairing needs a reference label")
            reference = resolver.promote(pairing.reference_label)
        else:
            raise ValueError(f"unknown reference kind {pairing.reference_kind!r}")
        comparisons.append(
            DriftComparison(
                query_label=query.label,
                position=query.position,
                query=query.summary,
                reference=reference,
                alignment=scheme.default_alignment,
                reference_label=pairing.reference_label,
            )
        )
    return comparisons


def measure(
    comparisons: list[DriftComparison], distance: str = DEFAULT_DISTANCE
) -> dict[str, DriftResult]:
    """Measure each resolved comparison's drift, keyed by the query label.

    A thin pass over :func:`~analysis.drift.compute_drift`: each comparison aligns its query to its
    reference with its own alignment method and measures every aligned class with the shared
    ``distance``. No fitting happens here, so re-measuring with a different distance is cheap.
    """
    distancer = DISTANCES[distance]
    return {
        comparison.query_label: compute_drift(
            comparison.query,
            comparison.reference,
            ALIGNMENTS[comparison.alignment],
            distancer,
        )
        for comparison in comparisons
    }
