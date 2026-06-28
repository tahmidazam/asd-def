"""Binning policies for the stratification axes.

The stratified analysis (plan section 7) re-estimates the mixture model within strata of a
continuous variable: age at diagnosis in years, or the derived calendar year of diagnosis.
A *binning policy* maps that variable onto an ordered set of named strata. The downstream
fit consumes a :class:`StratumAssignment` and never sees which policy produced it, so the
analysis is independent of the binning choice and a policy can be swapped without touching
the fitting code.

Three policies are defined, a substantive scheme plus two distribution-free ones:

- :class:`FixedBands` cuts the variable at explicit, substantively motivated edges: clinical
  age bands, or calendar-era bands anchored on the DSM-5 boundary. The edges are a parameter,
  frozen at pre-registration (plan section 12) and provisional until then.
- :class:`QuantileBins` cuts the variable at its own empirical quantiles, giving
  equal-frequency strata whose edges follow the cohort rather than a fixed rule.
- :class:`MaxEqualBins` is the same equal-frequency idea but with the bin count chosen from
  the cohort size and the minimum stratum size, the finest split that keeps every bin above
  the floor, rather than a fixed number of bins.

Both use left-closed, right-open intervals :math:`[lo, hi)` with open outer bins, so a value
that lands on an interior edge falls in the upper band. For the era axis this places a
diagnosis recorded in a boundary year on the later side, matching DSM-5 taking effect in
2013.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StratumAssignment:
    """An ordered partition of a continuous variable into named strata.

    Attributes
    ----------
    labels : list of str
        The stratum names, in ascending order of the variable.
    edges : list of float
        The finite interior cut points, of length ``len(labels) - 1``. The outer bins are
        open, so the full partition is ``(-inf, edges[0])``, ``[edges[0], edges[1])``, ...,
        ``[edges[-1], +inf)``.
    codes : pandas.Series
        The per-row stratum label, an ordered categorical indexed as the input. Rows whose
        value is missing are unassigned (categorical ``NaN``).
    counts : dict of str to int
        Assigned rows per stratum, in label order.
    n_missing : int
        Rows dropped because the variable was missing.
    spec : dict
        The serialisable policy specification, recorded in the run manifest and the frozen
        pre-registration so a stratification is reproducible from it alone.
    """

    labels: list[str]
    edges: list[float]
    codes: pd.Series
    counts: dict[str, int]
    n_missing: int
    spec: dict[str, object]


@runtime_checkable
class BinningPolicy(Protocol):
    """The interface every stratification axis depends on.

    A policy turns a continuous variable into a :class:`StratumAssignment`. Downstream code
    is typed against this protocol, never against a concrete policy, which is what keeps the
    analysis independent of the binning choice.
    """

    name: str

    def assign(self, values: pd.Series) -> StratumAssignment:
        """Partition ``values`` into ordered, named strata."""
        ...

    def spec(self) -> dict[str, object]:
        """Return the serialisable policy specification."""
        ...


def _default_band_labels(edges: tuple[float, ...]) -> list[str]:
    """Build readable band labels from interior cut points.

    The first and last bands are open (``<e0`` and ``>=e_last``); each interior band is
    named by its half-open range. The ``:g`` format drops trailing zeros, so an integer edge
    renders without a decimal point.
    """
    rendered = [f"{edge:g}" for edge in edges]
    labels = [f"<{rendered[0]}"]
    labels += [f"{rendered[i]}-{rendered[i + 1]}" for i in range(len(rendered) - 1)]
    labels.append(f">={rendered[-1]}")
    return labels


def _assign(
    values: pd.Series,
    interior_edges: tuple[float, ...],
    labels: list[str],
    spec: dict[str, object],
) -> StratumAssignment:
    """Cut ``values`` at ``interior_edges`` into the ordered ``labels``.

    Shared by both policies. Uses left-closed, right-open intervals with open outer bins, so
    every finite value is assigned and a value on an interior edge falls in the upper band.
    """
    bins = [-np.inf, *interior_edges, np.inf]
    codes = pd.cut(values, bins=bins, labels=labels, right=False, ordered=True)
    counts = {label: int((codes == label).sum()) for label in labels}
    return StratumAssignment(
        labels=list(labels),
        edges=[float(edge) for edge in interior_edges],
        codes=codes,
        counts=counts,
        n_missing=int(values.isna().sum()),
        spec=spec,
    )


@dataclass(frozen=True)
class FixedBands:
    """Cut at explicit, substantively motivated edges (plan section 7).

    The edges are interior cut points in the units of the axis (age at diagnosis in years,
    or era as a calendar year). They are provisional until frozen at pre-registration.

    Attributes
    ----------
    edges : tuple of float
        Strictly ascending interior cut points; ``k`` edges give ``k + 1`` bands.
    labels : tuple of str, optional
        Band names in ascending order. Defaults to half-open range labels.
    name : str
        Policy name recorded in the spec.
    """

    edges: tuple[float, ...]
    labels: tuple[str, ...] | None = None
    name: str = "fixed"

    def __post_init__(self) -> None:
        """Validate that the edges are ascending and the label count matches."""
        if not self.edges:
            raise ValueError("FixedBands needs at least one edge.")
        if list(self.edges) != sorted(set(self.edges)):
            raise ValueError("FixedBands edges must be strictly ascending and unique.")
        if self.labels is not None and len(self.labels) != len(self.edges) + 1:
            raise ValueError(
                f"FixedBands needs {len(self.edges) + 1} labels for {len(self.edges)} edges, "
                f"got {len(self.labels)}."
            )

    def _labels(self) -> list[str]:
        return list(self.labels) if self.labels is not None else _default_band_labels(self.edges)

    def assign(self, values: pd.Series) -> StratumAssignment:
        """Partition ``values`` at the fixed edges."""
        return _assign(values, self.edges, self._labels(), self.spec())

    def spec(self) -> dict[str, object]:
        """Return the serialisable specification, including the edges."""
        return {
            "policy": self.name,
            "interval": "left-closed",
            "edges": [float(edge) for edge in self.edges],
            "labels": self._labels(),
        }


@dataclass(frozen=True)
class QuantileBins:
    """Cut at the variable's own empirical quantiles, for equal-frequency strata.

    The realised edges depend on the cohort, so the spec records the intended number of bins
    ``q`` and the assignment records the edges the cohort produced. Ties can collapse bins,
    so the realised count of strata can be below ``q``.

    Attributes
    ----------
    q : int
        The number of equal-frequency bins requested (at least 2).
    name : str
        Policy name recorded in the spec.
    """

    q: int
    name: str = "quantile"

    def __post_init__(self) -> None:
        """Validate that at least two bins are requested."""
        if self.q < 2:
            raise ValueError("QuantileBins needs q >= 2.")

    def assign(self, values: pd.Series) -> StratumAssignment:
        """Partition ``values`` at its interior quantiles."""
        finite = values.dropna()
        probabilities = np.linspace(0.0, 1.0, self.q + 1)[1:-1]
        interior = tuple(float(edge) for edge in np.unique(np.quantile(finite, probabilities)))
        labels = [f"Q{i + 1}" for i in range(len(interior) + 1)]
        spec = {**self.spec(), "edges": list(interior)}
        return _assign(values, interior, labels, spec)

    def spec(self) -> dict[str, object]:
        """Return the serialisable specification (the intended ``q``)."""
        return {"policy": self.name, "interval": "left-closed", "q": self.q}


@dataclass(frozen=True)
class MaxEqualBins:
    """Equal-frequency bins, as many as keep every bin above a minimum size.

    The bin count is not fixed: it is the largest ``q`` whose equal-frequency split still
    leaves every bin at or above ``min_bin_size``, starting from ``floor(n / min_bin_size)``
    and stepping down if ties or skew leave a bin short. The result is the finest
    equal-frequency partition that clears the floor, so the resolution follows the cohort size
    and the floor rather than a hand-picked ``q``. Like :class:`QuantileBins`, the edges are
    the variable's own quantiles, so the choice stays on the design side of the
    pre-registration firewall.

    Attributes
    ----------
    min_bin_size : int
        The size every bin must clear; sets both the starting bin count and the floor the
        step-down enforces. Defaults to the phase-2 recovery floor.
    name : str
        Policy name recorded in the spec.
    """

    min_bin_size: int = 1000
    name: str = "max-equal"

    def __post_init__(self) -> None:
        """Validate that the floor is positive."""
        if self.min_bin_size < 1:
            raise ValueError("MaxEqualBins needs min_bin_size >= 1.")

    @staticmethod
    def _cuts(finite: pd.Series, q: int) -> tuple[tuple[float, ...], list[str]]:
        probabilities = np.linspace(0.0, 1.0, q + 1)[1:-1]
        interior = tuple(float(edge) for edge in np.unique(np.quantile(finite, probabilities)))
        labels = [f"Q{i + 1}" for i in range(len(interior) + 1)]
        return interior, labels

    def assign(self, values: pd.Series) -> StratumAssignment:
        """Partition ``values`` into the finest equal-frequency split above the floor."""
        finite = values.dropna()
        q = max(2, int(len(finite) // self.min_bin_size))
        interior, labels = self._cuts(finite, q)
        while q > 2:
            counts = _assign(values, interior, labels, {}).counts
            if min(counts.values()) >= self.min_bin_size:
                break
            q -= 1
            interior, labels = self._cuts(finite, q)
        spec = {**self.spec(), "q_realised": len(labels), "edges": list(interior)}
        return _assign(values, interior, labels, spec)

    def spec(self) -> dict[str, object]:
        """Return the serialisable specification (the floor that sets the bin count)."""
        return {"policy": self.name, "interval": "left-closed", "min_bin_size": self.min_bin_size}


# Provisional parameter sets, illustrative only and NOT the frozen pre-registration values
# (plan section 12 freezes those after the distribution and feasibility check on this branch).
# Edges are in the units of each axis: age at diagnosis in years, era as a calendar year. The
# era bands are anchored on the DSM-5 boundary (2013); a boundary-year diagnosis falls on the
# later side under the left-closed convention.
PROVISIONAL_AGE_BANDS = FixedBands(edges=(4, 7, 11), labels=("<4", "4-6", "7-10", ">=11"))
PROVISIONAL_ERA_BANDS = FixedBands(
    edges=(2013, 2017, 2021), labels=("<=2012", "2013-2016", "2017-2020", ">=2021")
)
PROVISIONAL_QUANTILES = QuantileBins(q=4)
