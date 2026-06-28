r"""Acceptance requirements a binning policy must meet before it is frozen.

A binning policy (:mod:`analysis.strata`) is only eligible for the confirmatory stratified
fit (plan section 7) if its partition can actually be fitted and is not silently confounded.
This module evaluates a concrete policy against a tiered requirement set, computed from the
cohort, the stratifying variable, and (optionally) the measurement-to-diagnosis lag and a
covariate frame. Nothing here fits a per-stratum mixture model or reads class drift, so the
checks stay on the design side of the pre-registration firewall: they decide whether a
partition is eligible to be fitted, not whether the fit gives a wanted answer.

Three tiers:

- Tier 1, eligibility gates (hard pass or fail). Every bin clears the empirical
  :math:`N_\text{min}` floor and the projected smallest-class floor, coverage of the
  modelling cohort is high, and the partition is valid. A policy that fails any Tier 1 gate
  is ineligible.
- Tier 2, confound and balance (reported with a flag, not a hard fail). Size balance, lag
  entanglement and small-lag retention for the era axis, covariate balance, and edge
  robustness. These inform the covariate-versus-subsample decision rather than reject a
  policy outright.
- Tier 3, demographics. A per-bin summary with standardised differences across the extreme
  bins, both the manuscript's per-stratum table and a check that any drift is not trivially a
  composition artefact.

Thresholds are settable (:class:`RequirementThresholds`) and recorded in the report, so the
frozen pre-registration carries the exact values a policy was judged against. The defaults
follow the phase-2 findings: a per-bin floor of 1000 (the practical recovery floor from the
:math:`N_\text{min}` sweep) and a smallest class near 15 per cent of a bin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis.enrich import cohens_d
from analysis.strata import BinningPolicy, FixedBands, StratumAssignment


@dataclass(frozen=True)
class RequirementThresholds:
    """The numeric criteria a policy is judged against.

    Attributes
    ----------
    min_bin_size : int
        Smallest allowed assigned count in any bin (the phase-2 recovery floor).
    smallest_class_fraction : float
        Reference smallest-class proportion, used to project a bin's smallest class.
    min_projected_smallest_class : int
        Smallest allowed projected smallest-class count (bin size times the fraction).
    min_coverage : float
        Smallest allowed fraction of the modelling cohort assigned (non-missing variable).
    min_n_bins : int
        Fewest non-empty bins (a contrast needs at least two).
    max_bin_share : float
        Largest share of the assigned cohort a single bin may hold before it is flagged.
    small_lag_years : float
        Lag cut for the small-lag subsample used to probe the era axis.
    max_lag_correlation : float
        Largest tolerated Spearman correlation between the variable and the lag.
    max_reassigned_fraction : float
        Largest share of probands that may move bin under an edge perturbation.
    edge_perturbation : float
        Size of the edge perturbation, in the variable's units.
    smd_flag : float
        Standardised difference across the extreme bins above which a covariate is flagged.
    """

    min_bin_size: int = 1000
    smallest_class_fraction: float = 0.15
    min_projected_smallest_class: int = 150
    min_coverage: float = 0.90
    min_n_bins: int = 2
    max_bin_share: float = 0.65
    small_lag_years: float = 2.0
    max_lag_correlation: float = 0.30
    max_reassigned_fraction: float = 0.05
    edge_perturbation: float = 1.0
    smd_flag: float = 0.20


DEFAULT_THRESHOLDS = RequirementThresholds()


@dataclass(frozen=True)
class RequirementResult:
    """The outcome of one requirement.

    Attributes
    ----------
    key : str
        Short identifier.
    tier : int
        1, 2, or 3.
    description : str
        What the requirement checks.
    status : str
        ``"pass"`` or ``"fail"`` for Tier 1, ``"ok"`` or ``"flag"`` for Tier 2, ``"report"``
        for Tier 3, and ``"skipped"`` when an input was not supplied.
    observed : float or None
        The measured quantity.
    threshold : float or None
        The criterion it was compared against.
    detail : str
        A human-readable summary.
    """

    key: str
    tier: int
    description: str
    status: str
    observed: float | None
    threshold: float | None
    detail: str


@dataclass
class PolicyReport:
    """The full evaluation of one policy against the requirement set.

    Attributes
    ----------
    spec : dict
        The policy specification (from :meth:`~analysis.strata.BinningPolicy.spec`).
    n_total : int
        Rows in the modelling cohort.
    n_assigned : int
        Rows assigned to a bin (the rest are missing on the variable).
    counts : dict of str to int
        Assigned rows per bin, in label order.
    results : list of RequirementResult
        One entry per requirement, across all three tiers.
    demographics : pandas.DataFrame or None
        The per-bin covariate summary with the extreme-bin standardised differences, or
        ``None`` when no covariates were supplied.
    """

    spec: dict[str, object]
    n_total: int
    n_assigned: int
    counts: dict[str, int]
    results: list[RequirementResult]
    demographics: pd.DataFrame | None = field(default=None)

    @property
    def eligible(self) -> bool:
        """Whether every Tier 1 gate passed."""
        return not any(r.tier == 1 and r.status == "fail" for r in self.results)

    @property
    def flags(self) -> list[str]:
        """Keys of the Tier 2 checks that were flagged."""
        return [r.key for r in self.results if r.tier == 2 and r.status == "flag"]

    def to_frame(self) -> pd.DataFrame:
        """Return the requirement results as a table, one row per requirement."""
        return pd.DataFrame(
            [
                {
                    "key": r.key,
                    "tier": r.tier,
                    "status": r.status,
                    "observed": r.observed,
                    "threshold": r.threshold,
                    "detail": r.detail,
                }
                for r in self.results
            ]
        )


def _gate_min_bin_size(
    assignment: StratumAssignment, t: RequirementThresholds
) -> RequirementResult:
    smallest = min(assignment.counts.values())
    offenders = [label for label, n in assignment.counts.items() if n < t.min_bin_size]
    status = "fail" if offenders else "pass"
    detail = f"smallest bin {smallest}" + (f"; below floor: {offenders}" if offenders else "")
    return RequirementResult(
        "min_bin_size",
        1,
        "Every bin clears the N_min floor.",
        status,
        float(smallest),
        float(t.min_bin_size),
        detail,
    )


def _gate_smallest_class(
    assignment: StratumAssignment, t: RequirementThresholds
) -> RequirementResult:
    projected = {label: n * t.smallest_class_fraction for label, n in assignment.counts.items()}
    worst = min(projected.values())
    offenders = [label for label, p in projected.items() if p < t.min_projected_smallest_class]
    status = "fail" if offenders else "pass"
    detail = (
        f"smallest projected class {worst:.0f} (at {t.smallest_class_fraction:.0%} of a bin)"
        + (f"; below floor: {offenders}" if offenders else "")
    )
    return RequirementResult(
        "smallest_class",
        1,
        "Each bin can populate the smallest of four classes.",
        status,
        float(worst),
        float(t.min_projected_smallest_class),
        detail,
    )


def _gate_coverage(
    assignment: StratumAssignment, n_total: int, t: RequirementThresholds
) -> RequirementResult:
    n_assigned = sum(assignment.counts.values())
    coverage = n_assigned / n_total if n_total else 0.0
    status = "pass" if coverage >= t.min_coverage else "fail"
    detail = f"{n_assigned}/{n_total} assigned ({coverage:.1%}); {assignment.n_missing} missing"
    return RequirementResult(
        "coverage",
        1,
        "Most of the cohort is assigned (low missingness).",
        status,
        float(coverage),
        float(t.min_coverage),
        detail,
    )


def _gate_partition(
    assignment: StratumAssignment, n_total: int, t: RequirementThresholds
) -> RequirementResult:
    non_empty = sum(1 for n in assignment.counts.values() if n > 0)
    accounted = sum(assignment.counts.values()) + assignment.n_missing == n_total
    status = "pass" if non_empty >= t.min_n_bins and accounted else "fail"
    detail = f"{non_empty} non-empty bins; rows accounted for: {accounted}"
    return RequirementResult(
        "partition_validity",
        1,
        "At least two non-empty, exhaustive bins.",
        status,
        float(non_empty),
        float(t.min_n_bins),
        detail,
    )


def _check_size_balance(
    assignment: StratumAssignment, t: RequirementThresholds
) -> RequirementResult:
    n_assigned = sum(assignment.counts.values())
    shares = (
        np.array([n / n_assigned for n in assignment.counts.values()])
        if n_assigned
        else np.array([])
    )
    largest = float(shares.max()) if shares.size else 0.0
    status = "flag" if largest > t.max_bin_share else "ok"
    detail = f"largest bin holds {largest:.1%} of the assigned cohort"
    return RequirementResult(
        "size_balance",
        2,
        "No bin dominates the partition.",
        status,
        largest,
        t.max_bin_share,
        detail,
    )


def _check_lag(
    assignment: StratumAssignment, variable: pd.Series, lag: pd.Series, t: RequirementThresholds
) -> list[RequirementResult]:
    correlation = float(variable.corr(lag, method="spearman"))
    corr_status = "flag" if abs(correlation) > t.max_lag_correlation else "ok"
    corr_result = RequirementResult(
        "lag_correlation",
        2,
        "The variable is not strongly entangled with the lag.",
        corr_status,
        correlation,
        t.max_lag_correlation,
        f"Spearman r = {correlation:.2f} between variable and lag",
    )
    small = lag <= t.small_lag_years
    retained = sum(
        1
        for label in assignment.labels
        if int((assignment.codes[small] == label).sum()) >= t.min_bin_size
    )
    retain_status = "ok" if retained >= t.min_n_bins else "flag"
    retain_result = RequirementResult(
        "small_lag_retention",
        2,
        "A small-lag subsample still supports the strata.",
        retain_status,
        float(retained),
        float(t.min_n_bins),
        f"{retained} bins keep >= {t.min_bin_size} probands within "
        f"{t.small_lag_years:g} years of diagnosis",
    )
    return [corr_result, retain_result]


def _extreme_smd(values: pd.Series, codes: pd.Series, low: str, high: str) -> float:
    """Standardised difference of a covariate between the lowest and highest bins."""
    group_low = values[codes == low].dropna()
    group_high = values[codes == high].dropna()
    if group_low.empty or group_high.empty:
        return float("nan")
    return abs(cohens_d(group_high, group_low))


def _check_covariate_balance(
    assignment: StratumAssignment, covariates: pd.DataFrame, t: RequirementThresholds
) -> RequirementResult:
    low, high = assignment.labels[0], assignment.labels[-1]
    smds = {col: _extreme_smd(covariates[col], assignment.codes, low, high) for col in covariates}
    offenders = [col for col, smd in smds.items() if np.isfinite(smd) and smd > t.smd_flag]
    worst = max((smd for smd in smds.values() if np.isfinite(smd)), default=0.0)
    status = "flag" if offenders else "ok"
    detail = f"largest extreme-bin SMD {worst:.2f}" + (
        f"; imbalanced: {offenders}" if offenders else ""
    )
    return RequirementResult(
        "covariate_balance",
        2,
        "Covariates are not strongly imbalanced across bins.",
        status,
        float(worst),
        t.smd_flag,
        detail,
    )


def _check_edge_robustness(
    assignment: StratumAssignment, variable: pd.Series, t: RequirementThresholds
) -> RequirementResult:
    edges = assignment.edges
    labels = tuple(assignment.labels)
    valid = variable.notna()
    if not edges or int(valid.sum()) == 0:
        return RequirementResult(
            "edge_robustness",
            2,
            "Edges are stable to a small perturbation.",
            "skipped",
            None,
            t.max_reassigned_fraction,
            "no interior edges to perturb",
        )
    base = assignment.codes[valid].astype(object)
    worst_fraction = 0.0
    worst_min_size = min(assignment.counts.values())
    for i in range(len(edges)):
        for step in (-t.edge_perturbation, t.edge_perturbation):
            perturbed = list(edges)
            perturbed[i] = edges[i] + step
            try:
                shifted = FixedBands(edges=tuple(perturbed), labels=labels).assign(variable)
            except ValueError:
                continue
            moved = float((base != shifted.codes[valid].astype(object)).mean())
            worst_fraction = max(worst_fraction, moved)
            worst_min_size = min(worst_min_size, min(shifted.counts.values()))
    stable = worst_fraction <= t.max_reassigned_fraction and worst_min_size >= t.min_bin_size
    status = "ok" if stable else "flag"
    detail = (
        f"up to {worst_fraction:.1%} reassign under +/-{t.edge_perturbation:g}; "
        f"smallest perturbed bin {worst_min_size}"
    )
    return RequirementResult(
        "edge_robustness",
        2,
        "Edges are stable to a small perturbation.",
        status,
        worst_fraction,
        t.max_reassigned_fraction,
        detail,
    )


def _demographic_table(assignment: StratumAssignment, covariates: pd.DataFrame) -> pd.DataFrame:
    low, high = assignment.labels[0], assignment.labels[-1]
    rows: dict[str, dict[str, float]] = {}
    for col in covariates:
        means = {
            label: float(covariates[col][assignment.codes == label].mean())
            for label in assignment.labels
        }
        means["smd_extreme"] = _extreme_smd(covariates[col], assignment.codes, low, high)
        rows[col] = means
    return pd.DataFrame.from_dict(rows, orient="index")


def evaluate_policy(
    policy: BinningPolicy,
    variable: pd.Series,
    *,
    lag: pd.Series | None = None,
    covariates: pd.DataFrame | None = None,
    thresholds: RequirementThresholds = DEFAULT_THRESHOLDS,
) -> PolicyReport:
    """Evaluate a binning policy against the tiered requirement set.

    Parameters
    ----------
    policy : BinningPolicy
        The concrete policy to test.
    variable : pandas.Series
        The stratifying variable over the modelling cohort (age at diagnosis or era).
    lag : pandas.Series, optional
        The measurement-to-diagnosis lag in years, on the same index. Enables the Tier 2 lag
        checks (the era-axis defence).
    covariates : pandas.DataFrame, optional
        Numeric (one-hot encoded where categorical) covariates on the same index. Enables the
        Tier 2 covariate-balance check and the Tier 3 demographic table.
    thresholds : RequirementThresholds, optional
        The criteria to judge against. Defaults to the phase-2 values.

    Returns
    -------
    PolicyReport
        The per-requirement results, the eligibility verdict, and the demographic table.
    """
    assignment = policy.assign(variable)
    n_total = len(variable)
    results = [
        _gate_min_bin_size(assignment, thresholds),
        _gate_smallest_class(assignment, thresholds),
        _gate_coverage(assignment, n_total, thresholds),
        _gate_partition(assignment, n_total, thresholds),
        _check_size_balance(assignment, thresholds),
    ]
    if lag is not None:
        results += _check_lag(assignment, variable, lag, thresholds)
    if covariates is not None:
        results.append(_check_covariate_balance(assignment, covariates, thresholds))
    results.append(_check_edge_robustness(assignment, variable, thresholds))
    demographics = _demographic_table(assignment, covariates) if covariates is not None else None
    return PolicyReport(
        spec=policy.spec(),
        n_total=n_total,
        n_assigned=sum(assignment.counts.values()),
        counts=assignment.counts,
        results=results,
        demographics=demographics,
    )
