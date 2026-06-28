r"""Class drift between a stratum fit and the pooled reference, and its permutation null.

The stratified analysis (plan section 7, frozen in section 12a) asks whether the four
reference classes move when the mixture model is re-estimated within a stratum of age at
diagnosis or diagnostic era. This module measures that movement and calibrates it, with the
expensive part (the fits) separated from the cheap, method-dependent part (alignment and
distance), so a different alignment or distance can be tried without re-fitting.

The unit that is fitted and stored is a :class:`StratumSummary`: the class-by-feature
centroids plus the contingency table of the fit's labels against the pooled reference labels
on the same probands. Those two are the method-independent sufficient statistics. From them:

- an :class:`AlignmentMethod` maps the fit's arbitrary class ids to the reference classes.
  :class:`MembershipJaccard` (the default) aligns on who is in each class, since a stratum is
  a subset of the pooled cohort and so carries both labellings on the same probands; this is
  Litman's overlap currency and distinguishes a class that *moved* (same members, shifted
  centroid) from one that *reorganised* (different members), which a centroid-only alignment
  cannot. :class:`CentroidHungarian` aligns on centroid distance instead.
- a :class:`DistanceMethod` measures how far each aligned class's centroid moved, in pooled
  standard-deviation units. :class:`StandardisedEuclidean` (the default) is the standardised
  Euclidean distance; :class:`MeanAbsolute` is the more outlier-robust mean absolute shift.

The drift is read two ways. Against the *between-class separation* (the same distance between
distinct reference classes), so a shift is on the scale of the partition: a value near or
above 1 means the class moved as far as the gap to a neighbouring class. And against a
*permutation null*: pseudo-strata of the same sizes formed by randomly partitioning the
cohort remove the association with the axis while keeping sample size fixed, so the observed
shift is read against same-size random partitions (beyond the 95th percentile, then FDR
controlled across the strata-by-class tests). The alignment also reports its confidence (the
per-class Jaccard and the overall adjusted Rand index), so a large shift with low overlap is
flagged as reorganisation rather than reported as drift.

:func:`summarise_pseudo_stratum` is a top-level, picklable unit of work (fit one subset,
return its summary), so the null can be spread across a process pool: the StepMix fit is
single-core, so throughput comes from running many fits at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from analysis.cohort import CohortMatrix
from analysis.features import Typing
from analysis.model import class_centroids, fit_gfmm


@dataclass
class ReferenceModel:
    """The pooled reference solution, the fixed target every stratum is compared against.

    Attributes
    ----------
    centroids : pandas.DataFrame
        Reference class-by-feature centroids.
    pooled_sd : pandas.Series
        Per-feature standard deviation across the cohort, the distance normaliser.
    labels : pandas.Series
        The reference (pooled) class per proband, used to build a stratum's contingency.
    """

    centroids: pd.DataFrame
    pooled_sd: pd.Series
    labels: pd.Series


@dataclass
class StratumSummary:
    """The method-independent summary of one fit: centroids and reference contingency.

    These are the sufficient statistics for any alignment or distance, so they are stored
    once and re-measured cheaply when the method changes.

    Attributes
    ----------
    centroids : pandas.DataFrame
        Fit class-by-feature centroids.
    contingency : pandas.DataFrame
        Counts of the fit's classes (rows) against the reference classes (columns) over the
        fit's probands.
    n : int
        Number of probands.
    """

    centroids: pd.DataFrame
    contingency: pd.DataFrame
    n: int


@dataclass
class ClassAlignment:
    """A mapping from fit classes to reference classes, with its confidence.

    Attributes
    ----------
    mapping : dict of int to int
        Fit class id to reference class id.
    quality : dict of int to float
        Per reference class, the match confidence (Jaccard for membership, a normalised
        closeness for centroid alignment); higher is more confident.
    overall : float
        The adjusted Rand index between the two labellings (membership), or the mean per-pair
        quality (centroid). A low value means the partition reorganised rather than shifted.
    """

    mapping: dict[int, int]
    quality: dict[int, float]
    overall: float


def common_columns(source: pd.DataFrame, reference: pd.DataFrame) -> list[str]:
    """Return the feature columns shared by two centroid matrices, in reference order."""
    source_set = set(source.columns)
    return [c for c in reference.columns if c in source_set]


def contingency_table(fit_labels: pd.Series, reference_labels: pd.Series) -> pd.DataFrame:
    """Cross-tabulate a fit's labels against the reference labels over the shared probands."""
    idx = fit_labels.index.intersection(reference_labels.index)
    table = pd.crosstab(fit_labels.loc[idx], reference_labels.loc[idx])
    table.index = table.index.astype(int)
    table.columns = table.columns.astype(int)
    return table


def _comb2(counts: np.ndarray) -> float:
    """Sum of ``n choose 2`` over an array of counts."""
    counts = counts.astype(float)
    return float(np.sum(counts * (counts - 1.0) / 2.0))


def adjusted_rand_index(table: np.ndarray) -> float:
    """Return the adjusted Rand index between two labellings, from their contingency table.

    Chance-corrected agreement (0 is chance, 1 is identical partitions), computed from the
    counts directly so it needs only the stored contingency, not the per-proband labels.
    """
    n = float(table.sum())
    if n < 2:
        return float("nan")
    index = _comb2(table.ravel())
    a = _comb2(table.sum(axis=1))
    b = _comb2(table.sum(axis=0))
    total = n * (n - 1.0) / 2.0
    expected = a * b / total
    maximum = 0.5 * (a + b)
    if maximum == expected:
        return 0.0
    return float((index - expected) / (maximum - expected))


@runtime_checkable
class AlignmentMethod(Protocol):
    """Map a stratum fit's classes to the reference classes."""

    name: str

    def align(self, stratum: StratumSummary, reference: ReferenceModel) -> ClassAlignment:
        """Return the fit-to-reference class mapping and its confidence."""
        ...


@dataclass
class MembershipJaccard:
    """Align on shared membership: pair classes by maximal Jaccard overlap of their probands.

    The most direct alignment, since a stratum is a subset of the pooled cohort, so each
    proband carries both labellings. The Jaccard (intersection over union) normalises for the
    very unequal class sizes, so the largest class does not dominate the match. The overall
    confidence is the adjusted Rand index of the two labellings.
    """

    name: str = "membership"

    def align(self, stratum: StratumSummary, reference: ReferenceModel) -> ClassAlignment:
        """Align by Hungarian assignment on one minus the Jaccard overlap."""
        table = stratum.contingency.reindex(columns=[int(c) for c in reference.centroids.index])
        table = table.fillna(0.0)
        counts = table.to_numpy(dtype=float)
        row = counts.sum(axis=1, keepdims=True)
        col = counts.sum(axis=0, keepdims=True)
        union = row + col - counts
        jaccard = np.divide(counts, union, out=np.zeros_like(counts), where=union > 0)
        rows, cols = linear_sum_assignment(1.0 - jaccard)
        fit_ids = [int(i) for i in table.index]
        ref_ids = [int(c) for c in table.columns]
        mapping = {fit_ids[i]: ref_ids[j] for i, j in zip(rows, cols, strict=True)}
        quality = {ref_ids[j]: float(jaccard[i, j]) for i, j in zip(rows, cols, strict=True)}
        return ClassAlignment(mapping, quality, adjusted_rand_index(counts))


@dataclass
class CentroidHungarian:
    """Align on centroid distance: pair classes by the closest standardised centroids.

    A fallback that uses only the centroids, so it cannot tell a class that moved from one
    that reorganised. Kept to cross-check the membership alignment: a disagreement between the
    two flags an unsafe mapping.
    """

    name: str = "centroid"

    def align(self, stratum: StratumSummary, reference: ReferenceModel) -> ClassAlignment:
        """Align by Hungarian assignment on the standardised centroid distance."""
        cols = [
            c
            for c in common_columns(stratum.centroids, reference.centroids)
            if reference.pooled_sd.get(c, 0.0) > 0
        ]
        sd = reference.pooled_sd[cols].to_numpy()
        src = stratum.centroids[cols].to_numpy() / sd
        ref = reference.centroids[cols].to_numpy() / sd
        cost = np.linalg.norm(src[:, None, :] - ref[None, :, :], axis=2)
        rows, cols_ind = linear_sum_assignment(cost)
        fit_ids = [int(i) for i in stratum.centroids.index]
        ref_ids = [int(i) for i in reference.centroids.index]
        mapping = {fit_ids[i]: ref_ids[j] for i, j in zip(rows, cols_ind, strict=True)}
        # Closeness in [0, 1]: 1 when coincident, decaying with distance, for a comparable
        # confidence reading. The mapping itself is unaffected by this rescaling.
        quality = {
            ref_ids[j]: float(1.0 / (1.0 + cost[i, j])) for i, j in zip(rows, cols_ind, strict=True)
        }
        return ClassAlignment(mapping, quality, float(np.mean(list(quality.values()))))


@runtime_checkable
class DistanceMethod(Protocol):
    """Measure the distance a class centroid moved, in pooled standard-deviation units."""

    name: str

    def pairwise(self, a: np.ndarray, b: np.ndarray, sd: np.ndarray) -> float:
        """Distance between two centroid vectors, each feature normalised by its spread."""
        ...


@dataclass
class StandardisedEuclidean:
    """Standardised Euclidean distance: the root-mean-square per-feature shift in SD units.

    The geometric distance the centroid moved in standardised feature space, averaged over
    features so it does not grow with their number. The default, as the natural distance and
    the closest cheap proxy for a covariance-aware Mahalanobis distance.
    """

    name: str = "euclidean"

    def pairwise(self, a: np.ndarray, b: np.ndarray, sd: np.ndarray) -> float:
        """Root mean square of the standardised per-feature difference."""
        d = (a - b) / sd
        return float(np.sqrt(np.mean(d**2)))


@dataclass
class MeanAbsolute:
    """Mean absolute per-feature shift in SD units, an outlier-robust alternative."""

    name: str = "mean-abs"

    def pairwise(self, a: np.ndarray, b: np.ndarray, sd: np.ndarray) -> float:
        """Mean absolute standardised per-feature difference."""
        return float(np.mean(np.abs((a - b) / sd)))


ALIGNMENTS: dict[str, AlignmentMethod] = {
    "membership": MembershipJaccard(),
    "centroid": CentroidHungarian(),
}
DISTANCES: dict[str, DistanceMethod] = {
    "euclidean": StandardisedEuclidean(),
    "mean-abs": MeanAbsolute(),
}
DEFAULT_ALIGNMENT = "membership"
DEFAULT_DISTANCE = "euclidean"


def class_distances(
    stratum: StratumSummary,
    reference: ReferenceModel,
    mapping: dict[int, int],
    distance: DistanceMethod,
) -> dict[int, float]:
    """Per reference class, the distance its aligned stratum centroid sits from it."""
    cols = [
        c
        for c in common_columns(stratum.centroids, reference.centroids)
        if reference.pooled_sd.get(c, 0.0) > 0
    ]
    sd = reference.pooled_sd[cols].to_numpy()
    out: dict[int, float] = {}
    for fit_class, ref_class in mapping.items():
        a = stratum.centroids.loc[fit_class, cols].to_numpy()
        b = reference.centroids.loc[ref_class, cols].to_numpy()
        out[ref_class] = distance.pairwise(a, b, sd)
    return out


def class_separation(reference: ReferenceModel, distance: DistanceMethod) -> float:
    """Mean distance between distinct reference classes, the drift baseline.

    The same distance the drift uses, measured between the reference classes themselves and
    averaged over pairs, so drift can be read as a fraction of the gap between distinct
    classes.
    """
    cols = [c for c in reference.centroids.columns if reference.pooled_sd.get(c, 0.0) > 0]
    sd = reference.pooled_sd[cols].to_numpy()
    classes = list(reference.centroids.index)
    distances = [
        distance.pairwise(
            reference.centroids.loc[a, cols].to_numpy(),
            reference.centroids.loc[b, cols].to_numpy(),
            sd,
        )
        for i, a in enumerate(classes)
        for b in classes[i + 1 :]
    ]
    return float(np.mean(distances)) if distances else float("nan")


@dataclass
class DriftResult:
    """One stratum's drift: per-class distance plus the alignment that produced it."""

    distances: dict[int, float]
    alignment: ClassAlignment


def compute_drift(
    stratum: StratumSummary,
    reference: ReferenceModel,
    alignment: AlignmentMethod,
    distance: DistanceMethod,
) -> DriftResult:
    """Align a stratum to the reference and measure each aligned class's drift.

    Pure and cheap (no fitting): this is the method-dependent step run over stored summaries,
    so a different ``alignment`` or ``distance`` re-measures without re-fitting.
    """
    aligned = alignment.align(stratum, reference)
    distances = class_distances(stratum, reference, aligned.mapping, distance)
    return DriftResult(distances=distances, alignment=aligned)


def null_partition(index: pd.Index, sizes: list[int], seed: int) -> list[pd.Index]:
    """Partition ``index`` into consecutive random chunks of the given sizes.

    Shuffles the proband index with a seeded generator, then splits it into blocks of
    ``sizes``, so the pseudo-strata have the same sizes as the real strata but no relation to
    the stratifying axis. The seed is the permutation index, so a resumed null reproduces the
    same partitions.
    """
    rng = np.random.default_rng(seed)
    shuffled = index.to_numpy().copy()
    rng.shuffle(shuffled)
    chunks: list[pd.Index] = []
    start = 0
    for size in sizes:
        chunks.append(pd.Index(shuffled[start : start + size]))
        start += size
    return chunks


def summarise_pseudo_stratum(
    features: pd.DataFrame,
    covariates: pd.DataFrame,
    typing: Typing,
    reference_labels: pd.Series,
    n_init: int,
    seed: int,
) -> StratumSummary:
    """Fit the GFMM on one subset and return its method-independent summary.

    A top-level function so it pickles for a process pool. Stores the centroids and the
    contingency against the reference labels, not a drift value, so the alignment and distance
    can be chosen (and changed) afterwards without re-fitting.
    """
    matrix = CohortMatrix(features, covariates, "spark", "pseudo")
    fit = fit_gfmm(matrix, typing, n_init=n_init, random_state=seed, progress_bar=0, verbose=0)
    centroids = class_centroids(fit.measurement_data, fit.labels)
    contingency = contingency_table(fit.labels, reference_labels)
    return StratumSummary(centroids=centroids, contingency=contingency, n=int(len(fit.labels)))


def serialise_summary(summary: StratumSummary, perm: int, s_idx: int) -> dict:
    """Serialise a stratum summary to a JSON-able record for the null store.

    The null fits are stored as their summaries, not their drift, so the alignment and
    distance can be chosen afterwards. One record per pseudo-stratum, keyed by its permutation
    and stratum index.
    """
    return {
        "perm": int(perm),
        "s_idx": int(s_idx),
        "n": int(summary.n),
        "classes": [int(i) for i in summary.centroids.index],
        "features": [str(c) for c in summary.centroids.columns],
        "centroids": summary.centroids.to_numpy().tolist(),
        "cont_rows": [int(i) for i in summary.contingency.index],
        "cont_cols": [int(c) for c in summary.contingency.columns],
        "contingency": summary.contingency.to_numpy().astype(int).tolist(),
    }


def deserialise_summary(record: dict) -> StratumSummary:
    """Rebuild a :class:`StratumSummary` from a serialised null-store record."""
    centroids = pd.DataFrame(
        record["centroids"],
        index=pd.Index([int(i) for i in record["classes"]], name="class"),
        columns=record["features"],
    )
    contingency = pd.DataFrame(
        record["contingency"],
        index=[int(i) for i in record["cont_rows"]],
        columns=[int(c) for c in record["cont_cols"]],
    )
    return StratumSummary(centroids=centroids, contingency=contingency, n=int(record["n"]))


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Return a boolean mask of the hypotheses that pass Benjamini-Hochberg FDR control.

    Controls the false-discovery rate at ``q`` across the strata-by-class drift tests (plan
    section 12a). A hypothesis is rejected if its p-value is at or below the largest threshold
    ``q * rank / m`` it satisfies, where ``m`` is the number of finite p-values; NaN p-values
    (a degenerate stratum) never pass.
    """
    p = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p)
    reject = np.zeros(p.shape, dtype=bool)
    idx = np.where(finite)[0]
    if idx.size == 0:
        return reject
    order = idx[np.argsort(p[idx])]
    m = idx.size
    thresholds = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresholds
    if passed.any():
        cutoff = int(np.max(np.where(passed)[0]))
        reject[order[: cutoff + 1]] = True
    return reject


def read_against_null(observed: float, null_draws: list[float]) -> dict[str, float]:
    """Read an observed drift against its size-matched null distribution.

    Returns the null 95th percentile, whether the observed shift exceeds it, and the
    permutation p-value with the Phipson-Smyth add-one correction (so the smallest p is
    ``1 / (n + 1)`` rather than zero). The decision threshold and the FDR step across classes
    are applied by the caller over these per-class reads.
    """
    draws = np.asarray([d for d in null_draws if np.isfinite(d)], dtype=float)
    n = int(draws.size)
    p95 = float(np.percentile(draws, 95)) if n else float("nan")
    exceedances = int(np.sum(draws >= observed)) if n else 0
    p_value = (1 + exceedances) / (1 + n) if n else float("nan")
    return {
        "observed": float(observed),
        "null_p95": p95,
        "exceeds_p95": float(observed > p95) if n else float("nan"),
        "p_value": p_value,
        "n_null": float(n),
    }
