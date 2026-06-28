r"""Class drift between a stratum fit and the pooled reference, and its permutation null.

The stratified analysis (plan section 7, frozen in section 12a) asks whether the four
reference classes move when the mixture model is re-estimated within a stratum of age at
diagnosis or diagnostic era. This module measures that movement and calibrates it, with the
expensive part (the fits) separated from the cheap, method-dependent part (alignment and
distance), so a different alignment or distance can be tried without re-fitting.

The unit that is fitted and stored is a :class:`StratumSummary`: per-class feature means
(centroids) and dispersions (standard deviations), plus the contingency of the fit's labels
against the pooled reference labels on the same probands. These are the method-independent
sufficient statistics. From them:

- an :class:`AlignmentMethod` maps the fit's arbitrary class ids to the reference classes.
  :class:`MembershipJaccard` (the default) aligns on who is in each class, since a stratum is
  a subset of the pooled cohort and so carries both labellings on the same probands; this
  distinguishes a class that *moved* (same members, shifted centroid) from one that
  *reorganised* (different members), which a centroid-only alignment cannot.
  :class:`CentroidHungarian` aligns on centroid distance instead.
- a :class:`DistanceMethod` measures how far each aligned class moved. :class:`Mahalanobis`
  (the default) is the covariance-aware distance between centroids, so correlated features
  count once rather than many times; :class:`StandardisedEuclidean` and :class:`MeanAbsolute`
  are the diagonal (covariance-blind) distances between centroids; :class:`JensenShannon`
  compares the class-conditional distributions, treating each feature as Gaussian with the
  per-class mean and dispersion, so it sees a change in spread that the centroid distances
  miss.

The drift is read against the *between-class separation* (the same distance between distinct
reference classes) so a shift is on the scale of the partition, and against a *permutation
null*: pseudo-strata of the same sizes from random partitions of the cohort, so the observed
shift is read against same-size random partitions (beyond the 95th percentile, then FDR
controlled). The alignment also reports its confidence (per-class Jaccard, overall adjusted
Rand index), so a large shift with low overlap is flagged as reorganisation, not drift.

:func:`summarise_pseudo_stratum` is a top-level, picklable unit of work (fit one subset,
return its summary), so the null can be spread across a process pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.covariance import LedoitWolf

from analysis.cohort import CohortMatrix
from analysis.features import Typing
from analysis.model import fit_gfmm

_VARIANCE_FLOOR = 1e-6


@dataclass
class ReferenceModel:
    """The pooled reference solution, the fixed target every stratum is compared against.

    Attributes
    ----------
    centroids : pandas.DataFrame
        Reference class-by-feature centroids (means).
    dispersions : pandas.DataFrame
        Reference class-by-feature standard deviations, for the distributional distance.
    pooled_sd : pandas.Series
        Per-feature standard deviation across the cohort, the diagonal-distance normaliser.
    precision : numpy.ndarray
        Inverse of the Ledoit-Wolf-shrunk pooled within-class covariance, in the column order
        of ``centroids``. Shrinkage keeps it well-conditioned at 238 features.
    labels : pandas.Series
        The reference (pooled) class per proband, used to build a stratum's contingency.
    """

    centroids: pd.DataFrame
    dispersions: pd.DataFrame
    pooled_sd: pd.Series
    precision: np.ndarray
    labels: pd.Series

    def as_stratum(self) -> StratumSummary:
        """Return the reference as a stratum summary, for the between-class separation."""
        n_classes = len(self.centroids)
        identity = pd.DataFrame(
            np.eye(n_classes, dtype=int), index=self.centroids.index, columns=self.centroids.index
        )
        return StratumSummary(
            centroids=self.centroids,
            dispersions=self.dispersions,
            contingency=identity,
            n=int(self.labels.shape[0]),
        )


def build_reference(measurement_data: pd.DataFrame, labels: pd.Series) -> ReferenceModel:
    """Build the reference model from the pooled fit's measurement data and labels.

    Computes the per-class centroids and dispersions, the per-feature pooled spread, and the
    Ledoit-Wolf-shrunk precision matrix of the pooled within-class covariance (the residuals
    of each proband from its class mean). Shrinkage is what makes the 238-feature covariance
    invertible and stable.
    """
    aligned = labels.reindex(measurement_data.index)
    grouped = measurement_data.groupby(aligned.to_numpy())
    centroids = grouped.mean()
    dispersions = grouped.std().fillna(0.0)
    centroids.index = pd.Index(np.asarray(centroids.index, dtype=int), name="class")
    dispersions.index = centroids.index
    residuals = measurement_data.to_numpy() - centroids.loc[aligned.to_numpy()].to_numpy()
    precision = LedoitWolf().fit(residuals).precision_
    return ReferenceModel(
        centroids=centroids,
        dispersions=dispersions,
        pooled_sd=measurement_data.std(),
        precision=np.asarray(precision, dtype=float),
        labels=labels,
    )


@dataclass
class StratumSummary:
    """The method-independent summary of one fit: centroids, dispersions, and contingency.

    These are the sufficient statistics for any alignment or distance, so they are stored once
    and re-measured cheaply when the method changes.

    Attributes
    ----------
    centroids : pandas.DataFrame
        Fit class-by-feature means.
    dispersions : pandas.DataFrame
        Fit class-by-feature standard deviations.
    contingency : pandas.DataFrame
        Counts of the fit's classes (rows) against the reference classes (columns).
    n : int
        Number of probands.
    """

    centroids: pd.DataFrame
    dispersions: pd.DataFrame
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
    """Return the feature columns shared by two matrices, in reference order."""
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
    proband carries both labellings. The Jaccard normalises for the very unequal class sizes,
    so the largest class does not dominate the match. The overall confidence is the adjusted
    Rand index of the two labellings.
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
        quality = {
            ref_ids[j]: float(1.0 / (1.0 + cost[i, j])) for i, j in zip(rows, cols_ind, strict=True)
        }
        return ClassAlignment(mapping, quality, float(np.mean(list(quality.values()))))


@runtime_checkable
class DistanceMethod(Protocol):
    """Measure the distance one aligned class moved between a stratum and the reference."""

    name: str

    def class_distance(
        self, stratum: StratumSummary, fit_class: int, reference: ReferenceModel, ref_class: int
    ) -> float:
        """Distance between a stratum class and its aligned reference class."""
        ...


def _standardised_delta(
    stratum: StratumSummary, fit_class: int, reference: ReferenceModel, ref_class: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-feature centroid difference and pooled SD over the shared features."""
    cols = [
        c
        for c in common_columns(stratum.centroids, reference.centroids)
        if reference.pooled_sd.get(c, 0.0) > 0
    ]
    sd = reference.pooled_sd[cols].to_numpy()
    delta = (
        stratum.centroids.loc[fit_class, cols].to_numpy()
        - reference.centroids.loc[ref_class, cols].to_numpy()
    )
    return delta, sd


@dataclass
class StandardisedEuclidean:
    """Standardised Euclidean distance: the root-mean-square per-feature shift in SD units.

    A diagonal (covariance-blind) distance: it treats the features as independent.
    """

    name: str = "euclidean"

    def class_distance(
        self, stratum: StratumSummary, fit_class: int, reference: ReferenceModel, ref_class: int
    ) -> float:
        """Root mean square of the standardised per-feature difference."""
        delta, sd = _standardised_delta(stratum, fit_class, reference, ref_class)
        return float(np.sqrt(np.mean((delta / sd) ** 2)))


@dataclass
class MeanAbsolute:
    """Mean absolute per-feature shift in SD units, an outlier-robust diagonal distance."""

    name: str = "mean-abs"

    def class_distance(
        self, stratum: StratumSummary, fit_class: int, reference: ReferenceModel, ref_class: int
    ) -> float:
        """Mean absolute standardised per-feature difference."""
        delta, sd = _standardised_delta(stratum, fit_class, reference, ref_class)
        return float(np.mean(np.abs(delta / sd)))


@dataclass
class Mahalanobis:
    """Mahalanobis distance between centroids, using the shrunk within-class precision.

    The covariance-aware distance: correlated features contribute once rather than many times,
    so a coordinated shift across a correlated block of symptoms is not double-counted. The
    default, as the statistically proper multivariate distance. Centroids are reindexed to the
    reference feature order; a feature absent from the stratum contributes no shift.
    """

    name: str = "mahalanobis"

    def class_distance(
        self, stratum: StratumSummary, fit_class: int, reference: ReferenceModel, ref_class: int
    ) -> float:
        """Square root of the precision-weighted squared centroid difference."""
        cols = reference.centroids.columns
        src = stratum.centroids.reindex(columns=cols).loc[fit_class]
        ref = reference.centroids.loc[ref_class]
        delta = src.fillna(ref).to_numpy() - ref.to_numpy()
        return float(np.sqrt(max(0.0, delta @ reference.precision @ delta)))


def _gaussian_jsd(m1: np.ndarray, s1: np.ndarray, m2: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Per-feature Jensen-Shannon divergence between two Gaussians, in [0, 1].

    No closed form exists, so each feature's divergence is integrated on a per-feature grid
    spanning both Gaussians, then normalised by ``ln 2`` to the unit interval.
    """
    s1 = np.maximum(s1, _VARIANCE_FLOOR)
    s2 = np.maximum(s2, _VARIANCE_FLOOR)
    lo = np.minimum(m1 - 6 * s1, m2 - 6 * s2)
    hi = np.maximum(m1 + 6 * s1, m2 + 6 * s2)
    grid = lo[:, None] + (hi - lo)[:, None] * np.linspace(0.0, 1.0, 256)[None, :]
    dx = (hi - lo) / 255.0

    def pdf(m: np.ndarray, s: np.ndarray) -> np.ndarray:
        z = (grid - m[:, None]) / s[:, None]
        return np.exp(-0.5 * z**2) / (s[:, None] * np.sqrt(2.0 * np.pi))

    p = pdf(m1, s1)
    q = pdf(m2, s2)
    mix = 0.5 * (p + q)

    def kl(a: np.ndarray) -> np.ndarray:
        ratio = np.divide(a, mix, out=np.ones_like(a), where=(a > 0) & (mix > 0))
        return np.sum(np.where(a > 0, a * np.log(ratio), 0.0), axis=1) * dx

    return np.clip((0.5 * kl(p) + 0.5 * kl(q)) / np.log(2.0), 0.0, 1.0)


@dataclass
class JensenShannon:
    """Mean per-feature Jensen-Shannon divergence between the class-conditional distributions.

    Each feature's class-conditional is treated as Gaussian with the per-class mean and
    dispersion, so the divergence sees a change in spread, not only in location, which the
    centroid distances cannot. Bounded in [0, 1] per feature and averaged over the shared
    features. The Gaussian treatment is an approximation for the binary and categorical-coded
    features.
    """

    name: str = "jsd"

    def class_distance(
        self, stratum: StratumSummary, fit_class: int, reference: ReferenceModel, ref_class: int
    ) -> float:
        """Mean per-feature Jensen-Shannon divergence over the shared features."""
        cols = common_columns(stratum.centroids, reference.centroids)
        m1 = stratum.centroids.loc[fit_class, cols].to_numpy()
        s1 = stratum.dispersions.reindex(columns=cols).loc[fit_class].to_numpy()
        m2 = reference.centroids.loc[ref_class, cols].to_numpy()
        s2 = reference.dispersions.loc[ref_class, cols].to_numpy()
        return float(np.mean(_gaussian_jsd(m1, s1, m2, s2)))


ALIGNMENTS: dict[str, AlignmentMethod] = {
    "membership": MembershipJaccard(),
    "centroid": CentroidHungarian(),
}
DISTANCES: dict[str, DistanceMethod] = {
    "mahalanobis": Mahalanobis(),
    "euclidean": StandardisedEuclidean(),
    "mean-abs": MeanAbsolute(),
    "jsd": JensenShannon(),
}
DEFAULT_ALIGNMENT = "membership"
DEFAULT_DISTANCE = "mahalanobis"


def class_distances(
    stratum: StratumSummary,
    reference: ReferenceModel,
    mapping: dict[int, int],
    distance: DistanceMethod,
) -> dict[int, float]:
    """Per reference class, the distance its aligned stratum class sits from it."""
    return {
        ref_class: distance.class_distance(stratum, fit_class, reference, ref_class)
        for fit_class, ref_class in mapping.items()
    }


def class_separation(reference: ReferenceModel, distance: DistanceMethod) -> float:
    """Mean distance between distinct reference classes, the drift baseline.

    The same distance the drift uses, measured between the reference classes themselves and
    averaged over pairs, so drift can be read as a fraction of the gap between distinct classes.
    """
    ref_stratum = reference.as_stratum()
    classes = [int(c) for c in reference.centroids.index]
    distances = [
        distance.class_distance(ref_stratum, a, reference, b)
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

    Pure and cheap (no fitting): the method-dependent step run over stored summaries, so a
    different ``alignment`` or ``distance`` re-measures without re-fitting.
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


def summarise(measurement_data: pd.DataFrame, labels: pd.Series, reference_labels: pd.Series):
    """Build a :class:`StratumSummary` from a fit's measurement data and labels.

    Computes per-class means and standard deviations and the contingency against the reference
    labels, the method-independent statistics every distance and alignment is derived from.
    """
    grouped = measurement_data.groupby(labels.to_numpy())
    centroids = grouped.mean()
    dispersions = grouped.std().fillna(0.0)
    centroids.index = pd.Index(np.asarray(centroids.index, dtype=int), name="class")
    dispersions.index = centroids.index
    return StratumSummary(
        centroids=centroids,
        dispersions=dispersions,
        contingency=contingency_table(labels, reference_labels),
        n=int(len(labels)),
    )


def summarise_pseudo_stratum(
    features: pd.DataFrame,
    covariates: pd.DataFrame,
    typing: Typing,
    reference_labels: pd.Series,
    n_init: int,
    seed: int,
) -> StratumSummary:
    """Fit the GFMM on one subset and return its method-independent summary.

    A top-level function so it pickles for a process pool. Stores the centroids, dispersions,
    and reference contingency, not a drift value, so the alignment and distance can be chosen
    (and changed) afterwards without re-fitting.
    """
    matrix = CohortMatrix(features, covariates, "spark", "pseudo")
    fit = fit_gfmm(matrix, typing, n_init=n_init, random_state=seed, progress_bar=0, verbose=0)
    return summarise(fit.measurement_data, fit.labels, reference_labels)


def serialise_summary(summary: StratumSummary, perm: int, s_idx: int) -> dict:
    """Serialise a stratum summary to a JSON-able record for the null store.

    The null fits are stored as their summaries, not their drift, so the alignment and distance
    can be chosen afterwards. One record per pseudo-stratum, keyed by its permutation and
    stratum index.
    """
    return {
        "perm": int(perm),
        "s_idx": int(s_idx),
        "n": int(summary.n),
        "classes": [int(i) for i in summary.centroids.index],
        "features": [str(c) for c in summary.centroids.columns],
        "centroids": summary.centroids.to_numpy().tolist(),
        "dispersions": summary.dispersions.to_numpy().tolist(),
        "cont_rows": [int(i) for i in summary.contingency.index],
        "cont_cols": [int(c) for c in summary.contingency.columns],
        "contingency": summary.contingency.to_numpy().astype(int).tolist(),
    }


def deserialise_summary(record: dict) -> StratumSummary:
    """Rebuild a :class:`StratumSummary` from a serialised null-store record."""
    classes = pd.Index([int(i) for i in record["classes"]], name="class")
    return StratumSummary(
        centroids=pd.DataFrame(record["centroids"], index=classes, columns=record["features"]),
        dispersions=pd.DataFrame(record["dispersions"], index=classes, columns=record["features"]),
        contingency=pd.DataFrame(
            record["contingency"],
            index=[int(i) for i in record["cont_rows"]],
            columns=[int(c) for c in record["cont_cols"]],
        ),
        n=int(record["n"]),
    )


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
