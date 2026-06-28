r"""Class drift between a stratum fit and the pooled reference, and its permutation null.

The stratified analysis (plan section 7, frozen in section 12a) asks whether the four
reference classes move when the mixture model is re-estimated within a stratum of age at
diagnosis or diagnostic era. This module measures that movement and calibrates it.

The primary statistic is the *normalised centroid shift*: a stratum's classes are aligned to
the reference by the Hungarian algorithm on their class-by-feature centroids, and each aligned
class's shift is the mean per-feature change between the stratum and reference centroids, in
pooled standard-deviation units. A larger shift means the class sits in a different place in
feature space within the stratum than in the pooled solution.

A shift is only interpretable against a null, because the class-defining features themselves
correlate with the stratifying axis, so some movement is expected. The null re-fits within
*pseudo-strata* of the same sizes formed by randomly partitioning the cohort, which removes
the association with the axis while keeping sample size fixed. The observed shift is read
against the distribution of shifts from the size-matched pseudo-strata (section 12a: beyond
the 95th percentile, then FDR-controlled across the strata-by-class tests). Drift is also
expressed as a fraction of the mean distance between distinct reference classes, so its size
is read on the scale of the partition itself.

:func:`fit_pseudo_stratum` is a top-level, picklable unit of work (fit one subset, align,
return its drift), so the null can be spread across a process pool: the StepMix fit is
single-core, so throughput comes from running many fits at once, not from one fit using many
cores.

Not yet implemented here (deferred from the primary metric): the Mahalanobis and
Jensen-Shannon corroborating measures, the projection assignment-stability ARI, and the
per-stratum component-support grid. They are computed from the same stored fits when added.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.align import hungarian_align
from analysis.cohort import CohortMatrix
from analysis.features import Typing
from analysis.model import class_centroids, fit_gfmm


def common_columns(source: pd.DataFrame, reference: pd.DataFrame) -> list[str]:
    """Return the feature columns shared by two centroid matrices, in reference order.

    A stratum or a small pseudo-stratum can lose a categorical level, which drops a column
    from its mixed-data descriptor and so from its centroids. Aligning and differencing over
    the shared columns keeps the comparison well defined rather than raising on a mismatch.
    """
    source_set = set(source.columns)
    return [c for c in reference.columns if c in source_set]


def align_centroids(source: pd.DataFrame, reference: pd.DataFrame) -> dict[int, int]:
    """Map each source class to a reference class by Hungarian alignment on centroids.

    Parameters
    ----------
    source : pandas.DataFrame
        The stratum's class-by-feature centroid matrix.
    reference : pandas.DataFrame
        The pooled reference class-by-feature centroid matrix.

    Returns
    -------
    dict of int to int
        Source class id to reference class id. Computed over the shared columns with a
        Euclidean cost, since the centroids are positions in feature space rather than the
        seven-category profiles the correlation cost is meant for.
    """
    cols = common_columns(source, reference)
    mapping: dict = hungarian_align(source[cols], reference[cols], metric="euclidean").mapping
    return {int(k): int(v) for k, v in mapping.items()}


def normalised_centroid_shift(
    source: pd.DataFrame,
    reference: pd.DataFrame,
    pooled_sd: pd.Series,
    mapping: dict[int, int],
) -> dict[int, float]:
    """Per reference class, the mean per-feature centroid shift in pooled-SD units.

    Parameters
    ----------
    source : pandas.DataFrame
        The stratum's class-by-feature centroids.
    reference : pandas.DataFrame
        The reference class-by-feature centroids.
    pooled_sd : pandas.Series
        Per-feature pooled standard deviation, the normaliser. Features with zero spread are
        dropped (they carry no information and would divide by zero).
    mapping : dict of int to int
        Source-to-reference class map from :func:`align_centroids`.

    Returns
    -------
    dict of int to float
        Reference class id to its mean absolute normalised shift.
    """
    cols = [c for c in common_columns(source, reference) if pooled_sd.get(c, 0.0) > 0]
    sd = pooled_sd[cols].to_numpy()
    shifts: dict[int, float] = {}
    for src_class, ref_class in mapping.items():
        delta = source.loc[src_class, cols].to_numpy() - reference.loc[ref_class, cols].to_numpy()
        shifts[ref_class] = float(np.mean(np.abs(delta) / sd))
    return shifts


def class_separation(reference: pd.DataFrame, pooled_sd: pd.Series) -> float:
    """Mean normalised distance between distinct reference classes, the drift baseline.

    Expressing a shift as a fraction of this separation reads it on the scale of the partition:
    a shift much smaller than the gap between two classes is minor, one approaching it is large.
    """
    cols = [c for c in reference.columns if pooled_sd.get(c, 0.0) > 0]
    sd = pooled_sd[cols].to_numpy()
    classes = list(reference.index)
    distances = [
        float(
            np.mean(
                np.abs(reference.loc[a, cols].to_numpy() - reference.loc[b, cols].to_numpy()) / sd
            )
        )
        for i, a in enumerate(classes)
        for b in classes[i + 1 :]
    ]
    return float(np.mean(distances)) if distances else float("nan")


def stratum_drift(
    centroids: pd.DataFrame, reference: pd.DataFrame, pooled_sd: pd.Series
) -> dict[int, float]:
    """Align one stratum's centroids to the reference and return its per-class drift."""
    mapping = align_centroids(centroids, reference)
    return normalised_centroid_shift(centroids, reference, pooled_sd, mapping)


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


def fit_pseudo_stratum(
    features: pd.DataFrame,
    covariates: pd.DataFrame,
    typing: Typing,
    reference: pd.DataFrame,
    pooled_sd: pd.Series,
    n_init: int,
    seed: int,
) -> dict[int, float]:
    """Fit the GFMM on one subset, align to the reference, and return its per-class drift.

    A top-level function so it pickles for a process pool. Used for each null pseudo-stratum;
    the observed strata reuse their stored centroids instead of refitting.
    """
    matrix = CohortMatrix(features, covariates, "spark", "pseudo")
    fit = fit_gfmm(matrix, typing, n_init=n_init, random_state=seed, progress_bar=0, verbose=0)
    centroids = class_centroids(fit.measurement_data, fit.labels)
    return stratum_drift(centroids, reference, pooled_sd)


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Return a boolean mask of the hypotheses that pass Benjamini-Hochberg FDR control.

    Controls the false-discovery rate at ``q`` across the strata-by-class drift tests (plan
    section 12a). A hypothesis is rejected if its p-value is at or below the largest threshold
    ``q * rank / m`` it satisfies, where ``m`` is the number of finite p-values; NaN p-values
    (a degenerate stratum) never pass.

    Parameters
    ----------
    p_values : numpy.ndarray
        The per-test permutation p-values.
    q : float, default 0.05
        The target false-discovery rate.

    Returns
    -------
    numpy.ndarray
        Boolean array, ``True`` where the test is rejected (a real drift).
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
        cutoff = np.max(np.where(passed)[0])
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
