"""Project the classes into a discriminant space and quantify how their centroids move.

The stratified fits give, for each stratum, a class-by-feature centroid aligned to the
pooled reference (:mod:`analysis.drift`). This module turns those centroids into the
material a trajectory figure needs, and measures the shape of each class's path, all in
aggregate (class-level) terms so nothing per-proband leaves the stage.

Three pieces:

- an :class:`Embedding` is a linear-discriminant projection fitted on the pooled reference
  classes. With four classes it spans three axes ($K - 1$), the coordinates in which the
  classes are maximally separated, so a class moving towards another is read directly. The
  projection is linear, so distances in it are honest, unlike a nonlinear embedding. It is
  an illustration; the drift claim rests on the full-dimensional distances of
  :mod:`analysis.drift`, not on this picture.
- :func:`directional_test` asks whether a class moves *with* the stratifying axis. The
  statistic is the net displacement from the first third of the strata to the last third,
  in standardised units. Permuting the stratum order holds the non-directional
  between-stratum scatter fixed and destroys only the ordering, so a net displacement
  beyond the shuffled null is movement tied to the axis, not scatter. This is a pilot
  measure on the observed aligned centroids; the confirmatory test is the continuous-trend
  regression against the refit permutation null (plan section 12a).
- :func:`roughness_metrics` reports the mean step between adjacent strata against the step
  that independent sampling of a class of that size would produce, so a jagged path can be
  read as sampling noise rather than movement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Features with no spread across the cohort cannot be standardised; their divisor is set to
# one so the standardised value is a constant zero rather than a division by zero.
_SD_FLOOR = 1e-9


@dataclass
class Embedding:
    """A linear-discriminant projection of the pooled reference classes.

    Attributes
    ----------
    transformer : sklearn.discriminant_analysis.LinearDiscriminantAnalysis
        The fitted transformer, taking standardised feature vectors to discriminant axes.
    mean, sd : numpy.ndarray
        Per-feature pooled mean and standard deviation used to standardise before projecting,
        in ``columns`` order.
    columns : list of str
        Feature order the transformer was fitted on.
    explained_variance_ratio : numpy.ndarray
        Fraction of between-class variance carried by each discriminant axis.
    """

    transformer: LinearDiscriminantAnalysis
    mean: np.ndarray
    sd: np.ndarray
    columns: list[str]
    explained_variance_ratio: np.ndarray

    @property
    def n_components(self) -> int:
        """Return the number of discriminant axes."""
        return int(self.explained_variance_ratio.shape[0])


def fit_embedding(
    measurement_data: pd.DataFrame, labels: pd.Series, n_components: int = 3
) -> Embedding:
    """Fit a linear-discriminant embedding of the pooled classes.

    Parameters
    ----------
    measurement_data : pandas.DataFrame
        The pooled proband-by-feature matrix.
    labels : pandas.Series
        The reference class per proband, indexed like ``measurement_data``.
    n_components : int, optional
        Discriminant axes to keep, capped at the number of classes minus one. Defaults to 3,
        the full space for a four-class solution.

    Returns
    -------
    Embedding
        The fitted projection, with the standardisation it applies before transforming.
    """
    columns = list(measurement_data.columns)
    mean = measurement_data.mean().to_numpy(dtype=float)
    sd = measurement_data.std().to_numpy(dtype=float).copy()
    sd[sd < _SD_FLOOR] = 1.0
    classes = labels.reindex(measurement_data.index).to_numpy()
    standardised = (measurement_data.to_numpy(dtype=float) - mean) / sd
    keep = min(n_components, len(np.unique(classes)) - 1)
    transformer = LinearDiscriminantAnalysis(n_components=keep)
    transformer.fit(standardised, classes)
    ratio = np.asarray(
        getattr(transformer, "explained_variance_ratio_", np.full(keep, np.nan)), dtype=float
    )
    return Embedding(transformer, mean, sd, columns, ratio)


def project(embedding: Embedding, centroids: pd.DataFrame) -> np.ndarray:
    """Project class-by-feature centroids into the discriminant axes.

    Parameters
    ----------
    embedding : Embedding
        A fitted embedding.
    centroids : pandas.DataFrame
        Centroids to project, carrying at least the embedding's feature columns.

    Returns
    -------
    numpy.ndarray
        One row per input centroid, one column per discriminant axis.
    """
    standardised = (
        centroids.loc[:, embedding.columns].to_numpy(dtype=float) - embedding.mean
    ) / embedding.sd
    return np.asarray(embedding.transformer.transform(standardised), dtype=float)


def _third(n_strata: int) -> int:
    """Return the size of the leading and trailing group of strata."""
    return max(2, n_strata // 3)


def directional_test(trajectory: np.ndarray, *, seed: int, n_shuffle: int) -> dict[str, float]:
    """Test whether one class's trajectory moves with the stratifying axis.

    The statistic is the net displacement between the first and last third of the strata, in
    standardised units. The null permutes the stratum ordering, which preserves the
    non-directional scatter and removes only the tie to the axis, so the observed value is
    read as a percentile of the shuffled distribution.

    Parameters
    ----------
    trajectory : numpy.ndarray
        The class's standardised centroids, ordered by stratum, shape ``(n_strata,
        n_features)``.
    seed : int
        Seed for the ordering shuffle, for reproducibility.
    n_shuffle : int
        Number of ordering permutations.

    Returns
    -------
    dict
        ``net`` (observed net displacement), ``null95`` (95th percentile of the null), ``p``
        (one-sided, with the Phipson-Smyth add-one), and ``significant`` (``p < 0.05``).
    """
    n_strata = trajectory.shape[0]
    k = _third(n_strata)

    def net(order: np.ndarray) -> float:
        moved = trajectory[order]
        return float(np.linalg.norm(moved[:k].mean(axis=0) - moved[-k:].mean(axis=0)))

    observed = net(np.arange(n_strata))
    rng = np.random.default_rng(seed)
    null = np.array([net(rng.permutation(n_strata)) for _ in range(n_shuffle)])
    p_value = float((np.sum(null >= observed) + 1) / (n_shuffle + 1))
    return {
        "net": observed,
        "null95": float(np.percentile(null, 95)),
        "p": p_value,
        "significant": p_value < 0.05,
    }


def roughness_metrics(
    trajectory: np.ndarray, sizes: np.ndarray, within_sd: np.ndarray
) -> dict[str, float]:
    r"""Measure a class trajectory's step size against its sampling-noise expectation.

    A stratum's centroid is a mean over that stratum's members, so two adjacent strata
    differ by sampling noise even with no real movement. The expected step under sampling
    alone is $\sqrt{\sum_f w_f^2 (1/n_i + 1/n_j)}$, where $w_f$ is the within-class
    standard deviation of feature $f$ (standardised) and $n_i, n_j$ are the adjacent class
    sizes. A step near this expectation is noise; a step well above it is movement.

    Parameters
    ----------
    trajectory : numpy.ndarray
        The class's standardised centroids, ordered by stratum, shape ``(n_strata,
        n_features)``.
    sizes : numpy.ndarray
        The class size in each stratum, in the same order.
    within_sd : numpy.ndarray
        The class's per-feature within-class standard deviation, standardised.

    Returns
    -------
    dict
        ``step`` (mean step between adjacent strata), ``sampling_noise`` (mean expected step
        under sampling), and ``snr`` (mean of their per-step ratio).
    """
    steps = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    variance = within_sd**2
    expected = np.array(
        [
            float(np.sqrt((variance * (1.0 / sizes[i] + 1.0 / sizes[i + 1])).sum()))
            for i in range(len(sizes) - 1)
        ]
    )
    return {
        "step": float(steps.mean()),
        "sampling_noise": float(expected.mean()),
        "snr": float((steps / expected).mean()),
    }
