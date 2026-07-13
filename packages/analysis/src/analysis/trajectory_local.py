r"""Local class-profile displacement along an axis, from the single cached fit (plan section 7e).

The score-based invariance test (:mod:`analysis.invariance`) is saturated at the cohort's
sample size: exact measurement invariance is always rejected once the sample runs to the
thousands, so the bridge $p$-value cannot discriminate. This module recasts the same question
around a null-free effect size. It freezes the pooled responsibilities of the measurement-only
reference and reads how each class centroid moves as a smooth function of the axis, with the
uncertainty coming from a clustered bootstrap rather than a saturated analytic null. It refits
nothing.

The quantity. For the frozen responsibilities $r_{ik}$ (the pooled ``predict_proba``), the local
centroid of class $k$ at focal point $f$ is the kernel-and-responsibility-weighted mean

.. math::

    \mu_k(f) = \frac{\sum_i w_i(f)\,r_{ik}\,x_i}{\sum_i w_i(f)\,r_{ik}},

with $w_i(f)$ the Gaussian kernel weight of proband $i$'s axis value about $f$ (the
:func:`analysis.localise.gaussian_weights` window at the axis's chosen bandwidth). With
$w \equiv 1$ this is the pooled centroid $\mu_k$, which equals the fit's responsibility-weighted
class means. The primitive is the per-feature displacement $d_k(f) = \mu_k(f) - \mu_k$, kept
full-dimensional. Magnitudes are divided by the between-class separation (the mean pairwise
distance between distinct pooled centroids, :func:`analysis.drift.class_separation`), so a
displacement is read in units of the gap between classes and is comparable across axes.

Uncertainty is a clustered bootstrap: families are resampled with replacement and the local
centroids are recomputed on the resample (re-weighting only, the responsibilities stay frozen),
giving a per-focal-point envelope. Resampling families rather than probands respects the
within-family correlation, so a block of correlated features carries an honest, wider tube.
Everything the module returns is conditional on the pooled fit.

The 2D discriminant plane (:mod:`analysis.trajectory`) is a view of the full-dimensional
displacement, not the authority: :func:`capture_fraction` reports how much of a class's
displacement lies in that plane, so a drift that is mostly out of plane cannot be hidden by the
picture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis import drift as drift_mod
from analysis import invariance
from analysis.localise import gaussian_weights
from analysis.trajectory import Embedding

# A local centroid divides by the summed weight-times-responsibility of a class. Where that sum
# is negligible (a class with almost no responsibility in a region), the centroid is undefined
# and set to not-a-number rather than dividing by near-zero.
_WEIGHT_FLOOR = 1e-9


def local_centroids(
    x_values: np.ndarray, responsibilities: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    r"""Return the kernel-and-responsibility-weighted class centroids at one focal point.

    The centroid of class $k$ is $\sum_i w_i r_{ik} x_i / \sum_i w_i r_{ik}$, the local weighted
    mean of the feature matrix under the frozen responsibilities. With ``weights`` a Gaussian
    kernel window this is the local centroid $\mu_k(f)$; with unit weights it is the pooled
    centroid $\mu_k$ (:func:`pooled_centroids`).

    Parameters
    ----------
    x_values : numpy.ndarray
        The measurement matrix, shape ``(n_probands, n_features)``.
    responsibilities : numpy.ndarray
        The frozen posterior responsibilities $r_{ik}$, shape ``(n_probands, n_classes)``.
    weights : numpy.ndarray
        The per-proband kernel weight, shape ``(n_probands,)``.

    Returns
    -------
    numpy.ndarray
        The class centroids, shape ``(n_classes, n_features)``; a class with no local weight is
        all not-a-number.
    """
    wr = weights[:, None] * responsibilities
    denom = wr.sum(axis=0)
    numer = wr.T @ x_values
    out = np.full((responsibilities.shape[1], x_values.shape[1]), np.nan)
    good = denom > _WEIGHT_FLOOR
    out[good] = numer[good] / denom[good, None]
    return out


def pooled_centroids(x_values: np.ndarray, responsibilities: np.ndarray) -> np.ndarray:
    """Return the pooled (whole-cohort) responsibility-weighted class centroids.

    The local centroids with a unit weight on every proband, so this is the frozen-responsibility
    class mean each local centroid is measured against.
    """
    return local_centroids(x_values, responsibilities, np.ones(x_values.shape[0]))


def separation(reference: drift_mod.ReferenceModel) -> float:
    """Return the between-class separation, the drift baseline (delegated to :mod:`analysis.drift`).

    The mean standardised-Euclidean distance between distinct pooled centroids, exactly
    :func:`analysis.drift.class_separation` under the standardised-Euclidean distance, so the
    separation scaling here uses the same convention as the refit-based drift stage.
    """
    return drift_mod.class_separation(reference, drift_mod.StandardisedEuclidean())


def grain_magnitude(
    displacement: np.ndarray, pooled_sd: np.ndarray, columns: np.ndarray, separation_scale: float
) -> np.ndarray:
    r"""Return the separation-scaled displacement magnitude of a feature grain, per class.

    A grain is a set of feature columns (the 4 classes are read whole, and each class is also
    read within each of the 7 author categories). Its magnitude is the Euclidean norm of the
    per-feature displacement in pooled-standard-deviation units over the grain's features,
    $\lVert d_k / \sigma \rVert$, divided by the between-class separation. The raw (unaveraged)
    norm is deliberate: a larger grain carries a larger norm and a wider bootstrap tube, and the
    tube, not the bare magnitude, calls significance.

    Parameters
    ----------
    displacement : numpy.ndarray
        The per-feature displacement, shape ``(..., n_features)``; the leading axes are kept.
    pooled_sd : numpy.ndarray
        The per-feature pooled standard deviation, shape ``(n_features,)``.
    columns : numpy.ndarray
        The integer column indices of the grain's features.
    separation_scale : float
        The between-class separation the magnitude is divided by.

    Returns
    -------
    numpy.ndarray
        The separation-scaled magnitude, shape ``displacement.shape[:-1]``.
    """
    z = displacement[..., columns] / pooled_sd[columns]
    return np.sqrt(np.nansum(z**2, axis=-1)) / separation_scale


def discriminant_plane(embedding: Embedding) -> np.ndarray:
    """Return an orthonormal basis of the first two discriminant directions, in standardised space.

    The embedding maps a standardised feature vector to the discriminant axes by the linear
    ``scalings_``; the plane the trajectory figure draws is the span of the first two of those
    direction vectors. Orthonormalising that span (a thin QR) gives a projector onto the plane,
    so the in-plane part of a displacement can be measured honestly even though the raw scaling
    vectors are not orthogonal.

    Returns
    -------
    numpy.ndarray
        The orthonormal basis, shape ``(n_features, 2)``.
    """
    directions = np.asarray(embedding.transformer.scalings_, dtype=float)[:, :2]
    basis, _ = np.linalg.qr(directions)
    return basis


def capture_fraction(
    displacement_row: np.ndarray, pooled_sd: np.ndarray, plane: np.ndarray
) -> float:
    r"""Return the fraction of one class's displacement that lies in the discriminant plane.

    The honesty guard on the figure: $\lVert P\,d_k \rVert / \lVert d_k \rVert$, where $d_k$ is
    the class's standardised displacement and $P$ orthogonally projects onto the plane. A value
    near one means the 2D picture shows essentially all of the movement; a value near zero means
    the movement is mostly out of plane and the picture understates it.

    Parameters
    ----------
    displacement_row : numpy.ndarray
        One class's per-feature displacement, shape ``(n_features,)``.
    pooled_sd : numpy.ndarray
        The per-feature pooled standard deviation, shape ``(n_features,)``.
    plane : numpy.ndarray
        An orthonormal basis of the plane, shape ``(n_features, 2)`` (:func:`discriminant_plane`).

    Returns
    -------
    float
        The in-plane capture fraction, or not-a-number when the displacement is zero.
    """
    z = displacement_row / pooled_sd
    total = float(np.linalg.norm(z))
    if total == 0.0 or not np.isfinite(total):
        return float("nan")
    in_plane = plane @ (plane.T @ z)
    return float(np.linalg.norm(in_plane) / total)


def mahalanobis_magnitude(displacement_row: np.ndarray, precision: np.ndarray) -> float:
    r"""Return the Mahalanobis magnitude of one class's raw displacement.

    The covariance-aware corroborating magnitude, $\sqrt{d_k^\top \Sigma^{-1} d_k}$ with the
    Ledoit-Wolf-shrunk pooled within-class precision, so a coordinated shift across correlated
    features counts once. This is the :mod:`analysis.drift` Mahalanobis distance evaluated on the
    local displacement; bootstrap-calibrating it (its clustered-bootstrap band) makes it
    dimension-fair across grains of different size.

    Parameters
    ----------
    displacement_row : numpy.ndarray
        One class's raw per-feature displacement, shape ``(n_features,)``.
    precision : numpy.ndarray
        The pooled within-class precision, shape ``(n_features, n_features)``.
    """
    finite = np.where(np.isfinite(displacement_row), displacement_row, 0.0)
    return float(np.sqrt(max(0.0, finite @ precision @ finite)))


@dataclass
class ObservedTrajectory:
    """The observed local-displacement trajectory of a fit against one axis.

    Attributes
    ----------
    focal_points : numpy.ndarray
        The axis positions the local centroids were read at, shape ``(n_focal,)``.
    pooled : numpy.ndarray
        The pooled class centroids, shape ``(n_classes, n_features)``.
    displacement : numpy.ndarray
        The per-feature displacement $d_k(f)$, shape ``(n_classes, n_focal, n_features)``.
    ld : numpy.ndarray
        The local centroids' first two discriminant coordinates, shape
        ``(n_classes, n_focal, 2)``, the trajectory the plane figure draws.
    grain_magnitude : dict of str to numpy.ndarray
        Per grain, the separation-scaled magnitude, shape ``(n_classes, n_focal)``.
    mahalanobis : numpy.ndarray
        The whole-class Mahalanobis magnitude, shape ``(n_classes, n_focal)``.
    capture : numpy.ndarray
        The per-class in-plane capture fraction of the endpoint displacement, shape
        ``(n_classes,)``.
    focal_ref : int
        The focal index the capture fraction and per-feature inference are anchored at: the
        endpoint (the last focal point), pre-specified so the per-feature test is not selected on
        the observed magnitude. The endpoint carries the accumulated drift for a monotone or
        single-break axis.
    peak_focal : numpy.ndarray
        Per class, the focal index of the largest whole-class magnitude, reported as where the
        drift is strongest (informational, not a test anchor), shape ``(n_classes,)``.
    """

    focal_points: np.ndarray
    pooled: np.ndarray
    displacement: np.ndarray
    ld: np.ndarray
    grain_magnitude: dict[str, np.ndarray]
    mahalanobis: np.ndarray
    capture: np.ndarray
    focal_ref: int
    peak_focal: np.ndarray


def _project_ld(centroids: np.ndarray, embedding: Embedding) -> np.ndarray:
    """Project raw class centroids into the first two discriminant coordinates."""
    z = (centroids - embedding.mean) / embedding.sd
    return np.asarray(embedding.transformer.transform(z), dtype=float)[:, :2]


def observed_trajectory(
    x_values: np.ndarray,
    responsibilities: np.ndarray,
    axis_values: np.ndarray,
    focal_points: np.ndarray,
    bandwidth: float,
    *,
    pooled_sd: np.ndarray,
    separation_scale: float,
    grains: dict[str, np.ndarray],
    embedding: Embedding,
    precision: np.ndarray,
    plane: np.ndarray,
) -> ObservedTrajectory:
    """Compute the observed local-displacement trajectory of a fit against an axis.

    Reads the local centroids at each focal point under the frozen responsibilities, forms the
    per-feature displacement from the pooled centroid, and derives the discriminant-plane
    coordinates, the separation-scaled grain magnitudes, the Mahalanobis magnitude, and the
    endpoint capture fraction. Pure and cheap: no fitting, only re-weighting.

    Parameters
    ----------
    x_values, responsibilities, axis_values : numpy.ndarray
        The measurement matrix, the frozen responsibilities, and the per-proband axis value.
    focal_points : numpy.ndarray
        The axis positions to read local centroids at.
    bandwidth : float
        The Gaussian kernel bandwidth, in axis units.
    pooled_sd : numpy.ndarray
        The per-feature pooled standard deviation.
    separation_scale : float
        The between-class separation the magnitudes are divided by.
    grains : dict of str to numpy.ndarray
        The feature-column indices of each grain (whole-class plus per author category).
    embedding : analysis.trajectory.Embedding
        The fixed discriminant embedding of the pooled classes.
    precision : numpy.ndarray
        The pooled within-class precision, for the Mahalanobis magnitude.
    plane : numpy.ndarray
        The orthonormal discriminant plane, for the capture fraction.

    Returns
    -------
    ObservedTrajectory
        The observed trajectory and its derived magnitudes.
    """
    axis_values = np.asarray(axis_values, dtype=float)
    n_classes = responsibilities.shape[1]
    n_focal = len(focal_points)
    pooled = pooled_centroids(x_values, responsibilities)

    displacement = np.empty((n_classes, n_focal, x_values.shape[1]))
    ld = np.empty((n_classes, n_focal, 2))
    axis_series = pd.Series(axis_values)
    for j, focal in enumerate(focal_points):
        weights = gaussian_weights(axis_series, float(focal), bandwidth).to_numpy()
        centroids = local_centroids(x_values, responsibilities, weights)
        displacement[:, j, :] = centroids - pooled
        ld[:, j, :] = _project_ld(centroids, embedding)

    magnitudes = {
        name: grain_magnitude(displacement, pooled_sd, cols, separation_scale)
        for name, cols in grains.items()
    }
    mahalanobis = np.array(
        [
            [mahalanobis_magnitude(displacement[k, j], precision) for j in range(n_focal)]
            for k in range(n_classes)
        ]
    )
    whole = magnitudes["class"]
    focal_ref = n_focal - 1
    peak_focal = np.nanargmax(whole, axis=1)
    capture = np.array(
        [capture_fraction(displacement[k, focal_ref], pooled_sd, plane) for k in range(n_classes)]
    )
    return ObservedTrajectory(
        focal_points=np.asarray(focal_points, dtype=float),
        pooled=pooled,
        displacement=displacement,
        ld=ld,
        grain_magnitude=magnitudes,
        mahalanobis=mahalanobis,
        capture=capture,
        focal_ref=focal_ref,
        peak_focal=peak_focal,
    )


@dataclass
class BootstrapTube:
    """The clustered-bootstrap envelope of a displacement trajectory.

    Attributes
    ----------
    quantiles : tuple of float
        The bootstrap quantiles held in each band, in order (typically low, median, high).
    ld : numpy.ndarray
        The local centroids' discriminant coordinates over the replicates, shape
        ``(n_boot, n_classes, n_focal, 2)``, the centroid tube the plane figure draws.
    grain_bands : dict of str to numpy.ndarray
        Per grain, the quantile bands of the separation-scaled magnitude, shape
        ``(n_quantiles, n_classes, n_focal)``.
    mahalanobis_bands : numpy.ndarray
        The quantile bands of the whole-class Mahalanobis magnitude, shape
        ``(n_quantiles, n_classes, n_focal)``.
    feature_displacement : numpy.ndarray
        The standardised per-feature displacement at the reference (endpoint) focal point, over
        the replicates, shape ``(n_boot, n_classes, n_features)``.
    n_boot : int
        The number of bootstrap replicates.
    clustered : bool
        Whether families (``True``) or probands (``False``) were resampled.
    """

    quantiles: tuple[float, ...]
    ld: np.ndarray
    grain_bands: dict[str, np.ndarray]
    mahalanobis_bands: np.ndarray
    feature_displacement: np.ndarray
    n_boot: int
    clustered: bool


def _family_rows(families: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Return the row positions grouped by family and the array of family keys.

    The family key may be a string (SPARK's ``family_sf_id``), so the run boundaries are found by
    an elementwise inequality rather than a numeric difference.
    """
    if families.shape[0] == 0:
        return [], np.array([])
    order = np.argsort(families, kind="stable")
    sorted_fam = families[order]
    boundaries = np.flatnonzero(sorted_fam[1:] != sorted_fam[:-1]) + 1
    groups = np.split(order, boundaries)
    keys = sorted_fam[np.concatenate([[0], boundaries])]
    return groups, keys


def clustered_bootstrap_tube(
    x_values: np.ndarray,
    responsibilities: np.ndarray,
    axis_values: np.ndarray,
    families: np.ndarray,
    focal_points: np.ndarray,
    bandwidth: float,
    *,
    pooled_sd: np.ndarray,
    separation_scale: float,
    grains: dict[str, np.ndarray],
    embedding: Embedding,
    precision: np.ndarray,
    focal_ref: int,
    n_boot: int,
    seed: int,
    clustered: bool = True,
    quantiles: tuple[float, ...] = (2.5, 50.0, 97.5),
) -> BootstrapTube:
    """Bootstrap the displacement trajectory by resampling families (or probands).

    Each replicate resamples whole families with replacement (so a proband appearing twice
    contributes twice), recomputes the pooled and local centroids on the resample under the
    frozen responsibilities, and records the discriminant coordinates, the separation-scaled
    grain magnitudes, the Mahalanobis magnitude, and the per-feature displacement at each class's
    reported focal point. The per-focal-point quantiles of those are the tube. Setting
    ``clustered=False`` resamples individual probands instead, the independent-bootstrap
    comparison that shows the family clustering is real rather than cosmetic.

    Parameters
    ----------
    x_values, responsibilities, axis_values : numpy.ndarray
        The measurement matrix, frozen responsibilities, and per-proband axis value.
    families : numpy.ndarray
        The per-proband family identifier, shape ``(n_probands,)``; the clustering unit.
    focal_points : numpy.ndarray
        The axis positions to read local centroids at.
    bandwidth : float
        The Gaussian kernel bandwidth.
    pooled_sd, separation_scale, grains, embedding, precision
        As described for :func:`observed_trajectory`.
    focal_ref : int
        The focal index the per-feature displacement is recorded at (the endpoint).
    n_boot : int
        The number of bootstrap replicates.
    seed : int
        The base seed for the resampling.
    clustered : bool, optional
        Resample families (default) or individual probands.
    quantiles : tuple of float, optional
        The bootstrap quantiles kept in each band.

    Returns
    -------
    BootstrapTube
        The bootstrap replicates and their per-focal-point quantile bands.
    """
    axis_values = np.asarray(axis_values, dtype=float)
    n_classes = responsibilities.shape[1]
    n_features = x_values.shape[1]
    n_focal = len(focal_points)
    n_probands = x_values.shape[0]
    rng = np.random.default_rng(seed)

    groups, _ = _family_rows(np.asarray(families))
    n_groups = len(groups)

    # Precompute the kernel weight of every proband at every focal point once; a resample only
    # re-selects rows, it does not change a proband's weight at a focal point.
    axis_series = pd.Series(axis_values)
    focal_weights = np.column_stack(
        [gaussian_weights(axis_series, float(f), bandwidth).to_numpy() for f in focal_points]
    )

    ld = np.empty((n_boot, n_classes, n_focal, 2))
    grain_draws = {name: np.empty((n_boot, n_classes, n_focal)) for name in grains}
    maha_draws = np.empty((n_boot, n_classes, n_focal))
    feature_draws = np.empty((n_boot, n_classes, n_features))

    for b in range(n_boot):
        if clustered:
            chosen = rng.integers(0, n_groups, size=n_groups)
            rows = np.concatenate([groups[c] for c in chosen])
        else:
            rows = rng.integers(0, n_probands, size=n_probands)
        xb = x_values[rows]
        rb = responsibilities[rows]
        wb = focal_weights[rows]
        pooledb = pooled_centroids(xb, rb)
        for j in range(n_focal):
            centroids = local_centroids(xb, rb, wb[:, j])
            disp = centroids - pooledb
            ld[b, :, j, :] = _project_ld(centroids, embedding)
            for name, cols in grains.items():
                grain_draws[name][b, :, j] = grain_magnitude(
                    disp, pooled_sd, cols, separation_scale
                )
            maha_draws[b, :, j] = [
                mahalanobis_magnitude(disp[k], precision) for k in range(n_classes)
            ]
            if j == focal_ref:
                feature_draws[b, :, :] = disp / pooled_sd

    grain_bands = {
        name: np.stack([np.nanpercentile(draws, q, axis=0) for q in quantiles])
        for name, draws in grain_draws.items()
    }
    maha_bands = np.stack([np.nanpercentile(maha_draws, q, axis=0) for q in quantiles])
    return BootstrapTube(
        quantiles=quantiles,
        ld=ld,
        grain_bands=grain_bands,
        mahalanobis_bands=maha_bands,
        feature_displacement=feature_draws,
        n_boot=n_boot,
        clustered=clustered,
    )


@dataclass
class FeatureInference:
    """The per-feature displacement, its clustered-bootstrap interval, and the FDR decision.

    Attributes
    ----------
    displacement : numpy.ndarray
        The observed standardised per-feature displacement at each class's reported focal point,
        shape ``(n_classes, n_features)``.
    ci_low, ci_high : numpy.ndarray
        The bootstrap interval per feature, shape ``(n_classes, n_features)``.
    p_value : numpy.ndarray
        The two-sided bootstrap $p$-value that the displacement differs from zero, shape
        ``(n_classes, n_features)``.
    reject : numpy.ndarray
        The Benjamini-Hochberg decision across the ``n_classes * n_features`` tests, shape
        ``(n_classes, n_features)``.
    covers_zero : numpy.ndarray
        Whether the bootstrap interval covers zero, shape ``(n_classes, n_features)``; most being
        true is the readable "many features invariant".
    """

    displacement: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    p_value: np.ndarray
    reject: np.ndarray
    covers_zero: np.ndarray


def per_feature_inference(
    observed_displacement: np.ndarray, feature_draws: np.ndarray, *, q: float = 0.05
) -> FeatureInference:
    r"""Test each per-feature displacement against zero with a clustered-bootstrap interval and FDR.

    Reads the 95 per cent bootstrap interval of each ``(class, feature)`` displacement, forms a
    two-sided bootstrap $p$-value from the fraction of replicates on the far side of zero, and
    applies Benjamini-Hochberg control across the ``4 * n_features`` tests (the
    :func:`analysis.invariance.benjamini_hochberg` implementation, the repo convention). A
    displacement whose interval covers zero is invariant at this level; most covering zero is the
    "many features invariant" reading.

    Parameters
    ----------
    observed_displacement : numpy.ndarray
        The observed standardised per-feature displacement, shape ``(n_classes, n_features)``.
    feature_draws : numpy.ndarray
        The bootstrap replicates, shape ``(n_boot, n_classes, n_features)``
        (:attr:`BootstrapTube.feature_displacement`).
    q : float, optional
        The false-discovery-rate level.

    Returns
    -------
    FeatureInference
        The per-feature displacement, interval, $p$-value, and FDR decision.
    """
    n_boot = feature_draws.shape[0]
    ci_low = np.nanpercentile(feature_draws, 2.5, axis=0)
    ci_high = np.nanpercentile(feature_draws, 97.5, axis=0)
    frac_positive = np.mean(feature_draws > 0.0, axis=0)
    # Two-sided add-one bootstrap p, floored at 1/(n_boot + 1) so a displacement beyond every
    # replicate is not reported as impossible.
    tail = np.minimum(frac_positive, 1.0 - frac_positive)
    p_value = np.clip(2.0 * tail, 1.0 / (n_boot + 1), 1.0)
    reject = invariance.benjamini_hochberg(p_value.ravel(), q).reshape(p_value.shape)
    covers_zero = (ci_low <= 0.0) & (ci_high >= 0.0)
    return FeatureInference(
        displacement=observed_displacement,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        reject=reject,
        covers_zero=covers_zero,
    )


def category_grains(columns: list[str], category_map: dict[str, str]) -> dict[str, np.ndarray]:
    """Return the column indices of each presentation grain: whole-class and per author category.

    The whole-class grain (``"class"``) is every feature; each author category grain
    (``"category:<name>"``) is the features mapped to that category. A category with no present
    feature is omitted. The grains are the pre-specified aggregation levels, fixed before the
    data are seen.

    Parameters
    ----------
    columns : list of str
        The measurement-matrix feature columns, in order.
    category_map : dict of str to str
        The feature-to-category mapping.

    Returns
    -------
    dict of str to numpy.ndarray
        The grain name mapped to its integer column indices.
    """
    grains: dict[str, np.ndarray] = {"class": np.arange(len(columns))}
    by_category: dict[str, list[int]] = {}
    for i, feature in enumerate(columns):
        category = category_map.get(str(feature))
        if category is None or (isinstance(category, float) and np.isnan(category)):
            continue
        by_category.setdefault(str(category), []).append(i)
    for name in sorted(by_category):
        grains[f"category:{name}"] = np.asarray(by_category[name], dtype=int)
    return grains
