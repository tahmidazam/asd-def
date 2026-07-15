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
distance between distinct pooled centroids under the same full standardised-Euclidean norm,
:func:`separation`), so one separation unit is the mean inter-class gap and a displacement reads
as a genuine fraction of that gap, comparable across axes.

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

    The full (unaveraged) standardised-Euclidean distance
    (:class:`analysis.drift.FullStandardisedEuclidean`), the same sum-norm convention
    :func:`grain_magnitude` uses for a class's displacement. Numerator and denominator then share
    a scale, so a separation-scaled magnitude is a genuine fraction of the mean inter-class gap:
    one separation unit is the mean pairwise distance between distinct reference centroids. The
    refit-based drift stage keeps its own averaged convention, self-consistent within that stage.
    """
    return drift_mod.class_separation(reference, drift_mod.FullStandardisedEuclidean())


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
    signed_slope : numpy.ndarray or None
        The directional draws: per replicate, each class's signed net-projected slope (the
        directional statistic of :func:`directional_statistic`), shape ``(n_boot, n_classes)``.
        ``None`` when the tube was built without frozen net directions.
    net_trend : numpy.ndarray or None
        Per replicate, each class's separation-scaled net-trend displacement over the focal span,
        shape ``(n_boot, n_classes)``; ``None`` as above.
    signed_trajectory : numpy.ndarray or None
        Per replicate, each class's one-dimensional signed trajectory projected onto its frozen
        net direction, shape ``(n_boot, n_classes, n_focal)``; ``None`` as above. The band of this
        is what the directional figure draws.
    break_position : numpy.ndarray or None
        Per replicate, each class's single-break location on the signed trajectory, shape
        ``(n_boot, n_classes)``; ``None`` as above.
    """

    quantiles: tuple[float, ...]
    ld: np.ndarray
    grain_bands: dict[str, np.ndarray]
    mahalanobis_bands: np.ndarray
    feature_displacement: np.ndarray
    n_boot: int
    clustered: bool
    signed_slope: np.ndarray | None = None
    net_trend: np.ndarray | None = None
    signed_trajectory: np.ndarray | None = None
    break_position: np.ndarray | None = None


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
    net_directions: np.ndarray | None = None,
) -> BootstrapTube:
    """Bootstrap the displacement trajectory by resampling families (or probands).

    Each replicate resamples whole families with replacement (so a proband appearing twice
    contributes twice), recomputes the pooled and local centroids on the resample under the
    frozen responsibilities, and records the discriminant coordinates, the separation-scaled
    grain magnitudes, the Mahalanobis magnitude, and the per-feature displacement at each class's
    reported focal point. The per-focal-point quantiles of those are the tube. Setting
    ``clustered=False`` resamples individual probands instead, the independent-bootstrap
    comparison that shows the family clustering is real rather than cosmetic.

    When ``net_directions`` is given (each class's frozen unit direction from
    :func:`directional_statistic`), each replicate also records the directional draws: the signed
    net-projected slope, the separation-scaled net-trend displacement, the one-dimensional signed
    trajectory, and the single-break location. Their per-class spread is the clustered-bootstrap
    null the H0E directional test reads against; freezing the direction at the observed value
    keeps the projected slope a fixed linear functional, so it is signed and its interval can
    honestly cover zero.

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
    net_directions : numpy.ndarray, optional
        Each class's frozen unit net direction, shape ``(n_classes, n_features)``. When given, the
        directional draws are recorded; when ``None`` they are left off the tube.

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
    directional = net_directions is not None
    focal_arr = np.asarray(focal_points, dtype=float)
    span = float(focal_arr.max() - focal_arr.min()) if n_focal else 0.0
    signed_draws = np.empty((n_boot, n_classes)) if directional else None
    trend_draws = np.empty((n_boot, n_classes)) if directional else None
    signed_traj_draws = np.empty((n_boot, n_classes, n_focal)) if directional else None
    break_draws = np.empty((n_boot, n_classes)) if directional else None

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
        traj_std = np.empty((n_classes, n_focal, n_features)) if directional else None
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
            std_disp = disp / pooled_sd
            if traj_std is not None:
                traj_std[:, j, :] = std_disp
            if j == focal_ref:
                feature_draws[b, :, :] = std_disp
        if directional:
            assert traj_std is not None and net_directions is not None
            assert signed_draws is not None and trend_draws is not None
            assert signed_traj_draws is not None and break_draws is not None
            slope = slope_vectors(traj_std, focal_arr)
            signed = np.nansum(slope * net_directions, axis=1)
            traj_1d = project_onto(traj_std, net_directions)
            signed_draws[b] = signed
            trend_draws[b] = signed * span / separation_scale
            signed_traj_draws[b] = traj_1d
            break_draws[b] = [single_break(focal_arr, traj_1d[k]) for k in range(n_classes)]

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
        signed_slope=signed_draws,
        net_trend=trend_draws,
        signed_trajectory=signed_traj_draws,
        break_position=break_draws,
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


@dataclass
class ControlComparison:
    r"""A paired-bootstrap comparison of a timing axis against one control variable.

    Attributes
    ----------
    axis_magnitude, control_magnitude : float
        The observed, class-averaged, separation-scaled endpoint magnitude of the timing axis and
        of the control variable (household income, area deprivation, or a random ordering).
    difference : float
        ``axis_magnitude - control_magnitude``.
    diff_draws : numpy.ndarray
        The paired-bootstrap replicate differences, shape ``(n_boot,)``.
    p_value : float
        The two-sided bootstrap $p$-value that the difference is zero, floored at
        $1/(n_{\text{boot}} + 1)$.
    p_value_greater : float
        The one-sided bootstrap $p$-value that the axis magnitude exceeds the control's, floored
        the same way.
    n_boot : int
        The number of paired bootstrap replicates.
    """

    axis_magnitude: float
    control_magnitude: float
    difference: float
    diff_draws: np.ndarray
    p_value: float
    p_value_greater: float
    n_boot: int


def control_specificity_bootstrap(
    x_values: np.ndarray,
    responsibilities: np.ndarray,
    families: np.ndarray,
    axis_values: np.ndarray,
    axis_bandwidth: float,
    axis_focal: float,
    control_values: np.ndarray,
    control_bandwidth: float,
    control_focal: float,
    *,
    pooled_sd: np.ndarray,
    separation_scale: float,
    n_boot: int,
    seed: int,
) -> ControlComparison:
    r"""Paired family-bootstrap test that a timing axis's drift exceeds a control's.

    The specificity panel (the ``invariance-as-an-effect-size`` guide) reads
    the timing axis's endpoint magnitude as larger than a control's, but as a magnitude comparison
    only, because the axis and the control were each read from one point estimate. This adds a
    $p$-value: every bootstrap replicate resamples one set of families and recomputes *both* the
    axis and the control magnitude on that same resample, so the two quantities share their
    sampling variation and the difference is a genuine paired statistic, not the comparison of two
    separately noisy numbers. The observed difference then acts as its own bootstrap-inverted
    test: :math:`p` is the fraction of replicate differences on the far side of zero (doubled for
    the two-sided form), the same construction :func:`per_feature_inference` and
    :func:`directional_inference` use.

    ``x_values``, ``responsibilities``, and ``families`` must already be restricted to the
    probands finite on *both* ``axis_values`` and ``control_values``, so that a family resampled
    for one quantity is resampled for the other. ``axis_focal`` and ``control_focal`` are each
    variable's own endpoint focal position (its own bandwidth and grid), matching how the
    specificity panel reads each variable.

    Parameters
    ----------
    x_values, responsibilities : numpy.ndarray
        The measurement matrix and frozen responsibilities, restricted to the shared rows.
    families : numpy.ndarray
        The per-proband family identifier over the same rows, the clustering unit.
    axis_values, control_values : numpy.ndarray
        The timing axis and the control variable, over the same rows.
    axis_bandwidth, control_bandwidth : float
        Each variable's own Gaussian kernel bandwidth.
    axis_focal, control_focal : float
        Each variable's own endpoint focal position.
    pooled_sd : numpy.ndarray
        The per-feature pooled standard deviation.
    separation_scale : float
        The between-class separation the magnitude is divided by.
    n_boot : int
        The number of paired bootstrap replicates.
    seed : int
        The bootstrap seed.

    Returns
    -------
    ControlComparison
        The observed magnitudes, their difference, and its bootstrap $p$-values.
    """
    rng = np.random.default_rng(seed)
    groups, _ = _family_rows(np.asarray(families))
    n_groups = len(groups)
    columns = np.arange(x_values.shape[1])

    axis_weight = gaussian_weights(pd.Series(axis_values), float(axis_focal), axis_bandwidth)
    axis_weight = axis_weight.to_numpy()
    control_weight = gaussian_weights(
        pd.Series(control_values), float(control_focal), control_bandwidth
    )
    control_weight = control_weight.to_numpy()

    def mean_magnitude(xb: np.ndarray, rb: np.ndarray, wb: np.ndarray) -> float:
        pooledb = pooled_centroids(xb, rb)
        centroids = local_centroids(xb, rb, wb)
        disp = centroids - pooledb
        return float(np.mean(grain_magnitude(disp, pooled_sd, columns, separation_scale)))

    observed_axis = mean_magnitude(x_values, responsibilities, axis_weight)
    observed_control = mean_magnitude(x_values, responsibilities, control_weight)

    diff_draws = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, n_groups, size=n_groups)
        rows = np.concatenate([groups[c] for c in chosen])
        xb = x_values[rows]
        rb = responsibilities[rows]
        axis_b = mean_magnitude(xb, rb, axis_weight[rows])
        control_b = mean_magnitude(xb, rb, control_weight[rows])
        diff_draws[b] = axis_b - control_b

    floor = 1.0 / (n_boot + 1)
    frac_positive = float(np.mean(diff_draws > 0.0))
    tail = min(frac_positive, 1.0 - frac_positive)
    p_value = float(np.clip(2.0 * tail, floor, 1.0))
    p_value_greater = float(np.clip(1.0 - frac_positive, floor, 1.0))

    return ControlComparison(
        axis_magnitude=observed_axis,
        control_magnitude=observed_control,
        difference=observed_axis - observed_control,
        diff_draws=diff_draws,
        p_value=p_value,
        p_value_greater=p_value_greater,
        n_boot=n_boot,
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


# =============================================================================================
# H0G: the referent decomposition of the era drift (plan sections 6, 7e, 12b;
# H0G).
#
# H0F asks which symptom category the drift sits in; H0G asks a different, mechanism-
# discriminating question: does the drift concentrate in instruments that ask about the child's
# present state (RBS-R, CBCL 6-18) or in instruments that ask about the developmental history and
# whether a behaviour was ever present (the SCQ Lifetime form, the developmental milestones)?
# Concentration in the current-state instruments is the signature of a change in measurement
# timing; concentration in the retrospective and lifetime instruments is the signature of a
# genuine change in the diagnosed population. The instruments split into two referents, and each
# feature carries its instrument's referent.
#
# The statistic is the per-class current-minus-retrospective contrast of the per-feature root-
# mean-square displacement intensity. The mean square, not the raw sum, makes the contrast
# size-fair: the current-state grain holds many more features than the retrospective grain (193
# against 45 on the reference set), so a raw sum-of-squares would favour it by feature count
# alone. Under the null the drift is spread at equal per-feature intensity across referents, so
# the two grains carry the same root-mean-square and the contrast is zero. The additive sum-of-
# squares share is reported alongside as a descriptive decomposition (it sums to one over the
# disjoint referents), but the test reads the size-fair contrast.
# =============================================================================================


def referent_grains(
    columns: list[str], instrument_map: dict[str, str], referent_map: dict[str, str]
) -> dict[str, np.ndarray]:
    """Return the column indices of each referent grain: per instrument and per temporal referent.

    Each feature carries the instrument it comes from (``instrument_map``, derived from the data
    dictionary) and each instrument carries a pre-registered temporal referent (``referent_map``,
    ``analysis.features.INSTRUMENT_REFERENT``). The grains are the per-instrument column sets
    (``"instrument:<name>"``, the transparent underlay) and the per-referent column sets
    (``"referent:<name>"``, the two-way headline). Resolution fails loudly, mirroring
    :func:`analysis.features.reconcile`'s no-typing-signal guard: a feature with no instrument, or
    an instrument with no referent, raises rather than being dropped, so a mapping gap cannot pass
    silently as an empty grain.

    Parameters
    ----------
    columns : list of str
        The measurement-matrix feature columns, in order.
    instrument_map : dict of str to str
        The feature-to-instrument mapping.
    referent_map : dict of str to str
        The instrument-to-referent mapping.

    Returns
    -------
    dict of str to numpy.ndarray
        The grain name mapped to its integer column indices.

    Raises
    ------
    ValueError
        When a feature resolves to no instrument, or its instrument to no referent.
    """
    by_instrument: dict[str, list[int]] = {}
    by_referent: dict[str, list[int]] = {}
    for i, feature in enumerate(columns):
        instrument = instrument_map.get(str(feature))
        if instrument is None:
            raise ValueError(f"no instrument for feature {feature!r}")
        referent = referent_map.get(instrument)
        if referent is None:
            raise ValueError(f"no referent for instrument {instrument!r}")
        by_instrument.setdefault(instrument, []).append(i)
        by_referent.setdefault(referent, []).append(i)
    grains: dict[str, np.ndarray] = {}
    for name in sorted(by_instrument):
        grains[f"instrument:{name}"] = np.asarray(by_instrument[name], dtype=int)
    for name in sorted(by_referent):
        grains[f"referent:{name}"] = np.asarray(by_referent[name], dtype=int)
    return grains


# =============================================================================================
# H0E: the directionality of the drift (plan sections 7e, 12b; H0E).
#
# H0D asks how far a class drifts; H0E asks whether that drift has a systematic trend along
# the axis, as opposed to a non-directional excursion. The distinction matters because the local
# centroid is nearest the pooled centroid at the axis interior, where the kernel window is most
# balanced, so the magnitude |d_k(f)| is mechanically U-shaped and cannot answer direction. The
# directional statistic is therefore built on the signed displacement and its slope, never on the
# magnitude norm.
#
# The primitive is the per-feature ordinary-least-squares slope of the standardised displacement
# d_k(f) / sigma against the axis position f, a slope vector b_k. Reducing b_k to a scalar by its
# Euclidean norm would test direction, but the norm is positively biased (a class with no trend
# still returns a positive norm from noise), so it cannot honestly cover zero. Instead the slope
# is projected onto the class's net direction, the unit vector of its mean standardised
# displacement across the focal grid. On an evenly spaced focal grid the slope contrast and the
# mean contrast are orthogonal, so under no drift the projected slope has zero expectation: it is
# a signed, unbiased directional statistic whose clustered-bootstrap interval can cover zero.
# =============================================================================================


def slope_vectors(traj_std: np.ndarray, focal_points: np.ndarray) -> np.ndarray:
    r"""Return each class's per-feature ordinary-least-squares slope against the axis.

    For the standardised displacement trajectory $D_k(f) = d_k(f) / \sigma$, the slope of feature
    $m$ is $\sum_j (f_j - \bar f)\,D_k(f_j)[m] / \sum_j (f_j - \bar f)^2$, the univariate
    least-squares slope of that feature's displacement on the axis position. The focal grid is
    evenly spaced in axis units (:func:`analysis.localise.focal_grid`), so an equal weight per
    focal point is an honest per-axis-unit trend on an irregularly sampled axis. A class is
    regressed only over the focal points where its local centroid is defined; a class defined at
    fewer than two focal points has an all-not-a-number slope.

    Parameters
    ----------
    traj_std : numpy.ndarray
        The standardised displacement trajectory, shape ``(n_classes, n_focal, n_features)``.
    focal_points : numpy.ndarray
        The axis positions the trajectory was read at, shape ``(n_focal,)``.

    Returns
    -------
    numpy.ndarray
        The per-class slope vector $b_k$, shape ``(n_classes, n_features)``.
    """
    focal = np.asarray(focal_points, dtype=float)
    n_classes, _, n_features = traj_std.shape
    out = np.full((n_classes, n_features), np.nan)
    for k in range(n_classes):
        block = traj_std[k]
        valid = np.isfinite(block).all(axis=1)
        if valid.sum() < 2:
            continue
        centred = focal[valid] - focal[valid].mean()
        denom = float((centred**2).sum())
        if denom <= 0.0:
            continue
        out[k] = (centred @ block[valid]) / denom
    return out


def net_directions(traj_std: np.ndarray) -> np.ndarray:
    r"""Return each class's unit net direction, the direction of its mean displacement.

    The net direction $\hat u_k$ is the unit vector of the mean standardised displacement across
    the focal grid, $\overline{D_k} / \lVert \overline{D_k} \rVert$. It is the axis the signed
    directional statistic projects onto. A class whose mean displacement is negligible (a
    symmetric excursion that cancels, or no drift) has an ill-defined direction and is given the
    zero vector, so its projected slope is zero rather than a projection onto noise.

    Parameters
    ----------
    traj_std : numpy.ndarray
        The standardised displacement trajectory, shape ``(n_classes, n_focal, n_features)``.

    Returns
    -------
    numpy.ndarray
        The per-class unit net direction, shape ``(n_classes, n_features)``.
    """
    mean_disp = np.nan_to_num(np.nanmean(traj_std, axis=1))
    norms = np.linalg.norm(mean_disp, axis=1, keepdims=True)
    safe = np.where(norms > _WEIGHT_FLOOR, norms, 1.0)
    unit = mean_disp / safe
    unit[norms[:, 0] <= _WEIGHT_FLOOR] = 0.0
    return unit


def project_onto(traj_std: np.ndarray, directions: np.ndarray) -> np.ndarray:
    r"""Return each class's one-dimensional signed trajectory along its net direction.

    The projection $s_k(f) = \langle D_k(f), \hat u_k \rangle$ of the standardised displacement
    onto the class's frozen net direction, a signed scalar per focal point. This is the
    one-dimensional signed trajectory the directional figure draws and the changepoint read
    localises a break on; positive values sit on the net-drift side of the pooled centroid.

    Parameters
    ----------
    traj_std : numpy.ndarray
        The standardised displacement trajectory, shape ``(n_classes, n_focal, n_features)``.
    directions : numpy.ndarray
        The per-class unit net direction, shape ``(n_classes, n_features)``.

    Returns
    -------
    numpy.ndarray
        The signed trajectory $s_k(f)$, shape ``(n_classes, n_focal)``.
    """
    return np.nansum(traj_std * directions[:, None, :], axis=2)


def _segment_sse(positions: np.ndarray, values: np.ndarray) -> float:
    """Return the residual sum of squares of a least-squares line through a segment."""
    if positions.shape[0] < 2:
        return 0.0
    slope, intercept = np.polyfit(positions, values, 1)
    residual = values - (slope * positions + intercept)
    return float(residual @ residual)


def single_break(positions: np.ndarray, series: np.ndarray, *, min_segment: int = 3) -> float:
    r"""Return the single-break location of a one-dimensional signed trajectory.

    A descriptive changepoint read: the axis position that best splits the signed trajectory into
    two independent least-squares segments, minimising the combined residual sum of squares. The
    break is reported at the midpoint of the two focal points it falls between. It is deliberately
    two independent lines (a discontinuity is allowed), so a level shift such as a DSM-5 (2013)
    boundary on the era axis is localised, not smoothed over. It is labelled descriptive: the
    bridge supremum-LM confidence set saturates at the full sample size, so the break location is
    read with its bootstrap spread rather than a resolved confidence set.

    Parameters
    ----------
    positions : numpy.ndarray
        The focal positions, shape ``(n_focal,)``.
    series : numpy.ndarray
        The one-dimensional signed trajectory, shape ``(n_focal,)``.
    min_segment : int, optional
        The fewest focal points each segment must hold.

    Returns
    -------
    float
        The break location in axis units, or not-a-number when the series is too short or flat.
    """
    positions = np.asarray(positions, dtype=float)
    series = np.asarray(series, dtype=float)
    finite = np.isfinite(positions) & np.isfinite(series)
    positions = positions[finite]
    series = series[finite]
    n = positions.shape[0]
    if n < 2 * min_segment:
        return float("nan")
    best_sse = np.inf
    best_at = float("nan")
    for i in range(min_segment - 1, n - min_segment):
        sse = _segment_sse(positions[: i + 1], series[: i + 1]) + _segment_sse(
            positions[i + 1 :], series[i + 1 :]
        )
        if sse < best_sse:
            best_sse = sse
            best_at = 0.5 * (positions[i] + positions[i + 1])
    return best_at


@dataclass
class DirectionalResult:
    r"""The observed per-class directional statistic of a displacement trajectory.

    Attributes
    ----------
    slope : numpy.ndarray
        The per-feature slope vector $b_k$, shape ``(n_classes, n_features)``.
    net_direction : numpy.ndarray
        The per-class unit net direction $\\hat u_k$, shape ``(n_classes, n_features)``.
    signed_slope : numpy.ndarray
        The signed net-projected slope $\\langle b_k, \\hat u_k\\rangle$, in standardised
        displacement per axis unit, shape ``(n_classes,)``. The directional statistic.
    net_trend : numpy.ndarray
        The separation-scaled net-trend displacement, the signed slope times the focal span over
        the between-class separation, shape ``(n_classes,)``; the interpretable effect size (how
        far, in separation units, the linear trend carries the class across the axis).
    signed_trajectory : numpy.ndarray
        The one-dimensional signed trajectory $s_k(f)$, shape ``(n_classes, n_focal)``.
    slope_norm : numpy.ndarray
        The Euclidean norm of the slope vector, shape ``(n_classes,)``; reported for context and
        known to be positively biased, so not the test statistic.
    break_position : numpy.ndarray
        The single-break location on each signed trajectory, shape ``(n_classes,)``.
    span : float
        The focal span (maximum minus minimum focal position), in axis units.
    focal_points : numpy.ndarray
        The focal positions, shape ``(n_focal,)``.
    """

    slope: np.ndarray
    net_direction: np.ndarray
    signed_slope: np.ndarray
    net_trend: np.ndarray
    signed_trajectory: np.ndarray
    slope_norm: np.ndarray
    break_position: np.ndarray
    span: float
    focal_points: np.ndarray


def directional_statistic(
    displacement: np.ndarray,
    pooled_sd: np.ndarray,
    focal_points: np.ndarray,
    separation_scale: float,
) -> DirectionalResult:
    r"""Compute the observed per-class directional statistic of a displacement trajectory.

    Standardises the displacement, fits the per-feature slope against the axis, projects it onto
    the class's net direction to get the signed directional statistic, and scales it to a
    separation-unit net-trend effect size. Also returns the one-dimensional signed trajectory, the
    (biased) slope norm, and the single-break changepoint location. Pure and cheap: it consumes
    the trajectory :func:`observed_trajectory` already computed and refits nothing.

    Parameters
    ----------
    displacement : numpy.ndarray
        The per-feature displacement $d_k(f)$, shape ``(n_classes, n_focal, n_features)``
        (:attr:`ObservedTrajectory.displacement`).
    pooled_sd : numpy.ndarray
        The per-feature pooled standard deviation, shape ``(n_features,)``.
    focal_points : numpy.ndarray
        The focal positions, shape ``(n_focal,)``.
    separation_scale : float
        The between-class separation the net trend is divided by.

    Returns
    -------
    DirectionalResult
        The observed directional statistic and its parts.
    """
    focal = np.asarray(focal_points, dtype=float)
    traj_std = displacement / pooled_sd
    slope = slope_vectors(traj_std, focal)
    unit = net_directions(traj_std)
    signed = np.nansum(slope * unit, axis=1)
    span = float(focal.max() - focal.min()) if focal.size else 0.0
    net_trend = signed * span / separation_scale
    signed_traj = project_onto(traj_std, unit)
    slope_norm = np.sqrt(np.nansum(slope**2, axis=1))
    breaks = np.array([single_break(focal, signed_traj[k]) for k in range(traj_std.shape[0])])
    return DirectionalResult(
        slope=slope,
        net_direction=unit,
        signed_slope=signed,
        net_trend=net_trend,
        signed_trajectory=signed_traj,
        slope_norm=slope_norm,
        break_position=breaks,
        span=span,
        focal_points=focal,
    )


@dataclass
class DirectionalInference:
    """The per-class directional test: effect size, clustered-bootstrap interval, and FDR.

    Attributes
    ----------
    net_trend : numpy.ndarray
        The observed separation-scaled net-trend displacement per class, shape ``(n_classes,)``.
    net_trend_lo, net_trend_hi : numpy.ndarray
        The clustered-bootstrap interval of the net trend, shape ``(n_classes,)``.
    signed_slope : numpy.ndarray
        The observed signed net-projected slope per class, shape ``(n_classes,)``.
    signed_slope_lo, signed_slope_hi : numpy.ndarray
        The clustered-bootstrap interval of the signed slope, shape ``(n_classes,)``.
    p_value : numpy.ndarray
        The two-sided bootstrap $p$-value that the signed slope differs from zero, shape
        ``(n_classes,)``.
    reject : numpy.ndarray
        The Benjamini-Hochberg decision across the classes at level ``q``, shape ``(n_classes,)``;
        a rejected class is directional along this axis.
    break_position : numpy.ndarray
        The observed single-break location per class, shape ``(n_classes,)``.
    break_lo, break_hi : numpy.ndarray
        The bootstrap spread of the break location, shape ``(n_classes,)``; descriptive.
    """

    net_trend: np.ndarray
    net_trend_lo: np.ndarray
    net_trend_hi: np.ndarray
    signed_slope: np.ndarray
    signed_slope_lo: np.ndarray
    signed_slope_hi: np.ndarray
    p_value: np.ndarray
    reject: np.ndarray
    break_position: np.ndarray
    break_lo: np.ndarray
    break_hi: np.ndarray


def directional_inference(
    observed: DirectionalResult, tube: BootstrapTube, *, q: float = 0.05
) -> DirectionalInference:
    r"""Test each class's directional statistic against the clustered-bootstrap null.

    The signed net-projected slope is a signed scalar, so its clustered-bootstrap distribution
    (frozen net direction, families resampled) gives a two-sided add-one $p$-value that it differs
    from zero, floored at one over the replicate count plus one. Benjamini-Hochberg control is
    applied across the classes at level ``q``; a rejected class is directional along the axis. The
    net-trend and
    signed-slope intervals are the bootstrap percentiles, and the break location carries its own
    bootstrap spread. The clustered bootstrap, not the bare slope, calls significance, because the
    slope norm is positively biased.

    Parameters
    ----------
    observed : DirectionalResult
        The observed directional statistic (:func:`directional_statistic`).
    tube : BootstrapTube
        The clustered-bootstrap tube built with the frozen net directions, carrying the
        directional draws.
    q : float, optional
        The false-discovery-rate level across the classes.

    Returns
    -------
    DirectionalInference
        The per-class effect size, interval, $p$-value, FDR decision, and break spread.
    """
    if tube.signed_slope is None or tube.net_trend is None:
        raise ValueError(
            "the tube carries no directional draws; pass net_directions when building it"
        )
    signed_draws = tube.signed_slope
    n_boot = signed_draws.shape[0]
    frac_positive = np.mean(signed_draws > 0.0, axis=0)
    tail = np.minimum(frac_positive, 1.0 - frac_positive)
    p_value = np.clip(2.0 * tail, 1.0 / (n_boot + 1), 1.0)
    reject = invariance.benjamini_hochberg(p_value, q)
    break_draws = tube.break_position
    if break_draws is not None:
        break_lo = np.nanpercentile(break_draws, 2.5, axis=0)
        break_hi = np.nanpercentile(break_draws, 97.5, axis=0)
    else:
        break_lo = np.full_like(observed.break_position, np.nan)
        break_hi = np.full_like(observed.break_position, np.nan)
    return DirectionalInference(
        net_trend=observed.net_trend,
        net_trend_lo=np.nanpercentile(tube.net_trend, 2.5, axis=0),
        net_trend_hi=np.nanpercentile(tube.net_trend, 97.5, axis=0),
        signed_slope=observed.signed_slope,
        signed_slope_lo=np.nanpercentile(signed_draws, 2.5, axis=0),
        signed_slope_hi=np.nanpercentile(signed_draws, 97.5, axis=0),
        p_value=p_value,
        reject=reject,
        break_position=observed.break_position,
        break_lo=break_lo,
        break_hi=break_hi,
    )
