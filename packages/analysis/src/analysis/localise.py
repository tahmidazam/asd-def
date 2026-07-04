r"""Localisation schemes: how a proband is weighted into a set of local fits.

The stratified analysis (plan section 7) re-estimates the mixture model as a function of a
continuous axis (age at diagnosis, or the derived calendar year of diagnosis). A *local fit*
is one re-estimation, weighted toward one region of that axis. A :class:`LocalisationScheme`
turns the axis variable into an ordered set of such local fits.

The generalisation this module adds is that a local fit is a *weight vector over the whole
cohort*, not a subset. Two schemes sit under one interface:

- :class:`HardBins` wraps any :class:`~analysis.strata.BinningPolicy` and emits one local fit
  per stratum with an indicator weight (one inside the bin, zero outside). Fitting on that
  weight is identical to fitting on the bin's subset, so the frozen ``MaxEqualBins`` primary
  reproduces the current stratified fits exactly.
- :class:`KernelWindows` emits one local fit per focal point on a grid, with a smooth kernel
  weight $w_i = \exp(-(a_i - a_0)^2 / 2h^2)$ that falls off with distance from the focal age
  $a_0$. Every proband contributes to every nearby fit, weighted by relevance, so no proband
  is wasted at a boundary. This is the local-likelihood (LSEM) estimator: the class profiles
  come out as smooth trajectories along the axis rather than a handful of disjoint points.

Downstream (:mod:`analysis.drift`) consumes the fits through the method-independent
:class:`~analysis.drift.StratumSummary`, so alignment, distance, and the permutation null are
unchanged by which scheme produced a fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from analysis.cohort import CohortMatrix
from analysis.features import Typing
from analysis.model import FitResult, fit_gfmm
from analysis.strata import BinningPolicy

# Weight below which a proband is dropped from a local fit rather than passed to StepMix with a
# negligible weight. Dropping keeps the fit cheap (the excluded rows carry no information) and
# makes a hard-bin fit an exact subset fit. A Gaussian kernel is about 3.7e-6 at four
# bandwidths, so this floor excludes probands beyond roughly four bandwidths from the focal age.
DEFAULT_WEIGHT_FLOOR = 1e-4


@dataclass(frozen=True)
class LocalFit:
    """One local re-estimation: where it sits on the axis and how each proband is weighted.

    Attributes
    ----------
    label : str
        The fit's name, unique within a scheme (a stratum name, or ``focal=6.5``).
    position : float
        The fit's location on the axis, in the axis units. For a hard bin this is the
        within-bin median; for a kernel window it is the focal point. It is the abscissa the
        drift trajectory is plotted and the continuous trend is regressed against.
    weights : pandas.Series
        Per-proband weight over the whole cohort index, in ``[0, 1]``. Zero excludes a
        proband from this fit; one includes it in full.
    """

    label: str
    position: float
    weights: pd.Series


@runtime_checkable
class LocalisationScheme(Protocol):
    """Turn a continuous axis variable into an ordered set of weighted local fits.

    Downstream code is typed against this protocol, never a concrete scheme, so the analysis
    is independent of whether the fits are hard bins or kernel windows.
    """

    name: str

    def locales(self, values: pd.Series) -> list[LocalFit]:
        """Return the ordered local fits for ``values`` (the per-proband axis variable)."""
        ...

    def spec(self) -> dict[str, object]:
        """Return the serialisable scheme specification for the run manifest."""
        ...


@dataclass(frozen=True)
class HardBins:
    """Adapt a :class:`~analysis.strata.BinningPolicy` to a localisation scheme.

    Emits one local fit per stratum with an indicator weight: one for probands in the bin,
    zero for the rest. Fitting on that weight is the same as fitting on the bin's subset, so
    this reproduces the current stratified analysis while presenting it through the unified
    interface. The frozen primary is ``HardBins(MaxEqualBins(1000))``.

    Attributes
    ----------
    policy : analysis.strata.BinningPolicy
        The binning policy that defines the strata.
    name : str
        Scheme name recorded in the spec, derived from the policy's name.
    """

    policy: BinningPolicy
    name: str = field(default="hardbins", init=False)

    def locales(self, values: pd.Series) -> list[LocalFit]:
        """One indicator-weighted local fit per stratum, positioned at the within-bin median."""
        assignment = self.policy.assign(values)
        codes = assignment.codes.reindex(values.index)
        fits: list[LocalFit] = []
        for label in assignment.labels:
            member = (codes == label).to_numpy()
            weights = pd.Series(member.astype(float), index=values.index, name=label)
            position = float(values[member].median()) if member.any() else float("nan")
            fits.append(LocalFit(label=str(label), position=position, weights=weights))
        return fits

    def spec(self) -> dict[str, object]:
        """Return the specification (the scheme name and the wrapped policy's spec)."""
        return {"scheme": self.name, "policy": self.policy.spec()}


def gaussian_weights(values: pd.Series, focal: float, bandwidth: float) -> pd.Series:
    r"""Return the Gaussian kernel weight of each value about a focal point.

    The weight is $\\exp(-(a_i - a_0)^2 / 2h^2)$ for value $a_i$, focal point $a_0$, and
    bandwidth $h$, so a value at the focal point has weight one and the weight falls off with
    distance. Missing values get weight zero.
    """
    z = (values.to_numpy(dtype=float) - focal) / bandwidth
    weights = np.exp(-0.5 * z**2)
    weights[~np.isfinite(values.to_numpy(dtype=float))] = 0.0
    return pd.Series(weights, index=values.index)


def focal_grid(values: pd.Series, n_points: int, quantile_span: tuple[float, float]) -> list[float]:
    """Return ``n_points`` evenly spaced focal points spanning an inner quantile range.

    The grid runs between the ``quantile_span`` quantiles of the finite values rather than the
    full range, so the outermost focal points still sit on a populated part of the axis and
    the edge fits are not estimated from almost nothing.
    """
    finite = values.dropna()
    lo, hi = np.quantile(finite, quantile_span)
    return [float(x) for x in np.linspace(lo, hi, n_points)]


@dataclass(frozen=True)
class KernelWindows:
    """A local-likelihood (LSEM) scheme: one kernel-weighted fit per focal point.

    Attributes
    ----------
    bandwidth : float
        The Gaussian kernel bandwidth $h$, in axis units, the smoothing knob. A larger
        bandwidth pulls in more probands per fit (smoother, more biased toward the pooled
        solution); a smaller one localises more sharply (less biased, noisier).
    grid : tuple of float or int
        The focal points, in axis units, or an integer count of evenly spaced points to build
        with :func:`focal_grid`.
    kernel : str
        The kernel name; only ``gaussian`` is defined.
    quantile_span : tuple of float, optional
        The inner quantile range the integer grid spans, ignored when ``grid`` is explicit.
    name : str
        Scheme name recorded in the spec.
    """

    bandwidth: float
    grid: tuple[float, ...] | int
    kernel: str = "gaussian"
    quantile_span: tuple[float, float] = (0.025, 0.975)
    name: str = field(default="kernel", init=False)

    def __post_init__(self) -> None:
        """Validate the bandwidth, the grid, and the kernel."""
        if self.bandwidth <= 0:
            raise ValueError("KernelWindows needs bandwidth > 0.")
        if isinstance(self.grid, int) and self.grid < 2:
            raise ValueError("KernelWindows needs an integer grid of at least 2 points.")
        if self.kernel != "gaussian":
            raise ValueError(f"unknown kernel {self.kernel!r}; only 'gaussian' is defined.")

    def _focal_points(self, values: pd.Series) -> list[float]:
        if isinstance(self.grid, int):
            return focal_grid(values, self.grid, self.quantile_span)
        return [float(x) for x in self.grid]

    def locales(self, values: pd.Series) -> list[LocalFit]:
        """One kernel-weighted local fit per focal point."""
        fits: list[LocalFit] = []
        for focal in self._focal_points(values):
            weights = gaussian_weights(values, focal, self.bandwidth)
            fits.append(LocalFit(label=f"focal={focal:g}", position=focal, weights=weights))
        return fits

    def spec(self) -> dict[str, object]:
        """Return the serialisable specification (bandwidth, kernel, and the focal grid)."""
        grid = self.grid if isinstance(self.grid, int) else [float(x) for x in self.grid]
        return {
            "scheme": self.name,
            "bandwidth": float(self.bandwidth),
            "kernel": self.kernel,
            "grid": grid,
            "quantile_span": list(self.quantile_span),
        }


def effective_sample_sizes(
    values: pd.Series, focal_points: list[float], bandwidth: float
) -> np.ndarray:
    r"""Return the effective sample size of a kernel fit at each focal point.

    The effective sample size at a focal point $a_0$ is the sum of the Gaussian weights,
    $\sum_i \exp(-(a_i - a_0)^2 / 2h^2)$, the count of whole probands the weighted fit is worth
    there. It is what sets a local fit's power, so it is the natural quantity to fix a bandwidth
    against. Missing values contribute nothing.
    """
    finite = values.dropna().to_numpy(dtype=float)
    return np.array(
        [float(np.exp(-0.5 * ((finite - focal) / bandwidth) ** 2).sum()) for focal in focal_points]
    )


def bandwidth_for_effective_n(
    values: pd.Series,
    focal_points: list[float],
    target_n: float,
    *,
    reduce: str = "min",
    tol: float = 1e-3,
    max_iter: int = 60,
) -> float:
    """Return the smallest bandwidth whose focal fits reach a target effective sample size.

    The effective sample size at a focal point rises with the bandwidth, so a target on it maps
    to a bandwidth by bisection. ``reduce`` sets which focal point the target must hold at:
    ``"min"`` fixes the thinnest focal point at the target, so every focal fit clears it (the
    same guarantee the hard-bin floor gives every bin); ``"median"`` fixes the typical focal
    point, which lets the edges fall below the target. The bandwidth is in the axis units.

    Parameters
    ----------
    values : pandas.Series
        The axis variable (age at diagnosis in years, or diagnosis year).
    focal_points : list of float
        The focal grid the bandwidth is chosen for, as :func:`focal_grid` builds it.
    target_n : float
        The effective sample size each focal fit should reach (for example the recovery floor).
    reduce : str, optional
        ``"min"`` (default) or ``"median"``: which focal point the target is held at.
    tol, max_iter : float and int, optional
        Bisection tolerance (in bandwidth units) and iteration cap.

    Returns
    -------
    float
        The smallest bandwidth meeting the target.
    """
    reducer = {"min": np.min, "median": np.median}[reduce]
    finite = values.dropna()
    span = float(finite.max() - finite.min()) or 1.0
    lo, hi = 1e-6, span
    while reducer(effective_sample_sizes(finite, focal_points, hi)) < target_n and hi < 1e6:
        hi *= 2.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if reducer(effective_sample_sizes(finite, focal_points, mid)) >= target_n:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


def permute_axis(values: pd.Series, seed: int) -> pd.Series:
    """Return the axis values shuffled across probands, the permutation null.

    Breaking the pairing between a proband and its axis value, then re-running the same scheme
    on the shuffled values, removes only the association with the axis while preserving the
    fit sizes and the weight profile. For :class:`HardBins` this gives same-size random bins
    (as :func:`analysis.drift.null_partition` does); for :class:`KernelWindows` it gives kernel
    windows over randomly relabelled probands. One null definition serves both schemes. The
    seed is the permutation index, so a resumed null reproduces the same shuffles.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(values.to_numpy())
    return pd.Series(shuffled, index=values.index, name=values.name)


def fit_locale(
    matrix: CohortMatrix,
    typing: Typing,
    locale: LocalFit,
    *,
    n_init: int,
    random_state: int,
    weight_floor: float = DEFAULT_WEIGHT_FLOOR,
    progress_bar: int = 0,
    verbose: int = 0,
) -> FitResult:
    """Fit the GFMM for one local fit, dropping negligible-weight probands.

    Probands whose weight is at or below ``weight_floor`` carry no information and are dropped
    before fitting, so a hard-bin fit is an exact subset fit and a kernel fit skips the far
    tail. The remaining probands are passed to StepMix with their weights, so the fit is the
    weighted maximum-likelihood solution local to this fit's region of the axis.

    Parameters
    ----------
    matrix : analysis.cohort.CohortMatrix
        The full cohort matrix; the local fit selects and weights its rows.
    typing : analysis.features.Typing
        The reconciled feature typing.
    locale : LocalFit
        The local fit's per-proband weights and label.
    n_init : int
        StepMix restarts.
    random_state : int
        Seed for reproducible restarts.
    weight_floor : float, optional
        Probands at or below this weight are excluded from the fit.
    progress_bar, verbose : int, optional
        StepMix verbosity, off by default so a sweep of many fits stays quiet.

    Returns
    -------
    analysis.model.FitResult
        The local fit, its labels on the retained probands, and its metrics.
    """
    weights = locale.weights.reindex(matrix.features.index).fillna(0.0)
    keep = weights.to_numpy() > weight_floor
    sub = CohortMatrix(
        matrix.features.loc[keep], matrix.covariates.loc[keep], matrix.dataset, matrix.version
    )
    return fit_gfmm(
        sub,
        typing,
        n_init=n_init,
        random_state=random_state,
        sample_weight=weights.to_numpy()[keep],
        progress_bar=progress_bar,
        verbose=verbose,
    )
