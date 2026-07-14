r"""Support for the number of latent classes, per stratum, by a bootstrap likelihood-ratio search.

The H0C hypothesis (plan section 7, H0C) asks whether the *number* of
components the Litman four-class general finite mixture model supports is stable across strata
of age at diagnosis and diagnostic era, or whether a class splits or merges in some stratum.
This module answers that per stratum, and relative to the pooled cohort put through the
identical procedure, so the shared over-extraction from feature misspecification (which drove
the pooled information criteria to nine classes at this sample size) cancels: an order change
is a stratum whose supported order differs from the pooled cohort's, not a stratum whose raw
information criteria differ from four.

The measurement model is fitted measurement-only (``structural=None``, the
``fit --no-covariates`` recipe), on hard subsets of the 238-feature cohort matrix. The
confirmatory statistic is a warm-started parametric bootstrap likelihood-ratio test (BLRT):

- the observed statistic for a step is $\text{LR} = 2(\ell_{K+1} - \ell_K)$, the twice
  log-likelihood gain from one extra class, where the $K$-class fit uses a handful of random
  restarts and the $K+1$-class fit is warm-started by splitting each of the $K$ classes in
  turn (so $K$ warm starts) plus a couple of random restarts, keeping the best;
- the null is parametric: datasets are simulated from the fitted $K$-component model at the
  stratum's own sample size, and each is put through the *identical* fitting recipe, so the
  null LR is not biased low by under-fitting the alternative. The $p$-value is the
  Phipson-Smyth add-one permutation $p$ (as in :mod:`analysis.drift`).

The search is sequential and anchored at four classes (:func:`sequential_search`): the primary
(splitting) direction tests $4$ against $5$ and steps outward while each step rejects, up to a
cap; the secondary (merging) direction tests $4$ against $3$ and steps down while each step
fails to reject, down to a floor. Two corroborators, neither of which decides, sit beside the
BLRT: the cross-validated log-likelihood elbow, with the knee found by an inline Kneedle
(:func:`kneedle_knee`), and the proper adjusted Lo-Mendell-Rubin test (:func:`vlmr_test`,
Formula 15 of Lo, Mendell and Rubin 2001), not the naive one-degree-of-freedom proxy of
:func:`analysis.selection.lmr_lrt_proxy`.

The fits reuse the StepMix internals the rest of the package already depends on
(:func:`analysis.model.prepare_inputs`, ``set_parameters``/``get_parameters``, the measurement
emission ``sample`` and ``log_likelihood``), and the degeneracy handling of
:mod:`analysis.drift` (its ``DEGENERATE_FIT_ERRORS``). :func:`bootstrap_lr` is
a top-level, picklable unit of work (simulate one null dataset, refit, return its LR), so the
bootstrap draws spread across a process pool.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import chi2
from stepmix.stepmix import StepMix

from analysis import config, selection
from analysis.cohort import CohortMatrix
from analysis.drift import DEGENERATE_FIT_ERRORS
from analysis.features import Typing
from analysis.model import prepare_inputs

# A draw dispatcher: given the null number of classes, the parametric null parameters, the
# sample size, and a half-open block of draw indices, return that block's null LR statistics
# (``None`` for a degenerate draw). The CLI supplies one backed by a process pool and a
# resumable checkpoint; a test supplies an in-process one. Keeping it a callable is what lets
# :func:`blrt_step` stay free of the pool and checkpoint machinery.
DrawDispatcher = Callable[[int, dict, int, int, int], list[float | None]]

# StepMix constructor options shared by every measurement-only fit in this module. The
# structural key is omitted so ``fit(X, Y=None)`` runs the measurement model alone, matching
# ``fit_gfmm(structural=None)`` (the reference release's ``fit --no-covariates`` recipe).
_STEPMIX_BASE: dict[str, object] = {
    "n_steps": 1,
    "progress_bar": 0,
    "verbose": 0,
    "max_iter": config.DEFAULT_ORDER_MAX_ITER,
}


def measurement_inputs(matrix: CohortMatrix, typing: Typing) -> tuple[pd.DataFrame, dict]:
    """Return the descriptor-aligned measurement matrix and the mixed-data descriptor.

    A thin wrapper over :func:`analysis.model.prepare_inputs` that drops the covariate channel,
    since the H0C fits are measurement-only. The returned frame is rounded as the reference
    release rounds, so the model sees the same values the covariate reference was fitted on.

    Parameters
    ----------
    matrix : analysis.cohort.CohortMatrix
        The cohort feature and covariate matrices.
    typing : analysis.features.Typing
        The reconciled feature typing that sets each feature's density.

    Returns
    -------
    tuple
        The measurement matrix and the StepMix mixed-data descriptor.
    """
    measurement_data, descriptor, _covariates = prepare_inputs(matrix, typing)
    return measurement_data, descriptor


def total_log_likelihood(model: StepMix, x: np.ndarray) -> float:
    r"""Return the total measurement log-likelihood of ``x`` under a fitted model.

    The per-sample log-likelihood of a mixture is
    $\log \sum_k \pi_k \prod_j f_j(x_{ij}; \theta_{jk})$, read off the emission
    log-likelihood (``model._mm.log_likelihood``) and the mixing weights (``model.weights_``),
    the same primitives :mod:`analysis.invariance` uses. Summed over samples, this is the
    quantity the likelihood-ratio statistic differences.

    Parameters
    ----------
    model : StepMix
        A fitted measurement-only model.
    x : numpy.ndarray
        The measurement matrix in the descriptor column order the model was fitted on.

    Returns
    -------
    float
        The total log-likelihood, summed over samples.
    """
    log_joint = model._mm.log_likelihood(x) + np.log(model.weights_)[None, :]
    return float(logsumexp(log_joint, axis=1).sum())


def fit_k(
    measurement_data: pd.DataFrame,
    descriptor: dict,
    k: int,
    *,
    n_init: int,
    seed: int,
) -> StepMix:
    """Fit a measurement-only ``k``-class GFMM with random restarts.

    Parameters
    ----------
    measurement_data : pandas.DataFrame
        The descriptor-aligned measurement matrix.
    descriptor : dict
        The StepMix mixed-data measurement descriptor.
    k : int
        The number of latent classes.
    n_init : int
        Random restarts, delegated to StepMix; the best restart is kept.
    seed : int
        Random seed for reproducible restarts.

    Returns
    -------
    StepMix
        The fitted estimator.
    """
    model = StepMix(
        n_components=k, measurement=descriptor, n_init=n_init, random_state=seed, **_STEPMIX_BASE
    )
    model.fit(measurement_data)
    return model


def split_warm_params(
    model: StepMix, k: int, class_id: int, rng: np.random.Generator, jitter: float
) -> dict:
    r"""Return warm $K+1$ parameters by splitting one class of a fitted $K$-class model.

    The split halves class ``class_id``'s mixing weight into two children and perturbs its
    class-conditional parameters symmetrically by $\pm$ a Gaussian jitter, giving a starting
    point for the extra class near the class most likely to divide. Applied to each of the $K$
    classes in turn, this is the $K$-warm-start half of the alternative-model recipe (plan
    section 7): a good local optimum for $K+1$ that a purely random restart rarely reaches, so
    the observed and null alternative fits are on an equal footing.

    The measurement parameters are a nested dict of per-emission arrays; the class axis is the
    one of length ``k``. A block with no such axis (a shared parameter) is carried over
    unchanged.

    Parameters
    ----------
    model : StepMix
        The fitted $K$-class model to split.
    k : int
        Its number of classes.
    class_id : int
        The class to split into two children.
    rng : numpy.random.Generator
        The generator for the symmetric jitter.
    jitter : float
        Standard deviation of the Gaussian perturbation applied to the split class.

    Returns
    -------
    dict
        A parameter dict in ``get_parameters`` format with $K+1$ classes, for
        :func:`warm_em`.
    """
    params = model.get_parameters()
    weights = np.asarray(params["weights"], dtype=float)
    new_weights = np.concatenate([weights, [weights[class_id] / 2.0]])
    new_weights[class_id] = weights[class_id] / 2.0
    new_weights = new_weights / new_weights.sum()

    new_measurement: dict[str, dict] = {}
    for name, block in params["measurement"].items():
        new_block: dict[str, object] = {}
        for key, value in block.items():
            array = np.asarray(value, dtype=float)
            class_axes = [axis for axis, size in enumerate(array.shape) if size == k]
            if not class_axes:
                new_block[key] = value
                continue
            axis = class_axes[0]
            row = np.take(array, class_id, axis=axis)
            perturbation = rng.normal(0.0, jitter, size=row.shape)
            extended = np.concatenate([array, np.expand_dims(row - perturbation, axis)], axis=axis)
            index: list[slice | int] = [slice(None)] * extended.ndim
            index[axis] = class_id
            extended[tuple(index)] = row + perturbation
            new_block[key] = extended
        new_measurement[name] = new_block
    return {
        "weights": new_weights,
        "measurement": new_measurement,
        "measurement_in": params["measurement_in"],
    }


def warm_em(
    descriptor: dict,
    k: int,
    warm_params: dict,
    x: np.ndarray,
    *,
    max_iter: int = config.DEFAULT_ORDER_MAX_ITER,
    tol: float = 1e-6,
) -> StepMix:
    """Run the EM algorithm from given warm parameters, without reinitialising.

    StepMix's own ``fit`` reinitialises the emission parameters at the start of every EM run,
    so a warm start cannot go through it. This drives the EM loop directly instead: it sets the
    given parameters, then alternates the model's own E-step and M-step until the average
    log-likelihood stops improving. The E-step and M-step are StepMix's, so the fit is the same
    optimiser the random restarts use, only seeded from the split rather than from noise.

    Parameters
    ----------
    descriptor : dict
        The StepMix mixed-data measurement descriptor.
    k : int
        The number of latent classes (one more than the model being split).
    warm_params : dict
        The starting parameters, in ``get_parameters`` format (from :func:`split_warm_params`).
    x : numpy.ndarray
        The measurement matrix in the descriptor column order.
    max_iter : int, optional
        Maximum EM iterations.
    tol : float, optional
        Absolute tolerance on the average log-likelihood for convergence.

    Returns
    -------
    StepMix
        The warm-started fitted model.
    """
    model = StepMix(
        n_components=k, measurement=descriptor, n_init=1, random_state=0, **_STEPMIX_BASE
    )
    model._check_initial_parameters(x)
    model.set_parameters(copy.deepcopy(warm_params))
    previous = -np.inf
    for _ in range(max_iter):
        avg_ll, log_resp = model._e_step(x)
        model._m_step(x, np.exp(log_resp))
        if abs(avg_ll - previous) < tol:
            break
        previous = avg_ll
    return model


def fit_k_plus_one(
    model_k: StepMix,
    measurement_data: pd.DataFrame,
    descriptor: dict,
    k: int,
    x: np.ndarray,
    *,
    n_random: int,
    n_init_random: int,
    seed: int,
    jitter: float,
) -> tuple[StepMix | None, float]:
    """Fit the $K+1$-class alternative by class-splitting warm starts plus random restarts.

    The alternative-model recipe (plan section 7), used identically for the observed data and
    every bootstrap sample: split each of the $K$ classes of ``model_k`` in turn (so $K$ warm
    starts, :func:`split_warm_params` then :func:`warm_em`), add ``n_random`` random restarts,
    and keep whichever reaches the highest total log-likelihood. A single warm start that fails
    numerically is skipped rather than fatal; if every start fails the caller reads the returned
    log-likelihood of minus infinity as a degenerate fit.

    Parameters
    ----------
    model_k : StepMix
        The fitted $K$-class model whose classes seed the warm starts.
    measurement_data : pandas.DataFrame
        The descriptor-aligned measurement matrix (for the random restarts).
    descriptor : dict
        The StepMix mixed-data measurement descriptor.
    k : int
        The number of classes of ``model_k``; the alternative has ``k + 1``.
    x : numpy.ndarray
        The measurement matrix as an array (for the log-likelihood and the warm EM).
    n_random : int
        Random restarts to add to the $K$ warm starts.
    n_init_random : int
        Restarts within each random-restart fit.
    seed : int
        Base seed; the warm starts and random restarts derive their seeds from it.
    jitter : float
        Split-perturbation standard deviation.

    Returns
    -------
    tuple
        The best $K+1$ fit (``None`` when none succeeded) and its total log-likelihood (minus
        infinity when none succeeded).
    """
    best_model: StepMix | None = None
    best_ll = -np.inf
    for class_id in range(k):
        try:
            warm = split_warm_params(
                model_k, k, class_id, np.random.default_rng(seed + class_id), jitter
            )
            candidate = warm_em(descriptor, k + 1, warm, x)
            candidate_ll = total_log_likelihood(candidate, x)
        except DEGENERATE_FIT_ERRORS:
            continue
        if np.isfinite(candidate_ll) and candidate_ll > best_ll:
            best_ll, best_model = candidate_ll, candidate
    for restart in range(n_random):
        try:
            candidate = fit_k(
                measurement_data, descriptor, k + 1, n_init=n_init_random, seed=seed + 100 + restart
            )
            candidate_ll = total_log_likelihood(candidate, x)
        except DEGENERATE_FIT_ERRORS:
            continue
        if np.isfinite(candidate_ll) and candidate_ll > best_ll:
            best_ll, best_model = candidate_ll, candidate
    return best_model, best_ll


@dataclass
class FitPair:
    """A fitted $K$-class model and its warm-started $K+1$-class alternative.

    Attributes
    ----------
    model_k : StepMix
        The $K$-class (null) fit.
    model_k1 : StepMix or None
        The best $K+1$-class (alternative) fit, or ``None`` when every start was degenerate.
    ll_k : float
        Total log-likelihood of the $K$-class fit.
    ll_k1 : float
        Total log-likelihood of the $K+1$-class fit (minus infinity when degenerate).
    k : int
        The null number of classes.
    """

    model_k: StepMix
    model_k1: StepMix | None
    ll_k: float
    ll_k1: float
    k: int

    @property
    def observed_lr(self) -> float:
        r"""Return the observed likelihood-ratio statistic $2(\ell_{K+1} - \ell_K)$."""
        return 2.0 * (self.ll_k1 - self.ll_k)


def fit_pair(
    measurement_data: pd.DataFrame,
    descriptor: dict,
    k: int,
    *,
    n_init: int,
    n_random: int,
    seed: int,
    jitter: float,
) -> FitPair:
    """Fit the $K$-class null and the warm-started $K+1$-class alternative on one dataset.

    The observed side of one BLRT step. Applied unchanged to the real stratum and (through
    :func:`bootstrap_lr`) to every simulated null dataset, so the recipe is identical on both
    sides.

    Parameters
    ----------
    measurement_data : pandas.DataFrame
        The descriptor-aligned measurement matrix.
    descriptor : dict
        The StepMix mixed-data measurement descriptor.
    k : int
        The null number of classes.
    n_init : int
        Random restarts for the $K$-class fit.
    n_random : int
        Random restarts added to the $K$ warm starts for the $K+1$-class fit.
    seed : int
        Base seed for the fits.
    jitter : float
        Split-perturbation standard deviation.

    Returns
    -------
    FitPair
        The two fits and their log-likelihoods.
    """
    x = measurement_data.to_numpy()
    model_k = fit_k(measurement_data, descriptor, k, n_init=n_init, seed=seed)
    ll_k = total_log_likelihood(model_k, x)
    model_k1, ll_k1 = fit_k_plus_one(
        model_k,
        measurement_data,
        descriptor,
        k,
        x,
        n_random=n_random,
        n_init_random=1,
        seed=seed,
        jitter=jitter,
    )
    return FitPair(model_k=model_k, model_k1=model_k1, ll_k=ll_k, ll_k1=ll_k1, k=k)


def simulate_null(
    model_k: StepMix | dict, descriptor: dict, columns: pd.Index, n: int, seed: int
) -> pd.DataFrame:
    """Simulate one parametric-bootstrap dataset from a fitted $K$-class model.

    Reconstructs a sampleable StepMix from the fitted model's parameters and draws ``n``
    samples from the mixture (the measurement emissions and the mixing weights), the parametric
    null of the BLRT (plan section 7). The seed sets the emission generator, so a draw is
    reproducible from its index.

    Parameters
    ----------
    model_k : StepMix or dict
        The fitted $K$-class model, or its ``get_parameters`` dict (for a worker process).
    descriptor : dict
        The StepMix mixed-data measurement descriptor.
    columns : pandas.Index
        The descriptor column order, so the sampled array is framed like the real matrix.
    n : int
        The stratum's sample size.
    seed : int
        Seed for the emission sampler.

    Returns
    -------
    pandas.DataFrame
        A simulated measurement matrix of ``n`` rows in ``columns`` order.
    """
    params = model_k.get_parameters() if isinstance(model_k, StepMix) else model_k
    k = len(params["weights"])
    sampler = StepMix(
        n_components=k, measurement=descriptor, n_init=1, random_state=seed, **_STEPMIX_BASE
    )
    sampler.set_parameters(copy.deepcopy(params))
    x_sample, _y, _labels = sampler.sample(n)
    return pd.DataFrame(x_sample, columns=columns)


def bootstrap_lr(
    null_params: dict,
    descriptor: dict,
    columns: list[str],
    n: int,
    k: int,
    *,
    n_init: int,
    n_random: int,
    seed: int,
    jitter: float,
) -> float | None:
    """Simulate one null dataset, refit both models, and return the null LR statistic.

    A top-level, picklable unit of work (mirroring
    :func:`analysis.drift.summarise_pseudo_stratum`), so the bootstrap draws spread across a
    process pool. The worker samples its own dataset from ``null_params`` (small, passed once
    per pool), then puts it through the identical fitting recipe as the observed data. Returns
    ``None`` when a fit is numerically degenerate, so the caller drops that draw and counts it
    rather than letting one bad refit abort the run.

    Parameters
    ----------
    null_params : dict
        The fitted $K$-class model's ``get_parameters`` dict, the parametric null to sample from.
    descriptor : dict
        The StepMix mixed-data measurement descriptor.
    columns : list of str
        The descriptor column order.
    n : int
        The stratum's sample size to simulate.
    k : int
        The null number of classes.
    n_init : int
        Random restarts for the $K$-class refit.
    n_random : int
        Random restarts added to the warm starts for the $K+1$-class refit.
    seed : int
        Seed for this draw (sampling and fitting).
    jitter : float
        Split-perturbation standard deviation.

    Returns
    -------
    float or None
        The null likelihood-ratio statistic, or ``None`` if the draw was degenerate.
    """
    try:
        columns_index = pd.Index(columns)
        simulated = simulate_null(null_params, descriptor, columns_index, n, seed)
        pair = fit_pair(
            simulated, descriptor, k, n_init=n_init, n_random=n_random, seed=seed, jitter=jitter
        )
    except DEGENERATE_FIT_ERRORS:
        return None
    if pair.model_k1 is None or not np.isfinite(pair.observed_lr):
        return None
    return pair.observed_lr


def phipson_smyth_p(observed: float, null_draws: list[float | None]) -> float:
    """Return the Phipson-Smyth add-one bootstrap $p$-value.

    The proportion of null draws at or beyond the observed statistic, with the add-one
    correction so the smallest attainable $p$ is $1 / (B + 1)$ rather than zero (as in
    :func:`analysis.drift.read_against_null`). Non-finite draws (degenerate refits) are dropped.

    Parameters
    ----------
    observed : float
        The observed likelihood-ratio statistic.
    null_draws : list of float
        The bootstrap null statistics.

    Returns
    -------
    float
        The add-one $p$-value, or ``nan`` when no finite draw survived.
    """
    draws = np.asarray([d for d in null_draws if d is not None and np.isfinite(d)], dtype=float)
    b = int(draws.size)
    if b == 0:
        return float("nan")
    exceedances = int(np.sum(draws >= observed))
    return (1 + exceedances) / (1 + b)


def kneedle_knee(k_values: list[int], scores: list[float]) -> int:
    r"""Return the knee of a concave, increasing curve by the Kneedle rule.

    The cross-validated log-likelihood rises with the number of classes and flattens; the knee
    is where the diminishing returns set in, the elbow the H0C corroborator reads. This is the
    Kneedle construction (Satopaa et al. 2011) for a concave increasing curve: normalise both the
    class count and the score to the unit interval, take the difference between the normalised
    score and the normalised class count, and return the class count at its maximum. Ties and a
    flat curve fall back to the first candidate. This is an inline implementation (no new
    dependency), an objective substitute for reading the elbow off a plot.

    Parameters
    ----------
    k_values : list of int
        The number of classes, ascending.
    scores : list of float
        The corresponding score (validation log-likelihood), one per ``k_values`` entry.

    Returns
    -------
    int
        The number of classes at the knee.
    """
    xs = np.asarray(k_values, dtype=float)
    ys = np.asarray(scores, dtype=float)
    finite = np.isfinite(ys)
    xs, ys = xs[finite], ys[finite]
    ks = [int(k) for k, keep in zip(k_values, finite, strict=True) if keep]
    if len(ks) < 2:
        return int(ks[0]) if ks else 0
    x_span = xs.max() - xs.min()
    y_span = ys.max() - ys.min()
    if x_span == 0 or y_span == 0:
        return ks[0]
    x_norm = (xs - xs.min()) / x_span
    y_norm = (ys - ys.min()) / y_span
    difference = y_norm - x_norm
    return ks[int(np.argmax(difference))]


def vlmr_test(
    null_ll: float,
    null_params: int,
    null_classes: int,
    alt_ll: float,
    alt_params: int,
    alt_classes: int,
    n: int,
) -> dict[str, float]:
    r"""Return the adjusted Lo-Mendell-Rubin likelihood-ratio test for $K$ against $K+1$.

    The proper VLMR of Lo, Mendell and Rubin (2001, Biometrika 88(3):767-778, Formula 15), not
    the naive one-degree-of-freedom proxy of :func:`analysis.selection.lmr_lrt_proxy`. The
    likelihood-ratio statistic is corrected by an ad-hoc factor and referred to a chi-square:

    .. math::

        \text{LR} = 2(\ell_1 - \ell_0), \qquad
        \text{LMR} = \frac{\text{LR}}{1 + \big[((3k_1 - 1) - (3k_0 - 1)) \ln n\big]^{-1}},

    with degrees of freedom the difference in free parameters $p_1 - p_0$ and
    $p = \Pr(\chi^2_{p_1 - p_0} > \text{LMR})$. Here $k_0, k_1$ are the null and alternative
    class counts and $\ell_0, \ell_1$ their log-likelihoods. Reproduces ``calc_lrt`` of the R
    package ``tidyLPA``.

    Parameters
    ----------
    null_ll, alt_ll : float
        Log-likelihoods of the null ($k_0$-class) and alternative ($k_1$-class) fits.
    null_params, alt_params : int
        Numbers of free parameters (``StepMix.n_parameters``).
    null_classes, alt_classes : int
        The class counts $k_0$ and $k_1$.
    n : int
        The sample size.

    Returns
    -------
    dict of str to float
        ``lr`` (the raw statistic), ``lmr`` (the corrected statistic), ``df``, and ``p``.
    """
    lr = 2.0 * (alt_ll - null_ll)
    proxy_gap = (3 * alt_classes - 1) - (3 * null_classes - 1)
    correction = 1.0 + 1.0 / (proxy_gap * np.log(n))
    lmr = lr / correction
    df = alt_params - null_params
    p = float(chi2.sf(lmr, df)) if df > 0 and np.isfinite(lmr) else float("nan")
    return {"lr": float(lr), "lmr": float(lmr), "df": float(df), "p": p}


@dataclass
class StepResult:
    """One BLRT step: a $k$-against-$k+1$ comparison with its staged bootstrap $p$-value.

    Attributes
    ----------
    k_null : int
        The null number of classes.
    k_alt : int
        The alternative number of classes (``k_null + 1``).
    direction : str
        ``"split"`` for an upward (add-a-class) step, ``"merge"`` for a downward one.
    observed_lr : float
        The observed likelihood-ratio statistic.
    p_value : float
        The Phipson-Smyth bootstrap $p$-value over all draws used.
    b_used : int
        The number of finite bootstrap draws behind ``p_value``.
    escalated : bool
        Whether the step escalated from the screen count to the full count.
    n_dropped : int
        Degenerate bootstrap draws dropped for this step.
    rejected : bool
        Whether the step rejected its null at the step level (``p_value <= alpha``).
    """

    k_null: int
    k_alt: int
    direction: str
    observed_lr: float
    p_value: float
    b_used: int
    escalated: bool
    n_dropped: int
    rejected: bool


@dataclass
class SearchResult:
    """The sequential search outcome for one dataset (a stratum or the pooled cohort).

    Attributes
    ----------
    supported_k : int
        The supported number of classes.
    capped : bool
        Whether the search hit the upper cap while still rejecting, so ``supported_k`` is a
        lower bound (reported as ``">=cap"``).
    direction : str
        ``"split"`` if the order rose above the anchor, ``"merge"`` if it fell below, or
        ``"stable"`` if it stayed at the anchor.
    steps : list of StepResult
        Every step taken, in order.
    elbow_knee : int
        The cross-validated log-likelihood knee (Kneedle), a corroborator.
    cv_log_likelihood : dict of int to float
        The validation log-likelihood per number of classes behind the knee.
    vlmr : dict of str to float
        The adjusted Lo-Mendell-Rubin test at the decisive comparison (empty when the search
        stayed at the anchor with no decisive step).
    n_dropped : int
        Total degenerate bootstrap draws dropped across the search.
    """

    supported_k: int
    capped: bool
    direction: str
    steps: list[StepResult] = field(default_factory=list)
    elbow_knee: int = 0
    cv_log_likelihood: dict[int, float] = field(default_factory=dict)
    vlmr: dict[str, float] = field(default_factory=dict)
    n_dropped: int = 0


@dataclass(frozen=True)
class Recipe:
    """The identical fitting recipe applied to the observed data and every null draw.

    Attributes
    ----------
    n_init : int
        Random restarts for a $K$-class fit.
    n_random : int
        Random restarts added to the $K$ class-splitting warm starts for a $K+1$-class fit.
    jitter : float
        Standard deviation of the split perturbation.
    """

    n_init: int = config.DEFAULT_ORDER_N_INIT
    n_random: int = config.DEFAULT_ORDER_N_RANDOM
    jitter: float = config.DEFAULT_ORDER_SPLIT_JITTER

    def spec(self) -> dict[str, object]:
        """Return the serialisable recipe, folded into the run hash."""
        return {"n_init": self.n_init, "n_random": self.n_random, "jitter": self.jitter}


@dataclass(frozen=True)
class Schedule:
    """The staged bootstrap schedule and the step-level rejection level.

    Attributes
    ----------
    b_screen : int
        Draws every step is screened at.
    b_escalate : int
        Draws a step escalates to when its screen $p$ falls below ``escalate_threshold``.
    escalate_threshold : float
        The screen-$p$ threshold that triggers escalation.
    alpha : float
        The step-level rejection level.
    """

    b_screen: int = config.DEFAULT_ORDER_B_SCREEN
    b_escalate: int = config.DEFAULT_ORDER_B_ESCALATE
    escalate_threshold: float = config.DEFAULT_ORDER_ESCALATE_THRESHOLD
    alpha: float = config.DEFAULT_ORDER_ALPHA

    def spec(self) -> dict[str, object]:
        """Return the serialisable schedule, folded into the run hash."""
        return {
            "b_screen": self.b_screen,
            "b_escalate": self.b_escalate,
            "escalate_threshold": self.escalate_threshold,
            "alpha": self.alpha,
        }


def blrt_step(
    measurement_data: pd.DataFrame,
    descriptor: dict,
    k_null: int,
    *,
    direction: str,
    dispatch: DrawDispatcher,
    recipe: Recipe,
    schedule: Schedule,
    seed: int,
) -> tuple[StepResult, FitPair]:
    """Run one $k$-against-$k+1$ BLRT step with the staged bootstrap.

    Fits the observed null and alternative (:func:`fit_pair`), then draws the parametric
    bootstrap null through ``dispatch``: a screen of ``schedule.b_screen`` draws, escalated to
    ``schedule.b_escalate`` only when the screen $p$ falls below the threshold. The $p$-value is
    the Phipson-Smyth add-one $p$ over every finite draw.

    Parameters
    ----------
    measurement_data : pandas.DataFrame
        The dataset's descriptor-aligned measurement matrix.
    descriptor : dict
        The StepMix mixed-data measurement descriptor.
    k_null : int
        The null number of classes; the alternative has ``k_null + 1``.
    direction : str
        ``"split"`` or ``"merge"``, recorded on the step.
    dispatch : callable
        The bootstrap-draw dispatcher (pool-backed in the CLI, in-process in a test).
    recipe : Recipe
        The identical fitting recipe for the observed and null fits.
    schedule : Schedule
        The staged bootstrap schedule and rejection level.
    seed : int
        Base seed for the observed fit (draw seeds are the dispatcher's concern).

    Returns
    -------
    tuple
        The step outcome (:class:`StepResult`) and the observed fit pair (:class:`FitPair`, for
        the VLMR corroborator).
    """
    pair = fit_pair(
        measurement_data,
        descriptor,
        k_null,
        n_init=recipe.n_init,
        n_random=recipe.n_random,
        seed=seed,
        jitter=recipe.jitter,
    )
    observed = pair.observed_lr
    null_params = pair.model_k.get_parameters()
    n = int(len(measurement_data))

    draws: list[float | None] = list(dispatch(k_null, null_params, n, 0, schedule.b_screen))
    p_value = phipson_smyth_p(observed, draws)
    escalated = False
    if (
        np.isfinite(p_value)
        and p_value < schedule.escalate_threshold
        and schedule.b_escalate > schedule.b_screen
    ):
        escalated = True
        draws.extend(
            dispatch(
                k_null,
                null_params,
                n,
                schedule.b_screen,
                schedule.b_escalate - schedule.b_screen,
            )
        )
        p_value = phipson_smyth_p(observed, draws)

    finite = [d for d in draws if d is not None and np.isfinite(d)]
    n_dropped = len(draws) - len(finite)
    rejected = bool(np.isfinite(p_value) and p_value <= schedule.alpha)
    step = StepResult(
        k_null=k_null,
        k_alt=k_null + 1,
        direction=direction,
        observed_lr=float(observed),
        p_value=float(p_value),
        b_used=len(finite),
        escalated=escalated,
        n_dropped=n_dropped,
        rejected=rejected,
    )
    return step, pair


def cross_validated_log_likelihood(
    measurement_data: pd.DataFrame,
    covariates: pd.DataFrame,
    descriptor: dict,
    k_values: list[int],
    *,
    seed: int,
    n_init: int,
    cv: int = 3,
) -> dict[int, float]:
    """Return the 3-fold cross-validated log-likelihood per number of classes.

    Reuses :func:`analysis.selection.validation_log_likelihood` (the released ``compute_LL``),
    the elbow corroborator's input. The elbow only corroborates the BLRT decision, so this uses
    the covariate scoring the selection stage already implements rather than a separate
    measurement-only cross-validation.

    Parameters
    ----------
    measurement_data : pandas.DataFrame
        The dataset's descriptor-aligned measurement matrix.
    covariates : pandas.DataFrame
        The structural covariate matrix on the same index.
    descriptor : dict
        The StepMix mixed-data measurement descriptor.
    k_values : list of int
        The class counts to grid over.
    seed : int
        Random seed for the estimator.
    n_init : int
        Random restarts per fold fit.
    cv : int, default 3
        Cross-validation folds.

    Returns
    -------
    dict of int to float
        Each number of classes mapped to its mean validation log-likelihood.
    """
    return selection.validation_log_likelihood(
        measurement_data, covariates, descriptor, k_values, seed=seed, n_init=n_init, cv=cv
    )


def sequential_search(
    measurement_data: pd.DataFrame,
    descriptor: dict,
    *,
    dispatch: DrawDispatcher,
    recipe: Recipe,
    schedule: Schedule,
    anchor: int = config.DEFAULT_ORDER_K_ANCHOR,
    cap: int = config.DEFAULT_ORDER_K_CAP,
    floor: int = config.DEFAULT_ORDER_K_FLOOR,
    seed: int = config.DEFAULT_ORDER_SEED,
    cv_scores: dict[int, float] | None = None,
) -> SearchResult:
    """Search for the supported number of classes, anchored at ``anchor``.

    From the anchor the primary (splitting) direction tests ``anchor`` against ``anchor + 1``
    and steps up while each step rejects, to the cap (``">=cap"`` when it hits the cap still
    rejecting). If the first split step does not reject, the secondary (merging) direction tests
    ``anchor`` against ``anchor - 1`` and steps down while each step fails to reject, to the
    floor. Each step is a warm-started parametric-bootstrap likelihood-ratio test
    (:func:`blrt_step`). The supported order is where the outward stepping stops (plan
    section 7).

    The two corroborators are attached but do not decide: the cross-validated log-likelihood
    knee (from ``cv_scores`` via :func:`kneedle_knee`) and the adjusted Lo-Mendell-Rubin test at
    the decisive comparison (:func:`vlmr_test`).

    Parameters
    ----------
    measurement_data : pandas.DataFrame
        The dataset's descriptor-aligned measurement matrix.
    descriptor : dict
        The StepMix mixed-data measurement descriptor.
    dispatch : callable
        The bootstrap-draw dispatcher.
    recipe : Recipe
        The identical fitting recipe.
    schedule : Schedule
        The staged bootstrap schedule and rejection level.
    anchor : int, optional
        The anchor number of classes (four).
    cap : int, optional
        The upper cap; the search reports ``">=cap"`` there.
    floor : int, optional
        The lower floor.
    seed : int, optional
        Base seed; each step derives its seed from it and its null class count.
    cv_scores : dict of int to float, optional
        The validation log-likelihood per number of classes for the elbow. When absent the
        knee is reported as zero.

    Returns
    -------
    SearchResult
        The supported order, the steps taken, and the corroborators.
    """
    steps: list[StepResult] = []
    pairs: dict[int, FitPair] = {}

    def run_step(k_null: int, direction: str) -> StepResult:
        step, pair = blrt_step(
            measurement_data,
            descriptor,
            k_null,
            direction=direction,
            dispatch=dispatch,
            recipe=recipe,
            schedule=schedule,
            seed=seed + 1000 * k_null,
        )
        steps.append(step)
        pairs[k_null] = pair
        return step

    capped = False
    k = anchor
    while k < cap:
        step = run_step(k, "split")
        if step.rejected:
            k += 1
        else:
            break
    else:
        # The while-loop exhausted its condition (k reached the cap) without a non-rejecting
        # step, so the last split rejected: the order is at least the cap.
        capped = True

    supported = k
    if supported == anchor:
        # The anchor could not be split; test whether it can be merged downward instead.
        k = anchor
        while k > floor:
            step = run_step(k - 1, "merge")
            if step.rejected:
                # The alternative (k) beats the null (k - 1): k is supported, stop.
                break
            k -= 1
        supported = k

    if supported > anchor:
        direction = "split"
    elif supported < anchor:
        direction = "merge"
    else:
        direction = "stable"

    result = SearchResult(
        supported_k=supported,
        capped=capped,
        direction=direction,
        steps=steps,
        n_dropped=sum(s.n_dropped for s in steps),
    )

    if cv_scores:
        result.cv_log_likelihood = dict(cv_scores)
        result.elbow_knee = kneedle_knee(
            sorted(cv_scores), [cv_scores[k] for k in sorted(cv_scores)]
        )

    decisive = pairs.get(supported - 1)
    if (
        decisive is not None
        and decisive.model_k1 is not None
        and any(s.k_alt == supported and s.rejected for s in steps)
    ):
        n = int(len(measurement_data))
        result.vlmr = vlmr_test(
            decisive.ll_k,
            int(decisive.model_k.n_parameters),
            supported - 1,
            decisive.ll_k1,
            int(decisive.model_k1.n_parameters),
            supported,
            n,
        )
    return result


def agreement_flag(result: SearchResult, pooled_supported_k: int) -> bool:
    """Return whether a positive order-change claim is corroborated for a stratum.

    A positive order-change claim requires agreement (plan section 7): the BLRT supported order
    differs from the pooled cohort's, the cross-validated elbow knee has moved off the pooled
    order too, and the adjusted Lo-Mendell-Rubin test agrees at the decisive comparison. A
    stratum whose order matches the pooled order is not an order change however far the pooled
    order sits from four, so it never needs corroboration. BLRT non-rejection alone (the order
    matching the pooled order) is the stability claim.

    Parameters
    ----------
    result : SearchResult
        The stratum's search outcome.
    pooled_supported_k : int
        The pooled cohort's supported order under the identical procedure.

    Returns
    -------
    bool
        ``True`` when the order change is corroborated by all three, ``False`` otherwise
        (including when the order matches the pooled order).
    """
    if result.supported_k == pooled_supported_k:
        return False
    blrt_moved = result.supported_k != pooled_supported_k
    elbow_moved = result.elbow_knee != 0 and result.elbow_knee != pooled_supported_k
    vlmr_agrees = (
        bool(result.vlmr)
        and np.isfinite(result.vlmr.get("p", float("nan")))
        and (result.vlmr["p"] <= config.DEFAULT_ORDER_ALPHA)
    )
    return blrt_moved and elbow_moved and vlmr_agrees
