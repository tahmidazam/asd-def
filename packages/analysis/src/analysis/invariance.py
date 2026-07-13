r"""Score-based measurement invariance from a single cached fit (plan section 7e).

The stratified analysis (section 7) asks whether the four reference classes are stable as the
mixture is re-estimated within strata of age at diagnosis or diagnostic era. Every refit-based
scheme carries a fit cost and a permutation null. This module answers the same question from a
single cached fit with an analytic null and no refitting, following the empirical fluctuation
process of Merkle and Zeileis (2013, Psychometrika) and Merkle, Fan and Zeileis (2014).

The idea. At the maximum-likelihood estimate every proband contributes a score, the gradient of
its log-likelihood with respect to each class-profile parameter. Fisher's identity gives the
casewise score with respect to the class-$k$ value of feature $j$ as
$r_{ik}\,\partial_\theta \log f_j(x_{ij};\theta_{jk})$, where $r_{ik}$ is the posterior
responsibility (``predict_proba``). The focal parameters are the class-conditional locations,
the profiles the whole analysis measures:

- a Gaussian mean has score $r_{ik}(x_{ij}-\mu_{jk})/\sigma^2_{jk}$;
- a Bernoulli probability has score $r_{ik}(x_{ij}-p_{jk})/(p_{jk}(1-p_{jk}))$;
- a categorical outcome $l$ has the multinomial-logit score $r_{ik}(\mathbb{1}[x_{ij}=l]-p_{jkl})$.

All three are validated against a central finite difference of the per-sample log-likelihood to
machine precision (:func:`numerical_score`, the correctness gate). The categorical outcomes of a
feature sum to a redundant direction (the probabilities are a simplex), which the whitening step
below removes, so no reference outcome is dropped by hand.

Ordered by the axis, the standardised running sum of these scores is the empirical fluctuation
process $B(t)$, $t \in [0,1]$. Standardisation is by the inverse square root of the
outer-product-of-gradients covariance of the focal block, which decorrelates its dimensions; the
process is pinned to zero at both ends and, under stability, converges to a Brownian bridge.
Two functionals read the process: $\text{maxLM} = \max_t \lVert B(t) \rVert^2$ (power against an
abrupt break, with $\arg\max_t$ the estimated break position) and
$\text{CvM} = \int_0^1 \lVert B(t) \rVert^2\,\mathrm{d}t$ (power against gradual drift). The null
is drawn by simulating many $d$-dimensional Brownian bridges on the axis's own time grid and
reading the same functional off each, so the $p$-value is analytic rather than a refit.

The module is a pure consumer of a cached measurement-only fit and a per-proband axis, method
independent, mirroring the cheap half of :mod:`analysis.drift`. It does not refit the mixture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from analysis.features import Typing

# A focal score column is dropped before whitening when it never varies across the probands
# (a categorical outcome that is never observed, say), because a constant score carries no
# information and would make the outer-product covariance singular.
_ZERO_VARIANCE = 1e-12

# Relative floor on the eigenvalues kept when inverting the focal covariance. Directions below
# it are the redundant ones (the categorical simplex constraint contributes one per feature) and
# are dropped, so the effective dimension is the covariance rank, not the raw column count.
_EIGEN_RTOL = 1e-8


@dataclass(frozen=True)
class FocalParameter:
    """One class-conditional location parameter, the unit a casewise score is taken for.

    Attributes
    ----------
    cls : int
        The latent class the parameter belongs to.
    feature : str
        The feature (measurement column) the parameter describes.
    kind : str
        The emission type: ``"gaussian_mean"``, ``"bernoulli"`` or ``"multinomial_logit"``.
    column : int
        The feature's column index in the measurement matrix (for reading $x_{ij}$).
    local : int
        The feature's index within its emission sub-model (for reading the fitted parameter).
    outcome : int or None
        For a categorical feature, the outcome $l$ this score is taken for; ``None`` otherwise.
    """

    cls: int
    feature: str
    kind: str
    column: int
    local: int
    outcome: int | None = None


@dataclass
class CasewiseScores:
    """The casewise focal scores for a fit: one column per focal parameter, one row per proband.

    Attributes
    ----------
    values : numpy.ndarray
        The score matrix, shape ``(n_probands, n_focal)``.
    parameters : list of FocalParameter
        The focal parameter each column scores, in column order.
    probands : pandas.Index
        The proband ids, in row order (the measurement matrix index).
    responsibilities : numpy.ndarray
        The posterior class responsibilities, shape ``(n_probands, n_classes)``.
    """

    values: np.ndarray
    parameters: list[FocalParameter]
    probands: pd.Index
    responsibilities: np.ndarray


def _ordered_type_columns(measurement_data: pd.DataFrame, typing: Typing) -> dict[str, list[str]]:
    """Return the present feature columns of each emission type, in measurement-matrix order.

    The measurement matrix places the continuous, then binary, then categorical columns, each in
    the order its type is listed. The emission parameter arrays follow the same order, so this
    mapping is what aligns a fitted parameter to its column.
    """
    present = set(measurement_data.columns)
    return {
        "continuous": [c for c in typing.continuous if c in present],
        "binary": [c for c in typing.binary if c in present],
        "categorical": [c for c in typing.categorical if c in present],
    }


def _require_measurement_only(model) -> None:
    """Raise if the fit carries a structural model, which the score test is not defined for.

    The invariance test reads the measurement model's class-conditional locations from the
    marginal (measurement-only) reference. A covariate (one-step) fit mixes the structural
    likelihood into the responsibilities and is a different estimand, so it is refused rather
    than scored silently.
    """
    if hasattr(model, "_sm"):
        raise ValueError(
            "invariance requires a measurement-only fit (no structural model); "
            "resolve the `structural='measurement'` reference, not the covariate one."
        )


def per_sample_log_likelihood(model, x_values: np.ndarray) -> np.ndarray:
    r"""Return the marginal log-likelihood of each proband under a measurement-only fit.

    The per-sample log-likelihood is $\log \sum_k \pi_k \prod_j f_j(x_{ij};\theta_{jk})$,
    the log-sum-exp over classes of the class prior plus the measurement emission. This is the
    quantity the casewise scores are gradients of, and the finite-difference gate differentiates.

    Parameters
    ----------
    model : StepMix
        A fitted measurement-only estimator.
    x_values : numpy.ndarray
        The measurement matrix, shape ``(n_probands, n_features)``, in the fit's column order.

    Returns
    -------
    numpy.ndarray
        The per-proband log-likelihood, shape ``(n_probands,)``.
    """
    _require_measurement_only(model)
    log_joint = model._mm.log_likelihood(x_values) + np.log(model.weights_)[None, :]
    return logsumexp(log_joint, axis=1)


def responsibilities(model, x_values: np.ndarray) -> np.ndarray:
    """Return the posterior class responsibilities $r_{ik}$ for a measurement-only fit.

    Parameters
    ----------
    model : StepMix
        A fitted measurement-only estimator.
    x_values : numpy.ndarray
        The measurement matrix, in the fit's column order.

    Returns
    -------
    numpy.ndarray
        The responsibilities, shape ``(n_probands, n_classes)``, each row summing to one.
    """
    _require_measurement_only(model)
    log_joint = model._mm.log_likelihood(x_values) + np.log(model.weights_)[None, :]
    return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))


def casewise_scores(model, measurement_data: pd.DataFrame, typing: Typing) -> CasewiseScores:
    """Compute the casewise focal score of every class-conditional location parameter.

    For each class and each feature the analytic score of the fitted location parameter is
    formed by Fisher's identity: the responsibility times the emission gradient. Gaussian means
    and Bernoulli probabilities give one score column per feature; a categorical feature gives
    one column per observed outcome (the never-observed padding columns are skipped).

    Parameters
    ----------
    model : StepMix
        A fitted measurement-only estimator.
    measurement_data : pandas.DataFrame
        The measurement matrix the fit was estimated on, in its column order.
    typing : analysis.features.Typing
        The feature typing that assigns each feature its emission.

    Returns
    -------
    CasewiseScores
        The score matrix, the focal parameter per column, the proband index, and the
        responsibilities.
    """
    _require_measurement_only(model)
    x = measurement_data.to_numpy(dtype=float)
    resp = responsibilities(model, x)
    n, n_classes = resp.shape
    columns = {c: i for i, c in enumerate(measurement_data.columns)}
    ordered = _ordered_type_columns(measurement_data, typing)
    params = model.get_parameters()["measurement"]

    score_cols: list[np.ndarray] = []
    focal: list[FocalParameter] = []

    def add(vec: np.ndarray, param: FocalParameter) -> None:
        if np.all(np.isfinite(vec)) and float(np.nanstd(vec)) > _ZERO_VARIANCE:
            score_cols.append(vec)
            focal.append(param)

    # Gaussian means: r_ik (x_ij - mu_jk) / var_jk.
    if ordered["continuous"]:
        means = params["continuous"]["means"]
        variances = params["continuous"]["covariances"]
        for local, feature in enumerate(ordered["continuous"]):
            col = columns[feature]
            for k in range(n_classes):
                vec = resp[:, k] * (x[:, col] - means[k, local]) / variances[k, local]
                add(vec, FocalParameter(k, feature, "gaussian_mean", col, local))

    # Bernoulli probabilities: r_ik (x_ij - p_jk) / (p_jk (1 - p_jk)).
    if ordered["binary"]:
        pis = params["binary"]["pis"]
        for local, feature in enumerate(ordered["binary"]):
            col = columns[feature]
            for k in range(n_classes):
                p = pis[k, local]
                vec = resp[:, k] * (x[:, col] - p) / (p * (1.0 - p))
                add(vec, FocalParameter(k, feature, "bernoulli", col, local))

    # Categorical outcomes (multinomial logit): r_ik (1[x_ij = l] - p_jkl).
    if ordered["categorical"]:
        cat_model = model._mm.models["categorical"]
        cat_matrix = measurement_data[ordered["categorical"]].to_numpy(dtype=float)
        one_hot = cat_model.encode_features(cat_matrix)
        max_outcomes = int(cat_model.parameters["max_n_outcomes"])
        pis = params["categorical"]["pis"]
        for local, feature in enumerate(ordered["categorical"]):
            col = columns[feature]
            for outcome in range(max_outcomes):
                oh_col = local * max_outcomes + outcome
                indicator = one_hot[:, oh_col]
                if not np.isfinite(indicator).all() or indicator.sum() == 0:
                    # A never-observed outcome carries no information; skip it so the block does
                    # not gain a constant, redundant score column.
                    continue
                for k in range(n_classes):
                    vec = resp[:, k] * (indicator - pis[k, oh_col])
                    add(vec, FocalParameter(k, feature, "multinomial_logit", col, local, outcome))

    values = np.column_stack(score_cols) if score_cols else np.empty((n, 0))
    return CasewiseScores(values, focal, measurement_data.index, resp)


def numerical_score(
    model,
    measurement_data: pd.DataFrame,
    parameter: FocalParameter,
    *,
    eps: float = 1e-5,
) -> np.ndarray:
    """Return the finite-difference casewise score of one focal parameter (the correctness gate).

    Perturbs the single parameter by plus and minus ``eps``, recomputes the per-sample
    log-likelihood each way, restores the fit, and returns the central difference. A Gaussian
    mean and a Bernoulli probability are perturbed directly; a categorical outcome is perturbed
    on the logit scale (the outcome probabilities are renormalised by a softmax), which matches
    the analytic multinomial-logit score. The analytic :func:`casewise_scores` must match this to
    machine precision per proband; a mismatch means a wrong gradient, which would invalidate every
    $p$-value.

    Parameters
    ----------
    model : StepMix
        A fitted measurement-only estimator. Restored to its fitted parameters on return.
    measurement_data : pandas.DataFrame
        The measurement matrix, in the fit's column order.
    parameter : FocalParameter
        The focal parameter to differentiate.
    eps : float, optional
        The central-difference step.

    Returns
    -------
    numpy.ndarray
        The finite-difference score, shape ``(n_probands,)``.
    """
    import copy

    _require_measurement_only(model)
    x = measurement_data.to_numpy(dtype=float)
    base = model.get_parameters()

    def evaluate(mutate) -> np.ndarray:
        params = copy.deepcopy(base)
        mutate(params)
        model.set_parameters(params)
        out = per_sample_log_likelihood(model, x)
        model.set_parameters(copy.deepcopy(base))
        return out

    k, local, outcome = parameter.cls, parameter.local, parameter.outcome

    if parameter.kind == "gaussian_mean":

        def shift(params, sign):
            params["measurement"]["continuous"]["means"][k, local] += sign * eps

    elif parameter.kind == "bernoulli":

        def shift(params, sign):
            params["measurement"]["binary"]["pis"][k, local] += sign * eps

    elif parameter.kind == "multinomial_logit":
        cat_model = model._mm.models["categorical"]
        max_outcomes = int(cat_model.parameters["max_n_outcomes"])

        def shift(params, sign):
            block = slice(local * max_outcomes, (local + 1) * max_outcomes)
            probs = params["measurement"]["categorical"]["pis"][k, block].astype(float)
            log_p = np.log(np.clip(probs, 1e-300, None))
            log_p[outcome] += sign * eps
            params["measurement"]["categorical"]["pis"][k, block] = np.exp(log_p - logsumexp(log_p))

    else:  # pragma: no cover - guarded by the enumeration in casewise_scores
        raise ValueError(f"unknown focal kind {parameter.kind!r}")

    return (evaluate(lambda p: shift(p, +1)) - evaluate(lambda p: shift(p, -1))) / (2.0 * eps)


# ---------------------------------------------------------------------------------------------
# The empirical fluctuation process and its bridge functionals.
# ---------------------------------------------------------------------------------------------


@dataclass
class TimeGrid:
    """The ordinal time grid the fluctuation process is read on.

    Attributes
    ----------
    t : numpy.ndarray
        The cumulative sample fraction at each evaluation point, ending at one.
    dt : numpy.ndarray
        The spacing between consecutive points (``t`` differenced, with a leading ``t[0]``), the
        integration weight for the Cramer-von Mises functional and the increment variance for the
        simulated null.
    index : numpy.ndarray
        The proband count included at each evaluation point (the cumulative-sum row to read).
    positions : numpy.ndarray
        The axis value at each evaluation point, for reading a break off the process.
    """

    t: np.ndarray
    dt: np.ndarray
    index: np.ndarray
    positions: np.ndarray


def build_time_grid(sorted_axis: np.ndarray, *, max_points: int = 512) -> TimeGrid:
    """Build the evaluation grid from the axis-sorted probands, collapsing ties.

    The process changes value proband by proband, but within a run of tied axis values the
    proband order is arbitrary, so the process is only read at the end of each tied run (an
    order-invariant point). When there are more such points than ``max_points`` the grid is
    thinned to roughly equal ordinal spacing; the observed statistic and the simulated null use
    the same thinned grid, so the reading stays calibrated.

    Parameters
    ----------
    sorted_axis : numpy.ndarray
        The axis values of the covered probands, in ascending order.
    max_points : int, optional
        The largest grid the process is evaluated on.

    Returns
    -------
    TimeGrid
        The evaluation grid.
    """
    n = int(sorted_axis.shape[0])
    # The last index of each run of equal axis values (one-based counts of probands included).
    change = np.flatnonzero(np.diff(sorted_axis) != 0) + 1
    boundaries = np.append(change, n)
    if boundaries.shape[0] > max_points:
        pick = np.linspace(0, boundaries.shape[0] - 1, max_points).round().astype(int)
        boundaries = np.unique(boundaries[pick])
    t = boundaries / n
    dt = np.diff(np.concatenate([[0.0], t]))
    positions = sorted_axis[boundaries - 1]
    return TimeGrid(t=t, dt=dt, index=boundaries, positions=positions)


def _whitening(block_values: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the inverse-square-root whitening of a focal block and its effective dimension.

    The block covariance is the outer-product-of-gradients estimate, the mean over probands of
    the outer product of the score row. Its symmetric inverse square root decorrelates the block
    so its dimensions become independent standard directions under stability. Redundant
    directions (a categorical feature's outcomes are a simplex, so one direction per feature has
    no variance) are dropped by an eigenvalue floor, so the effective dimension is the covariance
    rank rather than the raw column count.

    Parameters
    ----------
    block_values : numpy.ndarray
        The focal scores of one block, shape ``(n_probands, d)``.

    Returns
    -------
    tuple
        The whitening matrix, shape ``(d_eff, d)``, and the effective dimension ``d_eff``.
    """
    n = block_values.shape[0]
    cov = (block_values.T @ block_values) / n
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    keep = eigenvalues > _EIGEN_RTOL * float(eigenvalues.max(initial=0.0))
    kept = eigenvectors[:, keep]
    inv_root = 1.0 / np.sqrt(eigenvalues[keep])
    whitening = (kept * inv_root).T
    return whitening, int(keep.sum())


def fluctuation_process(block_values: np.ndarray, grid: TimeGrid) -> np.ndarray:
    """Return the standardised, bridge-tied fluctuation process on the grid.

    The scores are cumulated in axis order, standardised by the whitening, scaled by the root
    of the sample size, and tied down linearly so the process is zero at both ends (the residual
    tilt from an approximate optimiser or from axis coverage below one is removed). Under
    stability the result converges to a Brownian bridge.

    Parameters
    ----------
    block_values : numpy.ndarray
        The focal scores of one block, in axis order, shape ``(n_probands, d)``.
    grid : TimeGrid
        The evaluation grid.

    Returns
    -------
    numpy.ndarray
        The process, shape ``(n_grid, d_eff)``.
    """
    n = block_values.shape[0]
    whitening, _ = _whitening(block_values)
    cumulative = np.cumsum(block_values, axis=0)
    partial = cumulative[grid.index - 1]
    process = (whitening @ partial.T).T / np.sqrt(n)
    endpoint = process[-1]
    return process - np.outer(grid.t, endpoint)


def bridge_functionals(process: np.ndarray, grid: TimeGrid) -> tuple[float, float, int]:
    """Return the maxLM and Cramer-von Mises functionals of a process and the break index.

    ``maxLM`` is the largest squared norm over the grid, powerful against an abrupt break, and
    its location is the estimated break. ``CvM`` is the grid integral of the squared norm,
    powerful against a gradual drift.

    Parameters
    ----------
    process : numpy.ndarray
        The fluctuation process, shape ``(n_grid, d_eff)``.
    grid : TimeGrid
        The evaluation grid.

    Returns
    -------
    tuple
        ``maxLM``, ``CvM``, and the grid index of the maximum.
    """
    squared_norm = np.einsum("gd,gd->g", process, process)
    argmax = int(np.argmax(squared_norm))
    max_lm = float(squared_norm[argmax])
    cvm = float(np.sum(squared_norm * grid.dt))
    return max_lm, cvm, argmax


def simulate_bridge_null(
    d_eff: int, grid: TimeGrid, *, n_sim: int, seed: int, chunk: int = 256
) -> dict[str, np.ndarray]:
    """Simulate the null distributions of the two functionals for a ``d_eff``-dimensional bridge.

    Draws independent ``d_eff``-dimensional Brownian bridges on the grid and reads the same
    functionals off each, so the null matches the observed statistic's grid exactly. This is a
    Gaussian-process simulation, not a model refit. The draws run in chunks to bound memory.

    Parameters
    ----------
    d_eff : int
        The effective dimension of the focal block.
    grid : TimeGrid
        The evaluation grid.
    n_sim : int
        The number of simulated bridges.
    seed : int
        The seed for the draw.
    chunk : int, optional
        The number of bridges simulated at once.

    Returns
    -------
    dict
        ``"maxLM"`` and ``"cvm"``, each an array of ``n_sim`` null draws.
    """
    rng = np.random.default_rng(seed)
    g = grid.t.shape[0]
    max_lm = np.empty(n_sim)
    cvm = np.empty(n_sim)
    root_dt = np.sqrt(grid.dt)[:, None]
    done = 0
    while done < n_sim:
        size = min(chunk, n_sim - done)
        # A Brownian motion on the grid, then tied down to a bridge, per simulation.
        increments = rng.standard_normal((size, g, d_eff)) * root_dt[None]
        motion = np.cumsum(increments, axis=1)
        bridge = motion - grid.t[None, :, None] * motion[:, -1:, :]
        squared_norm = np.einsum("sgd,sgd->sg", bridge, bridge)
        max_lm[done : done + size] = squared_norm.max(axis=1)
        cvm[done : done + size] = squared_norm @ grid.dt
        done += size
    return {"maxLM": max_lm, "cvm": cvm}


def simulate_bridge_band(
    d_eff: int, grid: TimeGrid, *, n_sim: int, seed: int, quantiles=(0.5, 0.95), chunk: int = 256
) -> dict[float, np.ndarray]:
    """Return pointwise quantiles of a simulated bridge's squared norm, the figure's null band.

    Unlike :func:`simulate_bridge_null`, which keeps only the maxLM and CvM of each draw, this
    keeps the squared norm at every grid point, so the observed process can be drawn against a
    pointwise null envelope.

    Parameters
    ----------
    d_eff : int
        The effective dimension of the block.
    grid : TimeGrid
        The evaluation grid.
    n_sim : int
        The number of simulated bridges.
    seed : int
        The seed for the draw.
    quantiles : sequence of float, optional
        The quantiles of the squared norm to return per grid point.
    chunk : int, optional
        The number of bridges simulated at once.

    Returns
    -------
    dict
        A mapping from each quantile to its per-grid-point curve, shape ``(n_grid,)``.
    """
    rng = np.random.default_rng(seed)
    g = grid.t.shape[0]
    root_dt = np.sqrt(grid.dt)[:, None]
    squared = np.empty((n_sim, g))
    done = 0
    while done < n_sim:
        size = min(chunk, n_sim - done)
        increments = rng.standard_normal((size, g, d_eff)) * root_dt[None]
        motion = np.cumsum(increments, axis=1)
        bridge = motion - grid.t[None, :, None] * motion[:, -1:, :]
        squared[done : done + size] = np.einsum("sgd,sgd->sg", bridge, bridge)
        done += size
    return {q: np.quantile(squared, q, axis=0) for q in quantiles}


def add_one_pvalue(observed: float, null_draws: np.ndarray) -> float:
    """Return the simulation $p$-value with the Phipson-Smyth add-one correction.

    The smallest attainable $p$-value is $1/(B+1)$ rather than zero, so a statistic beyond every
    draw is not reported as impossible.
    """
    draws = np.asarray(null_draws, dtype=float)
    exceed = int(np.sum(draws >= observed))
    return (1 + exceed) / (1 + draws.size)


def directional_slopes(block_values: np.ndarray, axis_values: np.ndarray) -> np.ndarray:
    """Return each focal parameter's ordinary-least-squares slope of score against the axis.

    A systematic non-zero slope is directional drift: the parameter's score trends with the
    ordering variable rather than fluctuating around zero. The sign carries the direction.

    Parameters
    ----------
    block_values : numpy.ndarray
        The focal scores of one block, shape ``(n_probands, d)``.
    axis_values : numpy.ndarray
        The axis value of each proband, in the same row order.

    Returns
    -------
    numpy.ndarray
        The per-parameter slope, shape ``(d,)``.
    """
    centred = axis_values - axis_values.mean()
    denominator = float(centred @ centred)
    if denominator == 0.0:
        return np.zeros(block_values.shape[1])
    return (centred @ block_values) / denominator


# ---------------------------------------------------------------------------------------------
# Focal blocks and the per-block test.
# ---------------------------------------------------------------------------------------------


@dataclass
class BlockResult:
    """The invariance reading for one focal block.

    Attributes
    ----------
    label : str
        The block identifier (a class, or a class crossed with a category).
    cls : int
        The class the block belongs to.
    category : str or None
        The feature category the block is restricted to, or ``None`` for a whole-class block.
    n : int
        The number of probands the process ran over.
    d : int
        The number of focal parameters in the block.
    d_eff : int
        The effective dimension after dropping redundant directions.
    max_lm : float
        The maxLM statistic.
    cvm : float
        The Cramer-von Mises statistic.
    p_max_lm : float
        The bridge $p$-value of maxLM.
    p_cvm : float
        The bridge $p$-value of CvM.
    break_position : float
        The axis value at the maxLM maximum, the estimated break.
    break_low : float
        The lower edge of the break confidence set.
    break_high : float
        The upper edge of the break confidence set.
    direction : float
        The signed directional slope of the block's strongest-trending parameter.
    direction_feature : str
        The feature of that strongest-trending parameter.
    reject_max_lm : bool
        Whether maxLM passes the Benjamini-Hochberg step across blocks (filled by the caller).
    reject_cvm : bool
        Whether CvM passes the Benjamini-Hochberg step across blocks (filled by the caller).
    """

    label: str
    cls: int
    category: str | None
    n: int
    d: int
    d_eff: int
    max_lm: float
    cvm: float
    p_max_lm: float
    p_cvm: float
    break_position: float
    break_low: float
    break_high: float
    direction: float
    direction_feature: str
    reject_max_lm: bool = False
    reject_cvm: bool = False


def test_block(
    block_values: np.ndarray,
    parameters: list[FocalParameter],
    sorted_axis: np.ndarray,
    order: np.ndarray,
    grid: TimeGrid,
    *,
    label: str,
    cls: int,
    category: str | None,
    n_sim: int,
    seed: int,
) -> BlockResult:
    """Run the fluctuation-process test on one focal block.

    Cumulates the block's scores in axis order, reads the two functionals, draws the analytic
    null, and estimates the break with a confidence set from the sup statistic. Pure and cheap:
    no fitting, only the stored scores.

    Parameters
    ----------
    block_values : numpy.ndarray
        The block's focal scores in the fit's row order, shape ``(n_probands, d)``.
    parameters : list of FocalParameter
        The focal parameter of each column, for the directional read.
    sorted_axis : numpy.ndarray
        The covered probands' axis values, ascending.
    order : numpy.ndarray
        The permutation that sorts the probands by the axis.
    grid : TimeGrid
        The evaluation grid for this axis.
    label, cls, category : str, int, str or None
        The block's identity.
    n_sim : int
        The number of simulated bridges for the null.
    seed : int
        The seed for the null draw.

    Returns
    -------
    BlockResult
        The block's statistics, $p$-values, break, and directional slope.
    """
    ordered = block_values[order]
    process = fluctuation_process(ordered, grid)
    max_lm, cvm, argmax = bridge_functionals(process, grid)
    d_eff = process.shape[1]
    null = simulate_bridge_null(d_eff, grid, n_sim=n_sim, seed=seed)
    p_max_lm = add_one_pvalue(max_lm, null["maxLM"])
    p_cvm = add_one_pvalue(cvm, null["cvm"])

    # The break confidence set: the axis positions whose squared norm clears the null's maxLM
    # critical value, so the break location is reported with the same sup statistic that tested it.
    critical = float(np.quantile(null["maxLM"], 0.95))
    squared_norm = np.einsum("gd,gd->g", process, process)
    in_set = np.flatnonzero(squared_norm >= critical)
    if in_set.size:
        break_low = float(grid.positions[in_set.min()])
        break_high = float(grid.positions[in_set.max()])
    else:
        break_low = break_high = float(grid.positions[argmax])

    slopes = directional_slopes(ordered, sorted_axis)
    strongest = int(np.argmax(np.abs(slopes))) if slopes.size else 0
    direction = float(slopes[strongest]) if slopes.size else 0.0
    direction_feature = parameters[strongest].feature if parameters else ""

    return BlockResult(
        label=label,
        cls=cls,
        category=category,
        n=int(block_values.shape[0]),
        d=int(block_values.shape[1]),
        d_eff=d_eff,
        max_lm=max_lm,
        cvm=cvm,
        p_max_lm=p_max_lm,
        p_cvm=p_cvm,
        break_position=float(grid.positions[argmax]),
        break_low=break_low,
        break_high=break_high,
        direction=direction,
        direction_feature=direction_feature,
    )


def _category_of(feature: str, category_map) -> str:
    """Return a feature's category, or ``"unmapped"`` for an absent or blank entry."""
    value = category_map.get(str(feature)) if category_map is not None else None
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unmapped"
    return str(value)


def focal_blocks(
    parameters: list[FocalParameter], category_map, *, by_category: bool
) -> dict[tuple[int, str | None], list[int]]:
    """Group focal-parameter column indices into the blocks the test runs over.

    The whole-class blocks pool every parameter of a class. The category blocks restrict a class
    to one feature category, so a drift can be localised to a class and a symptom domain.

    Parameters
    ----------
    parameters : list of FocalParameter
        The focal parameter of each score column.
    category_map : Mapping or None
        The feature-to-category mapping; only needed when ``by_category`` is set.
    by_category : bool
        Build class-by-category blocks in addition to whole-class blocks.

    Returns
    -------
    dict
        A mapping from ``(class, category)`` (category ``None`` for a whole-class block) to the
        list of column indices in that block.
    """
    blocks: dict[tuple[int, str | None], list[int]] = {}
    for col, param in enumerate(parameters):
        blocks.setdefault((param.cls, None), []).append(col)
        if by_category:
            category = _category_of(param.feature, category_map)
            blocks.setdefault((param.cls, category), []).append(col)
    return blocks


@dataclass
class TopProcess:
    r"""The empirical fluctuation process and null band of the most significant block.

    Kept so the exploratory figure has a real curve to draw without recomputing the scores.

    Attributes
    ----------
    label : str
        The block the process belongs to.
    t : numpy.ndarray
        The grid's cumulative sample fraction.
    positions : numpy.ndarray
        The axis value at each grid point.
    observed : numpy.ndarray
        The observed squared norm $\\lVert B(t) \\rVert^2$ at each grid point.
    null_q50 : numpy.ndarray
        The pointwise median of the simulated bridge's squared norm.
    null_q95 : numpy.ndarray
        The pointwise 95th percentile of the simulated bridge's squared norm.
    """

    label: str
    t: np.ndarray
    positions: np.ndarray
    observed: np.ndarray
    null_q50: np.ndarray
    null_q95: np.ndarray


@dataclass
class InvarianceResult:
    """The invariance reading for a fit against one axis.

    Attributes
    ----------
    blocks : list of BlockResult
        The per-block results, Benjamini-Hochberg decisions filled in.
    n_reference : int
        The number of probands in the reference fit.
    n_covered : int
        The number with a non-missing axis value, which the test ran over.
    coverage : float
        ``n_covered / n_reference``.
    axis : str
        The ordering variable.
    n_sim : int
        The number of simulated bridges per block.
    top_process : TopProcess or None
        The fluctuation process of the most significant block, for the figure.
    q : float
        The false-discovery-rate level of the Benjamini-Hochberg step.
    """

    blocks: list[BlockResult]
    n_reference: int
    n_covered: int
    coverage: float
    axis: str
    n_sim: int
    top_process: TopProcess | None = None
    q: float = 0.05


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    r"""Return a boolean mask of the hypotheses that pass Benjamini-Hochberg FDR control.

    A hypothesis is rejected if its $p$-value is at or below the largest threshold
    $q\,\text{rank}/m$ it satisfies, where $m$ is the number of finite $p$-values. Not-a-number
    entries never pass. This mirrors :func:`analysis.drift.benjamini_hochberg`, the repo
    convention for the strata-by-class tests.
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


def run_invariance(
    model,
    measurement_data: pd.DataFrame,
    typing: Typing,
    axis_values: pd.Series,
    *,
    axis: str,
    category_map=None,
    by_category: bool = True,
    n_sim: int = 2000,
    seed: int = 0,
    max_grid: int = 512,
    q: float = 0.05,
) -> InvarianceResult:
    """Run the score-based invariance test of a fit against an axis, over every focal block.

    Joins the axis to the reference probands, drops those with a missing axis value (reporting
    coverage), computes the casewise scores once, and runs the fluctuation-process test per
    block. Significance is Benjamini-Hochberg controlled across blocks within the axis.

    Parameters
    ----------
    model : StepMix
        The fitted measurement-only reference estimator.
    measurement_data : pandas.DataFrame
        The measurement matrix the fit was estimated on.
    typing : analysis.features.Typing
        The feature typing.
    axis_values : pandas.Series
        The per-proband axis value, indexed by proband id.
    axis : str
        The axis name, recorded on the result.
    category_map : Mapping or None, optional
        The feature-to-category mapping for the class-by-category blocks.
    by_category : bool, optional
        Add the class-by-category blocks.
    n_sim : int, optional
        The number of simulated bridges per block.
    seed : int, optional
        The base seed; each block's null uses a distinct derived seed.
    max_grid : int, optional
        The largest evaluation grid (ties collapsed, then thinned to this many points).
    q : float, optional
        The false-discovery-rate level.

    Returns
    -------
    InvarianceResult
        The per-block results with FDR decisions, and the axis coverage.
    """
    scores = casewise_scores(model, measurement_data, typing)
    n_reference = int(scores.values.shape[0])

    axis_on_index = axis_values.reindex(scores.probands)
    covered_mask = axis_on_index.notna().to_numpy()
    covered = scores.values[covered_mask]
    covered_axis = axis_on_index[covered_mask].to_numpy(dtype=float)
    n_covered = int(covered.shape[0])

    order = np.argsort(covered_axis, kind="stable")
    sorted_axis = covered_axis[order]
    grid = build_time_grid(sorted_axis, max_points=max_grid)

    block_columns = focal_blocks(scores.parameters, category_map, by_category=by_category)
    results: list[BlockResult] = []
    for block_seed, ((cls, category), cols) in enumerate(sorted(block_columns.items(), key=str)):
        block_values = covered[:, cols]
        block_params = [scores.parameters[c] for c in cols]
        label = f"class {cls}" if category is None else f"class {cls} x {category}"
        results.append(
            test_block(
                block_values,
                block_params,
                sorted_axis,
                order,
                grid,
                label=label,
                cls=cls,
                category=category,
                n_sim=n_sim,
                seed=seed + 1 + block_seed,
            )
        )

    reject_max = benjamini_hochberg(np.array([r.p_max_lm for r in results]), q)
    reject_cvm = benjamini_hochberg(np.array([r.p_cvm for r in results]), q)
    for r, rm, rc in zip(results, reject_max, reject_cvm, strict=True):
        r.reject_max_lm = bool(rm)
        r.reject_cvm = bool(rc)

    top_process = _top_block_process(
        results, block_columns, covered, order, grid, n_sim=n_sim, seed=seed
    )

    return InvarianceResult(
        blocks=results,
        n_reference=n_reference,
        n_covered=n_covered,
        coverage=(n_covered / n_reference) if n_reference else float("nan"),
        axis=axis,
        n_sim=n_sim,
        top_process=top_process,
        q=q,
    )


def _top_block_process(
    results: list[BlockResult],
    block_columns: dict[tuple[int, str | None], list[int]],
    covered: np.ndarray,
    order: np.ndarray,
    grid: TimeGrid,
    *,
    n_sim: int,
    seed: int,
) -> TopProcess | None:
    """Return the fluctuation process and null band of the strongest-drifting category block.

    Recomputes only the winning block's process (the scores are already in hand), so the figure
    can draw a real curve against a pointwise null envelope. The winner is the largest-maxLM
    class-by-category block, the most localised reading; whole-class blocks are set aside because
    their maxLM is inflated by their much larger dimension and so is not comparable. At the sample
    sizes here many blocks tie at the smallest attainable $p$-value, so the effect size, not the
    $p$-value, orders them.
    """
    if not results:
        return None
    category_blocks = [r for r in results if r.category is not None]
    top = max(category_blocks or results, key=lambda r: r.max_lm)
    ordered = covered[:, block_columns[(top.cls, top.category)]][order]
    process = fluctuation_process(ordered, grid)
    squared = np.einsum("gd,gd->g", process, process)
    band = simulate_bridge_band(process.shape[1], grid, n_sim=n_sim, seed=seed)
    return TopProcess(
        label=top.label,
        t=grid.t,
        positions=grid.positions,
        observed=squared,
        null_q50=band[0.5],
        null_q95=band[0.95],
    )
