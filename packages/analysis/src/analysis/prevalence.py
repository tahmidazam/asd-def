"""Prevalence drift: how the frozen class proportions vary along an axis (PREV).

The PREV hypothesis of ``.context/hypotheses.md`` asks whether the four Litman class mixing
proportions are constant across diagnostic era and across age at diagnosis, or whether at least
one class's proportion trends along an axis. It is distinct from profile drift (the INV family):
the classes are held fixed at the measurement-only reference fit and only their sizes are read as
a function of the axis, so nothing is re-estimated. The estimand is the mixing proportions as a
function of the axis.

The rigorous read is a three-step estimator (Vermunt 2010; Bakk, Tekle and Vermunt 2013). The
frozen reference gives each proband a posterior over the classes; a naive regression of the hard
label on the axis inherits the classification error of that assignment (the classify-analyse
bias), which attenuates a real slope and can manufacture a spurious one. The maximum-likelihood
(ML) correction removes it: the modal assignment is treated as a single categorical indicator
whose class-conditional error probabilities are fixed at the confusion matrix of the frozen
posteriors, and the structural model, a multinomial logit of the true latent class on the axis, is
fitted by expectation-maximisation with that measurement model held fixed. No mixture is refitted;
the correction reuses the same confusion matrix StepMix builds for its own three-step estimator
(``stepmix.stepmix.compute_bch_matrix``).

Because the structural model is a small logit rather than a weighted covariate emission, this
implements the correction directly rather than through StepMix's ``fit`` path, which always
re-estimates the measurement model in its first step and, on this cohort, is numerically unstable
under fractional weights (progress log, 2026-07-05). The naive hard-label regression is reported
beside the corrected one as a transparent, uncorrected cross-check.

Uncertainty is a family-clustered bootstrap resampling SPARK families, matching the
invariance-trajectory convention: the corrected slope, its odds ratio, and the predicted
proportion curve each carry a percentile interval, and the corrected slope carries a two-sided
add-one bootstrap $p$. The naive model additionally carries the closed-form Wald and
likelihood-ratio $p$-values. Significance is Benjamini-Hochberg controlled across the per-class
contrasts within an axis. Everything here is class or coefficient level; no per-proband quantity
is returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from analysis.invariance import benjamini_hochberg

# Convergence controls for the small expectation-maximisation loop of the ML correction and for
# the soft-target logit M-step. The structural models are low dimensional, so a tight tolerance
# and a modest iteration cap are ample.
_EM_MAX_ITER = 200
_EM_TOL = 1e-8
_RIDGE = 1e-6


def modal_assignment(responsibilities: np.ndarray) -> np.ndarray:
    """Return the hard (modal) class assignment from a posterior responsibility matrix.

    Parameters
    ----------
    responsibilities : numpy.ndarray
        Per-proband posterior over the classes, shape ``(n, K)``.

    Returns
    -------
    numpy.ndarray
        The index of the most probable class per proband, shape ``(n,)``.
    """
    return np.asarray(responsibilities).argmax(axis=1)


def bch_confusion(responsibilities: np.ndarray, assignment: str = "modal") -> np.ndarray:
    """Return the classification-error matrix of a frozen posterior.

    Wraps ``stepmix.stepmix.compute_bch_matrix``, the confusion matrix at the heart of the
    three-step corrections. The entry ``D[c, s]`` is the probability of assigning a proband to
    class ``s`` given that its true latent class is ``c``, estimated from the posterior itself.

    Parameters
    ----------
    responsibilities : numpy.ndarray
        Per-proband posterior over the classes, shape ``(n, K)``.
    assignment : {"modal", "soft"}, optional
        Whether the predicted class is the modal assignment or the soft posterior, matching
        StepMix's ``assignment`` option.

    Returns
    -------
    numpy.ndarray
        The confusion matrix ``D`` of shape ``(K, K)``, rows indexed by true class.
    """
    from stepmix.stepmix import compute_bch_matrix

    d, _d_inv = compute_bch_matrix(np.asarray(responsibilities, dtype=float), assignment)
    return np.asarray(d, dtype=float)


def _softmax(eta: np.ndarray) -> np.ndarray:
    """Return the row-wise softmax of a logit matrix, computed stably."""
    shifted = eta - eta.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class MultinomialFit:
    """A fitted multinomial logit with baseline-category coefficients.

    Attributes
    ----------
    coef : numpy.ndarray
        Coefficients of shape ``(p, K)`` with the baseline class column held at zero.
    loglik : float
        The model log-likelihood. For a naive fit this is the multinomial log-likelihood of the
        hard labels; for a corrected fit it is the observed-data log-likelihood of the ML
        correction.
    """

    coef: np.ndarray
    loglik: float

    def proportions(self, design: np.ndarray) -> np.ndarray:
        """Return the predicted class proportions for a design matrix.

        Parameters
        ----------
        design : numpy.ndarray
            Design matrix of shape ``(m, p)`` sharing the fit's columns.

        Returns
        -------
        numpy.ndarray
            Predicted proportions of shape ``(m, K)``, each row summing to one.
        """
        return _softmax(np.asarray(design, dtype=float) @ self.coef)


def fit_soft_multinomial(
    design: np.ndarray, targets: np.ndarray, *, ridge: float = _RIDGE
) -> np.ndarray:
    """Fit a multinomial logit to soft class targets.

    Minimises the cross-entropy between the softmax of ``design @ coef`` and the target rows,
    with the first class held as the baseline (its coefficient column fixed at zero) and a small
    ridge penalty for identifiability under quasi-separation. The targets are non-negative rows;
    they are the posterior responsibilities in the correction's M-step and one-hot label
    indicators for a naive fit.

    Parameters
    ----------
    design : numpy.ndarray
        Design matrix of shape ``(n, p)``, including an intercept column.
    targets : numpy.ndarray
        Non-negative class weights of shape ``(n, K)``.
    ridge : float, optional
        The L2 penalty on the free coefficients.

    Returns
    -------
    numpy.ndarray
        Coefficients of shape ``(p, K)`` with the baseline column at zero.
    """
    design = np.asarray(design, dtype=float)
    targets = np.asarray(targets, dtype=float)
    n, p = design.shape
    k = targets.shape[1]
    row_weight = targets.sum(axis=1)

    def unpack(theta: np.ndarray) -> np.ndarray:
        free = theta.reshape(p, k - 1)
        return np.column_stack([np.zeros(p), free])

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        coef = unpack(theta)
        eta = design @ coef
        proba = _softmax(eta)
        log_proba = np.log(np.clip(proba, 1e-300, None))
        loss = -float((targets * log_proba).sum()) + 0.5 * ridge * float((theta**2).sum())
        # Gradient of the cross-entropy wrt the free (non-baseline) coefficient columns.
        resid = proba * row_weight[:, None] - targets
        grad_full = design.T @ resid
        grad = grad_full[:, 1:].reshape(-1) + ridge * theta
        return loss, grad

    theta0 = np.zeros(p * (k - 1))
    result = minimize(objective, theta0, jac=True, method="L-BFGS-B")
    return unpack(result.x)


def _multinomial_loglik(coef: np.ndarray, design: np.ndarray, labels: np.ndarray) -> float:
    """Return the multinomial log-likelihood of hard labels under baseline-category coefficients."""
    proba = _softmax(design @ coef)
    rows = np.arange(labels.shape[0])
    return float(np.log(np.clip(proba[rows, labels], 1e-300, None)).sum())


def fit_naive_multinomial(
    design: np.ndarray, labels: np.ndarray, *, n_classes: int
) -> MultinomialFit:
    """Fit the naive multinomial logit of hard labels on a design (the uncorrected cross-check).

    Parameters
    ----------
    design : numpy.ndarray
        Design matrix of shape ``(n, p)``.
    labels : numpy.ndarray
        Hard class labels in ``0 .. n_classes - 1``, shape ``(n,)``.
    n_classes : int
        The number of classes ``K``.

    Returns
    -------
    MultinomialFit
        The fitted coefficients and multinomial log-likelihood.
    """
    design = np.asarray(design, dtype=float)
    labels = np.asarray(labels)
    targets = np.zeros((labels.shape[0], n_classes))
    targets[np.arange(labels.shape[0]), labels] = 1.0
    coef = fit_soft_multinomial(design, targets)
    return MultinomialFit(coef=coef, loglik=_multinomial_loglik(coef, design, labels))


def fit_corrected_multinomial(
    design: np.ndarray,
    responsibilities: np.ndarray,
    *,
    assignment: str = "modal",
    max_iter: int = _EM_MAX_ITER,
    tol: float = _EM_TOL,
) -> MultinomialFit:
    """Fit the ML-corrected multinomial logit of latent class on a design.

    The modal assignment is treated as a single categorical measurement with class-conditional
    error probabilities fixed at the confusion matrix of the frozen posteriors. The structural
    multinomial logit is fitted by expectation-maximisation with that measurement model held
    fixed, so the classify-analyse bias of the naive fit is removed without refitting the mixture.

    Parameters
    ----------
    design : numpy.ndarray
        Design matrix of shape ``(n, p)``, including an intercept column.
    responsibilities : numpy.ndarray
        Frozen per-proband posterior over the classes, shape ``(n, K)``.
    assignment : {"modal", "soft"}, optional
        The assignment rule for the confusion matrix.
    max_iter : int, optional
        The maximum number of expectation-maximisation iterations.
    tol : float, optional
        The absolute log-likelihood tolerance for convergence.

    Returns
    -------
    MultinomialFit
        The corrected coefficients and the observed-data log-likelihood of the correction model.
    """
    design = np.asarray(design, dtype=float)
    responsibilities = np.asarray(responsibilities, dtype=float)
    n, k = responsibilities.shape
    s = modal_assignment(responsibilities)
    d = bch_confusion(responsibilities, assignment)
    # The fixed emission: the likelihood of proband i's observed assignment given each true class.
    emission = np.clip(d[:, s].T, 1e-12, None)  # shape (n, K)

    coef = fit_soft_multinomial(design, responsibilities)
    prev_loglik = -np.inf
    for _ in range(max_iter):
        prior = _softmax(design @ coef)
        joint = prior * emission
        marginal = joint.sum(axis=1)
        loglik = float(np.log(np.clip(marginal, 1e-300, None)).sum())
        gamma = joint / marginal[:, None]
        coef = fit_soft_multinomial(design, gamma)
        if abs(loglik - prev_loglik) < tol:
            break
        prev_loglik = loglik
    prior = _softmax(design @ coef)
    marginal = (prior * emission).sum(axis=1)
    loglik = float(np.log(np.clip(marginal, 1e-300, None)).sum())
    return MultinomialFit(coef=coef, loglik=loglik)


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-eta))


def _fit_weighted_logit(
    design: np.ndarray, target: np.ndarray, *, ridge: float = _RIDGE
) -> np.ndarray:
    """Fit a binary logit to a fractional target by penalised iteratively reweighted least squares.

    The target may be a fractional posterior (the correction's M-step) or a zero/one indicator (a
    naive one-versus-rest fit). A small ridge keeps the solve finite under quasi-separation.

    Parameters
    ----------
    design : numpy.ndarray
        Design matrix of shape ``(n, p)`` including an intercept column.
    target : numpy.ndarray
        Fractional or binary target in ``[0, 1]``, shape ``(n,)``.
    ridge : float, optional
        The L2 penalty on the coefficients.

    Returns
    -------
    numpy.ndarray
        The fitted coefficients of shape ``(p,)``.
    """
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    p = design.shape[1]
    beta = np.zeros(p)
    penalty = ridge * np.eye(p)
    for _ in range(100):
        mu = _sigmoid(design @ beta)
        w = np.clip(mu * (1.0 - mu), 1e-9, None)
        z = design @ beta + (target - mu) / w
        wx = design * w[:, None]
        hessian = design.T @ wx + penalty
        rhs = design.T @ (w * z)
        new_beta = np.linalg.solve(hessian, rhs)
        if np.max(np.abs(new_beta - beta)) < 1e-10:
            beta = new_beta
            break
        beta = new_beta
    return beta


@dataclass
class SlopeResult:
    """A per-class one-versus-rest axis slope under one estimator.

    Attributes
    ----------
    ref_class : int
        The class index.
    slope : float
        The axis log-odds slope (log odds of membership per axis unit).
    odds_ratio : float
        ``exp(slope)``.
    ci_low, ci_high : float
        The slope confidence interval (family-clustered bootstrap percentiles).
    wald_p : float
        The closed-form Wald $p$-value on the slope (naive estimator only; not-a-number for the
        corrected estimator, which uses the bootstrap $p$).
    lrt_p : float
        The closed-form likelihood-ratio $p$-value for the axis term (naive estimator only).
    boot_p : float
        The two-sided add-one family-clustered-bootstrap $p$-value.
    reject : bool
        Whether the contrast survives Benjamini-Hochberg control within the axis.
    """

    ref_class: int
    slope: float
    odds_ratio: float
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    wald_p: float = float("nan")
    lrt_p: float = float("nan")
    boot_p: float = float("nan")
    reject: bool = False


def corrected_onevsrest(
    design: np.ndarray,
    responsibilities: np.ndarray,
    ref_class: int,
    axis_col: int,
    *,
    assignment: str = "modal",
    max_iter: int = _EM_MAX_ITER,
    tol: float = _EM_TOL,
) -> float:
    """Return the ML-corrected one-versus-rest axis slope for one class.

    Collapses the posterior to the indicator "belongs to ``ref_class``", forms the two-by-two
    confusion matrix of that binary assignment, and fits a binary logit of the latent indicator on
    the design by expectation-maximisation with the confusion matrix held fixed. The returned slope
    is the coefficient on the axis column.

    Parameters
    ----------
    design : numpy.ndarray
        Design matrix of shape ``(n, p)`` including an intercept column.
    responsibilities : numpy.ndarray
        Frozen posterior over the classes, shape ``(n, K)``.
    ref_class : int
        The class whose membership is modelled.
    axis_col : int
        The design column whose coefficient is the axis effect.
    assignment : {"modal", "soft"}, optional
        The assignment rule for the confusion matrix.
    max_iter : int, optional
        The maximum number of expectation-maximisation iterations.
    tol : float, optional
        The absolute log-likelihood tolerance for convergence.

    Returns
    -------
    float
        The corrected axis log-odds slope for the class.
    """
    design = np.asarray(design, dtype=float)
    responsibilities = np.asarray(responsibilities, dtype=float)
    resp_k = responsibilities[:, ref_class]
    soft2 = np.column_stack([resp_k, 1.0 - resp_k])
    s2 = soft2.argmax(axis=1)
    d2 = bch_confusion(soft2, assignment)
    emission = np.clip(d2[:, s2].T, 1e-12, None)  # (n, 2): columns [in class, not in class]

    beta = _fit_weighted_logit(design, resp_k)
    prev_loglik = -np.inf
    for _ in range(max_iter):
        prob_in = _sigmoid(design @ beta)
        joint_in = prob_in * emission[:, 0]
        joint_out = (1.0 - prob_in) * emission[:, 1]
        marginal = joint_in + joint_out
        loglik = float(np.log(np.clip(marginal, 1e-300, None)).sum())
        gamma = joint_in / marginal
        beta = _fit_weighted_logit(design, gamma)
        if abs(loglik - prev_loglik) < tol:
            break
        prev_loglik = loglik
    return float(beta[axis_col])


def naive_onevsrest(
    design: np.ndarray, labels: np.ndarray, ref_class: int, axis_col: int
) -> SlopeResult:
    """Return the naive one-versus-rest axis slope for one class, with Wald and LRT tests.

    A binary logit of the hard-label indicator on the design, fitted by statsmodels so the Wald
    standard error and the likelihood-ratio test against the axis-free model are exact. It is
    uncorrected for classification error and reported as a cross-check.

    Parameters
    ----------
    design : numpy.ndarray
        Design matrix of shape ``(n, p)`` including an intercept column.
    labels : numpy.ndarray
        Hard class labels, shape ``(n,)``.
    ref_class : int
        The class whose membership is modelled.
    axis_col : int
        The design column whose coefficient is the axis effect.

    Returns
    -------
    SlopeResult
        The slope, odds ratio, Wald $p$, and likelihood-ratio $p$ (bootstrap fields left unset).
    """
    import statsmodels.api as sm
    from scipy import stats

    design = np.asarray(design, dtype=float)
    indicator = (np.asarray(labels) == ref_class).astype(float)
    full = sm.Logit(indicator, design).fit(disp=0, maxiter=200)
    slope = float(full.params[axis_col])
    wald_p = float(full.pvalues[axis_col])

    reduced_cols = [c for c in range(design.shape[1]) if c != axis_col]
    reduced = sm.Logit(indicator, design[:, reduced_cols]).fit(disp=0, maxiter=200)
    lr_stat = 2.0 * (full.llf - reduced.llf)
    lrt_p = float(stats.chi2.sf(max(lr_stat, 0.0), df=1))
    return SlopeResult(
        ref_class=ref_class,
        slope=slope,
        odds_ratio=float(np.exp(slope)),
        wald_p=wald_p,
        lrt_p=lrt_p,
    )


@dataclass
class ProportionCurve:
    """Predicted class-proportion curves over an axis grid, with bootstrap bands.

    Attributes
    ----------
    positions : numpy.ndarray
        The axis grid, shape ``(G,)``.
    corrected, naive : numpy.ndarray
        Predicted proportions of shape ``(K, G)`` under each estimator.
    band_lo, band_hi : numpy.ndarray
        The family-clustered bootstrap percentile band on the corrected curve, shape ``(K, G)``.
    pooled : numpy.ndarray
        The pooled (axis-free) class proportion, the mean responsibility per class over the
        cohort, shape ``(K,)``. This is the original mixing weight the curve trends away from.
    """

    positions: np.ndarray
    corrected: np.ndarray
    naive: np.ndarray
    band_lo: np.ndarray
    band_hi: np.ndarray
    pooled: np.ndarray


@dataclass
class JointTest:
    """A joint likelihood-ratio test of ``class ~ axis`` against ``class ~ 1``.

    Attributes
    ----------
    estimator : str
        ``"corrected"`` or ``"naive"``.
    lr_stat : float
        The likelihood-ratio statistic.
    df : int
        The degrees of freedom, ``(K - 1)`` times the number of axis terms.
    p_value : float
        The chi-square tail probability.
    """

    estimator: str
    lr_stat: float
    df: int
    p_value: float


@dataclass
class PrevalenceResult:
    """The full PREV read for one axis.

    Attributes
    ----------
    axis : str
        The axis name.
    n : int
        The number of probands with a finite axis value.
    axis_mean : float
        The centring constant subtracted from the axis before fitting (slopes are per raw unit).
    corrected_slopes, naive_slopes : list of SlopeResult
        The per-class one-versus-rest axis slopes under each estimator (unadjusted).
    adjusted_slopes : list of SlopeResult
        The corrected axis slopes net of sex, the diagnosis-to-measurement lag, and age at
        evaluation.
    joint_tests : list of JointTest
        The joint likelihood-ratio tests under each estimator.
    curve : ProportionCurve
        The predicted proportion curves and their band.
    dsm_contrasts : list of SlopeResult
        For the era axis, the per-class pre/post-2013 (DSM-5) log-odds contrasts; empty otherwise.
    """

    axis: str
    n: int
    axis_mean: float
    corrected_slopes: list[SlopeResult]
    naive_slopes: list[SlopeResult]
    adjusted_slopes: list[SlopeResult]
    joint_tests: list[JointTest]
    curve: ProportionCurve
    dsm_contrasts: list[SlopeResult] = field(default_factory=list)


def _design(columns: list[np.ndarray]) -> np.ndarray:
    """Stack an intercept column and the given columns into a design matrix."""
    n = columns[0].shape[0]
    return np.column_stack([np.ones(n), *columns])


def _joint_lrt(
    full: MultinomialFit, null: MultinomialFit, *, n_classes: int, n_axis_terms: int, estimator: str
) -> JointTest:
    from scipy import stats

    df = (n_classes - 1) * n_axis_terms
    lr = 2.0 * (full.loglik - null.loglik)
    return JointTest(
        estimator=estimator,
        lr_stat=float(lr),
        df=df,
        p_value=float(stats.chi2.sf(max(lr, 0.0), df=df)),
    )


DSM5_YEAR = 2013.0


def prevalence_analysis(
    responsibilities: np.ndarray,
    labels: np.ndarray,
    axis_values: np.ndarray,
    families: np.ndarray,
    *,
    axis: str,
    covariates: dict[str, np.ndarray] | None = None,
    grid: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
    q: float = 0.05,
    assignment: str = "modal",
) -> PrevalenceResult:
    """Run the PREV prevalence-drift analysis for one axis.

    Fits the corrected and naive per-class one-versus-rest slopes, the joint likelihood-ratio
    tests, the predicted proportion curves, the family-clustered bootstrap uncertainty, and (for
    era) the DSM-5 pre/post-2013 contrast. The axis is centred before fitting for numerical
    conditioning; the reported slopes are per raw axis unit. Significance is Benjamini-Hochberg
    controlled across the per-class contrasts within the axis, separately for each estimator.

    Parameters
    ----------
    responsibilities : numpy.ndarray
        Frozen posterior over the classes, shape ``(n, K)``.
    labels : numpy.ndarray
        Hard class labels, shape ``(n,)``.
    axis_values : numpy.ndarray
        The axis value per proband, shape ``(n,)``; must be finite (filter upstream).
    families : numpy.ndarray
        The per-proband family identifier, the bootstrap clustering unit, shape ``(n,)``.
    axis : str
        The axis name, used to decide whether the DSM-5 contrast is read.
    covariates : dict of str to numpy.ndarray, optional
        The adjustment covariates (``sex``, ``lag``, ``age_at_eval``) for the sensitivity model.
    grid : numpy.ndarray
        The axis grid the proportion curves are read at, shape ``(G,)``.
    n_boot : int, optional
        The number of family-clustered bootstrap replicates.
    seed : int, optional
        The base seed for the bootstrap.
    q : float, optional
        The Benjamini-Hochberg level.
    assignment : {"modal", "soft"}, optional
        The assignment rule for the confusion matrices.

    Returns
    -------
    PrevalenceResult
        The per-class slopes, joint tests, proportion curves, and DSM contrasts for the axis.
    """
    responsibilities = np.asarray(responsibilities, dtype=float)
    labels = np.asarray(labels)
    axis_values = np.asarray(axis_values, dtype=float)
    families = np.asarray(families)
    n, n_classes = responsibilities.shape

    axis_mean = float(axis_values.mean())
    centred = axis_values - axis_mean
    design = _design([centred])
    grid_design = _design([np.asarray(grid, dtype=float) - axis_mean])

    # Observed unadjusted slopes under both estimators.
    corrected_slopes = [
        SlopeResult(
            ref_class=c,
            slope=(
                sl := corrected_onevsrest(design, responsibilities, c, 1, assignment=assignment)
            ),
            odds_ratio=float(np.exp(sl)),
        )
        for c in range(n_classes)
    ]
    naive_slopes = [naive_onevsrest(design, labels, c, 1) for c in range(n_classes)]

    # Joint likelihood-ratio tests, class ~ axis against class ~ 1.
    null_design = np.ones((n, 1))
    corrected_full = fit_corrected_multinomial(design, responsibilities, assignment=assignment)
    corrected_null = fit_corrected_multinomial(null_design, responsibilities, assignment=assignment)
    naive_full = fit_naive_multinomial(design, labels, n_classes=n_classes)
    naive_null = fit_naive_multinomial(null_design, labels, n_classes=n_classes)
    joint_tests = [
        _joint_lrt(
            corrected_full,
            corrected_null,
            n_classes=n_classes,
            n_axis_terms=1,
            estimator="corrected",
        ),
        _joint_lrt(naive_full, naive_null, n_classes=n_classes, n_axis_terms=1, estimator="naive"),
    ]

    # Predicted proportion curves.
    corrected_curve = corrected_full.proportions(grid_design).T  # (K, G)
    naive_curve = naive_full.proportions(grid_design).T

    # Adjusted sensitivity: corrected axis slope net of sex, lag, and age at evaluation.
    adjusted_slopes: list[SlopeResult] = []
    if covariates is not None:
        cov_cols = [
            covariates["sex"] - float(np.mean(covariates["sex"])),
            covariates["lag"] - float(np.mean(covariates["lag"])),
            covariates["age_at_eval"] - float(np.mean(covariates["age_at_eval"])),
        ]
        adj_design = _design([centred, *cov_cols])
        for c in range(n_classes):
            sl = corrected_onevsrest(adj_design, responsibilities, c, 1, assignment=assignment)
            adjusted_slopes.append(SlopeResult(ref_class=c, slope=sl, odds_ratio=float(np.exp(sl))))

    # DSM-5 pre/post-2013 contrast for the era axis.
    dsm_contrasts: list[SlopeResult] = []
    if axis == "era":
        post = (axis_values >= DSM5_YEAR).astype(float)
        dsm_design = _design([post])
        for c in range(n_classes):
            corrected = corrected_onevsrest(
                dsm_design, responsibilities, c, 1, assignment=assignment
            )
            naive = naive_onevsrest(dsm_design, labels, c, 1)
            dsm_contrasts.append(
                SlopeResult(
                    ref_class=c,
                    slope=corrected,
                    odds_ratio=float(np.exp(corrected)),
                    wald_p=naive.wald_p,
                    lrt_p=naive.lrt_p,
                )
            )

    # Family-clustered bootstrap: corrected slopes, adjusted slopes, DSM contrasts, and the curve.
    band_lo, band_hi = _bootstrap_prevalence(
        responsibilities=responsibilities,
        axis_values=axis_values,
        families=families,
        design_builder=lambda idx: _design([axis_values[idx] - axis_mean]),
        grid_design=grid_design,
        corrected_slopes=corrected_slopes,
        adjusted_slopes=adjusted_slopes,
        dsm_contrasts=dsm_contrasts,
        corrected_curve=corrected_curve,
        covariates=covariates,
        axis=axis,
        axis_mean=axis_mean,
        n_boot=n_boot,
        seed=seed,
        assignment=assignment,
    )

    # Benjamini-Hochberg within the axis, per estimator.
    _apply_fdr(corrected_slopes, q, use_boot=True)
    _apply_fdr(naive_slopes, q, use_boot=False)
    if adjusted_slopes:
        _apply_fdr(adjusted_slopes, q, use_boot=True)
    if dsm_contrasts:
        _apply_fdr(dsm_contrasts, q, use_boot=True)

    curve = ProportionCurve(
        positions=np.asarray(grid, dtype=float),
        corrected=corrected_curve,
        naive=naive_curve,
        band_lo=band_lo,
        band_hi=band_hi,
        pooled=responsibilities.mean(axis=0),
    )
    return PrevalenceResult(
        axis=axis,
        n=n,
        axis_mean=axis_mean,
        corrected_slopes=corrected_slopes,
        naive_slopes=naive_slopes,
        adjusted_slopes=adjusted_slopes,
        joint_tests=joint_tests,
        curve=curve,
        dsm_contrasts=dsm_contrasts,
    )


def _two_sided_boot_p(draws: np.ndarray) -> float:
    """Return the two-sided add-one bootstrap $p$-value that a statistic differs from zero."""
    draws = draws[np.isfinite(draws)]
    n = draws.shape[0]
    if n == 0:
        return float("nan")
    frac_positive = float(np.mean(draws > 0.0))
    tail = min(frac_positive, 1.0 - frac_positive)
    return float(np.clip(2.0 * tail, 1.0 / (n + 1), 1.0))


def _apply_fdr(slopes: list[SlopeResult], q: float, *, use_boot: bool) -> None:
    """Fill the ``reject`` field of each slope by Benjamini-Hochberg control within the axis."""
    p = np.array([s.boot_p if use_boot else s.wald_p for s in slopes], dtype=float)
    reject = benjamini_hochberg(p, q)
    for s, r in zip(slopes, reject, strict=True):
        s.reject = bool(r)


def _bootstrap_prevalence(
    *,
    responsibilities: np.ndarray,
    axis_values: np.ndarray,
    families: np.ndarray,
    design_builder,
    grid_design: np.ndarray,
    corrected_slopes: list[SlopeResult],
    adjusted_slopes: list[SlopeResult],
    dsm_contrasts: list[SlopeResult],
    corrected_curve: np.ndarray,
    covariates: dict[str, np.ndarray] | None,
    axis: str,
    axis_mean: float,
    n_boot: int,
    seed: int,
    assignment: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the family-clustered bootstrap, fill the slope fields in place, and return the band.

    Each replicate resamples whole families with replacement, rebuilds the design and the frozen
    posteriors on the resample, and re-estimates the corrected slopes, the adjusted slopes, the
    DSM contrasts, and the predicted proportion curve. The percentiles of the draws are the
    intervals and band, and each slope's two-sided add-one bootstrap $p$ is its significance. The
    slope intervals and $p$-values are filled in place; the proportion-curve band is returned.
    """
    from analysis.trajectory_local import _family_rows

    n_classes = responsibilities.shape[1]
    groups, _ = _family_rows(families)
    n_groups = len(groups)
    rng = np.random.default_rng(seed)

    corrected_draws = np.full((n_boot, n_classes), np.nan)
    adjusted_draws = np.full((n_boot, n_classes), np.nan) if adjusted_slopes else None
    dsm_draws = np.full((n_boot, n_classes), np.nan) if dsm_contrasts else None
    curve_draws = np.full((n_boot, *corrected_curve.shape), np.nan)

    for b in range(n_boot):
        chosen = rng.integers(0, n_groups, size=n_groups)
        rows = np.concatenate([groups[c] for c in chosen])
        resp_b = responsibilities[rows]
        design_b = design_builder(rows)
        try:
            for c in range(n_classes):
                corrected_draws[b, c] = corrected_onevsrest(
                    design_b, resp_b, c, 1, assignment=assignment
                )
            full_b = fit_corrected_multinomial(design_b, resp_b, assignment=assignment)
            curve_draws[b] = full_b.proportions(grid_design).T
            if adjusted_draws is not None and covariates is not None:
                cov_b = [
                    covariates["sex"][rows] - float(np.mean(covariates["sex"][rows])),
                    covariates["lag"][rows] - float(np.mean(covariates["lag"][rows])),
                    covariates["age_at_eval"][rows]
                    - float(np.mean(covariates["age_at_eval"][rows])),
                ]
                adj_design_b = _design([axis_values[rows] - axis_mean, *cov_b])
                for c in range(n_classes):
                    adjusted_draws[b, c] = corrected_onevsrest(
                        adj_design_b, resp_b, c, 1, assignment=assignment
                    )
            if dsm_draws is not None:
                post_b = (axis_values[rows] >= DSM5_YEAR).astype(float)
                dsm_design_b = _design([post_b])
                for c in range(n_classes):
                    dsm_draws[b, c] = corrected_onevsrest(
                        dsm_design_b, resp_b, c, 1, assignment=assignment
                    )
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            # A degenerate resample (a rank-deficient design, an all-one-class family draw) is
            # dropped to not-a-number and skipped, mirroring the drift stage's degeneracy guard.
            continue

    _fill_intervals(corrected_slopes, corrected_draws)
    if adjusted_draws is not None:
        _fill_intervals(adjusted_slopes, adjusted_draws)
    if dsm_draws is not None:
        _fill_intervals(dsm_contrasts, dsm_draws)
    return (
        np.nanpercentile(curve_draws, 2.5, axis=0),
        np.nanpercentile(curve_draws, 97.5, axis=0),
    )


def _fill_intervals(slopes: list[SlopeResult], draws: np.ndarray) -> None:
    """Fill each slope's bootstrap CI and two-sided $p$ from the per-class draw columns."""
    for c, s in enumerate(slopes):
        column = draws[:, c]
        s.ci_low = float(np.nanpercentile(column, 2.5))
        s.ci_high = float(np.nanpercentile(column, 97.5))
        s.boot_p = _two_sided_boot_p(column)
