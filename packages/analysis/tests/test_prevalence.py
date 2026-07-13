"""Correctness gates for the prevalence-drift test (PREV, plan sections 3 and 12b).

All on synthetic data (governance: no participant data in tests). The gates:

1. the confusion matrix of a perfect posterior is the identity, and a noisy one is diagonally
   dominant, so :func:`analysis.prevalence.bch_confusion` behaves;
2. the soft-target multinomial logit recovers planted coefficients on clean one-hot data;
3. a planted proportion trend is recovered with the right sign and rejects under the FDR;
4. the three-step correction reduces the classify-analyse bias: with a known classification
   error and a known slope, the corrected slope de-attenuates the naive hard-label slope and
   carries a smaller bias on average;
5. a flat proportion yields a null: the corrected slopes cover zero and the FDR rejection rate
   across seeds stays near nominal.
"""

from __future__ import annotations

import numpy as np
import pytest
from analysis import prevalence as prev


def _onehot(idx: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((idx.shape[0], k))
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


def _noisy_posterior(
    rng: np.random.Generator, true: np.ndarray, k: int, kappa: float
) -> np.ndarray:
    """Return a self-consistent noisy posterior over the true classes.

    The responsibility of proband ``i`` is a softmax of ``kappa`` times the one-hot true class
    plus standard Gaussian noise, so the modal assignment carries genuine classification error
    (lower ``kappa`` is more error) and the error is independent of any axis, exactly the setting
    the three-step correction assumes. The confusion matrix
    (:func:`analysis.prevalence.bch_confusion`) recovered from these posteriors then matches the
    error process that generated the modal labels.
    """
    logits = kappa * _onehot(true, k) + rng.normal(0.0, 1.0, size=(true.shape[0], k))
    resp = np.exp(logits - logits.max(axis=1, keepdims=True))
    resp /= resp.sum(axis=1, keepdims=True)
    return resp


def _planted_responsibilities(
    rng: np.random.Generator,
    *,
    n: int,
    slopes: np.ndarray,
    kappa: float,
    axis_lo: float = 2000.0,
    axis_hi: float = 2025.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a cohort with a planted per-class proportion trend and classification error.

    The true latent class follows a multinomial logit of the (centred) axis with the given
    baseline-category ``slopes`` (class zero is the baseline, slope zero). The posterior is the
    self-consistent noisy posterior of :func:`_noisy_posterior`, so the modal assignment carries
    genuine error at concentration ``kappa``. Returns the responsibilities, the modal labels, the
    axis, and singleton family ids.
    """
    k = slopes.shape[0]
    axis = rng.uniform(axis_lo, axis_hi, size=n)
    centred = axis - axis.mean()
    eta = np.outer(centred, slopes)  # (n, K), column 0 is the baseline
    proba = np.exp(eta - eta.max(axis=1, keepdims=True))
    proba /= proba.sum(axis=1, keepdims=True)
    true = np.array([rng.choice(k, p=proba[i]) for i in range(n)])
    resp = _noisy_posterior(rng, true, k, kappa)
    labels = resp.argmax(axis=1)
    families = np.arange(n)
    return resp, labels, axis, families


# ---------------------------------------------------------------------------------------------
# Gate 1: the confusion matrix.
# ---------------------------------------------------------------------------------------------
def test_confusion_of_perfect_posterior_is_identity() -> None:
    """A one-hot posterior has no classification error, so its confusion matrix is the identity."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=500)
    resp = np.zeros((500, 4))
    resp[np.arange(500), labels] = 1.0
    d = prev.bch_confusion(resp)
    assert np.allclose(d, np.eye(4), atol=1e-8)


def test_confusion_of_noisy_posterior_is_diagonally_dominant() -> None:
    """A confident but imperfect posterior gives a diagonally dominant confusion matrix."""
    rng = np.random.default_rng(1)
    resp, _labels, _axis, _fam = _planted_responsibilities(
        rng, n=2000, slopes=np.zeros(4), kappa=2.0
    )
    d = prev.bch_confusion(resp)
    assert np.all(np.diag(d) > np.max(d - np.eye(4) * d, axis=1))
    assert np.allclose(d.sum(axis=1), 1.0, atol=1e-8)


# ---------------------------------------------------------------------------------------------
# Gate 2: the soft-target multinomial logit.
# ---------------------------------------------------------------------------------------------
def test_soft_multinomial_recovers_planted_coefficients() -> None:
    """On clean one-hot targets the soft-target logit recovers the data-generating slopes."""
    rng = np.random.default_rng(2)
    n, k = 8000, 3
    x = rng.uniform(-2.0, 2.0, size=n)
    design = np.column_stack([np.ones(n), x])
    true_slopes = np.array([0.0, 1.2, -0.8])
    true_intercepts = np.array([0.0, 0.3, -0.5])
    eta = design @ np.vstack([true_intercepts, true_slopes])
    proba = np.exp(eta - eta.max(axis=1, keepdims=True))
    proba /= proba.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(k, p=proba[i]) for i in range(n)])
    targets = np.zeros((n, k))
    targets[np.arange(n), labels] = 1.0

    coef = prev.fit_soft_multinomial(design, targets)
    assert coef[:, 0] == pytest.approx(np.zeros(2))  # baseline held at zero
    assert coef[1, 1] == pytest.approx(1.2, abs=0.1)
    assert coef[1, 2] == pytest.approx(-0.8, abs=0.1)


# ---------------------------------------------------------------------------------------------
# Gate 3: a planted trend is recovered with the right sign and rejects.
# ---------------------------------------------------------------------------------------------
def test_planted_trend_recovered_and_rejects() -> None:
    """A planted proportion trend is recovered with the right sign and survives the FDR."""
    rng = np.random.default_rng(3)
    # Class 1 rises with the axis, class 2 falls; classes 0 and 3 are flat. Slopes are per year of
    # the 25-year axis, so a swing of about two log-odds across the span is a clear trend.
    slopes = np.array([0.0, 0.08, -0.06, 0.0])
    resp, labels, axis, families = _planted_responsibilities(rng, n=6000, slopes=slopes, kappa=2.0)
    grid = np.linspace(axis.min(), axis.max(), 8)
    result = prev.prevalence_analysis(
        resp, labels, axis, families, axis="era", grid=grid, n_boot=80, seed=1
    )

    corrected = result.corrected_slopes
    assert corrected[1].slope > 0 and corrected[1].reject
    assert corrected[2].slope < 0 and corrected[2].reject
    # The joint test rejects a constant-proportion null decisively.
    joint = {t.estimator: t for t in result.joint_tests}
    assert joint["corrected"].p_value < 1e-6
    # The era read carries a DSM-5 contrast with the same signs as the trend.
    assert result.dsm_contrasts[1].slope > 0
    assert result.dsm_contrasts[2].slope < 0


# ---------------------------------------------------------------------------------------------
# Gate 4: the three-step correction reduces the classify-analyse bias.
# ---------------------------------------------------------------------------------------------
def test_correction_reduces_classify_analyse_bias() -> None:
    """With a known slope and classification error, the correction beats the naive hard-label fit.

    A two-class latent with a single planted log-odds slope. The naive regression on the
    error-laden modal label attenuates the slope toward zero; the ML correction, fixing the
    confusion matrix of the posteriors, recovers a slope closer to the truth. Averaged over
    seeds so the comparison is not a single-draw accident.
    """
    true_slope = 0.9  # the true one-versus-rest log-odds slope of class 0 on the axis
    naive_gaps, corrected_gaps, deattenuated = [], [], []
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        n = 6000
        axis = rng.uniform(-2.0, 2.0, size=n)
        # True binary latent: class 0 membership rises with the axis at the known slope.
        true_class = np.where(rng.uniform(size=n) < prev._sigmoid(true_slope * axis), 0, 1)
        # A self-consistent noisy posterior at concentration 1.0 (about a quarter of the modal
        # labels are wrong), so the modal labels carry genuine, axis-independent error.
        resp = _noisy_posterior(rng, true_class, 2, kappa=1.0)
        labels = resp.argmax(axis=1)
        design = np.column_stack([np.ones(n), axis])

        corrected = prev.corrected_onevsrest(design, resp, 0, 1)
        naive = prev.naive_onevsrest(design, labels, 0, 1).slope
        naive_gaps.append(abs(naive - true_slope))
        corrected_gaps.append(abs(corrected - true_slope))
        # The naive slope is attenuated toward zero; the correction de-attenuates it.
        deattenuated.append(abs(naive) < true_slope and abs(corrected) > abs(naive))

    # Every replicate de-attenuates, and the corrected slope's bias is smaller on average.
    assert all(deattenuated)
    assert np.mean(corrected_gaps) < np.mean(naive_gaps)


# ---------------------------------------------------------------------------------------------
# Gate 5: a flat proportion yields a null with near-nominal FDR.
# ---------------------------------------------------------------------------------------------
def test_flat_proportion_is_null_with_nominal_fdr() -> None:
    """A flat proportion is a null: the slopes cover zero and the FDR stays near nominal."""
    n_seeds, k = 10, 3
    rejections, covers = 0, 0
    total = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(200 + seed)
        resp, labels, axis, families = _planted_responsibilities(
            rng, n=1500, slopes=np.zeros(k), kappa=2.0
        )
        grid = np.linspace(axis.min(), axis.max(), 6)
        result = prev.prevalence_analysis(
            resp, labels, axis, families, axis="age_at_diagnosis", grid=grid, n_boot=40, seed=seed
        )
        for s in result.corrected_slopes:
            total += 1
            rejections += int(s.reject)
            covers += int(s.ci_low <= 0.0 <= s.ci_high)

    # Under the null the false-discovery rate sits near nominal (allowing Monte-Carlo slack), and
    # the great majority of the bootstrap intervals cover zero.
    assert rejections / total <= 0.15
    assert covers / total >= 0.8
