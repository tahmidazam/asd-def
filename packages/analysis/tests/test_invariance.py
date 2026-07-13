"""Correctness gates for the score-based measurement-invariance test.

Three gates, all on synthetic data (governance: no participant data in tests):

1. score validation: the analytic casewise scores match a central finite difference of the
   per-sample log-likelihood to machine precision, for every emission type;
2. power: a synthetic mixture with a planted mean drift along the axis yields a small bridge
   $p$-value in the drifted block and localises the break near the planted point;
3. size: with no drift the bridge $p$-values are approximately uniform, so the analytic null is
   calibrated rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analysis import invariance
from analysis.features import Typing
from stepmix.stepmix import StepMix
from stepmix.utils import get_mixed_descriptor


def _mixed_fit(seed: int, *, n: int = 600, n_init: int = 4):
    """Fit a small mixed-emission StepMix on well-separated synthetic classes.

    Returns the fitted model, the descriptor-ordered measurement matrix, and the typing, the
    three inputs the score extraction needs.
    """
    rng = np.random.default_rng(seed)
    k = 3
    z = rng.integers(0, k, n)
    cont = np.empty((n, 4))
    for c in range(k):
        cont[z == c] = rng.normal(2.5 * c, 1.0, (int((z == c).sum()), 4))
    binary = np.empty((n, 3), dtype=int)
    for c in range(k):
        p = np.clip(0.15 + 0.3 * c, 0.02, 0.98)
        binary[z == c] = (rng.uniform(size=(int((z == c).sum()), 3)) < p).astype(int)
    categ = np.empty((n, 3), dtype=int)
    for j in range(3):
        for c in range(k):
            probs = rng.dirichlet(np.ones(4))
            categ[z == c, j] = rng.choice(4, size=int((z == c).sum()), p=probs)
    cont_cols = [f"c{i}" for i in range(4)]
    bin_cols = [f"b{i}" for i in range(3)]
    cat_cols = [f"k{i}" for i in range(3)]
    frame = pd.DataFrame(np.hstack([cont, binary, categ]), columns=cont_cols + bin_cols + cat_cols)
    data, descriptor = get_mixed_descriptor(
        frame, continuous=cont_cols, binary=bin_cols, categorical=cat_cols
    )
    model = StepMix(
        n_components=k,
        measurement=descriptor,
        n_steps=1,
        n_init=n_init,
        random_state=seed,
        progress_bar=0,
        verbose=0,
    )
    model.fit(data)
    typing = Typing(continuous=cont_cols, binary=bin_cols, categorical=cat_cols)
    return model, data, typing


# ---------------------------------------------------------------------------------------------
# Gate 1: score validation (the load-bearing check).
# ---------------------------------------------------------------------------------------------


def test_analytic_scores_match_finite_difference():
    """Every focal parameter's analytic score equals its finite-difference score to 1e-6."""
    model, data, typing = _mixed_fit(0)
    scores = invariance.casewise_scores(model, data, typing)

    # At least one focal parameter of each emission type is present and checked.
    kinds = {p.kind for p in scores.parameters}
    assert kinds == {"gaussian_mean", "bernoulli", "multinomial_logit"}

    worst = {kind: 0.0 for kind in kinds}
    for col, param in enumerate(scores.parameters):
        fd = invariance.numerical_score(model, data, param)
        err = float(np.max(np.abs(scores.values[:, col] - fd)))
        worst[param.kind] = max(worst[param.kind], err)
    for kind, err in worst.items():
        assert err < 1e-6, f"{kind} analytic score deviates from finite difference by {err:.2e}"


def test_full_sample_score_is_near_zero():
    """The full-sample score is pinned near zero, the property that makes B(t) a bridge."""
    model, data, typing = _mixed_fit(1)
    scores = invariance.casewise_scores(model, data, typing)
    column_sums = np.abs(scores.values.sum(axis=0))
    # The optimiser stops at a tolerance, so the sum is small but not exactly zero.
    assert float(column_sums.max()) < 1e-2


def test_responsibilities_match_stepmix():
    """The responsibilities used by the scores match StepMix ``predict_proba``."""
    model, data, typing = _mixed_fit(2)
    scores = invariance.casewise_scores(model, data, typing)
    assert np.allclose(scores.responsibilities, model.predict_proba(data), atol=1e-10)


def test_measurement_only_is_required():
    """A covariate fit is refused rather than scored with the wrong estimand."""

    class _WithStructural:
        _sm = object()

    with pytest.raises(ValueError, match="measurement-only"):
        invariance.responsibilities(_WithStructural(), np.zeros((2, 2)))


# ---------------------------------------------------------------------------------------------
# Gate 2: power on a planted drift.
# ---------------------------------------------------------------------------------------------


def _drift_fit(seed: int, *, n: int = 1600, break_at: float = 0.5, delta: float = 3.0):
    """Fit a pooled mixture on data whose one class drifts in mean past a break in the axis.

    The axis is a sorted uniform variable. One latent class shifts a feature mean by ``delta``
    for probands past ``break_at``, so a single pooled fit violates measurement invariance in
    that class at that point. Returns the model, measurement matrix, typing, and axis series.
    """
    rng = np.random.default_rng(seed)
    k = 3
    axis = np.sort(rng.uniform(0.0, 1.0, n))
    z = rng.integers(0, k, n)
    drift_class = 1
    means = np.array([0.0, 4.0, 8.0])
    cont = np.empty((n, 3))
    base = rng.normal(0.0, 1.0, (n, 3))
    for i in range(n):
        cont[i] = means[z[i]] + base[i]
    # Plant the drift: class 1's feature 0 mean jumps by delta past the break.
    late = (axis >= break_at) & (z == drift_class)
    cont[late, 0] += delta
    frame = pd.DataFrame(cont, columns=["c0", "c1", "c2"])
    data, descriptor = get_mixed_descriptor(frame, continuous=["c0", "c1", "c2"])
    model = StepMix(
        n_components=k,
        measurement=descriptor,
        n_steps=1,
        n_init=6,
        random_state=seed,
        progress_bar=0,
        verbose=0,
    )
    model.fit(data)
    typing = Typing(continuous=["c0", "c1", "c2"], binary=[], categorical=[])
    axis_series = pd.Series(axis, index=data.index, name="axis")
    return model, data, typing, axis_series


def test_planted_drift_is_detected_and_localised():
    """A planted mean drift yields a small bridge p and a break near the planted point."""
    model, data, typing, axis = _drift_fit(0)
    result = invariance.run_invariance(
        model, data, typing, axis, axis="synthetic", by_category=False, n_sim=1000, seed=0
    )
    best = min(result.blocks, key=lambda b: b.p_max_lm)
    assert best.p_max_lm < 0.05, f"planted drift not detected; min p={best.p_max_lm:.3f}"
    assert 0.4 <= best.break_position <= 0.6, f"break mislocalised at {best.break_position:.3f}"


def test_no_drift_block_is_not_flagged_on_average():
    """The drifted class is the most significant, not an unrelated one."""
    model, data, typing, axis = _drift_fit(3)
    result = invariance.run_invariance(
        model, data, typing, axis, axis="synthetic", by_category=False, n_sim=1000, seed=1
    )
    ranked = sorted(result.blocks, key=lambda b: b.p_max_lm)
    # The strongest signal clears the threshold; the weakest does not blow up the same way.
    assert ranked[0].p_max_lm < 0.05
    assert ranked[0].max_lm > ranked[-1].max_lm


# ---------------------------------------------------------------------------------------------
# Gate 3: size calibration.
# ---------------------------------------------------------------------------------------------


def _null_fit(seed: int, *, n: int = 500):
    """Fit a pooled mixture on drift-free mixed data, with a random axis unrelated to the fit."""
    rng = np.random.default_rng(seed)
    k = 2
    z = rng.integers(0, k, n)
    means = np.array([0.0, 5.0])
    cont = np.empty((n, 3))
    for i in range(n):
        cont[i] = means[z[i]] + rng.normal(0.0, 1.0, 3)
    frame = pd.DataFrame(cont, columns=["c0", "c1", "c2"])
    data, descriptor = get_mixed_descriptor(frame, continuous=["c0", "c1", "c2"])
    model = StepMix(
        n_components=k,
        measurement=descriptor,
        n_steps=1,
        n_init=3,
        random_state=seed,
        progress_bar=0,
        verbose=0,
    )
    model.fit(data)
    typing = Typing(continuous=["c0", "c1", "c2"], binary=[], categorical=[])
    axis = pd.Series(rng.uniform(0.0, 1.0, n), index=data.index, name="axis")
    return model, data, typing, axis


@pytest.mark.slow
def test_size_calibration_is_approximately_uniform():
    """Over many no-drift datasets the bridge p-values are close to uniform (size near nominal)."""
    reps = 150
    p_values: list[float] = []
    for rep in range(reps):
        model, data, typing, axis = _null_fit(1000 + rep)
        result = invariance.run_invariance(
            model, data, typing, axis, axis="null", by_category=False, n_sim=400, seed=rep
        )
        p_values.extend(b.p_max_lm for b in result.blocks)
    fraction = float(np.mean(np.array(p_values) < 0.05))
    # With ~300 null p-values the standard error of the rejection fraction is about 0.013.
    assert 0.015 <= fraction <= 0.10, f"size miscalibrated: fraction p<0.05 = {fraction:.3f}"
