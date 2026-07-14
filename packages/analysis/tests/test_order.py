"""Tests for the number-of-classes search (H0C), on synthetic data only.

The correctness gates the design fixes (plan section 7): a four-class synthetic keeps four
classes, a five-class synthetic search finds five, the observed and null datasets go through
the identical fitting recipe, a stratum matching the pooled order is not flagged even when the
pooled order sits above four, and numerically degenerate bootstrap draws are dropped and
counted rather than crashing. The heavy end-to-end fits carry the ``slow`` marker; the decision
logic, the corroborators (Kneedle, VLMR), and the bootstrap $p$-value are exercised fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analysis import order
from stepmix.utils import get_mixed_descriptor


def _synthetic(n: int, k: int, seed: int) -> pd.DataFrame:
    """Return a well-separated ``k``-class mixture of continuous and binary features."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, 3.0, size=(k, 6))
    probs = rng.uniform(0.1, 0.9, size=(k, 3))
    labels = rng.integers(0, k, size=n)
    continuous = centres[labels] + rng.normal(0.0, 1.0, size=(n, 6))
    binary = (rng.uniform(size=(n, 3)) < probs[labels]).astype(float)
    columns = [f"c{i}" for i in range(6)] + [f"b{i}" for i in range(3)]
    return pd.DataFrame(np.hstack([continuous, binary]), columns=columns)


def _descriptor(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    return get_mixed_descriptor(
        dataframe=frame,
        continuous=[c for c in frame.columns if c.startswith("c")],
        binary=[c for c in frame.columns if c.startswith("b")],
    )


def _in_process_dispatch(descriptor: dict, columns, recipe: order.Recipe):
    """A dispatcher that runs the real bootstrap draws in-process (no pool)."""
    columns_list = [str(c) for c in columns]

    def dispatch(k_null, null_params, n, start, count):
        return [
            order.bootstrap_lr(
                null_params,
                descriptor,
                columns_list,
                int(n),
                int(k_null),
                n_init=recipe.n_init,
                n_random=recipe.n_random,
                seed=9000 + 1000 * int(k_null) + start + offset,
                jitter=recipe.jitter,
            )
            for offset in range(count)
        ]

    return dispatch


# --- fast unit tests: corroborators and the bootstrap p-value -------------------------------


def test_phipson_smyth_add_one_and_drops_none() -> None:
    """The add-one p-value ignores degenerate (``None``) draws and never returns zero."""
    # observed above every finite draw: smallest attainable p is 1 / (B + 1).
    p = order.phipson_smyth_p(100.0, [1.0, 2.0, None, 3.0, None])
    assert p == pytest.approx(1 / (3 + 1))
    # observed below every draw: p is one.
    assert order.phipson_smyth_p(0.0, [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    # no finite draws leaves the p-value undefined.
    assert np.isnan(order.phipson_smyth_p(1.0, [None, None]))


def test_kneedle_finds_the_elbow_of_a_concave_curve() -> None:
    """The knee of a diminishing-returns curve sits where the gain flattens."""
    # A curve rising fast to K=4 then nearly flat: the knee is four.
    scores = {2: 0.0, 3: 6.0, 4: 9.0, 5: 9.5, 6: 9.7, 7: 9.8}
    knee = order.kneedle_knee(sorted(scores), [scores[k] for k in sorted(scores)])
    assert knee == 4


def test_kneedle_handles_flat_and_short_curves() -> None:
    """A flat curve falls back to the first candidate; a single point returns itself."""
    assert order.kneedle_knee([2, 3, 4], [5.0, 5.0, 5.0]) == 2
    assert order.kneedle_knee([3], [1.0]) == 3


def test_vlmr_matches_the_tidylpa_reference_arithmetic() -> None:
    """The adjusted LMR reproduces ``calc_lrt`` (Lo, Mendell and Rubin 2001, Formula 15)."""
    # tidyLPA example: calc_lrt(150, -741.02, 8, 1, -488.91, 13, 2).
    out = order.vlmr_test(-741.02, 8, 1, -488.91, 13, 2, 150)
    lr = 2 * (-488.91 - -741.02)
    correction = 1.0 + 1.0 / (((3 * 2 - 1) - (3 * 1 - 1)) * np.log(150))
    assert out["lr"] == pytest.approx(lr)
    assert out["lmr"] == pytest.approx(lr / correction)
    assert out["df"] == pytest.approx(13 - 8)
    assert 0.0 <= out["p"] <= 1.0


# --- gate (d): pooled-order-matching strata are never flagged --------------------------------


def _result(supported_k: int, elbow: int, vlmr_p: float) -> order.SearchResult:
    return order.SearchResult(
        supported_k=supported_k,
        capped=False,
        direction="split" if supported_k > 4 else "stable",
        steps=[],
        elbow_knee=elbow,
        vlmr={"p": vlmr_p},
    )


def test_stratum_matching_pooled_order_is_not_flagged_even_above_four() -> None:
    """A stratum whose order equals the pooled order is not an order change (gate d)."""
    # Pooled over-extracts to five; a stratum also at five is stable, never flagged.
    result = _result(supported_k=5, elbow=5, vlmr_p=0.001)
    assert order.agreement_flag(result, pooled_supported_k=5) is False


def test_order_change_needs_agreement_of_all_three() -> None:
    """A positive order change requires the BLRT, the elbow, and the VLMR to agree."""
    changed = _result(supported_k=5, elbow=5, vlmr_p=0.001)
    assert order.agreement_flag(changed, pooled_supported_k=4) is True
    # Elbow did not move off the pooled order: no agreement.
    elbow_stuck = _result(supported_k=5, elbow=4, vlmr_p=0.001)
    assert order.agreement_flag(elbow_stuck, pooled_supported_k=4) is False
    # VLMR does not agree: no agreement.
    vlmr_ns = _result(supported_k=5, elbow=5, vlmr_p=0.40)
    assert order.agreement_flag(vlmr_ns, pooled_supported_k=4) is False


# --- the sequential decision tree, with a mocked recipe (fast) -------------------------------


@pytest.fixture(scope="module")
def _base_model():
    """A real, tiny fitted model, so canned fit pairs expose valid StepMix parameters."""
    frame = _synthetic(200, 2, 0)
    measurement_data, descriptor = _descriptor(frame)
    return order.fit_k(measurement_data, descriptor, 2, n_init=1, seed=0)


def _mock_fit_pair(lr_by_k: dict[int, float], base_model):
    """Return a ``fit_pair`` replacement whose observed LR is scripted per null class count."""

    def fake(measurement_data, descriptor, k, *, n_init, n_random, seed, jitter):
        target = lr_by_k.get(k, 1.0)
        return order.FitPair(
            model_k=base_model, model_k1=base_model, ll_k=0.0, ll_k1=target / 2, k=k
        )

    return fake


def _constant_dispatch(null_level: float = 10.0):
    """A dispatcher returning a constant null LR, so rejection is set by the observed LR."""

    def dispatch(k_null, null_params, n, start, count):
        return [null_level] * count

    return dispatch


@pytest.mark.parametrize(
    ("lr_by_k", "expected_k", "expected_dir", "expected_capped"),
    [
        # 4-vs-5 fails, 3-vs-4 rejects: the anchor stands.
        ({4: 1.0, 3: 100.0}, 4, "stable", False),
        # 4-vs-5 rejects, 5-vs-6 fails: five is supported.
        ({4: 100.0, 5: 1.0}, 5, "split", False),
        # every upward step rejects to the cap: reported as ">=7".
        ({4: 100.0, 5: 100.0, 6: 100.0}, 7, "split", True),
        # 4-vs-5 fails and every merge fails down to the floor: two.
        ({4: 1.0, 3: 1.0, 2: 1.0}, 2, "merge", False),
    ],
)
def test_sequential_search_decision_tree(
    monkeypatch, _base_model, lr_by_k, expected_k, expected_dir, expected_capped
) -> None:
    """The anchored search steps up while rejecting and down while not, to the caps."""
    monkeypatch.setattr(order, "fit_pair", _mock_fit_pair(lr_by_k, _base_model))
    frame = _synthetic(120, 2, 1)
    measurement_data, descriptor = _descriptor(frame)
    result = order.sequential_search(
        measurement_data,
        descriptor,
        dispatch=_constant_dispatch(10.0),
        recipe=order.Recipe(n_init=1, n_random=1),
        schedule=order.Schedule(b_screen=19, b_escalate=19, alpha=0.05),
        anchor=4,
        cap=7,
        floor=2,
        seed=0,
    )
    assert result.supported_k == expected_k
    assert result.direction == expected_dir
    assert result.capped == expected_capped


# --- gate (e): degenerate draws are dropped and counted --------------------------------------


def test_blrt_step_drops_and_counts_degenerate_draws(monkeypatch, _base_model) -> None:
    """A step tolerates ``None`` draws: it counts them and keeps the finite ones (gate e)."""
    monkeypatch.setattr(order, "fit_pair", _mock_fit_pair({4: 100.0}, _base_model))
    frame = _synthetic(120, 2, 2)
    measurement_data, descriptor = _descriptor(frame)

    def flaky_dispatch(k_null, null_params, n, start, count):
        # Half the draws come back degenerate.
        return [None if (start + i) % 2 else 1.0 for i in range(count)]

    step, _pair = order.blrt_step(
        measurement_data,
        descriptor,
        4,
        direction="split",
        dispatch=flaky_dispatch,
        recipe=order.Recipe(n_init=1, n_random=1),
        schedule=order.Schedule(b_screen=8, b_escalate=8, alpha=0.05),
        seed=0,
    )
    assert step.n_dropped == 4
    assert step.b_used == 4
    assert np.isfinite(step.p_value)


# --- gate (c): observed and null share the identical fitting recipe --------------------------


@pytest.mark.slow
def test_observed_and_null_use_the_identical_fit_path(monkeypatch) -> None:
    """Every fit, observed or bootstrap, goes through ``fit_pair`` with the same recipe (gate c)."""
    frame = _synthetic(400, 3, 3)
    measurement_data, descriptor = _descriptor(frame)
    recipe = order.Recipe(n_init=2, n_random=1)

    calls: list[dict] = []
    original = order.fit_pair

    def spy(md, desc, k, *, n_init, n_random, seed, jitter):
        calls.append({"k": k, "n_init": n_init, "n_random": n_random, "jitter": jitter})
        return original(md, desc, k, n_init=n_init, n_random=n_random, seed=seed, jitter=jitter)

    monkeypatch.setattr(order, "fit_pair", spy)
    dispatch = _in_process_dispatch(descriptor, measurement_data.columns, recipe)
    order.blrt_step(
        measurement_data,
        descriptor,
        3,
        direction="split",
        dispatch=dispatch,
        recipe=recipe,
        schedule=order.Schedule(b_screen=3, b_escalate=3, alpha=0.05),
        seed=0,
    )
    # One observed fit plus one per bootstrap draw, all at the same null class count and recipe.
    assert len(calls) == 1 + 3
    assert all(c["k"] == 3 for c in calls)
    assert all(
        (c["n_init"], c["n_random"], c["jitter"]) == (recipe.n_init, recipe.n_random, recipe.jitter)
        for c in calls
    )


# --- gates (a) and (b): four-class stays, five-class is found (end to end) --------------------


@pytest.mark.slow
def test_four_class_synthetic_retains_four() -> None:
    """A genuine four-class dataset does not split: the supported order is four (gate a)."""
    frame = _synthetic(600, 4, 4)
    measurement_data, descriptor = _descriptor(frame)
    recipe = order.Recipe(n_init=3, n_random=1)
    dispatch = _in_process_dispatch(descriptor, measurement_data.columns, recipe)
    result = order.sequential_search(
        measurement_data,
        descriptor,
        dispatch=dispatch,
        recipe=recipe,
        schedule=order.Schedule(b_screen=19, b_escalate=19, alpha=0.05),
        anchor=4,
        cap=7,
        floor=2,
        seed=0,
    )
    assert result.supported_k == 4
    # The primary splitting step does not reject.
    split_step = next(s for s in result.steps if s.k_null == 4 and s.direction == "split")
    assert not split_step.rejected


@pytest.mark.slow
def test_five_class_synthetic_finds_five() -> None:
    """A genuine five-class dataset is recovered: the search steps up to five (gate b)."""
    frame = _synthetic(750, 5, 5)
    measurement_data, descriptor = _descriptor(frame)
    recipe = order.Recipe(n_init=3, n_random=2)
    dispatch = _in_process_dispatch(descriptor, measurement_data.columns, recipe)
    result = order.sequential_search(
        measurement_data,
        descriptor,
        dispatch=dispatch,
        recipe=recipe,
        schedule=order.Schedule(b_screen=19, b_escalate=19, alpha=0.05),
        anchor=4,
        cap=7,
        floor=2,
        seed=0,
    )
    assert result.supported_k == 5
    assert result.direction == "split"


def test_split_warm_params_adds_a_class_and_keeps_valid_weights(_base_model) -> None:
    """Splitting a class yields one more class whose mixing weights still sum to one."""
    params = order.split_warm_params(_base_model, 2, 0, np.random.default_rng(0), jitter=0.3)
    weights = np.asarray(params["weights"], dtype=float)
    assert weights.shape == (3,)
    assert weights.sum() == pytest.approx(1.0)
