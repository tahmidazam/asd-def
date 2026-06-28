"""Phase 3: the acceptance requirements a binning policy is judged against.

The evaluator is exercised on synthetic cohorts, so the tests need no real data and no
model fit. They check the Tier 1 gates pass or fail as designed, the Tier 2 confound and
balance checks flag the right partitions, and the Tier 3 demographic table is produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from analysis.requirements import (
    DEFAULT_THRESHOLDS,
    RequirementThresholds,
    evaluate_policy,
)
from analysis.strata import FixedBands, QuantileBins


def _result(report, key):
    return next(r for r in report.results if r.key == key)


def test_balanced_policy_passes_every_tier1_gate() -> None:
    rng = np.random.default_rng(0)
    variable = pd.Series(rng.uniform(2, 16, size=8000))
    report = evaluate_policy(FixedBands(edges=(6, 10)), variable)
    assert report.eligible
    assert all(
        _result(report, key).status == "pass"
        for key in ("min_bin_size", "smallest_class", "coverage", "partition_validity")
    )


def test_undersized_bin_fails_the_size_gate() -> None:
    # 3000 probands cut so the top band holds far fewer than 1000.
    variable = pd.Series([5.0] * 2700 + [20.0] * 300)
    report = evaluate_policy(FixedBands(edges=(10,)), variable)
    assert not report.eligible
    assert _result(report, "min_bin_size").status == "fail"


def test_smallest_class_gate_is_stricter_than_raw_size() -> None:
    # A bin of 1100 clears the 1000 size floor but not a projected smallest class of 200.
    variable = pd.Series([3.0] * 5000 + [12.0] * 1100)
    thresholds = RequirementThresholds(min_projected_smallest_class=200)
    report = evaluate_policy(FixedBands(edges=(8,)), variable, thresholds=thresholds)
    assert _result(report, "min_bin_size").status == "pass"
    assert _result(report, "smallest_class").status == "fail"
    assert not report.eligible


def test_high_missingness_fails_coverage() -> None:
    variable = pd.Series([5.0] * 4000 + [np.nan] * 4000)
    report = evaluate_policy(FixedBands(edges=(4,)), variable)
    assert _result(report, "coverage").status == "fail"
    assert not report.eligible


def test_dominant_bin_is_flagged_for_balance() -> None:
    variable = pd.Series([5.0] * 9000 + [20.0] * 1100)
    report = evaluate_policy(FixedBands(edges=(10,)), variable)
    assert _result(report, "size_balance").status == "flag"
    assert "size_balance" in report.flags


def test_lag_checks_flag_entanglement_and_thin_subsample() -> None:
    rng = np.random.default_rng(1)
    n = 8000
    era = pd.Series(rng.uniform(2008, 2024, size=n))
    # Lag mechanically anti-correlated with era, so the small-lag subsample is all recent.
    lag = pd.Series(2024.0 - era + rng.normal(0, 0.3, size=n))
    report = evaluate_policy(FixedBands(edges=(2013, 2018)), era, lag=lag)
    assert _result(report, "lag_correlation").status == "flag"
    assert _result(report, "small_lag_retention").status == "flag"


def test_lag_checks_pass_when_independent() -> None:
    rng = np.random.default_rng(2)
    n = 9000
    variable = pd.Series(rng.uniform(2, 16, size=n))
    lag = pd.Series(rng.uniform(0, 1.5, size=n))  # everyone measured soon after diagnosis
    report = evaluate_policy(FixedBands(edges=(6, 10)), variable, lag=lag)
    assert _result(report, "lag_correlation").status == "ok"
    assert _result(report, "small_lag_retention").status == "ok"


def test_covariate_balance_and_demographics_table() -> None:
    variable = pd.Series([4.0] * 4000 + [12.0] * 4000)
    # Sex skews hard across the two bins; age is flat.
    sex = pd.Series([1.0] * 3800 + [0.0] * 200 + [1.0] * 2000 + [0.0] * 2000)
    age = pd.Series(np.concatenate([np.full(4000, 9.0), np.full(4000, 9.0)]))
    covariates = pd.DataFrame({"sex_male": sex, "age_at_eval_years": age})
    report = evaluate_policy(FixedBands(edges=(8,)), variable, covariates=covariates)
    assert _result(report, "covariate_balance").status == "flag"
    assert report.demographics is not None
    assert "smd_extreme" in report.demographics.columns
    assert set(report.demographics.index) == {"sex_male", "age_at_eval_years"}


def test_edge_robustness_flags_an_edge_inside_a_cluster() -> None:
    # A dense cluster straddling the edge means a +/-1 shift moves many probands.
    rng = np.random.default_rng(3)
    cluster = rng.normal(10.0, 0.5, size=6000)
    tail = rng.uniform(13, 18, size=2000)
    variable = pd.Series(np.concatenate([cluster, tail]))
    report = evaluate_policy(FixedBands(edges=(10,)), variable)
    assert _result(report, "edge_robustness").status == "flag"


def test_quantile_policy_is_evaluable_and_reports_spec() -> None:
    rng = np.random.default_rng(4)
    variable = pd.Series(rng.uniform(2, 16, size=8000))
    report = evaluate_policy(QuantileBins(q=4), variable)
    assert report.spec["policy"] == "quantile"
    assert report.eligible
    assert not report.to_frame().empty


def test_default_thresholds_use_the_phase2_floor() -> None:
    assert DEFAULT_THRESHOLDS.min_bin_size == 1000
    assert DEFAULT_THRESHOLDS.smallest_class_fraction == 0.15
