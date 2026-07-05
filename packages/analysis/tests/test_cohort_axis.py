"""The cross-cohort stratification-axis provider (plan section 8), on synthetic data only.

Each backend resolves a named axis to a per-proband variable and its binning policy. The two
cognitive axes (``cognitive_impairment``, ``iq``) are shared; the two timing axes
(``age_at_diagnosis``, ``era``) are SPARK-only. The backends read SFARI data, so they are never
instantiated here: the dispatch is exercised on a bypass instance (``object.__new__``) whose
single-column read is stubbed, so no catalogue or CSV is involved. The cross-cohort claim under
test is that both cohorts label the intellectual-disability split with the same band names,
though the impaired band sits low on the SSC IQ score and high on the SPARK impairment flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from analysis import config
from analysis.cohort.spark import SparkCohort
from analysis.cohort.ssc import SscCohort
from analysis.strata import BinningPolicy, FixedBands, MaxEqualBins, StratumAssignment


def _spark(column: pd.Series) -> SparkCohort:
    cohort = object.__new__(SparkCohort)
    cohort._read_axis_column = lambda table, name, index: column.reindex(index)  # type: ignore[method-assign]
    return cohort


def _ssc(column: pd.Series) -> SscCohort:
    cohort = object.__new__(SscCohort)
    cohort._read_axis_column = lambda table, name, index: column.reindex(index)  # type: ignore[method-assign]
    return cohort


def _resolve(
    cohort: SparkCohort | SscCohort, name: str, index: pd.Index, min_bin_size: int = 1000
) -> tuple[pd.Series, BinningPolicy]:
    resolved = cohort.axis(name, index, pd.DataFrame(index=index), min_bin_size)
    assert resolved is not None
    return resolved


def test_spark_cognitive_impairment_is_a_binary_id_dichotomy() -> None:
    index = pd.Index([1, 2, 3, 4])
    flag = pd.Series([0.0, 1.0, 1.0, np.nan], index=index)
    values, policy = _resolve(_spark(flag), "cognitive_impairment", index)
    assert isinstance(policy, FixedBands)
    assignment: StratumAssignment = policy.assign(values)
    # A 0/1 flag: value 1 is impaired ("id"), value 0 is not ("no_id"), NaN unassigned.
    assert assignment.counts == {"no_id": 1, "id": 2}
    assert assignment.n_missing == 1


def test_ssc_cognitive_impairment_dichotomises_fsiq_at_the_id_threshold() -> None:
    index = pd.Index(["a", "b", "c"])
    fsiq = pd.Series([65.0, 80.0, 95.0], index=index)
    values, policy = _resolve(_ssc(fsiq), "cognitive_impairment", index)
    assignment = policy.assign(values)
    # Below the threshold is "id"; the threshold itself falls on the "no_id" side (left-closed).
    assert list(assignment.codes) == ["id", "no_id", "no_id"]
    assert policy.spec()["edges"] == [float(config.ID_IQ_THRESHOLD)]


def test_both_cohorts_use_the_same_id_band_names() -> None:
    index = pd.Index([1])
    _, spark_policy = _resolve(_spark(pd.Series([1.0], index=index)), "cognitive_impairment", index)
    _, ssc_policy = _resolve(_ssc(pd.Series([60.0], index=index)), "cognitive_impairment", index)
    assert isinstance(spark_policy, FixedBands)
    assert isinstance(ssc_policy, FixedBands)
    assert spark_policy.labels is not None and ssc_policy.labels is not None
    assert set(spark_policy.labels) == set(ssc_policy.labels) == {"id", "no_id"}


def test_iq_axis_is_continuous_equal_frequency_on_both_cohorts() -> None:
    index = pd.RangeIndex(12)
    scores = pd.Series(np.arange(12.0), index=index)
    _, spark_policy = _resolve(_spark(scores), "iq", index, min_bin_size=3)
    _, ssc_policy = _resolve(_ssc(scores), "iq", index, min_bin_size=3)
    assert isinstance(spark_policy, MaxEqualBins)
    assert isinstance(ssc_policy, MaxEqualBins)


def test_ssc_returns_none_for_the_spark_only_timing_axes() -> None:
    index = pd.Index(["a"])
    covariates = pd.DataFrame(index=index)
    cohort = _ssc(pd.Series([90.0], index=index))
    assert cohort.axis("age_at_diagnosis", index, covariates) is None
    assert cohort.axis("era", index, covariates) is None


def test_unknown_axis_returns_none() -> None:
    index = pd.Index([1])
    covariates = pd.DataFrame(index=index)
    assert _spark(pd.Series([1.0], index=index)).axis("nonsense", index, covariates) is None
    assert _ssc(pd.Series([1.0], index=index)).axis("nonsense", index, covariates) is None
