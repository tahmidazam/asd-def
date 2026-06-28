"""Phase 3: the pure timing-variable derivations behind the strata-describe stage.

The IO build (reading SPARK tables) is exercised on real data by the CLI stage; here the
arithmetic and the implausible-value cleaning are pinned on small synthetic frames.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from analysis.strata_data import derive_axes, summarise


def _inputs():
    # diagnosis_age in months; registration anchors and age at eval in years.
    diagnosis_age = pd.Series([24.0, 60.0, 120.0, 36.0])  # 2, 5, 10, 3 years
    registration_year = pd.Series([2018.0, 2020.0, 2022.0, 2019.0])
    age_at_registration = pd.Series([4.0, 7.0, 12.0, 1.0])
    age_at_eval = pd.Series([5.0, 8.0, 13.0, 4.0])
    return diagnosis_age, registration_year, age_at_registration, age_at_eval


def test_axes_and_lag_arithmetic() -> None:
    axes, lag = derive_axes(*_inputs())
    # age at diagnosis = months / 12
    assert list(axes["age_at_diagnosis_years"]) == [2.0, 5.0, 10.0, 3.0]
    # diagnosis year = registration_year - age_at_registration + age_at_diagnosis
    assert axes["diagnosis_year"].iloc[0] == 2018.0 - 4.0 + 2.0
    # lag = age_at_eval - age_at_diagnosis
    assert list(lag) == [3.0, 3.0, 3.0, 1.0]


def test_diagnosis_after_evaluation_drops_only_the_lag() -> None:
    # Diagnosed at 10 but evaluated at 5: the age at diagnosis is a valid recalled value and
    # is kept, as is the era; only the lag (clearly negative) drops to missing.
    diagnosis_age = pd.Series([120.0])
    axes, lag = derive_axes(diagnosis_age, pd.Series([2018.0]), pd.Series([5.0]), pd.Series([5.0]))
    assert axes["age_at_diagnosis_years"].iloc[0] == 10.0
    assert axes["diagnosis_year"].iloc[0] == 2018.0 - 5.0 + 10.0
    assert pd.isna(lag.iloc[0])


def test_rounding_band_negative_lag_is_kept() -> None:
    # age_at_eval is floored to an integer, so a lag within one year of zero is a rounding
    # artefact and is kept (here diagnosed at 6.5, evaluated at a floored 6).
    diagnosis_age = pd.Series([78.0])  # 6.5 years
    _axes, lag = derive_axes(diagnosis_age, pd.Series([2021.0]), pd.Series([6.0]), pd.Series([6.0]))
    assert lag.iloc[0] == -0.5


def test_zero_lag_is_kept() -> None:
    # Diagnosed and evaluated at the same age is valid: lag is exactly zero, not dropped.
    diagnosis_age = pd.Series([72.0])  # 6 years
    axes, lag = derive_axes(diagnosis_age, pd.Series([2021.0]), pd.Series([6.0]), pd.Series([6.0]))
    assert axes["age_at_diagnosis_years"].iloc[0] == 6.0
    assert lag.iloc[0] == 0.0


def test_implausible_year_is_dropped() -> None:
    # A broken registration anchor reconstructs a year far in the past.
    diagnosis_age = pd.Series([24.0])
    axes, _ = derive_axes(diagnosis_age, pd.Series([1900.0]), pd.Series([2.0]), pd.Series([3.0]))
    assert pd.isna(axes["diagnosis_year"].iloc[0])


def test_summarise_reports_coverage_and_quantiles() -> None:
    axes, lag = derive_axes(*_inputs())
    axes.loc[1, "age_at_diagnosis_years"] = np.nan
    summary = summarise(axes, lag, n_total=4)
    assert summary["n_total"] == 4
    age = cast("dict[str, object]", summary["age_at_diagnosis_years"])
    lag_summary = cast("dict[str, object]", summary["lag_years"])
    assert age["n_missing"] == 1
    assert age["coverage"] == 0.75
    assert "p50" in lag_summary
