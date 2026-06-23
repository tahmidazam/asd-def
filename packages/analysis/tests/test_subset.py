"""Unit tests for the V9 records cutoff, on synthetic data only.

The SPARK backend reads SFARI data, so it is never instantiated here. The records-cutoff
gates are pure given the cutoff year and a registration frame, so they are exercised on a
bypass instance (``object.__new__``) with synthetic frames, no catalogue or CSV involved.
"""

from __future__ import annotations

import pandas as pd
from analysis.cohort.spark import SparkCohort


def _gated_cohort(cutoff_year: int | None, registration: pd.DataFrame | None = None) -> SparkCohort:
    """Return a SparkCohort with only the cutoff state set, bypassing the CSV-reading init."""
    cohort = object.__new__(SparkCohort)
    cohort._cutoff_year = cutoff_year
    cohort._registration = registration
    return cohort


def test_eval_year_gate_keeps_on_or_before_cutoff_and_drops_missing() -> None:
    cohort = _gated_cohort(2022)
    df = pd.DataFrame(
        {
            "eval_year": [2020, 2022, 2023, None],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    gated = cohort._gate_eval_year(df)
    # 2020 and 2022 clear; 2023 is after the freeze; a missing year cannot be confirmed.
    assert list(gated["value"]) == [1.0, 2.0]
    # The eval_year column is consumed once it has done its work.
    assert "eval_year" not in gated.columns


def test_eval_year_gate_is_identity_without_a_cutoff() -> None:
    cohort = _gated_cohort(None)
    df = pd.DataFrame({"eval_year": [2024], "value": [1.0]})
    assert cohort._gate_eval_year(df).equals(df)


def test_roster_gate_keeps_registered_by_cutoff_year() -> None:
    registration = pd.DataFrame(
        {"registration_year": [2019, 2022, 2024], "age_at_registration_years": [5.0, 6.0, 7.0]},
        index=pd.Index(["a", "b", "c"], name="subject_sp_id"),
    )
    cohort = _gated_cohort(2022, registration)
    complete = pd.DataFrame({"feature": [1.0, 1.0, 1.0]}, index=["a", "b", "c"])
    kept = cohort._apply_roster_gate(complete)
    # The 2024 registrant was not in the v9 roster.
    assert list(kept.index) == ["a", "b"]


def test_cbcl_derived_year_gate() -> None:
    registration = pd.DataFrame(
        {"registration_year": [2021, 2021, 2020], "age_at_registration_years": [5.0, 5.0, 4.0]},
        index=pd.Index(["early", "late", "missing_anchor"], name="subject_sp_id"),
    )
    # "early": completed ~2021 + 1.0 = 2022.0 (clears). "late": 2021 + 2.5 = 2023.5 (after).
    # A proband with no registration anchor is unreconstructable and drops.
    age_at_eval = pd.Series(
        {"early": 6.0, "late": 7.5, "no_registration": 6.0}, name="age_at_eval_years"
    )
    keep = SparkCohort._cbcl_within_cutoff(age_at_eval, registration, 2022)
    assert list(keep) == ["early"]


def test_cbcl_derived_year_keeps_mid_year_completion() -> None:
    # floor(2022.5) = 2022 <= cutoff, so a mid-2022 completion is kept, not rounded up.
    registration = pd.DataFrame(
        {"registration_year": [2021], "age_at_registration_years": [5.0]},
        index=pd.Index(["mid"], name="subject_sp_id"),
    )
    age_at_eval = pd.Series({"mid": 6.5}, name="age_at_eval_years")
    keep = SparkCohort._cbcl_within_cutoff(age_at_eval, registration, 2022)
    assert list(keep) == ["mid"]
