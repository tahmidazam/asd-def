"""Construct the stratification variables and the demographics for the cohort.

The phase-3 feasibility stage needs the two stratifying axes, the measurement-to-diagnosis
lag, and a covariate frame, all built for the probands in the modelling cohort and on its
index. This module reads only the timing and demographic columns each needs through the
``dscat`` catalogue (never a whole instrument file) and derives:

- age at diagnosis in years, from ``core_descriptive_variables.diagnosis_age`` (months);
- the calendar year of diagnosis, the registration-anchor reconstruction
  ``registration_year - age_at_registration_years + age_at_diagnosis`` (plan section 5);
- the lag in years, ``age_at_eval_years - age_at_diagnosis``, the section 7a confound;
- a numeric demographics frame (sex, age at evaluation, a cognitive-impairment proxy,
  Hispanic ethnicity, and the race indicators) for the per-bin table and the balance check.

Implausible values are set to missing before any policy is evaluated, and counted in the
diagnostics so the cleaning is visible. The age and era axes are gated only on their own
validity (a childhood age at diagnosis, a reconstructed year in the cohort window), not on
evaluation timing, since age at diagnosis is recalled. Only the lag is gated on the
evaluation comparison. Nothing here fits a model: this is the design-side characterisation
the binning policy is judged on (plan section 12 firewall).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analysis.cohort import INDEX_COLUMN, open_catalogue, read_columns, source_csv

# Reconstruction window for the derived diagnosis year. SPARK enrolment is recent, so a
# reconstructed year far outside this range signals a bad anchor rather than an early
# diagnosis, and is dropped.
_YEAR_FLOOR = 1985
_YEAR_CEILING = 2026

# Plausible range for age at diagnosis in years (a childhood diagnosis in this 4-18 cohort).
_AGE_CEILING = 18

# Floor on the lag. ``age_at_eval_years`` is recorded as an integer (floored), so a lag down
# to about minus one year is a rounding artefact, not a diagnosis genuinely postdating
# evaluation. Only larger negative lags are dropped as inconsistent.
_LAG_FLOOR = -1.0

# Instruments that carry a native completion year, used to check the subtests were measured
# close together and close to diagnosis (the section 7a contemporaneity question). CBCL 6-18
# carries no ``eval_year`` and is left out.
_DATED_INSTRUMENTS = ("scq", "rbsr", "background_history_child")

_RACE_INDICATORS = (
    "race_white",
    "race_african_amer",
    "race_asian",
    "race_other",
    "race_more_than_one_calc",
)


@dataclass
class StrataData:
    """The stratification variables, lag, and demographics for the cohort.

    Attributes
    ----------
    axes : pandas.DataFrame
        Columns ``age_at_diagnosis_years`` and ``diagnosis_year``, cleaned of implausible
        values, indexed by proband.
    lag : pandas.Series
        Measurement-to-diagnosis lag in years, on the same index.
    demographics : pandas.DataFrame
        Numeric covariates (sex, age at evaluation, cognitive-impairment proxy, Hispanic
        ethnicity, race indicators) for the demographic table and balance check.
    instrument_years : pandas.DataFrame
        Per-instrument completion year for the dated instruments, for the contemporaneity
        summary.
    diagnostics : dict
        Per-variable counts (missing, implausible) and quantiles, plus the contemporaneity
        summary, all recorded in the manifest.
    """

    axes: pd.DataFrame
    lag: pd.Series
    demographics: pd.DataFrame
    instrument_years: pd.DataFrame
    diagnostics: dict[str, object]


def derive_axes(
    diagnosis_age_months: pd.Series,
    registration_year: pd.Series,
    age_at_registration_years: pd.Series,
    age_at_eval_years: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Derive the two axes and the lag from the raw timing fields.

    Age at diagnosis is a parent-recalled value, valid in the childhood range and independent
    of when the instruments were completed, so the two axes are gated only on their own
    validity. The lag alone depends on the evaluation timing. Implausible values set to
    missing: an age at diagnosis below zero or above the childhood ceiling; a reconstructed
    year outside the cohort window; and a lag more negative than the one-year rounding
    tolerance (a diagnosis genuinely postdating evaluation).
    """
    age_at_diagnosis = diagnosis_age_months / 12.0
    diagnosis_year = registration_year - age_at_registration_years + age_at_diagnosis
    lag = age_at_eval_years - age_at_diagnosis

    age_invalid = (age_at_diagnosis < 0) | (age_at_diagnosis > _AGE_CEILING)
    age_at_diagnosis = age_at_diagnosis.mask(age_invalid)
    diagnosis_year = diagnosis_year.mask(
        age_invalid | (diagnosis_year < _YEAR_FLOOR) | (diagnosis_year > _YEAR_CEILING)
    )
    lag = lag.mask(age_invalid | (lag < _LAG_FLOOR))

    axes = pd.DataFrame(
        {"age_at_diagnosis_years": age_at_diagnosis, "diagnosis_year": diagnosis_year}
    )
    return axes, lag


def _quantiles(series: pd.Series) -> dict[str, float]:
    valid = series.dropna()
    if valid.empty:
        return {}
    qs = valid.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {f"p{int(p * 100)}": float(qs.loc[p]) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}


def summarise(axes: pd.DataFrame, lag: pd.Series, n_total: int) -> dict[str, object]:
    """Summarise the missingness, implausible drops, and spread of each variable."""
    summary: dict[str, object] = {"n_total": n_total}
    for name, series in (
        ("age_at_diagnosis_years", axes["age_at_diagnosis_years"]),
        ("diagnosis_year", axes["diagnosis_year"]),
        ("lag_years", lag),
    ):
        summary[name] = {
            "n_present": int(series.notna().sum()),
            "n_missing": int(series.isna().sum()),
            "coverage": float(series.notna().mean()),
            **_quantiles(series),
        }
    return summary


def _contemporaneity(
    instrument_years: pd.DataFrame, diagnosis_year: pd.Series
) -> dict[str, object]:
    """Summarise how close the dated subtests sit to each other and to diagnosis."""
    spread = instrument_years.max(axis=1) - instrument_years.min(axis=1)
    calendar_lag = instrument_years.mean(axis=1) - diagnosis_year
    return {
        "median_subtest_spread_years": float(spread.median()),
        "frac_subtests_over_1yr_apart": float((spread > 1).mean()),
        "median_eval_minus_diagnosis_years": float(calendar_lag.median()),
        "p90_eval_minus_diagnosis_years": float(calendar_lag.quantile(0.9)),
    }


def build_strata_data(
    root: Path,
    version: str,
    index: pd.Index,
    age_at_eval: pd.Series,
    sex: pd.Series,
) -> StrataData:
    """Build the stratification variables and demographics for a cohort.

    Parameters
    ----------
    root : Path
        Repository root.
    version : str
        SPARK release version.
    index : pandas.Index
        The modelling-cohort proband index the variables are built for.
    age_at_eval : pandas.Series
        Age at evaluation per proband (from the cohort covariates), on ``index``.
    sex : pandas.Series
        Encoded sex per proband (from the cohort covariates), on ``index``.

    Returns
    -------
    StrataData
        The axes, lag, demographics, per-instrument years, and diagnostics.
    """
    cat = open_catalogue(root)

    def read_table(table: str, columns: list[str]) -> pd.DataFrame:
        path = source_csv(cat, root, "spark", version, table)
        frame = read_columns(path, [INDEX_COLUMN, *columns]).set_index(INDEX_COLUMN)
        return frame[~frame.index.duplicated(keep="first")].reindex(index)

    cdv = read_table(
        "core_descriptive_variables",
        [
            "diagnosis_age",
            "registration_year",
            "age_at_registration_years",
            "cognitive_impairment_latest",
        ],
    )
    registration = read_table("individuals_registration", ["hispanic", *_RACE_INDICATORS])

    age_at_eval = age_at_eval.reindex(index)
    axes, lag = derive_axes(
        cdv["diagnosis_age"],
        cdv["registration_year"],
        cdv["age_at_registration_years"],
        age_at_eval,
    )

    cognitive = cdv["cognitive_impairment_latest"].map(
        {"TRUE": 1.0, "FALSE": 0.0, True: 1.0, False: 0.0}
    )
    hispanic = registration["hispanic"].where(registration["hispanic"] != 888).astype(float)
    race = registration[list(_RACE_INDICATORS)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    demographics = pd.DataFrame(
        {
            "sex": sex.reindex(index).astype(float),
            "age_at_eval_years": age_at_eval.astype(float),
            "cognitive_impairment": cognitive.astype(float),
            "hispanic": hispanic,
            **{col: race[col].astype(float) for col in _RACE_INDICATORS},
        }
    )

    instrument_years = pd.DataFrame(
        {table: read_table(table, ["eval_year"])["eval_year"] for table in _DATED_INSTRUMENTS}
    )

    diagnostics = summarise(axes, lag, len(index))
    diagnostics["contemporaneity"] = _contemporaneity(instrument_years, axes["diagnosis_year"])

    return StrataData(
        axes=axes,
        lag=lag,
        demographics=demographics,
        instrument_years=instrument_years,
        diagnostics=diagnostics,
    )
