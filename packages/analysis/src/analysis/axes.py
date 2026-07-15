"""The ordering-axis catalogue for the displacement atlas (plan section 12b).

The specificity check reads a class's separation-scaled endpoint displacement along an ordering
variable. The timing axes (diagnostic era, age at diagnosis) are two such orderings; this module
generalises the ordering to any continuous or ordered variable that is not one of the 238 clustered
features, so the same displacement can be screened across many axes and sorted from the largest to
the smallest mover. The atlas stage (``analysis displacement-atlas``) consumes this catalogue.

The axes are the timing pair (diagnostic era and age at diagnosis, the mechanism under test), a
covariate pool the cohort already carries or that timing derives (the measurement-to-diagnosis lag,
age at evaluation, household income, and the area deprivation index), and a seeded random ordering.
The random ordering is the floor: the one control guaranteed to carry no real structure, so an axis
whose displacement clears it is above sampling noise. There is no covariate-orthogonality assumption
here; the atlas reports every axis's displacement and leaves the random floor as the only reference.

Two roles are deliberately absent. The 238 clustered features and any total taken over them (an SCQ
or RBS-R sum) would order probands by a feature the classes were built from, moving the class
centroids by construction, a circular self-drift rather than an external ordering. Held-out
phenotype instruments (adaptive behaviour, motor coordination, IQ) are still phenotype, correlated
with the clustered features by construction, so they are a non-null ceiling rather than a clean
ordering and are left out for the same reason a symptom total is.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import strata_data
from analysis.cohort import open_catalogue, read_columns, source_csv

# The nine ordered household-income bands of ``background_history_child.annual_household_income``,
# mapped to a monotone rank so the band order (not the raw label) drives the ordering.
_INCOME_BANDS = {
    "less_than_20000": 1,
    "21000_35000": 2,
    "36000_50000": 3,
    "51000_65000": 4,
    "66000_80000": 5,
    "81000_100000": 6,
    "101000_130000": 7,
    "131000_160000": 8,
    "over_161000": 9,
}


@dataclass(frozen=True)
class AxisContext:
    """The inputs an axis loader may draw on, built once per atlas run.

    Attributes
    ----------
    root : pathlib.Path
        Repository root, for resolving the catalogue and source CSVs.
    dataset, version : str
        The cohort release the atlas runs on.
    index : pandas.Index
        The modelling-cohort proband index every axis is reindexed onto.
    covariates : pandas.DataFrame
        The cohort covariate frame (sex, age at evaluation, and the rest).
    strata : analysis.strata_data.StrataData
        The derived timing axes (era, age at diagnosis) and the lag.
    seed : int
        Base seed for the random ordering.
    """

    root: Path
    dataset: str
    version: str
    index: pd.Index
    covariates: pd.DataFrame
    strata: strata_data.StrataData
    seed: int


@dataclass(frozen=True)
class AxisSpec:
    """One ordering axis for the displacement atlas.

    Attributes
    ----------
    name : str
        The axis key: its column in the atlas artefact and its ``displacement-atlas`` handle.
    label : str
        A human-readable name for the figure.
    kind : str
        ``"timing"``, ``"covariate"``, ``"phenotype"``, or ``"random"``. The random axis is the
        floor; the timing axes are the mechanism under test; the rest are external orderings.
    load : callable
        Maps an :class:`AxisContext` to a proband-indexed ordering on the cohort index, with
        not-a-number where the variable is missing. A higher rank means more of the named quantity.
    """

    name: str
    label: str
    kind: str
    load: Callable[[AxisContext], pd.Series] = field(compare=False)


def _timing(column: str) -> Callable[[AxisContext], pd.Series]:
    """Return a loader for a derived timing axis (era or age at diagnosis)."""

    def load(ctx: AxisContext) -> pd.Series:
        return ctx.strata.axes[column].reindex(ctx.index)

    return load


def _lag(ctx: AxisContext) -> pd.Series:
    """Return the measurement-to-diagnosis lag in years."""
    return ctx.strata.lag.reindex(ctx.index)


def _age_at_eval(ctx: AxisContext) -> pd.Series:
    """Return age at evaluation, the covariate Litman entered into the model."""
    return ctx.covariates["age_at_eval_years"].astype(float).reindex(ctx.index)


def _random(ctx: AxisContext) -> pd.Series:
    """Return a seeded uniform ordering: the floor a meaningless axis produces."""
    rng = np.random.default_rng(ctx.seed)
    return pd.Series(rng.uniform(size=len(ctx.index)), index=ctx.index, name="random")


def _household_income(ctx: AxisContext) -> pd.Series:
    """Return the nine-band household income ordinal, mapped to a monotone rank."""
    cat = open_catalogue(ctx.root)
    path = source_csv(cat, ctx.root, ctx.dataset, ctx.version, "background_history_child")
    frame = read_columns(path, ["subject_sp_id", "annual_household_income"]).set_index(
        "subject_sp_id"
    )
    frame = frame[~frame.index.duplicated(keep="first")]
    return frame["annual_household_income"].map(_INCOME_BANDS).reindex(ctx.index)


def _area_deprivation(ctx: AxisContext) -> pd.Series:
    """Return the 2019 area deprivation index national-rank percentile."""
    cat = open_catalogue(ctx.root)
    path = source_csv(cat, ctx.root, ctx.dataset, ctx.version, "area_deprivation_index")
    # The release ships the column with an R-style ``X`` prefix (it starts with a digit),
    # which the dscat catalogue records without; read the on-disk name.
    column = "X2019_adi_national_rank_percentile"
    frame = read_columns(path, ["subject_sp_id", column]).set_index("subject_sp_id")
    frame = frame[~frame.index.duplicated(keep="first")]
    return pd.to_numeric(frame[column], errors="coerce").reindex(ctx.index)


# The atlas axes, in a stable order. Every axis is continuous or ordered and outside the 238
# clustered features, so none is a circular self-ordering. The timing axes are the mechanism under
# test; the random axis is the floor; the covariate axes are external orderings the class profiles
# may or may not track. Held-out phenotype instruments are deliberately absent: they are still
# phenotype, correlated with the clustered features by construction, so they are a non-null ceiling
# rather than a clean external ordering.
ATLAS_AXES: tuple[AxisSpec, ...] = (
    AxisSpec("era", "Diagnostic era", "timing", _timing("diagnosis_year")),
    AxisSpec("age_at_diagnosis", "Age at diagnosis", "timing", _timing("age_at_diagnosis_years")),
    AxisSpec("lag", "Measurement-to-diagnosis lag", "covariate", _lag),
    AxisSpec("age_at_eval", "Age at evaluation", "covariate", _age_at_eval),
    AxisSpec("household_income", "Household income", "covariate", _household_income),
    AxisSpec("area_deprivation", "Area deprivation", "covariate", _area_deprivation),
    AxisSpec("random", "Random ordering", "random", _random),
)

AXES_BY_NAME: dict[str, AxisSpec] = {spec.name: spec for spec in ATLAS_AXES}
