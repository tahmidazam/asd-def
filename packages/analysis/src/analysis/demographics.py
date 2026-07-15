"""The demographic covariate catalogue for the drift-attribution screen (plan section 7f).

The block-attribution engine (:mod:`analysis.blocks`) asks two questions of any proband-level
quantity: does the phenotype partition drift along it (co-drift, generalised by the displacement
atlas of :mod:`analysis.axes`), and does the timing drift shrink when the phenotype is residualised
on it (conditioning). This module supplies the demographic covariates both reads consume: household
socioeconomic position, parental education and occupation, family structure, inferred parental age
at the child's birth, perinatal complications, and the individual's sex and race.

Each covariate is one :class:`DemographicSpec`. An ordered or scalar covariate (income band,
education level, a parental-age year, a complication count) also serves as an ordering axis, so the
atlas can read the class displacement along it; a nominal covariate (family type, marital status,
race) enters the conditioning read alone, as a low-dimensional one-hot block whose linear
contribution is partialled out of the features. The coverage floor and the drop of a near-degenerate
or thinly joined covariate are the command layer's job (plan section 7f); this module only declares
the covariates and reads them, one set of columns at a time, under the dscat guardrail.

Two facts about the reads govern how the covariates are built. First, a covariate orthogonal to the
timing axis cannot account for a timing-ordered drift, so the conditioning shrinkage of such a
covariate is near zero by construction; the screen reports each covariate's association with the
axis alongside its shrinkage so this ceiling is visible. Second, the shrinkage is a descriptive
partial association, not a causal claim: a covariate that is itself downstream of the diagnosed
phenotype (family living arrangement, say) would be over-adjusted, and the prose says so.

The orderings imposed on the ordinal covariates (the education ladder, the collapse of the
occupation and living-arrangement categories) are modelling choices, recorded here so a reader can
see and contest them rather than find them buried in a preprocessing step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis.axes import _INCOME_BANDS, AxisContext
from analysis.cohort import open_catalogue, read_columns, source_csv

# Covariate families, used to group the screen's rows and colour the figure.
SES = "ses"
FAMILY = "family"
PARENTAL = "parental"
INDIVIDUAL = "individual"

# The education ladder, lowest to highest. The placement of a trade school above a plain high-school
# diploma (post-secondary vocational training) and of the GED at diploma equivalence are the
# debatable rungs; the rest is the conventional years-of-schooling order.
_EDUCATION_RANK = {
    "did_not_attend_high_school": 1,
    "some_high_school": 2,
    "ged_diploma": 3,
    "high_school_graduate": 4,
    "trade_school": 5,
    "some_college": 6,
    "associate_degree": 7,
    "baccalaureate_degree": 8,
    "graduate_or_professional_degree": 9,
}

# The nine occupation categories collapsed to six, to cap the one-hot block's width. The collapse
# groups the two management tiers with the executive rank, and the two manual tiers together.
_OCCUPATION_GROUP = {
    "executive": "management_professional",
    "upper_management_administration_or_professional": "management_professional",
    "supervisor_middle_manager": "management_professional",
    "craftsmen_skilled_technician_clerical_or_office_work": "skilled_clerical",
    "service_industry": "service_labour",
    "laborer": "service_labour",
    "part_time_and_or_temporary_work": "part_time_temporary",
    "not_currently_employed_or_unable_to_work": "not_employed",
    "full_time_care_provider_for_my_children": "full_time_carer",
}

# The biological-parent marital-status categories collapsed to five: the three divorced sub-levels
# (neither, mother, or father remarried) become one, since remarriage is not the axis of interest.
_MARITAL_GROUP = {
    "married": "married",
    "never_married": "never_married",
    "separated": "separated",
    "widowed": "widowed",
    "divorced_neither_remarried": "divorced",
    "divorced_mother_remarried": "divorced",
    "divorced_father_remarried": "divorced",
}

# The living-arrangement categories collapsed to five, to cap the one-hot width.
_LIVES_WITH_GROUP = {
    "both_biological_parents": "both_biological",
    "both_joint": "both_biological",
    "biological_mother": "single_biological",
    "biological_father": "single_biological",
    "biological_mother_and_stepfather": "step_family",
    "biological_father_and_stepmother": "step_family",
    "adoptive_parent": "adoptive",
    "other_relative": "other",
    "group_or_residential_setting": "other",
    "self_independently": "other",
}

# The five genuine perinatal-complication items of ``basic_medical_screening``, kept apart from the
# congenital-anomaly ``birth_def_*`` checkboxes (a different construct) and their ``_calc`` rollups
# (which would double-count). Their dictionary definitions all read "Preg/Birth complications --".
_BIRTH_COMPLICATION_COLS = (
    "birth_prem",
    "birth_oxygen",
    "birth_ivh",
    "birth_etoh_subst",
    "birth_pg_inf",
)

# Sentinel tokens that mean "not answered", mapped to not-a-number before any coding. ``999`` is the
# count sentinel of the family-history items; the ``na_survey_*`` tokens mark a question absent from
# a survey version or skipped by branching logic.
_MISSING_TOKENS = frozenset({"", "999", "na_survey_version", "na_survey_logic"})


def _read_table(ctx: AxisContext, table: str, columns: list[str]) -> pd.DataFrame:
    """Read the subject id and the named columns from a table's source CSV (the dscat guardrail).

    Keeps the first row per proband and indexes by ``subject_sp_id``, so a caller can reindex onto
    the modelling cohort.
    """
    catalogue = open_catalogue(ctx.root)
    path = source_csv(catalogue, ctx.root, ctx.dataset, ctx.version, table)
    frame = read_columns(path, ["subject_sp_id", *columns]).set_index("subject_sp_id")
    return frame[~frame.index.duplicated(keep="first")]


def _clean(series: pd.Series) -> pd.Series:
    """Strip whitespace and map the missing sentinels to not-a-number, leaving a string series."""
    stripped = series.astype(str).str.strip()
    return stripped.where(~stripped.isin(_MISSING_TOKENS))


def _one_hot(series: pd.Series, prefix: str, *, drop_first: bool = True) -> pd.DataFrame:
    """One-hot encode a categorical series, keeping missing rows missing.

    A row whose category is not-a-number is not-a-number across every indicator column, so the
    coverage join treats it as missing rather than as an implicit all-zero level. The first level in
    sorted order is dropped as the reference when ``drop_first`` is set, so the block is not
    collinear with the intercept the conditioning residualiser adds.
    """
    valid = series.notna().to_numpy()
    levels = sorted(str(v) for v in pd.unique(series.dropna()))
    if drop_first:
        levels = levels[1:]
    data = {
        f"{prefix}={level}": np.where(valid, (series == level).to_numpy(dtype=float), np.nan)
        for level in levels
    }
    return pd.DataFrame(data, index=series.index)


def _ordinal(
    table: str, column: str, mapping: dict[str, int]
) -> Callable[[AxisContext], pd.DataFrame]:
    """Return a loader for an ordered categorical mapped to its rank."""

    def load(ctx: AxisContext) -> pd.DataFrame:
        series = _clean(_read_table(ctx, table, [column])[column]).map(mapping)
        return series.reindex(ctx.index).to_frame(name=column)

    return load


def _count(table: str, column: str) -> Callable[[AxisContext], pd.DataFrame]:
    """Return a loader for a numeric count with the missing sentinels cleared."""

    def load(ctx: AxisContext) -> pd.DataFrame:
        series = pd.to_numeric(_clean(_read_table(ctx, table, [column])[column]), errors="coerce")
        return series.reindex(ctx.index).to_frame(name=column)

    return load


def _binary(
    table: str, column: str, positive: set[str], negative: set[str]
) -> Callable[[AxisContext], pd.DataFrame]:
    """Return a loader for a two-level flag mapped to one and zero, other tokens not-a-number."""

    def load(ctx: AxisContext) -> pd.DataFrame:
        cleaned = _clean(_read_table(ctx, table, [column])[column])
        value = pd.Series(np.nan, index=cleaned.index, dtype=float)
        value[cleaned.isin(positive)] = 1.0
        value[cleaned.isin(negative)] = 0.0
        return value.reindex(ctx.index).to_frame(name=column)

    return load


def _calculated_flag(table: str, column: str) -> Callable[[AxisContext], pd.DataFrame]:
    """Return a loader for a computed zero-or-one flag where a blank is a genuine zero."""

    def load(ctx: AxisContext) -> pd.DataFrame:
        raw = _read_table(ctx, table, [column])[column].astype(str).str.strip()
        value = (raw == "1").astype(float)
        return value.reindex(ctx.index).to_frame(name=column)

    return load


def _collapsed_one_hot(
    table: str, column: str, groups: dict[str, str]
) -> Callable[[AxisContext], pd.DataFrame]:
    """Return a loader that collapses a nominal category to groups, then one-hot encodes it."""

    def load(ctx: AxisContext) -> pd.DataFrame:
        series = _clean(_read_table(ctx, table, [column])[column]).map(groups)
        return _one_hot(series.reindex(ctx.index), column)

    return load


def _race(ctx: AxisContext) -> pd.DataFrame:
    """Return the six race indicators, missing only when no race box is ticked at all.

    Each ``race_*`` column is one when the box is ticked and blank otherwise, so a blank is a real
    zero. A proband who ticked no box (the race-and-ethnicity item non-response) has no race
    information, so every indicator is not-a-number for that proband and the coverage join drops it,
    rather than reading it as a person of no race.
    """
    columns = [
        "race_asian",
        "race_african_amer",
        "race_native_amer",
        "race_native_hawaiian",
        "race_white",
        "race_other",
    ]
    frame = _read_table(ctx, "individuals_registration", columns)
    ticked = frame[columns].apply(lambda s: s.astype(str).str.strip().eq("1").astype(float))
    any_ticked = ticked.to_numpy().sum(axis=1) > 0
    ticked = ticked.where(pd.Series(any_ticked, index=ticked.index), other=np.nan)
    return ticked.reindex(ctx.index)


def _parental_age_at_birth(parent_id_column: str) -> Callable[[AxisContext], pd.DataFrame]:
    """Return a loader for the inferred parental age at the child's birth, in years.

    SPARK records no parental age at birth, but it is the enrolled parent's registration age minus
    the child's, joined through the biological-parent link. Implausible values (below 12 or above 60
    years) are dropped. Coverage follows how often the parent is enrolled with an age: strong for
    mothers, thinner for fathers.
    """

    def load(ctx: AxisContext) -> pd.DataFrame:
        frame = _read_table(
            ctx, "individuals_registration", ["age_at_registration_years", parent_id_column]
        )
        age = pd.to_numeric(frame["age_at_registration_years"], errors="coerce")
        parent_id = frame[parent_id_column].astype(str).str.strip()
        parent_age = pd.to_numeric(parent_id.map(age), errors="coerce")
        gap = parent_age - age
        gap = gap.where((gap >= 12.0) & (gap <= 60.0))
        return gap.reindex(ctx.index).to_frame(name=parent_id_column)

    return load


def _birth_complication_count(ctx: AxisContext) -> pd.DataFrame:
    """Return the count of endorsed perinatal complications, missing when the block was not asked.

    Sums the five genuine complication items (prematurity, oxygen deprivation, a brain bleed, fetal
    alcohol or substance exposure, a pregnancy infection). A ticked box is one and a blank is a
    not-endorsed zero, but a proband for whom every item is a skip-logic sentinel was never asked
    the block, so the count is not-a-number.
    """
    frame = _read_table(ctx, "basic_medical_screening", list(_BIRTH_COMPLICATION_COLS))
    stripped = frame[list(_BIRTH_COMPLICATION_COLS)].apply(lambda s: s.astype(str).str.strip())
    endorsed = stripped.eq("1").to_numpy(dtype=float)
    asked = ~stripped.isin({"na_survey_logic", "na_survey_version"}).to_numpy()
    count = np.where(asked.any(axis=1), endorsed.sum(axis=1), np.nan)
    return (
        pd.Series(count, index=frame.index, name="birth_complication_count")
        .reindex(ctx.index)
        .to_frame()
    )


@dataclass(frozen=True)
class DemographicSpec:
    """One demographic covariate for the drift-attribution screen.

    Attributes
    ----------
    name : str
        The covariate key: its handle on the command line and its stem in the artefacts.
    label : str
        A human-readable name for the figure.
    kind : str
        The covariate family: ``"ses"``, ``"family"``, ``"parental"``, or ``"individual"``.
    coding : str
        ``"ordinal"``, ``"scalar"``, ``"count"``, ``"binary"``, or ``"onehot"``. The first four give
        a single ordered column that also serves as an atlas ordering axis; ``"onehot"`` gives a
        low-dimensional block for the conditioning read alone.
    load : callable
        Maps an :class:`analysis.axes.AxisContext` to a proband-indexed frame on the cohort index,
        with not-a-number where the covariate is missing. Ordered codings return one column; a
        one-hot coding returns its indicator columns.
    """

    name: str
    label: str
    kind: str
    coding: str
    load: Callable[[AxisContext], pd.DataFrame] = field(compare=False)

    @property
    def ordered(self) -> bool:
        """Whether the covariate is a single ordered column the atlas can order probands by."""
        return self.coding in {"ordinal", "scalar", "count", "binary"}


# The demographic covariates, in a stable order grouped by family. Household income and the area
# deprivation index also appear in the atlas axis catalogue (:mod:`analysis.axes`); they are here
# too so the conditioning read covers them, the two reads being distinct questions.
DEMOGRAPHICS: tuple[DemographicSpec, ...] = (
    # Socioeconomic position.
    DemographicSpec(
        "household_income",
        "Household income",
        SES,
        "ordinal",
        _ordinal("background_history_child", "annual_household_income", _INCOME_BANDS),
    ),
    DemographicSpec(
        "area_deprivation",
        "Area deprivation",
        SES,
        "scalar",
        _count("area_deprivation_index", "X2019_adi_national_rank_percentile"),
    ),
    DemographicSpec(
        "mother_education",
        "Mother's education",
        SES,
        "ordinal",
        _ordinal("background_history_child", "mother_highest_education", _EDUCATION_RANK),
    ),
    DemographicSpec(
        "father_education",
        "Father's education",
        SES,
        "ordinal",
        _ordinal("background_history_child", "father_highest_education", _EDUCATION_RANK),
    ),
    # Family structure.
    DemographicSpec(
        "family_type",
        "Family type (simplex/multiplex)",
        FAMILY,
        "onehot",
        _collapsed_one_hot(
            "core_descriptive_variables",
            "family_type",
            {"Simplex": "simplex", "Multiplex": "multiplex"},
        ),
    ),
    DemographicSpec(
        "marital_status",
        "Parents' marital status",
        FAMILY,
        "onehot",
        _collapsed_one_hot(
            "background_history_child", "marital_status_biological_parents", _MARITAL_GROUP
        ),
    ),
    DemographicSpec(
        "child_lives_with",
        "Living arrangement",
        FAMILY,
        "onehot",
        _collapsed_one_hot("background_history_child", "child_lives_with", _LIVES_WITH_GROUP),
    ),
    DemographicSpec(
        "num_asd_parents",
        "Number of parents with ASD",
        FAMILY,
        "count",
        _count("individuals_registration", "num_asd_parents"),
    ),
    DemographicSpec(
        "num_asd_siblings",
        "Number of siblings with ASD",
        FAMILY,
        "count",
        _count("individuals_registration", "num_asd_siblings"),
    ),
    DemographicSpec(
        "multiple_birth",
        "Multiple birth",
        FAMILY,
        "binary",
        _binary(
            "individuals_registration",
            "multiple_birth",
            positive={"Twin", "Triplet", "Quadruplet"},
            negative={"No"},
        ),
    ),
    DemographicSpec(
        "excluded_family_members",
        "Excluded family members",
        FAMILY,
        "binary",
        _binary(
            "individuals_registration",
            "excluded_family_members",
            positive={"1"},
            negative={"0"},
        ),
    ),
    DemographicSpec(
        "split_bio_family",
        "Split biological family",
        FAMILY,
        "binary",
        _calculated_flag("individuals_registration", "split_bio_family"),
    ),
    # Parental characteristics.
    DemographicSpec(
        "maternal_age_at_birth",
        "Maternal age at birth (inferred)",
        PARENTAL,
        "scalar",
        _parental_age_at_birth("biomother_sp_id"),
    ),
    DemographicSpec(
        "paternal_age_at_birth",
        "Paternal age at birth (inferred)",
        PARENTAL,
        "scalar",
        _parental_age_at_birth("biofather_sp_id"),
    ),
    DemographicSpec(
        "mother_occupation",
        "Mother's occupation",
        PARENTAL,
        "onehot",
        _collapsed_one_hot("background_history_child", "mother_occupation", _OCCUPATION_GROUP),
    ),
    DemographicSpec(
        "father_occupation",
        "Father's occupation",
        PARENTAL,
        "onehot",
        _collapsed_one_hot("background_history_child", "father_occupation", _OCCUPATION_GROUP),
    ),
    # Individual characteristics.
    DemographicSpec(
        "sex",
        "Sex assigned at birth",
        INDIVIDUAL,
        "binary",
        _binary(
            "individuals_registration",
            "sex",
            positive={"Male"},
            negative={"Female"},
        ),
    ),
    DemographicSpec(
        "birth_complications",
        "Perinatal complication count",
        INDIVIDUAL,
        "count",
        _birth_complication_count,
    ),
    DemographicSpec("race", "Race", INDIVIDUAL, "onehot", _race),
)

DEMOGRAPHICS_BY_NAME: dict[str, DemographicSpec] = {spec.name: spec for spec in DEMOGRAPHICS}
