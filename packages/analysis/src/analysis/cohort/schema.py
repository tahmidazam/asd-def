"""The shared feature schema and the cross-cohort harmonisation maps.

The feature set is the authors' final 238-feature list. The CBCL competence items arrive
as strings in SPARK and are recoded to the ordinal integers the released preprocessing
used. The SSC rename maps carry the second cohort's column names onto the SPARK names, so
both cohorts present the same schema to the rest of the pipeline (plan section 10).
"""

from __future__ import annotations

import csv
import html
import math
import re
from pathlib import Path

# Recoding of the CBCL competence items from their string answers to ordinal integers,
# verbatim from the released ``process_integrate_phenotype_data.py``. Applied across the
# CBCL sub-frame; the numeric t-score and item columns are unaffected because their values
# are not among these string keys.
CBCL_REPLACEMENTS: dict[str, int] = {
    "above_average": 0,
    "average": 1,
    "below_average": 2,
    "failing": 3,
    # close_friends
    "none": 3,
    "1": 2,
    "2_3": 1,
    "4_more": 0,
    # contact friends outside school
    "less_1": 2,
    "1_2": 1,
    "3_more": 0,
    # gets-along / behaviour items
    "worse": 2,
    "better": 0,
    "has no brothers or sisters": 1,
}

# How the parent-reported sex string is encoded, as in the released SCQ preprocessing.
SEX_ENCODING: dict[str, int] = {"Male": 1, "Female": 0}

# SSC-to-SPARK column renames per instrument, verbatim from ``generate_ssc_data``. These
# define the harmonisation onto the shared schema for the second cohort (plan section 10).
SSC_CBCL_RENAME: dict[str, str] = {
    "add_adhd_t_score": "dsm5_attention_deficit_hyperactivity_t_score",
    "affective_problems_t_score": "dsm5_depressive_problems_t_score",
    "anxiety_problems_t_score": "dsm5_anxiety_problems_t_score",
    "conduct_problems_t_score": "dsm5_conduct_problems_t_score",
    "oppositional_defiant_t_score": "dsm5_oppositional_defiant_t_score",
    "rule_breaking_t_score": "rule_breaking_behavior_t_score",
    "withdrawn_t_score": "withdrawn_depressed_t_score",
    "somatic_prob_t_score": "dsm5_somatic_problems_t_score",
}

SSC_SCQ_RENAME: dict[str, str] = {
    "q08_hits_self_object": "q08_hits_self_against_object",
    "q09_hits_self_object": "q09_hits_self_with_object",
    "q28_communicatiion": "q28_communication",
    "summary_score": "final_score",
}

SSC_RBSR_RENAME: dict[str, str] = {"q39_insists_palce": "q39_insists_time"}

# SSC raw background-history milestone columns mapped onto the SPARK names. Nine of the 11
# SPARK background-history features map cleanly; ``combined_phrases_age_mos`` and
# ``repeat_grade`` have no clean SSC equivalent, so the SSC backend provides a subset.
# Litman et al. read these milestones from a hand-cleaned file that was not released, so
# both the mapping and the free-text parsing (``parse_age_months``) are ours; the fidelity
# is confirmed in the SSC replication stage (phase 2).
SSC_BH_RENAME: dict[str, str] = {
    "age_smiled": "smiled_age_mos",
    "age_sat_wo_support": "sat_wo_support_age_mos",
    "age_crawled": "crawled_age_mos",
    "age_walked_alone": "walked_age_mos",
    "age_fed_self_w_spoon": "fed_self_spoon_age_mos",
    "age_used_words": "used_words_age_mos",
    "age_combined_words_short_sen": "combined_words_age_mos",
    "age_bladder_trained_day": "bladder_trained_age_mos",
    "age_bowel_trained": "bowel_trained_age_mos",
}

# How the SSC sex string and the SCQ yes/no answers are encoded onto the SPARK coding.
SSC_SEX_ENCODING: dict[str, int] = {"male": 1, "female": 0}
SSC_YES_NO: dict[str, int] = {"yes": 1, "no": 0}

# The milestone features the SSC provides, after ``SSC_BH_RENAME``. Their raw values are
# free text, so they are parsed to months by ``parse_age_months`` before typing.
MILESTONE_AGE_FEATURES: frozenset[str] = frozenset(SSC_BH_RENAME.values())

# Free-text milestone entries that carry no numeric age; parsed to missing.
_NO_AGE: frozenset[str] = frozenset(
    {"na", "n/a", "none", "never", "not yet", "unknown", "normal", "varies", "?"}
)

_NUMBER = r"\d+(?:\.\d+)?"
# Units, longest spelling first so an anchored match prefers the full word.
_YEARS = r"(?:years|year|yrs|yr|y)"
_MONTHS_UNIT = r"(?:months|month|moths|moth|mnths|mnth|mths|mth|mons|mon|mos|mo|ms|mm|m)"
_WEEKS = r"(?:weeks|week|wks|wk|w)"
_DAYS_PER_MONTH = 30.4375  # 365.25 / 12, to convert a week count into months
_YEARS_MONTHS_RE = re.compile(rf"^({_NUMBER})\s*{_YEARS}(?:\s*({_NUMBER})\s*{_MONTHS_UNIT})?$")
_WEEKS_RE = re.compile(rf"^({_NUMBER})\s*{_WEEKS}$")
_MONTHS_RE = re.compile(rf"^({_NUMBER})\s*{_MONTHS_UNIT}?$")
_UNIT_TOKEN_RE = re.compile(rf"({_NUMBER})\s*({_YEARS}|{_WEEKS}|{_MONTHS_UNIT})\b")
_BARE_NUMBER_RE = re.compile(rf"^{_NUMBER}$")
_NUMBER_PREFIX_RE = re.compile(rf"^{_NUMBER}\s*")
_LEADING_RE = re.compile(
    r"^(?:at|about|around|approx\.?|approximately|circa|c\.|almost|nearly|roughly|close\s+to)\s+"
)
# Bounds and inequalities: stripped, then the stated age is taken (the bound is dropped). A
# bare number left after a bound has an ambiguous scale (years or months), so it is treated
# as missing rather than guessed.
_DIRECTIONAL_RE = re.compile(
    r"^(?:[<>]=?|less\s+than|more\s+than|greater\s+than|under|over|before|after|by|up\s+to)\s*"
)
_TRAILING_OLD_RE = re.compile(r"\s+old$")
_YO_RE = re.compile(r"\by\s*[./]\s*o\b")  # "y.o" / "y/o" -> years old
_AGE_RE = re.compile(rf"^age\s+({_NUMBER})\b")  # "age 4" -> four years
_YM_COLON_RE = re.compile(r"^(\d+):(\d+)\b")  # "3:6" -> three years six months
# Narrative cues (loss, regression, sequencing); a compound carrying these is not a single age.
_NARRATIVE_RE = re.compile(
    r"lost|stopped|regress|declin|until|again|gone|echolalia|babble|\bthen\b|\bbut\b"
)
_RANGE_SEPARATORS: tuple[str, ...] = (" to ", " or ", " - ", "-")
# Vulgar-fraction glyphs (half, quarter, three-quarters), kept ASCII via chr().
_HALF, _QUARTER, _THREE_QUARTER = chr(0x00BD), chr(0x00BC), chr(0x00BE)
# Half- and quarter-year fractions written as "N 1/2", normalised to a decimal before parsing.
_FRACTION_RES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(\d+)\s+1/2\b"), r"\g<1>.5"),
    (re.compile(r"(\d+)\s+1/4\b"), r"\g<1>.25"),
    (re.compile(r"(\d+)\s+3/4\b"), r"\g<1>.75"),
)


def _compound_months(text: str) -> float | None:
    """Sum the unit-tagged numbers in a compound entry like "2 yrs 10 mos", or None.

    A compound age is a sum of distinct units (years, then months, then weeks). A repeated
    unit (``"12 mos ... 15 mos"``) or a narrative cue marks a range or a regression rather
    than one age, and is returned as missing.
    """
    if _NARRATIVE_RE.search(text):
        return None
    counts = {"y": 0, "w": 0, "m": 0}
    total = 0.0
    for match in _UNIT_TOKEN_RE.finditer(text):
        number = float(match.group(1))
        unit = match.group(2)
        if re.fullmatch(_YEARS, unit):
            counts["y"] += 1
            total += number * 12.0
        elif re.fullmatch(_WEEKS, unit):
            counts["w"] += 1
            total += number * 7.0 / _DAYS_PER_MONTH
        else:
            counts["m"] += 1
            total += number
    if not any(counts.values()) or any(c > 1 for c in counts.values()):
        return None
    return total


def _single_age_months(text: str) -> float | None:
    """Parse one milestone entry, with no range, into months, or None when unrecognised."""
    m = _YEARS_MONTHS_RE.match(text)
    if m:
        months = float(m.group(2)) if m.group(2) else 0.0
        return float(m.group(1)) * 12.0 + months
    m = _WEEKS_RE.match(text)
    if m:
        return float(m.group(1)) * 7.0 / _DAYS_PER_MONTH
    m = _MONTHS_RE.match(text)
    if m:
        return float(m.group(1))
    return _compound_months(text)


def _parse_range(text: str) -> float | None:
    """Parse an "X to Y" age range into the midpoint of its endpoints, or None.

    A bare left endpoint borrows the right endpoint's unit, so ``"1-2 years"`` reads as one
    to two years rather than one month to two years.
    """
    for sep in _RANGE_SEPARATORS:
        if sep not in text:
            continue
        left, _, right = text.partition(sep)
        left, right = left.strip(), right.strip()
        if not left or not right:
            return None
        if _BARE_NUMBER_RE.fullmatch(left):
            right_unit = _NUMBER_PREFIX_RE.sub("", right, count=1).strip()
            left = f"{left} {right_unit}".strip()
        low = _single_age_months(left)
        high = _single_age_months(right)
        if low is None or high is None:
            return None
        return (low + high) / 2.0
    return None


def parse_age_months(value: object) -> float | None:
    r"""Parse a free-text developmental-milestone age into months.

    The SSC records milestone ages as free text, whereas the SPARK features are ages in
    months. The recognised forms are mapped onto months: a bare number or a number with a
    month unit (``"13 months"``, ``"13 mos"``, ``"12 mon"``, ``"13m"``) is taken as months; a
    number with a year unit is multiplied by twelve, with an optional trailing months part
    (``"1 yr 6 mo"``); a number with a week unit (``"6 weeks"``) is converted from weeks; a
    compound (``"2 yrs 10 mos"``) sums its distinct unit parts; a half- or quarter-year
    fraction (``"3 1/2 yrs"``), a ``"y.o."`` suffix (``"3 y/o"``), an ``"age N"`` phrase, and a
    trailing ``"old"`` are handled; ``"at birth"`` is zero; and a range or an "N or M"
    (``"12-14"``, ``"18 months to 2 years"``, ``"7 or 8 months"``) is read as its midpoint. A
    bound (``"<3 mos"``, ``"before 1 year"``) has the bound dropped and the stated age taken.

    Some entries are deliberately left missing, so they drop at the complete-case step as in
    the released pipeline: text with no numeric age (``"never"``, ``"normal"``, ``"on time"``);
    a calendar date entered in the age field (``"03/2003"``); a regression or loss narrative
    (``"12 mos (lost at 15 mos)"``); and a bare number left after a bound (``"under 2"``),
    whose scale (years or months) is ambiguous. The parsing rules and the forms left missing
    are set out in the package's milestone-parsing guide.

    The parsed ages stay continuous in months rather than being snapped to the SPARK dropdown
    grid, because the milestone features are modelled as continuous.

    Parameters
    ----------
    value : object
        A raw milestone cell: a string, a number, or a missing value.

    Returns
    -------
    float or None
        The age in months, or None when the entry carries no recognisable age.
    """
    if not isinstance(value, str):
        if isinstance(value, (int, float)) and not math.isnan(value):
            return float(value)  # an already-numeric cell (numpy floats included)
        return None
    text = html.unescape(value).strip().lower()
    text = text.rstrip(".").replace("~", "").replace(",", "").replace("+", "")
    text = text.replace(_HALF, ".5").replace(_QUARTER, ".25").replace(_THREE_QUARTER, ".75")
    text = _LEADING_RE.sub("", text)
    stripped = _DIRECTIONAL_RE.sub("", text, count=1)
    bounded = stripped != text
    text = stripped.strip()
    if bounded and _BARE_NUMBER_RE.fullmatch(text):
        return None
    text = _TRAILING_OLD_RE.sub("", text)
    text = _YO_RE.sub(" years", text)
    text = _AGE_RE.sub(r"\g<1> years", text)
    text = _YM_COLON_RE.sub(r"\g<1> years \g<2> months", text)
    for pattern, repl in _FRACTION_RES:
        text = pattern.sub(repl, text)
    text = text.strip()
    if not text or text in _NO_AGE:
        return None
    if "birth" in text:
        return 0.0
    months = _parse_range(text)
    if months is not None:
        return months
    return _single_age_months(text)


def load_feature_list(path: Path) -> list[str]:
    """Read the authors' feature list from its one-column CSV.

    Parameters
    ----------
    path : Path
        Location of ``mixture_model_columns.csv``, which has a ``feature`` header.

    Returns
    -------
    list of str
        The feature names, in file order.
    """
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # discard the header
        return [row[0].strip() for row in reader if row and row[0].strip()]
