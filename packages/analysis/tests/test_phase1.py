"""Unit tests for the phase-1 pipeline, on synthetic data only.

No participant data is read here. The cohort backends, which read SFARI data, are exercised
through their pure helpers and the public typing, enrichment, and alignment logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analysis import cache
from analysis.align import hungarian_align
from analysis.cohort.schema import build_milestone_disambiguator, parse_age_months
from analysis.enrich import SEVEN_CATEGORIES, category_signature, feature_enrichment
from analysis.features import (
    Typing,
    infer_from_dictionary,
    n_value_levels,
    reconcile,
)
from analysis.reference import align_to_named, published_signature
from analysis.run import run_context


# ---- cache + run lifecycle ---------------------------------------------------
def test_hash_is_order_invariant_and_input_sensitive() -> None:
    a = {"dataset": "spark", "version": "2026-03-23", "seed": 0}
    reordered = {"seed": 0, "version": "2026-03-23", "dataset": "spark"}
    changed = {"dataset": "spark", "version": "2026-03-23", "seed": 1}
    assert cache.compute_hash(a) == cache.compute_hash(reordered)
    assert cache.compute_hash(a) != cache.compute_hash(changed)


def test_run_context_caches_and_captures_log(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANALYSIS_ROOT", str(tmp_path))
    calls = {"n": 0}

    def do() -> bool:
        with run_context("demo", {"x": 1}) as ctx:
            if ctx.cache_hit:
                return True
            calls["n"] += 1
            print("captured stdout line")
            cache.save_frame(pd.DataFrame({"a": [1]}), ctx.path("out.parquet"))
            ctx.metrics = {"ok": True}
            return False

    assert do() is False
    assert do() is True
    assert calls["n"] == 1
    rdir = tmp_path / "artefacts" / "demo" / cache.short_hash(cache.compute_hash({"x": 1}))
    manifest = cache.read_manifest(rdir)
    assert manifest is not None
    assert manifest["status"] == "ok"
    assert manifest["metrics"] == {"ok": True}
    assert "captured stdout line" in (rdir / "run.log").read_text()


# ---- typing inference + reconciliation ---------------------------------------
@pytest.mark.parametrize(
    ("field_type", "value_coding", "expected"),
    [
        ("calculated", "integer", "continuous"),
        ("dropdown", "1-24 = age in months\n27 = 2 years 3 months", "continuous"),
        ("radio", "0 = no\n1 = yes", "binary"),
        ("radio", "0 = none\n1 = mild\n2 = severe", "categorical"),
        ("radio", "", "categorical"),
    ],
)
def test_infer_from_dictionary(field_type: str, value_coding: str, expected: str) -> None:
    assert infer_from_dictionary(field_type, value_coding) == expected


def test_n_value_levels() -> None:
    assert n_value_levels("0 = no\n1 = yes") == 2
    assert n_value_levels("") == 0
    assert n_value_levels(None) == 0


# ---- SSC free-text milestone-age parsing -------------------------------------
@pytest.mark.parametrize(
    ("raw", "months"),
    [
        ("13 months", 13.0),
        ("13 mos", 13.0),
        ("13 mo", 13.0),
        ("13m", 13.0),
        ("13", 13.0),
        ("12 ", 12.0),
        ("1 year", 12.0),
        ("1 yr", 12.0),
        ("2yrs", 24.0),
        ("1.5 years", 18.0),
        ("1 yr 6 mo", 18.0),
        ("18.5 months", 18.5),
        ("at birth", 0.0),
        ("approx. 12 months", 12.0),
        ("~14 mos", 14.0),
        ("12-14", 13.0),
        ("1-2 years", 18.0),
        ("12 to 18 mo", 15.0),
        ("12 mon", 12.0),
        ("18 mon", 18.0),
        ("3 1/2 yrs", 42.0),
        ("2 1/2 years", 30.0),
        ("4 years old", 48.0),
        ("12 months old", 12.0),
        (12, 12.0),
        (12.0, 12.0),
        # B4: y.o. / y/o, compound year+month, "N or M", unit typos, colon
        ("3 y.o.", 36.0),
        ("4 y/o", 48.0),
        ("2 yrs. 10 mos", 34.0),
        ("3y;3m", 39.0),
        ("7 or 8 months", 7.5),
        ("11ms", 11.0),
        ("3:6 years", 42.0),
        ("18 months to 2 years", 21.0),
        ("9 mo - 1 yr", 10.5),
        # B3: bounds and inequalities drop the bound and take the stated age
        ("3+ years", 36.0),
        ("<3 mos", 3.0),
        (">42 mos", 42.0),
        ("&lt;1y", 12.0),
        ("before 1 year", 12.0),
        ("under 12 months", 12.0),
        ("by age 4", 48.0),
        ("age 3", 36.0),
    ],
)
def test_parse_age_months_recognised_forms(raw: object, months: float) -> None:
    assert parse_age_months(raw) == months


@pytest.mark.parametrize(
    ("raw", "months"),
    [
        ("6 weeks", 6 * 7 / 30.4375),
        ("1 week", 1 * 7 / 30.4375),
        ("6 wks", 6 * 7 / 30.4375),
        ("2-3 weeks", 2.5 * 7 / 30.4375),
        ("1 yr 2 weeks", 12 + 2 * 7 / 30.4375),
    ],
)
def test_parse_age_months_weeks(raw: str, months: float) -> None:
    assert parse_age_months(raw) == pytest.approx(months)


def test_parse_age_months_unicode_fraction() -> None:
    # a vulgar-fraction glyph (half), built with chr() to keep the test source ASCII
    assert parse_age_months("3" + chr(0x00BD) + " yrs") == 42.0


@pytest.mark.parametrize(
    "raw",
    [
        "normal",
        "n/a",
        "?",
        "",
        "five",
        None,
        float("nan"),
        # bare number after a bound has an ambiguous scale, so it stays missing
        "under 2",
        "over 3",
        # calendar dates and regression narratives are not single ages
        "03/2003",
        "12/01",
        "12 mos (lost at 15 mos)",
    ],
)
def test_parse_age_months_missing_forms(raw: object) -> None:
    assert parse_age_months(raw) is None


@pytest.mark.parametrize(
    "raw",
    ["never", "not yet", "hasn't walked", "has not", "doesn't", "unable", "can't", "not able"],
)
def test_parse_age_months_not_yet_maps_to_spark_code(raw: str) -> None:
    # A milestone stated as never reached takes the SPARK "888 = Not yet" code, not missing,
    # so the severe-delay proband is kept rather than dropped at the complete-case step.
    assert parse_age_months(raw) == 888.0


@pytest.mark.parametrize(("raw", "months"), [("18 years", 85.0), ("100 months", 85.0)])
def test_parse_age_months_caps_high_values(raw: str, months: float) -> None:
    # A parsed age above the SPARK "over 7 years" code caps at 85 months, which also discards
    # mis-parsed outliers.
    assert parse_age_months(raw) == months


# A late milestone whose SPARK ages cluster around three to four and a half years, and an early
# one clustering around a year, give the scale resolver its reference distribution.
_LATE_PRIOR = build_milestone_disambiguator(np.array([36.0, 42.0, 48.0, 54.0] * 50))
_EARLY_PRIOR = build_milestone_disambiguator(np.array([11.0, 12.0, 13.0, 14.0, 15.0] * 50))


def test_build_milestone_disambiguator_late_reads_small_as_years() -> None:
    # On a milestone whose SPARK ages are years, a bare "4" is four years, not four months.
    assert _LATE_PRIOR(4.0) == 48.0
    assert _LATE_PRIOR(3.0) == 36.0
    # A value already in the months range of the reference is kept.
    assert _LATE_PRIOR(42.0) == 42.0


def test_build_milestone_disambiguator_early_keeps_small_as_months() -> None:
    # On an early milestone (walking ~13 months), a small number stays months.
    assert _EARLY_PRIOR(6.0) == 6.0
    assert _EARLY_PRIOR(13.0) == 13.0


def test_build_milestone_disambiguator_degenerate_is_identity() -> None:
    # A reference with no spread cannot inform the scale, so the resolver is the identity.
    prior = build_milestone_disambiguator(np.array([42.0, 42.0, 42.0]))
    assert prior(4.0) == 4.0


def test_parse_age_months_disambiguates_unitless_and_bounds() -> None:
    # A unit-less number and a bare number left after a bound are resolved by the prior; an
    # explicit unit is trusted as written and a numeric cell is resolved like a bare number.
    assert parse_age_months("4", disambiguate=_LATE_PRIOR) == 48.0
    assert parse_age_months("under 2", disambiguate=_LATE_PRIOR) == 24.0
    assert parse_age_months(4.0, disambiguate=_LATE_PRIOR) == 48.0
    assert parse_age_months("4 months", disambiguate=_LATE_PRIOR) == 4.0
    assert parse_age_months("4 years", disambiguate=_LATE_PRIOR) == 48.0
    # An early-milestone prior keeps a small unit-less number as months.
    assert parse_age_months("13", disambiguate=_EARLY_PRIOR) == 13.0
    # Without a prior, behaviour is unchanged: a bare number is months, a bounded one is missing.
    assert parse_age_months("4") == 4.0
    assert parse_age_months("under 2") is None


def test_reconcile_defers_to_pickle_and_flags_conflict() -> None:
    features = ["a", "b", "repeat_grade"]
    dict_typing = {"a": "continuous", "b": "binary", "repeat_grade": "binary"}
    pickle_typing: dict[str, str | None] = {
        "a": "continuous",
        "b": "binary",
        "repeat_grade": "continuous",
    }
    observed = {"a": 40, "b": 2, "repeat_grade": 2}
    typing, report = reconcile(features, dict_typing, pickle_typing, observed)
    assert typing.continuous == ["a", "repeat_grade"]
    assert typing.binary == ["b"]
    conflict = report.set_index("feature").loc["repeat_grade"]
    assert not conflict["dictionary_pickle_agree"]
    assert conflict["chosen"] == "continuous"


def test_typing_as_dict_and_counts() -> None:
    typing = Typing(continuous=["a"], binary=["b", "c"], categorical=["d"])
    assert typing.as_dict() == {"a": "continuous", "b": "binary", "c": "binary", "d": "categorical"}
    assert typing.counts == {"continuous": 1, "binary": 2, "categorical": 1}


# ---- enrichment + signature --------------------------------------------------
def _synthetic_cohort() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 80
    labels = pd.Series(np.repeat([0, 1, 2, 3], n // 4), name="class")
    # binary feature present only in class 0; continuous feature high in class 1.
    binary = (labels == 0).astype(int)
    continuous = rng.normal(0, 1, n) + (labels == 1).to_numpy() * 5
    data = pd.DataFrame({"binmark": binary.to_numpy(), "contmark": continuous})
    return data, labels


def test_feature_enrichment_directions() -> None:
    data, labels = _synthetic_cohort()
    enr = feature_enrichment(data, labels, n_classes=4)
    assert enr.loc["binmark", "class0_dir"] == 1.0
    assert enr.loc["binmark", "is_binary"] == 1.0
    assert enr.loc["contmark", "class1_dir"] == 1.0
    assert enr.loc["contmark", "is_binary"] == 0.0


def test_category_signature_shape_and_sign() -> None:
    data, labels = _synthetic_cohort()
    enr = feature_enrichment(data, labels, n_classes=4)
    category_map = {"binmark": "self-injury", "contmark": "anxiety/mood"}
    sig = category_signature(enr, category_map, n_classes=4, reverse_coded=())
    assert list(sig.columns) == list(SEVEN_CATEGORIES)
    assert sig.shape == (4, 7)
    assert sig.loc[0, "self-injury"] == 1.0
    assert sig.loc[1, "anxiety/mood"] == 1.0


# ---- alignment ---------------------------------------------------------------
def test_hungarian_align_recovers_permutation() -> None:
    # Distinct, non-constant profiles so the correlation metric is well-defined.
    target = pd.DataFrame(
        [
            [1.0, 0.2, -0.5, 0.0],
            [-1.0, 0.5, 0.3, 0.1],
            [0.0, -0.7, 1.0, -0.2],
            [0.4, 0.4, -1.0, 0.9],
        ],
        index=["A", "B", "C", "D"],
        columns=["w", "x", "y", "z"],
    )
    shuffled = target.iloc[[2, 0, 3, 1]].copy()
    shuffled.index = pd.Index([10, 11, 12, 13], name="class")
    result = hungarian_align(shuffled, target, metric="correlation")
    assert result.mapping[10] == "C"
    assert result.mapping[11] == "A"
    assert all(r > 0.99 for r in result.correlations.values())


def test_align_to_named_anchors_hold_for_published_profile() -> None:
    target = published_signature().reset_index(drop=True)
    proportions = {0: 0.10, 1: 0.19, 2: 0.37, 3: 0.34}  # Broadly, Mixed, Social, Moderate
    named = align_to_named(target, proportions)
    assert named.mapping[0] == "Broadly affected"
    assert named.mapping[2] == "Social/behavioral"
    assert named.anchors_hold
