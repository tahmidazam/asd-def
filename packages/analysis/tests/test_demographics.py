"""Gates for the demographic covariate catalogue and the conditioning ceiling.

The scientific engine (``blocks.conditioning_shrinkage``: a covariate that carries the drift shrinks
it, an irrelevant one does not, and a joint mask keeps each block's own sample) is gated in
``test_blocks.py``. These gates cover what the demographic screen adds on top: the covariate
encoders, the missing-sentinel handling, the race item-nonresponse rule, the inferred parental-age
derivation, the perinatal-complication count, and the axis-span ceiling the stage reports beside the
shrinkage. The one input-output seam, ``_read_table``, is monkeypatched so no participant data is
read.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from analysis import demographics as dm


def _ctx(index: pd.Index) -> SimpleNamespace:
    """Return a minimal axis context; loaders touch only ``index`` once ``_read_table`` patched."""
    return SimpleNamespace(index=index, root=None, dataset="spark", version="test")


def _patch_read_table(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> None:
    """Make ``_read_table`` return the requested columns of a synthetic, subject-indexed frame."""

    def fake(_ctx: object, _table: str, columns: list[str]) -> pd.DataFrame:
        return frame[columns]

    monkeypatch.setattr(dm, "_read_table", fake)


def test_one_hot_drops_reference_and_keeps_missing_missing() -> None:
    series = pd.Series(["a", "b", "c", np.nan], index=["p0", "p1", "p2", "p3"])
    encoded = dm._one_hot(series, "x")
    # The first level in sorted order (a) is the dropped reference, so two columns remain.
    assert list(encoded.columns) == ["x=b", "x=c"]
    # The reference row is a real all-zero, not missing.
    assert encoded.loc["p0"].tolist() == [0.0, 0.0]
    # The category rows are one in their own column.
    assert encoded.loc["p1", "x=b"] == 1.0
    assert encoded.loc["p2", "x=c"] == 1.0
    # A missing category is missing across every indicator, so the coverage join drops it.
    assert encoded.loc["p3"].isna().all()


def test_clean_maps_the_missing_sentinels() -> None:
    series = pd.Series(["married", "999", "na_survey_version", "", "na_survey_logic", "5"])
    cleaned = dm._clean(series)
    assert cleaned.iloc[0] == "married"
    assert cleaned.iloc[5] == "5"
    assert cleaned.iloc[1:5].isna().all()


def test_ordered_property_tracks_the_coding() -> None:
    for spec in dm.DEMOGRAPHICS:
        assert spec.ordered == (spec.coding in {"ordinal", "scalar", "count", "binary"})


def test_registry_is_consistent() -> None:
    names = [spec.name for spec in dm.DEMOGRAPHICS]
    assert len(names) == len(set(names))
    assert dm.DEMOGRAPHICS_BY_NAME.keys() == set(names)
    for spec in dm.DEMOGRAPHICS:
        assert spec.kind in {dm.SES, dm.FAMILY, dm.PARENTAL, dm.INDIVIDUAL}
        assert spec.coding in {"ordinal", "scalar", "count", "binary", "onehot"}
        assert callable(spec.load)


def test_race_item_nonresponse_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    columns = [
        "race_asian",
        "race_african_amer",
        "race_native_amer",
        "race_native_hawaiian",
        "race_white",
        "race_other",
    ]
    # The columns read back as floats (a blank forces the column to float), so "1.0" not "1"; the
    # loader must coerce numerically. p_white ticks white, p_asian asian, p_none nothing.
    frame = pd.DataFrame(
        {c: [np.nan, np.nan, np.nan] for c in columns},
        index=["p_white", "p_asian", "p_none"],
    )
    frame.loc["p_white", "race_white"] = 1.0
    frame.loc["p_asian", "race_asian"] = 1.0
    _patch_read_table(monkeypatch, frame)

    out = dm.DEMOGRAPHICS_BY_NAME["race"].load(_ctx(frame.index))
    # A ticked proband is finite across all six indicators (blanks read as a real zero).
    assert out.loc["p_white"].notna().all()
    assert out.loc["p_white", "race_white"] == 1.0
    assert out.loc["p_white", "race_asian"] == 0.0
    # A proband who ticked no box has no race information at all.
    assert out.loc["p_none"].isna().all()


def test_parental_age_derivation(monkeypatch: pytest.MonkeyPatch) -> None:
    # A child (age 5) whose biomother (age 34) is enrolled gives a 29-year gap; a child with an
    # implausible gap (mother recorded younger than child) drops; a child with no link drops.
    frame = pd.DataFrame(
        {
            "age_at_registration_years": [5.0, 8.0, 3.0, 34.0, 40.0],
            "biomother_sp_id": ["mom_a", "mom_b", "", "", ""],
        },
        index=["child_a", "child_b", "child_c", "mom_a", "mom_b"],
    )
    frame.loc["mom_b", "age_at_registration_years"] = 6.0  # implausible: younger than the child
    _patch_read_table(monkeypatch, frame)

    out = dm.DEMOGRAPHICS_BY_NAME["maternal_age_at_birth"].load(_ctx(frame.index))
    gap = out.iloc[:, 0]
    assert gap.loc["child_a"] == pytest.approx(29.0)
    assert np.isnan(gap.loc["child_b"])  # 6 - 8 = -2, outside [12, 60]
    assert np.isnan(gap.loc["child_c"])  # no biomother link


def test_birth_complication_count(monkeypatch: pytest.MonkeyPatch) -> None:
    cols = list(dm._BIRTH_COMPLICATION_COLS)
    # p_two endorses two items; p_none was asked but endorsed nothing; p_skip was never asked.
    frame = pd.DataFrame(
        {c: ["", "", "na_survey_logic"] for c in cols},
        index=["p_two", "p_none", "p_skip"],
    )
    frame.loc["p_two", cols[0]] = "1"
    frame.loc["p_two", cols[1]] = "1"
    _patch_read_table(monkeypatch, frame)

    out = dm._birth_complication_count(_ctx(frame.index)).iloc[:, 0]
    assert out.loc["p_two"] == 2.0
    assert out.loc["p_none"] == 0.0
    assert np.isnan(out.loc["p_skip"])  # every item a skip-logic sentinel, so never asked


def test_covariate_axis_r2_is_the_span_ceiling() -> None:
    from analysis.cli import _covariate_axis_r2

    axis = np.arange(200.0)
    # A covariate that linearly determines the axis spans all of it.
    assert _covariate_axis_r2(axis, 2.0 * axis + 1.0) == pytest.approx(1.0)
    # A covariate independent of the axis spans almost none of it.
    rng = np.random.default_rng(0)
    assert _covariate_axis_r2(axis, rng.normal(size=200)) < 0.05
    # A two-column covariate uses both columns.
    two = np.column_stack([axis, rng.normal(size=200)])
    assert _covariate_axis_r2(axis, two) == pytest.approx(1.0)
