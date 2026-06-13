"""Unit tests for the order-sensitive, real-world-messy pure logic."""

from __future__ import annotations

import json

import pytest
from dscat.cli import _codes
from dscat.config import version_sort_key
from dscat.index import Catalogue
from dscat.ingest import _bind, _norm_key
from dscat.output import Format, render
from dscat.queries import expand_query
from dscat.synonyms import _pairs
from sqlalchemy import create_engine

_ROWS = [{"a": 1, "b": "x,y"}, {"a": 2, "b": "z"}]


def test_version_sort_key_orders_iso_dates():
    assert max(["2025-03-31", "2026-03-23", "2024-01-01"], key=version_sort_key) == "2026-03-23"


def test_version_sort_key_orders_dotted_numerically():
    assert max(["15.3", "15.10", "9.9"], key=version_sort_key) == "15.10"


def test_norm_key_collapses_separators():
    assert _norm_key("cbcl1-5") == _norm_key("cbcl_1_5") == "cbcl15"
    assert _norm_key("area deprivation index") == "areadeprivationindex"


def test_bind_exact_and_separator_insensitive():
    sheets = {"cbcl1-5", "scq", "area deprivation index"}
    assert _bind("scq", sheets) == "scq"
    assert _bind("cbcl_1_5", sheets) == "cbcl1-5"
    assert _bind("area_deprivation_index", sheets) == "area deprivation index"
    assert _bind("nope", sheets) is None


def test_bind_handles_excel_31char_truncation():
    sheets = {"approximated_cognitive_impairme"}  # 31 chars, truncated by Excel
    assert _bind("approximated_cognitive_impairment", sheets) == "approximated_cognitive_impairme"


def test_bind_ambiguous_returns_none():
    assert _bind("abc", {"ab-c", "ab_c"}) is None  # two sheets normalise to the same key


def test_codes_treats_empty_placeholders_as_blank():
    assert _codes("[]") == ""
    assert _codes("   ") == ""
    assert _codes("0 = no\n1 = yes") == "0 = no; 1 = yes"


def _mem_catalogue() -> Catalogue:
    cat = Catalogue(create_engine("sqlite://"))
    cat.init_schema()
    cat.insert_synonyms([("sleep", "insomnia"), ("insomnia", "sleep")])
    cat.commit()
    return cat


def test_expand_query_expands_synonyms_into_or_group():
    expr = expand_query(_mem_catalogue(), "sleep", raw=False)
    assert "sleep*" in expr and "insomnia*" in expr and " OR " in expr


def test_expand_query_raw_is_passthrough():
    assert expand_query(_mem_catalogue(), "foo OR bar", raw=True) == "foo OR bar"


def test_expand_query_rejects_empty():
    with pytest.raises(ValueError):
        expand_query(_mem_catalogue(), "!!!", raw=False)


def test_render_json_round_trips():
    assert json.loads(render(_ROWS, Format.json)) == _ROWS


def test_render_empty_is_json_array_or_blank():
    assert render([], Format.json) == "[]"
    assert render([], Format.csv) == ""
    assert render([], Format.table) == ""


def test_render_csv_quotes_embedded_delimiter():
    out = render(_ROWS, Format.csv).splitlines()
    assert out[0] == "a,b"
    assert out[1] == '1,"x,y"'  # the comma in the value forces quoting


def test_render_tsv_uses_tabs():
    assert render(_ROWS, Format.tsv).splitlines()[0] == "a\tb"


def test_render_table_and_markdown_carry_headers():
    assert "a" in render(_ROWS, Format.table).splitlines()[0]
    assert render(_ROWS, Format.markdown).startswith("|")


def test_render_table_truncates_wide_cells():
    wide = [{"x": "y" * 100}]
    assert "…" in render(wide, Format.table, max_col_width=10)
    assert "…" not in render(wide, Format.csv)  # serialisation keeps full values


def test_pairs_expands_groups_bidirectionally():
    pairs = _pairs([["Sleep", "insomnia"], ["solo"]])
    assert ("sleep", "insomnia") in pairs and ("insomnia", "sleep") in pairs
    assert all("solo" not in pair for pair in pairs)  # a one-term group makes no pairs
