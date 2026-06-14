"""Unit tests for the order-sensitive, real-world-messy pure logic."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from dscat import docs
from dscat.cli import _codes
from dscat.config import version_sort_key
from dscat.docs import Engine, cache_path, convert_doc, resolve_engine
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


def test_engine_values_are_stable():
    assert [e.value for e in Engine] == ["markitdown", "marker", "textutil"]


def test_resolve_engine_defaults_by_format():
    assert resolve_engine(Path("a.pdf")) is Engine.marker  # PDFs default to marker
    assert resolve_engine(Path("a.docx")) is Engine.markitdown  # other formats to markitdown
    assert resolve_engine(Path("a.txt")) is Engine.markitdown
    assert resolve_engine(Path("a.pdf"), Engine.markitdown) is Engine.markitdown  # -e overrides


def test_resolve_engine_routes_legacy_formats_to_textutil():
    assert resolve_engine(Path("a.doc")) is Engine.textutil
    assert resolve_engine(Path("a.rtf"), Engine.marker) is Engine.textutil  # textutil wins over -e


def test_cache_path_names_file_by_engine(tmp_path):
    src = tmp_path / "welcome.pdf"
    default = cache_path(tmp_path, "spark", "2026-03-23", src)
    markitdown = cache_path(tmp_path, "spark", "2026-03-23", src, Engine.markitdown)
    assert default.name == "welcome.marker.md"  # PDFs default to marker
    assert markitdown.name == "welcome.markitdown.md"  # each engine writes to its own file
    assert default.parent == markitdown.parent


def test_cache_path_uses_markitdown_for_non_pdf_default(tmp_path):
    src = tmp_path / "table.docx"
    assert cache_path(tmp_path, "spark", "v", src).name == "table.markitdown.md"


def test_cache_path_uses_textutil_for_legacy_formats(tmp_path):
    src = tmp_path / "legacy.rtf"
    # .rtf is named (and converted) by textutil even when marker is requested
    assert cache_path(tmp_path, "spark", "v", src, Engine.marker).name == "legacy.textutil.md"


def test_convert_doc_reuses_cache_newer_than_source(tmp_path):
    src = tmp_path / "notes.txt"
    src.write_text("original\n", encoding="utf-8")
    dest = tmp_path / "notes.md"
    dest.write_text("cached\n", encoding="utf-8")
    os.utime(dest, (src.stat().st_mtime + 10, src.stat().st_mtime + 10))
    # a cache newer than the source is returned without invoking any engine
    assert convert_doc(src, dest).read_text(encoding="utf-8") == "cached\n"


def _completed(cmd: list[str], code: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(cmd, code, stdout="", stderr=stderr)


def test_convert_doc_markitdown_invokes_cli(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_text("# from markitdown\n", encoding="utf-8")
        return _completed(cmd)

    monkeypatch.setattr(docs, "_run", fake_run)
    src = tmp_path / "paper.pdf"
    src.write_text("x", encoding="utf-8")
    md = convert_doc(src, tmp_path / "paper.markitdown.md", Engine.markitdown)
    assert calls[0][0] == "markitdown"
    assert md.read_text(encoding="utf-8").startswith("# from markitdown")


def test_convert_doc_markitdown_errors_when_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(docs, "_run", lambda cmd: None)
    src = tmp_path / "paper.pdf"
    src.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not installed"):
        convert_doc(src, tmp_path / "paper.markitdown.md", Engine.markitdown)


def test_convert_doc_marker_collects_folder_output(tmp_path, monkeypatch):
    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        out_dir = Path(cmd[cmd.index("--output_dir") + 1]) / "paper"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "paper.md").write_text("# from marker\n", encoding="utf-8")
        return _completed(cmd)

    monkeypatch.setattr(docs, "_run", fake_run)
    src = tmp_path / "paper.pdf"
    src.write_text("x", encoding="utf-8")
    md = convert_doc(src, tmp_path / "paper.marker.md", Engine.marker)
    assert md.read_text(encoding="utf-8").startswith("# from marker")


def test_convert_doc_marker_errors_without_output(tmp_path, monkeypatch):
    monkeypatch.setattr(docs, "_run", lambda cmd: _completed(cmd, code=1, stderr="boom"))
    src = tmp_path / "paper.pdf"
    src.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="marker could not convert"):
        convert_doc(src, tmp_path / "paper.marker.md", Engine.marker)


def test_convert_doc_routes_rtf_to_textutil(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        Path(cmd[cmd.index("-output") + 1]).write_text("plain text\n", encoding="utf-8")
        return _completed(cmd)

    monkeypatch.setattr(docs, "_run", fake_run)
    src = tmp_path / "legacy.rtf"
    src.write_text("x", encoding="utf-8")
    # marker is requested, but a .rtf must convert through textutil regardless
    md = convert_doc(src, tmp_path / "legacy.textutil.md", Engine.marker)
    assert calls[0][0] == "textutil"
    assert md.read_text(encoding="utf-8") == "plain text\n"


def test_convert_doc_default_routes_non_pdf_to_markitdown(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_text("# md\n", encoding="utf-8")
        return _completed(cmd)

    monkeypatch.setattr(docs, "_run", fake_run)
    src = tmp_path / "table.docx"
    src.write_text("x", encoding="utf-8")
    convert_doc(src, tmp_path / "table.markitdown.md")  # no engine -> markitdown for a .docx
    assert calls[0][0] == "markitdown"
