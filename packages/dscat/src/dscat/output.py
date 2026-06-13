"""Render query results to the console in a chosen serialisation format.

A command builds a list of records (dictionaries that share their keys) and hands
it to :func:`render`, which turns it into an aligned table, CSV, TSV, JSON, or a
Markdown table. Keeping the formatting here means the commands no longer hand-build
delimited strings, and every command offers the same ``--format`` choices.
"""

from __future__ import annotations

import csv as _csv
import io
import json as _json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from tabulate import tabulate


class Format(StrEnum):
    """A serialisation format for tabular command output.

    Attributes
    ----------
    table
        An aligned plain-text table, the default for reading in a terminal.
    json
        A JSON array of record objects.
    csv
        Comma-separated values with a header row.
    tsv
        Tab-separated values with a header row.
    markdown
        A GitHub-flavoured Markdown table.
    """

    table = "table"
    json = "json"
    csv = "csv"
    tsv = "tsv"
    markdown = "markdown"


def _one_line(value: object, width: int | None) -> str:
    """Collapse a value to a single line, truncating to ``width`` with an ellipsis."""
    text = " ".join(str("" if value is None else value).split())
    if width is not None and len(text) > width:
        return text[: width - 1] + "…"
    return text


def render(
    rows: Sequence[Mapping[str, Any]],
    fmt: Format = Format.table,
    *,
    max_col_width: int | None = 60,
) -> str:
    """Render records in the requested format.

    Parameters
    ----------
    rows : sequence of mapping
        The records to render. Every mapping shares the same keys, which become the
        columns in their declared order.
    fmt : Format, default Format.table
        The output format.
    max_col_width : int or None, default 60
        For the ``table`` and ``markdown`` formats, collapse each cell to one line
        and truncate it to this many characters (``None`` leaves cells untouched).
        The ``csv``, ``tsv``, and ``json`` formats always keep the full values.

    Returns
    -------
    str
        The rendered text, without a trailing newline. Empty ``rows`` renders to
        ``"[]"`` for JSON and to the empty string for every other format.
    """
    records = [dict(r) for r in rows]
    if fmt is Format.json:
        return _json.dumps(records, ensure_ascii=False, indent=2)
    if not records:
        return ""
    headers = list(records[0].keys())
    if fmt in (Format.csv, Format.tsv):
        buffer = io.StringIO()
        writer = _csv.writer(
            buffer, delimiter="\t" if fmt is Format.tsv else ",", lineterminator="\n"
        )
        writer.writerow(headers)
        for record in records:
            writer.writerow(["" if record.get(h) is None else record.get(h) for h in headers])
        return buffer.getvalue().rstrip("\n")
    table_format = "github" if fmt is Format.markdown else "simple"
    body = [[_one_line(record.get(h), max_col_width) for h in headers] for record in records]
    return tabulate(body, headers=headers, tablefmt=table_format, disable_numparse=True)
