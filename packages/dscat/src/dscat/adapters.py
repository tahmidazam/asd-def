"""Layout engines that turn a dictionary into normalised ``FeatureRow``s.

``sheet_per_table`` (SPARK): one sheet per table; the variable column is found by
alias, falling back to column 0 when a sheet has a junk/blank header cell. An
``ADOS File Name`` column (only in the combined ADOS sheet) is folded into notes.

``single_sheet`` (SSC): all variables on one sheet; columns are mapped from the
machine-id header row (``fieldid_row``); the Mother/Father/Proband/Sibling/Family/other
columns become each variable's ``roles_applicable`` set.
"""

from __future__ import annotations

from dscat.config import DatasetConfig, Version
from dscat.dictionary import read_sheet, sheet_names
from dscat.model import FeatureRow

APPLICABILITY_LABELS = {"mother", "father", "proband", "sibling", "family", "other"}


def parse(cfg: DatasetConfig, version: Version) -> tuple[list[FeatureRow], dict[str, str]]:
    """Parse a version's dictionary into feature rows using the configured layout.

    Parameters
    ----------
    cfg : DatasetConfig
        The dataset's adapter configuration.
    version : Version
        The version to parse; its ``dictionary_path`` is read.

    Returns
    -------
    list of FeatureRow
        One row per variable found in the dictionary.
    dict of str to str
        Logical table name mapped to its display title.

    Raises
    ------
    ValueError
        When ``cfg.layout`` is neither ``"sheet_per_table"`` nor ``"single_sheet"``.
    """
    if version.dictionary_path is None:
        return [], {}
    if cfg.layout == "sheet_per_table":
        return _parse_sheet_per_table(cfg, version)
    if cfg.layout == "single_sheet":
        return _parse_single_sheet(cfg, version)
    raise ValueError(f"unknown layout: {cfg.layout!r}")


def _col_index(header: list[str], aliases: list[str]) -> int | None:
    low = [h.strip().lower() for h in header]
    for alias in aliases:
        a = alias.strip().lower()
        for i, h in enumerate(low):
            if h == a:
                return i
    return None


def _get(row: list[str], idx: int | None) -> str:
    if idx is None or idx < 0 or idx >= len(row):
        return ""
    return row[idx]


def _truthy(v: str) -> bool:
    return str(v).strip().lower() not in ("", "0", "0.0", "no", "false", "n")


def _parse_sheet_per_table(
    cfg: DatasetConfig, version: Version
) -> tuple[list[FeatureRow], dict[str, str]]:
    path = version.dictionary_path
    assert path is not None
    skip = {s.strip().lower() for s in cfg.skip_sheets}
    features: list[FeatureRow] = []
    display: dict[str, str] = {}
    for sheet in sheet_names(path):
        if sheet.strip().lower() in skip:
            continue
        rows = read_sheet(path, sheet)
        if not rows:
            continue
        header = rows[0]
        cmap = {f: _col_index(header, a) for f, a in cfg.columns.items()}
        name_col = cmap.get("name")
        if name_col is None:  # junk/blank header cell -> variable is column 0
            name_col = 0
        table_name = sheet.strip().lower()
        seen: set[str] = set()  # ADOS_combined repeats each variable once per module
        for r in rows[1:]:
            name = _get(r, name_col)
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            notes = _get(r, cmap.get("notes"))
            ados_file = _get(r, cmap.get("ados_file"))
            if ados_file:
                notes = f"[ADOS file: {ados_file}] {notes}".strip()
            features.append(
                FeatureRow(
                    dataset=cfg.name,
                    version=version.version,
                    table_name=table_name,
                    name=name,
                    definition=_get(r, cmap.get("definition")),
                    field_type=_get(r, cmap.get("field_type")),
                    value_coding=_get(r, cmap.get("value_coding")),
                    notes=notes,
                    source_sheet=sheet,
                )
            )
        display[table_name] = sheet
    return features, display


def _parse_single_sheet(
    cfg: DatasetConfig, version: Version
) -> tuple[list[FeatureRow], dict[str, str]]:
    path = version.dictionary_path
    assert path is not None and cfg.sheet is not None
    rows = read_sheet(path, cfg.sheet)
    if len(rows) <= cfg.fieldid_row:
        return [], {}
    header = rows[cfg.fieldid_row]
    cmap = {f: _col_index(header, a) for f, a in cfg.columns.items()}
    tcol = cmap.get("table_name")
    ncol = cmap.get("name")
    tdt_col = cmap.get("table_display_title")
    role_cols: dict[int, str] = {}
    for i, h in enumerate(header):
        hl = h.strip().lower()
        if hl in APPLICABILITY_LABELS:
            role_cols[i] = hl
    features: list[FeatureRow] = []
    display: dict[str, str] = {}
    for r in rows[cfg.fieldid_row + 1 :]:
        table_name = _get(r, tcol).strip().lower()
        name = _get(r, ncol)
        if not table_name or not name:
            continue
        roles = [role for i, role in role_cols.items() if _truthy(_get(r, i))]
        features.append(
            FeatureRow(
                dataset=cfg.name,
                version=version.version,
                table_name=table_name,
                name=name,
                qualified_id=_get(r, cmap.get("qualified_id")),
                definition=_get(r, cmap.get("definition")),
                measurement_scale=_get(r, cmap.get("measurement_scale")),
                value_coding=_get(r, cmap.get("value_coding")),
                notes=_get(r, cmap.get("notes")),
                display_title=_get(r, cmap.get("display_title")),
                display_hint=_get(r, cmap.get("display_hint")),
                roles_applicable=",".join(roles),
                source_sheet=cfg.sheet,
            )
        )
        tdt = _get(r, tdt_col)
        if tdt and table_name not in display:
            display[table_name] = tdt
    return features, display
