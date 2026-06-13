"""Normalised rows produced by adapters and written to the index.

There is one ``FeatureRow`` per dictionary variable, defined once per logical
table even when the physical CSV repeats across family-role folders (as in SSC).
There is one ``TableRow`` per physical CSV, so SSC yields one row per role and
SPARK one per table.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FeatureRow:
    """One dictionary variable, normalised for the index.

    Adapters emit one row per variable defined in a dataset's data dictionary. A
    variable is defined once per logical table, even when the physical CSV repeats
    across family-role folders (as in SSC).

    Attributes
    ----------
    dataset : str
        Dataset id the variable belongs to.
    version : str
        Dataset version the variable was read from.
    table_name : str
        Logical table the variable belongs to (the CSV stem the user sees).
    name : str
        Variable (column) name as it appears in the data files.
    qualified_id : str
        Fully qualified id from the dictionary, when one is given (SSC).
    definition : str
        The variable's definition or question text.
    field_type : str
        Storage or answer type recorded by the vendor (SPARK).
    measurement_scale : str
        Measurement scale such as nominal or ordinal (SSC).
    value_coding : str
        Coded answer choices, for example ``0 = no / 1 = yes``.
    notes : str
        Free-text notes from the dictionary.
    display_title : str
        Human-facing label for the variable (SSC).
    display_hint : str
        Short hint shown alongside the label (SSC).
    roles_applicable : str
        Comma-separated family roles the variable applies to (SSC); empty for SPARK.
    source_sheet : str
        Dictionary sheet the row was parsed from.
    """

    dataset: str
    version: str
    table_name: str
    name: str
    qualified_id: str = ""
    definition: str = ""
    field_type: str = ""
    measurement_scale: str = ""
    value_coding: str = ""
    notes: str = ""
    display_title: str = ""
    display_hint: str = ""
    roles_applicable: str = ""  # comma-separated family roles (SSC); "" for SPARK
    source_sheet: str = ""


@dataclass(slots=True)
class TableRow:
    """One physical CSV file in a dataset version.

    Ingestion emits one row per CSV on disk. SSC repeats a measure across role
    folders and so yields one row per (table, role); SPARK yields one per table.

    Attributes
    ----------
    dataset : str
        Dataset id.
    version : str
        Dataset version.
    table_name : str
        Logical table name (the CSV stem).
    display_title : str
        Human-facing table title from the dictionary, when available.
    role : str
        Family role for SSC (proband, mother, and so on); empty for SPARK.
    file_path : str
        POSIX path to the CSV, relative to the repository root.
    n_rows : int
        Data-row count, excluding the header.
    n_cols : int
        Column count.
    file_bytes : int
        File size in bytes.
    notes : str
        Ingestion notes, for example when no dictionary sheet matched.
    """

    dataset: str
    version: str
    table_name: str
    display_title: str = ""
    role: str = ""  # "" for SPARK; family role (proband/mother/...) for SSC
    file_path: str = ""  # POSIX path relative to repo root
    n_rows: int = 0
    n_cols: int = 0
    file_bytes: int = 0
    notes: str = ""
