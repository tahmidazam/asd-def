# Adding a new dataset

SPARK and SSC have different dictionary layouts: SPARK uses one sheet per table,
while SSC uses a single flat sheet with a family-role applicability matrix. An adapter
config maps a dataset's dictionary onto the common index. Two layout engines cover
both shapes, so a new dataset is usually a new JSON file, not new code.

## 1. Drop the data in

Place the vendor's ship folder, unchanged, under a container directory in `data/`:

```text
data/<container>/<VendorShipFolder>/
```

The folder name encodes the version (see [Adding a dataset version](adding-a-version.md)).

## 2. Write the adapter config

Built-in adapters live in `packages/dscat/src/dscat/datasets/*.json`. A
project can add its own at `<repo-root>/datasets/*.json`; a project config overrides a
built-in with the same `name`. Re-run `uv run dscat ingest` after adding one.

### Common keys

| Key | Meaning |
| --- | --- |
| `name` | Dataset id, for example `spark`. |
| `display_name` | Human-facing label. |
| `container` | Subdirectory of `data/` to scan. |
| `layout` | `sheet_per_table` or `single_sheet`. |
| `version_pattern` | Regex with a named group `(?P<v>...)` that extracts the version from each ship-folder name. |
| `dictionary_glob` | Glob, relative to a version directory, locating the dictionary `.xlsx`. |
| `file_glob` | Glob for the data CSVs (default `**/*.csv`). |
| `strip_version_suffix` | Strip a trailing `-<version>` from CSV stems before binding them to dictionary tables. |
| `columns` | Map each canonical field to a list of header aliases; the first alias that matches a header wins, case-insensitively. |

### The sheet_per_table layout (SPARK-like)

One dictionary sheet per table. The canonical columns are `name`, `definition`,
`field_type`, `value_coding`, and `notes`, plus an optional `ados_file`. When no
`name` alias matches a sheet's header, the variable column falls back to column 0,
which tolerates a junk or blank header cell. `skip_sheets` lists non-table sheets to
ignore. CSV stems are bound to sheet names by exact match, by Excel's 31-character
truncation, or by a separator-insensitive match (so `cbcl1-5` binds to `cbcl_1_5`),
and the feature's table is re-keyed to the full CSV stem.

### The single_sheet layout (SSC-like)

All variables on one `sheet`. `fieldid_row` (0-based) is the row holding the
machine-id header, and `group_by` is the column that names each variable's table.
Columns headed Mother, Father, Proband, Sibling, Family, or other become each
feature's applicable roles. The `roles` object maps each role to its physical folder
or folders under a version directory, so one table row is created per (table, role)
where a CSV exists.

```json
{
  "name": "mycohort",
  "display_name": "My cohort phenotype dataset",
  "container": "mycohort",
  "layout": "single_sheet",
  "version_pattern": "Release_(?P<v>\\d{4}-\\d{2}-\\d{2})",
  "dictionary_glob": "**/*Dictionary*.xlsx",
  "sheet": "Variables",
  "group_by": "table.name",
  "fieldid_row": 0,
  "columns": {
    "table_name": ["table.name"],
    "name": ["variable"],
    "definition": ["description"],
    "value_coding": ["allowed values"]
  },
  "roles": {
    "proband": "Proband"
  }
}
```

The shipped `spark.json` and `ssc.json` are complete, working examples to copy from.

## 3. Add synonyms (optional)

Search recall improves with domain synonyms. Append your own to
`<repo-root>/synonyms.json` as an array of equivalent-term groups, and re-ingest. See
[Synonyms](synonyms.md) for the full explanation.

## 4. Verify

```bash
uv run dscat ingest -d mycohort
uv run dscat datasets
uv run dscat tables -d mycohort
```
