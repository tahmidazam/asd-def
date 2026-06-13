# Catalogue schema

The catalogue is one SQLite database at `.catalogue/index.db`, rebuilt by `dscat ingest`.
`dataset` and `version` are columns on every content table, so a query can scope to a
single release. The tables are declared in `dscat.index` (see
{py:class}`dscat.index.Catalogue`) as
[SQLAlchemy Core](https://docs.sqlalchemy.org/en/20/core/) `Table` objects; this page
describes the shape they produce.

For anything the CLI does not cover, query the database directly:

```bash
sqlite3 .catalogue/index.db "SELECT ... ;"
```

## Tables

```sql
dataset(name, display_name)

version(dataset, version, ship_folder, dictionary_path, ingested_at)

tbl(                          -- one row per physical CSV
  id, dataset, version, table_name, display_title,
  role,                       -- '' for SPARK; proband/mother/father/sibling/mz_twin for SSC
  file_path,                  -- POSIX, relative to the repository root
  n_rows, n_cols, file_bytes, notes)

feature(                      -- one row per dictionary variable, defined once per table
  feature_uid, dataset, version, table_name, name,
  qualified_id, definition, field_type, measurement_scale,
  value_coding, notes, display_title, display_hint,
  roles_applicable,           -- comma-separated family roles (SSC); '' for SPARK
  source_sheet)

document(id, dataset, version, path, kind, title, md_path)

synonym(term, expansion)      -- directed; every member of a group maps to every other
```

`feature_fts` is a SQLite FTS5 virtual table over `name`, `display_title`,
`definition`, `value_coding`, and `notes`. It uses external content (the `feature`
table) and porter stemming, and is rebuilt from `feature` on each ingest. Rank matches
with `bm25(feature_fts)`, where a lower score is a better match.

## Indexes

- `feature(dataset, version, table_name, name)`
- `feature(name)`
- `tbl(dataset, version, table_name, role)`

## Example queries

```sql
-- Features whose value coding mentions a sentinel, in the latest SPARK version
SELECT table_name, name, value_coding FROM feature
WHERE dataset = 'spark' AND version = '2026-03-23' AND value_coding LIKE '%-999%';

-- Which SSC measures exist for fathers
SELECT DISTINCT table_name FROM tbl WHERE dataset = 'ssc' AND role = 'father';

-- The ranked search that `dscat search` runs, by hand
SELECT f.table_name, f.name, bm25(feature_fts) AS rank
FROM feature_fts JOIN feature f ON f.feature_uid = feature_fts.rowid
WHERE feature_fts MATCH 'anxiety OR worry' AND f.dataset = 'ssc'
ORDER BY rank LIMIT 10;
```

The dataclasses written into these tables are {py:class}`dscat.model.FeatureRow` and
{py:class}`dscat.model.TableRow`; the read and write helpers are on
{py:class}`dscat.index.Catalogue`.
