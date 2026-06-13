# Adding a dataset version

A version is one vendor ship folder, dropped unchanged into `data/<dataset>/`. The
version id comes from the folder's name, so adding a release is two steps: drop the
folder in, then re-ingest.

## Drop the folder in

Place the vendor's ship folder under the dataset's container directory, with its name
unchanged:

```text
data/spark/SPARKDataRelease_2027-03-15/
data/ssc/SSC Version 16.1 Phenotype Dataset/
```

`data/` is gitignored, so releases are never committed. The adapter reads the version
id from the folder name through its `version_pattern`: `SPARKDataRelease_2026-03-23`
becomes `2026-03-23`, and `SSC Version 15.3 Phenotype Dataset` becomes `15.3`. ISO
dates and dotted-numeric versions order on their own, so the latest version is always
well defined.

The dictionary file may be named differently from one release to the next, for
example `SPARK Data Dictionary.xlsx` against `SPARK Data Dictionary-2026-03-23.xlsx`.
The adapter locates it by glob, so no configuration change is needed.

## Re-ingest

```bash
uv run dscat ingest -d spark   # just this dataset
uv run dscat ingest            # or every dataset
```

Confirm the new version is indexed:

```bash
uv run dscat versions -d spark
```

## Compare against the previous version

`diff` reports what changed between two releases:

```bash
uv run dscat diff -d spark --from 2025-03-31 --to 2026-03-23
```

It compares the two versions by `(table, feature)` identity and reports:

- tables added or removed,
- features added (`+`) or removed (`-`),
- features changed (`~`), meaning the definition or value coding differs after
  whitespace and case normalisation. The tag says which changed: `definition`,
  `values`, or both.

Because the comparison normalises whitespace and case, CRLF and trivial spacing never
show as a change. Narrow the report with `--tables` or `--features`, and use `--json`
to capture the full list.

## A note on renamed tables

A table renamed between versions shows as one table removed and one added, and its
features as removed and added rather than changed. Feature identity is `(table, name)`
within a dataset, so a different table name is a different feature.
