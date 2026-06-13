# dscat

A searchable catalogue over versioned tabular research datasets. The `dscat`
command indexes each dataset's data dictionary into a local SQLite and full-text
database, so you can list tables, search features, and read documentation without
opening the multi-gigabyte data files.

## Overview

The SPARK and SSC phenotype datasets are too large to read directly: a single CSV
can reach hundreds of thousands of rows and hundreds of columns. The meaning of
each column lives in the dataset's data dictionary. `dscat ingest` parses those
dictionaries into `.catalogue/index.db`, and the read commands query that index,
returning a few rows at a time.

A typical session searches for a feature by meaning, reads its metadata card to
find the column name and source file, and only then touches the data:

```bash
uv run dscat ingest                 # build the catalogue from data/
uv run dscat search "sleep problems"
uv run dscat feature scq.q01_phrases
```

## Guides

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Using the CLI
:link: guides/cli
:link-type: doc

The command set, the scope flags, and the search-then-read workflow.
:::

:::{grid-item-card} Adding a dataset version
:link: guides/adding-a-version
:link-type: doc

Drop in a new SPARK or SSC release and compare it against the previous one.
:::

:::{grid-item-card} Adding a new dataset
:link: guides/adding-a-dataset
:link-type: doc

Write a JSON adapter for a dataset with a different dictionary layout.
:::

:::{grid-item-card} Synonyms
:link: guides/synonyms
:link-type: doc

How search expands query terms, and how to add your own synonym groups.
:::

::::

## Reference

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Python API
:link: reference/index
:link-type: doc

Every module, class, and function in the `dscat` package.
:::

:::{grid-item-card} Catalogue schema
:link: reference/schema
:link-type: doc

The SQLite tables, columns, and indexes the catalogue is built from.
:::

::::

:::{toctree}
:hidden:
:caption: Guides

guides/cli
guides/adding-a-version
guides/adding-a-dataset
guides/synonyms
:::

:::{toctree}
:hidden:
:caption: Reference

reference/index
reference/schema
:::
