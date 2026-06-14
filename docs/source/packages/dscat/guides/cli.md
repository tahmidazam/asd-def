# Using the CLI

`dscat` reads a prebuilt catalogue of the dataset dictionaries. Run every command from
the repository root, where `data/` and `.catalogue/` live, through uv:

```bash
uv run dscat <command> [options]
```

Every read command prints an aligned table by default and can serialise its results
to JSON, CSV, TSV, or a Markdown table with `--format`. See [Output formats](#output-formats).

## Build the catalogue first

The read commands need the index. Build or refresh it from `data/`:

```bash
uv run dscat ingest                # all datasets
uv run dscat ingest -d spark       # one dataset
uv run dscat ingest --convert-docs # also pre-convert every document to markdown
```

`ingest` is safe to re-run: it rebuilds the named datasets and leaves the rest
intact. A single progress bar reports its progress, since scanning the data CSVs for
row counts takes a moment. By default documents are converted to markdown lazily, on
first `dscat doc`; pass `--convert-docs` to convert them all up front instead (slower,
using the same per-format engines), which is handy before working offline.

## The search-then-read workflow

The catalogue holds metadata, not data values. The usual path is:

1. `dscat search "<intent>"` finds candidate features by meaning or keyword.
2. `dscat feature <table.name>` shows one feature's full card, including its
   measurement scale, value codings, and the source CSV path.
3. With the column name and file path from step 2, read just that column from the
   CSV with a streaming tool. Reading the whole file is rarely necessary.

## Commands

| Command | What it does |
| --- | --- |
| `ingest` | Build or refresh the catalogue from `data/`. |
| `datasets` | List indexed datasets with version and feature counts. |
| `versions` | List versions per dataset, newest first. |
| `tables` | List tables: name, role(s), rows, columns, title. |
| `describe <table>` | Show a table's stats and a page of its features. |
| `search <query>` | Full-text feature search with synonym expansion. |
| `feature <key>` | Show one feature's full metadata card. |
| `diff` | Compare a dataset's dictionary between two versions. |
| `docs` | List a version's non-dictionary documentation files. |
| `doc <name>` | Convert a documentation file to markdown and show a section. |

Run `uv run dscat <command> --help` for the full options of any command.

## Scope flags

Read commands default to the latest version of each dataset. Three flags change that
scope:

| Flag | Meaning |
| --- | --- |
| `-d, --dataset NAME` | Limit to one dataset (`spark`, `ssc`). |
| `-v, --version VER` | Pin a version; requires `--dataset`. |
| `--all-versions` | Consider every version, not only the latest. |

Other common filters are `-t/--table`, `-s/--scale`, `-r/--role` (an SSC family
role: proband, mother, father, sibling, mz_twin), `-g/--grep`, and `-n/--limit`.

## Output formats

Read commands print an aligned text table by default. Pick another format with
`-f/--format`:

| Format | Use |
| --- | --- |
| `table` | Aligned columns for reading in a terminal (the default). |
| `json` | A JSON array of records, for scripts and pipelines. |
| `csv` | Comma-separated values with a header row. |
| `tsv` | Tab-separated values with a header row. |
| `markdown` | A GitHub-flavoured Markdown table, for pasting into docs or an issue. |

`--json` is kept as a shortcut for `--format json`. The table and Markdown views
truncate long free-text cells for readability; the `csv`, `tsv`, and `json` formats
carry the full values.

```bash
uv run dscat search "sleep problems" -d spark -f csv > sleep.csv
uv run dscat tables -d ssc -r proband -f markdown
uv run dscat datasets --json
```

## The feature key

`feature` accepts a bare variable name (`q01_phrases`), a `table.name`
(`scq.q01_phrases`), or a qualified id. When a key matches more than one feature, the
command lists the matches so you can narrow with `--dataset` or `--table`.

## Search expressions

`search` expands each query term through a synonym table, so `sleep` also matches
insomnia and naps, and `iq` matches cognitive and Mullen. It then runs an FTS5 BM25
match and orders results best first. Pass `--raw` to send an FTS5 MATCH expression
through unchanged, which lets you use `OR`, `NEAR`, prefix `*`, and column filters
directly. See [Synonyms](synonyms.md) for how the expansion works and how to extend it.

## Converting documents

`docs` lists the non-dictionary files shipped with a version (welcome packets,
protocols, release notes). `doc <name>` converts one of them to Markdown, caches it
under `.catalogue/docs/`, and prints either a preview or the windows of text matching a
regex passed to `-s/--section`.

By default the engine is chosen by file type:

| Format | Engine | Why |
| --- | --- | --- |
| PDF | `marker` | Layout-aware extraction; higher quality, but slower (loads ML models on first use). |
| `.docx`, `.txt`, and the rest | `markitdown` | Fast, with broad format coverage. |
| `.doc`, `.rtf` | `textutil` | The macOS tool; the only one of the three that reads these. |

Pass `-e/--engine` to force `marker` or `markitdown` for a given file (legacy `.doc` and
`.rtf` always use `textutil`). markitdown and marker ship as dependencies, so `uv sync`
installs them; `textutil` is part of macOS. Each engine caches to its own file, so you
can convert the same document with two engines and compare.

```bash
uv run dscat doc "Welcome Packet" -d spark                 # PDF, so marker
uv run dscat doc "Welcome Packet" -d spark -e markitdown   # force the faster engine
```

## Examples

```bash
uv run dscat search "adaptive behaviour" -d ssc -s standard
uv run dscat feature vineland-3.composite_standard_score -d spark
uv run dscat tables -d ssc -r proband -g ados
uv run dscat diff -d spark --from 2025-03-31 --to 2026-03-23 --features -n 100
uv run dscat doc "Welcome Packet" -d spark -s "data access"
```
