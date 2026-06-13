# Synonyms

`dscat search` expands each query term through a synonym table before it runs the
full-text match, so a search finds features that use a different word for the same
idea. Searching `sleep` also matches insomnia and naps; searching `iq` also matches
cognitive, Mullen, and WISC.

## How expansion works

A query is split into terms. Each term becomes an OR-group of the term and its
synonyms, and the groups are combined with AND. So `sleep problems` becomes, roughly:

```text
(sleep* OR insomnia* OR naps*) AND (problems*)
```

The expanded expression is matched with SQLite FTS5 (BM25 ranking), and results are
ordered best first. Expansion improves recall: you find the relevant features without
having to guess the exact wording the dataset used.

To search without expansion, pass `--raw`, which sends your text straight to FTS5 as a
MATCH expression (useful for `OR`, `NEAR`, prefix `*`, and column filters).

## The synonym file

Synonyms live in a JSON file: an array of groups, where each group is an array of
equivalent terms.

```json
[
  ["sleep", "insomnia", "somnolence", "bedtime", "nap"],
  ["iq", "cognitive", "intelligence", "mullen", "wisc"]
]
```

A group is bidirectional, so every term expands to every other and the first term is
simply a convenient label. Terms are lower-cased and matched against single query
tokens, so they apply to words rather than phrases. Keep entries lexical (word stems
the full-text tokeniser will match), rather than building an exhaustive ontology.

## Built-in and project synonyms

Two files are read, in order:

1. The built-in groups that ship with the package, at
   `packages/dscat/src/dscat/synonyms.json`.
2. An optional project file at `<repo-root>/synonyms.json`, whose groups are appended to
   the built-ins.

To add your own, create or edit `<repo-root>/synonyms.json`, add a group as an array of
terms, and re-run `uv run dscat ingest` to load them. The synonym table is rebuilt on
each ingest.

## A note on spelling

The datasets ship American spellings in their variable names (for example `behavior`),
so the built-in groups bridge British and American forms where that helps recall.
Searching `behaviour` still matches the `behavior` variables, because the two sit in the
same group.
