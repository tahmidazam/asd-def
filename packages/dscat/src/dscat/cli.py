"""dscat command-line interface (Typer).

Read commands print an aligned table by default and serialise to JSON, CSV, TSV, or
a Markdown table via ``--format`` (with ``--json`` kept as an alias). The formatting
lives in :mod:`dscat.output`, so the output stays stable and easy to read or parse.
Rich only decorates ``--help``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import typer

from dscat.docs import Engine
from dscat.output import Format, render

app = typer.Typer(
    name="dscat",
    help="Catalogue and search versioned tabular research datasets from their data dictionaries.",
    no_args_is_help=True,
    add_completion=False,
)


# ---- output helpers ----------------------------------------------------------
def _flat(s: str | None, n: int = 80) -> str:
    """Collapse whitespace/newlines to a single line and truncate with an ellipsis."""
    out = " ".join((s or "").split())
    return out if len(out) <= n else out[: n - 1] + "…"


def _type_of(row: Mapping[Any, Any]) -> str:
    """Return a feature's field type, or its measurement scale when the type is blank."""
    return (str(row["field_type"] or "") or str(row["measurement_scale"] or "")).strip()


def _codes(s: str | None) -> str:
    """Inline a value-coding string; treat empty placeholders like ``[]`` as blank."""
    t = (s or "").strip()
    if t in ("", "[]", "[ ]", "()", "[,]"):
        return ""
    return t.replace("\n", "; ")


def _resolve(fmt: Format, json: bool) -> Format:
    """Resolve the effective format, honouring the ``--json`` alias."""
    return Format.json if json else fmt


def _emit(rows: Sequence[Mapping[Any, Any]], fmt: Format, *, empty: str | None = None) -> None:
    """Render ``rows`` in ``fmt`` and print it, or print ``empty`` for an empty table."""
    text = render(rows, fmt)
    if text:
        typer.echo(text)
    elif fmt is Format.table and empty:
        typer.echo(empty)


def _open_catalogue():
    from dscat.index import Catalogue
    from dscat.paths import find_repo_root, index_path

    root = find_repo_root()
    try:
        return Catalogue.open(index_path(root)), root
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@app.command()
def version() -> None:
    """Print the dscat version."""
    from dscat import __version__

    typer.echo(__version__)


@app.command()
def ingest(
    dataset: list[str] = typer.Option(
        None, "--dataset", "-d", help="Only (re)ingest these datasets; default: all."
    ),
    convert_docs: bool = typer.Option(
        False, "--convert-docs", help="Also convert every discovered document to markdown (slow)."
    ),
) -> None:
    """Build/refresh the catalogue index from data/ (auto-discovers dataset versions)."""
    from dscat.ingest import run_ingest
    from dscat.paths import find_repo_root, index_path

    root = find_repo_root()
    summaries = run_ingest(root, only=list(dataset) if dataset else None, convert_docs=convert_docs)
    rows = [
        {
            "dataset": s.dataset,
            "versions": ", ".join(s.versions) or "(none)",
            "features": s.n_features,
            "tables": s.n_tables,
        }
        for s in summaries
    ]
    typer.echo(render(rows, Format.table))
    typer.echo(f"\nindex: {index_path(root)}")


@app.command()
def datasets(
    fmt: Format = typer.Option(Format.table, "--format", "-f", help="Output format."),
    json: bool = typer.Option(False, "--json", help="Alias for --format json."),
) -> None:
    """List indexed datasets."""
    cat, _ = _open_catalogue()
    rows = [
        {
            "dataset": r["name"],
            "display_name": r["display_name"],
            "versions": r["n_versions"],
            "features": r["n_features"],
        }
        for r in cat.datasets()
    ]
    _emit(rows, _resolve(fmt, json), empty="No datasets indexed. Run `dscat ingest`.")


@app.command()
def versions(
    dataset: str = typer.Option(None, "--dataset", "-d"),
    fmt: Format = typer.Option(Format.table, "--format", "-f", help="Output format."),
    json: bool = typer.Option(False, "--json", help="Alias for --format json."),
) -> None:
    """List versions per dataset (newest first)."""
    cat, _ = _open_catalogue()
    rows = [
        {
            "dataset": r["dataset"],
            "version": r["version"],
            "tables": r["n_tables"],
            "features": r["n_features"],
            "ingested": r["ingested_at"][:10],
        }
        for r in cat.versions(dataset)
    ]
    _emit(rows, _resolve(fmt, json), empty="No versions found.")


@app.command()
def tables(
    dataset: str = typer.Option(None, "--dataset", "-d"),
    version: str = typer.Option(None, "--version", "-v", help="Default: latest."),
    all_versions: bool = typer.Option(False, "--all-versions"),
    role: str = typer.Option(None, "--role", "-r", help="SSC family role, e.g. proband."),
    grep: str = typer.Option(None, "--grep", "-g", help="Filter by table name / title substring."),
    limit: int = typer.Option(200, "--limit", "-n"),
    fmt: Format = typer.Option(Format.table, "--format", "-f", help="Output format."),
    json: bool = typer.Option(False, "--json", help="Alias for --format json."),
) -> None:
    """List tables (dataset, version, name, role(s), rows, cols, title)."""
    from dscat import queries

    out = _resolve(fmt, json)
    cat, _ = _open_catalogue()
    result = queries.list_tables(cat, dataset, version, all_versions, role, grep)
    rows = [
        {
            "dataset": r["dataset"],
            "version": r["version"],
            "table": r["table_name"],
            "roles": r["roles"] or "",
            "rows": r["n_rows"],
            "cols": r["n_cols"],
            "title": r["display_title"] or "",
        }
        for r in result[:limit]
    ]
    _emit(rows, out, empty="No tables match.")
    if out is Format.table and len(result) > limit:
        typer.echo(f"\n{len(result) - limit} more; narrow with --grep/--role or raise --limit.")


@app.command()
def describe(
    table: str,
    dataset: str = typer.Option(None, "--dataset", "-d"),
    version: str = typer.Option(None, "--version", "-v", help="Default: latest."),
    limit: int = typer.Option(60, "--limit", "-n"),
    offset: int = typer.Option(0, "--offset"),
    fmt: Format = typer.Option(Format.table, "--format", "-f", help="Output format."),
    json: bool = typer.Option(False, "--json", help="Alias for --format json."),
) -> None:
    """Show a table's row and column stats and its features (name, type, short def)."""
    from dscat import queries

    out = _resolve(fmt, json)
    cat, _ = _open_catalogue()
    tbls, feats, total = queries.describe(cat, table, dataset, version, limit, offset)
    rows = [
        {
            "name": f["name"],
            "type": _type_of(f),
            "definition": _flat(f["definition"] or f["display_title"], 80),
        }
        for f in feats
    ]
    if out is not Format.table:
        _emit(rows, out)
        return
    if not tbls and not feats:
        typer.echo(f"No table '{table}' found (scope: {dataset or 'all datasets'}, latest).")
        return
    for t in tbls:
        role = f"  role={t['role']}" if t["role"] else ""
        title = f' "{_flat(t["display_title"], 50)}"' if t["display_title"] else ""
        typer.echo(
            f"{t['dataset']}/{t['version']}  {t['table_name']}{title}{role}  "
            f"({t['n_rows']} rows, {t['n_cols']} cols)  {t['file_path']}"
        )
    typer.echo(f"\nfeatures ({offset + 1}-{offset + len(feats)} of {total}):")
    if rows:
        typer.echo(render(rows, Format.table))
    if offset + len(feats) < total:
        typer.echo(f"\n{total - offset - len(feats)} more; page with --offset {offset + limit}.")


@app.command()
def search(
    query: str,
    dataset: str = typer.Option(None, "--dataset", "-d"),
    version: str = typer.Option(None, "--version", "-v", help="Default: latest."),
    all_versions: bool = typer.Option(False, "--all-versions"),
    table: str = typer.Option(None, "--table", "-t"),
    scale: str = typer.Option(None, "--scale", "-s", help="Filter by type/measurement scale."),
    limit: int = typer.Option(25, "--limit", "-n"),
    raw: bool = typer.Option(False, "--raw", help="Pass the FTS5 MATCH expression verbatim."),
    fmt: Format = typer.Option(Format.table, "--format", "-f", help="Output format."),
    json: bool = typer.Option(False, "--json", help="Alias for --format json."),
) -> None:
    """Fuzzy-search features by name/definition/value-coding (FTS5 + synonyms)."""
    from dscat import queries

    out = _resolve(fmt, json)
    cat, _ = _open_catalogue()
    try:
        result = queries.search(
            cat, query, dataset, version, all_versions, table, scale, limit, raw
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    rows = [
        {
            "dataset": r["dataset"],
            "version": r["version"],
            "table": r["table_name"],
            "name": r["name"],
            "type": _type_of(r),
            "definition": r["definition"] or r["display_title"] or _codes(r["value_coding"]),
        }
        for r in result
    ]
    _emit(rows, out, empty=f"No matches for '{query}'. Try fewer/other terms or --all-versions.")
    if out is Format.table and len(result) == limit:
        typer.echo(f"\n(showing {limit}; refine with --table/--scale or raise --limit)")


@app.command()
def feature(
    key: str,
    dataset: str = typer.Option(None, "--dataset", "-d"),
    version: str = typer.Option(None, "--version", "-v", help="Default: latest."),
    all_versions: bool = typer.Option(False, "--all-versions"),
    table: str = typer.Option(None, "--table", "-t"),
    fmt: Format = typer.Option(Format.table, "--format", "-f", help="Output format."),
    json: bool = typer.Option(False, "--json", help="Alias for --format json."),
) -> None:
    """Show one feature's full metadata card. KEY is a name, table.name, or qualified id."""
    from dscat import queries

    out = _resolve(fmt, json)
    cat, _ = _open_catalogue()
    rows = queries.find_feature(cat, key, dataset, version, all_versions, table)
    if out is not Format.table:
        _emit(rows, out)
        return
    if not rows:
        typer.echo(f"No feature '{key}' found. Try `dscat search {key}` or --all-versions.")
        return
    if len(rows) > 1:
        typer.echo(f"{len(rows)} matches for '{key}'; disambiguate with --dataset/--table:")
        for r in rows:
            typer.echo(f"  {r['dataset']}/{r['version']}  {r['table_name']}.{r['name']}")
        return
    r = rows[0]
    typer.echo(f"{r['dataset']}/{r['version']}  {r['table_name']}.{r['name']}")

    def line(label: str, value: str | None) -> None:
        if value and value.strip():
            typer.echo(f"  {label:<11}: {_flat(value, 200)}")

    line("display", r["display_title"])
    line("hint", r["display_hint"])
    line("definition", r["definition"])
    line("type", r["field_type"])
    line("scale", r["measurement_scale"])
    line("values", _codes(r["value_coding"]))
    line("roles", r["roles_applicable"])
    line("qualified", r["qualified_id"])
    line("notes", r["notes"])
    for s in queries.feature_sources(cat, r["dataset"], r["version"], r["table_name"]):
        tag = f" ({s['role']})" if s["role"] else ""
        line("source", f"{s['file_path']}{tag}")


@app.command()
def diff(
    dataset: str = typer.Option(..., "--dataset", "-d"),
    from_: str = typer.Option(..., "--from", help="Older version."),
    to: str = typer.Option(..., "--to", help="Newer version."),
    tables_only: bool = typer.Option(False, "--tables", help="Only table-level changes."),
    features_only: bool = typer.Option(False, "--features", help="Only feature-level changes."),
    limit: int = typer.Option(40, "--limit", "-n"),
    fmt: Format = typer.Option(Format.table, "--format", "-f", help="Output format."),
    json: bool = typer.Option(False, "--json", help="Alias for --format json."),
) -> None:
    """Diff a dataset's dictionary between two versions (added / removed / changed)."""
    from dscat import diff as diffmod

    out = _resolve(fmt, json)
    cat, _ = _open_catalogue()
    res = diffmod.diff_versions(cat, dataset, from_, to)
    changes: list[dict[str, str]] = []
    if not features_only:
        changes += [
            {"change": "table_added", "table": t, "feature": "", "detail": ""}
            for t in res.tables_added
        ]
        changes += [
            {"change": "table_removed", "table": t, "feature": "", "detail": ""}
            for t in res.tables_removed
        ]
    if not tables_only:
        changes += [
            {"change": "added", "table": t, "feature": n, "detail": ""} for t, n in res.added
        ]
        changes += [
            {"change": "removed", "table": t, "feature": n, "detail": ""} for t, n in res.removed
        ]
        changes += [
            {"change": "changed", "table": t, "feature": n, "detail": w} for t, n, w in res.changed
        ]
    if out is not Format.table:
        _emit(changes, out)
        return
    typer.echo(f"{dataset}: {from_} to {to}")
    if not features_only:
        typer.echo(f"tables: +{len(res.tables_added)} -{len(res.tables_removed)}")
    if not tables_only:
        typer.echo(f"features: +{len(res.added)} -{len(res.removed)} ~{len(res.changed)}")
    if changes[:limit]:
        typer.echo(render(changes[:limit], Format.table))
    if len(changes) > limit:
        typer.echo(f"\n{len(changes) - limit} more; raise --limit or use --format json.")


@app.command()
def docs(
    dataset: str = typer.Option(None, "--dataset", "-d"),
    version: str = typer.Option(None, "--version", "-v", help="Default: latest."),
    all_versions: bool = typer.Option(False, "--all-versions"),
    fmt: Format = typer.Option(Format.table, "--format", "-f", help="Output format."),
    json: bool = typer.Option(False, "--json", help="Alias for --format json."),
) -> None:
    """List non-dictionary documentation (PDF/Word/txt) for each dataset version."""
    from dscat import queries

    cat, _ = _open_catalogue()
    rows = [
        {"dataset": r["dataset"], "version": r["version"], "kind": r["kind"], "title": r["title"]}
        for r in queries.list_documents(cat, dataset, version, all_versions)
    ]
    _emit(rows, _resolve(fmt, json), empty="No documents found.")


@app.command()
def doc(
    name: str,
    dataset: str = typer.Option(None, "--dataset", "-d"),
    version: str = typer.Option(None, "--version", "-v", help="Default: latest."),
    all_versions: bool = typer.Option(False, "--all-versions"),
    engine: Engine | None = typer.Option(
        None, "--engine", "-e", help="Force engine; default: PDF=marker, else markitdown."
    ),
    section: str = typer.Option(None, "--section", "-s", help="Regex; show only matching context."),
    head: int = typer.Option(40, "--head", help="Preview lines when no --section."),
) -> None:
    """Convert a documentation file to markdown (cached) and show a section or preview."""
    from dscat import queries
    from dscat.docs import cache_path, convert_doc, extract_sections, resolve_engine

    cat, root = _open_catalogue()
    matches = queries.find_documents(cat, name, dataset, version, all_versions)
    if not matches:
        typer.echo(f"No document matching '{name}'. Try `dscat docs`.", err=True)
        raise typer.Exit(1)
    if len(matches) > 1:
        typer.echo(f"{len(matches)} documents match '{name}'; be more specific:")
        for m in matches:
            typer.echo(f"  {m['dataset']}/{m['version']}: {m['title']} ({m['kind']})")
        raise typer.Exit(1)
    m = matches[0]
    src = root / m["path"]
    try:
        md = convert_doc(src, cache_path(root, m["dataset"], m["version"], src, engine), engine)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{m['dataset']}/{m['version']}: {m['title']} ({m['kind']})")
    typer.echo(f"markdown ({resolve_engine(src, engine)}): {md}")
    if section:
        blocks = extract_sections(md, section)
        if not blocks:
            typer.echo(f"(no lines match /{section}/)")
            return
        typer.echo(f"\n--- {len(blocks)} section(s) matching /{section}/ ---")
        for b in blocks[:20]:
            typer.echo(b)
            typer.echo("...")
    else:
        lines = md.read_text(encoding="utf-8", errors="replace").splitlines()
        typer.echo(
            f"({len(lines)} lines; first {head} shown; use --section <regex> or read the file)\n"
        )
        typer.echo("\n".join(lines[:head]))


if __name__ == "__main__":
    app()
