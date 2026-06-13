"""Read queries over the catalogue, built with the SQLAlchemy Core expression language.

Default scope is the *latest* version of each dataset; callers opt into a pinned
``--version`` or ``--all-versions`` (resolved to a set of (dataset, version) pairs,
applied as a single ``_scope_clause``). Search expands query terms through the synonym
table and runs an FTS5 BM25 match (lower rank = better).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from sqlalchemy import (
    ColumnElement,
    RowMapping,
    and_,
    column,
    distinct,
    func,
    literal_column,
    or_,
    select,
    table,
    text,
)

from dscat.config import version_sort_key
from dscat.index import Catalogue, document_t, feature_t, synonym_t, tbl_t, version_t

# Lightweight handle on the FTS5 virtual table (not part of the declarative metadata)
# so it can be joined to feature on rowid = feature_uid.
feature_fts = table("feature_fts", column("rowid"))


def latest_version_map(cat: Catalogue, dataset: str | None = None) -> dict[str, str]:
    """Map each dataset to its newest version id.

    Parameters
    ----------
    cat : Catalogue
        Open catalogue to read from.
    dataset : str, optional
        Limit to one dataset; ``None`` covers all.

    Returns
    -------
    dict of str to str
        Dataset id mapped to its latest version.
    """
    stmt = select(version_t.c.dataset, version_t.c.version)
    if dataset:
        stmt = stmt.where(version_t.c.dataset == dataset)
    latest: dict[str, str] = {}
    for row in cat.conn.execute(stmt):
        if row.dataset not in latest or version_sort_key(row.version) > version_sort_key(
            latest[row.dataset]
        ):
            latest[row.dataset] = row.version
    return latest


def _scope_pairs(
    cat: Catalogue, dataset: str | None, version: str | None, all_versions: bool
) -> list[tuple[str, str]]:
    """Return the (dataset, version) pairs a query should consider."""
    if all_versions:
        stmt = select(version_t.c.dataset, version_t.c.version).distinct()
        if dataset:
            stmt = stmt.where(version_t.c.dataset == dataset)
        return [(r.dataset, r.version) for r in cat.conn.execute(stmt)]
    if version:
        if not dataset:
            raise ValueError("--version requires --dataset")
        return [(dataset, version)]
    return list(latest_version_map(cat, dataset).items())


def _scope_clause(
    dcol: ColumnElement[str], vcol: ColumnElement[str], pairs: list[tuple[str, str]]
) -> ColumnElement[bool]:
    return or_(*(and_(dcol == d, vcol == v) for d, v in pairs))


# ---- tables ------------------------------------------------------------------
def list_tables(
    cat: Catalogue,
    dataset: str | None,
    version: str | None,
    all_versions: bool,
    role: str | None,
    grep: str | None,
) -> Sequence[RowMapping]:
    """List tables in scope, one aggregated row per logical table.

    SSC's per-role CSVs are folded into a single row whose ``roles`` lists the
    roles and whose ``n_rows`` sums across them.

    Parameters
    ----------
    cat : Catalogue
        Open catalogue to read from.
    dataset : str or None
        Restrict to one dataset.
    version : str or None
        Pin a version (needs ``dataset``); the default scope is the latest.
    all_versions : bool
        Cover every version instead of only the latest.
    role : str or None
        Keep only this SSC family role.
    grep : str or None
        Keep only tables whose name or title contains this substring.

    Returns
    -------
    sequence of RowMapping
        One row per table with ``dataset``, ``version``, ``table_name``,
        ``display_title``, ``roles``, ``n_rows``, ``n_cols``, and ``n_files``.
    """
    pairs = _scope_pairs(cat, dataset, version, all_versions)
    if not pairs:
        return []
    stmt = (
        select(
            tbl_t.c.dataset,
            tbl_t.c.version,
            tbl_t.c.table_name,
            func.max(tbl_t.c.display_title).label("display_title"),
            func.group_concat(distinct(tbl_t.c.role)).label("roles"),
            func.sum(tbl_t.c.n_rows).label("n_rows"),
            func.max(tbl_t.c.n_cols).label("n_cols"),
            func.count().label("n_files"),
        )
        .where(_scope_clause(tbl_t.c.dataset, tbl_t.c.version, pairs))
        .group_by(tbl_t.c.dataset, tbl_t.c.version, tbl_t.c.table_name)
        .order_by(tbl_t.c.dataset, tbl_t.c.version.desc(), tbl_t.c.table_name)
    )
    if role:
        stmt = stmt.where(tbl_t.c.role == role)
    if grep:
        like = f"%{grep}%"
        stmt = stmt.where(or_(tbl_t.c.table_name.like(like), tbl_t.c.display_title.like(like)))
    return cat.conn.execute(stmt).mappings().all()


# ---- describe ----------------------------------------------------------------
def describe(
    cat: Catalogue,
    table_name: str,
    dataset: str | None,
    version: str | None,
    limit: int,
    offset: int,
) -> tuple[Sequence[RowMapping], Sequence[RowMapping], int]:
    """Return a table's rows, a page of its features, and the total feature count.

    Parameters
    ----------
    cat : Catalogue
        Open catalogue to read from.
    table_name : str
        Logical table to describe.
    dataset : str or None
        Restrict to one dataset.
    version : str or None
        Pin a version (needs ``dataset``); the default scope is the latest.
    limit : int
        Maximum number of feature rows to return.
    offset : int
        Number of feature rows to skip, for paging.

    Returns
    -------
    tuple
        ``(tables, features, total)``: the matching table rows, one page of
        feature rows, and the total feature count for the table.
    """
    pairs = _scope_pairs(cat, dataset, version, all_versions=False)
    tbls = (
        cat.conn.execute(
            select(tbl_t)
            .where(
                _scope_clause(tbl_t.c.dataset, tbl_t.c.version, pairs),
                tbl_t.c.table_name == table_name,
            )
            .order_by(tbl_t.c.dataset, tbl_t.c.version.desc(), tbl_t.c.role)
        )
        .mappings()
        .all()
    )
    fscope = _scope_clause(feature_t.c.dataset, feature_t.c.version, pairs)
    total = cat.conn.execute(
        select(func.count())
        .select_from(feature_t)
        .where(fscope, feature_t.c.table_name == table_name)
    ).scalar_one()
    feats = (
        cat.conn.execute(
            select(
                feature_t.c.name,
                feature_t.c.field_type,
                feature_t.c.measurement_scale,
                feature_t.c.definition,
                feature_t.c.value_coding,
                feature_t.c.display_title,
                feature_t.c.display_hint,
                feature_t.c.roles_applicable,
            )
            .where(fscope, feature_t.c.table_name == table_name)
            .order_by(feature_t.c.feature_uid)
            .limit(limit)
            .offset(offset)
        )
        .mappings()
        .all()
    )
    return tbls, feats, total


# ---- search ------------------------------------------------------------------
def _fts_term(t: str) -> str:
    return f'"{t}"' if " " in t else f"{t}*"


def expand_query(cat: Catalogue, query: str, raw: bool) -> str:
    """Build an FTS5 MATCH expression from a free-text query.

    Each query token becomes an OR-group of the token (as a prefix match) and its
    synonyms, and the groups are ANDed together. With ``raw`` set, the query is
    returned unchanged.

    Parameters
    ----------
    cat : Catalogue
        Open catalogue, read for synonym expansions.
    query : str
        Free-text query, or a raw MATCH expression when ``raw`` is set.
    raw : bool
        Return ``query`` unchanged.

    Returns
    -------
    str
        An FTS5 MATCH expression.

    Raises
    ------
    ValueError
        When ``query`` has no searchable tokens.
    """
    if raw:
        return query
    tokens = re.findall(r"[A-Za-z0-9_]+", query.lower())
    if not tokens:
        raise ValueError("empty search query")
    groups: list[str] = []
    for tok in tokens:
        expansions = (
            cat.conn.execute(select(synonym_t.c.expansion).where(synonym_t.c.term == tok))
            .scalars()
            .all()
        )
        members: dict[str, None] = {}
        for m in (tok, *expansions):
            members.setdefault(m, None)
        groups.append("(" + " OR ".join(_fts_term(m) for m in members) + ")")
    return " AND ".join(groups)


def search(
    cat: Catalogue,
    query: str,
    dataset: str | None,
    version: str | None,
    all_versions: bool,
    table_name: str | None,
    scale: str | None,
    limit: int,
    raw: bool,
) -> Sequence[RowMapping]:
    """Full-text search features by name, title, definition, value coding, and notes.

    Query terms expand through the synonym table and run as an FTS5 BM25 match,
    where a lower rank is a better match. Set ``raw`` to pass an FTS5 MATCH
    expression through unchanged.

    Parameters
    ----------
    cat : Catalogue
        Open catalogue to read from.
    query : str
        Search text, or a raw FTS5 MATCH expression when ``raw`` is set.
    dataset : str or None
        Restrict to one dataset.
    version : str or None
        Pin a version (needs ``dataset``); the default scope is the latest.
    all_versions : bool
        Search every version instead of only the latest.
    table_name : str or None
        Restrict to one table.
    scale : str or None
        Keep only features with this field type or measurement scale.
    limit : int
        Maximum number of rows to return.
    raw : bool
        Treat ``query`` as a verbatim FTS5 MATCH expression.

    Returns
    -------
    sequence of RowMapping
        Matching features with their ``rank``, best first.
    """
    pairs = _scope_pairs(cat, dataset, version, all_versions)
    if not pairs:
        return []
    match = expand_query(cat, query, raw)
    rank = func.bm25(literal_column("feature_fts")).label("rank")
    stmt = (
        select(
            feature_t.c.dataset,
            feature_t.c.version,
            feature_t.c.table_name,
            feature_t.c.name,
            feature_t.c.definition,
            feature_t.c.field_type,
            feature_t.c.measurement_scale,
            feature_t.c.value_coding,
            feature_t.c.display_title,
            feature_t.c.roles_applicable,
            rank,
        )
        .select_from(feature_t.join(feature_fts, feature_fts.c.rowid == feature_t.c.feature_uid))
        .where(text("feature_fts MATCH :match"))
        .where(_scope_clause(feature_t.c.dataset, feature_t.c.version, pairs))
        .order_by(rank)
        .limit(limit)
    )
    if table_name:
        stmt = stmt.where(feature_t.c.table_name == table_name)
    if scale:
        stmt = stmt.where(
            or_(feature_t.c.measurement_scale == scale, feature_t.c.field_type == scale)
        )
    return cat.conn.execute(stmt, {"match": match}).mappings().all()


# ---- feature lookup ----------------------------------------------------------
def find_feature(
    cat: Catalogue,
    key: str,
    dataset: str | None,
    version: str | None,
    all_versions: bool,
    table_name: str | None,
) -> Sequence[RowMapping]:
    """Look up features by name, ``table.name``, or qualified id.

    Parameters
    ----------
    cat : Catalogue
        Open catalogue to read from.
    key : str
        A bare variable name, a ``table.name``, or a qualified id.
    dataset : str or None
        Restrict to one dataset.
    version : str or None
        Pin a version (needs ``dataset``); the default scope is the latest.
    all_versions : bool
        Search every version instead of only the latest.
    table_name : str or None
        Restrict to one table.

    Returns
    -------
    sequence of RowMapping
        Every feature row matching ``key`` in scope; more than one row means the
        key is ambiguous.
    """
    pairs = _scope_pairs(cat, dataset, version, all_versions)
    if not pairs:
        return []
    stmt = select(feature_t).where(
        _scope_clause(feature_t.c.dataset, feature_t.c.version, pairs),
        or_(
            feature_t.c.name == key,
            feature_t.c.qualified_id == key,
            feature_t.c.table_name + "." + feature_t.c.name == key,
        ),
    )
    if table_name:
        stmt = stmt.where(feature_t.c.table_name == table_name)
    stmt = stmt.order_by(feature_t.c.dataset, feature_t.c.version.desc(), feature_t.c.table_name)
    return cat.conn.execute(stmt).mappings().all()


def feature_sources(
    cat: Catalogue, dataset: str, version: str, table_name: str
) -> Sequence[RowMapping]:
    """Return up to four ``(role, file_path)`` source rows for a table.

    Parameters
    ----------
    cat : Catalogue
        Open catalogue to read from.
    dataset : str
        Dataset id.
    version : str
        Version id.
    table_name : str
        Logical table whose physical CSVs to list.

    Returns
    -------
    sequence of RowMapping
        The role and file path of each physical CSV backing the table.
    """
    return (
        cat.conn.execute(
            select(tbl_t.c.role, tbl_t.c.file_path)
            .where(
                tbl_t.c.dataset == dataset,
                tbl_t.c.version == version,
                tbl_t.c.table_name == table_name,
            )
            .order_by(tbl_t.c.role)
            .limit(4)
        )
        .mappings()
        .all()
    )


# ---- documents ---------------------------------------------------------------
def list_documents(
    cat: Catalogue, dataset: str | None, version: str | None, all_versions: bool
) -> Sequence[RowMapping]:
    """List non-dictionary documentation files in scope.

    Parameters
    ----------
    cat : Catalogue
        Open catalogue to read from.
    dataset : str or None
        Restrict to one dataset.
    version : str or None
        Pin a version (needs ``dataset``); the default scope is the latest.
    all_versions : bool
        Cover every version instead of only the latest.

    Returns
    -------
    sequence of RowMapping
        Document rows with ``dataset``, ``version``, ``kind``, ``title``, and ``path``.
    """
    pairs = _scope_pairs(cat, dataset, version, all_versions)
    if not pairs:
        return []
    stmt = (
        select(
            document_t.c.dataset,
            document_t.c.version,
            document_t.c.kind,
            document_t.c.title,
            document_t.c.path,
        )
        .where(_scope_clause(document_t.c.dataset, document_t.c.version, pairs))
        .order_by(document_t.c.dataset, document_t.c.version.desc(), document_t.c.title)
    )
    return cat.conn.execute(stmt).mappings().all()


def find_documents(
    cat: Catalogue, name: str, dataset: str | None, version: str | None, all_versions: bool
) -> Sequence[RowMapping]:
    """Find documentation files whose title or path contains every token in ``name``.

    Splitting ``name`` on whitespace lets ``Welcome Packet`` match
    ``..._Welcome_Packet...`` across separator differences in file names.

    Parameters
    ----------
    cat : Catalogue
        Open catalogue to read from.
    name : str
        Whitespace-separated tokens that must all appear in the title or path.
    dataset : str or None
        Restrict to one dataset.
    version : str or None
        Pin a version (needs ``dataset``); the default scope is the latest.
    all_versions : bool
        Cover every version instead of only the latest.

    Returns
    -------
    sequence of RowMapping
        Matching document rows.
    """
    pairs = _scope_pairs(cat, dataset, version, all_versions)
    if not pairs:
        return []
    stmt = select(
        document_t.c.dataset,
        document_t.c.version,
        document_t.c.kind,
        document_t.c.title,
        document_t.c.path,
    ).where(_scope_clause(document_t.c.dataset, document_t.c.version, pairs))
    # Every whitespace-separated token must appear (so "Welcome Packet" matches
    # "..._Welcome_Packet...", tolerating separator differences in file names).
    for tok in name.split() or [name]:
        like = f"%{tok}%"
        stmt = stmt.where(or_(document_t.c.title.like(like), document_t.c.path.like(like)))
    stmt = stmt.order_by(document_t.c.dataset, document_t.c.version.desc(), document_t.c.title)
    return cat.conn.execute(stmt).mappings().all()
