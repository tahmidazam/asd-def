"""Cross-version diff of a dataset's dictionary.

Compares two versions by ``(table, feature)`` identity. It reports added and
removed features, and for features present in both whether the definition or
value coding changed (after whitespace and case normalisation, so CRLF noise does
not show as a change).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import distinct, select

from dscat.index import Catalogue, feature_t, tbl_t


def _norm(s: str | None) -> str:
    return " ".join((s or "").split()).lower()


def _features(cat: Catalogue, dataset: str, version: str) -> dict[tuple[str, str], tuple[str, str]]:
    rows = cat.conn.execute(
        select(
            feature_t.c.table_name,
            feature_t.c.name,
            feature_t.c.definition,
            feature_t.c.value_coding,
        ).where(feature_t.c.dataset == dataset, feature_t.c.version == version)
    ).all()
    return {(r.table_name, r.name): (_norm(r.definition), _norm(r.value_coding)) for r in rows}


def _table_names(cat: Catalogue, dataset: str, version: str) -> set[str]:
    rows = cat.conn.execute(
        select(distinct(tbl_t.c.table_name)).where(
            tbl_t.c.dataset == dataset, tbl_t.c.version == version
        )
    )
    return {r.table_name for r in rows}


@dataclass
class DiffResult:
    """The differences between two versions of a dataset's dictionary.

    Attributes
    ----------
    dataset : str
        Dataset id.
    from_v : str
        The older version compared.
    to_v : str
        The newer version compared.
    tables_added : list of str
        Table names present only in the newer version.
    tables_removed : list of str
        Table names present only in the older version.
    added : list of tuple
        ``(table, feature)`` pairs gained in the newer version.
    removed : list of tuple
        ``(table, feature)`` pairs lost in the newer version.
    changed : list of tuple
        ``(table, feature, what)`` for features in both versions whose definition
        or value coding differs; ``what`` is ``"definition"``, ``"values"``, or
        ``"definition+values"``.
    """

    dataset: str
    from_v: str
    to_v: str
    tables_added: list[str] = field(default_factory=list)
    tables_removed: list[str] = field(default_factory=list)
    added: list[tuple[str, str]] = field(default_factory=list)
    removed: list[tuple[str, str]] = field(default_factory=list)
    changed: list[tuple[str, str, str]] = field(default_factory=list)  # (table, name, what)


def diff_versions(cat: Catalogue, dataset: str, from_v: str, to_v: str) -> DiffResult:
    """Compare a dataset's dictionary between two versions.

    Features are matched by ``(table, name)`` identity. A feature in both versions
    counts as changed when its definition or value coding differs after whitespace
    and case normalisation, so CRLF and spacing noise is not reported.

    Parameters
    ----------
    cat : Catalogue
        Open catalogue to read from.
    dataset : str
        Dataset id.
    from_v : str
        Older version id.
    to_v : str
        Newer version id.

    Returns
    -------
    DiffResult
        The added, removed, and changed tables and features.
    """
    a, b = _features(cat, dataset, from_v), _features(cat, dataset, to_v)
    ak, bk = set(a), set(b)
    res = DiffResult(dataset, from_v, to_v)
    res.added = sorted(bk - ak)
    res.removed = sorted(ak - bk)
    for key in sorted(ak & bk):
        if a[key] != b[key]:
            what = []
            if a[key][0] != b[key][0]:
                what.append("definition")
            if a[key][1] != b[key][1]:
                what.append("values")
            res.changed.append((key[0], key[1], "+".join(what)))
    ta, tb = _table_names(cat, dataset, from_v), _table_names(cat, dataset, to_v)
    res.tables_added = sorted(tb - ta)
    res.tables_removed = sorted(ta - tb)
    return res
