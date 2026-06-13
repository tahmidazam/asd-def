"""The normalised SQLite catalogue: declarative schema, writes, and the FTS5 index.

Built with SQLAlchemy Core: tables are declared once as typed ``Table`` objects and
referenced by column (``feature_t.c.name``) everywhere, so queries compose instead of
concatenating strings. ``(dataset, version)`` pairs are first-class on every content table.
``feature_fts`` is a SQLite-specific external-content FTS5 table (BM25 + porter
stemming), created via raw DDL and rebuilt wholesale on ingest.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    Column,
    Engine,
    Index,
    Integer,
    MetaData,
    RowMapping,
    Text,
    create_engine,
    delete,
    distinct,
    func,
    insert,
    select,
    text,
)
from sqlalchemy import (
    Table as SATable,
)

from dscat.model import FeatureRow, TableRow

metadata = MetaData()

dataset_t = SATable(
    "dataset",
    metadata,
    Column("name", Text, primary_key=True),
    Column("display_name", Text),
)

version_t = SATable(
    "version",
    metadata,
    Column("dataset", Text, primary_key=True),
    Column("version", Text, primary_key=True),
    Column("ship_folder", Text),
    Column("dictionary_path", Text),
    Column("ingested_at", Text),
)

tbl_t = SATable(
    "tbl",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("dataset", Text),
    Column("version", Text),
    Column("table_name", Text),
    Column("display_title", Text),
    Column("role", Text),
    Column("file_path", Text),
    Column("n_rows", Integer),
    Column("n_cols", Integer),
    Column("file_bytes", Integer),
    Column("notes", Text),
)
Index("ix_tbl_lookup", tbl_t.c.dataset, tbl_t.c.version, tbl_t.c.table_name, tbl_t.c.role)

feature_t = SATable(
    "feature",
    metadata,
    Column("feature_uid", Integer, primary_key=True),
    Column("dataset", Text),
    Column("version", Text),
    Column("table_name", Text),
    Column("name", Text),
    Column("qualified_id", Text),
    Column("definition", Text),
    Column("field_type", Text),
    Column("measurement_scale", Text),
    Column("value_coding", Text),
    Column("notes", Text),
    Column("display_title", Text),
    Column("display_hint", Text),
    Column("roles_applicable", Text),
    Column("source_sheet", Text),
)
Index(
    "ix_feature_lookup",
    feature_t.c.dataset,
    feature_t.c.version,
    feature_t.c.table_name,
    feature_t.c.name,
)
Index("ix_feature_name", feature_t.c.name)

document_t = SATable(
    "document",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("dataset", Text),
    Column("version", Text),
    Column("path", Text),
    Column("kind", Text),
    Column("title", Text),
    Column("md_path", Text),
)

synonym_t = SATable(
    "synonym",
    metadata,
    Column("term", Text),
    Column("expansion", Text),
)

# FTS5 is a SQLite extension with no SQLAlchemy schema construct, so declare its
# DDL directly. External content (content='feature') keeps the index in sync from
# the feature table via the 'rebuild' command.
_FTS_CREATE = text(
    "CREATE VIRTUAL TABLE feature_fts USING fts5("
    "name, display_title, definition, value_coding, notes, "
    "content='feature', content_rowid='feature_uid', tokenize='porter unicode61')"
)
_FTS_REBUILD = text("INSERT INTO feature_fts(feature_fts) VALUES('rebuild')")


class Catalogue:
    """A connection to the SQLite catalogue index.

    Wraps a SQLAlchemy engine and an open connection, and exposes typed write and
    read helpers over the declarative schema in this module. Open one with
    :meth:`open`, then build the schema on a fresh database with
    :meth:`init_schema`.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.conn = engine.connect()

    @classmethod
    def open(cls, path: Path, create: bool = False) -> Catalogue:
        """Open the catalogue at ``path``.

        Parameters
        ----------
        path : Path
            Path to the SQLite index file.
        create : bool, default False
            When ``True``, create a fresh database, replacing any existing file.
            When ``False``, raise if the file does not exist.

        Returns
        -------
        Catalogue
            An open catalogue.

        Raises
        ------
        FileNotFoundError
            When ``create`` is ``False`` and no index exists at ``path``.
        """
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.unlink()
        elif not path.exists():
            raise FileNotFoundError(f"No catalogue index at {path}. Run `dscat ingest` first.")
        return cls(create_engine(f"sqlite:///{path}"))

    def init_schema(self) -> None:
        """Create every table and the full-text index on a fresh database."""
        metadata.create_all(self.conn)
        self.conn.execute(_FTS_CREATE)
        self.conn.commit()

    def commit(self) -> None:
        """Commit the current transaction."""
        self.conn.commit()

    # ---- writes ---------------------------------------------------------
    def upsert_dataset(self, name: str, display_name: str) -> None:
        """Insert or replace a dataset row."""
        self.conn.execute(
            insert(dataset_t).prefix_with("OR REPLACE").values(name=name, display_name=display_name)
        )

    def upsert_version(
        self, dataset: str, version: str, ship_folder: str, dictionary_path: str
    ) -> None:
        """Insert or replace a version row, stamping the current ingest time."""
        self.conn.execute(
            insert(version_t)
            .prefix_with("OR REPLACE")
            .values(
                dataset=dataset,
                version=version,
                ship_folder=ship_folder,
                dictionary_path=dictionary_path,
                ingested_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
        )

    def clear_dataset(self, dataset: str) -> None:
        """Delete every version, table, feature, and document row for a dataset."""
        for t in (version_t, tbl_t, feature_t, document_t):
            self.conn.execute(delete(t).where(t.c.dataset == dataset))

    def insert_features(self, rows: Iterable[FeatureRow]) -> None:
        """Bulk-insert feature rows."""
        if data := [asdict(r) for r in rows]:
            self.conn.execute(insert(feature_t), data)

    def insert_tables(self, rows: Iterable[TableRow]) -> None:
        """Bulk-insert table rows."""
        if data := [asdict(r) for r in rows]:
            self.conn.execute(insert(tbl_t), data)

    def insert_synonyms(self, pairs: Iterable[tuple[str, str]]) -> None:
        """Replace the synonym table with ``(term, expansion)`` pairs."""
        self.conn.execute(delete(synonym_t))
        if data := [{"term": t, "expansion": e} for t, e in pairs]:
            self.conn.execute(insert(synonym_t), data)

    def insert_documents(self, rows: Iterable[tuple[str, str, str, str, str]]) -> None:
        """Bulk-insert document rows from ``(dataset, version, path, kind, title)`` tuples."""
        cols = ("dataset", "version", "path", "kind", "title")
        if data := [dict(zip(cols, r, strict=True)) for r in rows]:
            self.conn.execute(insert(document_t), data)

    def rebuild_fts(self) -> None:
        """Rebuild the full-text index from the feature table."""
        self.conn.execute(_FTS_REBUILD)

    # ---- reads ----------------------------------------------------------
    def datasets(self) -> Sequence[RowMapping]:
        """Return each dataset with its version and feature counts."""
        n_versions = (
            select(func.count())
            .select_from(version_t)
            .where(version_t.c.dataset == dataset_t.c.name)
            .scalar_subquery()
        )
        n_features = (
            select(func.count())
            .select_from(feature_t)
            .where(feature_t.c.dataset == dataset_t.c.name)
            .scalar_subquery()
        )
        stmt = select(
            dataset_t.c.name,
            dataset_t.c.display_name,
            n_versions.label("n_versions"),
            n_features.label("n_features"),
        ).order_by(dataset_t.c.name)
        return self.conn.execute(stmt).mappings().all()

    def versions(self, dataset: str | None = None) -> Sequence[RowMapping]:
        """Return versions, newest first, with table and feature counts.

        Parameters
        ----------
        dataset : str, optional
            Limit to one dataset; ``None`` returns versions for all.
        """
        n_tables = (
            select(func.count(distinct(tbl_t.c.table_name)))
            .where(tbl_t.c.dataset == version_t.c.dataset, tbl_t.c.version == version_t.c.version)
            .scalar_subquery()
        )
        n_features = (
            select(func.count())
            .select_from(feature_t)
            .where(
                feature_t.c.dataset == version_t.c.dataset,
                feature_t.c.version == version_t.c.version,
            )
            .scalar_subquery()
        )
        stmt = select(
            version_t.c.dataset,
            version_t.c.version,
            version_t.c.ship_folder,
            version_t.c.ingested_at,
            n_tables.label("n_tables"),
            n_features.label("n_features"),
        )
        if dataset:
            stmt = stmt.where(version_t.c.dataset == dataset)
        stmt = stmt.order_by(version_t.c.dataset, version_t.c.version.desc())
        return self.conn.execute(stmt).mappings().all()

    def latest_version(self, dataset: str) -> str | None:
        """Return the newest version id for a dataset, or ``None`` when none exist."""
        versions = (
            self.conn.execute(select(version_t.c.version).where(version_t.c.dataset == dataset))
            .scalars()
            .all()
        )
        if not versions:
            return None
        from dscat.config import version_sort_key

        return max(versions, key=version_sort_key)
