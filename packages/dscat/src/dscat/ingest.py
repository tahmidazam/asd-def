"""Build the catalogue: parse each dataset's dictionary, resolve physical CSVs, index.

Physical-file resolution is generic so it handles both datasets' quirks:
- SPARK sheet names are truncated to Excel's 31-char limit, so a CSV stem is bound
  to its dictionary sheet by exact match or 31-char-prefix; the feature rows are then
  re-keyed to the full CSV stem the user actually sees.
- SSC's same measure repeats across role folders -> one ``tbl`` row per (table, role).
Row counts come from a CRLF-safe byte scan (never parse the 397 MB files).
"""

from __future__ import annotations

import csv
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from tqdm import tqdm

from dscat import adapters
from dscat.config import DatasetConfig, Version, discover_versions, load_configs
from dscat.docs import cache_path, convert_doc, discover_docs, resolve_engine
from dscat.index import Catalogue
from dscat.model import FeatureRow, TableRow
from dscat.paths import index_path
from dscat.synonyms import load_synonyms


@dataclass
class IngestSummary:
    """What one dataset contributed to the index during ingestion.

    Attributes
    ----------
    dataset : str
        Dataset id.
    versions : list of str
        Versions ingested.
    n_features : int
        Total feature rows written.
    n_tables : int
        Total table rows written.
    """

    dataset: str
    versions: list[str]
    n_features: int
    n_tables: int


def count_data_rows(path: Path) -> int:
    """Count the data rows in a CSV by scanning bytes for newlines.

    Counting bytes avoids parsing the file, so it stays fast on the largest data
    CSVs. The header line is excluded.

    Parameters
    ----------
    path : Path
        Path to the CSV.

    Returns
    -------
    int
        Number of rows after the header (never negative).
    """
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            n += chunk.count(b"\n")
    return max(n - 1, 0)  # minus header


def header_ncols(path: Path) -> int:
    """Return the column count from a CSV's header row (0 when the header is blank)."""
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        line = f.readline()
    return len(next(csv.reader([line]))) if line.strip() else 0


def _strip_suffix(stem: str, version: str, enabled: bool) -> str:
    suffix = f"-{version}"
    return stem[: -len(suffix)] if enabled and stem.endswith(suffix) else stem


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _bind(stem: str, sheet_norms: set[str]) -> str | None:
    """Bind a CSV stem to a dictionary sheet name.

    Tolerates Excel's 31-char sheet-name truncation and separator drift across
    versions (e.g. the 2025 sheet ``cbcl1-5`` / ``area deprivation index`` vs the
    CSV stems ``cbcl_1_5`` / ``area_deprivation_index``).
    """
    if stem in sheet_norms:
        return stem
    trunc = [s for s in sheet_norms if len(s) == 31 and stem.startswith(s)]
    if len(trunc) == 1:
        return trunc[0]
    key = _norm_key(stem)
    sep = [s for s in sheet_norms if _norm_key(s) == key]
    if len(sep) == 1:
        return sep[0]
    return None


def _folder_roles(cfg: DatasetConfig) -> dict[str, str]:
    out: dict[str, str] = {}
    for role, folders in cfg.roles.items():
        if isinstance(folders, str):
            folder_list = [folders] if folders else []
        elif isinstance(folders, list):
            folder_list = [str(f) for f in folders]
        else:
            folder_list = []
        for fol in folder_list:
            out[fol.strip().lower()] = role
    return out


def resolve_tables(
    cfg: DatasetConfig,
    version: Version,
    features: list[FeatureRow],
    root: Path,
    progress: Callable[[str], None] | None = None,
) -> list[TableRow]:
    """Create one TableRow per physical CSV; re-key truncated SPARK feature tables.

    ``progress``, when given, is called with each CSV's stem as that file is reached,
    so a caller can drive a progress bar over the (slow) per-file row counting.
    """
    vdir = version.version_dir
    sheet_norms = {f.table_name for f in features}
    folder_role = _folder_roles(cfg)
    canonical: dict[str, str] = {}  # truncated sheet name -> full CSV stem
    tables: list[TableRow] = []
    for csvp in sorted(vdir.glob(cfg.file_glob)):
        stem = _strip_suffix(csvp.stem, version.version, cfg.strip_version_suffix).strip().lower()
        if progress is not None:
            progress(stem)
        rel_parts = csvp.relative_to(vdir).parts[:-1]
        role = next(
            (folder_role[p.strip().lower()] for p in rel_parts if p.strip().lower() in folder_role),
            "",
        )
        bound = _bind(stem, sheet_norms)
        if bound and bound != stem:
            canonical[bound] = stem
        tables.append(
            TableRow(
                dataset=cfg.name,
                version=version.version,
                table_name=stem,
                display_title="",  # filled from the dictionary after re-keying, below
                role=role,
                file_path=csvp.resolve().relative_to(root.resolve()).as_posix(),
                n_rows=count_data_rows(csvp),
                n_cols=header_ncols(csvp),
                notes="" if bound else "no dedicated dictionary sheet",
            )
        )
    # Re-key features whose table_name was a truncated sheet name to the full CSV stem.
    for f in features:
        if f.table_name in canonical:
            f.table_name = canonical[f.table_name]
    return tables


def _tick(bar: tqdm, label: str, name: str) -> None:
    """Advance the ingest bar one file, showing ``label`` and the current table."""
    bar.set_postfix_str(f"{label}: {name}")
    bar.update(1)


def run_ingest(
    root: Path, only: list[str] | None = None, convert_docs: bool = False
) -> list[IngestSummary]:
    """Build or refresh the catalogue index from ``data/``.

    For each configured dataset, discovers its versions, parses their
    dictionaries, resolves the physical CSVs, and writes the dataset, version,
    table, feature, and document rows into the index. Synonyms are reloaded and
    the full-text index is rebuilt at the end. Progress is reported on a single
    stderr progress bar, advancing as each data CSV is scanned for its row count.

    Parameters
    ----------
    root : Path
        Repository root holding ``data/`` and ``.catalogue/``.
    only : list of str, optional
        Restrict ingestion to these dataset ids; ``None`` ingests every
        configured dataset. Datasets not listed keep their existing rows.
    convert_docs : bool, default False
        After the index is committed, convert every discovered document to
        markdown (cached under ``.catalogue/docs/``) with the per-format engines,
        advancing the same progress bar. This is slow, so it is off by default;
        otherwise conversion happens lazily on first read. A document that fails
        to convert is reported and skipped.

    Returns
    -------
    list of IngestSummary
        One summary per ingested dataset.
    """
    cfgs = load_configs(root)
    ip = index_path(root)
    fresh = not ip.exists()
    cat = Catalogue.open(ip, create=fresh)
    if fresh:
        cat.init_schema()

    summaries: list[IngestSummary] = []
    targets = [c for c in cfgs.values() if only is None or c.name in only]
    plan = [(cfg, discover_versions(cfg, root)) for cfg in targets]
    total = sum(
        len(list(v.version_dir.glob(cfg.file_glob)))
        + (len(discover_docs(v.version_dir, root)) if convert_docs else 0)
        for cfg, versions in plan
        for v in versions
    )
    pending_docs: list[tuple[str, str, str]] = []  # (dataset, version, relative path)
    with tqdm(total=total, desc="ingesting", unit="file") as bar:
        for cfg, versions in plan:
            cat.clear_dataset(cfg.name)
            cat.upsert_dataset(cfg.name, cfg.display_name)
            seen_versions: list[str] = []
            n_feat = n_tbl = 0
            for v in versions:
                label = f"{cfg.name}/{v.version}"
                bar.set_postfix_str(f"{label}: reading dictionary")
                features, display = adapters.parse(cfg, v)
                report = partial(_tick, bar, label)
                tables = resolve_tables(cfg, v, features, root, progress=report)
                for t in tables:  # fill display titles now that features are re-keyed
                    t.display_title = display.get(t.table_name, t.display_title or "")
                cat.upsert_version(cfg.name, v.version, v.ship_folder, str(v.dictionary_path or ""))
                cat.insert_features(features)
                cat.insert_tables(tables)
                docs_found = discover_docs(v.version_dir, root)
                cat.insert_documents(
                    (cfg.name, v.version, path, kind, title) for path, kind, title in docs_found
                )
                if convert_docs:
                    pending_docs += [(cfg.name, v.version, path) for path, _, _ in docs_found]
                seen_versions.append(v.version)
                n_feat += len(features)
                n_tbl += len(tables)
            summaries.append(IngestSummary(cfg.name, seen_versions, n_feat, n_tbl))
        bar.set_postfix_str("rebuilding search index")
        cat.insert_synonyms(load_synonyms(root))
        cat.rebuild_fts()
        cat.commit()
        for dataset, version, relpath in pending_docs:
            src = root / relpath
            name = Path(relpath).name
            bar.set_postfix_str(f"{dataset}/{version}: {name} ({resolve_engine(src)})")
            try:
                convert_doc(src, cache_path(root, dataset, version, src))
            except RuntimeError as exc:
                tqdm.write(f"skip {name}: {exc}", file=sys.stderr)
            bar.update(1)
    return summaries
