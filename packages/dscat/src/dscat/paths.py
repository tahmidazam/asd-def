"""Locating the monorepo root and the generated catalogue paths.

The catalogue (SQLite index + converted docs) lives at ``<root>/.catalogue`` and is
regenerable; raw datasets live (gitignored) at ``<root>/data/<dataset>/<ship-folder>``.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (or CWD) to the workspace root.

    Honours ``DSCAT_ROOT`` if set. Otherwise the root is the nearest ancestor that
    holds a ``data/`` directory or a ``pyproject.toml`` declaring a uv workspace.
    """
    env = os.environ.get("DSCAT_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    start = (start or Path.cwd()).resolve()
    for d in (start, *start.parents):
        if (d / "data").is_dir():
            return d
        pp = d / "pyproject.toml"
        if pp.is_file():
            try:
                if "[tool.uv.workspace]" in pp.read_text(encoding="utf-8"):
                    return d
            except OSError:
                pass
    return start


def data_root(root: Path) -> Path:
    """Return the raw-datasets directory, ``<root>/data``."""
    return root / "data"


def catalogue_dir(root: Path) -> Path:
    """Return the generated-catalogue directory, ``<root>/.catalogue``."""
    return root / ".catalogue"


def index_path(root: Path) -> Path:
    """Return the path to the SQLite index, ``<root>/.catalogue/index.db``."""
    return catalogue_dir(root) / "index.db"


def docs_cache_dir(root: Path) -> Path:
    """Return the converted-documents cache directory, ``<root>/.catalogue/docs``."""
    return catalogue_dir(root) / "docs"
