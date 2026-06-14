"""Locating the monorepo root and the cached-artefact paths.

Each pipeline stage writes its outputs and a manifest under
``<root>/artefacts/<stage>/<run-hash>/``, where the run hash is a content hash over the
inputs that determine the output (dataset version, feature list, hyperparameters, seed,
stratum definition, and the package commit). The directory is gitignored, because it
holds participant-derived intermediates that the SFARI consent does not allow into the
committed history.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (or the working directory) to the monorepo root.

    Honours ``ANALYSIS_ROOT`` if set. Otherwise the root is the nearest ancestor that
    holds a ``data/`` directory or a ``pyproject.toml`` declaring a uv workspace.

    Parameters
    ----------
    start : Path, optional
        Directory to start the search from. Defaults to the working directory.

    Returns
    -------
    Path
        The resolved monorepo root.
    """
    env = os.environ.get("ANALYSIS_ROOT")
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


def artefacts_dir(root: Path) -> Path:
    """Return the cached-artefacts directory, ``<root>/artefacts``."""
    return root / "artefacts"


def stage_dir(root: Path, stage: str) -> Path:
    """Return the directory holding every run of one stage, ``<root>/artefacts/<stage>``."""
    return artefacts_dir(root) / stage


def run_dir(root: Path, stage: str, run_hash: str) -> Path:
    """Return one run's directory, ``<root>/artefacts/<stage>/<run-hash>``."""
    return stage_dir(root, stage) / run_hash


def manifest_path(run_dir: Path) -> Path:
    """Return the manifest path for a run, ``<run-dir>/manifest.json``."""
    return run_dir / "manifest.json"
