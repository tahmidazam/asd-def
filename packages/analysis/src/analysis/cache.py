"""Content-addressed cache: run hashes, manifests, and artefact serialization.

A stage's output is identified by a hash over the inputs that determine it (dataset and
version, the feature-list digest, model hyperparameters, the covariate set, the seed, the
stratum definition, and the package commit). A run is a cache hit when a manifest with the
same hash already exists and finished cleanly, so a later session recomputes only what
changed (plan section 11).

This module holds the hashing, the manifest read and write, the environment capture, and
the serialization helpers. The :mod:`analysis.run` module composes them into the
``run_context`` lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

# Packages whose resolved versions are recorded in every manifest, because mixture fits
# depend on them (plan section 11).
TRACKED_PACKAGES: tuple[str, ...] = (
    "stepmix",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "pyarrow",
)

MANIFEST_NAME = "manifest.json"


def _json_default(obj: object) -> str:
    """Serialise the few non-JSON types that appear in run parameters."""
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def canonical_json(obj: object) -> str:
    """Serialise ``obj`` to a stable JSON string.

    Keys are sorted and whitespace is removed, so the same logical parameters always
    produce the same string and therefore the same hash.

    Parameters
    ----------
    obj : object
        Any JSON-serialisable object, plus :class:`~pathlib.Path` (rendered as its string).

    Returns
    -------
    str
        The canonical JSON encoding.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_default
    )


def compute_hash(params: Mapping[str, Any]) -> str:
    """Return the SHA-256 hex digest of a run's parameters.

    Parameters
    ----------
    params : Mapping
        The inputs that determine the output (plan section 11).

    Returns
    -------
    str
        The 64-character hex digest.
    """
    return hashlib.sha256(canonical_json(params).encode("utf-8")).hexdigest()


def short_hash(full_hash: str, length: int = 16) -> str:
    """Return the leading ``length`` characters of a hash, for directory names."""
    return full_hash[:length]


def file_digest(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes.

    Used to fold the author feature list and the typing manifest into a run hash, so that
    editing either input invalidates the cache.

    Parameters
    ----------
    path : Path
        File to digest.

    Returns
    -------
    str
        The 64-character hex digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_versions() -> dict[str, str]:
    """Return the resolved versions of the packages that fits depend on.

    Returns
    -------
    dict of str to str
        Package name mapped to its installed version, or ``"unknown"`` when the package
        is not installed.
    """
    versions: dict[str, str] = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "unknown"
    return versions


def git_commit(root: Path) -> str:
    """Return the short git commit of the repository, or ``"unknown"``.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    str
        The short commit hash, suffixed with ``"-dirty"`` when the working tree has
        uncommitted changes, or ``"unknown"`` when git is unavailable.
    """
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{rev}-dirty" if dirty else rev
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---- serialization -----------------------------------------------------------
def save_frame(df: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to Parquet, preserving its index."""
    df.to_parquet(path, index=True)


def load_frame(path: Path) -> pd.DataFrame:
    """Read a dataframe from Parquet."""
    return pd.read_parquet(path)


def save_model(obj: object, path: Path) -> None:
    """Persist a fitted model (or any picklable object) with joblib."""
    joblib.dump(obj, path)


def load_model(path: Path) -> Any:
    """Load a joblib-persisted object."""
    return joblib.load(path)


def save_json(obj: object, path: Path) -> None:
    """Write ``obj`` to a human-readable JSON file."""
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


# ---- manifests ---------------------------------------------------------------
def read_manifest(run_dir: Path) -> dict[str, Any] | None:
    """Return a run's manifest, or ``None`` when it does not exist.

    Parameters
    ----------
    run_dir : Path
        The run directory ``artefacts/<stage>/<hash>``.

    Returns
    -------
    dict or None
        The parsed manifest, or ``None`` when absent.
    """
    path = run_dir / MANIFEST_NAME
    if not path.is_file():
        return None
    return load_json(path)


def write_manifest(run_dir: Path, data: Mapping[str, Any]) -> None:
    """Write a run's manifest to ``run_dir/manifest.json``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(dict(data), run_dir / MANIFEST_NAME)
