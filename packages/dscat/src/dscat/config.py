"""Declarative dataset adapters + version discovery.

An adapter config (JSON) describes how to turn one dataset's heterogeneous data
dictionary into the normalised index. Built-in adapters ship in ``dscat/datasets``;
projects may add or override them by dropping ``<root>/datasets/*.json``.

Versions are discovered from the filesystem: each immediate subfolder of
``data/<container>/`` whose name matches ``version_pattern`` is one version (the
vendor ship folder, dropped in unchanged).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path


@dataclass
class DatasetConfig:
    """One dataset's adapter configuration, loaded from a JSON file.

    Maps a dataset's data dictionary onto the normalised index. Built-in configs
    ship in ``dscat/datasets``; a project may override them by name with
    ``<root>/datasets/*.json``.

    Attributes
    ----------
    name : str
        Dataset id, for example ``"spark"``.
    display_name : str
        Human-facing label.
    container : str
        Subdirectory of ``data/`` to scan for versions.
    layout : str
        Layout engine, either ``"sheet_per_table"`` or ``"single_sheet"``.
    version_pattern : str
        Regex with a named group ``(?P<v>...)`` that extracts the version from
        each ship-folder name.
    dictionary_glob : str
        Glob, relative to a version directory, that locates the dictionary file.
    columns : dict of str to list of str
        Canonical field name mapped to header aliases; the first alias that
        matches a header wins.
    skip_sheets : list of str
        For ``sheet_per_table``, dictionary sheets to ignore.
    sheet : str or None
        For ``single_sheet``, the sheet holding every variable.
    group_by : str or None
        For ``single_sheet``, the column that names each variable's table.
    fieldid_row : int
        For ``single_sheet``, the 0-based row index of the machine-id header.
    roles : dict of str to object
        For ``single_sheet``, a role mapped to its physical folder or folders
        under each version directory.
    file_glob : str
        Glob for the data CSVs (default ``**/*.csv``).
    strip_version_suffix : bool
        Whether to strip a trailing ``-<version>`` from CSV stems before binding
        them to dictionary tables.
    """

    name: str
    display_name: str
    container: str
    layout: str  # "sheet_per_table" | "single_sheet"
    version_pattern: str
    dictionary_glob: str
    columns: dict[str, list[str]] = field(default_factory=dict)
    # sheet_per_table
    skip_sheets: list[str] = field(default_factory=list)
    # single_sheet
    sheet: str | None = None
    group_by: str | None = None
    fieldid_row: int = 0  # 0-based row index holding machine column ids
    roles: dict[str, object] = field(default_factory=dict)  # role -> folder | [folders] | ""
    # file resolution
    file_glob: str = "**/*.csv"
    strip_version_suffix: bool = False


@dataclass
class Version:
    """One discovered version of a dataset: a vendor ship folder.

    Attributes
    ----------
    dataset : str
        Dataset id.
    version : str
        Version id parsed from the ship-folder name.
    ship_folder : str
        Name of the vendor folder under ``data/<container>/``.
    version_dir : Path
        Absolute path to that folder.
    dictionary_path : Path or None
        Path to the data dictionary, or ``None`` when none was found.
    """

    dataset: str
    version: str
    ship_folder: str
    version_dir: Path
    dictionary_path: Path | None

    @property
    def sort_key(self) -> tuple[int, tuple[int, ...], str]:
        """Return the key that orders this version (latest compares greatest)."""
        return version_sort_key(self.version)


def version_sort_key(v: str) -> tuple[int, tuple[int, ...], str]:
    """Order versions: ISO dates > dotted-numeric > raw string. Latest = max.

    The fixed (kind, numbers, raw) shape keeps every key mutually comparable.
    """
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", v)
    if m:
        return (2, tuple(int(x) for x in m.groups()), "")
    if re.fullmatch(r"[\d.]+", v):
        return (1, tuple(int(x) for x in v.split(".") if x != ""), "")
    return (0, (), v)


def _parse_cfg(data: dict) -> DatasetConfig:
    return DatasetConfig(
        name=data["name"],
        display_name=data.get("display_name", data["name"]),
        container=data.get("container", data["name"]),
        layout=data["layout"],
        version_pattern=data["version_pattern"],
        dictionary_glob=data["dictionary_glob"],
        columns={k: list(v) for k, v in data.get("columns", {}).items()},
        skip_sheets=list(data.get("skip_sheets", [])),
        sheet=data.get("sheet"),
        group_by=data.get("group_by"),
        fieldid_row=int(data.get("fieldid_row", 0)),
        roles=dict(data.get("roles", {})),
        file_glob=data.get("file_glob", "**/*.csv"),
        strip_version_suffix=bool(data.get("strip_version_suffix", False)),
    )


def load_configs(root: Path) -> dict[str, DatasetConfig]:
    """Built-in adapters, with optional project overrides from ``<root>/datasets``."""
    cfgs: dict[str, DatasetConfig] = {}
    for entry in resources.files("dscat.datasets").iterdir():
        if entry.name.endswith(".json"):
            cfg = _parse_cfg(json.loads(entry.read_text(encoding="utf-8")))
            cfgs[cfg.name] = cfg
    proj = root / "datasets"
    if proj.is_dir():
        for f in sorted(proj.glob("*.json")):
            cfg = _parse_cfg(json.loads(f.read_text(encoding="utf-8")))
            cfgs[cfg.name] = cfg
    return cfgs


def _find_dictionary(version_dir: Path, glob: str) -> Path | None:
    matches = sorted(version_dir.glob(glob))
    return matches[0] if matches else None


def discover_versions(cfg: DatasetConfig, root: Path) -> list[Version]:
    """Return versions for one dataset, newest first."""
    container = root / "data" / cfg.container
    if not container.is_dir():
        return []
    pat = re.compile(cfg.version_pattern)
    out: list[Version] = []
    for child in sorted(container.iterdir()):
        if not child.is_dir():
            continue
        m = pat.search(child.name)
        if not m:
            continue
        ver = m.group("v")
        out.append(
            Version(
                dataset=cfg.name,
                version=ver,
                ship_folder=child.name,
                version_dir=child,
                dictionary_path=_find_dictionary(child, cfg.dictionary_glob),
            )
        )
    out.sort(key=lambda v: v.sort_key, reverse=True)
    return out
