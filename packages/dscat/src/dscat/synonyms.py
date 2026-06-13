"""Domain synonym groups for query expansion (bidirectional within a group).

The file is a JSON array of groups, each group a JSON array of equivalent terms;
every member expands to every other during search. Built-ins ship in
``dscat/synonyms.json``; a project may append ``<root>/synonyms.json``.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path


def _pairs(groups: list[list[str]]) -> list[tuple[str, str]]:
    """Expand synonym groups into directed ``(term, expansion)`` pairs."""
    pairs: list[tuple[str, str]] = []
    for raw_group in groups:
        group = [g for g in (str(member).strip().lower() for member in raw_group) if g]
        pairs.extend((x, y) for x in group for y in group if x != y)
    return pairs


def load_synonyms(root: Path) -> list[tuple[str, str]]:
    """Load synonym pairs from the built-in groups and any project file.

    Parameters
    ----------
    root : Path
        Repository root; a ``synonyms.json`` there is appended to the built-ins.

    Returns
    -------
    list of tuple
        Directed ``(term, expansion)`` pairs, where each group member expands to
        every other member.
    """
    text = resources.files("dscat").joinpath("synonyms.json").read_text(encoding="utf-8")
    pairs = _pairs(json.loads(text))
    proj = root / "synonyms.json"
    if proj.is_file():
        pairs.extend(_pairs(json.loads(proj.read_text(encoding="utf-8"))))
    return pairs
