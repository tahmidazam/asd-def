"""Discover and convert non-dictionary documentation (PDF/Word/RTF/txt) to markdown.

Conversion runs through a selectable engine, each shelling out to its own command-line
tool. By default the engine follows the file type: ``marker`` (a layout-aware PDF
converter) for PDFs, ``markitdown`` (Microsoft's faster, broader-coverage converter)
for other formats, and macOS ``textutil`` for legacy ``.doc`` and ``.rtf``, which
neither reads. Passing an engine forces it for that file. markitdown and marker ship as
dependencies of dscat; ``textutil`` is part of macOS.

Results are cached under ``.catalogue/docs/<dataset>/<version>/``, keyed by engine, so
a converted file can be read or section-extracted without converting it again.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

from dscat.paths import docs_cache_dir

DOC_EXTS = {".pdf", ".docx", ".doc", ".rtf", ".txt"}


class Engine(StrEnum):
    """A document-to-markdown conversion engine.

    Attributes
    ----------
    markitdown
        Microsoft's ``markitdown`` CLI; fast, with broad document-format coverage. The
        default for every format except PDF.
    marker
        The ``marker`` PDF converter (the ``marker_single`` CLI); higher-quality,
        layout-aware extraction, and the default for PDFs.
    textutil
        The macOS ``textutil`` tool, used automatically for legacy ``.doc`` and ``.rtf``
        files that markitdown and marker cannot read.
    """

    markitdown = "markitdown"
    marker = "marker"
    textutil = "textutil"


TEXTUTIL_EXTS = {".doc", ".rtf"}
MARKER_EXTS = {".pdf"}  # default to marker; every other (non-legacy) format uses markitdown


def resolve_engine(src: Path, engine: Engine | None = None) -> Engine:
    """Return the engine used for ``src``.

    With no explicit ``engine``, the choice follows the file type: ``marker`` for PDFs
    and ``markitdown`` for every other format. Legacy ``.doc`` and ``.rtf`` always use
    ``textutil``, which markitdown and marker cannot read, even when an ``engine`` is
    given; for any other format an explicit ``engine`` overrides the default.
    """
    ext = src.suffix.lower()
    if ext in TEXTUTIL_EXTS:
        return Engine.textutil
    if engine is not None:
        return engine
    return Engine.marker if ext in MARKER_EXTS else Engine.markitdown


def discover_docs(version_dir: Path, root: Path) -> list[tuple[str, str, str]]:
    """Return (relative_path, kind, title) for documentation files under a version."""
    out: list[tuple[str, str, str]] = []
    for p in sorted(version_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in DOC_EXTS:
            rel = p.resolve().relative_to(root.resolve()).as_posix()
            out.append((rel, p.suffix.lower().lstrip("."), p.stem))
    return out


def cache_path(
    root: Path, dataset: str, version: str, src: Path, engine: Engine | None = None
) -> Path:
    """Return the cache path for a document converted by ``engine``.

    The effective engine (see :func:`resolve_engine`) is part of the filename
    (``<stem>.<engine>.md``) so caches from different engines never collide.
    """
    name = f"{src.stem}.{resolve_engine(src, engine)}.md"
    return docs_cache_dir(root) / dataset / version / name


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run ``cmd`` with captured output, or return ``None`` if its tool is not on PATH."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return None


def _tail(text: str) -> str:
    """Return the last non-empty line of ``text``, for a compact error detail."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _convert_markitdown(src: Path, dest: Path) -> None:
    """Convert with Microsoft's ``markitdown`` CLI, which writes ``dest`` itself."""
    proc = _run(["markitdown", str(src), "-o", str(dest)])
    if proc is None:
        raise RuntimeError("markitdown is not installed; run `uv sync`")
    if proc.returncode != 0:
        raise RuntimeError(f"markitdown could not convert {src.name}: {_tail(proc.stderr)}")


def _convert_marker(src: Path, dest: Path) -> None:
    """Convert with the ``marker_single`` CLI, then copy its markdown to ``dest``.

    ``marker`` writes a folder per document (markdown plus any extracted images); only
    the markdown is cached, preferring the file whose stem matches the source.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["marker_single", str(src), "--output_dir", tmp, "--output_format", "markdown"]
        proc = _run(cmd)
        if proc is None:
            raise RuntimeError("marker is not installed; run `uv sync`")
        produced = sorted(Path(tmp).rglob("*.md"))
        if not produced:
            raise RuntimeError(f"marker could not convert {src.name}: {_tail(proc.stderr)}")
        chosen = next((p for p in produced if p.stem == src.stem), produced[0])
        shutil.copyfile(chosen, dest)


def _convert_textutil(src: Path, dest: Path) -> None:
    """Convert legacy ``.doc``/``.rtf`` with macOS ``textutil`` (to plain text)."""
    proc = _run(["textutil", "-convert", "txt", "-output", str(dest), str(src)])
    if proc is None:
        raise RuntimeError("textutil is not available (macOS only)")
    if proc.returncode != 0:
        raise RuntimeError(f"textutil could not convert {src.name}: {_tail(proc.stderr)}")


_CONVERTERS: dict[Engine, Callable[[Path, Path], None]] = {
    Engine.markitdown: _convert_markitdown,
    Engine.marker: _convert_marker,
    Engine.textutil: _convert_textutil,
}


def convert_doc(src: Path, dest: Path, engine: Engine | None = None) -> Path:
    """Convert ``src`` to markdown at ``dest`` with ``engine``, caching by mtime.

    Parameters
    ----------
    src : Path
        The source document.
    dest : Path
        Where to write the markdown. A ``dest`` newer than ``src`` is reused as is.
    engine : Engine or None, default None
        Force a conversion engine. When ``None`` the engine is chosen by file type:
        ``marker`` for PDFs, ``markitdown`` otherwise, and ``textutil`` for legacy
        ``.doc``/``.rtf`` (see :func:`resolve_engine`).

    Returns
    -------
    Path
        ``dest``.

    Raises
    ------
    RuntimeError
        When the engine cannot convert the document, for example because its tool is
        not installed or it does not support the document's format.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
        return dest
    _CONVERTERS[resolve_engine(src, engine)](src, dest)
    return dest


def extract_sections(md_path: Path, pattern: str, context: int = 3) -> list[str]:
    """Return windows of lines around each case-insensitive match of ``pattern``.

    Each match contributes the lines within ``context`` lines on either side;
    overlapping windows are merged.

    Parameters
    ----------
    md_path : Path
        Markdown file to scan.
    pattern : str
        Regular expression matched against each line, case-insensitively.
    context : int, default 3
        Number of lines to include on each side of a match.

    Returns
    -------
    list of str
        One newline-joined block of text per merged window.
    """
    rx = re.compile(pattern, re.IGNORECASE)
    lines = md_path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = [i for i, ln in enumerate(lines) if rx.search(ln)]
    blocks: list[tuple[int, int]] = []
    for i in hits:
        lo, hi = max(0, i - context), min(len(lines), i + context + 1)
        if blocks and lo <= blocks[-1][1]:
            blocks[-1] = (blocks[-1][0], max(blocks[-1][1], hi))
        else:
            blocks.append((lo, hi))
    return ["\n".join(lines[lo:hi]) for lo, hi in blocks]
