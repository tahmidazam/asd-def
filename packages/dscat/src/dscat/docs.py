"""Discover and convert non-dictionary documentation (PDF/Word/RTF/txt) to markdown.

Conversion shells out to tools already on the system: ``pandoc`` for docx and rtf
(to GFM), ``pdftotext -layout`` for pdf (to text), and ``textutil`` as a macOS
fallback for legacy ``.doc``. Results are cached under
``.catalogue/docs/<dataset>/<version>/`` so a converted file can be read or
section-extracted without converting it again.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from dscat.paths import docs_cache_dir

DOC_EXTS = {".pdf", ".docx", ".doc", ".rtf", ".txt"}


def discover_docs(version_dir: Path, root: Path) -> list[tuple[str, str, str]]:
    """Return (relative_path, kind, title) for documentation files under a version."""
    out: list[tuple[str, str, str]] = []
    for p in sorted(version_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in DOC_EXTS:
            rel = p.resolve().relative_to(root.resolve()).as_posix()
            out.append((rel, p.suffix.lower().lstrip("."), p.stem))
    return out


def cache_path(root: Path, dataset: str, version: str, src: Path) -> Path:
    """Return the cache path for a converted document under ``.catalogue/docs``."""
    return docs_cache_dir(root) / dataset / version / (src.stem + ".md")


def _run(cmd: list[str]) -> bool:
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_doc(src: Path, dest: Path) -> Path:
    """Convert ``src`` to markdown/text at ``dest`` (cached by mtime). Returns ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
        return dest
    ext = src.suffix.lower()
    if ext in (".docx", ".rtf"):
        fmt = "docx" if ext == ".docx" else "rtf"
        if _run(["pandoc", "-f", fmt, "-t", "gfm", "-o", str(dest), str(src)]):
            return dest
    if ext in (".docx", ".doc", ".rtf"):  # fallback (and the only path for legacy .doc)
        if _run(["textutil", "-convert", "txt", "-output", str(dest), str(src)]):
            return dest
        raise RuntimeError(f"could not convert {src.name} (pandoc/textutil unavailable?)")
    if ext == ".pdf":
        if _run(["pdftotext", "-layout", str(src), str(dest)]):
            return dest
        raise RuntimeError(f"could not convert {src.name} (pdftotext unavailable?)")
    if ext == ".txt":
        shutil.copyfile(src, dest)
        return dest
    raise RuntimeError(f"unsupported document type: {ext}")


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
