"""docs command-line interface (Typer).

Builds and previews the asd-def Sphinx site, a thin wrapper over sphinx-build and
sphinx-autobuild, so the documentation has a single entry point.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="docs",
    help="Build, version, and live-preview the asd-def Sphinx documentation.",
    no_args_is_help=True,
    add_completion=False,
)


# ---- helpers -----------------------------------------------------------------
def _docs_root() -> Path:
    """Walk up from the CWD to the nearest ancestor holding docs/source/conf.py."""
    for d in (Path.cwd(), *Path.cwd().parents):
        if (d / "docs" / "source" / "conf.py").is_file():
            return d
    typer.echo("no docs/source/conf.py at or above the current directory", err=True)
    raise typer.Exit(1)


def _tool(name: str) -> str:
    """Resolve a console script (sphinx-build, sphinx-autobuild) from the environment."""
    found = shutil.which(name) or str(Path(sys.executable).parent / name)
    if not Path(found).exists():
        typer.echo(f"{name!r} not found; run `uv sync --group docs`", err=True)
        raise typer.Exit(1)
    return found


def _run(cmd: list[str]) -> None:
    """Run a docs subprocess, exiting non-zero if it fails."""
    try:
        code = subprocess.call(cmd)
    except KeyboardInterrupt:
        code = 130
    if code:
        raise typer.Exit(code)


@app.command()
def build(
    builder: str = typer.Option("html", "--builder", "-b", help="Sphinx builder."),
    strict: bool = typer.Option(False, "--strict", "-W", help="Turn warnings into errors."),
) -> None:
    """Build the docs once into docs/build/<builder> (HTML by default)."""
    root = _docs_root()
    cmd = [_tool("sphinx-build"), "-b", builder]
    if strict:
        cmd += ["-W", "--keep-going"]
    cmd += [str(root / "docs" / "source"), str(root / "docs" / "build" / builder)]
    _run(cmd)


@app.command()
def versions(
    local: bool = typer.Option(
        False, "--local", "-l", help="Build only the working tree with mock version data."
    ),
) -> None:
    """Build every released version into docs/build, driven by docs/poly.py."""
    root = _docs_root()
    cmd = [_tool("sphinx-polyversion"), str(root / "docs" / "poly.py")]
    if local:
        cmd.append("--local")
    _run(cmd)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
) -> None:
    """Serve the docs with live reload, rebuilding on edits to docs/ and packages/."""
    root = _docs_root()
    cmd = [
        _tool("sphinx-autobuild"),
        str(root / "docs" / "source"),
        str(root / "docs" / "build" / "html"),
        "--watch",
        str(root / "packages"),
        "--host",
        host,
        "--port",
        str(port),
        "--open-browser",
    ]
    _run(cmd)


@app.command()
def clean() -> None:
    """Remove the docs/build directory."""
    shutil.rmtree(_docs_root() / "docs" / "build", ignore_errors=True)


if __name__ == "__main__":
    app()
