"""sphinx-polyversion configuration: build every released version of the docs.

Driven by the launcher: ``uv run docs versions`` builds all ``v*`` tags, and
``uv run docs versions --local`` builds the working tree alone with mock data.

Each revision is checked out into a temporary directory, provisioned with
``uv sync`` against its own pinned dependencies, and built with a strict
``sphinx-build``. The results are merged under ``docs/build/<version>/``,
alongside a root redirect and the pydata-sphinx-theme ``switcher.json`` rendered
from the revision list.
"""

from __future__ import annotations

import asyncio
import os
from asyncio.subprocess import PIPE
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any, cast

from sphinx_polyversion.api import apply_overrides
from sphinx_polyversion.builder import BuildError
from sphinx_polyversion.driver import DefaultDriver
from sphinx_polyversion.git import Git, GitRef, GitRefType, file_predicate
from sphinx_polyversion.pyvenv import VirtualPythonEnvironment
from sphinx_polyversion.sphinx import SphinxBuilder

#: Public base URL of the site. Keep in sync with SITE_URL in docs/source/conf.py.
SITE_URL = "https://tahmidazam.github.io/asd-def"

TAG_REGEX = r"v\d+\.\d+\.\d+"  # release tags only
BRANCH_REGEX = r"(?!)"  # matches nothing: no dev build from a branch
OUTPUT_DIR = "docs/build"
SOURCE_DIR = "docs/source"
SPHINX_ARGS = ["-W", "--keep-going"]  # warnings (incl. nitpicky) fail the release
UV_ARGS = ["--locked", "--no-dev", "--group", "docs"]
MOCK = False


class Uv(VirtualPythonEnvironment):
    """Provision a revision with ``uv sync``, then build inside its ``.venv``.

    sphinx-polyversion ships no working uv environment (its ``uv`` module is an
    unfinished stub), so this mirrors the built-in ``Poetry`` environment.
    """

    def __init__(
        self,
        path: Path,
        name: str,
        *,
        args: Iterable[str],
        env: dict[str, str] | None = None,
    ) -> None:
        """Record the ``uv sync`` arguments for the revision checked out at ``path``."""
        super().__init__(path, name, path / ".venv", env=env)
        self.args = list(args)

    async def __aenter__(self) -> Uv:
        """Create and populate the revision's virtual environment with uv."""
        cmd = ["uv", "sync", *self.args]
        env = self.apply_overrides(os.environ.copy())
        env.pop("VIRTUAL_ENV", None)  # force uv to use the checkout's own .venv
        process = await asyncio.create_subprocess_exec(
            *cmd, cwd=self.path, env=env, stdout=PIPE, stderr=PIPE
        )
        out, err = await process.communicate()
        if process.returncode:
            raise BuildError from CalledProcessError(
                cast(int, process.returncode),
                " ".join(cmd),
                out.decode(errors="ignore"),
                err.decode(errors="ignore"),
            )
        self.venv = (self.path / ".venv").resolve()
        return self


def root_data(driver: DefaultDriver) -> dict[str, Any]:
    """Build the jinja context for the root templates (switcher.json, index.html)."""
    by_name: dict[str, GitRef] = {}
    for ref in sorted(driver.builds, key=lambda r: r.date, reverse=True):
        by_name.setdefault(ref.name, ref)
    revisions = list(by_name.values())
    return {
        "revisions": revisions,
        "latest": revisions[0] if revisions else None,
        "SITE_URL": SITE_URL,
    }


apply_overrides(globals())  # --local sets MOCK=True
root = Git.root(Path(__file__).parent)
_mock_ref = GitRef("local", "", "", GitRefType.BRANCH, datetime.now())

DefaultDriver(
    root,
    OUTPUT_DIR,
    vcs=Git(
        branch_regex=BRANCH_REGEX,
        tag_regex=TAG_REGEX,
        predicate=file_predicate([Path(SOURCE_DIR)]),  # skip tags without docs/source
    ),
    builder=SphinxBuilder(Path(SOURCE_DIR), args=SPHINX_ARGS),
    env=Uv.factory(args=UV_ARGS),
    root_data_factory=root_data,
    template_dir=root / "docs" / "_polyversion" / "templates",
    mock={"current": _mock_ref, "revisions": [_mock_ref]},
).run(mock=MOCK, sequential=True)
