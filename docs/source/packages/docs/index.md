# docs

The `docs` command builds and live-previews this Sphinx site. It is a thin wrapper
over `sphinx-build` and `sphinx-autobuild`, so the documentation has a single entry
point.

## Overview

The `docs` package owns the documentation toolchain as its own dependencies, and sits in the
`docs` dependency group. Install it with `uv sync --group docs`, and run it with
`uv run --group docs`:

```bash
uv run --group docs docs build     # build HTML into docs/build/html
uv run --group docs docs serve     # live-reloading preview at http://127.0.0.1:8000
uv run --group docs docs clean     # remove docs/build
```

`build` takes `-b/--builder` to select another Sphinx builder, such as `linkcheck`,
and `-W/--strict` to turn warnings into errors. `serve` takes `--host` and `--port`,
and rebuilds on edits to `docs/` and `packages/` so autodoc picks up docstring changes.

## Reference

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Python API
:link: reference
:link-type: doc

The `docs_cli.cli` module: the `build`, `serve`, and `clean` commands.
:::

::::

:::{toctree}
:hidden:

reference
:::
