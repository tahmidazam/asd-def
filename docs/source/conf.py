"""Sphinx configuration for the asd-def documentation.

Build once:  uv run --group docs docs build
Live server: uv run --group docs docs serve

Both commands come from the docs launcher (packages/docs).

Settings reference:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sys
from datetime import date
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Make every package's source importable so autodoc reads the live source in this
# monorepo without a reinstall. New packages under packages/<name>/src are picked
# up automatically.
_PACKAGES = Path(__file__).resolve().parents[2] / "packages"
for _src in sorted(_PACKAGES.glob("*/src")):
    sys.path.insert(0, str(_src))

# -- Project information -----------------------------------------------------

project = "asd-def"
author = "Tahmid Azam"
copyright = f"{date.today():%Y}, {author}"

# -- General configuration ---------------------------------------------------

extensions = [
    # Built in, ship with Sphinx:
    "sphinx.ext.autodoc",  # pull API docs from docstrings
    "sphinx.ext.autosummary",  # generate per-object summary tables and stubs
    "sphinx.ext.napoleon",  # parse Google- and NumPy-style docstrings
    "sphinx.ext.intersphinx",  # link to other projects' documentation
    "sphinx.ext.viewcode",  # add "[source]" links next to documented objects
    # Third party, from the `docs` dependency group:
    "myst_parser",  # author pages in Markdown (MyST)
    "sphinx_design",  # grids, cards, tabs; pairs with the pydata theme
    "sphinx_copybutton",  # copy button on code blocks
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "en"

# Author pages in either reStructuredText or Markdown.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Autodoc and autosummary -------------------------------------------------

autosummary_generate = True  # build stub pages for autosummary entries
autodoc_member_order = "bysource"
autodoc_typehints = "description"  # render type hints in the parameter list
autodoc_typehints_description_target = "documented"  # don't repeat documented types
autodoc_default_options = {
    "members": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Map the short type names used in the NumPy docstrings to their canonical targets, so
# the rendered API cross-references resolve: Path to the standard library, the
# SQLAlchemy types via their intersphinx inventory.
napoleon_preprocess_types = True  # apply napoleon_type_aliases to the docstring types
# The NumPy docstrings name types by their short name; map each to its canonical target
# so the rendered cross-references link out: internal classes to their reference page,
# external types to the standard library or the SQLAlchemy inventory.
napoleon_type_aliases = {
    "Path": "pathlib.Path",
    "Engine": "sqlalchemy.engine.Engine",
    "RowMapping": "sqlalchemy.engine.RowMapping",
    "sequence": "collections.abc.Sequence",
    "mapping": "collections.abc.Mapping",
    "Catalogue": "dscat.index.Catalogue",
    "FeatureRow": "dscat.model.FeatureRow",
    "TableRow": "dscat.model.TableRow",
    "DatasetConfig": "dscat.config.DatasetConfig",
    "Version": "dscat.config.Version",
    "DiffResult": "dscat.diff.DiffResult",
    "IngestSummary": "dscat.ingest.IngestSummary",
    "Format": "dscat.output.Format",
}
python_use_unqualified_type_names = True  # display the short type names, still linked

# Treat an unresolved cross-reference as an error under -W, so a broken link to a
# package or one of its objects fails the docs build instead of silently rendering as
# plain text.
nitpicky = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/20/", None),
}

# -- MyST (Markdown) ---------------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html

myst_enable_extensions = [
    "colon_fence",  # ::: fenced directives
    "deflist",  # definition lists
    "fieldlist",  # field lists
    "smartquotes",  # typographic quotes
    "substitution",  # |variable| substitutions
    "tasklist",  # GitHub-style task lists
]
myst_heading_anchors = 3  # auto-generate anchors for h1-h3 headings

# -- HTML output (pydata-sphinx-theme) ---------------------------------------
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = project

# Theme options are kept minimal. See the theme's user guide for the full set;
# add navbar entries, icon links (GitHub, etc.), and a version switcher here as
# the site grows.
html_theme_options = {
    "show_toc_level": 2,
    "navigation_with_keys": False,
}

# -- sphinx-copybutton -------------------------------------------------------

# Strip interactive prompts (">>> ", "... ", "$ ") when copying code blocks.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
