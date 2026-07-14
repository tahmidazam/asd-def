"""Sphinx configuration for the asd-def documentation.

Build once:  uv run --group docs docs build
Live server: uv run --group docs docs serve

Both commands come from the docs launcher (packages/docs).

Settings reference:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import contextlib
import sys
import warnings
from datetime import date
from importlib.metadata import version as _package_version
from pathlib import Path

from sphinx_polyversion.api import load
from sphinx_polyversion.git import GitRef  # import also registers GitRef for load()

# -- Path setup --------------------------------------------------------------
# Make every package's source importable so autodoc reads the live source in this
# monorepo without a reinstall. New packages under packages/<name>/src are picked
# up automatically.
_PACKAGES = Path(__file__).resolve().parents[2] / "packages"
for _src in sorted(_PACKAGES.glob("*/src")):
    sys.path.insert(0, str(_src))

# A strict build (``docs build --strict`` passes ``-W``) fails on any warning, including the
# SyntaxWarning that stepmix raises when autodoc imports it: one of its docstrings carries an
# invalid ``\_`` escape. That is a third-party bug we cannot fix in their source, so drop it here.
# The warning fires at compile time and reports its origin as ``<unknown>``, so it is matched on
# message rather than module. Invalid escapes in this repo's own code are still caught by ruff
# (W605), so the filter does not hide a warning we would otherwise want to see.
warnings.filterwarnings("ignore", message=r"invalid escape sequence", category=SyntaxWarning)

# -- Project information -----------------------------------------------------

project = "asd-def"
author = "Tahmid Azam"
copyright = f"{date.today():%Y}, {author}"

# -- Versioning --------------------------------------------------------------
# Built per release by sphinx-polyversion (docs/poly.py), which injects the
# revision data via POLYVERSION_DATA. A plain sphinx-build (CI check, local
# `docs build`) has none, so fall back to the installed package version.
SITE_URL = "https://tahmidazam.github.io/asd-def"  # keep in sync with docs/poly.py

html_context: dict = {}
with contextlib.suppress(Exception):
    load(globals())  # fills html_context with "revisions" and "current"

current = html_context.get("current")
if isinstance(current, GitRef):
    release = current.name
    html_baseurl = f"{SITE_URL}/{current.name}/"
else:
    release = "v" + _package_version("dscat")
    html_baseurl = SITE_URL
version = release

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
    "sphinxcontrib.mermaid",  # render mermaid diagrams in the docs
    "sphinxcontrib.bibtex",  # cite from a shared BibTeX file via :footcite:
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
    "BinningPolicy": "analysis.strata.BinningPolicy",
    "LocalisationScheme": "analysis.localise.LocalisationScheme",
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
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Types from dependencies without an intersphinx inventory: linking them is not possible,
# so they render as plain text rather than failing the strict build.
nitpick_ignore = [
    ("py:class", "tqdm.tqdm"),
    ("py:class", "StepMix"),
    ("py:class", "sklearn.discriminant_analysis.LinearDiscriminantAnalysis"),
    # Forward-reference artefacts of ``from __future__ import annotations`` in dscat's
    # signatures: a type alias (``Version``) renders as the internal ``TypeAliasForwardRef``,
    # and a class nested in a stringised return annotation (``FeatureRow``) renders quoted.
    # Neither is resolvable as written, so both render as plain text rather than failing the
    # strict build.
    ("py:class", "TypeAliasForwardRef"),
    ("py:class", "'dscat.model.FeatureRow'"),
    # The ``callable`` builtin used as a NumPy-docstring parameter type: napoleon emits a
    # ``py:class`` cross-reference, but ``callable`` is a builtin function, not a class, so it
    # cannot resolve and renders as plain text.
    ("py:class", "callable"),
]

# -- MyST (Markdown) ---------------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html

myst_enable_extensions = [
    "colon_fence",  # ::: fenced directives
    "deflist",  # definition lists
    "dollarmath",  # $...$ inline and $$...$$ display LaTeX maths
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

# Override the theme's base and heading fonts with Helvetica (see _static/custom.css).
html_css_files = ["custom.css"]

# See the theme's user guide for the full set of options.
html_theme_options = {
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "switcher": {
        "json_url": f"{SITE_URL}/switcher.json",
        "version_match": release,
    },
    # json_url is fetched in the browser, not at build time; probing it during
    # the strict (-W) build would fail offline, so skip the check.
    "check_switcher": False,
}

# -- sphinx-copybutton -------------------------------------------------------

# Strip interactive prompts (">>> ", "... ", "$ ") when copying code blocks.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- sphinxcontrib-bibtex ----------------------------------------------------
# One shared bibliography, seeded from the brief's verified entries. Pages cite with the
# ``:footcite:`` role and carry a per-page ``{footbibliography}`` so each article renders
# its own footnotes; the References page holds the full ``{bibliography}``. Every entry is
# verified with the user before it lands here.
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "label"
bibtex_default_style = "unsrt"

# -- sphinxcontrib-mermaid ---------------------------------------------------

# Diagrams render client-side from the mermaid runtime, pinned to an exact version and
# fetched from the jsDelivr CDN at view time. Pinning keeps the rendered output stable
# across builds; the version here is the one the extension defaults to, set explicitly so
# an extension upgrade cannot change it silently.
mermaid_version = "11.12.1"
