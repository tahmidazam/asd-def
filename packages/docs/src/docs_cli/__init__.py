"""The `docs` command for building and previewing the asd-def docs."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("docs")
except PackageNotFoundError:
    # Not installed (e.g. imported straight from the source tree); use a sentinel.
    __version__ = "0.0.0"
