"""dscat: a searchable catalogue over versioned tabular research datasets."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dscat")
except PackageNotFoundError:
    # Not installed (e.g. imported straight from the source tree); use a sentinel.
    __version__ = "0.0.0"
