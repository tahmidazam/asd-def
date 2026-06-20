"""Stability analysis of the Litman data-driven autism classes.

The package's goal is to reproduce the Litman et al. general finite mixture model on the
SPARK phenotype releases held in this monorepo, then re-estimate it within strata of age at
diagnosis and diagnostic era to test whether the four recovered classes survive a change in
who is sampled.

The package is in early development, built one pipeline stage at a time. The command-line
entry point is :mod:`analysis.cli`; :mod:`analysis.paths` locates the cached artefacts each
stage will write.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("analysis")
except PackageNotFoundError:
    # Not installed (e.g. imported straight from the source tree); use a sentinel.
    __version__ = "0.0.0"
