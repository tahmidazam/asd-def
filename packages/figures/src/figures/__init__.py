"""Matplotlib figures for the Litman stability analysis.

The package turns the cached artefacts written by :mod:`analysis` into figures. Each figure
is built by a function that takes a dataframe and returns a Matplotlib figure, so the figures
are testable without a display; the command-line entry point :mod:`figures.cli` resolves a
cached run, builds the figure, and writes it under ``artefacts/figures/``.

There is one figure per phase-1 and phase-2 result: the reproduction of the named classes
(:mod:`figures.reproduction`), the model-selection criteria (:mod:`figures.selection`), the
stability of the reference fit (:mod:`figures.stability`), the minimum viable stratum size
(:mod:`figures.nmin`), and the cross-cohort replication (:mod:`figures.replication`). The
``publish`` step (:mod:`figures.publish`) copies the rendered PNGs into the committed
documentation tree.
"""

from __future__ import annotations

import matplotlib

# The package writes figures to files in headless pipelines and tests, so the non-interactive
# Agg backend is selected here, before any submodule imports pyplot.
matplotlib.use("Agg")

__version__ = "0.3.0"
