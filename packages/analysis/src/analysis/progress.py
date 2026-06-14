"""A single progress bar per command, with live state in its postfix.

Each pipeline command opens one bar whose total is the sum of all units of work across
every loop it runs (for example the sum of initialisations over a grid of component
counts, or the features in an enrichment pass). The bar is updated once per unit and its
postfix carries the live state: which stage or stratum is running, the best log-likelihood
so far, the smallest class proportion, and so on.

The bar is written to the real standard error so it survives the standard-output
redirection that :mod:`analysis.run` uses to capture a run's log, and so its control
characters do not pollute ``run.log``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from tqdm import tqdm


@contextmanager
def task_bar(total: int, desc: str, **kwargs: Any) -> Iterator[tqdm]:
    """Open the command's single progress bar as a context manager.

    Parameters
    ----------
    total : int
        Total units of work across every loop the command runs.
    desc : str
        Short label shown to the left of the bar.
    **kwargs
        Forwarded to :class:`tqdm.tqdm`.

    Yields
    ------
    tqdm.tqdm
        The bar. Call ``update`` once per unit of work and ``set_postfix`` to publish the
        live state.
    """
    bar = tqdm(
        total=total,
        desc=desc,
        file=sys.__stderr__,
        dynamic_ncols=True,
        leave=True,
        **kwargs,
    )
    try:
        yield bar
    finally:
        bar.close()
