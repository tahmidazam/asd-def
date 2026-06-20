"""Append-only checkpoint logs that let a long iterative stage resume after an interrupt.

The content-addressed cache (:mod:`analysis.cache`, :mod:`analysis.run`) works at the
granularity of a whole stage: a run is reused only once it has finished cleanly, and an
interrupted run leaves no reusable output, so re-running recomputes the stage from the
start. That is fine for the short stages, but the multi-seed loops (model selection, the
multi-initialisation and subsampling stability runs, and the minimum-stratum-size sweep) can
take tens of minutes to hours, and losing all of it to one interrupt is wasteful.

A :class:`CheckpointLog` records each unit of work as it completes, one JSON line per unit,
appended and flushed to disk. When the stage runs again over the same parameters (hence the
same run directory), it reads the completed units back and continues from the first one that
is missing. The seeds are derived deterministically from the unit index, so a resumed run
reproduces exactly what an uninterrupted run would have computed; the checkpoint changes only
how much is recomputed, never the result.

The unit of resumption is one line. Each line holds the whole payload for one unit (for the
selection grid, every criterion row for one seeded iteration; for stability, one fit and its
comparison). A process killed mid-write can leave a torn final line; :meth:`CheckpointLog.load`
parses up to the first line that does not decode and stops there, so the last, incomplete
unit is dropped and recomputed rather than read back half-written. The logs use Python's
``json`` non-finite extension (``NaN`` is written and read back unchanged), since they are
read only by this module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Checkpoint files share this suffix so a stage can clear every checkpoint in its run
# directory without naming each one (a stage may keep more than one, e.g. the fits and the
# comparisons of a multi-initialisation run).
SUFFIX = ".checkpoint.jsonl"


class CheckpointLog:
    """An append-only log of completed units of work for one resumable loop.

    Each call to :meth:`append` writes one unit's payload as a JSON line and flushes it to
    disk; :meth:`load` reads the completed units back in order. The payload is any
    JSON-serialisable value: a caller that produces several records per unit stores the list
    of them, so one line maps to one resumable unit.

    Parameters
    ----------
    path : Path
        The log file. It lives inside the stage's content-addressed run directory, so it is
        specific to the run's parameters and is never shared between different parameter sets.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[Any]:
        """Return the completed unit payloads in the order they were written.

        Parsing stops at the first line that does not decode as JSON, which drops a torn
        final line left by a process killed mid-write. Because units are appended and
        flushed one at a time, only the last line can be incomplete.

        Returns
        -------
        list
            One entry per completed unit, in append order. Empty when the log does not exist.
        """
        if not self.path.is_file():
            return []
        payloads: list[Any] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payloads.append(json.loads(line))
            except json.JSONDecodeError:
                break
        return payloads

    def append(self, payload: Any) -> None:
        """Append one unit's payload as a JSON line and flush it to disk.

        The write is flushed and ``fsync``-ed so a completed unit survives an interrupt that
        kills the process before the stage finishes.

        Parameters
        ----------
        payload : object
            Any JSON-serialisable value describing one completed unit.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def clear(self) -> None:
        """Delete the log file if it exists."""
        self.path.unlink(missing_ok=True)


def clear_checkpoints(directory: Path) -> None:
    """Remove every checkpoint log in a run directory.

    Called when a stage is forced to recompute (so a stale partial run is not resumed) and
    once it has finished cleanly (so the now-redundant checkpoints do not linger beside the
    final artefacts).

    Parameters
    ----------
    directory : Path
        A stage's run directory.
    """
    if not directory.is_dir():
        return
    for path in directory.glob(f"*{SUFFIX}"):
        path.unlink(missing_ok=True)
