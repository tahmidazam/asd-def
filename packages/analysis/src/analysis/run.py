"""The run lifecycle: a content-addressed directory, a captured log, and a manifest.

``run_context`` is the single entry point every expensive stage uses. It hashes the run's
parameters, opens (or reuses) the directory ``artefacts/<stage>/<hash>``, captures the
run's standard output into ``run.log`` while still showing it on the console, and writes a
``manifest.json`` recording the inputs, status, timing, resolved package versions, the
repository commit, and the caller's metrics.

A run whose manifest already exists and finished cleanly is a cache hit: the caller is told
so through ``RunContext.cache_hit`` and loads the cached artefacts instead of recomputing
(plan section 11). Pass ``force=True`` to recompute regardless.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
import time
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analysis import cache
from analysis.paths import find_repo_root
from analysis.paths import run_dir as _run_dir

LOG_NAME = "run.log"


class _Tee(io.TextIOBase):
    """A text stream that forwards every write to several underlying streams.

    Used to send a run's standard output to both the console and ``run.log`` at once.
    """

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, text: str) -> int:  # type: ignore[override]
        """Write ``text`` to every underlying stream and return its length."""
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        """Flush every underlying stream, ignoring streams already closed.

        The tee does not own its streams: a stream may be closed by the run before the
        tee is garbage-collected, so flushing it then is a no-op rather than an error.
        """
        for stream in self._streams:
            with contextlib.suppress(ValueError):
                stream.flush()

    def detach_streams(self) -> None:
        """Stop forwarding to the underlying streams, so the tee can be discarded safely."""
        self._streams = ()


@dataclass
class RunContext:
    """Handle to one run: where its artefacts live and what to record about it.

    Attributes
    ----------
    stage : str
        The pipeline stage, used as the artefact subdirectory.
    run_hash : str
        The full hash over the run's parameters.
    run_dir : Path
        The directory holding this run's artefacts and manifest.
    cache_hit : bool
        ``True`` when a clean manifest already existed, so the caller should load cached
        artefacts rather than recompute.
    params : dict
        The parameters that determined the hash.
    metrics : dict
        Scalar results the caller wants recorded in the manifest (final log-likelihood,
        class proportions, and so on).
    log : logging.Logger
        Logger writing to both the console and ``run.log``.
    """

    stage: str
    run_hash: str
    run_dir: Path
    cache_hit: bool
    params: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("analysis"))

    def path(self, name: str) -> Path:
        """Return the path to a named artefact inside this run's directory."""
        return self.run_dir / name


@contextmanager
def run_context(
    stage: str,
    params: Mapping[str, Any],
    *,
    root: Path | None = None,
    force: bool = False,
) -> Iterator[RunContext]:
    """Open a run: resolve its directory, capture its log, and manage its manifest.

    Parameters
    ----------
    stage : str
        The pipeline stage (the artefact subdirectory and logger name).
    params : Mapping
        The inputs that determine the output. Their hash names the run directory and is
        recorded in the manifest, so editing any input invalidates the cache.
    root : Path, optional
        Repository root. Defaults to the discovered root.
    force : bool, default False
        Recompute even when a clean manifest already exists.

    Yields
    ------
    RunContext
        The run handle. When ``cache_hit`` is ``True`` the body should load cached
        artefacts; otherwise it computes, writes artefacts under ``run_dir``, and may set
        ``metrics`` for the manifest.
    """
    root = root or find_repo_root()
    full_hash = cache.compute_hash(dict(params))
    rdir = _run_dir(root, stage, cache.short_hash(full_hash))

    existing = cache.read_manifest(rdir)
    hit = (not force) and existing is not None and existing.get("status") == "ok"

    logger = logging.getLogger(f"analysis.{stage}")
    ctx = RunContext(stage, full_hash, rdir, hit, dict(params), log=logger)

    if hit:
        # The cached run stands. Hand back the context without touching the log or
        # manifest, so the caller can load the existing artefacts.
        yield ctx
        return

    rdir.mkdir(parents=True, exist_ok=True)
    logfile = (rdir / LOG_NAME).open("a", encoding="utf-8")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    to_console = logging.StreamHandler(sys.stdout)
    to_file = logging.StreamHandler(logfile)
    for handler in (to_console, to_file):
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    started = datetime.now(UTC)
    clock = time.monotonic()
    manifest: dict[str, Any] = {
        "stage": stage,
        "hash": full_hash,
        "status": "running",
        "params": dict(params),
        "started_at": started.isoformat(timespec="seconds"),
        "git_commit": cache.git_commit(root),
        "environment": cache.environment_versions(),
    }
    cache.write_manifest(rdir, manifest)

    saved_stdout = sys.stdout
    tee = _Tee(saved_stdout, logfile)
    sys.stdout = tee
    try:
        yield ctx
    except BaseException as exc:  # noqa: BLE001 - recorded then re-raised
        manifest["status"] = "failed"
        manifest["error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        raise
    else:
        manifest["status"] = "ok"
    finally:
        sys.stdout = saved_stdout
        tee.detach_streams()
        manifest["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        manifest["duration_s"] = round(time.monotonic() - clock, 3)
        manifest["metrics"] = ctx.metrics
        manifest["outputs"] = sorted(
            p.name for p in rdir.iterdir() if p.name not in {cache.MANIFEST_NAME, LOG_NAME}
        )
        cache.write_manifest(rdir, manifest)
        for handler in (to_console, to_file):
            logger.removeHandler(handler)
            handler.close()
        logfile.close()
