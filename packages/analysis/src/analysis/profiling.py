r"""Hardware capture and per-unit resource measurement for the fitting stages.

The stratified fits and the permutation null are the heaviest compute in the pipeline, and
how feasible they are depends on the machine that runs them: a laptop overnight, or a
Harvard O2 (SLURM) job array. This module measures that cost rather than guessing it. It
captures the hardware once per run and the resource use of each fitted unit, so a short
calibration run yields the per-fit cost that the full run, and any later SLURM resource
request, are projected from.

Two pieces, neither of which knows anything about the model, so the same instrumentation
serves the stratified fits, the drift null, and any later stage:

- :func:`capture_hardware` records the CPU, core counts, memory, BLAS thread pools, and
  platform. It is written into the run manifest, so every artefact carries the machine that
  produced it and a laptop run and an O2 run are directly comparable.
- :func:`measure` is a context manager around one unit of work (one ``fit_gfmm`` call). It
  times the unit, reads its process CPU time, samples the resident set size on a background
  thread to catch the peak (including the native BLAS allocations that :mod:`tracemalloc`
  cannot see), and records the bytes the unit writes, returning a :class:`UnitMetrics`.

The four measured quantities map onto the four numbers a SLURM job needs: wall time sets
``--time``, peak resident memory sets ``--mem``, CPU utilisation (CPU time over wall time)
shows whether a fit uses more than one core and so sets ``--cpus-per-task``, and the output
bytes set the scratch storage the run consumes.

Scope. The CPU-time and memory readings are for the calling process and its threads, which
is what a StepMix fit is: BLAS-threaded work in one process, no child processes. A stage
that forks worker processes would need the children measured too; that is noted where it
would matter rather than handled here, since the fitting stages do not fork.
"""

from __future__ import annotations

import os
import platform
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import psutil

# Resident memory is sampled on a timer while a unit runs. The interval trades the accuracy
# of the captured peak against the sampler's own overhead; at 50 ms it adds negligible load
# to a fit that takes seconds while still catching a short-lived allocation spike.
_DEFAULT_SAMPLE_INTERVAL_S = 0.05

# The threading environment variables that govern how many threads the numerical libraries
# spawn. Recorded with the hardware so a fit's core usage is interpretable.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextmanager
def single_threaded_blas() -> Iterator[None]:
    """Force every BLAS and OpenMP thread pool to one thread for the enclosed block.

    A lone fit is already single-core (:func:`measure` shows ``cpu_utilisation`` about one),
    but scikit-learn bundles its own OpenMP runtime (``libomp``) sized to every logical core
    by default. Running several fits at once through a
    :class:`~concurrent.futures.ProcessPoolExecutor` gives each worker its own copy of that
    pool, so N concurrent workers compete for N times the machine's threads the moment any of
    them calls a parallelised scikit-learn routine. A spawned worker inherits the parent's
    environment, and each numerical library reads its thread count once, the first time it is
    used, so setting these variables before the pool starts keeps every worker to the one
    core it was given. Not needed around a solitary fit, so it is applied only around a
    concurrent pool, not a whole CLI invocation.

    Yields
    ------
    None
    """
    previous = {var: os.environ.get(var) for var in _THREAD_ENV_VARS}
    try:
        for var in _THREAD_ENV_VARS:
            os.environ[var] = "1"
        yield
    finally:
        for var, value in previous.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


def _blas_info() -> list[dict[str, object]]:
    """Return the loaded BLAS or OpenMP thread pools, or an empty list if unavailable.

    Uses :mod:`threadpoolctl`, which inspects the numerical shared libraries already loaded
    into the process, so :mod:`numpy` is imported first to ensure its BLAS backend is
    present. Each entry names the backend (OpenBLAS, MKL, or BLIS) and the number of threads
    it is configured to use, which is what determines how many cores a fit can occupy.
    """
    try:
        import numpy  # noqa: F401  (imported for its BLAS backend, not used directly)
        from threadpoolctl import threadpool_info

        return [
            {
                "backend": pool.get("internal_api"),
                "num_threads": pool.get("num_threads"),
                "version": pool.get("version"),
            }
            for pool in threadpool_info()
        ]
    except Exception:  # noqa: BLE001 - diagnostic capture must never break a run
        return []


_cpu_model_cache: tuple[str | None] | None = None


def _cpu_model() -> str | None:
    """Return the CPU brand string, looked up once and cached for the process.

    :mod:`py-cpuinfo` can take up to a second on its first call (it probes the processor), so
    the result is memoised: the CPU does not change while a run executes, and capturing the
    hardware on every run must stay cheap. Falls back to the platform string if the lookup
    fails.
    """
    global _cpu_model_cache
    if _cpu_model_cache is None:
        try:
            from cpuinfo import get_cpu_info

            _cpu_model_cache = (get_cpu_info().get("brand_raw"),)
        except Exception:  # noqa: BLE001 - fall back to the platform string
            _cpu_model_cache = (platform.processor() or None,)
    return _cpu_model_cache[0]


def capture_hardware() -> dict[str, object]:
    """Capture the hardware and threading configuration of the current machine.

    The result is JSON-serialisable and recorded once per run, so every artefact carries the
    machine that produced it. Each probe is guarded: a field that cannot be read is set to
    ``None`` rather than raising, since the capture is diagnostic and partial information is
    still useful.

    Returns
    -------
    dict
        Keys: ``cpu_model``, ``architecture``, ``physical_cores``, ``logical_cores``,
        ``cpu_freq_mhz``, ``total_memory_bytes``, ``available_memory_bytes``, ``hostname``,
        ``platform``, ``system``, ``python_version``, ``blas`` (the loaded BLAS thread
        pools), and ``thread_env`` (the threading environment variables).
    """
    try:
        freq = psutil.cpu_freq()
        cpu_freq_mhz = round(freq.max, 1) if freq and freq.max else None
    except Exception:  # noqa: BLE001 - cpu_freq is unavailable on some platforms
        cpu_freq_mhz = None

    memory = psutil.virtual_memory()
    return {
        "cpu_model": _cpu_model(),
        "architecture": platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": cpu_freq_mhz,
        "total_memory_bytes": int(memory.total),
        "available_memory_bytes": int(memory.available),
        "hostname": platform.node() or None,
        "platform": platform.platform(),
        "system": platform.system(),
        "python_version": platform.python_version(),
        "blas": _blas_info(),
        "thread_env": {var: os.environ.get(var) for var in _THREAD_ENV_VARS},
    }


@dataclass
class UnitMetrics:
    """The measured cost of one unit of work (one fit).

    Attributes
    ----------
    wall_s : float
        Elapsed wall-clock seconds.
    cpu_s : float
        Process CPU seconds (user plus system) spent during the unit. Summed across threads,
        so a BLAS-threaded fit reports more CPU than wall time.
    peak_rss_bytes : int
        Highest resident set size seen while the unit ran, the figure a SLURM ``--mem``
        request must cover.
    start_rss_bytes : int
        Resident set size when the unit started, so the unit's own allocation can be read as
        a delta against the process baseline.
    n_samples : int
        How many memory samples the background thread took, recorded so a peak built from too
        few samples is visible rather than silently trusted.
    output_bytes : int or None
        Bytes the unit wrote to disk, set by the caller after the artefacts are saved. Feeds
        the scratch-storage projection. ``None`` when not recorded.
    """

    wall_s: float
    cpu_s: float
    peak_rss_bytes: int
    start_rss_bytes: int
    n_samples: int
    output_bytes: int | None = None

    @property
    def cpu_utilisation(self) -> float:
        """CPU seconds per wall second, with 1 one fully-used core and above 1 multi-core."""
        return self.cpu_s / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def peak_rss_delta_bytes(self) -> int:
        """Peak resident memory above the process baseline at the unit's start."""
        return max(0, self.peak_rss_bytes - self.start_rss_bytes)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable record, including the derived ratios."""
        return {
            "wall_s": self.wall_s,
            "cpu_s": self.cpu_s,
            "cpu_utilisation": round(self.cpu_utilisation, 3),
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_rss_delta_bytes": self.peak_rss_delta_bytes,
            "start_rss_bytes": self.start_rss_bytes,
            "n_samples": self.n_samples,
            "output_bytes": self.output_bytes,
        }


@dataclass
class MeasureHandle:
    """The handle :func:`measure` yields, so the caller can attach the unit's output size.

    The metrics are filled in when the context exits; ``output_bytes`` is set by the caller
    inside the block (for example to the size of the artefact it just wrote) and is copied
    into the final :class:`UnitMetrics`.
    """

    output_bytes: int | None = None
    metrics: UnitMetrics | None = None


@contextmanager
def measure(*, sample_interval_s: float = _DEFAULT_SAMPLE_INTERVAL_S) -> Iterator[MeasureHandle]:
    """Measure the wall time, CPU time, and peak resident memory of the enclosed work.

    A background thread samples the resident set size every ``sample_interval_s`` seconds and
    keeps the maximum, so the peak captures the native BLAS allocations a Python-level memory
    tracer would miss. On exit the handle's ``metrics`` holds the :class:`UnitMetrics`; set
    ``handle.output_bytes`` inside the block to record what the unit wrote.

    Parameters
    ----------
    sample_interval_s : float, optional
        Seconds between resident-memory samples. The default suits fits that run for seconds.

    Yields
    ------
    MeasureHandle
        The handle whose ``metrics`` are populated when the block exits.

    Examples
    --------
    >>> with measure() as unit:  # doctest: +SKIP
    ...     result = fit_gfmm(inputs)
    ...     unit.output_bytes = save(result)
    >>> unit.metrics.wall_s  # doctest: +SKIP
    2.41
    """
    proc = psutil.Process()
    handle = MeasureHandle()
    start_rss = proc.memory_info().rss
    peak = {"rss": start_rss, "n": 0}
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(sample_interval_s):
            try:
                rss = proc.memory_info().rss
            except psutil.Error:
                break
            peak["n"] += 1
            if rss > peak["rss"]:
                peak["rss"] = rss

    sampler = threading.Thread(target=sample, name="rss-sampler", daemon=True)
    cpu_start = proc.cpu_times()
    wall_start = time.monotonic()
    sampler.start()
    try:
        yield handle
    finally:
        wall_s = time.monotonic() - wall_start
        cpu_end = proc.cpu_times()
        stop.set()
        sampler.join(timeout=1.0)
        try:  # a final reading, in case the peak fell between the last sample and here
            rss = proc.memory_info().rss
            peak["rss"] = max(peak["rss"], rss)
        except psutil.Error:
            pass
        cpu_s = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
        handle.metrics = UnitMetrics(
            wall_s=round(wall_s, 4),
            cpu_s=round(cpu_s, 4),
            peak_rss_bytes=int(peak["rss"]),
            start_rss_bytes=int(start_rss),
            n_samples=int(peak["n"]),
            output_bytes=handle.output_bytes,
        )


def path_bytes(path: Path) -> int:
    """Return the size in bytes of a file, or the recursive size of a directory.

    Used to record what a unit wrote, so the per-unit output size projects to the storage a
    full run consumes. A path that does not exist contributes nothing.

    Parameters
    ----------
    path : Path
        A file or directory.

    Returns
    -------
    int
        Total size in bytes.
    """
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return 0


def _percentile(values: list[float], q: float) -> float:
    """Return the ``q``-th percentile (0 to 100) by linear interpolation, no numpy needed."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    low = int(rank)
    frac = rank - low
    if low + 1 >= len(ordered):
        return ordered[-1]
    return ordered[low] + frac * (ordered[low + 1] - ordered[low])


def summarise(metrics: list[UnitMetrics]) -> dict[str, object]:
    """Aggregate per-unit metrics into the distribution a projection is built from.

    The full-run and SLURM projections multiply these measured per-unit figures by the known
    unit count, so the spread (median to 90th percentile to maximum) is what turns a
    calibration sample into a bounded estimate rather than a single guess.

    Parameters
    ----------
    metrics : list of UnitMetrics
        The measured units.

    Returns
    -------
    dict
        ``n_units``; ``total_wall_s`` and ``total_cpu_s``; the wall-second
        ``median`` / ``p90`` / ``max``; the ``peak_rss_bytes`` median and max; the mean
        ``cpu_utilisation``; and ``total_output_bytes`` (``None`` when no unit recorded a
        size). Empty input returns zeroes.
    """
    if not metrics:
        return {"n_units": 0}
    walls = [m.wall_s for m in metrics]
    rss = [float(m.peak_rss_bytes) for m in metrics]
    outputs = [m.output_bytes for m in metrics if m.output_bytes is not None]
    return {
        "n_units": len(metrics),
        "total_wall_s": round(sum(walls), 3),
        "total_cpu_s": round(sum(m.cpu_s for m in metrics), 3),
        "wall_s_median": round(_percentile(walls, 50), 4),
        "wall_s_p90": round(_percentile(walls, 90), 4),
        "wall_s_max": round(max(walls), 4),
        "peak_rss_bytes_median": int(_percentile(rss, 50)),
        "peak_rss_bytes_max": int(max(rss)),
        "cpu_utilisation_mean": round(sum(m.cpu_utilisation for m in metrics) / len(metrics), 3),
        "total_output_bytes": sum(outputs) if outputs else None,
    }
