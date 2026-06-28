"""Tests for the resource-measurement helpers.

Timing and memory are inherently noisy, so these assert direction and rough magnitude (a
busy loop spends CPU, a held allocation raises resident memory, a sleep does not), not exact
figures.
"""

from __future__ import annotations

import time

import pytest
from analysis.profiling import (
    UnitMetrics,
    _percentile,
    capture_hardware,
    measure,
    path_bytes,
    summarise,
)


def test_capture_hardware_reports_plausible_fields() -> None:
    hw = capture_hardware()
    logical, physical, memory = hw["logical_cores"], hw["physical_cores"], hw["total_memory_bytes"]
    assert isinstance(logical, int) and logical >= 1
    assert isinstance(physical, int) and physical >= 1
    assert isinstance(memory, int) and memory > 0
    assert isinstance(hw["blas"], list)
    assert isinstance(hw["thread_env"], dict)
    assert hw["system"] in {"Darwin", "Linux", "Windows"}


def test_measure_times_a_sleep_without_spending_cpu() -> None:
    with measure(sample_interval_s=0.01) as unit:
        time.sleep(0.15)
    assert unit.metrics is not None
    assert unit.metrics.wall_s >= 0.13
    # Sleeping holds no core, so CPU time stays far below wall time.
    assert unit.metrics.cpu_utilisation < 0.5


def test_measure_charges_cpu_for_a_busy_loop() -> None:
    with measure(sample_interval_s=0.01) as unit:
        deadline = time.monotonic() + 0.15
        total = 0
        while time.monotonic() < deadline:
            total += 1
    assert unit.metrics is not None
    assert unit.metrics.cpu_s > 0
    # A tight Python loop keeps one core busy, so utilisation approaches one.
    assert unit.metrics.cpu_utilisation > 0.5


def test_measure_catches_a_held_allocation_as_peak_rss() -> None:
    with measure(sample_interval_s=0.01) as unit:
        block = bytearray(64 * 1024 * 1024)  # 64 MiB, touched so the pages are resident
        block[::4096] = b"\x01" * len(block[::4096])
        time.sleep(0.05)
        del block
    assert unit.metrics is not None
    assert unit.metrics.n_samples > 0
    # The peak should sit well above the baseline; allow generous slack for the allocator.
    assert unit.metrics.peak_rss_delta_bytes >= 20 * 1024 * 1024


def test_measure_records_caller_output_bytes() -> None:
    with measure(sample_interval_s=0.01) as unit:
        unit.output_bytes = 4096
    assert unit.metrics is not None
    assert unit.metrics.output_bytes == 4096
    assert unit.metrics.to_dict()["output_bytes"] == 4096


def test_path_bytes_for_file_and_directory(tmp_path) -> None:
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"y" * 50)
    assert path_bytes(tmp_path / "a.bin") == 100
    assert path_bytes(tmp_path) == 150
    assert path_bytes(tmp_path / "missing") == 0


def test_percentile_interpolates() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert _percentile(values, 0) == 1.0
    assert _percentile(values, 50) == pytest.approx(2.5)
    assert _percentile(values, 100) == 4.0
    assert _percentile([], 50) == 0.0


def test_summarise_aggregates_the_distribution() -> None:
    metrics = [
        UnitMetrics(
            wall_s=1.0,
            cpu_s=2.0,
            peak_rss_bytes=100,
            start_rss_bytes=10,
            n_samples=5,
            output_bytes=1000,
        ),
        UnitMetrics(
            wall_s=3.0,
            cpu_s=6.0,
            peak_rss_bytes=300,
            start_rss_bytes=10,
            n_samples=5,
            output_bytes=2000,
        ),
    ]
    out = summarise(metrics)
    assert out["n_units"] == 2
    assert out["total_wall_s"] == 4.0
    assert out["wall_s_median"] == pytest.approx(2.0)
    assert out["wall_s_max"] == 3.0
    assert out["peak_rss_bytes_max"] == 300
    assert out["cpu_utilisation_mean"] == pytest.approx(2.0)
    assert out["total_output_bytes"] == 3000


def test_summarise_handles_empty_input() -> None:
    assert summarise([])["n_units"] == 0
