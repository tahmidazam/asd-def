"""Tests for the figures package: the selection figure, saving, and run resolution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from figures import data, paths, style
from figures.nmin import nmin_figure
from figures.publish import FigureSpec, publish_figure
from figures.replication import replication_figure
from figures.reproduction import reproduction_figure
from figures.selection import selection_figure
from figures.stability import stability_figure
from matplotlib.figure import Figure

_CATEGORIES = [
    "anxiety/mood",
    "attention",
    "disruptive behavior",
    "self-injury",
    "social/communication",
    "restricted/repetitive",
    "developmental",
]


def _synthetic_summary(n_max: int = 10) -> pd.DataFrame:
    """Build a summary frame shaped like an ``analysis select`` summary."""
    k = np.arange(1, n_max + 1, dtype=float)
    rows: dict[str, np.ndarray] = {"n_components": k}
    for name in ("bic", "aic", "caic", "sabic", "awe"):
        rows[f"{name}_mean"] = 7_000_000.0 - 50_000.0 * k
        rows[f"{name}_std"] = 1_000.0 + 100.0 * k
    rows["val_log_likelihood_mean"] = -330.0 + 40.0 * (1.0 - np.exp(-(k - 1.0) / 2.0))
    rows["val_log_likelihood_std"] = 1.0 + 0.1 * k
    rows["relative_entropy_mean"] = 0.9 - 0.03 * k
    rows["relative_entropy_std"] = np.full_like(k, 0.02)
    rows["smallest_class_proportion_mean"] = 0.4 / k
    rows["smallest_class_proportion_std"] = np.full_like(k, 0.01)
    return pd.DataFrame(rows)


def test_selection_figure_structure() -> None:
    fig = selection_figure(_synthetic_summary(), reference_k=4, criteria=("bic", "aic", "caic"))
    assert isinstance(fig, Figure)

    axes = fig.get_axes()
    assert len(axes) == 3  # one axis per panel

    criterion_lines = [ln for ln in axes[0].get_lines() if ln.get_label() in {"BIC", "AIC", "CAIC"}]
    assert len(criterion_lines) == 3
    assert axes[0].get_xlabel()

    reference_lines = []
    for ax in axes:
        for ln in ax.get_lines():
            xdata = np.asarray(ln.get_xdata())
            if xdata.size == 2 and xdata[0] == 4 and xdata[-1] == 4:
                reference_lines.append(ln)
    assert reference_lines  # a reference line at K = 4 in the panels


def test_selection_figure_missing_columns() -> None:
    summary = _synthetic_summary().drop(columns=["bic_mean"])
    with pytest.raises(ValueError, match="missing columns"):
        selection_figure(summary)


def test_save_figure_writes_files(tmp_path: Path) -> None:
    fig = selection_figure(_synthetic_summary())
    stem = tmp_path / "select" / "abcd" / "selection_criteria"
    written = style.save_figure(fig, stem, formats=("pdf", "png"))

    assert [path.suffix for path in written] == [".pdf", ".png"]
    for path in written:
        assert path.is_file()
        assert path.stat().st_size > 0
    meta = json.loads(stem.with_suffix(".json").read_text())
    assert "figures_version" in meta
    assert "generated_at" in meta


def _write_manifest(run_dir: Path, *, status: str, finished_at: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": status, "finished_at": finished_at}), encoding="utf-8"
    )


def test_resolve_run_picks_latest_ok(tmp_path: Path) -> None:
    stage = tmp_path / "artefacts" / "select"
    _write_manifest(stage / "aaaa", status="ok", finished_at="2026-01-01T00:00:00+00:00")
    _write_manifest(stage / "bbbb", status="ok", finished_at="2026-06-01T00:00:00+00:00")
    _write_manifest(stage / "cccc", status="failed", finished_at="2026-12-01T00:00:00+00:00")

    assert data.resolve_run(tmp_path, "select").name == "bbbb"


def test_resolve_run_explicit_and_missing(tmp_path: Path) -> None:
    stage = tmp_path / "artefacts" / "select"
    _write_manifest(stage / "aaaa", status="ok", finished_at="2026-01-01T00:00:00+00:00")

    assert data.resolve_run(tmp_path, "select", "aaaa").name == "aaaa"
    with pytest.raises(FileNotFoundError):
        data.resolve_run(tmp_path, "select", "zzzz")


def test_resolve_run_none_completed(tmp_path: Path) -> None:
    stage = tmp_path / "artefacts" / "select"
    _write_manifest(stage / "cccc", status="failed", finished_at="2026-12-01T00:00:00+00:00")
    with pytest.raises(FileNotFoundError):
        data.resolve_run(tmp_path, "select")


# ---- replication figure ------------------------------------------------------
def _signature_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    index = pd.Index(range(4), name="class")
    spark = pd.DataFrame(rng.uniform(-1, 1, (4, 7)), index=index, columns=_CATEGORIES)
    ssc = (spark + rng.normal(0, 0.1, (4, 7))).clip(-1, 1)
    return spark, ssc


def _replication_metrics() -> dict:
    return {
        "overall_correlation": 0.76,
        "category_correlation": dict.fromkeys(_CATEGORIES, 0.9),
        "p_value": 0.005,
        "n_ssc": 771,
    }


def test_replication_figure_structure() -> None:
    spark, ssc = _signature_pair()
    fig = replication_figure(spark, ssc, _replication_metrics())
    assert isinstance(fig, Figure)
    assert len(fig.get_axes()) == 2  # scatter + per-category bars


def test_replication_figure_mismatched_signatures() -> None:
    spark, ssc = _signature_pair()
    with pytest.raises(ValueError, match="share shape and columns"):
        replication_figure(spark, ssc.iloc[:, :5], _replication_metrics())


# ---- per-category replication comparison -------------------------------------
def test_replication_figure_with_comparison() -> None:
    spark, ssc = _signature_pair()
    primary = _replication_metrics()
    comparison = {**_replication_metrics(), "overall_correlation": 0.875}
    fig = replication_figure(spark, ssc, primary, comparison)
    assert isinstance(fig, Figure)
    assert len(fig.get_axes()) == 2


# ---- stability figure --------------------------------------------------------
def _stability_inputs() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rng = np.random.default_rng(1)
    comparisons = pd.DataFrame(
        {
            "overall_correlation": rng.uniform(0.85, 0.98, 50),
            "adjusted_rand_index": rng.uniform(0.5, 0.8, 50),
        }
    )
    aggregate = {
        "category_correlation_mean": dict.fromkeys(_CATEGORIES, 0.9),
        "n_reps": 50,
        "frac": 0.5,
    }
    overlap = pd.DataFrame(
        np.eye(4) * 0.8 + 0.05, index=pd.Index(range(4), name="source"), columns=range(4)
    )
    return comparisons, aggregate, overlap


def test_stability_figure_structure() -> None:
    comparisons, aggregate, overlap = _stability_inputs()
    fig = stability_figure(comparisons, aggregate, overlap)
    assert isinstance(fig, Figure)
    assert len(fig.get_axes()) >= 3  # three panels (plus the overlap colourbar)


def test_stability_figure_missing_column() -> None:
    comparisons, aggregate, overlap = _stability_inputs()
    with pytest.raises(ValueError, match="missing column"):
        stability_figure(comparisons.drop(columns=["adjusted_rand_index"]), aggregate, overlap)


# ---- nmin figure -------------------------------------------------------------
def _nmin_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(2)
    sizes = [600, 1000, 1500, 2200, 3200, 5000]
    rows = [
        {
            "size": s,
            "overall_correlation": min(0.7 + 0.00005 * s + float(rng.normal(0, 0.03)), 1.0),
            "smallest_class_proportion": 0.15,
        }
        for s in sizes
        for _ in range(5)
    ]
    per_fit = pd.DataFrame(rows)
    summary = per_fit.groupby("size", as_index=False)[
        ["overall_correlation", "smallest_class_proportion"]
    ].mean()
    metrics = {"floor": 1000, "floor_ci90": [600, 3000], "benchmark": 0.9}
    return per_fit, summary, metrics


def test_nmin_figure_structure() -> None:
    per_fit, summary, metrics = _nmin_inputs()
    fig = nmin_figure(per_fit, summary, metrics)
    assert isinstance(fig, Figure)
    assert len(fig.get_axes()) == 2  # recovery + smallest-class panels


def test_nmin_figure_missing_column() -> None:
    per_fit, summary, metrics = _nmin_inputs()
    with pytest.raises(ValueError, match="missing columns"):
        nmin_figure(per_fit.drop(columns=["smallest_class_proportion"]), summary, metrics)


# ---- reproduction figure -----------------------------------------------------
_NAMED = ["Social/behavioral", "Moderate challenges", "Mixed ASD with DD", "Broadly affected"]


def _reproduction_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    rng = np.random.default_rng(3)
    our = pd.DataFrame(
        rng.uniform(-1, 1, (4, 7)), index=pd.Index(range(4), name="class"), columns=_CATEGORIES
    )
    published = pd.DataFrame(
        rng.uniform(-1, 1, (4, 7)), index=pd.Index(_NAMED, name="named_class"), columns=_CATEGORIES
    )
    alignment = {
        "mapping": {"2": _NAMED[0], "0": _NAMED[1], "1": _NAMED[2], "3": _NAMED[3]},
        "correlations": {"2": 0.85, "0": None, "1": 0.97, "3": None},
        "overall_correlation": 0.9,
        "anchors_hold": True,
    }
    our_proportions = {0: 0.29, 1: 0.18, 2: 0.39, 3: 0.15}
    published_proportions = {_NAMED[0]: 0.37, _NAMED[1]: 0.34, _NAMED[2]: 0.19, _NAMED[3]: 0.10}
    return our, published, alignment, our_proportions, published_proportions


def test_reproduction_figure_structure() -> None:
    our, published, alignment, our_props, pub_props = _reproduction_inputs()
    fig = reproduction_figure(our, published, alignment, our_props, pub_props)
    assert isinstance(fig, Figure)
    assert len(fig.get_axes()) == 4  # one panel per named class


def test_reproduction_figure_mismatched_categories() -> None:
    our, published, alignment, our_props, pub_props = _reproduction_inputs()
    with pytest.raises(ValueError, match="share their categories"):
        reproduction_figure(our.iloc[:, :5], published, alignment, our_props, pub_props)


# ---- publishing --------------------------------------------------------------
def _render_artefact(root: Path, stage: str, run_hash: str, file_name: str) -> None:
    """Write a source-run manifest and a rendered PNG, as the build commands would."""
    run_directory = root / "artefacts" / stage / run_hash
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "manifest.json").write_text(
        json.dumps(
            {"status": "ok", "finished_at": "2026-06-01T00:00:00+00:00", "git_commit": "abc1234"}
        ),
        encoding="utf-8",
    )
    png = paths.figure_stem(root, stage, run_hash, file_name).with_suffix(".png")
    png.parent.mkdir(parents=True, exist_ok=True)
    png.write_bytes(b"\x89PNG\r\n")


def test_publish_figure_copies_and_records(tmp_path: Path) -> None:
    spec = FigureSpec("select", "select", "selection_criteria")
    _render_artefact(tmp_path, "select", "abcd1234abcd1234", "selection_criteria")

    destination = publish_figure(tmp_path, spec)

    assert destination == paths.docs_figures_dir(tmp_path) / "selection_criteria.png"
    assert destination.is_file()
    sidecar = json.loads(destination.with_suffix(".json").read_text())
    assert sidecar["figure"] == "select"
    assert sidecar["source_run"] == "abcd1234abcd1234"
    assert sidecar["source_git_commit"] == "abc1234"


def test_publish_figure_missing_render(tmp_path: Path) -> None:
    spec = FigureSpec("select", "select", "selection_criteria")
    _write_manifest(
        tmp_path / "artefacts" / "select" / "abcd",
        status="ok",
        finished_at="2026-06-01T00:00:00+00:00",
    )
    with pytest.raises(FileNotFoundError, match="run `figures select` first"):
        publish_figure(tmp_path, spec)
