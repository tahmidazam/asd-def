"""Tests for the figures package: the selection figure, saving, and run resolution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from figures import data, paths, style
from figures.atlas import atlas_figure
from figures.attribution import attribution_figure, mover_contrast_figure
from figures.category_decomposition import (
    category_decomposition_figure,
    category_heatmaps_figure,
)
from figures.demographic_conditioning import demographic_conditioning_figure
from figures.dense_features import dense_feature_figure
from figures.invariance import invariance_process_figure
from figures.nmin import nmin_figure
from figures.pairwise import pairwise_trajectory_figure
from figures.prevalence import proportion_curve_figure, stacked_area_figure
from figures.publish import FigureSpec, publish_figure
from figures.referent_decomposition import referent_decomposition_figure
from figures.replication import replication_figure
from figures.reproduction import reproduction_figure
from figures.roughness import roughness_figure
from figures.selection import selection_figure
from figures.stability import stability_figure
from figures.trajectory import trajectory_figure
from figures.trajectory_local import (
    panels_figure,
    plane_figure,
    referent_figure,
    specificity_figure,
)
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


# ---- trajectory and roughness figures -----------------------------------------
_CLASS_NAMES = ("Moderate challenges", "Mixed ASD with DD", "Social/behavioral", "Broadly affected")


def _trajectory_embedding() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for c in range(4):
        rows.append(
            {
                "kind": "anchor",
                "ref_class": c,
                "class_name": _CLASS_NAMES[c],
                "stratum": "",
                "order": -1,
                "ld1": float(c),
                "ld2": float(-c),
                "ld3": 0.0,
                "jaccard": float("nan"),
                "reorganised": False,
            }
        )
        for s in range(6):
            jaccard = float(rng.uniform(0.3, 0.9))
            rows.append(
                {
                    "kind": "stratum",
                    "ref_class": c,
                    "class_name": _CLASS_NAMES[c],
                    "stratum": f"Q{s + 1}",
                    "order": s,
                    "ld1": float(c) + rng.normal() * 0.2,
                    "ld2": float(-c) + rng.normal() * 0.2,
                    "ld3": 0.0,
                    "jaccard": jaccard,
                    "reorganised": jaccard < 0.5,
                }
            )
    return pd.DataFrame(rows), {"axis": "age_at_diagnosis", "n_strata": 6}


def test_trajectory_figure_structure() -> None:
    embedding, meta = _trajectory_embedding()
    fig = trajectory_figure(embedding, meta)
    assert isinstance(fig, Figure)
    assert len(fig.get_axes()) >= 4


def test_trajectory_figure_missing_column() -> None:
    embedding, meta = _trajectory_embedding()
    with pytest.raises(ValueError, match="missing columns"):
        trajectory_figure(embedding.drop(columns=["ld2"]), meta)


def _pairwise_trajectory() -> tuple[pd.DataFrame, dict]:
    """Build an adjacent pairwise-trajectory table shaped like a pairwise ``drift`` run."""
    positions = [1.0, 2.0, 3.0, 4.0]
    rows: list[dict] = []
    for step, position in enumerate(positions):
        for ref_class in range(4):
            drift = 0.2 + 0.1 * ref_class + 0.05 * step
            rows.append(
                {
                    "query_stratum": f"Q{step + 1}",
                    "reference_stratum": f"Q{step + 2}",
                    "position": position,
                    "ref_class": ref_class,
                    "drift": drift,
                    "drift_vs_separation": drift,
                    "centroid_quality": 0.3,
                    "overall_quality": 0.25 + 0.02 * step,
                }
            )
    return pd.DataFrame(rows), {"axis": "age_at_diagnosis", "mode": "adjacent"}


def test_pairwise_trajectory_figure_structure() -> None:
    trajectory, meta = _pairwise_trajectory()
    fig = pairwise_trajectory_figure(trajectory, dict(enumerate(_CLASS_NAMES)), meta)
    assert isinstance(fig, Figure)
    assert len(fig.get_axes()) == 2


def test_pairwise_trajectory_figure_missing_column() -> None:
    trajectory, meta = _pairwise_trajectory()
    with pytest.raises(ValueError, match="missing columns"):
        pairwise_trajectory_figure(trajectory.drop(columns=["drift_vs_separation"]), {}, meta)


def test_pairwise_trajectory_figure_rejects_all_pairs() -> None:
    trajectory, meta = _pairwise_trajectory()
    # Two comparisons at the same position for one class is the all-pairs shape, not a trajectory.
    doubled = pd.concat([trajectory, trajectory.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="all-pairs"):
        pairwise_trajectory_figure(doubled, {}, meta)


def _roughness_frames() -> tuple[dict, dict]:
    def rough() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ref_class": range(4),
                "class_name": _CLASS_NAMES,
                "step": [1.0, 2.0, 1.5, 2.5],
                "sampling_noise": [0.5, 0.5, 0.5, 0.5],
                "snr": [2.0, 4.0, 3.0, 5.0],
            }
        )

    def direction() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ref_class": range(4),
                "class_name": _CLASS_NAMES,
                "net": [2.0, 5.0, 3.0, 6.0],
                "null95": [1.0, 1.0, 1.0, 1.0],
                "p": [0.01, 0.001, 0.2, 0.001],
                "significant": [True, True, False, True],
            }
        )

    roughness_by_axis = {"age at diagnosis": rough(), "diagnostic era": rough()}
    directional_by_axis = {"age at diagnosis": direction(), "diagnostic era": direction()}
    return roughness_by_axis, directional_by_axis


def test_roughness_figure_structure() -> None:
    roughness_by_axis, directional_by_axis = _roughness_frames()
    fig = roughness_figure(roughness_by_axis, directional_by_axis)
    assert isinstance(fig, Figure)
    assert len(fig.get_axes()) == 2


def test_roughness_figure_mismatched_axes() -> None:
    roughness_by_axis, directional_by_axis = _roughness_frames()
    directional_by_axis = {"only one": next(iter(directional_by_axis.values()))}
    with pytest.raises(ValueError, match="share the same axis labels"):
        roughness_figure(roughness_by_axis, directional_by_axis)


def test_resolve_run_axis_filter(tmp_path: Path) -> None:
    stage = tmp_path / "artefacts" / "trajectory"

    def manifest(name: str, axis: str, finished: str) -> None:
        run = stage / name
        run.mkdir(parents=True, exist_ok=True)
        (run / "manifest.json").write_text(
            json.dumps({"status": "ok", "finished_at": finished, "params": {"axis": axis}}),
            encoding="utf-8",
        )

    manifest("aaaa", "age_at_diagnosis", "2026-01-01T00:00:00+00:00")
    manifest("bbbb", "era", "2026-02-01T00:00:00+00:00")
    manifest("cccc", "age_at_diagnosis", "2026-03-01T00:00:00+00:00")
    assert data.resolve_run(tmp_path, "trajectory", axis="age_at_diagnosis").name == "cccc"
    assert data.resolve_run(tmp_path, "trajectory", axis="era").name == "bbbb"


def _attribution_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic attribute-stage tables: two classes, three strata, three categories."""
    strata = ["Q1", "Q2", "Q10"]
    names = {0: "Broadly affected", 1: "Social/behavioral"}
    cats = ["developmental", "anxiety/mood", "attention"]
    srows, crows, mrows = [], [], []
    rng = np.random.default_rng(0)
    for cls in (0, 1):
        for s in strata:
            srows.append(
                {
                    "stratum": s,
                    "ref_class": cls,
                    "class_name": names[cls],
                    "n_stayers": 100,
                    "n_leavers": 10 * (cls + 1),
                    "n_joiners": 5,
                    "churn": 0.2 + 0.1 * cls,
                    "jaccard": 0.4 if s == "Q1" else 0.7,
                    "ari": 0.5,
                    "top_shift_feature": "f0",
                    "top_shift_category": cats[cls],
                    "top_mover_feature": "f0",
                }
            )
            for cat in cats:
                crows.append(
                    {
                        "stratum": s,
                        "ref_class": cls,
                        "category": cat,
                        "contribution": rng.uniform(0, 1),
                    }
                )
            for f in range(4):
                mrows.append(
                    {
                        "stratum": s,
                        "ref_class": cls,
                        "feature": f"f{f}",
                        "effect": rng.uniform(-1, 1),
                        "magnitude": rng.uniform(0, 1),
                        "p_value": 0.01,
                        "fdr_significant": f < 2,
                    }
                )
    return pd.DataFrame(srows), pd.DataFrame(crows), pd.DataFrame(mrows)


def test_attribution_figure_structure() -> None:
    summary, category, movers = _attribution_tables()
    fig = attribution_figure(summary, category, {"axis": "age_at_diagnosis"})
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 2  # two panels plus the colourbar


def test_mover_contrast_figure_structure() -> None:
    summary, category, movers = _attribution_tables()
    fig = mover_contrast_figure(summary, movers, {"axis": "era"})
    assert isinstance(fig, Figure)


def test_mover_contrast_figure_handles_empty_class() -> None:
    summary, category, movers = _attribution_tables()
    movers = movers[movers["ref_class"] == 0]  # class 1 has no mover rows
    fig = mover_contrast_figure(summary, movers, {"axis": "age_at_diagnosis"})
    assert isinstance(fig, Figure)


def _invariance_process() -> tuple[pd.DataFrame, dict]:
    """A stored fluctuation process shaped like an ``invariance`` run's process table."""
    t = np.linspace(0.0, 1.0, 60)
    observed = 40.0 * t * (1.0 - t) + 1e-6  # a bump peaking mid-axis, pinned at the ends
    return pd.DataFrame(
        {
            "t": t,
            "position": 2.0 + 15.0 * t,
            "observed": observed,
            "null_q50": np.full_like(t, 0.05),
            "null_q95": np.full_like(t, 0.09),
        }
    ), {"axis": "age_at_diagnosis", "top_block": "class 2 x social/communication"}


def test_invariance_process_figure_structure() -> None:
    process, meta = _invariance_process()
    fig = invariance_process_figure(process, meta)
    assert isinstance(fig, Figure)
    ax = fig.get_axes()[0]
    assert ax.get_yscale() == "log"  # the null band and the excursion span orders of magnitude
    assert ax.get_xlabel()


# ---------------------------------------------------------------------------------------------
# The local-trajectory (score-invariance recast) figures.
# ---------------------------------------------------------------------------------------------


def _local_trajectory_tables(capture_value: float = 0.3):
    """Return a synthetic plane table and capture table for the local-trajectory figures."""
    plane_rows: list[dict] = []
    capture_rows: list[dict] = []
    for c in range(4):
        plane_rows.append(
            {
                "kind": "anchor",
                "ref_class": c,
                "class_name": _CLASS_NAMES[c],
                "focal_index": -1,
                "position": float("nan"),
                "ld1": float(c),
                "ld2": float(-c),
                "ld1_lo": float("nan"),
                "ld1_hi": float("nan"),
                "ld2_lo": float("nan"),
                "ld2_hi": float("nan"),
                "cov11": 0.4,
                "cov12": 0.05,
                "cov22": 0.3,
                "capture": capture_value,
            }
        )
        for s in range(8):
            ld1 = float(c) + s * 0.03
            ld2 = float(-c) + s * 0.02
            plane_rows.append(
                {
                    "kind": "focal",
                    "ref_class": c,
                    "class_name": _CLASS_NAMES[c],
                    "focal_index": s,
                    "position": 2010.0 + s,
                    "ld1": ld1,
                    "ld2": ld2,
                    "ld1_lo": ld1 - 0.04,
                    "ld1_hi": ld1 + 0.04,
                    "ld2_lo": ld2 - 0.03,
                    "ld2_hi": ld2 + 0.03,
                    "cov11": float("nan"),
                    "cov12": float("nan"),
                    "cov22": float("nan"),
                    "capture": float("nan"),
                }
            )
        capture_rows.append(
            {"ref_class": c, "class_name": _CLASS_NAMES[c], "capture": capture_value}
        )
    return pd.DataFrame(plane_rows), pd.DataFrame(capture_rows)


def test_local_plane_figure_structure() -> None:
    plane, capture = _local_trajectory_tables()
    fig = plane_figure(plane, capture, {"axis": "era"})
    assert isinstance(fig, Figure)
    assert fig.get_axes()  # a panel and its colourbar


def test_local_panels_figure_structure() -> None:
    plane, capture = _local_trajectory_tables()
    fig = panels_figure(plane, capture, {"axis": "age_at_diagnosis"})
    assert isinstance(fig, Figure)
    assert len(fig.get_axes()) >= 4


def test_local_specificity_figure_orders_axes() -> None:
    rng = np.random.default_rng(1)
    rows: list[dict] = []
    magnitudes = {
        "era": 2.8,
        "age_at_diagnosis": 6.0,
        "area_deprivation": 2.1,
        "household_income": 1.9,
        "random": 1.3,
    }
    for axis_name, base in magnitudes.items():
        for c in range(4):
            rows.append(
                {
                    "axis_name": axis_name,
                    "ref_class": c,
                    "class_name": _CLASS_NAMES[c],
                    "endpoint_magnitude": base + rng.normal() * 0.1,
                }
            )
    fig = specificity_figure(pd.DataFrame(rows), {"timing_axes": ["era", "age_at_diagnosis"]})
    assert isinstance(fig, Figure)
    ax = fig.get_axes()[0]
    # The two timing axes and the three controls are all drawn.
    assert len(ax.get_xticklabels()) == 5


def _referent_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return synthetic ``referent`` and ``referent_contrast`` frames for the H0G figure."""
    instruments = {
        "current_state": {"rbsr": 49, "cbcl_6_18": 144},
        "retrospective": {"scq": 34, "background_history_child": 11},
    }
    grain_rows: list[dict] = []
    contrast_rows: list[dict] = []
    for c in range(4):
        rms_current = 0.06 + 0.02 * c
        rms_retrospective = 0.15 + 0.05 * c
        for referent, rms in (("current_state", rms_current), ("retrospective", rms_retrospective)):
            n = sum(instruments[referent].values())
            grain_rows.append(
                {
                    "axis": "era",
                    "grain_kind": "referent",
                    "grain": referent,
                    "referent": referent,
                    "ref_class": c,
                    "class_name": _CLASS_NAMES[c],
                    "n_features": n,
                    "rms": rms,
                    "share": 0.4,
                    "n_feature_reject": 10,
                }
            )
            for instrument, ni in instruments[referent].items():
                grain_rows.append(
                    {
                        "axis": "era",
                        "grain_kind": "instrument",
                        "grain": instrument,
                        "referent": referent,
                        "ref_class": c,
                        "class_name": _CLASS_NAMES[c],
                        "n_features": ni,
                        "rms": rms * (0.8 + 0.1 * (instrument == "scq")),
                        "share": 0.2,
                        "n_feature_reject": 5,
                    }
                )
        contrast = rms_current - rms_retrospective
        contrast_rows.append(
            {
                "axis": "era",
                "ref_class": c,
                "class_name": _CLASS_NAMES[c],
                "contrast": contrast,
                "ci_low": contrast - 0.02,
                "ci_high": contrast + 0.02,
                "p_value": 0.002,
                "reject": True,
                "rms_current": rms_current,
                "rms_retrospective": rms_retrospective,
                "share_current": 0.4,
                "share_retrospective": 0.6,
                "mechanism": "timing" if contrast > 0 else "population",
            }
        )
    return pd.DataFrame(grain_rows), pd.DataFrame(contrast_rows)


def test_local_referent_figure_structure() -> None:
    grains, contrast = _referent_tables()
    fig = referent_figure(grains, contrast, {"axis": "era"})
    assert isinstance(fig, Figure)
    # One panel per class.
    assert len([ax for ax in fig.get_axes() if ax.has_data()]) >= 4


def _synthetic_prevalence() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ``proportion_curve`` and ``slopes`` frames shaped like a ``prevalence`` run."""
    positions = np.linspace(2009.0, 2025.0, 8)
    pooled = (0.25, 0.16, 0.39, 0.20)
    slopes = (-0.05, -0.08, 0.05, 0.06)
    curve_rows: list[dict] = []
    slope_rows: list[dict] = []
    for c in range(4):
        centred = positions - positions.mean()
        proportion = pooled[c] * np.exp(slopes[c] * centred)
        for pos, prop in zip(positions, proportion, strict=True):
            curve_rows.append(
                {
                    "ref_class": c,
                    "class_name": _CLASS_NAMES[c],
                    "position": float(pos),
                    "corrected": float(prop),
                    "naive": float(prop) * 0.99,
                    "band_lo": float(prop) * 0.95,
                    "band_hi": float(prop) * 1.05,
                    "pooled": pooled[c],
                }
            )
        slope_rows.append(
            {
                "kind": "corrected",
                "ref_class": c,
                "class_name": _CLASS_NAMES[c],
                "slope": slopes[c],
                "odds_ratio": float(np.exp(slopes[c])),
                "reject": True,
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(slope_rows)


def test_prevalence_panels_figure_structure() -> None:
    curve, slopes = _synthetic_prevalence()
    fig = proportion_curve_figure(curve, slopes, {"axis": "era"})
    assert isinstance(fig, Figure)
    # One panel per class.
    assert len(fig.get_axes()) == 4


def test_prevalence_stacked_figure_structure() -> None:
    curve, slopes = _synthetic_prevalence()
    fig = stacked_area_figure(curve, {"axis": "age_at_diagnosis"})
    assert isinstance(fig, Figure)
    ax = fig.get_axes()[0]
    # The composition axis runs from zero to one, and all four classes appear in the legend.
    assert ax.get_ylim() == (0.0, 1.0)
    legend = ax.get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == 4


def _category_decomposition_tables() -> tuple[dict, dict]:
    """Return synthetic grain-magnitude and feature-displacement frames for both timing axes."""
    rng = np.random.default_rng(0)
    grains: dict[str, pd.DataFrame] = {}
    features: dict[str, pd.DataFrame] = {}
    for axis in ("era", "age_at_diagnosis"):
        grain_rows = []
        feature_rows = []
        for c in range(4):
            for focal in range(3):
                grain_rows.append(
                    {
                        "grain": "class",
                        "ref_class": c,
                        "class_name": _CLASS_NAMES[c],
                        "focal_index": focal,
                        "position": float(focal),
                        "magnitude": 1.0,
                        "band_lo": 0.5,
                        "band_hi": 1.5,
                    }
                )
                for cat in _CATEGORIES:
                    grain_rows.append(
                        {
                            "grain": f"category:{cat}",
                            "ref_class": c,
                            "class_name": _CLASS_NAMES[c],
                            "focal_index": focal,
                            "position": float(focal),
                            "magnitude": float(rng.uniform(0.1, 1.0)),
                            "band_lo": 0.0,
                            "band_hi": 1.0,
                        }
                    )
            for j, cat in enumerate(_CATEGORIES):
                feature_rows.append(
                    {
                        "ref_class": c,
                        "class_name": _CLASS_NAMES[c],
                        "feature": f"f{c}_{j}_age_mos",
                        "category": cat,
                        "displacement": float((j - 3) * 0.2),
                        "ci_low": -0.1,
                        "ci_high": 0.1,
                        "p_value": 0.001,
                        "reject": True,
                        "covers_zero": False,
                    }
                )
        grains[axis] = pd.DataFrame(grain_rows)
        features[axis] = pd.DataFrame(feature_rows)
    return grains, features


def test_category_decomposition_figure_structure() -> None:
    grains, features = _category_decomposition_tables()
    fig = category_decomposition_figure(grains, features, {"axes": ["era", "age_at_diagnosis"]})
    assert isinstance(fig, Figure)
    # Two category heatmaps, four per-class leading-feature panels, and the shared colour bar.
    assert len(fig.get_axes()) == 7
    legend = fig.legends[0]
    assert len(legend.get_texts()) >= 1


def test_category_heatmaps_figure_structure() -> None:
    grains, _ = _category_decomposition_tables()
    fig = category_heatmaps_figure(grains, {"axes": ["era", "age_at_diagnosis"]})
    assert isinstance(fig, Figure)
    # Two category heatmaps and their shared colour bar, and no lollipop panels or legend.
    assert len(fig.get_axes()) == 3
    assert not fig.legends


def test_dense_feature_figure_structure() -> None:
    _, features = _category_decomposition_tables()
    # Mark enough features significant that the dense matrix has rows to draw.
    for axis in features:
        features[axis]["reject"] = True
    fig = dense_feature_figure(features, {"axes": ["era", "age_at_diagnosis"]})
    assert isinstance(fig, Figure)
    # The group sidebar, the heatmap, and the colour bar.
    assert len(fig.get_axes()) == 3
    legend = fig.legends[0]
    assert len(legend.get_texts()) >= 1


def test_referent_decomposition_figure_structure() -> None:
    classes = list(range(4))
    contrast = pd.DataFrame(
        {
            "ref_class": classes,
            "class_name": [_CLASS_NAMES[c] for c in classes],
            "contrast": [-0.15, -0.05, -0.09, -0.23],
            "ci_low": [-0.17, -0.07, -0.10, -0.27],
            "ci_high": [-0.12, -0.02, -0.07, -0.17],
            "p_value": [0.002] * 4,
            "reject": [True] * 4,
            "mechanism": ["population"] * 4,
        }
    )
    grain_rows = []
    instruments = {
        "cbcl_6_18": "current_state",
        "rbsr": "current_state",
        "scq": "retrospective",
        "background_history_child": "retrospective",
    }
    for c in classes:
        for name, referent in instruments.items():
            grain_rows.append(
                {
                    "grain_kind": "instrument",
                    "grain": name,
                    "referent": referent,
                    "ref_class": c,
                    "class_name": _CLASS_NAMES[c],
                    "rms": 0.1 + 0.05 * c,
                    "share": 0.25,
                    "n_features": 40,
                    "n_feature_reject": 10,
                }
            )
    grains = pd.DataFrame(grain_rows)
    fig = referent_decomposition_figure(grains, contrast, {"axis": "era"})
    assert isinstance(fig, Figure)
    assert len(fig.get_axes()) == 2


def test_atlas_figure_structure() -> None:
    classes = list(range(4))
    axes = [
        ("era", "Diagnostic era", "timing", 0.20),
        ("age_at_diagnosis", "Age at diagnosis", "timing", 0.40),
        ("household_income", "Household income", "covariate", 0.21),
        ("area_deprivation", "Area deprivation", "covariate", 0.15),
        ("random", "Random ordering", "random", 0.09),
    ]
    rows = []
    for name, label, kind, base in axes:
        for c in classes:
            rows.append(
                {
                    "axis_name": name,
                    "label": label,
                    "kind": kind,
                    "ref_class": c,
                    "class_name": _CLASS_NAMES[c],
                    "endpoint_magnitude": base + 0.03 * c,
                    "n_joint": 11000,
                    "bandwidth": 1.5,
                }
            )
    fig = atlas_figure(pd.DataFrame(rows), {})
    assert isinstance(fig, Figure)
    # Three kind panels (timing, covariate, random) plus the shared colour bar.
    assert len(fig.get_axes()) == 4
    timing_panel = fig.get_axes()[0]
    # The timing panel holds both timing axes, the larger mover (age at diagnosis) on top.
    ytick_labels = [t.get_text() for t in timing_panel.get_yticklabels()]
    assert ytick_labels == ["Age at diagnosis", "Diagnostic era"]
    # Only the bottom panel (the random floor, index 2 before the colour bar) carries the labels.
    bottom_panel = fig.get_axes()[2]
    assert [t.get_text() for t in bottom_panel.get_xticklabels()][0] == _CLASS_NAMES[0]


def _demographic_conditioning_table(axis: str) -> pd.DataFrame:
    """Build a synthetic ``demographic_conditioning_<axis>`` table for the smoke test."""
    covariates = [
        ("household_income", "Household income", "ses", "ordinal", 0.004),
        ("family_type", "Family type", "family", "onehot", 0.006),
        ("maternal_age_at_birth", "Maternal age at birth", "parental", "scalar", 0.001),
        ("sex", "Sex assigned at birth", "individual", "binary", 0.015),
    ]
    rows = []
    for name, label, kind, coding, r2 in covariates:
        for c in range(4):
            rows.append(
                {
                    "name": name,
                    "label": label,
                    "kind": kind,
                    "coding": coding,
                    "ref_class": c,
                    "class_name": _CLASS_NAMES[c],
                    "shrinkage": r2 * (c + 1),
                    "raw_magnitude": 1.0,
                    "conditioned_magnitude": 1.0 - r2 * (c + 1),
                    "axis_r2": r2,
                    "n_columns": 1,
                    "n_joint": 11000,
                }
            )
    return pd.DataFrame(rows)


def test_demographic_conditioning_figure_structure() -> None:
    tables = {
        "era": _demographic_conditioning_table("era"),
        "age_at_diagnosis": _demographic_conditioning_table("age_at_diagnosis"),
    }
    fig = demographic_conditioning_figure(tables, {"axes": list(tables)})
    assert isinstance(fig, Figure)
    # The kind strip, the axis-R^2 ceiling panel, the shrinkage panel, and the shared colour bar.
    assert len(fig.get_axes()) == 4
    # The shrinkage panel carries eight columns: four classes for each of the two axes.
    shrinkage_panel = fig.get_axes()[2]
    assert len(shrinkage_panel.get_xticklabels()) == 8
    # The family strip rows the socioeconomic covariate above the individual one.
    strip = fig.get_axes()[0]
    labels = [t.get_text() for t in strip.get_yticklabels()]
    assert any("Household income" in label for label in labels)
    assert any("Sex assigned at birth" in label for label in labels)
