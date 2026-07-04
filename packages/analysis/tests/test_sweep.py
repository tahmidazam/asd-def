"""Tests for the sweep conductor: scheme parsing and the shared decision table (synthetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analysis.cohort import CohortMatrix
from analysis.drift import ALIGNMENTS, DISTANCES, ReferenceModel, StratumSummary
from analysis.features import Typing
from analysis.localise import HardBins, KernelWindows
from analysis.strata import MaxEqualBins, QuantileBins
from analysis.sweep import (
    LocaleSummary,
    fit_local_summaries,
    fit_null_summaries,
    parse_scheme,
    summarise_local_worker,
    sweep_decision,
)


def _reference() -> ReferenceModel:
    centroids = pd.DataFrame(
        {"f0": [0.0, 10.0, 0.0, 10.0], "f1": [0.0, 0.0, 10.0, 10.0]},
        index=pd.Index([0, 1, 2, 3], name="class"),
    )
    dispersions = pd.DataFrame(1.0, index=centroids.index, columns=centroids.columns)
    return ReferenceModel(
        centroids=centroids,
        dispersions=dispersions,
        pooled_sd=pd.Series({"f0": 2.0, "f1": 4.0}),
        precision=np.eye(2),
        labels=pd.Series([0, 1, 2, 3], index=["a", "b", "c", "d"]),
    )


def _aligned_summary(centroids: pd.DataFrame) -> StratumSummary:
    dispersions = pd.DataFrame(1.0, index=centroids.index, columns=centroids.columns)
    contingency = pd.DataFrame(np.diag([100] * 4), index=[0, 1, 2, 3], columns=[0, 1, 2, 3])
    return StratumSummary(
        centroids=centroids, dispersions=dispersions, contingency=contingency, n=400
    )


def test_parse_scheme_grammar() -> None:
    max_equal = parse_scheme("hardbins:max-equal:1000")
    assert isinstance(max_equal, HardBins)
    assert max_equal.policy == MaxEqualBins(min_bin_size=1000)
    quantile = parse_scheme("hardbins:quantile:4")
    assert isinstance(quantile, HardBins)
    assert quantile.policy == QuantileBins(q=4)
    kernel = parse_scheme("kernel:1.5:20")
    assert isinstance(kernel, KernelWindows)
    assert kernel.bandwidth == 1.5
    assert kernel.grid == 20


@pytest.mark.parametrize(
    "bad", ["hardbins:max-equal", "kernel:2", "spline:1:2", "hardbins:bogus:1"]
)
def test_parse_scheme_rejects_bad_specs(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_scheme(bad)


def test_sweep_decision_shape_and_flags() -> None:
    reference = _reference()
    ref_centroids = reference.centroids
    observed = [
        LocaleSummary("Q1", 5.0, _aligned_summary(ref_centroids)),
        LocaleSummary("Q2", 12.0, _aligned_summary(ref_centroids + 0.1)),
    ]
    null = [(0, _aligned_summary(ref_centroids)), (1, _aligned_summary(ref_centroids))]

    decision = sweep_decision(
        "hardbins", observed, null, reference, ALIGNMENTS["membership"], DISTANCES["euclidean"]
    )

    assert len(decision) == len(observed) * 4  # one row per local fit per reference class
    assert {"scheme", "position", "stratum", "ref_class", "p_value"} <= set(decision.columns)
    assert decision["fdr_significant"].dtype == bool
    assert decision["reorganised"].dtype == bool
    assert set(decision["position"]) == {5.0, 12.0}
    assert (decision["scheme"] == "hardbins").all()


def _cohort(n: int = 160, seed: int = 0) -> tuple[CohortMatrix, Typing]:
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1, 2, 3], n // 4)
    features = pd.DataFrame(
        {
            "c1": rng.normal(0, 1, n) + (labels == 1) * 5,
            "c2": rng.normal(0, 1, n) + (labels == 3) * 5,
            "b1": ((labels == 0) | (rng.random(n) < 0.05)).astype(int),
            "b2": ((labels == 2) | (rng.random(n) < 0.05)).astype(int),
            "cat1": np.clip(labels, 0, 2),
        },
        index=pd.Index([f"p{i}" for i in range(n)], name="proband"),
    )
    covariates = pd.DataFrame(
        {"sex": rng.integers(0, 2, n), "age_at_eval_years": rng.integers(4, 18, n)},
        index=features.index,
    )
    return CohortMatrix(features, covariates, "synthetic", "v0"), Typing(
        continuous=["c1", "c2"], binary=["b1", "b2"], categorical=["cat1"]
    )


def test_end_to_end_hardbins_sweep_runs() -> None:
    matrix, typing = _cohort()
    rng = np.random.default_rng(1)
    axis = pd.Series(rng.uniform(2, 18, len(matrix.features)), index=matrix.features.index)
    # A pooled reference from the whole cohort, then measure the two bins against a size-one null.
    from analysis.drift import build_reference
    from analysis.model import fit_gfmm

    pooled = fit_gfmm(matrix, typing, n_init=2, random_state=0, verbose=0)
    reference = build_reference(pooled.measurement_data, pooled.labels)
    observed = fit_local_summaries(
        matrix, typing, axis, HardBins(QuantileBins(q=2)), pooled.labels, n_init=2, seed=0
    )
    null = fit_null_summaries(
        matrix,
        typing,
        axis,
        HardBins(QuantileBins(q=2)),
        pooled.labels,
        n_init=2,
        n_permutations=1,
        seed=0,
    )
    decision = sweep_decision(
        "hardbins:quantile:2",
        observed,
        null,
        reference,
        ALIGNMENTS["membership"],
        DISTANCES["euclidean"],
    )
    assert len(decision) == 2 * pooled.labels.nunique()
    assert decision["p_value"].between(0, 1).all()


def test_worker_returns_none_on_degenerate_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    # A singular covariate GLM raises LinAlgError; the worker drops it rather than propagating,
    # so one bad refit cannot abort an hours-long null.
    import analysis.model as model_mod

    def boom(*args: object, **kwargs: object) -> None:
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(model_mod, "fit_gfmm", boom)
    matrix, typing = _cohort()
    ref_labels = pd.Series(0, index=matrix.features.index)
    out = summarise_local_worker(
        matrix.features, matrix.covariates, typing, "s", "v", None, ref_labels, n_init=1, seed=0
    )
    assert out is None
