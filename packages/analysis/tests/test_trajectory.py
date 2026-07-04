"""Tests for the trajectory embedding, directional test, and roughness metrics (synthetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from analysis.trajectory import (
    Embedding,
    directional_test,
    fit_embedding,
    project,
    roughness_metrics,
)


def _pooled() -> tuple[pd.DataFrame, pd.Series]:
    """Return a synthetic four-class pooled matrix with well-separated classes."""
    rng = np.random.default_rng(0)
    centres = {0: [0, 0, 0, 0], 1: [6, 0, 0, 0], 2: [0, 6, 0, 0], 3: [6, 6, 0, 0]}
    blocks, labels = [], []
    for cls, centre in centres.items():
        blocks.append(rng.normal(centre, 1.0, size=(60, 4)))
        labels += [cls] * 60
    index = pd.RangeIndex(len(labels))
    matrix = pd.DataFrame(np.vstack(blocks), columns=[f"f{i}" for i in range(4)], index=index)
    return matrix, pd.Series(labels, index=index)


def test_fit_embedding_and_project_shapes() -> None:
    matrix, labels = _pooled()
    embedding = fit_embedding(matrix, labels)
    assert isinstance(embedding, Embedding)
    assert embedding.n_components == 3  # K - 1 for four classes
    assert embedding.explained_variance_ratio.shape == (3,)
    centroids = matrix.groupby(labels.to_numpy()).mean()
    coords = project(embedding, centroids)
    assert coords.shape == (4, 3)


def test_directional_test_detects_directed_drift() -> None:
    rng = np.random.default_rng(1)
    n_strata, n_features = 10, 20
    axis = rng.normal(size=n_features)
    axis /= np.linalg.norm(axis)
    drift = np.outer(np.arange(n_strata), 2.0 * axis) + rng.normal(
        scale=0.2, size=(n_strata, n_features)
    )
    result = directional_test(drift, seed=0, n_shuffle=2000)
    assert result["significant"]
    assert result["p"] < 0.05
    assert result["net"] > result["null95"]


def test_directional_test_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    trajectory = rng.normal(size=(8, 12))
    first = directional_test(trajectory, seed=7, n_shuffle=500)
    second = directional_test(trajectory, seed=7, n_shuffle=500)
    assert first == second
    assert 0.0 < first["p"] <= 1.0


def test_roughness_metrics_matches_hand_computation() -> None:
    # Two steps of length 5 in standardised space, class of 100 in every stratum, unit within-
    # class spread. Expected step under sampling is sqrt(2 * (1/100 + 1/100)) = 0.2.
    trajectory = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    sizes = np.array([100.0, 100.0, 100.0])
    within_sd = np.array([1.0, 1.0])
    metrics = roughness_metrics(trajectory, sizes, within_sd)
    assert metrics["step"] == 5.0
    assert abs(metrics["sampling_noise"] - 0.2) < 1e-9
    assert abs(metrics["snr"] - 25.0) < 1e-9
