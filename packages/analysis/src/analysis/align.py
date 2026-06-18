"""Hungarian alignment of one set of classes to another by profile similarity.

StepMix assigns class ids arbitrarily on every fit, so a recovered class has no fixed
meaning. To compare two solutions (our fit against the published classes, or a stratum
against the reference) we align them by matching their profiles with the Hungarian
algorithm on a cost matrix of profile distances (Kuhn 1955), as the plan specifies for
cross-stratum and cross-cohort comparison (plan section 6, deviations).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass
class Alignment:
    """The result of aligning a source set of classes to a target set.

    Attributes
    ----------
    mapping : dict
        Each source class id mapped to the target label it aligns to.
    cost : pandas.DataFrame
        The source-by-target cost matrix the assignment minimised.
    correlations : dict
        Each source class id mapped to the Pearson correlation of its profile with its
        assigned target profile.
    total_cost : float
        The minimised total assignment cost.
    """

    mapping: dict[object, object]
    cost: pd.DataFrame
    correlations: dict[object, float]
    total_cost: float


def _pairwise_cost(source: pd.DataFrame, target: pd.DataFrame, metric: str) -> np.ndarray:
    """Return the source-by-target cost matrix under a distance metric."""
    s = source.to_numpy(dtype=float)
    t = target.to_numpy(dtype=float)
    cost = np.empty((len(s), len(t)))
    with np.errstate(invalid="ignore", divide="ignore"):
        for i, srow in enumerate(s):
            for j, trow in enumerate(t):
                if metric == "correlation":
                    # A constant profile has undefined correlation; treat it as no
                    # association (worst cost) rather than letting the NaN propagate.
                    r = np.corrcoef(srow, trow)[0, 1]
                    cost[i, j] = 1.0 - (0.0 if np.isnan(r) else r)
                elif metric == "euclidean":
                    cost[i, j] = float(np.linalg.norm(srow - trow))
                else:
                    raise ValueError(f"unknown metric {metric!r}")
    return cost


def hungarian_align(
    source: pd.DataFrame, target: pd.DataFrame, metric: str = "correlation"
) -> Alignment:
    """Align source classes to target classes by minimising profile distance.

    Parameters
    ----------
    source : pandas.DataFrame
        Source class-by-profile matrix (for example our four classes by seven categories).
    target : pandas.DataFrame
        Target class-by-profile matrix with the same columns (for example the published
        named classes).
    metric : str, default "correlation"
        ``"correlation"`` (cost is one minus Pearson r) or ``"euclidean"``.

    Returns
    -------
    Alignment
        The mapping, the cost matrix, the per-pair correlations, and the total cost.

    Raises
    ------
    ValueError
        When the source and target columns differ.
    """
    if list(source.columns) != list(target.columns):
        raise ValueError("source and target must share the same profile columns")
    cost = _pairwise_cost(source, target, metric)
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping: dict[object, object] = {}
    correlations: dict[object, float] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for i, j in zip(row_ind, col_ind, strict=True):
            src_label = source.index[i]
            tgt_label = target.index[j]
            mapping[src_label] = tgt_label
            r = np.corrcoef(source.iloc[i].to_numpy(float), target.iloc[j].to_numpy(float))[0, 1]
            correlations[src_label] = float(0.0 if np.isnan(r) else r)
    cost_df = pd.DataFrame(cost, index=source.index, columns=target.index)
    return Alignment(
        mapping=mapping,
        cost=cost_df,
        correlations=correlations,
        total_cost=float(cost[row_ind, col_ind].sum()),
    )


def greedy_overlap_align(source: pd.Series, target: pd.Series) -> dict[int, int]:
    """Align source class labels to target labels by greedy proband overlap.

    This reproduces the released ``match_class_labels`` rule, which compares two
    clusterings of the *same* probands (across seeds or subsamples). Each source class
    claims the target class it overlaps most, where overlap is the proportion of the
    *source* class that falls in the target class; a collision is resolved in favour of the
    larger overlap, and any classes left unclaimed are paired in order. The rule needs a
    shared index, so it applies to same-sample comparison only; for disjoint strata and
    across cohorts the plan aligns by profile similarity with :func:`hungarian_align`
    instead (plan section 6, deviations).

    Parameters
    ----------
    source : pandas.Series
        Class label per proband for the solution being aligned.
    target : pandas.Series
        Class label per proband for the reference solution, on an overlapping index.

    Returns
    -------
    dict of int to int
        Each source class id mapped to the target class id it aligns to.
    """
    shared = source.index.intersection(target.index)
    src = source.loc[shared]
    tgt = target.loc[shared]
    source_classes = sorted(int(c) for c in src.unique())
    target_classes = sorted(int(c) for c in tgt.unique())

    overlap: dict[tuple[int, int], float] = {}
    for s in source_classes:
        s_index = src.index[src == s]
        s_size = len(s_index)
        for t in target_classes:
            shared_count = len(s_index.intersection(tgt.index[tgt == t]))
            overlap[(s, t)] = shared_count / s_size if s_size else 0.0

    mapping: dict[int, int] = {}
    claimed: dict[int, int] = {}  # target class id mapped to the source class holding it
    for s in source_classes:
        best = max(target_classes, key=lambda t, s=s: overlap[(s, t)])
        holder = claimed.get(best)
        if holder is None:
            mapping[s] = best
            claimed[best] = s
        elif overlap[(holder, best)] < overlap[(s, best)]:
            del mapping[holder]
            mapping[s] = best
            claimed[best] = s

    leftover_source = [s for s in source_classes if s not in mapping]
    leftover_target = [t for t in target_classes if t not in claimed]
    for s, t in zip(leftover_source, leftover_target, strict=False):
        mapping[s] = t
    return mapping
