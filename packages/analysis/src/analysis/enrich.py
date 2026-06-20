"""Per-class feature enrichment and the seven-category signature.

Reproduces the released enrichment pipeline (plan section 6, step 7). Each feature is
tested one class against the rest, in both directions: a binomial test for binary features
(those with two observed values) and a Welch t-test for the rest. The :math:`p`-values are
Benjamini-Hochberg corrected within each class and direction; a corrected :math:`p` below
:math:`0.05` marks a feature as enriched or depleted in that class. The 24 reverse-coded
SCQ items have their direction flipped, and the features are summarised into the seven
literature-defined categories as the signed proportion enriched minus depleted, which is
the class signature used to align to the published classes (plan section 6a).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_ind
from statsmodels.stats.multitest import multipletests

from analysis.features import REVERSE_CODED_SCQ

# The seven categories Litman et al. summarise, in their reporting order. The category map
# also defines somatic, other problems, and thought problems, which the released summary
# drops; we drop them too.
SEVEN_CATEGORIES: tuple[str, ...] = (
    "anxiety/mood",
    "attention",
    "disruptive behavior",
    "self-injury",
    "social/communication",
    "restricted/repetitive",
    "developmental",
)

_ALPHA = 0.05


def cohens_d(group: pd.Series, reference: pd.Series) -> float:
    """Return Cohen's :math:`d` of a group against a reference, pooling their variances."""
    mean_diff = float(np.mean(group) - np.mean(reference))
    pooled_sd = float(np.sqrt((np.std(group, ddof=1) ** 2 + np.std(reference, ddof=1) ** 2) / 2))
    return mean_diff / pooled_sd if pooled_sd > 0 else 0.0


def feature_enrichment(data: pd.DataFrame, labels: pd.Series, n_classes: int = 4) -> pd.DataFrame:
    """Test each feature for one-versus-rest enrichment in every class.

    Parameters
    ----------
    data : pandas.DataFrame
        The proband-by-feature measurement matrix used for the fit.
    labels : pandas.Series
        The hard class label per proband, on the same index.
    n_classes : int, default 4
        Number of classes.

    Returns
    -------
    pandas.DataFrame
        One row per feature with, per class ``c``: ``class{c}_dir`` (+1 enriched, -1
        depleted, 0 neither, after correction) and ``class{c}_effect`` (fold enrichment for
        binary features, Cohen's :math:`d` otherwise), plus an ``is_binary`` flag.
    """
    labels = labels.reindex(data.index)
    groups = {c: data[labels == c] for c in range(n_classes)}
    # A class with no probands (a collapsed fit, or a projection that lands no one in a
    # class, which is plausible on a small cohort) has no data to test. Such a class is
    # marked enriched-in-nothing rather than crashing on a zero-size test.
    empty = {c: len(groups[c]) == 0 for c in range(n_classes)}
    greater: dict[int, dict[str, float]] = {c: {} for c in range(n_classes)}
    lesser: dict[int, dict[str, float]] = {c: {} for c in range(n_classes)}
    effect: dict[int, dict[str, float]] = {c: {} for c in range(n_classes)}
    is_binary: dict[str, bool] = {}

    for feature in data.columns:
        column = data[feature]
        binary = column.nunique(dropna=True) == 2
        is_binary[feature] = binary
        if binary:
            total = len(column)
            background = int(column.sum())
            p_background = background / total
            for c in range(n_classes):
                if empty[c]:
                    greater[c][feature] = lesser[c][feature] = effect[c][feature] = np.nan
                    continue
                values = groups[c][feature]
                n = len(values)
                successes = int(values.sum())
                greater[c][feature] = binomtest(
                    successes, n=n, p=p_background, alternative="greater"
                ).pvalue
                lesser[c][feature] = binomtest(
                    successes, n=n, p=p_background, alternative="less"
                ).pvalue
                effect[c][feature] = (successes / n) / p_background if p_background > 0 else np.nan
        else:
            for c in range(n_classes):
                if empty[c]:
                    greater[c][feature] = lesser[c][feature] = effect[c][feature] = np.nan
                    continue
                rest = pd.concat([groups[o][feature] for o in range(n_classes) if o != c])
                greater[c][feature] = ttest_ind(
                    groups[c][feature], rest, equal_var=False, alternative="greater"
                ).pvalue
                lesser[c][feature] = ttest_ind(
                    groups[c][feature], rest, equal_var=False, alternative="less"
                ).pvalue
                effect[c][feature] = cohens_d(groups[c][feature], column)

    records: dict[str, dict[str, float]] = {
        f: {"is_binary": float(is_binary[f])} for f in data.columns
    }
    for c in range(n_classes):
        enriched = _significant(greater[c])
        depleted = _significant(lesser[c])
        for f in data.columns:
            direction = -1 if depleted.get(f) else (1 if enriched.get(f) else 0)
            records[f][f"class{c}_dir"] = float(direction)
            records[f][f"class{c}_effect"] = float(effect[c][f])
    out = pd.DataFrame.from_records([{"feature": f, **records[f]} for f in data.columns])
    return out.set_index("feature")


def _significant(pvalues: dict[str, float]) -> dict[str, bool]:
    r"""Correct a class-direction's :math:`p`-values and flag those below :math:`\alpha`."""
    features = [f for f, p in pvalues.items() if not np.isnan(p)]
    if not features:
        return {}
    corrected = multipletests([pvalues[f] for f in features], method="fdr_bh")[1]
    return {f: bool(p < _ALPHA) for f, p in zip(features, corrected, strict=True)}


def _safe_pearson(a: np.ndarray, b: np.ndarray, min_std: float = 1e-9) -> float | None:
    """Return the Pearson correlation, or ``None`` when either vector is near-constant."""
    if float(np.std(a)) < min_std or float(np.std(b)) < min_std:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        r = float(np.corrcoef(a, b)[0, 1])
    return None if np.isnan(r) else r


def profile_correlation(
    signature_a: pd.DataFrame, signature_b: pd.DataFrame
) -> tuple[float | None, dict[str, float | None]]:
    """Correlate two seven-category class signatures, overall and per category.

    This is the comparison Litman et al. use to declare reproduction and replication: the
    profile is the signed net-proportion-enriched vector per category, and the correlation
    is taken over the class-by-category matrix (plan section 6a). The two signatures must
    already have their classes aligned to a common ordering (for example by
    :func:`analysis.align.greedy_overlap_align` for same-sample comparison or
    :func:`analysis.align.hungarian_align` across cohorts).

    Parameters
    ----------
    signature_a, signature_b : pandas.DataFrame
        Class-by-category signed signatures on the same class index and the seven
        categories in ``SEVEN_CATEGORIES`` order.

    Returns
    -------
    tuple
        The overall Pearson correlation over the flattened class-by-category matrix, and a
        mapping from each category to its Pearson correlation across classes. A near-constant
        profile (overall or within a category) has an undefined correlation, returned as
        ``None``.

    Raises
    ------
    ValueError
        When the two signatures do not share their class index.
    """
    if list(signature_a.index) != list(signature_b.index):
        raise ValueError("signatures must share the same class index (align them first)")
    a = signature_a.loc[:, list(SEVEN_CATEGORIES)]
    b = signature_b.loc[:, list(SEVEN_CATEGORIES)]
    overall = _safe_pearson(a.to_numpy(float).ravel(), b.to_numpy(float).ravel())
    per_category = {
        cat: _safe_pearson(a[cat].to_numpy(float), b[cat].to_numpy(float))
        for cat in SEVEN_CATEGORIES
    }
    return overall, per_category


def bootstrap_overall_correlation(
    measurement: pd.DataFrame,
    labels: pd.Series,
    target: pd.DataFrame,
    category_map: dict[str, str],
    *,
    n_boot: int,
    seed: int,
    n_classes: int = 4,
    level: float = 0.95,
    reverse_coded: tuple[str, ...] = REVERSE_CODED_SCQ,
    keep: set[str] | None = None,
) -> dict[str, float | int]:
    r"""Bootstrap the overall profile correlation by resampling probands.

    The class labels are held fixed and the probands are resampled with replacement; for
    each resample the seven-category signature is recomputed and correlated against a fixed
    target on the same class index. The spread of those correlations is the sampling
    uncertainty in the reproduction or replication statistic from the finite cohort, with the
    model fit held fixed. It does not capture uncertainty from refitting the model (the
    stability stage does that) nor, for the reproduction, the resolution of the figure-read
    target.

    Parameters
    ----------
    measurement : pandas.DataFrame
        The proband-by-feature matrix.
    labels : pandas.Series
        The hard class label per proband, positionally aligned with ``measurement``.
    target : pandas.DataFrame
        The fixed signature to correlate against, on the same class index the recomputed
        signature carries (class ids ``0`` to ``n_classes - 1``).
    category_map : dict of str to str
        Feature-to-category map for the signatures.
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        Seed for the resampling, so the interval is reproducible.
    n_classes : int, default 4
        Number of classes.
    level : float, default 0.95
        Central probability of the reported percentile interval.
    reverse_coded : tuple of str, optional
        SCQ items whose enrichment direction is flipped before the signature.
    keep : set of str, optional
        The contributory feature set, applied to every resample.

    Returns
    -------
    dict
        ``ci_low`` and ``ci_high`` (the percentile interval at ``level``), ``median``, the
        ``level``, and ``n_valid`` (the resamples that yielded a defined correlation; a
        resample that empties or flattens a class is dropped, as in the permutation null).
    """
    rng = np.random.default_rng(seed)
    measurement = measurement.reset_index(drop=True)
    label_values = labels.to_numpy()
    target = target.loc[:, list(SEVEN_CATEGORIES)]
    n = len(measurement)
    correlations: list[float] = []
    for _ in range(n_boot):
        take = rng.integers(0, n, size=n)
        boot_measurement = measurement.iloc[take].reset_index(drop=True)
        boot_labels = pd.Series(label_values[take], name="class")
        enrichment = feature_enrichment(boot_measurement, boot_labels, n_classes=n_classes)
        signature = category_signature(
            enrichment, category_map, n_classes=n_classes, reverse_coded=reverse_coded, keep=keep
        )
        overall, _ = profile_correlation(signature, target)
        if overall is not None:
            correlations.append(overall)

    if not correlations:
        return {
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "median": float("nan"),
            "level": level,
            "n_valid": 0,
        }
    tail = (1.0 - level) / 2.0
    values = np.asarray(correlations)
    return {
        "ci_low": float(np.quantile(values, tail)),
        "ci_high": float(np.quantile(values, 1.0 - tail)),
        "median": float(np.median(values)),
        "level": level,
        "n_valid": len(correlations),
    }


def contributory_features(enrichment: pd.DataFrame, n_classes: int = 4) -> list[str]:
    """Return the features that contribute to the class signatures.

    Reproduces the released non-contributory feature exclusion (plan section 6, step 7). A
    feature is dropped when it is significantly enriched or depleted in no class, or when its
    effect size stays below the magnitude threshold in every class: a fold enrichment below
    :math:`1.5` for binary features, an absolute Cohen's :math:`d` below :math:`0.2` for the
    rest. The surviving features are the universe over which the seven-category proportions
    are computed, so the same set (taken from the reference solution) is applied to both
    sides of a comparison.

    Parameters
    ----------
    enrichment : pandas.DataFrame
        The per-feature enrichment from :func:`feature_enrichment`.
    n_classes : int, default 4
        Number of classes.

    Returns
    -------
    list of str
        The contributory feature names, in the enrichment frame's order.
    """
    directions = enrichment[[f"class{c}_dir" for c in range(n_classes)]].abs()
    effects = enrichment[[f"class{c}_effect" for c in range(n_classes)]].abs()
    is_binary = enrichment["is_binary"] > 0
    significant = (directions > 0).any(axis=1)
    binary_strong = (effects >= 1.5).any(axis=1)
    continuous_strong = (effects >= 0.2).any(axis=1)
    strong = binary_strong.where(is_binary, continuous_strong)
    keep = significant & strong
    return enrichment.index[keep].tolist()


def category_signature(
    enrichment: pd.DataFrame,
    category_map: dict[str, str],
    n_classes: int = 4,
    reverse_coded: tuple[str, ...] = REVERSE_CODED_SCQ,
    keep: set[str] | None = None,
) -> pd.DataFrame:
    """Summarise feature enrichment into the signed seven-category class signature.

    For each class and category the signature is the proportion of the category's features
    enriched minus the proportion depleted, after flipping the reverse-coded SCQ items.

    Parameters
    ----------
    enrichment : pandas.DataFrame
        The per-feature enrichment from :func:`feature_enrichment`.
    category_map : dict of str to str
        Feature-to-category map.
    n_classes : int, default 4
        Number of classes.
    reverse_coded : tuple of str, optional
        SCQ items whose enriched and depleted directions are swapped.
    keep : set of str, optional
        When given, only these features contribute to the proportions (the contributory set
        from :func:`contributory_features`). When omitted, every feature contributes, which
        is the right behaviour for the reference signature compared to the published figure.

    Returns
    -------
    pandas.DataFrame
        Class-by-category signed signature, indexed by class id, columns the seven
        categories in reporting order.
    """
    directions = enrichment[[f"class{c}_dir" for c in range(n_classes)]].copy()
    directions.columns = pd.Index(range(n_classes))
    if keep is not None:
        directions = directions.loc[[f for f in directions.index if f in keep]]
    for feature in reverse_coded:
        if feature in directions.index:
            directions.loc[feature] = -directions.loc[feature]

    category = directions.index.map(category_map)
    signature = pd.DataFrame(index=range(n_classes), columns=SEVEN_CATEGORIES, dtype=float)
    for cat in SEVEN_CATEGORIES:
        members = directions[category == cat]
        if members.empty:
            signature[cat] = 0.0
            continue
        n = len(members)
        for c in range(n_classes):
            enriched = (members[c] > 0).sum()
            depleted = (members[c] < 0).sum()
            signature.loc[c, cat] = (enriched - depleted) / n
    signature.index.name = "class"
    return signature
