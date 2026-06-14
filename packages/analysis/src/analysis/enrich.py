"""Per-class feature enrichment and the seven-category signature.

Reproduces the released enrichment pipeline (plan section 6, step 7). Each feature is
tested one class against the rest, in both directions: a binomial test for binary features
(those with two observed values) and a Welch t-test for the rest. The p-values are
Benjamini-Hochberg corrected within each class and direction; a corrected p below 0.05
marks a feature as enriched or depleted in that class. The 24 reverse-coded SCQ items have
their direction flipped, and the features are summarised into the seven literature-defined
categories as the signed proportion enriched minus depleted, which is the class signature
used to align to the published classes (plan section 6a).
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
    """Return Cohen's d of a group against a reference, pooling their variances."""
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
        binary features, Cohen's d otherwise), plus an ``is_binary`` flag.
    """
    labels = labels.reindex(data.index)
    groups = {c: data[labels == c] for c in range(n_classes)}
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
    """Benjamini-Hochberg correct a class-direction's p-values and flag those below alpha."""
    features = [f for f, p in pvalues.items() if not np.isnan(p)]
    if not features:
        return {}
    corrected = multipletests([pvalues[f] for f in features], method="fdr_bh")[1]
    return {f: bool(p < _ALPHA) for f, p in zip(features, corrected, strict=True)}


def category_signature(
    enrichment: pd.DataFrame,
    category_map: dict[str, str],
    n_classes: int = 4,
    reverse_coded: tuple[str, ...] = REVERSE_CODED_SCQ,
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

    Returns
    -------
    pandas.DataFrame
        Class-by-category signed signature, indexed by class id, columns the seven
        categories in reporting order.
    """
    directions = enrichment[[f"class{c}_dir" for c in range(n_classes)]].copy()
    directions.columns = pd.Index(range(n_classes))
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
