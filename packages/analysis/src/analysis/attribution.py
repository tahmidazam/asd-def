r"""Attribute a class's movement between two fits to features and to probands (archived).

Archived. This is the refit-era attribution: it interprets the drift of a mixture re-estimated
within a stratum. The category attribution ($H_0^F$) is now read from the single cached fit by
the block-attribution engine (:mod:`analysis.blocks`, the additive category decomposition), so
this module is no longer $H_0^F$'s evidence. It is kept because it renders the membership-churn
and mover-versus-stayer figures on the
:doc:`refit pilot </packages/analysis/archive/tracking-the-classes-across-strata>` page, which
the single-fit engine cannot reproduce (a frozen fit relabels no proband).

The drift stage (:mod:`analysis.drift`) measures how far a reference class moves when the
mixture is re-estimated within a stratum, as one distance per class. This module opens that
distance up. It asks which features carry the shift, and which probands changed class, so a
movement reads as "these features, these people" rather than a single number.

Two families, both cheap readouts over the stored fits (no re-fitting):

- Centroid-shift decomposition splits a class's distance into signed per-feature
  contributions that sum back to it. For the Mahalanobis distance the split is the term-by-term
  expansion of the quadratic form $\Delta\mu^\top P \Delta\mu = \sum_i \Delta\mu_i (P\Delta\mu)_i$,
  so a coordinated shift across correlated features is charged together, matching the distance
  the drift stage reports. The diagonal split $(\Delta\mu_i / \sigma_i)^2$ is the
  covariance-blind cross-check.
- Mover and stayer attribution labels each shared proband a class member that stayed or left
  between the two fits, then contrasts the two groups over a feature frame (the clustered
  features, the held-out SPARK variables, or, later, the genetic scores). The movement is then
  explained by what marks the probands the class shed.

The unit both families read is a :class:`Comparison`: two labellings on the same proband index
plus the fit summaries and their alignment. It is agnostic to how the second labelling was
produced (a hard stratum bin, a kernel-weighted focal point, a partition-tree node), so the
same attribution runs on any of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy import stats

from analysis.drift import (
    ClassAlignment,
    ReferenceModel,
    StratumSummary,
    benjamini_hochberg,
)


@dataclass(frozen=True)
class Comparison:
    """Two labellings on a shared proband index, with their summaries and alignment.

    Attributes
    ----------
    reference : analysis.drift.ReferenceModel
        The pooled reference: centroids, dispersions, pooled spread, and the within-class
        precision.
    stratum : analysis.drift.StratumSummary
        The second fit's method-independent summary (centroids, dispersions, contingency).
    ref_labels : pandas.Series
        Reference class per proband, over the shared index.
    fit_labels : pandas.Series
        Second-fit class per proband, over the shared index, in the fit's own class ids.
    alignment : analysis.drift.ClassAlignment
        The fit-to-reference class mapping and its confidence.
    """

    reference: ReferenceModel
    stratum: StratumSummary
    ref_labels: pd.Series
    fit_labels: pd.Series
    alignment: ClassAlignment

    @property
    def shared_index(self) -> pd.Index:
        """Probands carrying both labellings."""
        return self.ref_labels.index.intersection(self.fit_labels.index)

    def movements(self) -> list[Movement]:
        """One :class:`Movement` per reference class the alignment mapped, in class order."""
        inverse = {int(ref): int(fit) for fit, ref in self.alignment.mapping.items()}
        return [
            Movement(self, ref_class=ref, fit_class=fit) for ref, fit in sorted(inverse.items())
        ]


@dataclass(frozen=True)
class Movement:
    """One aligned class pair: how reference ``ref_class`` re-expressed as ``fit_class``.

    Attributes
    ----------
    comparison : Comparison
        The two-fit pairing this movement is drawn from.
    ref_class : int
        The fixed reference class being tracked.
    fit_class : int
        Its aligned partner in the second fit.
    """

    comparison: Comparison
    ref_class: int
    fit_class: int


def _delta(movement: Movement) -> pd.Series:
    """Per-feature centroid difference (stratum minus reference) over the reference columns.

    Mirrors :class:`analysis.drift.Mahalanobis`: the stratum centroid is reindexed to the
    reference feature order and a feature absent from the stratum contributes no shift.
    """
    ref = movement.comparison.reference
    cols = ref.centroids.columns
    src = movement.comparison.stratum.centroids.reindex(columns=cols).loc[movement.fit_class]
    tgt = ref.centroids.loc[movement.ref_class]
    return src.fillna(tgt) - tgt


def signed_shift(movement: Movement) -> pd.Series:
    """Per-feature centroid shift in pooled-SD units, signed (stratum minus reference).

    The direction of movement per feature, so a decomposition magnitude can be read together
    with whether the stratum class sits above or below the reference class on that feature.
    """
    ref = movement.comparison.reference
    delta = _delta(movement)
    sd = ref.pooled_sd.reindex(delta.index).replace(0.0, np.nan)
    return (delta / sd).rename("signed_shift")


@runtime_checkable
class DecompositionMethod(Protocol):
    """Split a class's squared distance into signed per-feature contributions."""

    name: str

    def contributions(self, movement: Movement) -> pd.Series:
        """Signed per-feature contributions that sum to the class's squared distance."""
        ...


@dataclass
class MahalanobisContribution:
    """Term-by-term split of the Mahalanobis distance: ``c_i = delta_i (P delta)_i``.

    The contributions sum to the squared Mahalanobis distance the drift stage reports, so a
    class's movement has an additive, covariance-aware feature breakdown. A contribution can be
    negative when a feature's shift offsets a correlated block, which the diagonal split cannot
    show. The default, matching the default drift distance.
    """

    name: str = "mahalanobis"

    def contributions(self, movement: Movement) -> pd.Series:
        """Return the precision-weighted per-feature contributions to the squared distance."""
        ref = movement.comparison.reference
        delta = _delta(movement).to_numpy()
        contrib = delta * (ref.precision @ delta)
        return pd.Series(contrib, index=ref.centroids.columns, name="contribution")


@dataclass
class StandardisedContribution:
    """Diagonal split ``c_i = (delta_i / sigma_i)**2``: non-negative and covariance-blind.

    Sums to the total squared standardised shift, the cross-check on the Mahalanobis split: a
    feature that ranks high here but low under Mahalanobis is one whose shift is shared with a
    correlated block rather than its own.
    """

    name: str = "standardised"

    def contributions(self, movement: Movement) -> pd.Series:
        """Return the squared standardised per-feature shift."""
        ref = movement.comparison.reference
        delta = _delta(movement)
        sd = ref.pooled_sd.reindex(delta.index).replace(0.0, np.nan)
        return ((delta / sd) ** 2).fillna(0.0).rename("contribution")


def category_of(feature: object, category_map: Mapping[str, object]) -> str:
    """Return a feature's category, or ``"unmapped"`` for an absent or blank entry.

    The author map leaves a few CBCL composites (for example ``total_problems_t_score``) with a
    blank category, which loads as ``NaN``, and does not list every clustered feature. Both
    resolve to ``"unmapped"`` here so the category is always a clean string, and so a blank
    category never silently drops a feature's contribution from the totals.
    """
    cat = category_map.get(str(feature))
    return "unmapped" if cat is None or pd.isna(cat) else str(cat)


def category_totals(contributions: pd.Series, category_map: Mapping[str, object]) -> pd.Series:
    """Aggregate per-feature contributions into the literature categories.

    Sums the signed contributions within each category (plan section 6, step 7), so a class's
    movement reads at the level of the categories, not only the individual features. Every
    feature is kept (blanks and absentees under ``"unmapped"``), so the totals sum to the same
    squared distance the per-feature contributions do.
    """
    cats = pd.Series({f: category_of(f, category_map) for f in contributions.index}, dtype=object)
    totals = contributions.groupby(cats.to_numpy()).sum()
    return totals.sort_values(ascending=False)


def _membership(movement: Movement) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return the reference-member, stratum-member, and stayer masks over the shared index.

    A proband is a reference member of the class when the reference assigns it there, a stratum
    member when its second-fit class maps to the class through the alignment, and a stayer when
    both hold. From these three masks the leavers (reference member, not stratum member) and the
    joiners (stratum member, not reference member) follow.
    """
    comp = movement.comparison
    idx = comp.shared_index
    ref = comp.ref_labels.loc[idx]
    mapping = comp.alignment.mapping
    mapped = comp.fit_labels.loc[idx].map(lambda c: mapping.get(int(c), -1))
    is_ref = ref == movement.ref_class
    is_fit = mapped == movement.ref_class
    return is_ref, is_fit, is_ref & is_fit


def membership_counts(movement: Movement) -> dict[str, int]:
    """Count the stayers, leavers, and joiners for one class between the two fits.

    Returns
    -------
    dict
        ``n_stayers`` (in the class under both fits), ``n_leavers`` (a reference member the
        second fit dropped), and ``n_joiners`` (a second-fit member the reference did not
        assign to the class). Leavers plus joiners over their union is the class churn, one
        minus the Jaccard overlap the alignment reports.
    """
    is_ref, is_fit, stayer = _membership(movement)
    return {
        "n_stayers": int(stayer.sum()),
        "n_leavers": int((is_ref & ~is_fit).sum()),
        "n_joiners": int((is_fit & ~is_ref).sum()),
    }


def movers(movement: Movement, kind: str = "either") -> pd.Series:
    """Mark the probands that changed class membership between the two fits.

    A class can move in two ways: it can shed members (leavers, reference members the second
    fit drops) and it can absorb members (joiners, second-fit members the reference did not
    assign there). A class that keeps every member but pulls in new ones still drifts, so a
    leaver-only view misses it; ``kind="either"`` (the default) captures both.

    Parameters
    ----------
    movement : Movement
        The aligned class pair to score.
    kind : str, default "either"
        ``"either"`` marks leavers and joiners against the stayers over the union of the two
        memberships (the class churn). ``"leavers"`` restricts to the reference members and
        marks those the second fit dropped. ``"joiners"`` restricts to the second-fit members
        and marks those the reference did not assign to the class.

    Returns
    -------
    pandas.Series
        Boolean ``moved`` over the relevant probands: the complement are the stayers the
        contrast is run against.
    """
    idx = movement.comparison.shared_index
    is_ref, is_fit, stayer = _membership(movement)
    if kind == "leavers":
        subset = is_ref
        moved = is_ref & ~is_fit
    elif kind == "joiners":
        subset = is_fit
        moved = is_fit & ~is_ref
    elif kind == "either":
        subset = is_ref | is_fit
        moved = ~stayer
    else:
        raise ValueError(f"kind must be 'either', 'leavers', or 'joiners', not {kind!r}")
    members = idx[subset.to_numpy()]
    return moved.loc[members].rename("moved")


@dataclass
class AttributionResult:
    """Ranked feature attributions distinguishing movers from stayers.

    Attributes
    ----------
    importances : pandas.DataFrame
        One row per feature: the signed ``effect`` (a standardised mean difference or a model
        coefficient), its ``magnitude`` for ranking, and, where the method provides them, a
        ``p_value`` and ``fdr_significant`` flag. Sorted by magnitude, most distinguishing
        first. Empty when the contrast is undegenerate (only movers or only stayers).
    n_movers : int
        Number of probands that left the class.
    n_stayers : int
        Number that stayed.
    method : str
        The contrast method's name.
    """

    importances: pd.DataFrame
    n_movers: int
    n_stayers: int
    method: str


_EMPTY_IMPORTANCES = ["feature", "effect", "magnitude", "p_value", "fdr_significant"]


@runtime_checkable
class ContrastMethod(Protocol):
    """Rank the features that distinguish movers from stayers."""

    name: str

    def contrast(self, moved: pd.Series, features: pd.DataFrame) -> AttributionResult:
        """Return the ranked feature attributions for one class's movers against its stayers."""
        ...


def _split_groups(moved: pd.Series, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the mover and stayer feature blocks, aligned on the mover index."""
    mask = moved.reindex(features.index).to_numpy(dtype=bool)
    values = features.to_numpy(dtype=float)
    return values[mask], values[~mask]


@dataclass
class UnivariateContrast:
    """Per-feature standardised mean difference between movers and stayers.

    Cohen's $d$ per feature (positive when movers score higher), with a Welch $t$-test,
    Benjamini-Hochberg controlled across features. Model-free, fast, and always defined; the
    first read on what marks the movers, feature by feature, before any multivariate model.
    Missing values are handled per feature, so it runs on the held-out variables as it does on
    the clustered features.
    """

    name: str = "univariate"

    def contrast(self, moved: pd.Series, features: pd.DataFrame) -> AttributionResult:
        """Contrast movers and stayers with a per-feature effect size and Welch test."""
        movers_block, stayers_block = _split_groups(moved, features)
        n_movers, n_stayers = int(movers_block.shape[0]), int(stayers_block.shape[0])
        if n_movers < 2 or n_stayers < 2:
            return AttributionResult(
                pd.DataFrame(columns=_EMPTY_IMPORTANCES), n_movers, n_stayers, self.name
            )
        with np.errstate(invalid="ignore", divide="ignore"):
            n1 = np.sum(~np.isnan(movers_block), axis=0)
            n0 = np.sum(~np.isnan(stayers_block), axis=0)
            mean1 = np.nanmean(movers_block, axis=0)
            mean0 = np.nanmean(stayers_block, axis=0)
            var1 = np.nanvar(movers_block, axis=0, ddof=1)
            var0 = np.nanvar(stayers_block, axis=0, ddof=1)
            pooled = np.sqrt(
                ((n1 - 1) * var1 + (n0 - 1) * var0) / np.maximum(n1 + n0 - 2, 1),
            )
            effect = np.divide(mean1 - mean0, pooled, out=np.zeros_like(pooled), where=pooled > 0)
            se = np.sqrt(var1 / np.maximum(n1, 1) + var0 / np.maximum(n0, 1))
            tstat = np.divide(mean1 - mean0, se, out=np.zeros_like(se), where=se > 0)
            df_num = (var1 / np.maximum(n1, 1) + var0 / np.maximum(n0, 1)) ** 2
            df_den = (var1 / np.maximum(n1, 1)) ** 2 / np.maximum(n1 - 1, 1) + (
                var0 / np.maximum(n0, 1)
            ) ** 2 / np.maximum(n0 - 1, 1)
            df = np.divide(df_num, df_den, out=np.ones_like(df_num), where=df_den > 0)
            p_value = 2.0 * stats.t.sf(np.abs(tstat), np.maximum(df, 1.0))
        p_value = np.where(np.isfinite(p_value), p_value, 1.0)
        table = pd.DataFrame(
            {
                "feature": [str(c) for c in features.columns],
                "effect": effect,
                "magnitude": np.abs(effect),
                "p_value": p_value,
                "fdr_significant": benjamini_hochberg(p_value, q=0.05),
            }
        )
        table = table.sort_values("magnitude", ascending=False, ignore_index=True)
        return AttributionResult(table, n_movers, n_stayers, self.name)


@dataclass
class LogisticContrast:
    """L1-regularised logistic regression of mover status on standardised features.

    Signed coefficients rank the features that jointly distinguish movers from stayers, so
    correlated features share credit rather than each scoring the marginal difference. Features
    are standardised and missing values median-imputed within the contrasted probands; a
    constant or fully missing feature is dropped. The L1 penalty keeps the ranking sparse, so a
    short list of features carries the movement.
    """

    name: str = "logistic"
    c: float = 1.0

    def contrast(self, moved: pd.Series, features: pd.DataFrame) -> AttributionResult:
        """Fit the penalised model and return the signed coefficients, ranked by magnitude."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        target = moved.reindex(features.index).to_numpy(dtype=bool)
        n_movers, n_stayers = int(target.sum()), int((~target).sum())
        columns = np.asarray([str(c) for c in features.columns])
        values = features.to_numpy(dtype=float)
        medians = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
        filled = np.where(np.isfinite(values), values, medians)
        keep = np.nanstd(filled, axis=0) > 0
        if n_movers < 2 or n_stayers < 2 or not keep.any():
            return AttributionResult(
                pd.DataFrame(columns=_EMPTY_IMPORTANCES), n_movers, n_stayers, self.name
            )
        design = StandardScaler().fit_transform(filled[:, keep])
        # ``l1_ratio=1`` selects a pure L1 penalty in the current scikit-learn API (the older
        # ``penalty="l1"`` argument is deprecated); liblinear handles it on this small design.
        model = LogisticRegression(solver="liblinear", l1_ratio=1.0, C=self.c, max_iter=1000)
        model.fit(design, target)
        coef = model.coef_.ravel()
        table = pd.DataFrame(
            {
                "feature": columns[keep],
                "effect": coef,
                "magnitude": np.abs(coef),
                "p_value": np.nan,
                "fdr_significant": False,
            }
        )
        table = table.sort_values("magnitude", ascending=False, ignore_index=True)
        return AttributionResult(table, n_movers, n_stayers, self.name)


DECOMPOSITIONS: dict[str, DecompositionMethod] = {
    "mahalanobis": MahalanobisContribution(),
    "standardised": StandardisedContribution(),
}
CONTRASTS: dict[str, ContrastMethod] = {
    "univariate": UnivariateContrast(),
    "logistic": LogisticContrast(),
}
DEFAULT_DECOMPOSITION = "mahalanobis"
DEFAULT_CONTRAST = "univariate"
