r"""The block-attribution engine: what carries or co-moves with the class drift (plan section 7f).

The drift stage reports each class's movement as one separation-scaled number, and the section-7e
recast (:mod:`analysis.trajectory_local`) resolves it to a per-class, per-feature displacement
trajectory read from the single cached fit. This module asks what that movement is about, and it
does so for two kinds of candidate.

A *block* is a named, proband-indexed set of columns. An *internal* block is a subset of the
reference's own features, so its sub-displacement is a slice of the drift vector the recast already
computed and the per-feature squared magnitudes sum back to the whole-class distance. The seven
author categories (H0F) and the two instrument referents (H0G) are the internal blocks
the earlier stages already read; this module is the generalisation of their
``dict[str, ndarray]`` grain seam. An *external* block is data the model was never fit on (a
held-out phenotype, a polygenic score, a microbiome summary). Its class profile is still defined,
because the frozen responsibilities $r_{ik}$ weight any proband-level quantity, but that profile is
a separate quantity in its own space that cannot be summed into the phenotype drift, only compared
with it.

Two modes follow.

*Co-drift* is defined for any block. The trap it must avoid is the axis: the phenotype drift is a
function of era or age, so any block that varies with the axis correlates with the drift trivially,
and beating a random control ordering is no evidence when the block genuinely depends on the real
axis. The signal that is not a shared-axis artefact is *which classes move*. The phenotype drifts
hardest in the developmental class along age; a uniform axis effect on a block would move all four
classes alike. Co-drift is therefore the correlation between the block's per-class
displacement-magnitude profile and the phenotype's per-class profile, read at the axis endpoint,
with a family-clustered bootstrap tube. A positive, zero-excluding correlation is co-drift (the
block moves in the same classes); a correlation covering zero is dissociation. The four-class
profile is coarse, so the read stays qualitative, aligned against dissociated, with the tube
carrying the uncertainty. This is the general form of the H0J question.

*Conditioning* asks whether a block accounts for the drift: remove the block's linear contribution
to the features and report whether the class displacement shrinks. It is well posed only for a
low-dimensional covariate, so a high-dimensional block earns a conditioning score only through an
explicit, named reduction (a composite score, a chosen taxon, a single polygenic score). The read
is a descriptive partial association, not a causal mediation claim: a covariate that is itself a
consequence of the same developmental process would be over-adjusted.

Everything here is a pure consumer of the cached fit's frozen responsibilities and the strata
axis; it refits nothing. The functions take arrays and return results, leaving the cohort joins and
artefact writing to the command layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.invariance import benjamini_hochberg
from analysis.localise import gaussian_weights
from analysis.trajectory_local import (
    _family_rows,
    grain_magnitude,
    local_centroids,
    pooled_centroids,
)


def _grain_rms(standardised: np.ndarray, columns: np.ndarray) -> np.ndarray:
    r"""Return the per-feature root-mean-square intensity of a grain, keeping the leading axes.

    The size-fair intensity $\lVert d_k/\sigma \rVert_\text{grain} / \sqrt{n_\text{grain}}$, the
    quadratic mean of the standardised displacement over the grain's features. Dividing by the
    square root of the feature count makes grains of different size comparable, unlike the raw norm
    :func:`analysis.trajectory_local.grain_magnitude` reports. This is the size-fair primitive the
    grain contrast reads; the referent split of the era drift is one instance of it.
    """
    return np.sqrt(np.nanmean(standardised[..., columns] ** 2, axis=-1))


# A profile correlation over four classes is undefined when a profile has no spread (a block that
# moves every class alike). Below this standard deviation the profile is treated as flat and the
# alignment is reported as zero rather than as a divide-by-near-zero artefact.
_PROFILE_STD_FLOOR = 1e-12


def per_class_profile(
    x_values: np.ndarray,
    responsibilities: np.ndarray,
    weights: np.ndarray,
    pooled_sd: np.ndarray,
    separation_scale: float,
) -> np.ndarray:
    r"""Return each class's separation-scaled displacement magnitude at one focal point.

    The local centroid of every class under the frozen responsibilities and the kernel window,
    minus the pooled centroid, read as a per-class magnitude over the block's columns. With the
    block's own ``pooled_sd`` and ``separation_scale`` this is the block's per-class drift profile;
    with the reference feature matrix it is the phenotype's.

    Parameters
    ----------
    x_values : numpy.ndarray
        The block's measurement matrix, shape ``(n_probands, n_columns)``.
    responsibilities : numpy.ndarray
        The frozen posterior responsibilities $r_{ik}$, shape ``(n_probands, n_classes)``.
    weights : numpy.ndarray
        The per-proband Gaussian kernel weight at the focal point, shape ``(n_probands,)``.
    pooled_sd : numpy.ndarray
        The block's per-column pooled standard deviation, shape ``(n_columns,)``.
    separation_scale : float
        The between-class separation the magnitude is divided by. A single scalar, so it cancels in
        the co-drift correlation; it is kept for the magnitude to stay interpretable.

    Returns
    -------
    numpy.ndarray
        The per-class magnitude, shape ``(n_classes,)``; a class with no local weight is
        not-a-number.
    """
    displacement = local_centroids(x_values, responsibilities, weights) - pooled_centroids(
        x_values, responsibilities
    )
    columns = np.arange(x_values.shape[1])
    return grain_magnitude(displacement, pooled_sd, columns, separation_scale)


def profile_alignment(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    r"""Return the class-resolved co-drift: the correlation of two per-class magnitude profiles.

    A Pearson correlation of the two four-class profiles, so a block that peaks in the same class
    as the phenotype scores near $+1$ (co-drift), a block that peaks in a different class scores
    negative (dissociation), and a block that moves every class alike, a flat profile, scores near
    zero. Centring makes it insensitive to the shared positive level of two magnitude vectors,
    which a raw cosine is not. Classes not-a-number in either profile (no local weight) are dropped;
    with fewer than two shared finite classes, or with either profile flat, the alignment is zero.

    Parameters
    ----------
    profile_a, profile_b : numpy.ndarray
        The two per-class magnitude profiles, each shape ``(n_classes,)``.

    Returns
    -------
    float
        The correlation in ``[-1, 1]``, or ``0.0`` when it is undefined.
    """
    a = np.asarray(profile_a, dtype=float)
    b = np.asarray(profile_b, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < 2:
        return 0.0
    a, b = a[good], b[good]
    if a.std() < _PROFILE_STD_FLOOR or b.std() < _PROFILE_STD_FLOOR:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@dataclass(frozen=True)
class CoDriftResult:
    r"""The class-resolved co-drift of one block with the phenotype drift.

    Attributes
    ----------
    phenotype_profile, block_profile : numpy.ndarray
        The observed per-class magnitude profiles, each shape ``(n_classes,)``.
    alignment : float
        The observed profile correlation (:func:`profile_alignment`).
    ci_low, ci_high : float
        The family-bootstrap 2.5 and 97.5 percentiles of the alignment.
    p_value : float
        The two-sided add-one bootstrap $p$ against no alignment, floored at one over the number
        of bootstrap replicates plus one.
    aligned : bool
        Whether the interval excludes zero on the positive side (co-drift rather than dissociation).
    n_joint : int
        The number of probands finite on both the axis and the block, the paired sample.
    """

    phenotype_profile: np.ndarray
    block_profile: np.ndarray
    alignment: float
    ci_low: float
    ci_high: float
    p_value: float
    aligned: bool
    n_joint: int


def co_drift(
    x_phenotype: np.ndarray,
    x_block: np.ndarray,
    responsibilities: np.ndarray,
    families: np.ndarray,
    axis_values: np.ndarray,
    bandwidth: float,
    focal: float,
    *,
    pooled_sd_phenotype: np.ndarray,
    pooled_sd_block: np.ndarray,
    separation_phenotype: float,
    separation_block: float,
    n_boot: int,
    seed: int,
) -> CoDriftResult:
    r"""Test whether a block co-drifts with the phenotype in the same classes.

    Both profiles are read along the *same* axis at the *same* endpoint focal point, since the
    block co-drifts along era or age, not along a variable of its own. Every bootstrap replicate
    resamples one set of families and recomputes both the phenotype and the block profile and their
    correlation on that same resample, so the two share their sampling variation and the alignment
    is a genuine paired statistic. The observed alignment then acts as its own bootstrap-inverted
    test, the construction :func:`analysis.trajectory_local.control_specificity_bootstrap` uses.

    The inputs must already be restricted to the probands finite on both the axis and the block, so
    that a family resampled for one profile is resampled for the other.

    Parameters
    ----------
    x_phenotype : numpy.ndarray
        The reference feature matrix, shape ``(n_joint, n_features)``.
    x_block : numpy.ndarray
        The block's matrix over the same rows, shape ``(n_joint, n_columns)``.
    responsibilities : numpy.ndarray
        The frozen responsibilities over the same rows, shape ``(n_joint, n_classes)``.
    families : numpy.ndarray
        The per-proband family identifier over the same rows, the clustering unit.
    axis_values : numpy.ndarray
        The timing axis over the same rows.
    bandwidth : float
        The axis Gaussian kernel bandwidth.
    focal : float
        The axis endpoint focal position both profiles are read at.
    pooled_sd_phenotype, pooled_sd_block : numpy.ndarray
        Each matrix's per-column pooled standard deviation.
    separation_phenotype, separation_block : float
        Each matrix's between-class separation (cancels in the correlation, kept for the magnitude).
    n_boot : int
        The number of paired bootstrap replicates.
    seed : int
        The bootstrap seed.

    Returns
    -------
    CoDriftResult
        The observed profiles, their alignment, and its bootstrap interval and $p$-value.
    """
    weights = gaussian_weights(pd.Series(axis_values), float(focal), bandwidth).to_numpy()

    def alignment_of(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        pheno = per_class_profile(
            x_phenotype[rows],
            responsibilities[rows],
            weights[rows],
            pooled_sd_phenotype,
            separation_phenotype,
        )
        block = per_class_profile(
            x_block[rows],
            responsibilities[rows],
            weights[rows],
            pooled_sd_block,
            separation_block,
        )
        return pheno, block, profile_alignment(pheno, block)

    all_rows = np.arange(x_phenotype.shape[0])
    pheno_obs, block_obs, observed = alignment_of(all_rows)

    groups, _ = _family_rows(np.asarray(families))
    n_groups = len(groups)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, n_groups, size=n_groups)
        rows = np.concatenate([groups[c] for c in chosen])
        draws[b] = alignment_of(rows)[2]

    ci_low, ci_high = (float(v) for v in np.percentile(draws, (2.5, 97.5)))
    floor = 1.0 / (n_boot + 1)
    # Inclusive tails, so a dissociated block whose profiles are flat gives alignment draws pinned
    # at zero and an honest $p$ near one, rather than a strict ``> 0`` count reading zero draws on
    # the far side as significance.
    frac_at_or_below = float(np.mean(draws <= 0.0))
    frac_at_or_above = float(np.mean(draws >= 0.0))
    p_value = float(np.clip(2.0 * min(frac_at_or_below, frac_at_or_above), floor, 1.0))
    return CoDriftResult(
        phenotype_profile=pheno_obs,
        block_profile=block_obs,
        alignment=float(observed),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        aligned=bool(ci_low > 0.0),
        n_joint=int(x_phenotype.shape[0]),
    )


@dataclass(frozen=True)
class Decomposition:
    """An internal block's per-partition share of a class's drift.

    Attributes
    ----------
    partitions : list of str
        The partition names (the grain keys), in a fixed order.
    squared_magnitude : numpy.ndarray
        The separation-scaled squared magnitude of each partition, shape
        ``(n_classes, n_partitions)``.
    whole : numpy.ndarray
        The whole-block squared magnitude per class, shape ``(n_classes,)``; the partition squared
        magnitudes sum to this (the partitions tile the columns).
    share : numpy.ndarray
        Each partition's fraction of the whole per class, shape ``(n_classes, n_partitions)``.
    """

    partitions: list[str]
    squared_magnitude: np.ndarray
    whole: np.ndarray
    share: np.ndarray


def decompose(
    displacement: np.ndarray,
    pooled_sd: np.ndarray,
    partitions: dict[str, np.ndarray],
    separation_scale: float,
) -> Decomposition:
    r"""Split an internal block's class displacement into per-partition squared magnitudes.

    For a partition of the block's columns (the seven author categories, the two referents, or any
    tiling), the separation-scaled squared magnitude is additive: because the magnitude is a
    Euclidean norm over standardised per-feature displacements, the squared magnitudes of a set of
    disjoint column groups that cover the block sum to the whole-block squared magnitude. This is
    the additive sum-of-squares share the H0G referent decomposition already reports, lifted
    to an arbitrary partition.

    Parameters
    ----------
    displacement : numpy.ndarray
        The per-feature class displacement at the reported focal point, shape
        ``(n_classes, n_columns)``.
    pooled_sd : numpy.ndarray
        The per-column pooled standard deviation, shape ``(n_columns,)``.
    partitions : dict of str to numpy.ndarray
        The partition name mapped to its integer column indices. The groups must be disjoint and
        cover every column for the shares to sum to one.
    separation_scale : float
        The between-class separation the magnitude is divided by.

    Returns
    -------
    Decomposition
        The per-partition squared magnitudes, the whole, and the shares.
    """
    names = list(partitions)
    squared = np.column_stack(
        [
            grain_magnitude(displacement, pooled_sd, partitions[name], separation_scale) ** 2
            for name in names
        ]
    )
    whole = squared.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(whole[:, None] > 0.0, squared / whole[:, None], 0.0)
    return Decomposition(partitions=names, squared_magnitude=squared, whole=whole, share=share)


@dataclass(frozen=True)
class GrainContrast:
    r"""The size-fair contrast between two internal grains and its bootstrap test.

    The general form of the H0G referent contrast, over any two disjoint column groups of an
    internal block (current against retrospective instruments, one category against the rest, one
    instrument against another). The statistic is size-fair: each grain's intensity is the
    root-mean-square standardised displacement over its own features, so a grain does not win by
    holding more features. The additive sum-of-squares shares are returned alongside as the
    descriptive split, but the contrast, not the share, is the test.

    Attributes
    ----------
    contrast : numpy.ndarray
        The observed group-A-minus-group-B root-mean-square contrast per class, shape
        ``(n_classes,)``.
    ci_low, ci_high : numpy.ndarray
        The clustered-bootstrap interval of the contrast per class, shape ``(n_classes,)``.
    p_value : numpy.ndarray
        The two-sided add-one bootstrap $p$-value per class, shape ``(n_classes,)``.
    reject : numpy.ndarray
        The Benjamini-Hochberg decision across the classes at level ``q``, shape ``(n_classes,)``.
    rms_a, rms_b : numpy.ndarray
        Each grain's size-fair root-mean-square intensity per class, shape ``(n_classes,)``.
    share_a, share_b : numpy.ndarray
        Each grain's additive sum-of-squares share per class (summing to one over the two grains),
        shape ``(n_classes,)``.
    """

    contrast: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    p_value: np.ndarray
    reject: np.ndarray
    rms_a: np.ndarray
    rms_b: np.ndarray
    share_a: np.ndarray
    share_b: np.ndarray


def grain_contrast(
    feature_draws: np.ndarray,
    observed_displacement: np.ndarray,
    group_a_cols: np.ndarray,
    group_b_cols: np.ndarray,
    *,
    q: float = 0.05,
) -> GrainContrast:
    r"""Test the per-class size-fair contrast between two internal grains from the tube draws.

    Reads the size-fair root-mean-square intensity of two disjoint column groups from the
    standardised endpoint displacement, forms the group-A-minus-group-B contrast per class, and
    calls significance from the clustered-bootstrap replicates the tube already holds (the
    standardised per-feature displacement at the endpoint over the same family resamples). No new
    bootstrap loop runs: the draws are paired (both grains re-read on the same resample) and
    family-clustered. The two-sided add-one $p$-value is the fraction of replicate contrasts on the
    far side of zero, doubled and floored, and Benjamini-Hochberg control is applied across the
    classes. The H0G referent split of the era drift is the instance where group A is the
    current-state grain and group B the retrospective grain (the `invariance-trajectory` era stage
    calls this with those two grains).

    Parameters
    ----------
    feature_draws : numpy.ndarray
        The standardised per-feature displacement replicates at the endpoint, shape
        ``(n_boot, n_classes, n_features)``.
    observed_displacement : numpy.ndarray
        The observed standardised per-feature displacement at the endpoint, shape
        ``(n_classes, n_features)``.
    group_a_cols, group_b_cols : numpy.ndarray
        The column indices of the two grains being contrasted.
    q : float, optional
        The false-discovery-rate level across the classes.

    Returns
    -------
    GrainContrast
        The per-class contrast, its interval, $p$-value, and FDR decision, and the per-grain
        root-mean-square intensities and additive shares.
    """
    group_a_cols = np.asarray(group_a_cols, dtype=int)
    group_b_cols = np.asarray(group_b_cols, dtype=int)

    rms_a = _grain_rms(observed_displacement, group_a_cols)
    rms_b = _grain_rms(observed_displacement, group_b_cols)
    contrast = rms_a - rms_b

    ss_a = np.nansum(observed_displacement[:, group_a_cols] ** 2, axis=1)
    ss_b = np.nansum(observed_displacement[:, group_b_cols] ** 2, axis=1)
    total = ss_a + ss_b
    safe = np.where(total > 0.0, total, 1.0)
    share_a = np.where(total > 0.0, ss_a / safe, np.nan)
    share_b = np.where(total > 0.0, ss_b / safe, np.nan)

    draw_a = _grain_rms(feature_draws, group_a_cols)
    draw_b = _grain_rms(feature_draws, group_b_cols)
    contrast_draws = draw_a - draw_b
    n_boot = contrast_draws.shape[0]
    ci_low = np.nanpercentile(contrast_draws, 2.5, axis=0)
    ci_high = np.nanpercentile(contrast_draws, 97.5, axis=0)
    frac_positive = np.mean(contrast_draws > 0.0, axis=0)
    tail = np.minimum(frac_positive, 1.0 - frac_positive)
    p_value = np.clip(2.0 * tail, 1.0 / (n_boot + 1), 1.0)
    reject = benjamini_hochberg(p_value, q)

    return GrainContrast(
        contrast=contrast,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        reject=reject,
        rms_a=rms_a,
        rms_b=rms_b,
        share_a=share_a,
        share_b=share_b,
    )


def residualise(x_values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    r"""Remove a covariate's linear contribution from every column of a matrix.

    The ordinary-least-squares residual of each column of ``x_values`` on ``[1, z]``, so the
    returned matrix is the part of the features that a linear reading of the covariate does not
    explain. Conditioning reads the class displacement on this residual: if the drift was carried
    by variation the covariate tracks, the residual displacement shrinks.

    Parameters
    ----------
    x_values : numpy.ndarray
        The feature matrix, shape ``(n_probands, n_features)``.
    covariate : numpy.ndarray
        The reduced covariate, shape ``(n_probands,)`` or ``(n_probands, n_reduced)``.

    Returns
    -------
    numpy.ndarray
        The residualised matrix, shape ``(n_probands, n_features)``.
    """
    z = np.asarray(covariate, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    design = np.column_stack([np.ones(z.shape[0]), z])
    coefficients, _, _, _ = np.linalg.lstsq(design, x_values, rcond=None)
    return x_values - design @ coefficients


@dataclass(frozen=True)
class ConditioningResult:
    """How much a class's drift survives removing a covariate's linear contribution.

    Attributes
    ----------
    raw_magnitude, conditioned_magnitude : numpy.ndarray
        The per-class endpoint magnitude before and after residualising the features on the
        covariate, each in a fixed pooled-standard-deviation metric, shape ``(n_classes,)``.
    shrinkage : numpy.ndarray
        ``1 - conditioned / raw`` per class, the fraction of the drift the covariate accounts for;
        near one when the covariate carries the drift, near zero when it is irrelevant.
    """

    raw_magnitude: np.ndarray
    conditioned_magnitude: np.ndarray
    shrinkage: np.ndarray


def conditioning_shrinkage(
    x_values: np.ndarray,
    responsibilities: np.ndarray,
    weights: np.ndarray,
    covariate: np.ndarray,
    pooled_sd: np.ndarray,
    separation_scale: float,
) -> ConditioningResult:
    r"""Report how much of each class's drift a reduced covariate accounts for.

    The class displacement is read twice in the *same* fixed metric (the raw features' pooled
    standard deviation), once on the features and once on the features residualised on the
    covariate (:func:`residualise`). A covariate that carries the along-axis movement leaves a
    small residual displacement, so the magnitude shrinks; an irrelevant covariate leaves the
    displacement almost unchanged. The shared metric and the shared ``separation_scale`` cancel in
    the ratio, so the shrinkage reads how much of the movement the covariate linearly explains, not
    a change of units.

    This is a descriptive partial association, H0H's adjustment generalised to a declared
    reduction, not a causal mediation claim: a covariate that is itself a downstream consequence of
    the drift would be over-adjusted.

    Parameters
    ----------
    x_values : numpy.ndarray
        The reference feature matrix, shape ``(n_probands, n_features)``.
    responsibilities : numpy.ndarray
        The frozen responsibilities, shape ``(n_probands, n_classes)``.
    weights : numpy.ndarray
        The per-proband kernel weight at the endpoint focal point, shape ``(n_probands,)``.
    covariate : numpy.ndarray
        The low-dimensional reduction, shape ``(n_probands,)`` or ``(n_probands, n_reduced)``.
    pooled_sd : numpy.ndarray
        The fixed per-feature metric both readings use, shape ``(n_features,)``.
    separation_scale : float
        The between-class separation both readings divide by (cancels in the shrinkage).

    Returns
    -------
    ConditioningResult
        The raw and conditioned per-class magnitudes and their shrinkage.
    """
    raw = per_class_profile(x_values, responsibilities, weights, pooled_sd, separation_scale)
    conditioned = per_class_profile(
        residualise(x_values, covariate),
        responsibilities,
        weights,
        pooled_sd,
        separation_scale,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        shrinkage = np.where(raw > 0.0, 1.0 - conditioned / raw, 0.0)
    return ConditioningResult(
        raw_magnitude=raw, conditioned_magnitude=conditioned, shrinkage=shrinkage
    )
