"""Correctness gates for the block-attribution engine (plan section 7f).

All on synthetic data (governance: no participant data in tests). The engine reads how a candidate
block co-moves with, decomposes, or accounts for the frozen-fit class drift:

1. co-drift, aligned: an external block that drifts in the same class as the phenotype gives a
   positive profile alignment whose bootstrap interval excludes zero;
2. co-drift, dissociated: an external block that drifts uniformly across the classes (the shared-
   axis confound the class-resolved statistic must reject) gives an alignment covering zero;
3. internal decomposition: a partition's per-class squared magnitudes sum to the whole-block
   squared magnitude, and the shares sum to one;
4. conditioning: a covariate that carries the drift shrinks the class displacement, while an
   irrelevant covariate leaves it almost unchanged;
5. the joint mask: restricting to the probands finite on both the axis and the block gives a co-
   drift read whose paired sample is that finite count.

The internal-block reads then reproduce the two hypotheses the engine subsumes:

6. H0F: the engine's decomposition equals the trajectory stage's per-category grain magnitude;
7. H0G: the size-fair grain contrast flags a grain-concentrated drift and is size-fair.
"""

from __future__ import annotations

import numpy as np
from analysis import blocks
from analysis import trajectory_local as tl
from analysis.localise import gaussian_weights


class _Blocks:
    """A synthetic frozen fit with a phenotype drift and a configurable external block.

    Four well-separated classes with (near-hard) one-hot responsibilities, a monotone axis, and a
    planted along-axis drift. ``block_drift`` selects the external block's movement: ``"same"``
    drifts the same class as the phenotype (co-drift), ``"uniform"`` drifts every class alike (the
    dissociated confound), and ``None`` leaves the block still. ``covariate_driven`` instead makes
    the phenotype drift a global linear function of a returned covariate, so conditioning on that
    covariate should remove it.
    """

    def __init__(
        self,
        seed: int,
        *,
        block_drift: str | None = "same",
        covariate_driven: bool = False,
        n_per: int = 200,
        n_features: int = 10,
        n_block: int = 6,
        n_classes: int = 4,
        drift_class: int = 1,
        delta: float = 8.0,
    ) -> None:
        rng = np.random.default_rng(seed)
        n = n_per * n_classes
        z = np.repeat(np.arange(n_classes), n_per)
        axis = rng.uniform(0.0, 1.0, n)
        families = np.arange(n) // 2  # two probands per family, so the bootstrap clusters

        # Phenotype: separated class centres plus a planted along-axis drift.
        centres = rng.normal(0.0, 4.0, (n_classes, n_features))
        x = centres[z] + rng.normal(0.0, 0.4, (n, n_features))
        pre_sd = x.std(axis=0)
        unit = np.zeros(n_features)
        unit[: n_features // 2] = 1.0
        unit /= np.linalg.norm(unit)

        covariate = rng.uniform(0.0, 1.0, n)
        self.irrelevant = rng.uniform(0.0, 1.0, n)
        if covariate_driven:
            # The drift is a global linear function of the covariate (every class), so residualising
            # the features on the covariate removes it and the displacement collapses.
            self.covariate = axis + 0.05 * rng.normal(0.0, 1.0, n)
            x = x + (delta * self.covariate)[:, None] * unit * pre_sd
        else:
            self.covariate = covariate
            moved = z == drift_class
            x = x.copy()
            x[moved] += (delta * axis[moved])[:, None] * unit * pre_sd

        # External block: its own space, with a drift matched to the requested pattern.
        centres_b = rng.normal(0.0, 3.0, (n_classes, n_block))
        xb = centres_b[z] + rng.normal(0.0, 0.4, (n, n_block))
        pre_sd_b = xb.std(axis=0)
        unit_b = np.zeros(n_block)
        unit_b[: n_block // 2] = 1.0
        unit_b /= np.linalg.norm(unit_b)
        if block_drift == "same":
            moved_b = z == drift_class
            xb[moved_b] += (delta * axis[moved_b])[:, None] * unit_b * pre_sd_b
        elif block_drift == "uniform":
            xb = xb + (delta * axis)[:, None] * unit_b * pre_sd_b
        elif block_drift is not None:  # pragma: no cover
            raise ValueError(block_drift)

        self.x = x
        self.xb = xb
        self.responsibilities = np.eye(n_classes)[z]
        self.axis = axis
        self.families = families
        self.pooled_sd = x.std(axis=0)
        self.pooled_sd_block = xb.std(axis=0)
        self.n_features = n_features
        self.n_block = n_block
        self.n_classes = n_classes
        self.drift_class = drift_class
        self.bandwidth = 0.15
        self.focal = 0.9

    def weights(self) -> np.ndarray:
        import pandas as pd

        return gaussian_weights(pd.Series(self.axis), self.focal, self.bandwidth).to_numpy()

    def run_co_drift(self, *, n_boot: int, seed: int) -> blocks.CoDriftResult:
        return blocks.co_drift(
            self.x,
            self.xb,
            self.responsibilities,
            self.families,
            self.axis,
            self.bandwidth,
            self.focal,
            pooled_sd_phenotype=self.pooled_sd,
            pooled_sd_block=self.pooled_sd_block,
            separation_phenotype=1.0,
            separation_block=1.0,
            n_boot=n_boot,
            seed=seed,
        )


# ---------------------------------------------------------------------------------------------
# Gate 1: co-drift, aligned.
# ---------------------------------------------------------------------------------------------


def test_aligned_block_co_drifts():
    """A block drifting in the same class as the phenotype aligns and excludes zero."""
    world = _Blocks(0, block_drift="same")
    result = world.run_co_drift(n_boot=200, seed=1)

    # The phenotype and block both peak in the drifted class.
    assert int(np.nanargmax(result.phenotype_profile)) == world.drift_class
    assert int(np.nanargmax(result.block_profile)) == world.drift_class
    # The alignment is strongly positive and its interval clears zero.
    assert result.alignment > 0.8, f"weak alignment ({result.alignment:.3f})"
    assert result.ci_low > 0.0, f"interval touches zero (lo={result.ci_low:.3f})"
    assert result.aligned
    assert result.p_value < 0.05


# ---------------------------------------------------------------------------------------------
# Gate 2: co-drift, dissociated (the shared-axis confound).
# ---------------------------------------------------------------------------------------------


def test_uniform_block_does_not_co_drift():
    """A block that drifts every class alike (pure axis effect) covers zero, not co-drift."""
    world = _Blocks(0, block_drift="uniform")
    result = world.run_co_drift(n_boot=200, seed=1)

    assert not result.aligned, f"uniform block called co-drift (align={result.alignment:.3f})"
    assert result.ci_low <= 0.0 <= result.ci_high, "interval should straddle zero"
    assert result.p_value > 0.05


# ---------------------------------------------------------------------------------------------
# Gate 3: internal decomposition sums to the whole.
# ---------------------------------------------------------------------------------------------


def test_decomposition_sums_to_whole():
    """A partition of the columns tiles the whole-block squared magnitude, shares sum to one."""
    world = _Blocks(0, block_drift="same")
    weights = world.weights()
    displacement = blocks.local_centroids(
        world.x, world.responsibilities, weights
    ) - blocks.pooled_centroids(world.x, world.responsibilities)

    half = world.n_features // 2
    partitions = {
        "first": np.arange(half),
        "second": np.arange(half, world.n_features),
    }
    result = blocks.decompose(displacement, world.pooled_sd, partitions, 1.0)

    whole = blocks.grain_magnitude(displacement, world.pooled_sd, np.arange(world.n_features), 1.0)
    assert np.allclose(result.squared_magnitude.sum(axis=1), result.whole)
    assert np.allclose(result.whole, whole**2)
    assert np.allclose(result.share.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------------------------
# Gate 4: conditioning shrinks for a carrier covariate, not for an irrelevant one.
# ---------------------------------------------------------------------------------------------


def test_conditioning_shrinks_for_the_carrier():
    """Residualising on the covariate that carries the drift collapses it; noise leaves it."""
    world = _Blocks(0, covariate_driven=True)
    weights = world.weights()

    carrier = blocks.conditioning_shrinkage(
        world.x, world.responsibilities, weights, world.covariate, world.pooled_sd, 1.0
    )
    noise = blocks.conditioning_shrinkage(
        world.x, world.responsibilities, weights, world.irrelevant, world.pooled_sd, 1.0
    )

    assert np.nanmean(carrier.shrinkage) > 0.8, "the carrier covariate did not shrink the drift"
    assert np.nanmax(np.abs(noise.shrinkage)) < 0.2, "an irrelevant covariate shrank the drift"


# ---------------------------------------------------------------------------------------------
# Gate 5: the joint mask sets the paired sample.
# ---------------------------------------------------------------------------------------------


def test_joint_mask_sets_the_paired_sample():
    """Dropping probands missing the block leaves a co-drift read over the finite count."""
    world = _Blocks(0, block_drift="same")
    block = world.xb.copy().astype(float)
    missing = np.zeros(block.shape[0], dtype=bool)
    missing[::5] = True  # a fifth of probands lack the block
    block[missing, 0] = np.nan

    finite = np.isfinite(block).all(axis=1) & np.isfinite(world.axis)
    result = blocks.co_drift(
        world.x[finite],
        block[finite],
        world.responsibilities[finite],
        world.families[finite],
        world.axis[finite],
        world.bandwidth,
        world.focal,
        pooled_sd_phenotype=world.pooled_sd,
        pooled_sd_block=world.pooled_sd_block,
        separation_phenotype=1.0,
        separation_block=1.0,
        n_boot=100,
        seed=1,
    )
    assert result.n_joint == int(finite.sum())
    assert result.n_joint < block.shape[0]
    assert result.aligned  # the co-drift survives the smaller paired sample


# ---------------------------------------------------------------------------------------------
# Gate 6: H0F agreement. The engine's decomposition reproduces the trajectory-stage category read.
# ---------------------------------------------------------------------------------------------


def test_decomposition_agrees_with_grain_magnitude():
    """Each partition's block-engine magnitude equals the trajectory stage's grain magnitude."""
    world = _Blocks(0, block_drift="same")
    weights = world.weights()
    displacement = blocks.local_centroids(
        world.x, world.responsibilities, weights
    ) - blocks.pooled_centroids(world.x, world.responsibilities)

    half = world.n_features // 2
    partitions = {"first": np.arange(half), "second": np.arange(half, world.n_features)}
    result = blocks.decompose(displacement, world.pooled_sd, partitions, 1.3)

    for j, name in enumerate(result.partitions):
        grain = tl.grain_magnitude(displacement, world.pooled_sd, partitions[name], 1.3)
        assert np.allclose(result.squared_magnitude[:, j], grain**2)


# ---------------------------------------------------------------------------------------------
# Gate 7: H0G. The size-fair grain contrast flags a grain-concentrated drift and is size-fair.
# ---------------------------------------------------------------------------------------------


def test_grain_contrast_flags_a_concentrated_grain():
    """A drift concentrated in group A gives a positive, zero-excluding, rejecting contrast."""
    rng = np.random.default_rng(3)
    n_boot, n_classes, n_features = 300, 4, 12
    group_a = np.arange(0, 8)  # the larger grain, as in the real 193:45 referent split
    group_b = np.arange(8, n_features)

    # Class 0 carries a drift concentrated in group A; the other classes carry only noise.
    observed = rng.normal(0.0, 0.02, (n_classes, n_features))
    observed[0, group_a] += 1.0
    draws = observed[None, :, :] + rng.normal(0.0, 0.05, (n_boot, n_classes, n_features))

    result = blocks.grain_contrast(draws, observed, group_a, group_b, q=0.05)

    # Size-fair: group A wins on intensity, not on its larger feature count.
    assert result.contrast[0] > 0.0
    assert result.rms_a[0] > result.rms_b[0]
    assert result.ci_low[0] > 0.0
    assert result.reject[0]
    # The additive shares sum to one over the two disjoint grains.
    assert np.allclose(result.share_a[0] + result.share_b[0], 1.0)
