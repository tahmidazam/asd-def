"""Correctness gates for the local class-profile displacement (plan section 7e recast).

Five gates, all on synthetic data (governance: no participant data in tests):

1. the frozen-responsibility identity: a whole-cohort window reproduces the pooled class means;
2. a planted in-plane drift: the path moves in the planted direction, the tube excludes zero at
   the drift, the capture fraction is near one, and the separation scaling matches
   :mod:`analysis.drift`;
3. a planted orthogonal drift: a large full-dimensional magnitude but a capture fraction near
   zero, so the 2D view cannot silently hide the movement;
4. no drift: the per-feature FDR sits near nominal and the path stays inside its tube;
5. the clustered bootstrap actually clusters: strongly within-family-correlated data yields a
   wider tube than an independent proband bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analysis import drift as drift_mod
from analysis import trajectory as trajectory_mod
from analysis import trajectory_local as tl
from analysis.features import Typing
from stepmix.stepmix import StepMix
from stepmix.utils import get_mixed_descriptor

# ---------------------------------------------------------------------------------------------
# Gate 1: the frozen-responsibility identity.
# ---------------------------------------------------------------------------------------------


def _mixed_fit(seed: int, *, n: int = 700, n_init: int = 4):
    """Fit a small measurement-only mixed-emission StepMix on well-separated synthetic classes."""
    rng = np.random.default_rng(seed)
    k = 3
    z = rng.integers(0, k, n)
    cont = np.empty((n, 4))
    for c in range(k):
        cont[z == c] = rng.normal(2.5 * c, 1.0, (int((z == c).sum()), 4))
    binary = np.empty((n, 3), dtype=int)
    for c in range(k):
        p = np.clip(0.15 + 0.3 * c, 0.02, 0.98)
        binary[z == c] = (rng.uniform(size=(int((z == c).sum()), 3)) < p).astype(int)
    cont_cols = [f"c{i}" for i in range(4)]
    bin_cols = [f"b{i}" for i in range(3)]
    frame = pd.DataFrame(np.hstack([cont, binary]), columns=cont_cols + bin_cols)
    data, descriptor = get_mixed_descriptor(frame, continuous=cont_cols, binary=bin_cols)
    model = StepMix(
        n_components=k,
        measurement=descriptor,
        n_steps=1,
        n_init=n_init,
        random_state=seed,
        progress_bar=0,
        verbose=0,
    )
    model.fit(data)
    typing = Typing(continuous=cont_cols, binary=bin_cols, categorical=[])
    return model, data, typing


def test_whole_cohort_window_reproduces_pooled_means():
    """A unit-weight window equals the pooled centroid, which matches the fit's class means."""
    model, data, _typing = _mixed_fit(0)
    x = data.to_numpy(dtype=float)
    resp = model.predict_proba(data)

    pooled = tl.pooled_centroids(x, resp)
    unit_window = tl.local_centroids(x, resp, np.ones(x.shape[0]))
    # The exact frozen-responsibility identity: the whole-cohort window is the pooled centroid.
    assert np.allclose(unit_window, pooled, atol=1e-10)
    # And it is the responsibility-weighted mean, so its Gaussian block matches the fit's stored
    # means to the EM fixed-point tolerance (a much looser, but still tight, agreement).
    means = model.get_parameters()["measurement"]["continuous"]["means"]
    pooled_frame = pd.DataFrame(pooled, columns=data.columns)
    stored = pooled_frame[[f"c{i}" for i in range(4)]].to_numpy()
    assert np.allclose(stored, means, atol=1e-5)

    # A very wide Gaussian window also converges to the pooled centroid.
    axis = pd.Series(np.linspace(0.0, 1.0, x.shape[0]))
    from analysis.localise import gaussian_weights

    wide = gaussian_weights(axis, 0.5, 50.0).to_numpy()
    assert np.allclose(tl.local_centroids(x, resp, wide), pooled, atol=1e-6)


# ---------------------------------------------------------------------------------------------
# A planted-drift world shared by gates 2 to 5.
# ---------------------------------------------------------------------------------------------


class _World:
    """A synthetic pooled fit with an optional planted centroid drift along the axis.

    Well-separated classes give a meaningful discriminant plane; the frozen responsibilities are
    the (near-hard) class indicators, so the geometry of the planted drift is exact. ``direction``
    selects where the drifted class moves: ``"in-plane"`` along the first discriminant direction,
    ``"orthogonal"`` off the plane, or ``None`` for no drift.
    """

    def __init__(
        self,
        seed: int,
        *,
        direction: str | None,
        n_per: int = 260,
        n_features: int = 12,
        n_classes: int = 4,
        delta: float = 6.0,
        break_at: float = 0.5,
        drift_class: int = 1,
        family_size: int = 1,
        family_scale: float = 0.0,
    ) -> None:
        rng = np.random.default_rng(seed)
        n = n_per * n_classes
        z = np.repeat(np.arange(n_classes), n_per)
        centres = rng.normal(0.0, 4.0, (n_classes, n_features))
        x = centres[z] + rng.normal(0.0, 0.4, (n, n_features))

        # Families: consecutive probands share a family, and (when family_scale > 0) a shared
        # family-level offset, so members are correlated within a family.
        families = np.arange(n) // family_size
        if family_scale > 0.0:
            offsets = rng.normal(0.0, family_scale, (families.max() + 1, n_features))
            x = x + offsets[families]

        axis = rng.uniform(0.0, 1.0, n)

        # The embedding is a fixed transform, fitted once on the pooled classes (as in the real
        # pipeline), so it is fitted here on the pre-drift data and never refitted. The planted
        # direction is then chosen relative to that fixed plane.
        columns = [f"f{i}" for i in range(n_features)]
        base_frame = pd.DataFrame(x, columns=columns)
        labels = pd.Series(z, index=base_frame.index)
        self.embedding = trajectory_mod.fit_embedding(
            base_frame, labels, n_components=n_classes - 1
        )
        self.plane = tl.discriminant_plane(self.embedding)

        pooled_sd_pre = base_frame.std().to_numpy()
        if direction == "in-plane":
            unit = self.plane[:, 0]
        elif direction == "orthogonal":
            raw = rng.normal(0.0, 1.0, n_features)
            unit = raw - self.plane @ (self.plane.T @ raw)
            unit = unit / np.linalg.norm(unit)
        elif direction is None:
            unit = np.zeros(n_features)
        else:  # pragma: no cover
            raise ValueError(direction)
        moved = (z == drift_class) & (axis >= break_at)
        x = x.copy()
        x[moved] += delta * unit * pooled_sd_pre

        # The pooled spread, reference, and precision are read on the observed (drifted) data; the
        # embedding stays the fixed transform above.
        frame = pd.DataFrame(x, columns=columns)
        self.reference = drift_mod.build_reference(frame, labels)

        self.x = x
        self.columns = columns
        self.responsibilities = np.eye(n_classes)[z]
        self.axis = axis
        self.families = families
        self.pooled_sd = frame.std().to_numpy()
        self.separation = tl.separation(self.reference)
        self.precision = self.reference.precision
        self.grains = {"class": np.arange(n_features)}
        self.focal_points = np.linspace(0.1, 0.9, 17)
        self.drift_class = drift_class
        self.unit = unit
        self.n_classes = n_classes

    def observed(self) -> tl.ObservedTrajectory:
        return tl.observed_trajectory(
            self.x,
            self.responsibilities,
            self.axis,
            self.focal_points,
            0.12,
            pooled_sd=self.pooled_sd,
            separation_scale=self.separation,
            grains=self.grains,
            embedding=self.embedding,
            precision=self.precision,
            plane=self.plane,
        )

    def tube(self, *, n_boot: int, seed: int, clustered: bool, focal_ref: int) -> tl.BootstrapTube:
        return tl.clustered_bootstrap_tube(
            self.x,
            self.responsibilities,
            self.axis,
            self.families,
            self.focal_points,
            0.12,
            pooled_sd=self.pooled_sd,
            separation_scale=self.separation,
            grains=self.grains,
            embedding=self.embedding,
            precision=self.precision,
            focal_ref=focal_ref,
            n_boot=n_boot,
            seed=seed,
            clustered=clustered,
        )


# ---------------------------------------------------------------------------------------------
# Gate 2: a planted in-plane drift.
# ---------------------------------------------------------------------------------------------


def test_in_plane_drift_moves_captures_and_excludes_zero():
    """An in-plane drift: right direction, capture near one, tube excludes zero, drift-scaled."""
    world = _World(1, direction="in-plane")
    obs = world.observed()
    c = world.drift_class

    # The endpoint displacement points along the planted direction (in standardised space).
    endpoint = obs.displacement[c, -1] / world.pooled_sd
    cosine = float(endpoint @ world.unit / (np.linalg.norm(endpoint) * np.linalg.norm(world.unit)))
    assert cosine > 0.9, f"path off planted direction (cos={cosine:.3f})"

    # Almost all of the movement lies in the plane.
    assert obs.capture[c] > 0.9, f"in-plane capture too low ({obs.capture[c]:.3f})"

    # The separation scaling is the drift stage's convention on this shared example.
    assert world.separation == pytest.approx(
        drift_mod.class_separation(world.reference, drift_mod.StandardisedEuclidean())
    )

    # The tube excludes zero at the drift: the drifted class's whole-class magnitude has a lower
    # band well above zero at the endpoint.
    tube = world.tube(n_boot=200, seed=0, clustered=True, focal_ref=obs.focal_ref)
    lo = tube.grain_bands["class"][0, c, obs.focal_ref]
    assert lo > 0.1, f"tube touches zero at the drift (lo={lo:.3f})"
    # And the strongest drifted feature's per-feature interval excludes zero.
    observed_disp = obs.displacement[:, obs.focal_ref] / world.pooled_sd
    inference = tl.per_feature_inference(observed_disp, tube.feature_displacement)
    strongest = int(np.argmax(np.abs(world.unit)))
    assert not inference.covers_zero[c, strongest], "drifted feature interval covers zero"
    assert inference.reject[c, strongest], "drifted feature not rejected by FDR"


# ---------------------------------------------------------------------------------------------
# Gate 3: a planted orthogonal drift.
# ---------------------------------------------------------------------------------------------


def test_orthogonal_drift_is_large_but_out_of_plane():
    """An orthogonal drift has a large full-dimensional magnitude but a near-zero capture."""
    world = _World(2, direction="orthogonal")
    obs = world.observed()
    c = world.drift_class

    endpoint = obs.displacement[c, -1] / world.pooled_sd
    cosine = float(endpoint @ world.unit / (np.linalg.norm(endpoint) * np.linalg.norm(world.unit)))
    assert cosine > 0.9, f"path off planted direction (cos={cosine:.3f})"

    # The full-dimensional magnitude is large (a clear drift) ...
    assert obs.grain_magnitude["class"][c, obs.focal_ref] > 0.3
    # ... yet almost none of it lies in the plane, so the picture cannot hide it.
    assert obs.capture[c] < 0.1, f"out-of-plane capture too high ({obs.capture[c]:.3f})"


# ---------------------------------------------------------------------------------------------
# Gate 4: no drift, tube calibrated.
# ---------------------------------------------------------------------------------------------


def test_no_drift_stays_in_tube_and_fdr_near_nominal():
    """With no drift the path sits inside its tube and few per-feature tests survive FDR."""
    world = _World(5, direction=None)
    obs = world.observed()
    tube = world.tube(n_boot=300, seed=1, clustered=True, focal_ref=obs.focal_ref)

    # The observed magnitude never pokes above the tube's upper edge: it is everywhere within
    # what sampling noise alone produces, so no class is flagged as drifting. (The lower edge is
    # not a containment test: the bootstrap of a norm sits above the observed value, so the
    # observed can dip below the lower percentile without any drift.)
    hi = tube.grain_bands["class"][2]
    under_envelope = obs.grain_magnitude["class"] <= hi + 1e-9
    assert under_envelope.mean() > 0.95, (
        f"path pokes above the tube ({under_envelope.mean():.2f} under)"
    )

    # The per-feature intervals mostly cover zero, and FDR keeps rejections near nominal.
    observed_disp = obs.displacement[:, obs.focal_ref] / world.pooled_sd
    inference = tl.per_feature_inference(observed_disp, tube.feature_displacement)
    assert inference.covers_zero.mean() > 0.9, "too few intervals cover zero under no drift"
    assert inference.reject.mean() < 0.1, (
        f"FDR over-rejects under no drift ({inference.reject.mean():.2f})"
    )


# ---------------------------------------------------------------------------------------------
# Gate 5: the clustered bootstrap actually clusters.
# ---------------------------------------------------------------------------------------------


def test_family_bootstrap_is_wider_than_iid_under_within_family_correlation():
    """Strong within-family correlation widens the family tube over an independent-proband tube."""
    world = _World(7, direction=None, family_size=8, family_scale=3.0)
    obs = world.observed()

    family = world.tube(n_boot=250, seed=3, clustered=True, focal_ref=obs.focal_ref)
    iid = world.tube(n_boot=250, seed=3, clustered=False, focal_ref=obs.focal_ref)

    # The centroid tube: the spread of the local centroid position over the replicates. A shared
    # family offset inflates it under family resampling but not under proband resampling (a
    # magnitude band would partly hide this, since a common-mode offset cancels in the
    # local-minus-pooled displacement).
    def centroid_spread(tube: tl.BootstrapTube) -> float:
        return float(np.mean(np.std(tube.ld, axis=0)))

    family_spread = centroid_spread(family)
    iid_spread = centroid_spread(iid)
    assert family_spread > 1.3 * iid_spread, (
        f"family bootstrap not wider (family={family_spread:.3f}, iid={iid_spread:.3f})"
    )
