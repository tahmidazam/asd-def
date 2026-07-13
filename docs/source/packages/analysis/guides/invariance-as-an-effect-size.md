# Invariance as an effect size

The score-based invariance test ({doc}`score-based-invariance`) reads the class profiles for
stability along an axis from the single cached fit. At the reference sample size, roughly 11,700
probands, it has so much power that exact invariance is always rejected: on the real runs every
whole-class block and almost every class-by-category block rejects at the smallest attainable
$p$-value, and the break confidence sets span nearly the whole axis. Exact measurement invariance
is a point null, and no real class profile sits exactly on it, so the bridge $p$-value stops
discriminating once the sample runs to the thousands. This is the known large-sample regime for
measurement invariance (Meredith 1993, *Psychometrika*; Putnick and Bornstein 2016, *Dev Rev*).

This guide describes the recast that the `invariance-trajectory` stage runs. It keeps the same
frozen fit and the same question, whether the class profiles move along age at diagnosis or
diagnosis year, but makes a null-free effect size the headline and demotes the bridge $p$-value to
corroboration. It reuses the cached measurement-only reference fit and refits nothing.

## The quantity

Freeze the pooled responsibilities $r_{ik}$ of the measurement-only reference, its `predict_proba`.
For a focal point $f$ on the axis, weight each proband by a Gaussian kernel around $f$ and read the
responsibility-weighted local centroid of class $k$,

$$ \mu_k(f) = \frac{\sum_i w_i(f)\,r_{ik}\,x_i}{\sum_i w_i(f)\,r_{ik}}, $$

where $w_i(f)$ is the kernel weight of proband $i$'s axis value about $f$ at the axis's bandwidth
(the same kernel and bandwidth the local-likelihood sweep uses, so the bandwidth is chosen by the
effective sample size of a focal fit, one focal fit worth about 1,000 probands). With a unit weight
on every proband this is the pooled centroid $\mu_k$, which equals the fit's own responsibility-
weighted class means; that identity is the first correctness gate below.

The primitive is the per-feature displacement $d_k(f) = \mu_k(f) - \mu_k$, kept full-dimensional. A
magnitude is the Euclidean norm of the standardised displacement over a group of features, divided
by the between-class separation (the mean distance between distinct pooled centroids, the same
quantity and convention the drift stage uses), so a displacement is read on the scale of the gap
between classes and is comparable across the two axes. The displacement is aggregated at two
pre-specified grains: the four classes whole, and each class within each of the seven author
categories. The whole-class grain is not a headline on its own, because a larger group of features
carries a larger norm; the bootstrap tube below is what makes grains of different size comparable.

## The discriminant plane is a view, not the authority

The class trajectories are drawn in the fixed between-class discriminant plane, the same linear-
discriminant embedding the {doc}`trajectory <../investigations/tracking-the-classes-across-strata>`
figure fits once on the pooled classes. Because the plane is two-dimensional and the displacement is
full-dimensional, a picture can flatter or hide the movement. Each class therefore carries an
in-plane capture fraction, the fraction of its displacement that lies in the plane,
$\lVert P\,d_k \rVert / \lVert d_k \rVert$ with $P$ the orthogonal projection onto the plane. A
capture near one means the picture shows essentially all of the movement; a capture near zero means
the movement is mostly out of plane and the picture understates it. The full-dimensional
separation-scaled displacement is the authoritative magnitude; the plot is a view of it, captioned
with the capture so a low-capture class is flagged rather than trusted.

:::{figure} /_figures/local_plane_era.png
:alt: Four class centroids in the discriminant plane, each with its local trajectory along diagnosis year
:width: 100%
:align: center

Local class trajectories along diagnosis year. The four pooled anchors sit far apart; each class's
centroid traces a short path coloured from the earliest to the latest focal year, with a time arrow
for the net displacement and the clustered-bootstrap tube drawn as the pale ellipses along the path.
Every class is flagged (`*`): its capture fraction is below one half, so most of the drift is out of
this plane and the short in-plane paths understate it.
:::

:::{figure} /_figures/local_plane_age_at_diagnosis.png
:alt: The same four class centroids with their local trajectories along age at diagnosis
:width: 100%
:align: center

The same read along age at diagnosis. The in-plane paths are again short and again flagged as mostly
out of plane, yet the full-dimensional magnitudes (reported below) are the largest of either axis,
which is the point of the capture fraction: the plane cannot show where most of the movement goes.
:::

## Uncertainty is a clustered bootstrap

The bridge null is replaced by a family-clustered bootstrap. Families are resampled with
replacement (SPARK groups probands into families, so siblings move together), the local centroids
are recomputed on the resample under the same frozen responsibilities, and the per-focal-point
envelope is taken. This is the test of whether the drift is above sampling noise; it is conditional
on the pooled fit, and resampling families rather than probands respects the within-family
correlation, so a block of correlated features carries an honest, wider tube. The tube also makes
grains of different size comparable, because a larger group of features carries a wider tube.

The per-feature displacements each carry a clustered-bootstrap interval, and the tests across the
four classes and the features are Benjamini-Hochberg controlled at $q = 0.05$, the repo convention.
Most intervals covering zero is the readable form of many features being invariant. The per-class
panels draw the centroid tube over the faint within-class member ellipse; the two answer different
questions, where the centroid sits and where the members spread, so both belong.

:::{figure} /_figures/local_panels_age_at_diagnosis.png
:alt: One panel per class, the local trajectory and its tight bootstrap tube over the faint member ellipse
:width: 100%
:align: center

The per-class panels for age at diagnosis. Each panel draws the class's local trajectory and its
tight centroid tube over the faint grey member ellipse. The member ellipse is broad because the
class spreads widely; the centroid tube is small because the centroid is estimated from about 1,000
effective probands at each focal point.
:::

## Specificity: the control panel

A large magnitude alone does not separate genuine drift from noise accumulated across many features,
because the norm of pure sampling noise is positive too. The specificity check reads the same
separation-scaled endpoint displacement along control variables that are not the mechanism under
test: sex, the area deprivation index, and a random ordering. The claim is comparative: the timing
axes exceed the controls. The random ordering is the floor a meaningless axis produces, and area
deprivation is a real socio-geographic variable that the class profiles may track, so it is a harder
control than the random floor. This is a magnitude comparison, not a reject-or-not decision, because
at this sample size everything rejects.

:::{figure} /_figures/local_specificity.png
:alt: Bar chart of endpoint displacement for era, age, area deprivation, sex, and random order
:width: 100%
:align: center

Endpoint displacement by axis, in separation units. The two timing axes (highlighted) sit above the
control panel, with a dot per class on each bar. Age at diagnosis is the largest, carried by the
developmental class; both timing axes clear the control mean (the dotted line), so the drift is above
the noise floor a random ordering gives and larger than the sex and deprivation controls.
:::

## Corroboration

Two corroborating reads sit beside the effect size. The covariance-aware Mahalanobis magnitude uses
the drift stage's Ledoit-Wolf-shrunk within-class precision, so a coordinated shift across correlated
features counts once; it is bootstrap-calibrated, which makes it dimension-fair across grains of
different size. The score-based bridge $p$-value of {doc}`score-based-invariance` is carried through
as well, so the saturated test that motivated the recast stays on the record rather than being
dropped.

## The correctness gates

The stage underwrites a scientific claim, so five gates, on synthetic data, guard it before any real
reading is trusted:

1. a whole-cohort window reproduces the pooled fit's class means to machine precision, the frozen-
   responsibility identity;
2. a planted in-plane drift moves the path in the planted direction, the tube excludes zero at the
   drift, the capture fraction is near one, and the separation scaling matches the drift stage;
3. a planted orthogonal drift has a large full-dimensional magnitude but a capture fraction near
   zero, so the two-dimensional view cannot silently hide it;
4. with no drift the path stays inside the tube and the per-feature false-discovery rate sits near
   nominal, so the tube is calibrated;
5. strongly within-family-correlated data yields a wider tube under family resampling than under an
   independent-proband bootstrap, so the family clustering is real rather than cosmetic.

## Running it

```
uv run analysis invariance-trajectory --axis era
uv run analysis invariance-trajectory --axis age_at_diagnosis
```

The stage resolves the measurement-only reference (the `analysis fit --no-covariates` fit), derives
the axis bandwidth at the recovery floor, reads the local trajectory and its clustered-bootstrap
tube, and writes the per-feature displacement table with its intervals and false-discovery
decisions, the separation-scaled grain magnitudes with their tube, the discriminant-plane trajectory
with the centroid tube and the member ellipses, the per-class capture fractions, and the specificity
table. Every output is class or feature level, so nothing per-proband leaves the stage. `--no-controls`
skips the control panel, `--n-boot` sets the bootstrap replicates (500 by default), and the run hash
folds in the reference fit, the axis, the bandwidth, and the bootstrap settings.

The figures are built with `uv run figures local-trajectory --axis <axis>` (the combined plane),
`uv run figures local-panels --axis <axis>` (the per-class panels), and `uv run figures
local-specificity` (the control-panel small-multiple, which reads both timing axes).

## Reading the result

On the reference release the two axes tell a consistent story. Diagnosis year and age at diagnosis
both drift beyond the control panel: the separation-scaled endpoint displacement averages about 2.8
along diagnosis year and about 6.0 along age at diagnosis, against 2.1 for area deprivation, 1.3 for
the random floor, and 0.8 for sex. The age effect is the larger, carried by the developmental class
(the Mixed ASD with developmental delay class moves furthest). Roughly a third of the per-feature
displacements survive the false-discovery step for diagnosis year and over half for age at
diagnosis, so the drift is pervasive but not total: most single features stay within sampling noise
while the class profiles as a whole move.

The capture fractions are all small, near what a random direction would give, so the drift is
high-dimensional and mostly out of the between-class discriminant plane. The short in-plane paths in
the trajectory figures are not the size of the effect; the full-dimensional magnitude is. In
root-mean-square terms the diagnosis-year magnitude is about a fifth of a class gap, which matches
the movement the refit-based drift stage measured on the hard bins, so the two methods triangulate.

Whether this recast is re-registered as the primary invariance read, and how it sits beside the
frozen refit null of the pre-registration, is a decision recorded in the progress log rather than
taken by the stage.
