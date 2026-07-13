# H0A / H0D: invariance as an effect size

:::{admonition} The question
:class: note

Are the four class profiles invariant to diagnostic era and age at diagnosis (H0A), and if not, is
the drift small relative to the between-class separation (H0D)? The score-based invariance test
({doc}`score-based-invariance`) reads the class profiles for stability along an axis from the
single cached fit, but at the reference sample size, roughly 11,700 probands, it has so much power
that exact invariance is always rejected: on the real runs every whole-class block and almost every
class-by-category block rejects at the smallest attainable $p$-value, and the break confidence sets
span nearly the whole axis. Exact measurement invariance is a point null, and no real class profile
sits exactly on it, so the bridge $p$-value stops discriminating once the sample runs to the
thousands, the known large-sample regime for measurement invariance (Meredith 1993, *Psychometrika*;
Putnick and Bornstein 2016, *Dev Rev*). This page recasts both nulls as a null-free effect size, the
separation-scaled local-centroid displacement along an axis, with a family-clustered bootstrap tube,
an in-plane capture fraction, and a specificity check against a control panel, so a class-profile
drift is read by its size against the gap between classes rather than by whether it is exactly zero.
:::

:::{admonition} The result
:class: tip

Both nulls are rejected. The separation-scaled endpoint displacement averages about 2.8 along
diagnosis year and about 6.0 along age at diagnosis, against a control panel of about 3.2 for
household income, 2.1 for area deprivation, and 1.3 for the random floor: both timing axes clear
the noise floor, and age at diagnosis clears every covariate control for all four classes (H0A
rejected, paired specificity bootstrap floor $p = 0.0005$ on both axes). The drift is not small
relative to the between-class separation either: roughly a third of the per-feature displacements
survive the false-discovery step for diagnosis year and over half for age at diagnosis, and in
root-mean-square terms the diagnosis-year magnitude is about a fifth of a class gap (H0D rejected,
same bootstrap tube, comparative specificity $p = 0.0005$). The capture fractions are all small, so
the drift is high-dimensional and mostly out of the between-class discriminant plane the trajectory
figures draw; the full-dimensional magnitude is the authoritative read, not the short in-plane
paths.
:::

This page describes the recast that the `invariance-trajectory` stage runs. It keeps the same
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
discriminant embedding the {doc}`trajectory <tracking-the-classes-across-strata>`
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
test, and asks whether the timing axes exceed them.

The controls are screened, not assumed. A valid control is a real proband-level covariate that is
not the phenotype and is orthogonal to the timing axes. The phenotype is excluded on principle: the
classes are defined on the 238 clustered features, so ordering probands by one of them, or by a
symptom total correlated with them, moves the class centroids by construction, which is circular
rather than a null. Orthogonality to timing is checked on the data: across the modelling cohort the
rank correlation of each control with year of diagnosis and with age at diagnosis is near zero
(household income about 0.07, area deprivation about 0.04), against 0.4 to 0.7 for variables that are
timing by construction. Two graded covariates pass both tests, household income (a nine-band
ordinal) and the 2019 area deprivation index, so the displacement trajectory has an axis to walk.
Sex, used in an earlier version of the panel, is dropped: it is binary, so the trajectory is
degenerate, and it is the least timing-orthogonal of the candidates.

The panel brackets the timing effect. The random ordering is the floor a meaningless axis produces
(mean displacement about 1.3 separation units). Household income (about 3.2) and area deprivation
(about 2.1) are real covariates that the class profiles track through a non-timing pathway, so they
are harder controls than the floor. A timing axis is above the noise floor if it clears the random
ordering, and specific to timing if it also clears the two covariate controls. This is a magnitude
comparison, not a reject-or-not decision, because at this sample size everything rejects.

:::{figure} /_figures/local_specificity.png
:alt: Bar chart of endpoint displacement for era, age, area deprivation, household income, and random order
:width: 100%
:align: center

Endpoint displacement by axis, in separation units. The two timing axes (highlighted) sit above the
control panel, with a dot per class on each bar. Age at diagnosis is the largest, carried by the
developmental class; both timing axes clear the control mean (the dotted line), above the noise floor
a random ordering gives and above the income and deprivation covariates.
:::

Specificity is read per class, not on the between-class mean, because a mean hides drift concentrated
in one class. Along age at diagnosis all four classes clear every control, so the effect is specific
throughout, and largest for the developmental class (endpoint about 10.9 separation units). Along
diagnostic era the picture splits: the moderate-challenges and the developmental (Mixed ASD with DD)
classes clear the covariate controls, while the broadly-affected and social-or-behavioral classes sit
above the random floor but below their own household-income control, so their era drift is above
noise but not specific to timing.

## Directionality (DIREC)

The magnitude answers how far a class drifts; it does not answer whether the drift has a systematic
trend along the axis. That distinction matters, because the local centroid sits nearest the pooled
centroid at the axis interior, where the kernel window is most balanced, so the magnitude is
mechanically U-shaped and cannot read direction. The directional statistic is therefore built on the
signed displacement, never on the magnitude norm.

For each class the standardised displacement $d_k(f)/\sigma$ is regressed on the axis position $f$,
one ordinary-least-squares slope per feature, giving a slope vector $b_k$. The focal grid is evenly
spaced in axis units, so an equal weight per focal point is an honest per-axis-unit trend on an
irregularly sampled axis. Reducing $b_k$ to its Euclidean norm would read direction but is positively
biased, so the slope is instead projected onto the class's net direction, the unit vector of its mean
displacement across the grid. On an evenly spaced grid the slope contrast and the mean contrast are
orthogonal, so under no drift this projected slope has zero expectation: a signed, unbiased statistic
whose clustered-bootstrap interval can honestly cover zero. Scaled by the focal span over the
between-class separation, it is the net-trend effect size, how far in separation units the linear
trend carries the class across the axis.

Significance comes from the family-clustered bootstrap, not the bare slope, because of that bias: the
net direction is frozen at the observed value and the families are resampled, so the projected slope
stays a fixed linear functional and its two-sided bootstrap $p$-value is calibrated. The decision is
Benjamini-Hochberg controlled across the four classes within an axis, and across the four classes and
both axes when the two runs are read together. A secondary, descriptive changepoint read fits a
single break to the one-dimensional signed trajectory (two independent least-squares segments, so a
level shift such as the DSM-5 (2013) boundary on diagnosis year is localised rather than smoothed),
reported with its bootstrap spread; it is labelled descriptive because the bridge supremum-LM
confidence set saturates at the full sample size.

## Corroboration

Two corroborating reads sit beside the effect size. The covariance-aware Mahalanobis magnitude uses
the drift stage's Ledoit-Wolf-shrunk within-class precision, so a coordinated shift across correlated
features counts once; it is bootstrap-calibrated, which makes it dimension-fair across grains of
different size. The score-based bridge $p$-value of {doc}`score-based-invariance` is carried through
as well, so the saturated test that motivated the recast stays on the record rather than being
dropped.

## The correctness gates

The stage underwrites a scientific claim, so gates on synthetic data guard it before any real
reading is trusted. The magnitude and tube gates:

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

The DIREC directionality gates then separate direction from magnitude:

6. a planted monotone drift is flagged directional, its net-trend interval excluding zero;
7. a symmetric excursion (down then back) has a large magnitude but a net trend whose interval
   covers zero, so it is not mistaken for direction, the gate the whole read exists for;
8. with no drift the directional test rejects at about the nominal rate across seeds;
9. within-family correlation on the axis widens the slope interval under family resampling over an
   independent-proband one, so the clustered bootstrap is doing the work significance rests on;
10. a planted step drift's single-break read localises near the planted boundary.

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

It also writes the DIREC tables: the per-class directional summary (the net-trend effect size with
its interval, the two-sided bootstrap $p$-value, the Benjamini-Hochberg decision, and the descriptive
break with its spread) and the one-dimensional signed trajectory with its band.

The figures are built with `uv run figures local-trajectory --axis <axis>` (the combined plane),
`uv run figures local-panels --axis <axis>` (the per-class panels), `uv run figures
local-directional --axis <axis>` (the signed directional trajectories), and `uv run figures
local-specificity` (the control-panel small-multiple, which reads both timing axes).

## Reading the result

On the reference release the two axes tell a consistent story with an honest split. The
separation-scaled endpoint displacement averages about 2.8 along diagnosis year and about 6.0 along
age at diagnosis, against a control panel of about 3.2 for household income, 2.1 for area
deprivation, and 1.3 for the random floor. Age at diagnosis clears every control, on the mean and for
all four classes. Diagnosis year clears the random floor and area deprivation, but its mean sits just
below the household-income control, and per class only the moderate-challenges and developmental
classes clear the covariate controls. The age effect is the larger, carried by the developmental
class (the Mixed ASD with developmental delay class moves furthest). Roughly a third of the per-feature
displacements survive the false-discovery step for diagnosis year and over half for age at
diagnosis, so the drift is pervasive but not total: most single features stay within sampling noise
while the class profiles as a whole move.

The capture fractions are all small, near what a random direction would give, so the drift is
high-dimensional and mostly out of the between-class discriminant plane. The short in-plane paths in
the trajectory figures are not the size of the effect; the full-dimensional magnitude is. In
root-mean-square terms the diagnosis-year magnitude is about a fifth of a class gap, which matches
the movement the refit-based drift stage measured on the hard bins, so the two methods triangulate.

The drift is also directional. Along age at diagnosis all four classes carry a net trend whose
interval excludes zero, in the same sense, with the developmental class moving furthest; the drift is
monotone, not a symmetric excursion. Along diagnosis year the picture is a split: the Broadly affected
and Social or behavioral classes trend one way while the Mixed ASD with developmental delay class
trends the other, and the Moderate challenges class shows a large endpoint magnitude but no
directional trend (its net-trend interval covers zero). So the Moderate era movement is a
non-directional excursion, exactly the case the magnitude alone cannot tell from a trend. The
descriptive single-break locations on diagnosis year cluster around 2017, a few years after the
DSM-5 boundary of 2013, consistent with the score-test break estimates; they are read with their
bootstrap spread, not as a resolved changepoint, because the bridge confidence set saturates.

## Limits

The read is conditional on the pooled fit: it freezes the reference responsibilities and reweights,
so it measures where the class centroids sit along the axis, not a re-estimated partition. The
capture fractions warn that the discriminant-plane figures understate the movement; the
full-dimensional, separation-scaled magnitude is the authoritative number. The specificity split on
diagnostic era, where two classes clear the covariate controls and two do not, means the era verdict
is not uniform across classes even though the joint test rejects. Whether this recast is
re-registered as the primary invariance read, and how it sits beside the frozen refit null of the
pre-registration, is a decision recorded in the progress log rather than taken by the stage.
