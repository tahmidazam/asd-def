# Measuring how far a class drifts

Once a stratum's classes are matched to the reference (the
{doc}`alignment step <aligning-stratum-classes>`), the drift is how far each matched class has
moved. "How far" needs a distance, and the right distance depends on what kind of movement
matters: a shift in the class centre, whether the correlations between features should temper
that shift, and whether a change in a feature's spread should count as well as a change in its
mean. The package makes the distance a swappable method, with four defined, so the same stored
fits can be read under any of them.

This guide describes the distance interface, the four distances, and the between-class separation
they are read against. The matching that decides which stratum class is compared with which
reference class is the {doc}`sibling step <aligning-stratum-classes>`.

## The distance interface

A distance method measures one matched pair: given a stratum class and the reference class it was
aligned to, it returns one non-negative number for how far the one sits from the other. The drift
stage is typed against the interface, so the distance is chosen at the command line and the
measurement code does not change.

The reference model carries the ingredients a distance might need: the per-class centroids, the
per-class dispersions (standard deviations), the per-feature pooled standard deviation, and the
inverse covariance described below. Each distance uses the subset it needs. A stratum is
summarised once into the matching per-class centroids and dispersions and stored, so changing the
distance re-measures over the stored summaries in seconds rather than re-fitting. The summary
keeps the dispersions, not only the centroids, precisely so a distributional distance can be
added later without re-running any fit.

## The four distances

**Mahalanobis distance** is the default. It is the distance between the two class centroids
weighted by the inverse of the pooled within-class covariance, so a coordinated shift across a
block of correlated features counts once rather than once per feature. For stratum and reference
centroids $\mu_s$ and $\mu_r$ and a within-class covariance $S$,

$$ d_M = \sqrt{(\mu_s - \mu_r)^\top S^{-1} (\mu_s - \mu_r)}. $$

The covariance is estimated from the residuals of each proband about its class mean and shrunk
with the Ledoit-Wolf estimator. The shrinkage is what makes the 238-feature covariance invertible
and well-conditioned, which a raw sample covariance at this dimension is not. A feature missing
from a stratum contributes no shift.

**Standardised Euclidean distance** is the root mean square of the per-feature centroid shift,
each feature in units of its pooled standard deviation $\sigma_j$ over the $p$ shared features,
$d = \sqrt{p^{-1} \sum_j ((\mu_{s,j} - \mu_{r,j}) / \sigma_j)^2}$. It treats the features as
independent (a diagonal covariance), so a shift shared across correlated features is counted in
each of them. It is the plainest reading of the movement and the easiest to trace feature by
feature.

**Mean absolute distance** is the same diagonal, standardised shift summarised by its mean
absolute value rather than its root mean square,
$d = p^{-1} \sum_j |(\mu_{s,j} - \mu_{r,j}) / \sigma_j|$. Averaging absolute shifts rather than
squared ones makes it less sensitive to a few features that move a long way, so it reports the
typical per-feature movement.

**Jensen-Shannon distance** compares the class-conditional distributions rather than their
centres alone. Each feature is treated as Gaussian with the class mean and dispersion, and the
Jensen-Shannon divergence between the stratum's Gaussian and the reference's is averaged over the
features. It is bounded in $[0, 1]$ per feature and so overall, and it is the only one of the four
that responds to a change in a feature's spread within the class as well as to a change in its
mean. The
divergence has no closed form for two Gaussians, so each feature's value is integrated numerically
on a grid spanning both. The Gaussian treatment is an approximation, since most of the features
are binary or categorical-coded rather than continuous.

The first three measure how far the class centre has moved and differ only in whether and how
they weight the features; the Jensen-Shannon measures how far the whole class-conditional
distribution has moved.

## Reading the distance against the separation

A distance in standard-deviation units or in divergence units is hard to judge on its own. The
drift is read against the between-class separation: the same chosen distance measured between the
distinct reference classes themselves and averaged over the pairs. This is the natural yardstick,
the typical gap between two different named classes under the metric in force, so a stratum
class's drift can be reported as a fraction of it.

The separation also bounds what a credible drift looks like. A class moving an appreciable
fraction of the way towards a different class is a large but possible drift; a class moving as far
as the whole between-class gap is rarely genuine and usually marks a misaligned or reorganised
pairing rather than a class that drifted (see the reorganisation flag in
{doc}`the alignment step <aligning-stratum-classes>`). The separation is computed under whichever
distance is in force, so the drift and its yardstick stay on one scale.

The drift is also read against a permutation null, same-size random partitions of the cohort, to
decide how much of a shift exceeds chance. That calibration belongs to the drift stage; the
distance methods here supply the number it calibrates.

## Choosing the method

```
uv run analysis drift --axis age_at_diagnosis --distance mahalanobis
```

`--distance` takes `mahalanobis` (the default), `euclidean`, `mean-abs`, or `jsd`. The choice
enters the run hash, so each distance caches as its own run. The stratum fits and the permutation
null are keyed on the fitting parameters alone, not on the distance, so re-measuring under a
different distance reuses them: the costly refitting happens once, and trying another distance is
cheap.
