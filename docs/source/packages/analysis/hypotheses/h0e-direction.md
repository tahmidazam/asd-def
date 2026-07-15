# $H_0^E$: does the drift have a direction along the axis?

:::{admonition} Definition
:class: note

$H_0^E$ (direction). Null: the drift is non-directional, with no systematic trend along the axis; a
class's signed displacement swings out and back with no net movement. Alternative: the drift is
directional, either monotone in the axis (increasing with later diagnosis year, or ordered across
{term}`age at diagnosis`) or discontinuous at the {term}`DSM-5 boundary`. Estimand: the sign and
shape of the class-parameter-by-axis function, read per class as the net-projected slope of the
same displacement whose size $H_0^A$ and $H_0^D$ report. The test is conditional on rejecting
$H_0^A$: it is evaluated only on the axes where an invariance null falls, and it separates the
direction of a drift already established as real.
:::

:::{admonition} Status
:class: tip

Partially rejected: directional in 7 of the 8 class-by-axis tests under the joint
{term}`false discovery rate` step ($q<0.05$ across the four classes and both axes). Along
{term}`age at diagnosis` all four classes move the same way, a coherent monotone trend led by the
developmental class (Mixed ASD with developmental delay, net trend $+0.75$ separation units,
interval $[+0.60, +0.92]$; Moderate challenges lowest at $+0.33$), every interval clear of zero. Along
{term}`diagnostic era` the classes split: Broadly affected ($-0.43$, $[-0.46, -0.39]$) and Social or
behavioral ($-0.40$, $[-0.42, -0.37]$) trend one way, Mixed ASD with developmental delay ($+0.41$,
$[+0.37, +0.45]$) the other. Moderate challenges is the exception: its era net trend is $-0.02$ with an
interval $[-0.05, +0.00]$ that touches zero, so its era movement is a non-directional magnitude
excursion, the one test that does not clear the joint false-discovery step. On era the descriptive
single break clusters around 2017, a few years after the DSM-5 boundary of 2013.
:::

## Method

Whether a class drifts is one question; whether the drift points somewhere is another, and a
magnitude cannot answer the second. The local centroid of the effect-size read is a kernel-weighted
average around a focal point on the axis, so at the interior of the axis the window is balanced and
the centroid sits close to the pooled profile, while at either end it is pulled towards the local
data. A displacement that grows steadily across the axis and one that swings out at the extremes and
returns through the middle both read as a U-shaped magnitude. Only the sign separates them, so the
directional statistic works on the signed displacement, on the same cached fit and the same
{term}`family-clustered bootstrap` the magnitude uses, with no refit.

For each class the standardised displacement $d_k(f)/\sigma$ is regressed on axis position, one
ordinary-least-squares slope per feature, and the slope vector is projected onto the class's net
direction, the average direction it drifts. The projection is a signed number: positive when the
class moves further from the pooled profile as the axis advances, negative when it moves back through
it. On an evenly spaced focal grid the slope and the average are orthogonal contrasts, which is what
lets the projection sit honestly at zero under no trend. Scaled by the span of the axis over the
{term}`between-class separation`, it is a net-trend effect size in the same units as the endpoint
displacement that $H_0^A$ and $H_0^D$ report. A secondary, descriptive read fits a single break to
each signed one-dimensional trajectory, two independent least-squares segments so a level shift is
allowed rather than smoothed, to localise a possible discontinuity at the DSM-5 boundary.

This is part of the `analysis invariance-trajectory` stage, which also carries the per-class
magnitude; the per-class directional net-trend statistic is computed alongside it. The displacement,
the separation it is scaled against, and the correctness gates that separate a planted trend from a
planted excursion are the {doc}`class-drift machinery <../guides/measuring-class-drift>`.

## Experimental design decisions

- The projected slope, not the norm. Reducing the slope vector to its length would answer direction
  too, but the length of a noisy vector is biased upward and can never sit at zero, so a class with
  no trend would still look directional. The projection onto the net direction keeps the statistic
  signed and unbiased, with zero expectation under no drift.
- The family bootstrap freezes the direction. The net direction is fixed at its observed value and
  whole families are resampled, so siblings move together and the projected slope stays a fixed
  linear read whose two-sided bootstrap interval can cover zero. Significance comes from that
  interval, not the slope on its own, because of the upward bias the projection removes.
- A descriptive single-break read. The break localises a discontinuity on the signed trajectory, but
  the score-based test's supremum-LM confidence set spans the whole axis at this sample size, so the
  break is reported with its bootstrap spread, not as a resolved changepoint.
- Per-class net directions. Each sign is read against that class's own net direction, so the era
  split is a statement about how each class drifts rather than a shared axis the four are ranked on.

## Results

The runs are on SPARK `2026-03-23` (11,704 probands), against the measurement-only reference fit
`41ab0e38`, at 500 bootstrap replicates: the era run (coverage 99.5 per cent, bandwidth 1.87 years)
and the age run (coverage 99.7 per cent, bandwidth 2.21 years). Age at diagnosis moves all four
classes the same way, a monotone trend with no reversal, the net-trend effect sizes running from
$0.33$ for Moderate challenges to $0.75$ for Mixed ASD with developmental delay (separation units, a
fraction of the mean inter-class gap), every interval clear of zero, so the developmental class both
moves furthest and trends hardest. Diagnostic era divides
the classes: Broadly affected and Social or behavioral are most distinct from the pooled profile
among the earliest-diagnosed and converge through it as diagnosis year advances, while Mixed ASD with
developmental delay goes the other way, growing more distinct in the recent years. Moderate
challenges is the exception, its endpoint displacement clearing the era specificity controls yet its
net trend covering zero, so its era movement is a non-directional excursion.

:::{figure} /_figures/local_directional_era.png
:alt: Each class's signed displacement along its net direction across diagnostic era, with bootstrap bands
:width: 100%
:align: center

Directional drift along diagnostic era. Each line is a class's local centroid projected onto its own
net direction, the shaded band its family-clustered bootstrap interval, the horizontal line the
pooled profile, and the dotted verticals the descriptive single-break locations. How to read it: a
line sloping away from the pooled level is a class that shifts steadily as diagnosis year advances,
while a flat line near the pooled level has no directional trend whatever its magnitude; where a band
clears the pooled level the trend is resolved, and where it straddles it the class is not
directional. Broadly affected and Social or behavioral converge through the pooled profile while
Mixed ASD with developmental delay diverges, and Moderate challenges stays flat. Rendered by
{py:mod}`figures.trajectory_local` (`figures local-directional --axis era`).
:::

:::{figure} /_figures/local_directional_age_at_diagnosis.png
:alt: Each class's signed displacement along its net direction across age at diagnosis, with bootstrap bands
:width: 100%
:align: center

The same read for age at diagnosis. How to read it: as above, with each line a class projected onto
its own net direction and the band its family-clustered bootstrap interval. All four lines slope away
from the pooled level in the same sense with no reversal, so the age drift is a coherent monotone
trend, led by the developmental class. Rendered by {py:mod}`figures.trajectory_local`
(`figures local-directional --axis age_at_diagnosis`).
:::

The secondary single-break read places the era breaks around 2017 (2016.5 for Moderate challenges,
2017.8 for the other three), a few years after the DSM-5 boundary of 2013, and consistent with the
2015 to 2018 breaks the {term}`score-based invariance test`
{footcite}`merkleTestsMeasurementInvariance2013` placed on the same axis. On age at diagnosis the
breaks fall between 4.75 and 7.25 years, matching that test's five-to-six-year estimates.

## Handling the null

The decision is a per-class net-trend comparison: a class is directional if its two-sided
family-clustered-bootstrap interval on the projected slope excludes zero, and non-directional if the
interval covers it. Significance is Benjamini-Hochberg-controlled ({term}`false discovery rate`)
across the four classes within an axis and across the four classes and both axes jointly. Under that
joint step 7 of the 8 class-by-axis tests reject: all four along age at diagnosis, and Broadly
affected, Social or behavioral, and Mixed ASD with developmental delay along diagnostic era. The
eighth, Moderate challenges on era, has a net trend of $-0.4$ with an interval $[-0.7, 0.0]$ that
touches zero and does not clear the joint false-discovery step, so it is read as a magnitude-only
excursion. $H_0^E$ is therefore partially rejected. The descriptive single break is not part of the
verdict: the score-based test's supremum-LM confidence set saturates and spans the whole axis at this
sample size, so the break location is reported with its bootstrap spread rather than resolved.

## Discussion

The age drift is a coherent trend across all four classes, and the era drift is directional for three
classes and a magnitude-only excursion for the fourth, the case a magnitude on its own cannot tell
apart. The read is conditional on the pooled fit: it freezes the reference responsibilities and
re-weights, so it measures where the class centroids sit along the axis, not a re-estimated
partition. The break locations are descriptive for the reason above, and the signs are per class,
each against that class's own net direction, so the era split describes how each class drifts rather
than ranking the four on a shared axis. What the directional read settles is narrow and firm, and it
separates direction from size on the same displacement the magnitude read reports.

## See also

- {doc}`Are the class profiles invariant, and is any drift small? <../hypotheses/h0a-invariance>`
  ($H_0^A$ and $H_0^D$), which reads the magnitude of the same displacement whose signed trend this
  page reads.
- {doc}`Invariance as an effect size <../hypotheses/h0a-invariance>`, the recast
  and the correctness gates that separate a planted trend from a planted excursion.
- {doc}`The score-based invariance test <../guides/the-score-based-invariance-test>`, the corroborating
  read that places the era and age breaks.
- {doc}`Measuring how far a class drifts <../guides/measuring-class-drift>`, the class-drift machinery
  this rests on.
- {doc}`The Python API <../reference>` for the `invariance-trajectory` stage.

```{footbibliography}
```
