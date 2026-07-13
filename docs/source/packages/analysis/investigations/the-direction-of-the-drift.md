# The direction of the drift

:::{admonition} The question
:class: note

The class profiles are not invariant to diagnostic era or age at diagnosis, and the movement is
large. This page asks the next thing: does the movement have a direction? A class could be pulled a
consistent way as diagnosis year advances, or it could wander out and back with no net trend. The two
look the same to a magnitude, because the local centroid sits nearest the pooled centroid at the
interior of the axis whatever the shape of the drift. So the read is built on the signed displacement
and its slope, on the same cached fit and the same family-clustered bootstrap the effect size uses,
with no refit.
:::

:::{admonition} The result
:class: tip

The drift is directional in seven of the eight class-by-axis tests (Benjamini-Hochberg at
$q = 0.05$ across the four classes and both axes). Age at diagnosis moves every class the same way, a
coherent monotone trend led by the developmental class. Diagnostic era splits: the Broadly affected
and Social or behavioral classes trend one way while the Mixed ASD with developmental delay class
trends the other, and the Moderate challenges class has a large displacement but no direction, its
net-trend interval covering zero. So the Moderate era movement is an excursion, not a trend, the case
a magnitude on its own cannot tell apart. On era the single-break read falls around 2017, a few years
after the DSM-5 boundary of 2013.
:::

## Direction is not magnitude

Whether a class drifts is one question; whether the drift points somewhere is another. The magnitude
cannot answer the second, and not by accident. The local centroid is a kernel-weighted average around
a focal point on the axis, so at the interior of the axis the window is balanced and the centroid sits
close to the pooled profile, while at either end it is pulled towards the local data. A displacement
that grows steadily across the axis and one that swings out at the extremes and returns through the
middle both read as a U-shaped magnitude. Only the sign separates them.

So the directional statistic works on the signed displacement. For each class the standardised
displacement is regressed on the axis position, one ordinary-least-squares slope per feature, and that
slope vector is projected onto the class's net direction, the average direction it drifts. The
projection is a signed number: positive when the class moves further from the pooled profile as the
axis advances, negative when it moves back through it. Reducing the slope to its length instead would
answer direction too, but the length of a noisy vector is biased upward and can never sit at zero, so
a class with no trend would still look directional. On an evenly spaced focal grid the slope and the
average are orthogonal contrasts, which is what lets the projection sit honestly at zero under no
trend. Scaled by the span of the axis over the between-class separation, it is a net-trend effect
size in the same units as the endpoint displacement.

Significance comes from the family-clustered bootstrap, not the slope on its own, because of that
upward bias. The net direction is fixed at its observed value and whole families are resampled, so
siblings move together and the projected slope stays a fixed linear read whose two-sided bootstrap
interval can cover zero. The mechanics, and the correctness gates that separate a planted trend from a
planted excursion, are in the guide on {doc}`invariance as an effect size <../guides/invariance-as-an-effect-size>`.

The runs are on SPARK `2026-03-23` (11,704 probands), against the measurement-only reference fit
`41ab0e38`, at 500 bootstrap replicates: era run `7a62da33` (coverage 99.5 per cent, bandwidth 1.87
years) and age run `c1ca6d76` (coverage 99.7 per cent, bandwidth 2.21 years).

## The signed trajectories

:::{figure} /_figures/local_directional_era.png
:alt: Each class's signed displacement along its net direction across diagnostic era, with bootstrap bands
:width: 100%
:align: center

Directional drift along diagnostic era. Each line is a class's local centroid projected onto its own
net direction, the shaded band its family-clustered bootstrap interval, and the horizontal line the
pooled profile. The dotted verticals mark the descriptive single-break locations. The legend carries
each class's net-trend effect size with its interval and whether it is directional.
:::

:::{figure} /_figures/local_directional_age_at_diagnosis.png
:alt: Each class's signed displacement along its net direction across age at diagnosis, with bootstrap bands
:width: 100%
:align: center

The same read for age at diagnosis.
:::

### How to read it

A line sloping away from the pooled level is a class whose profile shifts steadily as the axis
advances; a flat line near the pooled level is a class with no directional trend, whatever its
magnitude elsewhere. A line that crosses the pooled level has a local centroid on one side of the
pooled profile among the earliest-diagnosed and the other side among the latest. Where a band clears
the pooled level the trend is resolved; where it straddles it the class is not directional. The dotted
vertical is where a single break best splits the trajectory, read with its bootstrap spread rather
than as a resolved changepoint.

### What it shows

Age at diagnosis moves all four classes the same way, a monotone trend with no reversal. The net-trend
effect sizes run from 5.1 for Moderate challenges to 11.6 for Mixed ASD with developmental delay
(interval $[9.2, 14.2]$), every interval clear of zero, so the developmental class both moves furthest
and trends hardest. Diagnostic era divides the classes. Broadly affected ($-6.6$, interval
$[-7.1, -6.1]$) and Social or behavioral ($-6.1$, $[-6.5, -5.7]$) are most distinct from the pooled
profile among the earliest-diagnosed and converge through it as diagnosis year advances, while Mixed
ASD with developmental delay ($+6.3$, $[+5.6, +7.0]$) goes the other way, growing more distinct in the
recent years. Moderate challenges is the exception: its endpoint displacement clears the era specificity
controls, yet its net trend is $-0.4$ with an interval $[-0.7, 0.0]$ that touches zero, so its era
movement is a non-directional excursion. It is the one class-by-axis test that does not clear the joint
false-discovery step.

## The DSM-5 boundary

The secondary read fits a single break to each signed trajectory, two independent least-squares
segments so a level shift is allowed rather than smoothed. On diagnostic era the breaks cluster
around 2017 (2016.5 for Moderate challenges, 2017.8 for the other three), a few years after the DSM-5
boundary of 2013, and consistent with the 2015 to 2018 breaks the score-based test placed on the same
axis. It is labelled descriptive: the score test's supremum-LM confidence set spans the whole axis at
this sample size, so the break is reported with its bootstrap spread, not as a resolved changepoint.
On age at diagnosis the breaks fall between 4.75 and 7.25 years, matching the score test's five-to-six
year estimates.

## Limits

The read is conditional on the pooled fit: it freezes the reference responsibilities and re-weights,
so it measures where the class centroids sit along the axis, not a re-estimated partition. The break
locations are descriptive for the reason above. The signs are per class, each against that class's own
net direction, so the era split is a statement about how each class drifts rather than a shared axis
the four are ranked on. What the directional read settles is narrower and firm: the age drift is a
coherent trend across all four classes, the era drift is directional for three and a magnitude-only
excursion for the fourth, and both separate direction from size on the same effect size the
{doc}`invariance recast <../guides/invariance-as-an-effect-size>` measures.
