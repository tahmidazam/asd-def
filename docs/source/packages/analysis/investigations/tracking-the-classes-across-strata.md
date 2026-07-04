# Tracking the classes across the strata

:::{admonition} The question
:class: note

When the four-class mixture is re-estimated on its own within a stratum of age at diagnosis, or
within a stratum of diagnostic era, do the classes stay put? The stratified stage refits the model
independently in each stratum, aligns the stratum classes back to the named reference, and measures
how far each class moved, against a permutation null that breaks only the link to the axis. This
page reports the first such run: both axes, one hundred permutations, under the frozen primary
scheme.
:::

:::{admonition} The result
:class: tip

No single stratum is an outlier: no class drifts beyond its matched-size null on either axis (0 of
44 class-by-stratum tests on age at diagnosis, 0 of 40 on era, Benjamini-Hochberg at $q = 0.05$).
Across the strata as a whole, though, the classes move in a consistent direction: the young-to-old
displacement is larger than an ordering-shuffle null for all four classes on the age axis and for
three of four on era. The movement is carried by the two developmental classes and by the
developmental and milestone features, and the membership turns over most at the extremes of each
axis. This is the middle case the design anticipated: a gradual, feature-driven trajectory rather
than either fixed classes or a partition that breaks apart. It is a pilot at one hundred
permutations, not the confirmatory run.
:::

## Two tests, one answer

The two findings look opposed only until the two tests are separated. The drift test asks a local
question: is any one stratum's class further from the reference than a random stratum of the same
size would be? The directional test asks a global one: taken in order from the youngest-diagnosed
stratum to the oldest, do the class centroids move a consistent net distance, more than a reshuffle
of the stratum order would give? A set of classes can pass the first and fail the second at once.
That is what happens here: no stratum stands out on its own, yet the strata in sequence trace a
coherent path. A slow drift spread evenly across eleven strata leaves each single step within the
noise while the sum of the steps is not.

The run is on SPARK 2026-03-23 (11,704 probands), under the frozen primary scheme: equal-frequency
bins at the recovery floor (`MaxEqualBins(1000)`, giving 11 age bins of 1,003 to 1,097 and 10 era
bins of 1,148 to 1,196), membership alignment by Jaccard overlap, and the Mahalanobis centroid
distance. The drift runs are `18b35b34` (age) and `51b335b0` (era); the trajectory runs are
`db8195db` and `5864719e`; the attribution runs are `46208ce1` and `e3281c14`.

## No stratum is an outlier

The drift stage reports each class's shift as a fraction of the between-class separation, the mean
Mahalanobis distance between distinct reference classes, so a shift is read on the scale of the
partition itself. On both axes the two least developmental classes barely move: Moderate challenges
and Social or behavioral sit at 0.22 to 0.24 of the class gap on average. The two developmental
classes move more, Mixed ASD with developmental delay and Broadly affected at 0.33 to 0.44 on
average. None of this clears the null. The smallest permutation p-value is 0.01, the resolution
floor at one hundred permutations ($1/(N+1)$), and no class-by-stratum test survives the
Benjamini-Hochberg correction on either axis.

Where a shift does exceed the whole between-class separation, in two cells on the age axis (Mixed
ASD with developmental delay and Broadly affected, both at the age extremes), the membership tells
the real story. The alignment there is by name only: the Jaccard overlap falls to 0.07 for Broadly
affected in the youngest stratum, so almost none of the reference class's probands are in the
stratum class it was matched to. That is reorganisation, a different set of people wearing the same
label, not a class that moved with its members intact. Five such reorganised cells appear on the
age axis and seven on era, concentrated in Broadly affected on both axes and in Mixed ASD with
developmental delay at the age extremes.

## The class trajectories

:::{figure} /_figures/trajectory_age_at_diagnosis.png
:alt: Each class's centroid path across age-at-diagnosis strata in the pooled discriminant space
:width: 100%
:align: center

Class trajectories across age at diagnosis, from the `analysis trajectory` run. One panel per
class; the focal class is drawn as nested grey coverage contours (50 to 95 per cent of its
members), the other three as their centroid alone. The stratum centroids are coloured from the
youngest-diagnosed stratum to the oldest, a red ring marks a stratum where membership reorganised
(Jaccard below 0.5), and the arrow is the net displacement from the first third of the strata to
the last.
:::

:::{figure} /_figures/trajectory_era.png
:alt: Each class's centroid path across diagnostic-era strata in the pooled discriminant space
:width: 100%
:align: center

The same figure for diagnostic era. The projection is a linear discriminant embedding of the pooled
four-class space (the first two axes hold 54 and 36 per cent of the between-class variance), so
positions and distances are honest, but it is an illustration; the drift claim rests on the
full-dimensional statistics, not on the picture.
:::

How to read it. Each panel fixes on one class. The grey shading is where that class's probands sit
in the pooled discriminant space, drawn as a density so no individual is plotted. The coloured dots
are the same class re-estimated within each stratum, laid out youngest to oldest, so a short tight
run of dots is a class that holds its position and a long coloured track is a class that moves. The
arrow summarises the track as a single young-to-old displacement. A red ring flags a dot where the
stratum class shares few members with the reference, so its position should be read as a relabelling
rather than a move.

What it shows. The two developmental classes trace the longest tracks. Mixed ASD with developmental
delay and Broadly affected move furthest from youngest to oldest on the age axis, and their tracks
carry the red rings at the extreme strata, where the class both moves and reorganises. Moderate
challenges and Social or behavioral keep tighter tracks with no reorganisation. On the era axis the
tracks are shorter and Social or behavioral in particular barely moves, consistent with its weaker
directional result below.

## Movement or noise?

:::{figure} /_figures/roughness.png
:alt: Trajectory step against sampling noise, and net displacement against a shuffle null
:width: 100%
:align: center

Trajectory roughness and directional movement, from both axes' `analysis trajectory` runs. Panel A
plots each class's mean step between adjacent strata against the step that resampling a class of
that size would produce on its own, so a point above the diagonal is a path rougher than sampling
noise. Panel B plots each class's net young-to-old displacement against the 95th percentile of an
ordering-shuffle null, so a bar past its marker is a displacement tied to the axis rather than to
scatter.
:::

How to read it. Panel A separates real step from sampling wobble: a small stratum resampled twice
gives two slightly different centroids even with no drift, and that expected wobble is the
comparison line. A class whose step-to-step movement sits above the line is moving more than its
size alone would explain. Panel B is the directional test: it shuffles the order of the strata
twenty thousand times and asks whether the real young-to-old ordering gives a larger net
displacement than a random ordering, so a class clearing its null marker is drifting with the axis,
not wandering.

What it shows. Every class clears the sampling-noise line on both axes, so the paths are not just
the wobble of small strata; the ratio of step to noise runs from 1.4 to 1.9 on age and 1.8 to 3.4
on era. The directional test is significant for all four classes on age at diagnosis, with the
largest net displacements in the two developmental classes (Mixed ASD with developmental delay at
6.7 against a null of 4.5, Broadly affected at 7.5 against 5.6, both $p < 0.001$). On era it is
significant for three: Moderate challenges, Mixed ASD with developmental delay, and Broadly
affected, while Social or behavioral does not clear its null ($p = 0.086$). Social or behavioral is
also the smoothest path on both axes, the highest step-to-noise ratio with the least directional
pull, a class that stays where it is.

:::{admonition} A pilot, not the confirmatory directional test
:class: caution

The directional test shuffles the order of the observed stratum centroids. It is a cheap check on
the fixed set of fits, so it can say the young-to-old ordering matters, but it does not refit. The
confirmatory version regresses the drift on the continuous axis against the refit permutation null,
and the kernel (local-likelihood) trajectory estimates the same movement as a smooth curve with a
band. Both are separate stages; this page reports the pilot.
:::

## What carries the movement

:::{figure} /_figures/attribution_age_at_diagnosis.png
:alt: Per-class churn across strata and the category decomposition of each class's shift, age axis
:width: 100%
:align: center

Movement attribution for age at diagnosis, from the `analysis attribute` run. Panel A is a heatmap
of churn, the fraction of a class's membership that changed in each stratum (one minus the Jaccard
overlap), with a box on the reorganised cells. Panel B stacks each class's centroid shift by
literature category, pooled across strata, so the bar shows which kinds of feature carry the move.
:::

:::{figure} /_figures/attribution_era.png
:alt: Per-class churn across strata and the category decomposition of each class's shift, era axis
:width: 100%
:align: center

The same decomposition for diagnostic era.
:::

How to read it. Panel A is a map of where each class turns over its membership: a dark cell is a
stratum in which many of that class's probands are assigned differently from the reference, and the
boxed cells are the reorganised ones from the drift stage. Panel B answers what the movement is made
of: each class's total centroid shift is split across the seven literature categories plus the
composite features outside them, so a tall developmental segment means the class moved mainly on its
developmental and milestone items.

What it shows. Mean churn is 0.37 on age and 0.32 on era, and the darkest cells fall at the axis
extremes, matching the reorganised trajectories above. The category split is where the two
developmental classes separate from the rest. For Mixed ASD with developmental delay and Broadly
affected the shift is dominated by the developmental category (pooled contributions of 29.9 and 25.5
on age, far above any other category), so these classes move on exactly the milestone features that
are ascertained earliest and define them. Social or behavioral moves mainly on the anxiety or mood
category, and Moderate challenges on the composite features outside the seven categories. The
movement is not spread evenly across the phenotype; it is concentrated in the developmental and
milestone features, which is the feature-driven middle case the design set out to distinguish from a
genuine break in the partition.

## Who moves

:::{figure} /_figures/attribution_movers_age_at_diagnosis.png
:alt: Per-class mover-versus-stayer feature contrast at each class's peak-churn stratum, age axis
:width: 100%
:align: center

The mover contrast for age at diagnosis, from the same `analysis attribute` run. One panel per
class, at the stratum where that class churns most; each bar is a signed standardised mean
difference between the probands that left the class and the stable core, so the features at the top
are what most distinguish who moved.
:::

:::{figure} /_figures/attribution_movers_era.png
:alt: Per-class mover-versus-stayer feature contrast at each class's peak-churn stratum, era axis
:width: 100%
:align: center

The same contrast for diagnostic era.
:::

How to read it. This figure drops from the class down to the individual features that mark its
movers. For each class it takes the stratum where it churns most, splits that class into the
probands who left and those who stayed, and ranks the features by the standardised difference
between the two groups. A large positive bar is a feature on which the movers score higher than the
stayers, a negative bar the reverse.

What it shows. The recurring discriminators are the developmental milestones and the broad problem
composites. Who moves is marked by the timing of early milestones (age at combined words and
phrases, age toilet-trained) and by the CBCL problem scores (total, conduct, and anxious or
depressed t-scores), all Benjamini-Hochberg significant at each class's peak-churn stratum. The
probands who change class are those whose milestone timing and overall problem burden sit between two
class profiles, the same boundary cases that the [stability analysis](stability-under-refitting) found
move between fits. The movement across strata and the softness of the class boundaries are two views
of one thing: a proband near the edge of a class is assigned by both which stratum and which fit it
falls in.

## What a hundred permutations can and cannot say

This is a pilot. At one hundred permutations the smallest reportable p-value is about 0.01, so the
null is resolved only to two figures and the Benjamini-Hochberg step across forty-odd tests has
little room; a class could drift and still not clear the correction at this count. The frozen
confirmatory value is one thousand permutations, which the [pipeline](../guides/running-the-pipeline)
runs on both axes once the fits are in place. Read at that limit, the pilot supports a consistent
reading rather than a final one: no stratum is an outlier, the classes drift gradually in a
direction tied to the axis, the drift is carried by the developmental classes and features, and the
membership reorganises at the extremes.

Two checks remain before the reading is confirmatory. The binned result has to agree with the
bin-free continuous trend, which the kernel trajectory estimates directly, and the pilot directional
test has to be replaced by the refit permutation null. Both are the pre-registered next steps, and
both are set up to be read the same way as this run, against the same reference and the same
distance. For the mechanics behind each panel, see the guides on
[measuring how far a class drifts](../guides/measuring-class-drift) and
[attributing a class's movement](../guides/attributing-class-movement).
