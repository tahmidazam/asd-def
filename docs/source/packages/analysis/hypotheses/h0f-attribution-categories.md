# $H_0^F$: is the drift spread evenly across the seven phenotype categories?

:::{admonition} Definition
:class: note

$H_0^F$ (category attribution). Null: any drift is distributed uniformly across the
{term}`seven phenotype categories` (anxiety or mood, attention, disruptive behaviour, self-injury,
social or communication, restricted or repetitive, and developmental), so no category carries more
than its share of a class's movement. Alternative: the drift is concentrated in the
developmental-history, milestone, and intellectual-disability-linked features, and therefore in the
two developmentally defined classes (Mixed ASD with developmental delay; Broadly affected),
consistent with feature-driven movement rather than a reorganisation of the four-way partition.
Estimand: the per-category share of the total class drift.

This test is conditional. It is read only where an invariance null is rejected, so it inherits the
verdict of {doc}`$H_0^A$ <../hypotheses/h0a-invariance>`: the class {term}`profile`s are not
invariant to {term}`diagnostic era` or {term}`age at diagnosis`, which is what makes a decomposition
of the drift meaningful.
:::

:::{admonition} Status
:class: tip

Rejected. The drift is concentrated, not uniform. It sits in the developmental-history and milestone
features, and so in the two developmentally defined classes: Mixed ASD with developmental delay and
Broadly affected move mainly on the developmental category (pooled category contributions of $29.9$
and $25.5$ on age at diagnosis, far above any other category), while Social or behavioral moves on
anxiety or mood and Moderate challenges on the composite features outside the seven. Because this is
a per-feature decomposition of an already-measured distance, the read is a count of features that
clear {term}`false discovery rate` control rather than a single p-value: the recurring
discriminators are the developmental milestones (age at combined words and phrases, age
toilet-trained) and the broad CBCL problem scores, which clear the Benjamini-Hochberg step at each
class's peak-churn stratum.
:::

## Method

The drift stage reports how far each class moved as one number, a shift scaled by the
{term}`between-class separation`. That number says a class changed; it does not say which features
carry the change. The `analysis attribute` stage opens the number up. It re-reads the stored
reference and stratum fits and refits nothing, so the decomposition is a cheap consumer of work
already on disk.

The default decomposition is the Mahalanobis split, which matches the default drift distance. The
squared {term}`Mahalanobis distance` is a quadratic form in the centroid shift and the shrunk
within-class precision, and it expands term by term into per-feature contributions that sum back to
the squared distance, so a coordinated shift across a correlated block of symptoms is charged to the
block once rather than to each feature. A contribution can be negative, when a feature's shift
offsets a correlated block. The standardised split, squared shift in pooled-standard-deviation units,
is the covariance-blind cross-check: a feature that ranks high there but low under the Mahalanobis
split is one whose apparent movement is shared with a correlated block rather than its own. Each
per-feature contribution is reported with its signed shift and its literature category, and the
contributions are summed into the {term}`seven phenotype categories`, with every feature kept in the
totals (blanks and unlisted features under an `unmapped` bucket) so the category totals sum to the
same distance the per-feature contributions do. Uncertainty is the per-feature {term}`confidence band`
of the {term}`family-clustered bootstrap`, and the per-feature tests are {term}`false discovery
rate`-controlled.

A second reading works at the level of the probands. A stratum is a subset of the pooled cohort, so
every proband carries both labellings, the reference class and the stratum-fit class, and a class can
both shed members and absorb them. The churn, one minus the Jaccard overlap, is the honest count of
how much the membership turned over. Contrasting the churned probands with the stable core says what
marks the probands whose membership changed: the default is a standardised mean difference (Cohen's
$d$) per feature with a Welch test and Benjamini-Hochberg control. The full method, both the
decomposition and the mover contrast, is set out in
{doc}`attributing a class's movement <../guides/attributing-class-movement>`.

## Experimental design decisions

- Category totals keep every feature. Unmapped and composite features go into their own buckets, so
  the seven category totals plus the remainder sum to exactly the drift distance and nothing is lost
  from the accounting.
- Mahalanobis split as the default, standardised split as the cross-check. The default charges a
  correlated block of symptoms once; the standardised split, covariance-blind, exposes movement that
  is borrowed from a block rather than a feature's own.
- Movers counted as churn, not leavers only. A class that keeps its members but pulls in new ones has
  still drifted, so the count folds in joiners as well as leavers.
- The contrast runs over the clustered feature frame, the one lever that fixes what the movement is
  explained by. The held-out SPARK variables and, later, the genetic scores present the same
  interface, so the question extends to them without new attribution code.
- Descriptive, not confirmatory. The decomposition and the contrast interpret an already-measured
  drift rather than deciding whether it is real, so they sit outside the pre-registered confirmatory
  freeze.

## Results

The category split separates the developmental classes from the rest cleanly. Mixed ASD with
developmental delay and Broadly affected move mainly on the developmental category (pooled
contributions of $29.9$ and $25.5$ on age at diagnosis, far above any other category for those
classes), Social or behavioral moves on anxiety or mood, and Moderate challenges moves on the
composite features outside the seven. The movement is concentrated in the developmental and milestone
features, not spread across the phenotype. Membership turnover is highest at the extremes of each
axis: mean churn is $0.37$ on age at diagnosis and $0.32$ on diagnostic era, darkest in the boundary
strata.

:::{figure} /_figures/attribution_age_at_diagnosis.png
:alt: Per-class membership churn across the age-at-diagnosis strata and the category composition of each class shift.
:width: 100%
:align: center

Movement attribution along age at diagnosis. Panel A is a heatmap of churn, one minus the Jaccard
overlap, per class and stratum, boxed on the reorganised cells; panel B stacks each class's centroid
shift by literature category, pooled across strata. How to read it: a dark cell in panel A is a
stratum where the membership turned over, and a tall developmental segment in panel B means the class
moved on its milestone items. The two developmental classes stack almost entirely on the
developmental category, which is the concentration that rejects the uniform null. Rendered by
{py:mod}`figures.attribution` (`figures attribution --axis age_at_diagnosis`).
:::

:::{figure} /_figures/attribution_era.png
:alt: Per-class membership churn across the diagnostic-era strata and the category composition of each class shift.
:width: 100%
:align: center

The same decomposition along diagnostic era. How to read it as above: the category composition of
each class's shift, read against its churn across the era strata. The developmental concentration
holds on this axis too, with the same two classes carrying their shift on the developmental features.
Rendered by {py:mod}`figures.attribution` (`figures attribution --axis era`).
:::

The mover contrast names the probands behind the churn. At each class's peak-churn stratum the
recurring discriminators are the developmental milestones (age at combined words and phrases, age
toilet-trained) and the broad CBCL problem scores (total, conduct, anxious or depressed), all
Benjamini-Hochberg significant. The probands who change class are the boundary cases whose milestone
timing and problem burden sit between two profiles.

:::{figure} /_figures/attribution_movers_age_at_diagnosis.png
:alt: Per-class mover-versus-stayer feature contrast at each class's peak-churn stratum along age at diagnosis.
:width: 100%
:align: center

The mover contrast along age at diagnosis. One panel per class, at its peak-churn stratum; each bar
is a signed standardised mean difference between the probands who changed class and those who stayed.
How to read it: a large positive bar is a feature on which the movers score higher, and the leading
bars are the developmental milestones and broad problem scores that mark the boundary cases. Rendered
by {py:mod}`figures.attribution` (`figures attribution --axis age_at_diagnosis`).
:::

:::{figure} /_figures/attribution_movers_era.png
:alt: Per-class mover-versus-stayer feature contrast at each class's peak-churn stratum along diagnostic era.
:width: 100%
:align: center

The same contrast for diagnostic era. How to read it as above: the features that most separate the
probands who changed class from the stable core at each class's peak-churn era stratum. Rendered by
{py:mod}`figures.attribution` (`figures attribution --axis era`).
:::

## Handling the null

$H_0^F$ has no single omnibus statistic, and by design. The estimand is the per-category share of a
class's drift, so the decomposition splits one already-measured distance into per-feature
contributions that sum back to it, then rolls them up to the seven categories. The uncertainty
attaches to each contribution as its own {term}`family-clustered bootstrap` interval, and the tests
run per feature across the four classes and the features under {term}`false discovery rate` control at
$q < 0.05$. The output is therefore a count of features, and of categories, that clear the threshold,
not a single p-value: an omnibus test would collapse the very concentration the hypothesis is about.

Read that way, the null is rejected. A uniform drift would spread the contributions evenly across the
seven categories; instead the developmental category dominates the two developmentally defined
classes, and the surviving per-feature discriminators are the developmental milestones and the broad
problem scores rather than a flat scatter across the phenotype. The concentration is the finding.

## Discussion

The drift is feature-driven. It moves on developmental history and milestones, and so on the two
classes those features define, rather than reorganising the four-way partition evenly. This is the
middle case the design anticipated: a gradual, feature-carried trajectory rather than fixed classes
or a broken partition. The read has limits. The decomposition is descriptive, interpreting a drift
that {doc}`$H_0^A$ <../hypotheses/h0a-invariance>` already established rather than testing it afresh,
so it inherits that verdict and its conditioning on the pooled fit. It also inherits the illustrative
status of the discriminant projections that draw the trajectories: the claim rests on the
full-dimensional per-feature contributions, and the plane figures are a picture of them.

## See also

- {doc}`Attributing a class's movement <../guides/attributing-class-movement>`, the method behind the
  decomposition and the mover contrast.
- {doc}`Are the class profiles invariant? <../hypotheses/h0a-invariance>` ($H_0^A$), the invariance
  null this test is conditional on.
- {doc}`Measuring how far a class drifts <../guides/measuring-class-drift>`, the drift distance the
  decomposition splits.
- {doc}`Tracking the classes across the strata <../archive/tracking-the-classes-across-strata>`,
  the stratified refit run these attribution figures were first drawn on.
- {doc}`The Python API <../reference>` for the `attribute` stage.
