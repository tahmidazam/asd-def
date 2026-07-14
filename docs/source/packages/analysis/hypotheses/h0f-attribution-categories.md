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

Rejected. The drift is concentrated, not uniform, and it is concentrated differently by class and by
axis. Mixed ASD with developmental delay moves on the developmental category above all else ($43\%$
of its drift along diagnostic era, $64\%$ along age at diagnosis), carried by the motor and language
milestones. The other three classes move on social or communication along era ($53$ to $59\%$) and on
the internalizing categories, anxiety or mood and somatic, along age at diagnosis. A uniform drift
would spread the shares evenly across the categories; instead one category dominates each class, so
the null is rejected.
:::

## Method

The drift stage reports how far each class moved as one number, a shift scaled by the
{term}`between-class separation`. That number says a class changed; it does not say which features
carry the change. The category decomposition opens it up, and it does so from the single cached fit,
with no refit.

The decomposition is one read of the {doc}`block-attribution engine <../reference>`
({py:mod}`analysis.blocks`). The engine treats a named set of columns as a block: an internal block
is a subset of the reference's own features, so its sub-displacement is a slice of the drift vector
the {doc}`effect-size recast <../hypotheses/h0a-invariance>` already computed from the frozen
responsibilities. The seven author symptom categories are the internal block here. Each feature's
displacement is standardised by its pooled standard deviation, and the squared standardised
displacements of a category sum, over the disjoint categories, to the whole-class squared distance,
so a category's share is additive and the shares over the categories add to one. This is the same
additive decomposition the {doc}`referent split <../hypotheses/h0g-attribution-referent>` ($H_0^G$)
reports, applied to the symptom categories rather than the instrument referents.

Uncertainty is the family-clustered bootstrap of the recast: the per-feature displacements carry a
{term}`confidence band` from resampling whole SPARK families, and the per-feature tests are
{term}`false discovery rate`-controlled at $q < 0.05$. Because the estimand is a share of an
already-measured distance rather than a single statistic, the read is a count of features, and of
categories, that clear the threshold, not one $p$-value.

The decomposition inherits the reference fit and the axis, and it refits nothing. It is a consumer
of the cached `invariance-trajectory` run: the per-category magnitudes it splits are the
`grain_magnitude_<axis>.parquet` category grains, and the per-feature displacements are
`feature_displacement_<axis>.parquet`.

## Experimental design decisions

- The categories tile the features. Every feature is charged to exactly one category (a feature
  without an author category falls to a composite bucket), so the category shares plus the composite
  remainder sum to the whole-class distance and nothing is lost from the accounting.
- Litman's seven, plus three the paper sets aside. The feature-to-category map Litman et al. shared
  resolves the 238 features into ten categories: their seven author symptom categories and three
  further Child Behavior Checklist problem scales (somatic, thought problems, other problems). Their
  own category signature uses only the seven (their code's `features_to_visualize`), so the figure
  leads with those seven and marks the three additional scales off with a divider. The three are kept
  in view rather than folded away because one of them, somatic, carries a real part of the age drift
  for the non-developmental classes (the dizziness and tiredness items), which a seven-category-only
  read would hide inside the accounting.
- Additive squared-magnitude shares, not a size-fair contrast. $H_0^F$ asks how the drift is
  distributed across the categories, so the additive share is the natural read; the size-fair
  root-mean-square contrast that {doc}`$H_0^G$ <../hypotheses/h0g-attribution-referent>` uses is for
  a two-way comparison where the grains differ sharply in feature count.
- One fit, no refit. The decomposition reads the frozen-responsibility displacement of $H_0^A$, so
  it measures the drift of the fixed four-class partition rather than re-estimating classes within a
  stratum. The earlier refit-based reading (a per-stratum re-estimation, its membership churn, and a
  mover-versus-stayer contrast) is kept for the record in
  {doc}`the refit pilot <../archive/tracking-the-classes-across-strata>`; a frozen fit relabels no
  proband, so churn and movers have no counterpart here.
- Descriptive, not confirmatory. The decomposition interprets an already-measured drift rather than
  deciding whether it is real, so it sits outside the pre-registered confirmatory freeze.

## Results

The category split separates the developmental class from the rest cleanly, and it separates the two
timing axes. Along diagnostic era, three of the four classes move on social or communication (Moderate
challenges $59\%$, Social or behavioral $53\%$, Broadly affected $43\%$), while Mixed ASD with
developmental delay moves on the developmental category ($43\%$) with social or communication second
($24\%$). Along age at diagnosis the developmental concentration sharpens: Mixed ASD with developmental
delay puts $64\%$ of its drift on the developmental category, and the other three classes shift to the
internalizing categories, anxiety or mood and somatic, with restricted or repetitive alongside.

:::{figure} /_figures/category_decomposition.png
:alt: Category-share heatmaps of each class's drift along era and age, and the leading features of the age drift per class.
:width: 100%
:align: center

The category decomposition. Panels A and B are heatmaps of each class's drift split into category
shares, along diagnostic era and age at diagnosis; a darker cell is a category that carries more of
that class's movement. The dashed divider separates Litman's seven author categories (left) from the
three additional Child Behavior Checklist problem scales the paper sets aside (somatic, thought
problems, other problems). Panel C names the leading features of the age drift for each class, the
largest false-discovery-rate survivors, coloured by category and each panel on its own scale. How to
read it: the single dark cell in each heatmap row is the concentration that rejects the uniform null,
and the lollipops behind it are the developmental milestones (age crawled, age sat without support,
age at combined words and phrases) for Mixed ASD with developmental delay and the internalizing items
(feels worthless, unhappy, overtired) for the others. Rendered by
{py:mod}`figures.category_decomposition` (`figures category-decomposition`).
:::

The leading features name the movement. For Mixed ASD with developmental delay along age, the
developmental milestones dominate: the ages at crawling and sitting without support move furthest,
with the ages at combined words and phrases behind them. For the other classes the age drift is an
internalizing shift, led by the Child Behavior Checklist items for feeling worthless, unhappiness, and
tiredness. Along era the recurring discriminators are the social-communication items of the SCQ.

The summary figure names the leading features; the dense matrix below shows them all. Of the 238
features, 227 clear false-discovery-rate control in at least one class on at least one axis, so the
matrix is close to complete.

:::{figure} /_figures/dense_features.png
:alt: A heatmap of every significant feature's signed displacement, per class and axis, grouped by category.
:width: 100%
:align: center

Every significant feature. Each row is a feature clearing false-discovery-rate control in at least one
class on at least one axis, grouped by category (the colour sidebar); each column is a class, along
diagnostic era (left) and age at diagnosis (right); the cell is the signed separation-standardised
displacement, red for a rise and blue for a fall, with non-significant cells left faint. How to read
it: the age block carries far more colour than the era block, the developmental rows light up only in
Mixed ASD with developmental delay, the restricted-or-repetitive rows fall (blue) with age across the
non-developmental classes, and the anxiety-or-mood and somatic rows rise (red) with age, the
internalizing shift the summary figure reports. Rendered by {py:mod}`figures.dense_features`
(`figures dense-features`).
:::

## Handling the null

$H_0^F$ has no single omnibus statistic, and by design. The estimand is the per-category share of a
class's drift, so the decomposition splits one already-measured distance into per-feature
contributions that sum back to it, then rolls them up to the categories. The uncertainty attaches to
each contribution as its own {term}`family-clustered bootstrap` interval, and the tests run per
feature across the four classes under {term}`false discovery rate` control at $q < 0.05$. The output
is therefore a count of features, and of categories, that clear the threshold, not a single
$p$-value: an omnibus test would collapse the very concentration the hypothesis is about.

Read that way, the null is rejected. A uniform drift would spread the contributions evenly across the
categories; instead one category dominates each class, and the surviving per-feature discriminators
are the developmental milestones and the internalizing items rather than a flat scatter across the
phenotype. The concentration is the finding.

## Discussion

The drift is feature-driven. It moves on developmental history and milestones for the developmental
class, and on the internalizing and social-communication items for the others, rather than
reorganising the four-way partition evenly. This is the middle case the design anticipated: a gradual,
feature-carried trajectory rather than fixed classes or a broken partition. The read has limits. The
decomposition is descriptive, interpreting a drift that {doc}`$H_0^A$ <../hypotheses/h0a-invariance>`
already established rather than testing it afresh, so it inherits that verdict and its conditioning on
the pooled fit. It also inherits the illustrative status of the discriminant projections that draw the
trajectories: the claim rests on the full-dimensional per-feature contributions, and the plane figures
are a picture of them.

## Removing a category: the causal question

A natural next question is causal: remove a category's features and does the drift disappear? The
decomposition answers the additive form of it directly. Because the drift magnitude is a Euclidean
norm over features, dropping a category's columns leaves the norm over the rest, which for a category
carrying a fraction $s$ of the squared drift is $\sqrt{1 - s}$ of the magnitude. Drop the
developmental category from Mixed ASD with developmental delay along age at diagnosis, where it carries
$64\%$, and about $60\%$ of the drift remains: the drift shrinks, it does not vanish, because the other
categories move too.

The deeper form, whether one category's shift is *downstream* of another (is the internalizing rise
carried by the developmental shift?), is not identifiable from the drift. Both shifts are age effects:
they co-move because both track age at diagnosis, so residualising one category on another removes
their shared age variance whether or not one causes the other. A check on synthetic data confirms it:
two feature blocks that both drift with the axis but are independent given the axis still show a large
apparent mediation, all of it the shared axis. The block engine's conditioning mode
({func}`~analysis.blocks.conditioning_shrinkage`) is therefore reserved for a low-dimensional
*confounder*, not another symptom block.

The genuine causal test in this analysis is {doc}`$H_0^H$ <../hypotheses/h0h-attribution-timing>`
($H_0^H$): it conditions the era drift on the measurement-to-diagnosis lag and age at evaluation,
the timing confounder, and asks whether the drift survives. A drift that persists net of the timing is
a change in the diagnosed population rather than an artefact of when the child was rated, which is the
"remove it and see if the drift disappears" test that is well posed here.

## See also

- {doc}`The block-attribution engine <../reference>` ({py:mod}`analysis.blocks`), the engine this
  decomposition is one read of.
- {doc}`Are the class profiles invariant? <../hypotheses/h0a-invariance>` ($H_0^A$), the invariance
  null this test is conditional on.
- {doc}`Is the era drift measurement or population? <../hypotheses/h0g-attribution-referent>`
  ($H_0^G$), the sibling decomposition by instrument referent.
- {doc}`Measuring how far a class drifts <../guides/measuring-class-drift>`, the drift distance the
  decomposition splits.
- {doc}`The refit pilot <../archive/tracking-the-classes-across-strata>`, the archived refit-based
  reading, its membership churn, and the mover-versus-stayer contrast.
