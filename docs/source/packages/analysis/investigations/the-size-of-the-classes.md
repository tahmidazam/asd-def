# H0B: the size of the classes

:::{admonition} The question
:class: note

The class profiles are not invariant to diagnostic era or age at diagnosis, and the movement is
large and, on most classes, directional. Those reads are all about shape: how each class's profile
moves. This page asks a separate question about size. Do the four classes make up the same share of
the diagnosed population across the axis, or does a class grow while another shrinks? A class can
hold its shape while its size moves, or move its shape while its size holds, so prevalence is read
on its own. The classes are frozen at the measurement-only reference fit and only their mixing
proportions are regressed on the axis, so nothing is re-estimated.
:::

:::{admonition} The result
:class: tip

Every class's proportion trends along both axes. Across age at diagnosis the shift is stark: the
Mixed ASD with developmental delay class falls from about a third of the earliest-diagnosed to
almost none of the latest, while Social or behavioral rises from about a quarter to three fifths.
Across diagnostic era the same pattern is milder: the developmental class shrinks and the social
class grows as diagnosis year advances. At the full sample the omnibus test rejects a
constant-proportion null by a wide margin on both axes, so the reading is the size and shape of each
class's trend, not the binary reject. The age trends survive adjustment for measurement timing and
sex; on era, the Social or behavioral rise is nullified by that adjustment while the developmental
decline and the Broadly affected rise persist.
:::

## Size is not shape

The invariance reads measure where each class's profile sits along the axis. Prevalence is a
different quantity: the fraction of the cohort each class holds, read as a function of the axis. The
two can move independently. A per-stratum refit conflates them, because re-estimating the mixture
inside a stratum lets the classes change both their profiles and their sizes at once, and reports a
single "proportion" that mixes the two. Freezing the classes separates the questions: with the
profiles fixed, a trend in the mixing proportion is prevalence drift alone.

So the estimand is the mixing proportion $\pi_k$ as a function of the axis. The frozen reference
gives every proband a posterior over the four classes; the stage regresses class membership on the
axis using those posteriors, with no refit.

## Reading proportions without a refit

The direct approach regresses each proband's hard class label on the axis. It is transparent but
biased. The hard label is a noisy reading of the true latent class, because the posterior is not a
point mass, so a share of the labels are wrong. Regressing a mislabelled outcome on the axis pulls a
real slope towards zero and can invent one where there is none, the classify-analyse bias of
three-step latent-class analysis.

The primary read corrects it. The modal assignment is treated as a single categorical measurement
of the true class whose error probabilities are fixed at the confusion matrix of the frozen
posteriors, and the structural model, a multinomial logit of the true class on the axis, is fitted
with that measurement model held fixed. The correction removes the attenuation without touching the
mixture. A naive multinomial logit on the hard labels runs beside it as an uncorrected cross-check.
The mechanics, and the correctness gates (including that the correction de-attenuates a known slope
where the naive one does not), are in the guide on
{doc}`measuring prevalence drift <../guides/measuring-prevalence-drift>`. Uncertainty is a
family-clustered bootstrap resampling SPARK families, the same convention the effect-size reads use,
so each slope and the proportion curve carry a percentile interval and each slope a two-sided
bootstrap $p$-value, controlled for multiplicity across the four classes with Benjamini-Hochberg.

On this cohort the corrected and naive estimates agree to three or four decimal places, so the
reference posteriors are confident, the bias is small, and the naive read corroborates rather than
contradicts. The correction still matters as a guarantee: it is what lets the read stand without
assuming the posteriors are error-free.

The runs are on SPARK `2026-03-23` (11,704 probands), against the measurement-only reference fit
`41ab0e38`, at 500 bootstrap replicates: era run `b0125247` (98.3 per cent of probands have a
diagnosis year) and age run `7ff81e49` (98.4 per cent have an age at diagnosis). The pooled
proportions, the classes' shares over the whole cohort, are 0.39 for Social or behavioral, 0.25 for
Moderate challenges, 0.21 for Broadly affected, and 0.16 for Mixed ASD with developmental delay.

## The proportion curves

:::{figure} /_figures/prevalence_age_at_diagnosis.png
:alt: Predicted proportion of each class as a function of age at diagnosis, one panel per class, with bootstrap bands
:width: 100%
:align: center

Class prevalence along age at diagnosis. Each panel is one class: the solid line is the corrected
proportion curve, the shaded band its family-clustered bootstrap interval, the thin dashed line the
naive hard-label cross-check, and the dotted horizontal line the pooled (axis-free) proportion the
class trends away from. The title carries the per-year log-odds slope and its odds ratio.
:::

:::{figure} /_figures/prevalence_era.png
:alt: Predicted proportion of each class as a function of diagnostic era, one panel per class, with bootstrap bands
:width: 100%
:align: center

The same read for diagnostic era.
:::

### How to read it

Each curve is the fraction of the diagnosed population the class holds at that point on the axis. A
curve above its dotted pooled line is a class over-represented there; below it, under-represented.
An upward slope is a class growing along the axis, a downward slope a class shrinking. Where the
band clears the pooled line the class is off its overall share; the slope's bootstrap interval,
reported in the guide's tables, is what calls the trend significant. The dashed naive curve sitting
on the solid one is the visible sign that the classification-error correction changes little here.

### What it shows

Across age at diagnosis all four classes trend, and the developmental split is the headline. Mixed
ASD with developmental delay falls from about 0.35 among those diagnosed near age two to under 0.01
among those diagnosed in the early teens, a per-year log-odds slope of $-0.35$ (odds ratio 0.71).
Social or behavioral runs the other way, from about 0.24 to about 0.60 (slope $+0.13$). Broadly
affected rises gently and Moderate challenges declines. This is the expected shape of a diagnosed
population: the developmental class is recognised early, so it dominates the young-diagnosed and
thins out among the late-diagnosed, where the social presentations take over.

Across diagnostic era the same two classes lead, more gently. Mixed ASD with developmental delay
falls from about 0.26 among the earliest-diagnosed years to about 0.09 in the recent years (slope
$-0.08$), while Social or behavioral, Broadly affected and, against them, Moderate challenges all
shift by smaller amounts. The omnibus likelihood-ratio test of class on the axis against a constant
gives $\chi^2 = 365$ on 3 degrees of freedom for era and $\chi^2 = 1204$ for age, so a
constant-proportion null is rejected outright and the interpretation rests on the per-class curves
rather than the reject.

## The composition

:::{figure} /_figures/prevalence_stacked_age_at_diagnosis.png
:alt: The four class proportions stacked to one across age at diagnosis
:width: 100%
:align: center

The same age result as a composition. The four corrected proportions sum to one at every age, so the
figure reads as a single shifting mixture: the Social or behavioral base grows as the developmental
band at the top thins to almost nothing.
:::

:::{figure} /_figures/prevalence_stacked_era.png
:alt: The four class proportions stacked to one across diagnostic era
:width: 100%
:align: center

The same for diagnostic era, a milder version of the same shift.
:::

Because the proportions sum to one, the compositional view makes the trade explicit: the population
diagnosed later, whether later in life or in a more recent year, is progressively less
developmental and more social in its makeup.

## The DSM-5 boundary

Diagnostic era carries a secondary read, a pre and post-2013 contrast that marks the DSM-5 revision.
Every class shifts across it in the same sense as its continuous trend and by a larger step: the
odds of the Broadly affected class are about 1.6 times higher after 2013 and of the Social or
behavioral class about 1.5 times, while the odds of the Moderate challenges and Mixed ASD with
developmental delay classes are about 0.68 and 0.58 times their pre-2013 values. The boundary is a
descriptor overlaid on a continuous trend, not a claim that the change happens at the boundary.

## Timing, or the population?

A prevalence trend across diagnostic era could reflect a change in who is diagnosed, or an artefact
of when they were measured relative to diagnosis. The adjusted read fits the corrected axis slope net
of sex, the measurement-to-diagnosis lag, and age at evaluation. On age at diagnosis every class
keeps its trend under adjustment, the developmental decline and the social rise included, so the age
composition shift is more than a measurement-timing effect. On diagnostic era the picture divides:
the Broadly affected rise and the Mixed ASD with developmental delay decline survive, but the Social
or behavioral era rise is nullified, its adjusted slope near zero with an interval covering it. So
the growing share of the social class across recent years tracks measurement timing rather than a
change in the diagnosed population, a prevalence-side echo of the timing question the attribution
work pursues for the profiles.

## Limits

The read is conditional on the pooled fit. It freezes the reference posteriors and regresses on the
axis, so it measures how the fixed classes' sizes shift, not a re-estimated partition. At this sample
size the omnibus test rejects by construction, which is why the interpretation is the effect size and
the curve, not the $p$-value. Earlier per-stratum refits hinted that a class's proportion moved with
the axis; where that hint does not match a frozen-class curve here, the refit's apparent size change
was profile drift reassigning members between re-estimated classes rather than a genuine change in
share, and the two questions come apart exactly as the frozen read is built to make them. What this
investigation settles is firm: the diagnosed autism population shifts from developmental towards
social presentations as diagnosis moves later in life and, more mildly, into more recent years, and
on the age axis that shift is not an artefact of when the measurements were taken.
