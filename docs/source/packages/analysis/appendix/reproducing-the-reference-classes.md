# Reproducing the reference classes

:::{admonition} The question
:class: note

Do the four data-driven classes of Litman et al. (2025) reproduce on the SPARK release held
here, and can they be named as the authors named them? A stable, named fit is the fixed reference
the stratified analysis is measured against, which re-estimates the mixture model within strata of
age at diagnosis and diagnostic era.
:::

:::{admonition} The result
:class: tip

The four classes reproduce. The model recovers proportions of about $39$, $29$, $18$, and $15$ per
cent (Social or behavioural, Moderate challenges, Mixed ASD with developmental delay, and Broadly
affected) against the published $37$, $34$, $19$, and $10$; every named-class anchor holds; and the
overall seven-category profile correlates with the published figure at $r = 0.902$, close to the
authors' own SSC-replication value of $r = 0.927$.
:::

:::{figure} /_figures/reproduction.png
:alt: Recovered class signatures against the published figure-1b profile, one panel per class
:width: 100%
:align: center

Recovered class signatures against the values read from figure 1b of Litman et al. (dashed), one
panel per named class, ordered by published class size. Each panel draws two recovered conditions,
the full `2026-03-23` release ($n = 11{,}704$) and the cohort cut back to the records present at the
authors' V9 freeze ($n = 5{,}324$; see
{doc}`subsetting the cohort to the V9 freeze <../appendix/subsetting-to-the-v9-freeze>`), and notes
the two recovered proportions against the published one. The two signatures sit almost on top of
each other in every panel, so the V9 cut leaves the class shapes unchanged and moves only the
proportions.
:::

## What reproduces

The four recovered classes map cleanly onto the four named ones: every anchor holds, and the
proportions ($39$, $29$, $18$, $15$ per cent) are of the same order as the published $37$, $34$,
$19$, $10$. The smallest class, Broadly affected, is the one proportion that stands apart, about
$15$ per cent here against the published $10$.

The overall seven-category profile correlates with the published figure at $r = 0.902$, taken over
the full four-class, seven-category matrix ($28$ points). Per-class correlations, each over only the
seven category points of one class and so coarse, are $r = 0.97$ for Mixed ASD with developmental
delay and $r = 0.85$ for Social or behavioural. Broadly affected and Moderate challenges have
saturated profiles (uniformly high and uniformly low), so their per-class correlation is undefined
and they rest on the anchors instead. The main per-class divergence is that Social or behavioural
shows weaker social-communication and restricted-or-repetitive enrichment here than in the paper.

## How solid it is

Resampling the $11{,}704$ probands with replacement (the fitted labels held fixed) over $500$
resamples puts a $95$ per cent interval of $[0.893, 0.916]$ on the overall $r$, so the reproduction
is precise to about $\pm 0.01$ from sampling alone. Holding the fit fixed, this captures the
sampling variability in the signature, not the variability from refitting the model, and the
figure-read resolution of the target is a further, separate source of uncertainty.

The seven-category profile is itself stable, which [stability under refitting](stability-under-refitting)
shows directly: across $200$ re-initialisations and $50$ random halves of the cohort, the profile
reproduces against the reference at about $0.91$ to $0.92$, and no fit ever collapsed a class. The
reproduction is therefore not an artefact of one fit or one sample.

## The V9 freeze cut

The authors fit on SPARK v9 ($n = 5{,}392$); the release held here is later and broader ($n =
11{,}704$). Cutting the cohort back to the records present at the V9 freeze ($n = 5{,}324$, close to
the authors' count) shifts the two largest classes toward the published split, Social or behavioural
from $39$ to $34$ per cent and Moderate challenges from $29$ to $31$, against the published $37$ and
$34$, while leaving the smallest class at $15$ and the recovered signatures unchanged. A size-matched
random subsample of the full release does not reproduce those shifts, so they track the V9 records
rather than the smaller sample, and the inflated smallest class is unmoved by the cut.

## How the reproduction is built

:::{dropdown} Feature typing

StepMix fits a Gaussian density to each continuous feature, a Bernoulli to each binary feature, and
a multinomial to each categorical feature, so each feature has to be typed correctly. The typing is
derived three ways and reconciled:

1. from the data dictionary, through `dscat`: a calculated score or a dropdown age coding is
   continuous; a radio item with exactly two coded levels is binary, otherwise categorical;
2. from the authors' released typing files, which carry their own assignment;
3. from the observed number of distinct values in the cohort, as a cross-check.

The dictionary inference and the released typing agree on $237$ of the $238$ features. The one
disagreement is `repeat_grade`, a yes/no item the authors placed with the continuous features rather
than the binary ones. The run defers to the released typing there, since the aim is to reproduce the
authors' model, and records the disagreement in a reconciliation report alongside the cohort matrix.
The reconciled typing is $38$ continuous, $33$ binary, and $167$ categorical features.
:::

:::{dropdown} The mixture model

The `fit` stage trains a StepMix general finite mixture model with four components: a one-step joint
estimation, the measurement densities set by the reconciled typing, and sex and age at evaluation as
structural covariates. The $200$ random restarts are delegated to StepMix, as in the released code,
and the best restart by log-likelihood is kept. The released rounding of the feature matrix to
integers is applied at fit time, so the cached cohort matrix stays unrounded while the model sees the
values the authors fit on. The fit takes around ten minutes and predicts a hard class label per
proband.
:::

:::{dropdown} Per-class enrichment and the class signature

A feature is *enriched* in a class when probands there carry it more often, or score higher on it,
than the rest of the cohort, and *depleted* when they carry or score less. Each feature is tested for
this in every class against the rest, in both directions: a binomial test for binary features and a
Welch $t$-test for the others, Benjamini-Hochberg corrected within each class and direction. A
corrected $p$-value below $0.05$ marks the feature enriched or depleted. The $24$ reverse-coded SCQ
social items have their direction flipped, and the features are summarised into the seven
literature-defined categories (anxiety or mood, attention, disruptive behaviour, self-injury, social
or communication, restricted or repetitive, and developmental) as the signed proportion enriched
minus depleted. That seven-category vector is each class's *signature*, the currency every later
investigation compares against.
:::

:::{dropdown} Naming the classes

The authors' three routes to naming a recovered class are mostly closed here: there are no shared
probands to match on, and no reference model was released. The classes are therefore aligned on the
published class signatures, the same currency the authors used to declare replication in the SSC.

The primary mechanism is the set of named-class anchors, the substantive characteristics that define
each class:

- the highest-difficulty class overall, which is also the smallest, is Broadly affected;
- the most developmental of the rest is Mixed ASD with developmental delay;
- the largest of the rest, high on core, attention, and anxiety with no developmental delay, is
  Social or behavioural;
- the uniformly lowest is Moderate challenges.

The assignment is cross-checked for mutual consistency: Broadly affected should be both the highest
overall and the smallest, Social or behavioural should be the largest, and so on. The published
seven-category signature, read from the paper's figure, gives a profile correlation for each class
and overall as a second measure.
:::

:::{dropdown} The values read from figure 1b

The published seven-category signatures are not released as a numeric table, so they are read from
figure 1b of Litman et al. (2025), the per-category proportion-and-direction plot, at the figure's
resolution. The signed values run from $-1$ (depleted across that category) to $+1$ (enriched), in
the seven-category order:

| Class | anxiety/mood | attention | disruptive | self-injury | social/comm | restricted/rep | developmental |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Social or behavioural | +1.0 | +1.0 | +0.95 | +0.50 | +0.50 | +0.45 | -0.90 |
| Moderate challenges | -0.90 | -0.95 | -1.0 | -0.95 | -1.0 | -1.0 | -1.0 |
| Mixed ASD with developmental delay | -0.90 | -0.45 | -0.65 | -0.10 | +0.10 | +0.05 | +0.45 |
| Broadly affected | +1.0 | +1.0 | +1.0 | +1.0 | +1.0 | +1.0 | +1.0 |

Broadly affected sits near $+1$ and Moderate challenges near $-1$ across every category, so both
profiles are saturated and their per-class correlation is undefined; they rest on the anchors
instead. The published class proportions the anchors use, also from the paper, are about $37$, $34$,
$19$, and $10$ per cent for Social or behavioural, Moderate challenges, Mixed ASD with developmental
delay, and Broadly affected. Every value here is read to the figure's resolution, so it is
approximate; numeric supplementary tables, if obtained, would replace them.
:::

## Caveats

The reproduction is benchmarked on the published profile and proportions, not on a per-proband
agreement, for two reasons. The authors' SPARK v9 release and their per-proband labels are not
available, so an exact reproduction of their cohort and a label-level comparison are out of reach.
And the published seven-category profile is read from the paper's figure at the figure's resolution,
not from a numeric supplementary table.

Reproducibility is necessary for the stratified test but is not the same as validity. A partition can
reproduce across samples and still reflect parent-reported, deficit-framed measurement rather than a
biological kind, a distinction the genetics arm and the construct-validity checks speak to and this
investigation does not.
