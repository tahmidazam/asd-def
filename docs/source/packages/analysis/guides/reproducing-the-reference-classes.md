# Reproducing the reference classes

The first goal is to recover the four data-driven autism classes of Litman et al. (2025) on
the SPARK release held here, and to name them as the authors did. That named fit is the
fixed reference every later comparison is measured against, so it has to reproduce the
published solution and align to the published classes cleanly. This guide describes the
feature typing, the model, the per-class enrichment, and how the classes are named, then
reports the reproduction result and its caveats.

## Feature typing

StepMix fits a Gaussian density to each continuous feature, a Bernoulli to each binary
feature, and a multinomial to each categorical feature, so each feature has to be typed
correctly. The typing is derived three ways and reconciled:

1. from the data dictionary, through `dscat`: a calculated score or a dropdown age coding is
   continuous; a radio item with exactly two coded levels is binary, otherwise categorical;
2. from the authors' released typing files, which carry their own assignment;
3. from the observed number of distinct values in the cohort, as a cross-check.

The dictionary inference and the released typing agree on 237 of the 238 features. The one
disagreement is `repeat_grade`, a yes/no item the authors placed with the continuous
features rather than the binary ones. The run defers to the released typing there, since the
aim is to reproduce the authors' model, and records the disagreement in a reconciliation
report alongside the cohort matrix. The reconciled typing is 38 continuous, 33 binary, and
167 categorical features.

## The model

The `fit` stage trains a StepMix general finite mixture model with four components: a
one-step joint estimation, the measurement densities set by the reconciled typing, and sex
and age at evaluation as structural covariates. The 200 random restarts are delegated to
StepMix, as in the released code, and the best restart by log-likelihood is kept. The
released rounding of the feature matrix to integers is applied at fit time, so the cached
cohort matrix stays unrounded while the model sees the values the authors fit on. The fit
takes around ten minutes and predicts a hard class label per proband.

## Per-class enrichment

Each feature is then tested for enrichment in each class against the rest, in both
directions: a binomial test for binary features and a Welch t-test for the others, with
Benjamini-Hochberg correction within each class and direction. A feature is enriched or
depleted in a class when its corrected $p$-value falls below $0.05$. The 24 reverse-coded SCQ
social items have their direction flipped, and the features are summarised into the seven
literature-defined categories (anxiety or mood, attention, disruptive behaviour, self-injury,
social or communication, restricted or repetitive, and developmental) as the signed
proportion enriched minus depleted. That seven-category vector is each class's signature.

## Naming the classes

The authors' three routes to naming a recovered class are mostly closed here: there are no
shared probands to match on, and no reference model was released. The classes are therefore
aligned on the published class signatures, the same currency the authors used to declare
replication in the SSC.

The primary mechanism is the set of named-class anchors, the substantive characteristics
that define each class:

- the highest-difficulty class overall, which is also the smallest, is Broadly affected;
- the most developmental of the rest is Mixed ASD with developmental delay;
- the largest of the rest, high on core, attention, and anxiety with no developmental delay,
  is Social or behavioural;
- the uniformly lowest is Moderate challenges.

The assignment is cross-checked for mutual consistency: the class named Broadly affected
should be both the highest overall and the smallest, the class named Social or behavioural
should be the largest, and so on. The published seven-category signature, read from the
paper's figure, gives a profile correlation for each class and overall as a second measure.

## The values read from figure 1b

The published seven-category signatures are not released as a numeric table, so they are read
from figure 1b of Litman et al. (2025), the per-category proportion-and-direction plot, at the
figure's resolution. The signed values run from $-1$ (depleted across that category) to $+1$
(enriched), in the seven-category order:

| Class | anxiety/mood | attention | disruptive | self-injury | social/comm | restricted/rep | developmental |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Social or behavioural | +1.0 | +1.0 | +0.95 | +0.50 | +0.50 | +0.45 | -0.90 |
| Moderate challenges | -0.90 | -0.95 | -1.0 | -0.95 | -1.0 | -1.0 | -1.0 |
| Mixed ASD with developmental delay | -0.90 | -0.45 | -0.65 | -0.10 | +0.10 | +0.05 | +0.45 |
| Broadly affected | +1.0 | +1.0 | +1.0 | +1.0 | +1.0 | +1.0 | +1.0 |

Broadly affected sits near $+1$ and Moderate challenges near $-1$ across every category, so
both profiles are saturated: their per-class correlation is uninformative and they rest on the
anchors instead. The published class proportions the anchors use, also taken from the paper,
are about 37, 34, 19, and 10 per cent for Social or behavioural, Moderate challenges, Mixed ASD
with developmental delay, and Broadly affected. Every value in this section is read from the
figure to its resolution, so it is approximate; obtaining the numeric supplementary tables would
replace these with exact values and let the alignment use the published profile directly.

## The reproduction result

On SPARK 2026-03-23 the model recovers four classes whose proportions are about 39, 29, 18,
and 15 per cent (Social or behavioural, Moderate challenges, Mixed ASD with developmental
delay, and Broadly affected), against the published 37, 34, 19, and 10. Every anchor holds,
so the four recovered classes map cleanly onto the four named classes.

The overall seven-category profile correlation against the published figure is $r = 0.902$,
which matches the authors' own SSC-replication value of $r = 0.927$. The per-class correlations
are $r = 0.97$ for Mixed ASD with developmental delay and $r = 0.85$ for Social or behavioural;
Broadly affected and Moderate challenges have saturated profiles (uniformly high and uniformly low
respectively), so their per-class correlation is uninformative and they rest on the anchors.
The main per-class divergence from the published profile is that Social or behavioural shows
weaker social-communication and restricted-or-repetitive enrichment here than in the paper.

## Caveats

The reproduction is benchmarked on the published profile and proportions, not on a
per-proband agreement, for two reasons stated up front. The authors' SPARK v9 release and
their per-proband labels are not available, so an exact reproduction of their cohort and a
label-level comparison are out of reach. And the published seven-category profile is read
from the paper's figure at the figure's resolution, not from a numeric supplementary table,
so the correlation is read against the figure. The cohort is also larger than the authors'
(11,704 against 5,392), because the release is later and broader.

With those caveats, the four classes reproduce with the correct structure, clean naming, and
matching proportions, and this named fit stands as the reference for the planned stability
work.
