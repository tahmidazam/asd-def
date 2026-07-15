# Stability under refitting

:::{admonition} The question
:class: note

A mixture-model fit depends on its random start and its sample. Is the four-class solution a
property of the cohort, or an artefact of one fit? This stage refits the reference model many
times, from different seeds and on random halves of the cohort, and compares each fit back to the
named reference.
:::

:::{admonition} The result
:class: tip

The class profiles are stable; membership is softer at the edges. The seven-category profile
reproduces against the reference at a mean correlation of $0.91$ across re-initialisations and
$0.92$ across random halves, and no fit ever collapsed a class ($0$ of $100$, $0$ of $50$). The
adjusted Rand index, which tracks individual proband assignment, is more moderate at $0.63$ to
$0.65$: substantial agreement, not near-perfect, because probands near a class boundary move
between fits.
:::

:::{figure} /_figures/stability.png
:alt: Profile and membership stability of the reference fit under subsampling
:width: 100%
:align: center

Stability of the reference fit under subsampling, from an `analysis stability` run ($50$ refits on
random halves). (A) The seven-category profile correlation clusters near $0.92$, while the adjusted
Rand index is more spread, around $0.65$: the class definitions reproduce, the proband-level
membership less so. (B) The per-category correlation is uniformly high, developmental included, in
contrast to its weaker cross-cohort replication. (C) The mean class-overlap matrix; its diagonal,
each class's retention, runs from $0.83$ to $0.87$.
:::

## Two metrics, two questions

The profile correlation and the adjusted Rand index answer different questions, and the gap
between them is the result.

The **profile correlation** asks whether the class definitions reproduce. They do. Across the
best $100$ of $200$ re-initialisations the mean is $0.91$ (standard deviation $0.09$); across $50$
random halves it is $0.92$ (standard deviation $0.05$). No fit collapsed a class in either mode
($0$ of $100$, $0$ of $50$), so the four-class solution is recovered every time. Per-category, the
correlations run from $0.87$ (disruptive behaviour) to $0.98$ (anxiety or mood) under subsampling.

The **adjusted Rand index** asks whether individual probands keep the same class. Agreement there
is only partial: $0.63$ (standard deviation $0.14$) across re-initialisations and $0.65$ (standard
deviation $0.14$) across subsamples. The index is chance-corrected, so $0$ is the agreement
expected at random and $1$ is identical assignment; a value near $0.65$ is substantial, not
near-perfect, because probands near a class boundary are assigned differently from fit to fit.

Both modes run on SPARK 2026-03-23 against the named reference fit; each random half is about
$5{,}850$ probands.

## Developmental is stable within SPARK

The developmental category replicates as well as the rest here, at $0.94$ under subsampling. Its
weaker value in the cross-cohort [SSC replication](replicating-in-the-ssc), $0.86$, is not
intrinsic instability: there the category rests on a reduced, milestone-only feature set and a
correlation taken over the four classes alone. Within SPARK, where every category is measured as
in the reference, neither limitation applies.

## What the pattern means

Stable profiles with softer boundary membership is what a continuum with regions would produce:
the regions (the profiles) are reproducible, while a proband sitting between two regions is
assigned differently from fit to fit. This does not decide between four discrete classes and a
graded continuum with four regions; it shows the question is live, and the planned sensitivity work
tests it directly by comparing the four-class model against a lower-dimensional severity gradient.
The reference classes are stable as structures, with membership that is softer at the edges.

## How the fits are compared

:::{dropdown} The comparison currency

Each fit is compared to the named reference in the
[seven-category signature](reproducing-the-reference-classes) from the reproduction stage: the
signed proportion of each category's features enriched minus depleted, per class. Two solutions of
the same probands have their class labels aligned first, by the released greedy overlap rule, so
that profiles are compared class for class.
:::

:::{dropdown} The two stability modes

In `multi-init` mode the stage runs many single-initialisation fits from different seeds, ranks
them by log-likelihood, and compares the best of them to the reference; the released default runs
$2{,}000$ fits and reports the best $100$, both configurable. In `subsample` mode it refits on
random halves of the cohort and compares each half back to the full-sample reference.
:::

:::{dropdown} How each fit is scored

Every fit is scored three ways. The seven-category profile correlation is the authors' own
measure. The class-overlap matrix records, for each reference class, the proportion of its probands
the fit places in each class; after alignment, the diagonal is each class's retention. The adjusted
Rand index is added because it is label-invariant and chance-corrected, so it needs no prior
alignment. The released code reports the correlation and the overlap; the Rand index is a
deliberate addition.
:::

Both modes are seeded and resumable; see
[the pipeline and its cache](../guides/pipeline-and-caching) for the seeding and checkpointing,
the one deliberate divergence from the released (unseeded) stability procedure.
