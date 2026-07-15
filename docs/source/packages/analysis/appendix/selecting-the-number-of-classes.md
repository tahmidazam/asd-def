# Selecting the number of classes

:::{admonition} The question
:class: note

How many classes do the data support? Reproducing four is necessary but not sufficient: an
automatic rule might prefer a different number. This appendix grids a panel of selection criteria
over $1$ to $10$ classes and asks whether they back the choice of four.
:::

:::{admonition} The result
:class: tip

No automatic criterion selects four. The information criteria fall across the whole grid and reach
their minimum at $9$ classes on the full release, the over-extraction expected at a large sample.
Four is kept by reading the criteria rather than minimising one: the out-of-sample log-likelihood
gains little past it, and every higher solution degenerates into a class of a few per cent. This is
the authors' own procedure, which fixes four instead of taking an argmin.
:::

:::{figure} /_figures/selection_criteria.png
:alt: Model-selection criteria across one to ten latent classes
:width: 100%
:align: center

Model selection across $1$ to $10$ latent classes, from `analysis select` runs on the full
`2026-03-23` release and on the cohort cut back to the records at the authors' V9 freeze (the
subset; see {doc}`subsetting the cohort to the V9 freeze <../appendix/subsetting-to-the-v9-freeze>`).
(A) The Bayesian information criterion, normalised within each cut since it scales with sample size,
reaches its minimum at $9$ classes on the full release and $8$ on the V9 subset, both far along the
grid. (B) The cross-validated log-likelihood gains little beyond four classes in either cut.
(C) The smallest class proportion falls towards zero as classes are added. The dashed line marks
the four classes chosen by Litman et al.
:::

## Why no criterion selects four

The naive rule is to take the number of classes that minimises an information criterion. Here that
rule picks $9$ on the full release and $8$ on the V9 subset, not four. At a sample this large the
likelihood gain from each added class outruns the penalty on its parameters, so the criteria keep
falling well past any interpretable count. This over-extraction is a known property of the criteria,
not a signal that the data hold eight or nine classes. Cutting the cohort back to the V9 freeze
moves the minimum only from $9$ to $8$, so it is not an artefact of the later release's size.

## Why four is kept

Two out-of-sample readings agree on four. The cross-validated log-likelihood, which scores fit on
held-out folds rather than rewarding raw complexity, flattens after the fourth class: its per-class
gain drops from about $5$ points at the fourth to $3$ at the fifth and shrinks from there. The
higher solutions also degenerate. The smallest class falls from roughly $13\%$ at four classes to
about $5\%$ by nine, while classification certainty stays high throughout, so it is class size, not
mislabelling, that makes those solutions uninterpretable.

The four-class choice is deliberate, not automatic. The released code asserts it visually, with a
reference line at four on every criterion panel, and hard-codes four for the final model. The
`select` stage matches that: it reports the full criteria table and the count that minimises each
information criterion, and leaves the four-class decision to the methods write-up.

## How the criteria are scored

:::{dropdown} The selection panel

The `select` stage grids over $1$ to $10$ classes and scores each fit on a panel of criteria: the
cross-validated log-likelihood ($3$ folds), the Akaike, Bayesian, sample-size-adjusted Bayesian,
and consistent Akaike information criteria, the approximate weight of evidence, the scaled relative
entropy, the average latent-class posterior probability, and the smallest class proportion. The
grid is repeated over several seeds and summarised as a mean and a standard deviation per class
count.

StepMix 3.0.0 provides the AIC, BIC, sample-size-adjusted BIC, and consistent AIC directly, and
their formulas match the authors' hand-written helpers, so those are used unchanged; the approximate
weight of evidence is computed in the package. The released "Lo-Mendell-Rubin likelihood-ratio test"
is a naive chi-square on the cross-validated log-likelihood differences, with the degrees of freedom
fixed at $1$ rather than the difference in free parameters. The package reproduces that
approximation and labels it a proxy, rather than substituting the analytically correct adjusted
test.
:::

The selection grid is seeded and resumable, like the other multi-seed stages; see
[the pipeline and its cache](../guides/pipeline-and-caching) for the seeding and checkpointing,
which are the one deliberate divergence from the released (unseeded) selection procedure.
