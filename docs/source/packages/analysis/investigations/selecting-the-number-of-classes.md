# Selecting the number of classes

:::{admonition} The question
:class: note

How many classes do the data support? Reproducing four classes is necessary but not sufficient:
an automatic model-selection rule might prefer a different number. This investigation asks what a
panel of selection criteria says across one to ten classes, and whether they back the authors'
choice of four.
:::

:::{admonition} The result
:class: tip

The criteria do not select four. The information criteria fall across the whole grid and reach
their minimum at nine classes, the over-extraction these criteria are known to show at a large
sample. The cross-validated log-likelihood, the out-of-sample measure, gains little past four,
and the higher-class solutions degenerate, their smallest class falling towards a few per cent of
the cohort. Four classes is retained as the authors chose it, by reading the criteria rather than
by an automatic rule.
:::

:::{figure} /_figures/selection_criteria.png
:alt: Model-selection criteria across one to ten latent classes
:width: 100%
:align: center

Model selection across one to ten latent classes, from an `analysis select` run. (A) The
information criteria fall throughout and reach their minimum at nine classes, the over-extraction
expected at a sample of this size. (B) The cross-validated log-likelihood gains little beyond
four classes. (C) The smallest class proportion falls towards zero as classes are added; it is
class size, not classification certainty, that marks the higher-class solutions as
uninterpretable. The dashed line marks the four classes chosen by Litman et al.
:::

## Reading the result

On SPARK 2026-03-23 the information criteria do not select four: they fall across the whole grid
and reach their minimum at nine classes. This is the over-extraction these criteria are known to
show at a large sample, where the likelihood gain from an extra class outweighs the penalty on
the parameters it adds. Cutting the cohort back to the records present at the authors' V9 freeze
moves the minimum only to eight classes, still far from four (see
[isolating the records added since V9](isolating-the-new-records)), so the over-extraction is not
simply an artefact of the later release's size. The cross-validated log-likelihood, the
out-of-sample measure, gains little past four classes, and the higher-class solutions degenerate,
their smallest class falling towards a few per cent of the cohort.

The choice of four classes is not made by an automatic rule. The released code asserts it
visually, with a reference line at four on every criterion panel, and hard-codes four for the
final model. The stage therefore reports the full criteria table and the component count that
minimises each information criterion, leaving the four-class choice to the methods write-up. Four
classes is retained as the authors chose it, by reading the criteria.

## How the criteria are scored

:::{dropdown} The selection panel

The `select` stage grids over one to ten classes and scores each fit on a panel of criteria: the
cross-validated validation log-likelihood (three folds), the Akaike, Bayesian,
sample-size-adjusted Bayesian, and consistent Akaike information criteria, the approximate weight
of evidence, the scaled relative entropy, the average latent-class posterior probability, and the
smallest class proportion. The grid is repeated over several seeds and summarised as a mean and a
standard deviation per class count.

StepMix 3.0.0 provides the AIC, BIC, sample-size-adjusted BIC, and consistent AIC directly, and
their formulas match the authors' hand-written helpers, so those are used unchanged; the
approximate weight of evidence is computed in the package. The released "Lo-Mendell-Rubin
likelihood-ratio test" is a naive chi-square on the cross-validated log-likelihood differences,
with the degrees of freedom fixed at one rather than the difference in free parameters. The
package reproduces that approximation, and labels it a proxy, rather than substituting the
analytically correct adjusted test.
:::

The selection grid is seeded and resumable, like the other multi-seed stages; see
[the pipeline and its cache](../guides/pipeline-and-caching) for the seeding and checkpointing,
which are the one deliberate divergence from the released (unseeded) selection procedure.
