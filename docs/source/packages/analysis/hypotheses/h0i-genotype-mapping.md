# $H_0^I$: does the genotype-to-class mapping hold across diagnostic timing?

:::{admonition} Definition
:class: note

Null: the genotype-to-class association is invariant across strata of {term}`diagnostic era` and
{term}`age at diagnosis`; the class-conditional polygenic scores (autism, ADHD, major depression,
schizophrenia, educational attainment, IQ) and the rare-variant burden (de novo loss-of-function in
constrained genes, rare inherited burden) show no class-by-stratum interaction. Alternative: the
genotype-to-class mapping drifts, showing a class-by-stratum interaction on one or more genetic
contrasts. Estimand: the class contrast on each genetic measure as a function of the axis.
:::

:::{admonition} Status
:class: tip

Waiting on data access. This test is gated on SFARI genotype access, not on any earlier rejection,
and no genetic data is held yet. Nothing is run. Until the data is in hand, the phenotypic half of
the headline question is answered by $H_0^A$, $H_0^B$, and $H_0^C$, and this article is a
placeholder for the genetic half.
:::

## Method

The planned test is a class-by-stratum interaction (a Wald or {term}`likelihood-ratio test`) in a
generalised linear model over each genetic measure, with cognitive impairment as a covariate. The
{term}`autism class` labels stay fixed at the reference, so no mixture is re-estimated: only the
association between genotype and the fixed classes is tested for a change along the axis. Effect
sizes are read against a resampling baseline. The read is deliberately powered as a test of a
*change* in the mapping rather than of its baseline strength, because the genotype-to-class alignment
is expected to be weak and diffuse.

This article will be completed once genotype access is granted and the stage is implemented and run.

## See also

- {doc}`Are the class profiles invariant, and is any drift small? <../hypotheses/h0a-invariance>`
  ($H_0^A$), the phenotypic drift this genetic test is set beside.
- {doc}`Does genetic drift track phenotypic drift? <h0j-genotype-dissociation>` ($H_0^J$), the
  model comparison that reads this result against the phenotype.
