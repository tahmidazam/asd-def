# $H_0^J$: does genetic drift track phenotypic drift?

:::{admonition} Definition
:class: note

A compound, model-comparison hypothesis over the joint outcome of $H_0^A$ (profile invariance) and
$H_0^I$ (genotype-to-class invariance), conditional on rejecting $H_0^A$. It reads two patterns:

- Dissociation: the profiles drift ($H_0^A$ rejected) while the genotype-to-class mapping is
  invariant ($H_0^I$ retained). Reading: the phenotypic drift is not tracked by the genetic
  architecture, implicating non-genetic drivers (environmental exposure, ascertainment, or
  measurement change) in how autism presents across strata.
- Co-drift: the profiles and the genotype-to-class mapping drift together. Reading: the genetic
  programs move with the phenotypic classes, implicating a shift in the underlying biology of the
  diagnosed classes.

Estimand: the joint pattern of phenotypic-parameter drift and genotype-class-mapping drift along each
axis.
:::

:::{admonition} Status
:class: tip

Waiting on data access. This test is gated on $H_0^I$, which is itself gated on SFARI genotype
access, so it cannot run until the genetic data is held. Nothing is run.
:::

## Method

The planned test is a cross-tabulation of the $H_0^A$ and $H_0^I$ outcomes per axis: where the
profile drifted against where the genotype-to-class map drifted. Because the genotype-to-class
alignment is expected to be weak and diffuse, $H_0^I$ is powered as a test of a change in the
mapping rather than of its baseline strength, so a
retained $H_0^I$ is read as no detected co-drift rather than as proof of none.

This article will be completed once genotype access is granted and both $H_0^A$ and $H_0^I$ are read
on the same strata.

## See also

- {doc}`Does the genotype-to-class mapping hold? <h0i-genotype-mapping>` ($H_0^I$), the genetic
  test this compares against the phenotype.
- {doc}`Are the class profiles invariant, and is any drift small? <../hypotheses/h0a-invariance>`
  ($H_0^A$), the phenotypic drift half of the comparison.
