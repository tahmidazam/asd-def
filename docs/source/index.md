# Phenotypic autism classes drift with diagnostic era and age at diagnosis

Tahmid Azam (School of Clinical Medicine, University of Cambridge) and Chirag J. Patel
(Department of Biomedical Informatics, Harvard Medical School).

## Background

Litman et al. {footcite}`litmanDecompositionPhenotypicHeterogeneity2025a` fitted a
{term}`general finite mixture model` to parent-reported phenotype data from {term}`SPARK` and
recovered four {term}`autism class`es (Social or behavioral, Mixed ASD with developmental delay,
Moderate challenges, and Broadly affected), which they map to genetic programs of common, de novo,
and inherited variation and present as biologically grounded. This work tests whether those classes
are invariant to {term}`diagnostic era` and to {term}`age at diagnosis`.

## The gap

Litman et al. enter age at evaluation as a model covariate and use age at diagnosis only for
external validation, so no peer-reviewed work has tested whether the classes are confounded by
diagnostic timing. The surrounding literature has questioned the stability of the autism diagnosis
over time: age at diagnosis indexes different developmental and polygenic profiles
{footcite}`zhangPolygenicDevelopmentalProfiles2025`, and the mean polygenic load of a diagnosis
declines with diagnostic year, with broadening criteria a candidate explanation
{footcite}`labiancaChangesGeneticContributions2026`. Both describe autism at the population level;
whether Litman's classes agree with those shifts, or are confounded by them, is untested.

## Why it matters

A data-driven class becomes consequential the moment people are sorted into it, for research, for
clinical narratives, and eventually for care. Who is diagnosed early, and who is diagnosed at all,
has never been random: it tracks sex, race, and access to services. A partition that is not
invariant to diagnostic era and age at diagnosis risks mistaking the history of who was noticed, and
when, for structure intrinsic to the disorder. Testing that invariance is what this work is for.

## Hypotheses

| Hypothesis | Null | Status |
| --- | --- | --- |
| [$H_0^A$](packages/analysis/hypotheses/h0a-invariance.md) | The four-class profiles are invariant to diagnostic era and age at diagnosis. | Rejected |
| [$H_0^B$](packages/analysis/hypotheses/h0b-prevalence.md) | The class proportions are invariant to both axes. | Rejected |
| [$H_0^C$](packages/analysis/hypotheses/h0c-order.md) | The supported number of classes is four in every stratum. | Not yet run |
| [$H_0^D$](packages/analysis/hypotheses/h0a-invariance.md) | Any drift is within sampling noise and small against the {term}`between-class separation`. | Rejected |
| [$H_0^E$](packages/analysis/hypotheses/h0e-direction.md) | The drift has no consistent direction along the axis. | Partially rejected |
| [$H_0^F$](packages/analysis/hypotheses/h0f-attribution-categories.md) | The drift is spread evenly across the {term}`seven phenotype categories`. | Rejected |
| [$H_0^G$](packages/analysis/hypotheses/h0g-attribution-referent.md) | The era drift is spread evenly across instruments regardless of {term}`referent`. | Rejected |
| [$H_0^H$](packages/analysis/hypotheses/h0h-attribution-timing.md) | The era drift is spread evenly across {term}`measurement lag`. | Partially written |
| [$H_0^I$](packages/analysis/hypotheses/h0i-genotype-mapping.md) | The genotype-to-class mapping is invariant to both axes. | Waiting on data access |
| [$H_0^J$](packages/analysis/hypotheses/h0j-genotype-dissociation.md) | Drift in the genotype-to-class mapping tracks the phenotype drift. | Waiting on data access |

## Sources

```{footbibliography}
```

## Packages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} analysis
:link: packages/analysis/index
:link-type: doc

Reproduce the four Litman autism classes and test whether they hold within strata of age at
diagnosis and diagnostic era.
:::

:::{grid-item-card} figures
:link: packages/figures/index
:link-type: doc

Render the analysis artefacts as Matplotlib figures for the manuscript.
:::

:::{grid-item-card} dscat
:link: packages/dscat/index
:link-type: doc

A searchable catalogue over the versioned SPARK and SSC data dictionaries.
:::

:::{grid-item-card} docs
:link: packages/docs/index
:link-type: doc

Build and live-preview this Sphinx site.
:::

::::

:::{toctree}
:hidden:
:maxdepth: 2

packages/analysis/index
glossary
references
packages/figures/index
packages/dscat/index
packages/docs/index
:::
