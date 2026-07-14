# asd-def

## Investigations

Each investigation tests one hypothesis from the labelled registry (H0A to H0J). See
{doc}`the analysis package <packages/analysis/index>` for the full list, and its archive for the
reproduction and replication work these tests depend on.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} H0A / H0D. Invariance as an effect size
:link: packages/analysis/investigations/invariance-as-an-effect-size
:link-type: doc

Are the class profiles invariant to diagnostic era or age at diagnosis, and is any drift small
relative to the between-class separation?
:::

:::{grid-item-card} H0A corroboration. Score-based measurement invariance
:link: packages/analysis/investigations/score-based-invariance
:link-type: doc

Does a score-based fluctuation-process test corroborate the invariance read from the single
cached fit?
:::

:::{grid-item-card} H0A pilot / H0F. Tracking the classes across the strata
:link: packages/analysis/investigations/tracking-the-classes-across-strata
:link-type: doc

Do the classes hold across the strata, and is the drift spread evenly across the seven
phenotypic categories?
:::

:::{grid-item-card} H0E. The direction of the drift
:link: packages/analysis/investigations/the-direction-of-the-drift
:link-type: doc

Does the drift have a direction, or is it a symmetric excursion?
:::

:::{grid-item-card} H0B. The size of the classes
:link: packages/analysis/investigations/the-size-of-the-classes
:link-type: doc

Do the four classes make up the same share of the diagnosed population across the axis?
:::

::::

## Packages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} analysis
:link: packages/analysis/index
:link-type: doc

Reproduce the Litman autism classes and test their stability across age at diagnosis and
diagnostic era.
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
