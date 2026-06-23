# asd-def

## Investigations

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 1. Reproducing the reference classes
:link: packages/analysis/investigations/reproducing-the-reference-classes
:link-type: doc

Do the four classes reproduce on the release held here, and can they be named as the authors
named them?
:::

:::{grid-item-card} 2. Selecting the number of classes
:link: packages/analysis/investigations/selecting-the-number-of-classes
:link-type: doc

How many classes do the data support, and do the selection criteria back the authors' choice of
four?
:::

:::{grid-item-card} 3. Stability under refitting
:link: packages/analysis/investigations/stability-under-refitting
:link-type: doc

Do the classes survive re-initialisation and resampling, or are they an artefact of one fit or
one sample?
:::

:::{grid-item-card} 4. The minimum viable stratum size
:link: packages/analysis/investigations/the-minimum-stratum-size
:link-type: doc

How small can a stratum be before four-class recovery breaks down, which bounds the
stratification bins?
:::

:::{grid-item-card} 5. Replicating in the SSC
:link: packages/analysis/investigations/replicating-in-the-ssc
:link-type: doc

Do the four classes reappear in a second, independent cohort?
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
packages/figures/index
packages/dscat/index
packages/docs/index
:::
