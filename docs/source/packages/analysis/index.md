# analysis

The analysis package for the Litman stability workstream: reproducing the data-driven autism
classes of Litman et al. (2025) and testing whether they hold within strata of age at
diagnosis and diagnostic era.

The package is built one pipeline stage at a time. Each stage is a CLI subcommand that reads
named inputs and writes its outputs plus a manifest under a content-addressed `artefacts/`
directory, so a later run recomputes only what changed.

## Implemented stages

- `analysis cohort` builds the harmonised proband-by-feature matrix from a SPARK release and
  the authors' final feature list, and writes the reconciled feature typing.
- `analysis fit` fits the reference general finite mixture model and predicts a class label
  per proband.
- `analysis align` summarises each class into the seven literature-defined categories and
  aligns the recovered classes to Litman's four named classes.

The cohort layer sits behind one interface with a SPARK and an SSC backend, so a stage runs
on either cohort. The SSC backend is in place; its fidelity to the authors' SSC pipeline is
confirmed in the replication stage. The remaining stages (model selection, stability, SSC
replication, the stratified analysis, and reporting) are listed under "planned" in
`analysis --help` and are added as the work proceeds.

## Guides

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} The pipeline and its cache
:link: guides/pipeline-and-caching
:link-type: doc

The staged commands, the content-addressed artefact cache, the run manifests, and the
reproducibility choices behind them.
:::

:::{grid-item-card} The cohort interface
:link: guides/the-cohort-interface
:link-type: doc

One interface over SPARK and the SSC, the pinned feature set, and the deliberate departures
from the released preprocessing.
:::

:::{grid-item-card} Reproducing the reference classes
:link: guides/reproducing-the-reference-classes
:link-type: doc

Feature typing, the mixture model, per-class enrichment, naming the classes, and the
reproduction result.
:::

::::

## Reference

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Python API
:link: reference
:link-type: doc

The command-line interface, the run and caching infrastructure, the cohort abstraction,
feature typing, the model wrapper, and the enrichment and alignment.
:::

::::

:::{toctree}
:hidden:
:caption: Guides

guides/pipeline-and-caching
guides/the-cohort-interface
guides/reproducing-the-reference-classes
:::

:::{toctree}
:hidden:
:caption: Reference

reference
:::
