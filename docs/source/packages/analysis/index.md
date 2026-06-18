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
- `analysis select` grids over the number of components and reports the information criteria
  (validation log-likelihood, AIC, BIC, sample-size-adjusted BIC, consistent AIC, the
  approximate weight of evidence, the relative entropy, the average posterior certainty, and
  the smallest class proportion).
- `analysis stability` summarises multi-initialisation stability (ranking many single-init
  fits by log-likelihood) or subsampling stability (refitting on random halves), comparing
  each fit to the reference by profile correlation, class overlap, and the adjusted Rand
  index.
- `analysis nmin` refits at descending sample sizes to fix the minimum viable stratum size
  for the later stratified work.
- `analysis replicate` fits a fresh model on the SPARK features shared with the SSC, projects
  it onto the SSC, and correlates the seven-category profiles against a permutation null.

The cohort layer sits behind one interface with a SPARK and an SSC backend, so a stage runs
on either cohort. The remaining stages (the stratified analysis and reporting) are listed
under "planned" in `analysis --help` and are added as the work proceeds.

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

:::{grid-item-card} Parsing the SSC milestone ages
:link: guides/parsing-ssc-milestone-ages
:link-type: doc

Turning the SSC's free-text developmental-milestone ages into months: the forms recognised,
and the entries left missing.
:::

:::{grid-item-card} Reproducing the reference classes
:link: guides/reproducing-the-reference-classes
:link-type: doc

Feature typing, the mixture model, per-class enrichment, naming the classes, and the
reproduction result.
:::

:::{grid-item-card} Stability, selection, and replication
:link: guides/stability-selection-and-replication
:link-type: doc

How many classes the data support, whether the solution survives re-initialisation and
resampling, the minimum viable stratum size, and cross-cohort replication.
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
guides/parsing-ssc-milestone-ages
guides/reproducing-the-reference-classes
guides/stability-selection-and-replication
:::

:::{toctree}
:hidden:
:caption: Reference

reference
:::
