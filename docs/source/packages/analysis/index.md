# analysis

The analysis package for the Litman stability workstream: reproducing the data-driven autism
classes of Litman et al. (2025) and testing whether they hold within strata of age at
diagnosis and diagnostic era.

The package is built one pipeline stage at a time. Each stage is a CLI subcommand that reads
named inputs and writes its outputs plus a manifest under a content-addressed `artefacts/`
directory, so a later run recomputes only what changed.

## The pipeline

Three stages build the named reference, and four more stress-test it. The `cohort` stage builds
the harmonised proband-by-feature matrix, `fit` fits the reference four-class mixture model, and
`align` names the classes; the four checks branch from there. The remaining stages (the
stratified analysis and reporting) are listed under "planned" in `analysis --help` and are added
as the work proceeds.

:::{mermaid}
flowchart LR
  subgraph reference [Build the reference]
    direction LR
    cohort[cohort] --> fit[fit] --> align[align]
  end
  align --> stability[stability]
  align --> nmin[nmin]
  cohort --> select[select]
  cohort --> replicate[replicate]
:::

The cohort layer sits behind one interface with a SPARK and an SSC backend, so a stage runs on
either cohort. Each stage and where its result is reported:

| Stage | What it does | Reported in |
| --- | --- | --- |
| `cohort` | Builds the harmonised proband-by-feature matrix from a SPARK release and the authors' final feature list, with the reconciled feature typing. | {doc}`The cohort interface <guides/the-cohort-interface>` |
| `fit` | Fits the reference four-class general finite mixture model and predicts a class label per proband. | {doc}`Reproducing the reference classes <investigations/reproducing-the-reference-classes>` |
| `align` | Summarises each class into the seven literature-defined categories and aligns the recovered classes to Litman's four named classes. | {doc}`Reproducing the reference classes <investigations/reproducing-the-reference-classes>` |
| `select` | Grids over the number of components and reports the information criteria. | {doc}`Selecting the number of classes <investigations/selecting-the-number-of-classes>` |
| `stability` | Ranks many single-init fits by log-likelihood, and refits on random halves, comparing each fit to the reference. | {doc}`Stability under refitting <investigations/stability-under-refitting>` |
| `nmin` | Refits at descending sample sizes to fix the minimum viable stratum size. | {doc}`The minimum viable stratum size <investigations/the-minimum-stratum-size>` |
| `replicate` | Fits on the SPARK features shared with the SSC, projects onto the SSC, and correlates the profiles against a permutation null. | {doc}`Replicating in the SSC <investigations/replicating-in-the-ssc>` |

## Technical guides

How the machinery works: the staged pipeline and its cache, the runbook, the cohort interface,
and the SSC milestone parsing.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} The pipeline and its cache
:link: guides/pipeline-and-caching
:link-type: doc

The staged commands, the content-addressed artefact cache, the run manifests, and the
reproducibility choices behind them.
:::

:::{grid-item-card} Running the pipeline
:link: guides/running-the-pipeline
:link-type: doc

The runbook: the commands to run everything implemented so far, in order, and what each
stage depends on.
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

:::{grid-item-card} Subsetting the cohort to the V9 freeze
:link: guides/subsetting-to-the-v9-freeze
:link-type: doc

Cutting a later SPARK release back in time to the probands present at Litman's V9 freeze: the
roster and completion gates, and what the cut recovers.
:::

::::

## Investigations

The investigations follow the analysis as a sequence of questions, each with its own figure and
result. Each page opens with the question it answers and its headline result, embeds the figure
it reports, and folds the method detail into expandable sections. Read top to bottom for the arc,
or jump to one.

1. {doc}`Do the four classes reproduce? <investigations/reproducing-the-reference-classes>` They
   do: proportions 39/29/18/15 against the published 37/34/19/10, every named-class anchor holds,
   and the overall profile correlates with the published figure at $r = 0.90$.
2. {doc}`How many classes do the data support? <investigations/selecting-the-number-of-classes>`
   The selection criteria over-extract at this sample size (their minimum is at nine classes);
   four is retained by reading them, as the authors did.
3. {doc}`Do the classes survive re-initialisation and resampling? <investigations/stability-under-refitting>`
   The profiles reproduce at 0.91 to 0.92 and no class ever collapses; proband-level membership is
   softer (adjusted Rand 0.63 to 0.65).
4. {doc}`How small a stratum stays viable? <investigations/the-minimum-stratum-size>` Recovery is
   reliable from about 1,000 probands (isotonic floor about 840), so the stratification bins are
   best kept nearer 2,000.
5. {doc}`Do the classes replicate in a second cohort? <investigations/replicating-in-the-ssc>` They
   do in the SSC, at $r = 0.89$ ($p = 0.006$), with a bootstrap interval $[0.79, 0.93]$ that
   includes the authors' published $0.927$; six of the seven categories correlate at $0.85$ or
   above, the developmental category lower at $0.79$.

Several of these investigations carry a third condition alongside the full `2026-03-23` release and
the published values: the cohort cut back to the records present at the authors' V9 freeze (see
{doc}`subsetting the cohort to the V9 freeze <guides/subsetting-to-the-v9-freeze>`). Comparing the
full release, the V9 subset, and the paper shows which differences from the published solution trace
to the records added since V9.

Taken together, the reproduction and these checks show the pooled reference is solid enough at the
profile level to anchor the stratified test. That matters because the test compares
stratum-specific fits against this reference, and a fragile baseline would make any drift
uninterpretable. The class profiles reproduce across initialisations, resamples, and a second
cohort, the developmental category lower than the rest there; the membership is softer at the
boundaries; and the stratum-size floor sets the lower bound on how finely the cohort can be split.
Cutting the cohort back to the records present at the authors' V9 freeze shows the differences from
their class proportions are partly compositional, carried by the records added since, while the
overall reproduction, the inflated smallest class, and the SSC developmental gap are unchanged by
the cut.

What comes next is the stratified analysis itself. With the stratification plan, the bins, the
drift metrics, the null, and the decision thresholds frozen in advance so the result is
confirmatory, the model is re-estimated within strata of age at diagnosis and diagnostic era, each
stratum's classes are aligned to this named reference, and the drift is read against two
baselines: a permutation null that re-fits within strata of the same sizes formed by shuffling the
stratum labels, and the distance between distinct reference classes. The era axis carries its own
threat, the lag between when the phenotype is measured and when the diagnosis was made, which is
quantified and tested rather than assumed away. Once the genotype data are available, the same
strata are the setting for testing whether the genotype-to-phenotype mapping drifts.

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
:caption: Technical guides

guides/pipeline-and-caching
guides/running-the-pipeline
guides/the-cohort-interface
guides/parsing-ssc-milestone-ages
guides/subsetting-to-the-v9-freeze
:::

:::{toctree}
:hidden:
:caption: Investigations

investigations/reproducing-the-reference-classes
investigations/selecting-the-number-of-classes
investigations/stability-under-refitting
investigations/the-minimum-stratum-size
investigations/replicating-in-the-ssc
:::

:::{toctree}
:hidden:
:caption: Reference

reference
:::
