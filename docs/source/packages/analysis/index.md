# analysis

The analysis package for the Litman stability workstream: reproducing the data-driven autism
classes of Litman et al. (2025) and testing whether they hold within strata of age at
diagnosis and diagnostic era.

Each stage is a CLI subcommand that reads named inputs and writes its outputs plus a manifest
under a content-addressed `artefacts/` directory, so a later run recomputes only what changed.

## The pipeline

Three stages build the named reference, and four more stress-test it. The `cohort` stage builds
the harmonised proband-by-feature matrix, `fit` fits the reference four-class mixture model, and
`align` names the classes; the four checks branch from there.

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
  cohort --> strata[strata-describe]
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
| `strata-describe` | Builds the age-at-diagnosis and diagnostic-era axes, the lag, and the demographics, and tests each binning policy against the acceptance requirements. | {doc}`Choosing the stratification bins <guides/choosing-the-stratification-bins>` |
| `strata`, `stratify` | Assigns every proband to a stratum under the frozen policy, then re-estimates the four-class model independently within each stratum. | {doc}`Tracking the classes across the strata <investigations/tracking-the-classes-across-strata>` |
| `drift` | Aligns each stratum's classes to the reference and reads their movement against the permutation null. | {doc}`Tracking the classes across the strata <investigations/tracking-the-classes-across-strata>` |
| `trajectory` | Traces each class's path across the strata and tests its net young-to-old displacement. | {doc}`Tracking the classes across the strata <investigations/tracking-the-classes-across-strata>` |
| `attribute` | Decomposes each class's movement onto features and categories, and contrasts the probands that moved with those that stayed. | {doc}`Tracking the classes across the strata <investigations/tracking-the-classes-across-strata>` |
| `invariance` | Tests the class profiles for stability along an axis from the single cached fit, reading a score-based fluctuation process against an analytic Brownian-bridge null. | {doc}`Score-based measurement invariance <guides/score-based-invariance>` |
| `invariance-trajectory` | Recasts the invariance read as a null-free effect size: the separation-scaled local-centroid displacement along an axis, with a family-clustered bootstrap tube, an in-plane capture fraction, a control-panel specificity check, and the per-class directional net-trend statistic (DIREC) with a descriptive break. | {doc}`Invariance as an effect size <guides/invariance-as-an-effect-size>` |
| `prevalence` | Tests whether the frozen class proportions trend along an axis (PREV): a maximum-likelihood three-step correction of a multinomial logit on the frozen posteriors, with a naive hard-label cross-check, per-class one-versus-rest slopes, predicted proportion curves, a family-clustered bootstrap, and (for era) a DSM-5 contrast. | {doc}`Measuring prevalence drift <guides/measuring-prevalence-drift>` |
| `order` | Tests whether the supported number of classes is stable across strata (ORDER): a warm-started parametric bootstrap likelihood-ratio search anchored at four, read relative to the pooled cohort under the identical procedure, corroborated by the cross-validated elbow and the adjusted Lo-Mendell-Rubin test. | {doc}`Testing the number of classes <guides/testing-the-number-of-classes>` |

## Technical guides

How the machinery works: the staged pipeline and its cache, the runbook, the cohort interface,
the SSC milestone parsing, choosing the stratification bins, the two halves of the drift read
(aligning a stratum to the reference and measuring how far each class moved), attributing a
movement to the features and probands that carry it, the score-based measurement-invariance
test that reads the same question from the single cached fit, and its recast as a null-free effect
size.

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

:::{grid-item-card} Choosing the stratification bins
:link: guides/choosing-the-stratification-bins
:link-type: doc

The binning policies, the acceptance requirements a partition must meet, and the
`strata-describe` check that fixes the bins the stratified analysis runs on.
:::

:::{grid-item-card} Aligning stratum classes to the reference
:link: guides/aligning-stratum-classes
:link-type: doc

Matching a stratum fit's arbitrarily numbered classes to the reference classes: the membership
and centroid methods, and how they separate a class that moved from one that reorganised.
:::

:::{grid-item-card} Measuring how far a class drifts
:link: guides/measuring-class-drift
:link-type: doc

The swappable distances between an aligned class and the reference (Mahalanobis, Euclidean,
mean-absolute, Jensen-Shannon), and the between-class separation they are read against.
:::

:::{grid-item-card} Attributing a class's movement
:link: guides/attributing-class-movement
:link-type: doc

Opening up a drift into the features that carry it (the centroid-shift decomposition) and the
probands that changed class (the mover-versus-stayer contrast).
:::

:::{grid-item-card} Testing the number of classes
:link: guides/testing-the-number-of-classes
:link-type: doc

The bootstrap likelihood-ratio search for the supported number of classes per stratum: the
warm-started fits, the parametric null, the anchored sequential search, and the elbow and
Lo-Mendell-Rubin corroborators.
:::

::::

## Investigations

Each investigation answers one question, with its own figure and headline result. Read in order
for the arc, or jump to one.

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
6. {doc}`Do the classes hold across the strata? <investigations/tracking-the-classes-across-strata>`
   In the first stratified run (one hundred permutations, both axes) no stratum drifts beyond its
   null, but the classes trace a gradual young-to-old trajectory carried by the developmental
   classes and features, with the membership reorganising at the extremes.
7. {doc}`Does the drift have a direction? <investigations/the-direction-of-the-drift>` In seven of
   the eight class-by-axis tests it does. Age at diagnosis moves every class the same way, led by
   the developmental class; diagnostic era splits, and the Moderate challenges class has a large
   era displacement but no directional trend, with the era breaks falling around 2017.
8. {doc}`Which controls does the specificity check use? <investigations/choosing-the-specificity-controls>`
   The negative controls are chosen by role and by a timing-orthogonality screen, not by convenience:
   the phenotype and the clustered features are excluded on principle, leaving household income and
   area deprivation against a random floor. The one control that decides the era verdict, household
   income, is shown to be a genuine socioeconomic gradient rather than leaked phenotype.
9. {doc}`Do the class sizes shift? <investigations/the-size-of-the-classes>` Every class's
   proportion trends on both axes. The developmental class falls from about a third of the
   earliest-diagnosed to almost none of the latest while the social class grows to three fifths; the
   era shift is milder, and its social rise is explained away by measurement timing while the age
   shift is not.

Several of these investigations carry a third condition alongside the full `2026-03-23` release
and the published values: the cohort cut back to the records present at the authors' V9 freeze
(see {doc}`subsetting the cohort to the V9 freeze <guides/subsetting-to-the-v9-freeze>`), which
isolates the differences that trace to the records added since.

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
guides/choosing-the-stratification-bins
guides/aligning-stratum-classes
guides/measuring-class-drift
guides/attributing-class-movement
guides/score-based-invariance
guides/invariance-as-an-effect-size
guides/measuring-prevalence-drift
guides/testing-the-number-of-classes
:::

:::{toctree}
:hidden:
:caption: Investigations

investigations/reproducing-the-reference-classes
investigations/selecting-the-number-of-classes
investigations/stability-under-refitting
investigations/the-minimum-stratum-size
investigations/replicating-in-the-ssc
investigations/tracking-the-classes-across-strata
investigations/the-direction-of-the-drift
investigations/choosing-the-specificity-controls
investigations/the-size-of-the-classes
:::

:::{toctree}
:hidden:
:caption: Reference

reference
:::
