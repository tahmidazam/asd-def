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
| `fit` | Fits the reference four-class general finite mixture model and predicts a class label per proband. | {doc}`Reproducing the reference classes <appendix/reproducing-the-reference-classes>` |
| `align` | Summarises each class into the seven literature-defined categories and aligns the recovered classes to Litman's four named classes. | {doc}`Reproducing the reference classes <appendix/reproducing-the-reference-classes>` |
| `select` | Grids over the number of components and reports the information criteria. | {doc}`Selecting the number of classes <appendix/selecting-the-number-of-classes>` |
| `stability` | Ranks many single-init fits by log-likelihood, and refits on random halves, comparing each fit to the reference. | {doc}`Stability under refitting <appendix/stability-under-refitting>` |
| `nmin` | Refits at descending sample sizes to fix the minimum viable stratum size. | {doc}`The minimum viable stratum size <appendix/the-minimum-stratum-size>` |
| `replicate` | Fits on the SPARK features shared with the SSC, projects onto the SSC, and correlates the profiles against a permutation null. | {doc}`Replicating in the SSC <appendix/replicating-in-the-ssc>` |
| `strata-describe` | Builds the age-at-diagnosis and diagnostic-era axes, the lag, and the demographics, and tests each binning policy against the acceptance requirements. | {doc}`Choosing the stratification bins <guides/choosing-the-stratification-bins>` |
| `strata`, `stratify` | Assigns every proband to a stratum under the frozen policy, then re-estimates the four-class model independently within each stratum. | {doc}`Tracking the classes across the strata <archive/tracking-the-classes-across-strata>` |
| `drift` | Aligns each stratum's classes to the reference and reads their movement against the permutation null. | {doc}`Tracking the classes across the strata <archive/tracking-the-classes-across-strata>` |
| `trajectory` | Traces each class's path across the strata and tests its net young-to-old displacement. | {doc}`Tracking the classes across the strata <archive/tracking-the-classes-across-strata>` |
| `attribute` | Decomposes each class's movement onto features and categories, and contrasts the probands that moved with those that stayed. | {doc}`$H_0^F$: attribution by category <hypotheses/h0f-attribution-categories>` |
| `invariance` | Tests the class profiles for stability along an axis from the single cached fit, reading a score-based fluctuation process against an analytic Brownian-bridge null. | {doc}`The score-based invariance test <guides/the-score-based-invariance-test>` |
| `invariance-trajectory` | Recasts the invariance read as a null-free effect size: the separation-scaled local-centroid displacement along an axis, with a family-clustered bootstrap band, an in-plane capture fraction, a control-panel specificity check, the per-class directional net-trend statistic with a descriptive break, and (era only) the current-versus-retrospective referent split. | {doc}`$H_0^A$: invariance <hypotheses/h0a-invariance>`, {doc}`Computing the invariance effect size <guides/computing-the-invariance-effect-size>` |
| `prevalence` | Tests whether the frozen class proportions trend along an axis: a maximum-likelihood three-step correction of a multinomial logit on the frozen posteriors, with a naive hard-label cross-check, per-class one-versus-rest slopes, predicted proportion curves, a family-clustered bootstrap, and (for era) a DSM-5 contrast. | {doc}`$H_0^B$: prevalence <hypotheses/h0b-prevalence>` |
| `order` | Tests whether the supported number of classes is stable across strata: a warm-started parametric bootstrap likelihood-ratio search anchored at four, read relative to the pooled cohort under the identical procedure, corroborated by the cross-validated elbow and the adjusted Lo-Mendell-Rubin test. | {doc}`$H_0^C$: order <hypotheses/h0c-order>` |

## Hypotheses

Each hypothesis has its own article: the null and estimand, the verdict up front, the method with
its figures, and the cross-links to the machinery it rests on. The front page carries the status
table across all ten.

- {doc}`$H_0^A$ and $H_0^D$: are the profiles invariant, and is any drift small? <hypotheses/h0a-invariance>` Both rejected on both axes.
- {doc}`$H_0^B$: do the class proportions shift? <hypotheses/h0b-prevalence>` Rejected: every proportion trends on both axes.
- {doc}`$H_0^C$: is the supported number of classes four in every stratum? <hypotheses/h0c-order>` Not yet run.
- {doc}`$H_0^E$: does the drift have a direction? <hypotheses/h0e-direction>` Partially rejected: directional in 7 of 8 tests.
- {doc}`$H_0^F$: is the drift spread evenly across the seven categories? <hypotheses/h0f-attribution-categories>` Rejected: concentrated in the developmental features.
- {doc}`$H_0^G$: is the era drift even across instruments regardless of referent? <hypotheses/h0g-attribution-referent>` Rejected: retrospective-dominant for all four classes.
- {doc}`$H_0^H$: is the era drift an artefact of measurement timing? <hypotheses/h0h-attribution-timing>` Partially written.
- {doc}`$H_0^I$: does the genotype-to-class mapping hold? <hypotheses/h0i-genotype-mapping>` Waiting on data access.
- {doc}`$H_0^J$: does genetic drift track phenotypic drift? <hypotheses/h0j-genotype-dissociation>` Waiting on data access.

## Technical guides

How the machinery works: the staged pipeline and its cache, the runbook, the cohort interface,
the SSC milestone parsing, choosing the stratification bins, the two halves of the drift read
(aligning a stratum to the reference and measuring how far each class moved), the two invariance
reads (the effect-size recast and the score-based test), attributing a movement to the features and
probands that carry it, splitting the era drift by referent, measuring prevalence drift, and testing
the number of classes.

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

:::{grid-item-card} Computing the invariance effect size
:link: guides/computing-the-invariance-effect-size
:link-type: doc

The separation-scaled local-centroid displacement, its family-clustered bootstrap band, the
in-plane capture fraction, the specificity panel, the directional statistic, and the correctness
gates.
:::

:::{grid-item-card} The score-based invariance test
:link: guides/the-score-based-invariance-test
:link-type: doc

The casewise score, the empirical fluctuation process and its Brownian-bridge null, the two
statistics, the focal blocks, and the correctness gates.
:::

:::{grid-item-card} Attributing a class's movement
:link: guides/attributing-class-movement
:link-type: doc

Opening up a drift into the features that carry it (the centroid-shift decomposition) and the
probands that changed class (the mover-versus-stayer contrast).
:::

:::{grid-item-card} Splitting the era drift by referent
:link: guides/splitting-the-drift-by-referent
:link-type: doc

The size-fair current-versus-retrospective decomposition of the era drift, discriminating a
measurement-timing signature from a diagnosed-population one.
:::

:::{grid-item-card} Measuring prevalence drift
:link: guides/measuring-prevalence-drift
:link-type: doc

The three-step correction of a multinomial logit on the frozen posteriors, the per-class slopes,
the predicted proportion curves, and the DSM-5 contrast.
:::

:::{grid-item-card} Testing the number of classes
:link: guides/testing-the-number-of-classes
:link-type: doc

The bootstrap likelihood-ratio search for the supported number of classes per stratum: the
warm-started fits, the parametric null, the anchored sequential search, and the elbow and
Lo-Mendell-Rubin corroborators.
:::

::::

## Appendix

Reproduction, replication, and other foundational work that the hypothesis tests depend on, but
that does not itself test a hypothesis. Several of these carry a third condition alongside the full
`2026-03-23` release and the published values: the cohort cut back to the records present at the
authors' V9 freeze (see {doc}`subsetting the cohort to the V9 freeze
<appendix/subsetting-to-the-v9-freeze>`), which isolates the differences that trace to the records
added since.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Do the four classes reproduce?
:link: appendix/reproducing-the-reference-classes
:link-type: doc

They do: proportions 39/29/18/15 against the published 37/34/19/10, every named-class anchor
holds, and the overall profile correlates with the published figure at $r = 0.90$.
:::

:::{grid-item-card} How many classes do the data support?
:link: appendix/selecting-the-number-of-classes
:link-type: doc

The selection criteria over-extract at this sample size (their minimum is at nine classes); four
is retained by reading them, as the authors did.
:::

:::{grid-item-card} Do the classes survive re-initialisation and resampling?
:link: appendix/stability-under-refitting
:link-type: doc

The profiles reproduce at 0.91 to 0.92 and no class ever collapses; proband-level membership is
softer (adjusted Rand 0.63 to 0.65).
:::

:::{grid-item-card} How small a stratum stays viable?
:link: appendix/the-minimum-stratum-size
:link-type: doc

Recovery is reliable from about 1,000 probands (isotonic floor about 840), so the stratification
bins are best kept nearer 2,000.
:::

:::{grid-item-card} Do the classes replicate in a second cohort?
:link: appendix/replicating-in-the-ssc
:link-type: doc

They do in the SSC, at $r = 0.89$ ($p = 0.006$), with a bootstrap interval $[0.79, 0.93]$ that
includes the authors' published $0.927$; six of the seven categories correlate at $0.85$ or above,
the developmental category lower at $0.79$.
:::

:::{grid-item-card} Which controls does the specificity check use?
:link: appendix/choosing-the-specificity-controls
:link-type: doc

The negative controls are chosen by role and by a timing-orthogonality screen, not by
convenience: the phenotype and the clustered features are excluded on principle, leaving
household income and area deprivation against a random floor.
:::

:::{grid-item-card} Subsetting the cohort to the V9 freeze
:link: appendix/subsetting-to-the-v9-freeze
:link-type: doc

Cutting a later SPARK release back in time to the probands present at Litman's V9 freeze: the
roster and completion gates, and what the cut recovers.
:::

::::

## Archive

Superseded work, kept for the record. The refit pilot is an earlier route to the drift question,
now answered by the effect-size recast and the hypothesis articles.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Tracking the classes across the strata (refit pilot)
:link: archive/tracking-the-classes-across-strata
:link-type: doc

The hundred-permutation refit pilot on both axes: no stratum drifts beyond its null, but a gradual
young-to-old trajectory carried by the developmental classes and features. Its category-attribution
result is now $H_0^F$.
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
:caption: Hypotheses

hypotheses/h0a-invariance
hypotheses/h0b-prevalence
hypotheses/h0c-order
hypotheses/h0e-direction
hypotheses/h0f-attribution-categories
hypotheses/h0g-attribution-referent
hypotheses/h0h-attribution-timing
hypotheses/h0i-genotype-mapping
hypotheses/h0j-genotype-dissociation
:::

:::{toctree}
:hidden:
:caption: Technical guides

guides/pipeline-and-caching
guides/running-the-pipeline
guides/the-cohort-interface
guides/parsing-ssc-milestone-ages
guides/choosing-the-stratification-bins
guides/aligning-stratum-classes
guides/measuring-class-drift
guides/computing-the-invariance-effect-size
guides/the-score-based-invariance-test
guides/attributing-class-movement
guides/splitting-the-drift-by-referent
guides/measuring-prevalence-drift
guides/testing-the-number-of-classes
:::

:::{toctree}
:hidden:
:caption: Appendix

appendix/reproducing-the-reference-classes
appendix/selecting-the-number-of-classes
appendix/stability-under-refitting
appendix/the-minimum-stratum-size
appendix/replicating-in-the-ssc
appendix/choosing-the-specificity-controls
appendix/subsetting-to-the-v9-freeze
:::

:::{toctree}
:hidden:
:caption: Archive

archive/tracking-the-classes-across-strata
:::

:::{toctree}
:hidden:
:caption: Reference

reference
:::
