# Stability, selection, and replication

Reproducing the four classes is necessary but not sufficient: a partition can reproduce and
still be an artefact of one fit, one initialisation, or one cohort. The second goal is to
test how solid the reference solution is, three ways, before the stratified work begins. The
`select` stage asks how many classes the data support; the `stability` stage asks whether the
solution survives re-initialisation and resampling; the `replicate` stage asks whether it
reappears in a second cohort. The `nmin` stage reuses the resampling machinery to fix how
small a stratum can be before four-class recovery breaks down, which bounds the bins in the
stratified analysis.

Each stage compares a fitted solution to the named reference fit. The comparison currency is
the seven-category signature from the reproduction stage: the signed proportion of each
category's features enriched minus depleted, per class. Two solutions of the same probands
have their class labels aligned first, by the released greedy overlap rule, so that profiles
are compared class for class.

## Model selection

The `select` stage grids over one to ten classes and scores each fit on a panel of criteria:
the cross-validated validation log-likelihood (three folds), the Akaike, Bayesian,
sample-size-adjusted Bayesian, and consistent Akaike information criteria, the approximate
weight of evidence, the scaled relative entropy, the average latent-class posterior
probability, and the smallest class proportion. The grid is repeated over several seeds and
summarised as a mean and a standard deviation per class count.

StepMix 3.0.0 provides the AIC, BIC, sample-size-adjusted BIC, and consistent AIC directly,
and their formulas match the authors' hand-written helpers, so those are used unchanged; the
approximate weight of evidence is computed in the package. The released "Lo-Mendell-Rubin
likelihood-ratio test" is a naive chi-square on the cross-validated log-likelihood
differences, with the degrees of freedom fixed at one rather than the difference in free
parameters. The package reproduces that approximation, and labels it a proxy, rather than
substituting the analytically correct adjusted test.

The choice of four classes is not made by an automatic rule. The released code asserts it
visually, with a reference line at four on every criterion panel, and hard-codes four for the
final model. The stage therefore reports the full criteria table and the component count that
minimises each information criterion, leaving the four-class choice to the methods write-up.

:::{figure} /_figures/selection_criteria.png
:alt: Model-selection criteria across one to ten latent classes
:width: 100%
:align: center

Model selection across one to ten latent classes, from an `analysis select` run. (a) The
information criteria fall throughout and reach their minimum at nine classes, the
over-extraction expected at a sample of this size. (b) The cross-validated log-likelihood
gains little beyond four classes. (c) The smallest class proportion falls towards zero as
classes are added, while the relative entropy stays high, so it is class size rather than
classification certainty that marks the higher-class solutions as uninterpretable. The dashed
line marks the four classes chosen by Litman et al.
:::

## Multi-initialisation and subsampling stability

Mixture-model fits depend on their random start, so the `stability` stage in its
`multi-init` mode runs many single-initialisation fits from different seeds, ranks them by
log-likelihood, and compares the best of them to the reference. The released analysis runs
2,000 fits and reports the best 100; both counts are configurable. In its `subsample` mode
the stage refits on random halves of the cohort and compares each half back to the
full-sample reference.

Every compared fit is scored three ways. The seven-category profile correlation is the
authors' own measure. The class-overlap matrix records, for each reference class, the
proportion of its probands that the fit places in each of its classes; after alignment, the
diagonal is the retention of each class. The adjusted Rand index is added to these because it
is label-invariant and chance-corrected, so it needs no prior alignment. The released code
reports the profile correlation and the overlap; the Rand index is a deliberate addition.

## The stability result

Both modes run on SPARK 2026-03-23 against the named reference fit. Multi-initialisation ranks
200 single-initialisation fits by log-likelihood and compares the best 100 to the reference;
subsampling refits on 50 random halves of the cohort (about 5,850 probands each). The
seven-category profile reproduces strongly under both: the mean overall correlation is 0.91
across the multi-initialisation fits and 0.92 (standard deviation 0.05) across the subsamples.
No fit collapsed a class in either mode (0 of 100 and 0 of 50), so the four-class solution is
recovered every time.

The per-category correlations are uniformly high under subsampling, from 0.87 (disruptive
behaviour) to 0.98 (anxiety or mood), with developmental at 0.94. That developmental figure
contrasts with the 0.49 it scores in the SSC replication: within SPARK, where the milestone ages
are measured the same way as in the reference, the developmental category is as stable as the
rest, so its weakness across cohorts is a property of the SSC milestone parsing, not of the
class.

The adjusted Rand index is more moderate, 0.63 across the multi-initialisation fits and 0.65
(standard deviation 0.14) across the subsamples. The two measures answer different questions:
the profile correlation asks whether the class definitions reproduce, and they do; the Rand
index asks whether individual probands keep the same class, and agreement there is partial
because probands near a class boundary move between fits. The reference classes are stable as
structures, with softer membership at the edges.

:::{figure} /_figures/stability.png
:alt: Profile and membership stability of the reference fit under subsampling
:width: 100%
:align: center

Stability of the reference fit under subsampling, from an `analysis stability` run (50 refits
on random halves). (a) The seven-category profile correlation clusters near 0.92, while the
adjusted Rand index is more spread, around 0.65: the class definitions reproduce, the
proband-level membership less so. (b) The per-category correlation is uniformly high,
developmental included, in contrast to its weaker cross-cohort replication. (c) The mean
class-overlap matrix; its diagonal, each class's retention, runs from 0.83 to 0.87.
:::

## The minimum viable stratum size

The stratified analysis splits the cohort by age at diagnosis and by diagnostic era, and a
four-class model over more than two hundred mixed features needs enough probands per class to
estimate its densities. The `nmin` stage fixes the floor empirically rather than by a rule of
thumb. It refits at descending sample sizes and records, at each size, the smallest class
proportion, the scaled relative entropy, the average posterior certainty, and the profile
correlation to the full-sample reference. The floor is read from a monotone (isotonic) fit of
profile correlation against log-size: the smallest size at which the fitted recovery reaches
the reproduction benchmark, with a bootstrap confidence interval. Pooling every fit makes this
robust to the scatter a small replicate count produces, where the smallest individually
clearing size is not (that size is still reported, for continuity). The floor becomes the lower
bound on the stratification bins.

## The stratum-size floor

A first sweep at three replicates per size gave a tidy-looking 1,170 from the
smallest-clearing-size rule, but its per-size correlation bounced from 0.81 to 0.98 with no
clean fall, so the figure was noise. A concentrated sweep settles it better: six sizes from 600
to 5,000 (where the crossing sits), ten replicates each, against the $r \ge 0.90$ benchmark the
reproduction meets. On SPARK 2026-03-23 the isotonic fit puts the floor at about 1,000 probands,
with a 90 per cent bootstrap interval of 600 to 3,955; the smallest individually clearing size
agrees at 1,000.

The recovery measure is noisy in its own right, not only under-sampled. Even at ten replicates
the per-size mean correlation hovers around the benchmark across the whole range (0.79 at 600,
then 0.93, 0.87, 0.91, 0.89, 0.92 up to 5,000) rather than climbing cleanly, so two independent
fits of a several-thousand-proband subsample still agree only to about 0.92. The monotone fit
imposes the ordering recovery is expected to follow and the interval carries the residual
scatter, which is why the floor is reported as a range. No fit collapsed a class at any size
down to 600, so the model recovers four non-empty classes throughout and the floor turns on
profile fidelity, the classes themselves surviving.

For setting the stratification bins the interval matters more than the point. At the point
estimate of 1,000 the smallest class (about 15 per cent) holds only around 150 probands for a
profile with several hundred free parameters, which is thin; at the upper bound of about 3,955
it holds around 580. The conservative floor is therefore the upper bound, and the phase-4 bins
are best kept above about 2,000 probands, nearer 4,000 where the smallest class is comfortable.
Tightening the interval further would take many more replicates for little gain, because much of
the scatter comes from the fit itself and only partly from sampling.

:::{figure} /_figures/stratum_size.png
:alt: Recovery against subsample size and the minimum viable stratum size
:width: 100%
:align: center

Recovery against subsample size, from an `analysis nmin` run (six sizes, ten refits each). (a)
Each refit's profile correlation to the full-sample reference, with the per-size mean, the
$r \ge 0.90$ benchmark, and the isotonic floor near 1,000 with its 90 per cent bootstrap
interval (600 to 3,955). The recovery measure hovers near the benchmark across the whole range
rather than climbing cleanly, so the floor is reported as a range. (b) The smallest class
proportion stays clear of zero at every size, so no class collapses.
:::

## Cross-cohort replication

The `replicate` stage tests whether the classes reappear in the SSC. A fresh model is fitted
on SPARK restricted to the features shared with the SSC, then that fitted model predicts
class labels on the SSC. Because both cohorts pass through the one model, the class ids
already correspond, so no cross-cohort label alignment is needed. The replication measure is
the correlation of the seven-category profiles between the two cohorts, the same currency the
authors used to declare replication.

Two points of care. StepMix validates a prediction input by its feature count, not its column
names, so the SSC measurement matrix is reindexed to the exact SPARK column order before
prediction. And the released code reports the correlation without a null; the package adds a
permutation null that shuffles the SSC class labels and recomputes the correlation, so the
observed value is read against chance.

## The replication result

On the SSC 15.3 release the SPARK model projects onto 771 probands, across the 100 features the
two cohorts share, and the seven-category profiles correlate at $r = 0.76$. The permutation null
puts that value beyond chance: among the label shuffles that yield a defined correlation, none
reaches the observed $r$, giving $p = 0.005$. A shuffle that flattens a class profile gives an
undefined correlation and drops from the null, so the $p$-value rests on the shuffles that
produced a usable profile.

:::{figure} /_figures/replication.png
:alt: Cross-cohort replication of the class signatures between SPARK and the SSC
:width: 90%
:align: center

Cross-cohort replication, from an `analysis replicate` run projecting the SPARK model onto 771
SSC probands. (a) Every class-by-category signature value, SSC against SPARK, around the line
of equality ($r = 0.76$). (b) The per-category correlation; the developmental category, the
one built from the SSC milestone parsing rather than a standard instrument, sits well below
the rest.
:::

The correlation is uneven across the seven categories:

| Category | Correlation |
| --- | --- |
| anxiety or mood | 0.91 |
| social or communication | 0.90 |
| restricted or repetitive | 0.80 |
| self-injury | 0.78 |
| attention | 0.71 |
| disruptive behaviour | 0.71 |
| developmental | 0.49 |

Every category built from a standard instrument, the CBCL, the RBS-R, and the SCQ, replicates at
$0.71$ or above, and the two highest, anxiety or mood and social or communication, reach $0.90$.
The developmental category is the exception at $0.49$, and it is the one category built from the
SSC background-history milestone ages, which are parsed from free text here rather than read from
the authors' unreleased clean file (see [parsing the SSC milestone ages](../guides/parsing-ssc-milestone-ages)).
The distance from the published $r = 0.927$ is concentrated in that category, where the SSC
pipeline departs from the authors'; the class structure itself holds across the others. Each
category correlation is taken over the four classes alone and is therefore coarse; the overall
$r$ is taken over the full four-class, seven-category profile.

The class proportions under projection differ from SPARK's: the SSC places about 64 per cent of
probands in Social or behavioural (39 per cent in SPARK) and about 1 per cent in Moderate
challenges (8 per cent in SPARK).

## Determinism and caveats

The released stability, subsampling, and selection fits are not seeded, so they are not
reproducible run to run. Every fit here takes an explicit seed, recorded in the run manifest,
which is the one deliberate divergence from the released procedure for these stages.

Because each unit's seed is derived from its index, these long stages also resume after an
interrupt: `select`, both `stability` modes, and `nmin` checkpoint each completed unit and
continue from where they stopped when re-run, reproducing the same result as an uninterrupted
run. See [the pipeline and its cache](../guides/pipeline-and-caching.md) for how the checkpoints work.

The replication carries two honest caveats. The SSC harmonisation relies on the package's own
milestone handling: the authors used a hand-cleaned background-history file that was not
released, so the raw free-text milestone ages are parsed into months here rather than read
from their clean file. That parsing is what the developmental category rests on, and it is
where the projection departs most from the published profile. And the shared-feature
complete-case reduction leaves a sample below the full SSC release, so the replication is
reported with its sample size and read against the published value rather than offered as an
exact reproduction.
