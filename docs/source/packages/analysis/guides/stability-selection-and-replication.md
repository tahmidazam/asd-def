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

## The minimum viable stratum size

The stratified analysis splits the cohort by age at diagnosis and by diagnostic era, and a
four-class model over more than two hundred mixed features needs enough probands per class to
estimate its densities. The `nmin` stage fixes the floor empirically rather than by a rule of
thumb. It refits at descending sample sizes and records, at each size, the smallest class
proportion, the scaled relative entropy, the average posterior certainty, and the profile
correlation to the full-sample reference. The minimum viable size is the smallest swept size
whose profile correlation still holds at the reproduction benchmark. That size becomes the
lower bound on the stratification bins.

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

## Determinism and caveats

The released stability, subsampling, and selection fits are not seeded, so they are not
reproducible run to run. Every fit here takes an explicit seed, recorded in the run manifest,
which is the one deliberate divergence from the released procedure for these stages.

The replication carries two honest caveats. The SSC harmonisation relies on the package's own
milestone handling: the authors used a hand-cleaned background-history file that was not
released, so the raw free-text milestone ages are parsed into months here rather than read
from their clean file. And the SSC release held here is small once reduced to complete cases
on the shared features, so the replication is reported with its sample size rather than as a
clean reproduction of the published value.
