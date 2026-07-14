# Glossary

## Domain terms

```{glossary}
autism class
  One of the four latent groups the reference model assigns each participant to, the classes
  of Litman et al. {footcite}`litmanDecompositionPhenotypicHeterogeneity2025a`: Social or
  behavioral, Mixed ASD with developmental delay, Moderate challenges, and Broadly affected. A
  class is a {term}`latent class` of the {term}`general finite mixture model`, not an observed
  label.

latent class
  A group that is not directly observed but inferred from the pattern across many measured
  features (a [latent class](https://en.wikipedia.org/wiki/Latent_class_model)). Each
  participant has a {term}`responsibilities` vector giving the probability of belonging to
  each class rather than a hard assignment.

general finite mixture model
  The model behind the classes (GFMM): a fixed number of latent classes, each with its own
  class-conditional distribution over the measured features, mixed in the population by the
  {term}`prevalence` weights (a
  [finite mixture model](https://en.wikipedia.org/wiki/Mixture_model)). Fitted by
  {term}`expectation-maximization` with StepMix
  {footcite}`morinStepMixPythonPackage2025`, which estimates this family of models, and whose
  three-step methods run the {term}`prevalence` test.

profile
  The class-conditional measurement parameters that define one class: the feature-level
  Gaussian means and variances, Bernoulli probabilities, and multinomial category
  probabilities. The invariance hypotheses ask whether the profiles move along an axis.

prevalence
  The [share of the population](https://en.wikipedia.org/wiki/Prevalence) in a class, also
  called the mixing proportion or latent class size. Whether the prevalences shift along an
  axis is tested separately from profile drift ($H_0^B$).

age at diagnosis
  The participant's age in years when autism was first diagnosed. One of the two axes; the
  corroborating axis for the headline question.

diagnostic era
  The reconstructed calendar year of diagnosis, treated as a continuous axis. The axis for the
  headline question of whether the classes drift with diagnostic timing.

DSM-5 boundary
  The 2013 revision of the [diagnostic manual](https://en.wikipedia.org/wiki/DSM-5), which
  merged the earlier autism sub-diagnoses into one spectrum. A candidate discontinuity in the
  {term}`diagnostic era` axis: diagnoses from 2013 onward.

referent
  The time frame an instrument asks about. A current-state instrument (for example RBS-R,
  CBCL 6-18) reports the present; a retrospective or lifetime instrument (SCQ-Lifetime,
  developmental milestones) reports the past. The split separates a measurement-timing change
  from a change in the diagnosed population ($H_0^G$).

measurement lag
  The gap between the diagnosis and the parent-report measurement. A nuisance covariate the
  era analysis adjusts for, since a change in the lag could otherwise be mistaken for era
  drift.

seven phenotype categories
  The author-defined symptom groupings the features fall into: anxiety or mood, attention,
  disruptive behaviour, self-injury, social or communication, restricted or repetitive, and
  developmental. Drift is attributed across these categories ($H_0^F$).

Litman feature set
  The 238 parent-report features from Litman et al.
  {footcite}`litmanDecompositionPhenotypicHeterogeneity2025a` that the reference model is
  fitted over, the measurement set for every stability test on this site.

SPARK
  The [Simons Foundation Powering Autism Research](https://www.sfari.org/resource/spark/)
  cohort, the primary parent-report dataset; the reference model is fitted to the SPARK cohort.

SSC
  The [Simons Simplex Collection](https://www.sfari.org/resource/simons-simplex-collection/), a
  separate autism cohort used to check whether the reference classes replicate outside SPARK.
```

## Method terms

```{glossary}
responsibilities
  The posterior class-membership probabilities the model assigns each participant, one per
  {term}`latent class`, summing to one. The soft counterpart of a hard class label: the
  {term}`three-step estimator` regresses on these rather than on a single assigned class.

expectation-maximization
  The iterative algorithm that fits the {term}`general finite mixture model`
  ([expectation-maximization](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)):
  it alternates estimating the {term}`responsibilities` given the parameters with
  re-estimating the parameters given the responsibilities.

measurement invariance
  The property that a measurement relates to the latent construct the same way across groups
  or along an axis
  ([measurement invariance](https://en.wikipedia.org/wiki/Measurement_invariance)). The
  invariance nulls ($H_0^A$) state that the {term}`profile` of each class is invariant to the
  axis.

score-based invariance test
  A test for {term}`measurement invariance` that scans the per-participant model scores,
  ordered along the axis, for a systematic fluctuation. The generalisation to a continuous
  axis without pre-set subgroups is the Merkle and Zeileis score-based
  test {footcite}`merkleTestsMeasurementInvariance2013`; it is built on an
  {term}`empirical fluctuation process` and corroborates the effect-size read.

empirical fluctuation process
  The cumulative sum of the ordered model scores, which behaves like a {term}`Brownian bridge`
  under the invariance null, so an unusual excursion signals a break along the axis.

Brownian bridge
  A [Brownian motion pinned to zero at both ends](https://en.wikipedia.org/wiki/Brownian_bridge).
  The reference process for the {term}`empirical fluctuation process` under the null; at the
  full sample size its $p$-value saturates, so the read moves to the effect size.

family-clustered bootstrap
  A [resampling scheme](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) that draws
  whole families rather than individuals, so the intervals respect the dependence between
  related participants. The source of every {term}`confidence band` on this site.

confidence band
  The [band](https://en.wikipedia.org/wiki/Confidence_and_prediction_bands) around an
  effect-size trajectory produced by the {term}`family-clustered bootstrap`. A band sitting
  above the control level is the positive read for a specific drift.

false discovery rate
  The [expected share of false positives](https://en.wikipedia.org/wiki/False_discovery_rate)
  among the rejected nulls. Controlled by the Benjamini-Hochberg procedure across the
  class-by-parameter tests within an axis.

Mahalanobis distance
  A [distance between a point and a distribution](https://en.wikipedia.org/wiki/Mahalanobis_distance)
  that accounts for the covariance, so it reads in the natural units of the spread. Used to
  scale a centroid shift against the {term}`between-class separation`.

linear discriminant analysis
  The [projection that best separates the classes](https://en.wikipedia.org/wiki/Linear_discriminant_analysis).
  Its plane defines the {term}`between-class separation`; a {term}`capture fraction` flags
  drift lying outside it.

between-class separation
  How far apart the classes sit: the mean per-feature (root-mean-square) distance between their
  centroids. A drift read as a fraction of it is a fraction of a class gap, so the size is judged
  against how distinct the classes are rather than in raw feature units, and a value near one is a
  class that moved about as far as two classes are apart. The trajectory figures instead plot a
  displacement "in separation units", an un-averaged norm that sums the per-feature displacements
  rather than averaging them, so it grows with the number of features and reads as a comparative
  scale across axes and controls, not a count of class gaps. Divide a separation-unit value by the
  square root of the number of features it sums over (about 15 for a whole-class read over the 238
  features) to recover the fraction of a class gap.

capture fraction
  The share of a class's centroid drift that lies within the {term}`linear discriminant
  analysis` plane. A low value flags drift in a direction the between-class axes do not span.

local structural equation modelling
  Re-estimating the model at each point on a continuous axis, weighting every proband by a
  kernel around that focal point so the class parameters come out as smooth trajectories
  rather than a handful of disjoint strata (LSEM {footcite}`hildebrandtExploringFactorModel2016`).
  The kernel bandwidth sets how local each fit is. The `KernelWindows` scheme in
  {py:mod}`analysis.localise` implements it.

multinomial logistic regression
  A [regression of a categorical outcome](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)
  with more than two levels on covariates. Used to regress latent class on an axis for the
  {term}`prevalence` test.

three-step estimator
  A way to relate latent class to a covariate without refitting the mixture: fit the mixture,
  fix the classification-error matrix of the posteriors, then regress class on the covariate
  with that error accounted for. Removes the bias of regressing on hard labels.

differential item functioning
  A [feature that relates to the latent class differently](https://en.wikipedia.org/wiki/Differential_item_functioning)
  across groups or along an axis, net of the class itself. The direct-effect route for testing
  whether era drift survives adjustment for {term}`measurement lag` ($H_0^H$).

likelihood-ratio test
  A [test comparing two nested models](https://en.wikipedia.org/wiki/Likelihood-ratio_test) by
  twice the difference in their log-likelihoods. Used for order selection (a bootstrap variant
  for the number of classes) and as an invariance cross-check.

cross-validated log-likelihood
  The held-out log-likelihood, averaged over
  [folds](https://en.wikipedia.org/wiki/Cross-validation_(statistics)). Its elbow selects the
  number of latent classes, in preference to the raw information criteria, which over-extract
  at this sample size ($H_0^C$).
```

## Sources

```{footbibliography}
```
