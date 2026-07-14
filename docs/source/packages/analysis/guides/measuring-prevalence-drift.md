# Measuring prevalence drift

The invariance stages ask whether the four reference classes change shape along an axis. The
prevalence stage asks a separate question: whether the classes change size. It tests the $H_0^B$
hypothesis, that the class mixing proportions are constant across diagnostic era and across age at
diagnosis, against the alternative that at least one class's proportion trends along an axis. The
estimand is the mixing proportions as a function of the axis.

This is cheap because nothing is re-estimated. The four classes are already fixed by the
measurement-only reference fit, which gives every proband a posterior over the classes. The stage
regresses class membership on the axis using those frozen posteriors, so there is no refit and no
per-stratum mixture. Prevalence drift is reported separately from profile drift: a class can hold
its shape while its size moves, or move its shape while its size holds, and conflating the two is
exactly the confusion a per-stratum refit invites.

## The classify-analyse bias, and why a naive slope is not enough

The obvious test regresses each proband's hard (modal) class label on the axis with a multinomial
logit. It is transparent but biased. The hard label is a noisy version of the true latent class:
the reference posterior is not a point mass, so a share of the modal labels are wrong. Regressing a
mislabelled outcome on the axis attenuates a real slope toward zero and can manufacture a spurious
one. This is the classify-analyse bias of three-step latent-class analysis (Vermunt 2010,
*Political Analysis*; Bakk, Tekle and Vermunt 2013, *Sociological Methodology*).

The stage therefore leads with a bias-corrected estimator and reports the naive one beside it as a
transparent cross-check.

## The maximum-likelihood three-step correction

The correction treats the modal assignment as a single categorical measurement of the true latent
class, whose class-conditional error probabilities are known. Write $D_{cs}$ for the probability of
assigning a proband to class $s$ when its true class is $c$. This confusion matrix is estimated
directly from the frozen posteriors (the same matrix StepMix builds for its own three-step
estimator, `stepmix.stepmix.compute_bch_matrix`), so it is fixed, not fitted.

The structural model is a multinomial logit of the true latent class on the axis. It is fitted by
expectation-maximisation with the measurement model held at $D$: the E-step forms each proband's
posterior over the true class from its axis-implied prior and the fixed likelihood $D_{c s_i}$ of
its observed assignment $s_i$, and the M-step refits the logit to those soft targets. No mixture is
re-estimated; only the small structural logit moves. Because the error probabilities enter
explicitly, the estimator removes the attenuation the naive slope carries.

This implements the correction directly rather than through StepMix's own `fit` path. That path
re-estimates the measurement model in its first step, which would defeat the point of freezing the
reference, and its covariate structural model is numerically unstable under fractional weights on
this cohort. The structural logit here is a few coefficients solved by a penalised Newton step, so
it is stable.

## What the stage reports

For each axis, and per class:

- the one-versus-rest axis slope, the log-odds of membership per axis unit, with its odds ratio.
  This is the most interpretable read of whether a class's prevalence trends with the axis. It is
  reported for the corrected estimator (with a bootstrap interval and $p$) and the naive estimator
  (with the closed-form Wald and likelihood-ratio $p$-values, uncorrected);
- the predicted proportion as a smooth function of the axis over a grid, with a confidence band.
  This is the estimand itself, the mixing proportion $\pi_k$ read as $\pi_k(\text{axis})$;
- the joint likelihood-ratio test of `class ~ axis` against `class ~ 1`, the omnibus question of
  whether any class's proportion trends, reported for both estimators.

Uncertainty is a family-clustered bootstrap resampling SPARK families, the same convention as the
invariance-trajectory stage: each replicate resamples whole families with replacement, recomputes
the confusion matrix and the corrected fit, and records the slopes and the proportion curve. The
percentiles of the draws are the intervals and the band, and each corrected slope's significance is
a two-sided add-one bootstrap $p$-value. Resampling families rather than probands keeps siblings
together, so a within-family correlation widens the interval honestly.

Significance is Benjamini-Hochberg controlled at $q = 0.05$ across the per-class contrasts within
an axis, separately for each estimator. Effect sizes with intervals accompany every $p$, and a null
result is reported symmetrically with a drift result.

## Directionality and the adjusted model

The linear axis term is the primary directional read. For the era axis the stage adds a DSM-5
pre/post-2013 contrast, the per-class log-odds of membership after the 2013 diagnostic-criteria
change against before it, matching the descriptor overlay of the other era reads.

An adjusted sensitivity fits the corrected axis slope net of sex, the measurement-to-diagnosis lag,
and age at evaluation. A prevalence trend that survives this adjustment is more than a
measurement-timing or sex-composition artefact, mirroring the $H_0^H$ conditioning of the
invariance work.

## Correctness gates

The stage is checked on synthetic data only (no participant data enters a test):

1. the confusion matrix of a one-hot posterior is the identity, and a noisy posterior gives a
   diagonally dominant matrix;
2. the soft-target multinomial logit recovers planted coefficients on clean data;
3. a planted proportion trend is recovered with the right sign and survives the FDR, and the joint
   test rejects a constant-proportion null;
4. with a known classification error and a known slope, the correction de-attenuates the naive
   hard-label slope and carries a smaller bias on average, the property that motivates it;
5. a flat proportion is a null: the corrected slopes cover zero and the FDR rejection rate stays
   near nominal across seeds.

## Reading the result

At the full sample size the joint test rejects a constant-proportion null decisively on both axes,
so the read is the per-class slope, its sign, and its size, not the binary reject. A positive slope
is a class growing along the axis, a negative slope a class shrinking. Because the classes are
frozen, a slope here is prevalence drift alone, uncontaminated by the profile drift that a
per-stratum refit would fold in. Where a per-stratum refit suggested a proportion change that $H_0^B$
does not confirm, the refit change was profile drift reassigning members rather than a genuine size
change, and the discrepancy is itself informative.

The `figures prevalence --axis {era, age_at_diagnosis}` command draws the estimand directly. The
default `--layout panels` gives one panel per class: the corrected proportion curve with its
bootstrap band, the naive hard-label curve as a thin dashed cross-check, a dotted line at the pooled
(axis-free) proportion the class trends away from, and the per-class slope and odds ratio in the
panel title. The `--layout stacked` view stacks the four corrected proportions to one across the
axis, so the compositional shift, one class growing as another shrinks, reads as a single figure.
