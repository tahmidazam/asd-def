# Score-based measurement invariance

The drift stages ask whether the four reference classes move when the mixture is re-estimated
inside strata of an axis, and pay for a fit and a permutation null in each stratum. The
score-based test asks the same question from the single cached fit, with an analytic null and no
refitting. It follows the empirical fluctuation process of Merkle and Zeileis (2013,
*Psychometrika*) and Merkle, Fan and Zeileis (2014), read here against the axis of age at
diagnosis or diagnosis year.

This guide describes the casewise score, the fluctuation process and its bridge null, the two
statistics, the focal blocks, the correctness gates, and how to read the result. The stage is a
pure consumer of the measurement-only reference fit and the axis, so it never refits the mixture.

## The casewise score

At the maximum-likelihood estimate, each proband contributes a score to every class-profile
parameter, the gradient of its log-likelihood with respect to that parameter. For a mixture,
Fisher's identity makes this the responsibility times the emission gradient: the casewise score
of proband $i$ on the class-$k$ value of feature $j$ is $r_{ik}\,\partial_\theta \log
f_j(x_{ij};\theta_{jk})$, where $r_{ik}$ is the posterior responsibility (the model's
`predict_proba`).

The focal parameters are the class-conditional locations, the profiles the whole analysis
measures. Each emission type has an exact score:

- a Gaussian mean gives $r_{ik}(x_{ij}-\mu_{jk})/\sigma^2_{jk}$;
- a Bernoulli probability gives $r_{ik}(x_{ij}-p_{jk})/(p_{jk}(1-p_{jk}))$;
- a categorical outcome $l$ gives the multinomial-logit score $r_{ik}(\mathbb{1}[x_{ij}=l]-
  p_{jkl})$.

On the reference release the measurement features are 38 continuous, 33 binary and 167
categorical, so the categorical block is the majority and is scored rather than dropped. The
outcomes of a categorical feature sum to a redundant direction, because the outcome probabilities
are a simplex; the whitening below removes it, so no reference outcome is chosen by hand.

## The fluctuation process and its bridge null

Order the probands by the axis and take the running sum of their scores. Standardise it by the
inverse square root of the outer-product-of-gradients covariance of the focal block, which
decorrelates the block's dimensions, and scale by the root of the sample size. The result is the
empirical fluctuation process $B(t)$, $t \in [0, 1]$. Because the fit forces the full-sample score
to zero, the process is pinned to zero at both ends; a small residual tilt, from an optimiser that
stops at a tolerance or from an axis that covers most but not all probands, is removed by a linear
tie-down. Under stability $B(t)$ converges to a Brownian bridge.

Two functionals read the process:

$$ \text{maxLM} = \max_t \lVert B(t) \rVert^2, \qquad
   \text{CvM} = \int_0^1 \lVert B(t) \rVert^2\,\mathrm{d}t. $$

The maxLM is the largest excursion, powerful against an abrupt break, and its location is the
estimated break. The Cramer-von Mises is the integral, powerful against a gradual drift. The null
is drawn by simulating many independent $d$-dimensional Brownian bridges on the axis's own time
grid ($d$ the effective block dimension) and reading the same functional off each. This is a
Gaussian-process simulation of microseconds, not a model refit, so the $p$-value is analytic. The
estimated break carries a confidence set, the axis positions whose excursion clears the null's
maxLM critical value.

A signed directional slope, the ordinary-least-squares slope of each focal parameter's score on
the axis, reports the sense of a drift: a systematic trend rather than a fluctuation about zero.

## Focal blocks and multiplicity

The process is read per block. A whole-class block pools every location parameter of a class; a
class-by-category block restricts a class to one feature category, so a drift can be localised to
a class and a symptom domain. Significance is Benjamini-Hochberg controlled across the blocks
within an axis, the repo convention for the strata-by-class tests. The whole-class blocks have a
much larger dimension than the category blocks, so their raw maxLM is larger; the two are compared
by $p$-value, not by raw statistic.

## The correctness gates

The test is a $p$-value engine, so its scores are validated before any real reading is trusted.
Three gates, on synthetic data, guard it:

1. the analytic casewise score matches a central finite difference of the per-sample
   log-likelihood to machine precision, for every emission type;
2. a synthetic mixture with a planted mean drift along the axis yields a small $p$ in the drifted
   block and localises the break near the planted point;
3. with no drift the $p$-values are uniform, so the analytic null is calibrated rather than
   assumed.

The score validation is the load-bearing one: a wrong gradient would invalidate every $p$-value,
so the same finite-difference check is also run on the real reference fit before it is read.

## Running it

```
uv run analysis invariance --axis age_at_diagnosis
uv run analysis invariance --axis era
```

The stage resolves the measurement-only reference (the `analysis fit --no-covariates` fit), joins
the axis to its probands, drops any with a missing axis value and reports the coverage, and writes
one row per block with the two statistics, their bridge $p$-values, the Benjamini-Hochberg
decisions, the estimated break and its confidence set, and the directional slope. It also stores
the fluctuation process of the strongest-drifting category block, which `uv run figures invariance
--axis age_at_diagnosis` draws against the simulated null band.

`--n-simulations` sets the number of null bridges (2000 by default), `--seed` the null draw, and
`--max-grid` the largest evaluation grid (tied axis values are collapsed, then thinned to this).
The run hash folds in the reference fit, the axis, the focal-block specification, the number of
simulations and the seed, so each configuration caches as its own run.

## Reading the result

The test consumes the marginal (measurement-only) reference, so its estimand matches the kernel
and pairwise arms rather than the covariate fit. At the reference sample size, roughly 11,700
probands, the test has great power, so a rejection means the class profiles are not perfectly
invariant along the axis, which is almost always true to some degree. Two habits keep the reading
honest. First, read the effect sizes and the break locations, not only the binary decisions: which
classes and categories drift most, and where. Second, confirm against a permuted axis, a random
ordering under which there is no drift; the $p$-values there are uniform and nothing survives the
false-discovery-rate step, which is what separates genuine axis-associated drift from the
large-sample tendency to reject.

The relationship of this test to the frozen refit null of the pre-registration, whether it
supplements or replaces it and any re-pre-registration, is a decision recorded in the progress log
rather than taken by the stage.
