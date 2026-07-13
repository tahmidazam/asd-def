# Testing the number of classes

The invariance and prevalence stages ask whether the four reference classes change shape or
size along an axis. The order stage asks a different question: whether the *number* of classes
the data support changes from one stratum to the next. It tests the ORDER hypothesis, that four
components are supported within every stratum of age at diagnosis and diagnostic era, against
the alternative that the supported number rises (a class splits) or falls (two classes merge)
in some stratum.

The estimand is the supported number of classes per stratum, and the read is comparative: each
stratum is judged relative to the pooled cohort put through the identical procedure, not against
the number four. This matters because the pooled information criteria over-extract at this sample
size (their minimum sits at nine classes, so a raw criterion is not a usable target). Whatever
drives that over-extraction, feature misspecification most likely, is shared by the pooled cohort
and every stratum, so comparing a stratum's supported order to the pooled cohort's cancels it. An
order change is a stratum whose supported order differs from the pooled cohort's; the pooled
order is itself a reported finding.

## The confirmatory statistic: a warm-started bootstrap likelihood-ratio test

The number of components is not a nested-model comparison a chi-square can adjudicate: the null
that puts one class's mixing weight at zero sits on the boundary of the parameter space, so the
usual asymptotics do not hold. The parametric bootstrap likelihood-ratio test (BLRT) sidesteps
that by building the null distribution of the statistic by simulation (McLachlan 1987).

For one step, comparing $K$ classes against $K+1$:

- the observed statistic is $\text{LR} = 2(\ell_{K+1} - \ell_K)$, twice the log-likelihood gain
  from the extra class;
- the null is parametric: datasets are simulated from the fitted $K$-component model at the
  stratum's own sample size, and each is put through the same fitting recipe to give a null
  $\text{LR}$;
- the $p$-value is the Phipson-Smyth add-one bootstrap $p$, so the smallest attainable value is
  $1 / (B + 1)$ rather than zero (the same correction the drift null uses).

The fits are measurement-only (`fit_gfmm(structural=None)`, the `fit --no-covariates` recipe),
on hard subsets of the 238-feature cohort matrix, so the order question is about the measurement
model alone.

### Why the recipe must be identical, and how $K+1$ is fitted

The BLRT is only valid if the observed and simulated datasets are fitted the same way. If the
$K+1$-class fit is allowed to under-fit on the null datasets (a random restart that misses the
best optimum), the null $\text{LR}$ is biased low and the test rejects too readily. So the recipe
is fixed and applied identically on both sides:

- the $K$-class fit uses a handful of random restarts, keeping the best;
- the $K+1$-class fit is warm-started by splitting each of the $K$ classes in turn. Splitting a
  class halves its mixing weight into two children and perturbs its class-conditional parameters
  symmetrically, giving a good starting point near the class most likely to divide. That is $K$
  warm starts; a couple of random restarts are added, and the best fit is kept.

Because StepMix reinitialises its emission parameters at the start of every EM run, a warm start
cannot go through its `fit`. The warm start drives the EM loop directly instead: it sets the split
parameters, then alternates StepMix's own expectation and maximisation steps until the average
log-likelihood settles. The optimiser is StepMix's; only the starting point changes.

## The anchored sequential search

The search is anchored at four classes and steps outward (plan section 7):

- the primary (splitting) direction tests four against five, and while a step rejects it steps
  up, five against six and on, to a cap of seven. If the search is still rejecting at the cap it
  reports the order as `>=7`;
- if the first splitting step does not reject, the secondary (merging) direction tests four
  against three, and while a step fails to reject it steps down, three against two, to a floor of
  two.

The search stops at the first splitting step that does not reject, or the first merging step that
does. The supported order is where it stops. Running the identical search on the pooled cohort
first gives the reference order $K^*_\text{pooled}$ that every stratum is read against.

## Staged bootstrap draws

The confirmatory number of draws is large ($B = 999$), and most steps are settled well before
that. Each step is therefore screened at $B = 199$ draws, and escalated to $B = 999$ only where
the screen $p$ falls below $0.10$, so the expensive draws are spent only where the decision is
close. The draws are independent and run concurrently; each is a self-contained unit (simulate one
null dataset, refit, return its $\text{LR}$), streamed to a resumable checkpoint, so an interrupt
continues from the first draw not yet computed.

## The two corroborators

Two further reads sit beside the BLRT. Neither decides on its own.

- The cross-validated log-likelihood elbow. The validation log-likelihood rises with the number
  of classes and flattens; the knee is where the diminishing returns set in. The knee is found by
  an inline Kneedle construction (Satopaa et al. 2011): normalise both axes to the unit interval,
  and take the class count at the maximum of the difference between the normalised curve and the
  diagonal. This is an objective substitute for reading the elbow off a plot.
- The adjusted Lo-Mendell-Rubin test. This is the proper VLMR (Lo, Mendell and Rubin 2001,
  Formula 15), not the naive one-degree-of-freedom proxy the selection stage reports. The
  likelihood-ratio statistic is corrected by an ad-hoc factor and referred to a chi-square:

  $$
  \text{LMR} = \frac{2(\ell_{K+1} - \ell_K)}
  {1 + \big[((3(K+1) - 1) - (3K - 1)) \ln n\big]^{-1}},
  $$

  with degrees of freedom the difference in free parameters and $p = \Pr(\chi^2_{\text{df}} >
  \text{LMR})$.

## The decision

A stratum whose supported order matches the pooled order is the stability result; the BLRT not
rejecting past the pooled order is enough for it. A positive order-change claim asks for more: the
BLRT order must differ from the pooled order, the cross-validated elbow knee must move off the
pooled order too, and the adjusted Lo-Mendell-Rubin test must agree at the decisive comparison.
Requiring the three to agree guards against reading a single close call as a genuine change.

Degeneracy is handled by the identical-recipe null itself: a spurious empty-class fit that
inflates the observed $\text{LR}$ inflates the null draws the same way, so it is self-calibrated.
Only numerically broken fits (a singular covariate general linear model, a non-finite
log-likelihood) are dropped, and a stratum is flagged if more than one draw in ten is dropped.

## Running it

The stage runs per axis and reuses the pooled search across both axes:

```
uv run analysis order --axis age_at_diagnosis
uv run analysis order --axis era
```

The strata are the four equal-frequency quantile bins of the axis. A thousand-proband bin cannot
estimate a fifth class, so the fine `MaxEqualBins` primary is deliberately not used here: a null
result in a bin that small would be a power artefact rather than evidence of stability. The era
axis adds the DSM-IV against DSM-5 (2013) split as one targeted secondary pair.

The knobs mirror the drift stage. `--b-screen` and `--b-escalate` set the staged schedule (the
frozen values are 199 and 999), `--escalate-threshold` the screen-$p$ that triggers escalation,
`--n-init` the restarts per fit, `--k-anchor`, `--k-cap`, and `--seed` the search. A quick
end-to-end check runs at a small `--b-screen` and `--b-escalate`; the confirmatory read uses the
frozen schedule.

Each run caches a per-stratum decision table (the supported order, the direction, the per-step
BLRT $p$-values, the escalation flag, the elbow knee, the VLMR $p$, the agreement flag, whether
the order changed against the pooled cohort, and the dropped-draw count), the pooled reference
row, and a manifest, under `artefacts/order/<hash>/`. Only class and stratum-level outputs leave
the stage; no per-proband data is written.
