# $H_0^C$: is the supported number of classes four in every stratum?

:::{admonition} Definition
:class: note

$H_0^C$ (order). Null: the supported number of {term}`latent class`es is $K = 4$ within every
stratum along both axes, with no class splitting or merging. Alternative: the supported order
changes in one or more strata. Estimand: the {term}`general finite mixture model` order per coarse
stratum, read comparatively against the pooled cohort put through the identical procedure.
:::

:::{admonition} Status
:class: tip

Not yet run. The stratified order stage has not been executed, so there is no per-stratum supported
order to report. Running it will decide whether four components are supported within every stratum
of {term}`age at diagnosis` and {term}`diagnostic era`, or whether a class splits or merges in some
stratum, and it will report the pooled reference order the strata are read against.
:::

## Method

The invariance and prevalence questions ask whether the four reference classes change shape or size
along an axis. This question is different: whether the *number* of classes the data support changes
from one stratum to the next. The estimand is the supported order per stratum, and the read is
comparative. Each stratum is judged against the pooled cohort put through the identical procedure,
not against the number four, because the pooled information criteria over-extract at this sample
size (their minimum sits at nine classes, so a raw criterion is not a usable target). Whatever
drives that over-extraction, most likely feature misspecification, is shared by the pooled cohort
and every stratum, so comparing a stratum's supported order to the pooled cohort's cancels it. An
order change is a stratum whose supported order differs from the pooled cohort's, and the pooled
order is itself a reported finding.

The confirmatory statistic is a warm-started parametric bootstrap {term}`likelihood-ratio test`.
The number of components is not a nested comparison a chi-square can adjudicate: the null that puts
one class's mixing weight at zero sits on the boundary of the parameter space, so the usual
asymptotics fail. The bootstrap test builds the null distribution of the statistic
by simulation instead. For one step comparing $K$ classes against $K+1$, the observed statistic is
twice the log-likelihood gain from the extra class, $\text{LR} = 2(\ell_{K+1} - \ell_K)$; datasets
are simulated from the fitted $K$-component model at the stratum's own sample size and each is put
through the same fitting recipe to give a null $\text{LR}$; and the $p$-value is the Phipson-Smyth
add-one bootstrap $p$, so its smallest attainable value is $1 / (B + 1)$ rather than zero. The fits
are measurement-only, on hard subsets of the cohort matrix, so the order question is about the
measurement model alone. The test is valid only when the observed and simulated datasets are fitted
the same way, so the recipe is fixed and applied identically on both sides: the $K$-class fit keeps
the best of a handful of random restarts, and the $K+1$-class fit is warm-started by splitting each
of the $K$ classes in turn (halving its mixing weight into two children and perturbing its
class-conditional parameters symmetrically), giving a good starting point near the class most likely
to divide.

The search is anchored at four classes and steps outward. The primary splitting direction tests
four against five, and while a step rejects it steps up to a cap of seven; the secondary merging
direction tests four against three when the first splitting step does not reject, and steps down to
a floor of two while a step fails to reject. The search stops at the first splitting step that does
not reject, or the first merging step that does, and the supported order is where it stops. The
identical search on the pooled cohort gives the reference order every stratum is read against. Two
further reads sit beside the bootstrap test as corroborators, neither deciding on its own: the
{term}`cross-validated log-likelihood` elbow, whose knee is found by an inline Kneedle construction,
and the adjusted Lo-Mendell-Rubin test, the proper corrected likelihood-ratio referred to a
chi-square. The confirmatory draws are staged: each
step is screened at a smaller number of draws and escalated to the full schedule only where the
screen $p$ is close, so the expensive draws are spent only where the decision is tight.

This is the `analysis order` stage, run per axis and reusing the pooled search across both axes. The
full procedure, its knobs, and its cache are documented in the guide,
{doc}`testing the number of classes <../guides/testing-the-number-of-classes>`.

## Experimental design decisions

- Comparative to the pooled cohort, not to four. The pooled information criteria over-extract at
  this sample size, so a raw criterion target is unusable. Reading each stratum against the pooled
  cohort through the identical recipe cancels the shared over-extraction, and the pooled order is
  reported in its own right.
- The cross-validated elbow, not the raw information criteria. The information criteria fall across
  the whole grid and reach their minimum at nine classes on the pooled cohort, the known
  over-extraction of these criteria at a large sample, so they do not point to a usable order. The
  out-of-sample cross-validated log-likelihood gains little past four classes and is the honest
  substitute; its knee is read objectively by the Kneedle construction rather than off a plot.
- Coarse strata, not the finest bins. The strata are the four equal-frequency quantile bins of the
  axis. A thousand-proband bin cannot estimate a fifth class, so a null result in a bin that small
  would be a power artefact rather than evidence of stability; the finer binning is deliberately not
  used for this question.
- Three corroborators must agree for a positive claim. A stratum matching the pooled order is the
  stability result, and the bootstrap test not rejecting past the pooled order is enough for it. A
  positive order-change claim asks for more: the bootstrap order must differ from the pooled order,
  the cross-validated elbow knee must move off it too, and the adjusted Lo-Mendell-Rubin test must
  agree at the decisive comparison. Requiring the three to agree guards against reading a single
  close call as a genuine change.

## Results

The stratified order test is not yet run, so there are no results to report. When it runs it will
report, per axis, the pooled reference order and each stratum's supported order anchored at four,
together with the per-step bootstrap $p$-values, the cross-validated elbow knee, the adjusted
Lo-Mendell-Rubin $p$, whether the three corroborators agree, and whether the order changed against
the pooled cohort. The read is comparative, so the finding will be stated as whether every stratum
matches the pooled order (stability) or some stratum departs from it (a class splitting or merging).
No numbers or figures are shown here until the stage has been executed.

## Handling the null

When the stage runs, the decision rule will be applied in the conditional. A stratum whose supported
order matches the pooled order is the stability result, and the bootstrap test failing to reject
past the pooled order is enough to keep $H_0^C$ for that stratum. $H_0^C$ would be rejected in a
stratum only if three conditions hold together: the bootstrap order differs from the pooled order,
the cross-validated elbow knee moves off the pooled order in the same stratum, and the adjusted
Lo-Mendell-Rubin test agrees at the decisive comparison. A single close call among the three is not
enough. Degeneracy is handled by the identical-recipe null itself, since a spurious empty-class fit
that inflates the observed statistic inflates the null draws the same way; only numerically broken
fits are dropped, and a stratum is flagged if more than one draw in ten is dropped.

## Discussion

$H_0^C$ is the third of the primary, unconditionally tested questions, alongside the invariance and
prevalence reads. It asks whether the four-way partition itself survives stratification, rather than
whether its class shapes or sizes drift. Because the pooled criteria over-extract, the answer is
framed relative to the pooled cohort rather than to an absolute count of four, and a positive
order-change claim is deliberately demanding. The verdict, and the pooled reference order it rests
on, await the run.

## See also

- {doc}`Testing the number of classes <../guides/testing-the-number-of-classes>`, the guide to the
  `analysis order` stage this page describes.
- {doc}`Selecting the number of classes <../archive/selecting-the-number-of-classes>`, the pooled
  investigation into how many classes the data support, with the selection-criteria figure.
- {doc}`Are the class profiles invariant, and is any drift small? <../hypotheses/h0a-invariance>`
  ($H_0^A$ and $H_0^D$), the sibling profile-invariance read.
- {doc}`The Python API <../reference>` for the `order` stage.
