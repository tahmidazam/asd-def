# Choosing the stratification bins

The stratified analysis re-estimates the mixture model within strata of a continuous variable,
age at diagnosis or the calendar year of diagnosis, and asks whether the four classes hold. The
finer the strata, the more homogeneous each is in the stratifying variable, but the fewer
probands it holds, and a four-class mixture over the author feature set needs enough probands
per class to estimate it. Granularity is therefore bounded below by sample size, and the bins
have to be chosen and checked before any stratum is fitted.

The package makes the binning a swappable policy and checks each candidate against a set of
acceptance requirements. This guide describes the policies, the requirements, and the
`strata-describe` stage that builds the axes and runs the check. No model is fitted here: this
is the feasibility step that fixes the bins, separate from the stratified fits that follow.

## Binning policies

A policy turns a continuous variable into an ordered set of named strata. Three are defined:

- **Fixed bands** cut at explicit, substantively chosen edges: clinical age bands, or
  calendar-era bands anchored on the 2013 DSM-5 boundary.
- **Quantile bins** cut at the variable's own quantiles, for a fixed number of equal-frequency
  strata.
- **Max-equal bins** are equal-frequency too, but the bin count is not fixed: it is the finest
  equal-frequency split that still keeps every bin at or above a minimum size, so the resolution
  follows the cohort and the floor rather than a hand-picked count.

Every policy returns the same thing, a stratum assignment: the ordered labels, the cut points,
the per-proband stratum, and a serialisable specification recorded in the run manifest. The
stages that consume a partition are typed against the policy interface, not against a concrete
policy, so the stratifying step is independent of which policy drew the bins and a policy can be
swapped without touching the fitting code.

## The acceptance requirements

A policy is eligible for the stratified fit only if its partition can be fitted and is not
silently confounded. Each candidate is evaluated against a tiered requirement set, computed from
the cohort, the stratifying variable, the measurement-to-diagnosis lag, and the demographics.

- **Eligibility gates** (hard pass or fail): every bin clears the recovery floor; each bin can
  populate the smallest of four classes; coverage of the modelling cohort is high; and the
  partition is valid. A policy that fails any gate is ineligible.
- **Confound and balance** (reported as a flag, not a failure): no bin dominates the partition;
  the variable is not strongly entangled with the lag, and a small-lag subsample still supports
  the strata; the demographics are not strongly imbalanced across bins; and the edges are stable
  to a small perturbation.
- **Demographics**: a per-bin summary with standardised differences across the extreme bins,
  both for the manuscript's per-stratum table and as a check that any later drift is not simply a
  composition artefact.

The floor follows the phase-2 finding that four-class recovery holds from about 1,000 probands
(see [the minimum viable stratum size](../investigations/the-minimum-stratum-size)), enough to
populate the smallest of the four classes. The thresholds are settable and recorded in the
report, so the values a policy was judged against travel with it.

## Running the check

```
uv run analysis strata-describe
```

The stage builds age at diagnosis and the reconstructed diagnosis year for the modelling cohort,
the lag between evaluation and diagnosis, and a numeric demographics frame, reading only the
timing and demographic columns each needs through the catalogue. It then evaluates the fixed
bands, a quantile split, and the max-equal split on both axes, and writes the requirement
results, the per-bin counts, the demographic table, and the variable distributions to
`artefacts/strata-describe/<run-hash>/`. The per-policy eligibility verdict and the flags are
printed and recorded in the manifest.

## The chosen scheme

On SPARK 2026-03-23 (11,704 probands, both axes present for over 99 per cent of them) the
max-equal split at a floor of 1,000 is the primary scheme for both axes. It gives 11
age-at-diagnosis bins of 1,003 to 1,097 probands and 10 diagnosis-year bins of 1,148 to 1,196.
The era axis tops out at 10 rather than 11 because the reconstructed year has ties that collapse
the finer quantile edges below the floor. The quantile split and a continuous, bin-free trend
are the sensitivity analyses; the substantive fixed bands are kept as a coarse check, with the
oldest age band (diagnosed at 11 or older, 906 probands) merged into a 7-and-older band to clear
the floor.

Two points carry into the stratified analysis. Each max-equal bin sits close to the floor, so
its smallest class is near 150, the thin end of the range where recovery holds. And the fine
resolution weakens one defence of the era axis: the small-lag subsample, used to separate a
genuine era effect from the lag between measurement and diagnosis, no longer keeps two bins above
the floor, so the continuous-trend and covariate-adjusted era checks carry more of that weight.
The stratified fits, the drift metrics, and their null come next; this stage only fixes the bins
they run on.
