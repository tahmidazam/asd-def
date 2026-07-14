# Choosing the stratification bins

The `strata-describe` stage builds the two timing axes and fixes the hard bins the stratified refit
runs on. No model is fitted here: it is the feasibility step that precedes the fits.

Two of its outputs serve every read on the site. It builds {term}`age at diagnosis` and the
reconstructed {term}`diagnostic era` for the modelling cohort, the {term}`measurement lag` between
evaluation and diagnosis, and a numeric demographics frame. The kernel reads run their focal grid
along these axes and adjust for the lag; only the hard bins below are specific to the refit path.

## Binning and the acceptance requirements

Binning is a swappable policy: fixed bands (clinical age bands, or era bands anchored on the
{term}`DSM-5 boundary`), quantile bins (equal-frequency at a fixed count), or max-equal bins (the
finest equal-frequency split that keeps every bin at or above a floor, so the resolution follows the
cohort). Each candidate is checked against a tiered requirement set before it is eligible: hard
eligibility gates (every bin clears the recovery floor, can populate the smallest of four classes,
and covers the cohort), and reported flags for confounding and balance (no dominant bin, tolerable
entanglement with the lag, balanced demographics, stable edges). The floor follows the phase-2
finding that four-class recovery holds from about 1,000 probands (see {doc}`the minimum stratum size
<../appendix/the-minimum-stratum-size>`).

```
uv run analysis strata-describe
```

## The chosen scheme

On SPARK 2026-03-23 (11,704 probands) the max-equal split at a floor of 1,000 is the primary scheme:
11 age-at-diagnosis bins of 1,003 to 1,097 probands and 10 diagnosis-year bins of 1,148 to 1,196.
The era axis tops out at 10 because the reconstructed year has ties that collapse the finer edges
below the floor. The quantile split and a continuous, bin-free trend are the sensitivity analyses,
and the fixed bands are a coarse check.

One caveat carries into the era analysis: at this resolution the small-lag subsample, used to
separate a genuine era effect from the {term}`measurement lag`, no longer keeps two bins above the
floor, so the continuous-trend and covariate-adjusted era checks carry more of that weight.
