## v0.7.0 (2026-07-15)

### Feat

- **figures**: stacked-pair prevalence composition
- **figures**: add presentation figure-builder command
- **analysis**: add timing covariates to conditioning screen
- **figures**: demographic conditioning heatmap
- **analysis**: condition the class drift on demographics
- displacement atlas over non-modelling ordering axes
- **analysis**: per-category bootstrap CI on the replication
- **figures**: H0F and H0G paper figures
- **analysis**: block-attribution engine, H0G on the size-fair grain contrast
- **figures**: add brief command for brief figures
- **analysis**: add ATTR-REF referent decomposition of era drift
- **analysis**: add ORDER number-of-classes search
- **figures**: swap sex for household income in specificity panel
- **analysis**: prevalence-drift test (PREV)
- **analysis**: add specificity paired-bootstrap p-value
- **analysis**: DIREC directional drift statistic
- **analysis**: null-free effect-size recast of the invariance test
- **analysis**: score-based measurement-invariance test
- **figures**: pairwise class-drift trajectory figure
- **analysis**: observed pairwise drift via --reference-scheme
- **analysis**: pluggable reference scheme for stratum drift
- **analysis**: cross-cohort cognitive stratification axis
- **analysis**: measurement-only reference via fit --no-covariates
- **analysis**: measurement-only fit option for the kernel sweep
- **analysis**: build the class trajectory from a kernel sweep
- **figures**: kernel-sweep trajectory figure
- **analysis**: choose the kernel bandwidth by effective sample size
- **analysis**: kernel/LSEM localisation and the sweep conductor
- **figures**: add class-churn attribution figures and count joiners
- **figures**: draw class coverage ellipses on the trajectory panels
- **figures**: plot class trajectories and roughness across the strata
- **analysis**: add class-movement attribution and the trajectory stage
- **analysis**: add Mahalanobis and Jensen-Shannon distances, default to Mahalanobis
- **analysis**: decouple drift measurement from fitting; align by membership
- **analysis**: measure class drift against the permutation null
- **analysis**: fit the GFMM within each stratum
- **analysis**: assign probands to the frozen strata
- **analysis**: add resource profiling and per-run hardware capture

### Fix

- **analysis**: scale invariance magnitudes by the full-norm gap
- **docs**: ignore stepmix's invalid-escape warning
- **analysis**: align a measurement-only sweep to the measurement-only reference
- **analysis**: resolve the reference fit at the canonical restart count
- **analysis**: drop fits that diverge to non-finite log-likelihood

### Perf

- **analysis**: fit every stage's seeded loops concurrently

## v0.6.0 (2026-06-28)

### Feat

- **analysis**: add binning policies and the strata feasibility check
- **analysis**: resolve SSC milestone age scale against the SPARK distribution
- **figures**: show the V9 cut as a condition in the existing figures
- **figures**: compare class proportions and selection across cohort cuts
- **analysis**: reconstruct the V9 cohort via a records cutoff

### Fix

- **analysis**: key the replicate cache on the SSC frame content

## v0.5.0 (2026-06-23)

### Feat

- **figures**: overlay the published per-category values on the replication figure
- **figures**: restyle the figures with bold left-aligned panel titles
- **figures**: show the bootstrap interval on the reproduction and replication figures
- **analysis**: add bootstrap confidence intervals to the reproduction and replication correlations

### Fix

- **analysis**: recover the SSC replication to match the authors
- derive package __version__ from installed metadata

## v0.4.0 (2026-06-20)

### Feat

- **figures**: render the analysis results and publish them into the documentation
- **analysis**: add resumable checkpointing and a non-convergence guard to the long stages
- **analysis**: add phase-2 stability, selection, and replication tooling
- **analysis**: parse SSC free-text milestone ages onto the schema
- **analysis**: build the reference pipeline (cohort, fit, align)

## v0.3.0 (2026-06-14)

### Feat

- **analysis**: scaffold the analysis package for the Litman stability study

## v0.2.0 (2026-06-14)

### Feat

- **dscat**: add --convert-docs flag and an ingest progress bar
- **docs**: build versioned docs with a pydata version switcher
- **dscat**: convert documents through per-format conversion engines

## v0.1.0 (2026-06-13)

### Feat

- add dscat and docs packages with uv monorepo setup
