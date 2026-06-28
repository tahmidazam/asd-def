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
