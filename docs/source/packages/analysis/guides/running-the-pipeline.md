# Running the pipeline

This guide is the runbook: the commands to reproduce everything implemented so far, in the
order their inputs require. It complements [the pipeline and its cache](pipeline-and-caching),
which explains the staging and caching design. Here the focus is what to run.

## Before the first run

The pipeline reads the SPARK and SSC releases through the `dscat` catalogue, so the datasets
have to be in place under `data/` and the catalogue built once with `uv run dscat ingest` (see
[the cohort interface](the-cohort-interface) for the releases used). The authors' feature list
and category map have to be present as well. Install the environment with `uv sync`. The
`analysis` package is a non-editable path dependency, so after editing its source run
`uv sync --reinstall-package analysis` before the change takes effect at the command line.

Every command is `uv run analysis <stage>`. With no options each stage uses the reference
settings: SPARK `2026-03-23`, four classes, and 200 restarts.

## The stages, in order

The reference solution comes first, then the stability and replication checks that lean on it.

1. Build the cohort matrix and the feature typing:
   ```
   uv run analysis cohort
   ```
   `fit` builds this for you if it is missing, so this step is mainly to inspect the cohort
   (the proband count, the typing counts, and any typing conflict) before fitting.

2. Fit the reference mixture model and predict a class label per proband:
   ```
   uv run analysis fit
   ```
   This is the expensive step, around ten minutes for the 200 restarts.

3. Summarise each class and align the classes to Litman's named classes:
   ```
   uv run analysis align
   ```
   `fit` then `align` are the reference solution. Their result, the per-category profile
   correlation and the named-class anchors, is the phase-1 reproduction gate (see
   [reproducing the reference classes](../investigations/reproducing-the-reference-classes)).

4. Grid over the number of components and report the information criteria:
   ```
   uv run analysis select
   ```

5. Check that the solution survives re-initialisation and resampling:
   ```
   uv run analysis stability --mode multi-init
   uv run analysis stability --mode subsample
   ```

6. Fix the minimum viable stratum size for the later stratified work:
   ```
   uv run analysis nmin
   ```

7. Project the model onto the SSC and correlate the category profiles:
   ```
   uv run analysis replicate
   ```

Stages 4 to 7 are the phase-2 stability and replication checks, each reported in its own
investigation: [selecting the number of classes](../investigations/selecting-the-number-of-classes),
[stability under refitting](../investigations/stability-under-refitting),
[the minimum viable stratum size](../investigations/the-minimum-stratum-size), and
[replicating in the SSC](../investigations/replicating-in-the-ssc). The SSC milestone ages
`replicate` reads are parsed from free text, described in
[parsing the SSC milestone ages](parsing-ssc-milestone-ages).

## What depends on what

- `fit` needs the cohort, and builds it if absent.
- `align` needs a completed `fit`.
- `stability` and `nmin` need a completed `fit` and `align` (the reference). They exit with
  guidance if either is missing for the chosen settings.
- `select` and `replicate` need only the cohort, not the reference, so they can run at any
  point once the data is in place.

## Re-running and the cache

Each stage writes its outputs and a manifest to `artefacts/<stage>/<run-hash>/`. The run hash
is taken over the stage's declared inputs (the dataset and version, the feature-list and
category-map file digests, the hyperparameters, and the seed), so a later run with the same
inputs is a cache hit and returns at once, while a changed input recomputes.

Two consequences are worth keeping in mind:

- A change to a stage's settings, a different release, seed, or restart count, gives a new
  hash and a fresh run, and leaves the earlier run in place.
- A change to the computation that the hash does not capture, for example the SSC
  milestone-age parsing that `replicate` applies, does not change the hash. Pass `--force` to
  recompute in that case, otherwise the stage returns the cached result from before the change.

A long run that is interrupted does not start over. The multi-seed stages, `select`, both
`stability` modes, and `nmin`, checkpoint each completed unit (a seeded iteration, a fit, a
replicate) inside the run directory, so re-running the same command resumes from where it
stopped and reproduces the same result. `--force` discards the checkpoint and recomputes from
the start.

## What is not yet runnable

The `strata`, `stratify`, `drift`, `sensitivity`, and `report` stages are planned for later
phases. They appear under "Planned" in `uv run analysis --help` and exit with a note rather
than running. The pipeline implemented so far ends at `replicate`.
