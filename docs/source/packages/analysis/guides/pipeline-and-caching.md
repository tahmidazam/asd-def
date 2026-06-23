# The pipeline and its cache

The analysis runs as a sequence of stages, each a CLI subcommand that reads named inputs
and writes its outputs plus a manifest under a content-addressed `artefacts/` directory.
Re-running a stage with the same inputs is a cache hit and returns at once; changing an
input recomputes only what depends on it. This guide covers how the cache, the manifests,
and the run logs work, and the design choices behind them.

## Stages

Each stage is one `analysis` subcommand. The implemented stages run the reproduction; the
rest are listed under "planned" in `analysis --help` and are added as the work proceeds.

```bash
uv run analysis cohort   # build the proband-by-feature matrix and its typing
uv run analysis fit      # fit the reference mixture model, predict class labels
uv run analysis align    # summarise the classes and name them
```

The stages form a directed acyclic graph: `fit` reads the cohort the `cohort` stage built,
and `align` reads the labels the `fit` stage predicted. Each stage records the hash of the
stage it depends on, so editing the cohort invalidates the fit and the alignment that rest
on it.

## Why a content-addressed cache

Fitting the mixture model is the expensive step: the reference fit runs 200 random restarts
and takes around ten minutes, and the planned stability and stratified work multiply that
across seeds and strata. Caching is what makes the work practical across sessions, and the
result has to be trustworthy, so a cached output must correspond exactly to the inputs that
produced it.

A run is therefore identified by a hash over everything that determines its output:

- the dataset and version (for example `spark` / `2026-03-23`),
- a digest of the authors' feature list and of the released typing files,
- the model hyperparameters, the covariate set, and the random seed,
- the stage it depends on (its hash), and
- the package's git commit.

The hash names the run directory, so the same inputs always resolve to the same place. A
stage is a cache hit when a manifest with that hash already exists and finished cleanly;
otherwise it recomputes. Pass `--force` to recompute regardless.

## Layout

Every run writes to `artefacts/<stage>/<hash>/`:

```
artefacts/fit/3156a623d289b89a/
  manifest.json    inputs, status, timing, environment, git commit, metrics
  model.joblib     the fitted estimator
  labels.parquet   the predicted class label per proband
  centroids.parquet
  run.log          the run's captured output
```

Data frames are written as Parquet, models with joblib, and manifests and scalar metrics
as JSON. `artefacts/` is gitignored: it holds participant-derived intermediates, which the
SFARI consent does not allow into version control, so only non-disclosive aggregate outputs
are ever promoted into the manuscript.

## Manifests

The manifest records what produced the run and what it found:

- the input parameters and their hash,
- the status (`running`, `ok`, or `failed`), start and finish times, and wall-clock duration,
- the git commit and the resolved versions of the modelling libraries (StepMix, numpy,
  pandas, scikit-learn, scipy, statsmodels, pyarrow),
- the stage's key metrics, for example the fit's class proportions and information criteria.

Recording the resolved versions matters because mixture fits depend on them. A library
upgrade does not silently invalidate a cache, it changes the recorded versions, and
re-running the reproduction benchmark catches any movement in the result.

## Run logs and observability

Each run captures its standard output and its log messages into `run.log` while still
showing them at the terminal. The reference fit shows StepMix's restart progress bar with
the best log-likelihood so far, and that progress is written to the log, so a long run
started in one session can be inspected later.

## Resuming an interrupted run

The cache works at the granularity of a whole stage: a run is reused only once it has
finished cleanly, so a stage interrupted partway leaves no reusable output and re-running it
starts over. That is fine for the short stages, but the multi-seed loops can run for tens of
minutes to hours, and losing all of it to one interrupt is wasteful. Four stages therefore
checkpoint their progress: `select`, both modes of `stability`, and `nmin`.

Each records one unit of work as it completes, appended to a `*.checkpoint.jsonl` file in the
run directory: a seeded iteration of the selection grid, one single-initialisation fit, one
subsample replicate, or one comparison to the reference. Re-running the same command (the
same parameters, hence the same run directory) reads the completed units back and continues
from the first one that is missing. Because each unit's seed is derived from its index, a
resumed run reproduces exactly what an uninterrupted run would have computed; the checkpoint
changes only how much is recomputed, never the result.

The unit of resumption is one line. A process killed mid-write can leave a torn final line,
which is dropped and that last unit recomputed, so a half-written record is never read back.
The multi-initialisation run keeps two checkpoints, one for the fits and one for the
comparisons; it does not store each fit's labels, so a comparison whose fit was restored from
a checkpoint refits that seed on demand (cheap, since only the best fits are compared). On
clean completion the checkpoints are removed, the Parquet outputs having superseded them, and
`--force` clears them before recomputing, so a forced run always starts fresh.

## Determinism and dependencies

Every fit takes an explicit seed, recorded in the manifest, so a run is reproducible given
the same environment. The released stability, subsampling, and selection fits are not seeded,
so they are not reproducible run to run; seeding every fit here, and deriving each unit's seed
from its index, is the one deliberate divergence from the released procedure for the multi-seed
stages, and it is what lets them resume (above). The modelling stack is kept current and shared across the monorepo in
one lockfile rather than pinned or isolated: a dependency such as pandas or scikit-learn is
a single consistent version across packages. The authors fit with an older StepMix (1.2.5);
this work uses the current major version (3.x), and the reproduction benchmark, not a
version pin, is what guards the result.

## Reproducing a run from scratch

With the SPARK data and the authors' feature list in place, the three implemented stages
rebuild the reference solution end to end:

```bash
uv run analysis cohort
uv run analysis fit
uv run analysis align
```

The first invocation computes; a second is a cache hit and returns immediately. The run
hash printed by each stage ties every output back to the exact inputs that produced it.
