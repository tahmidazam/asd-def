# figures

The figures package renders the analysis artefacts as Matplotlib figures. Each figure is
built by a function that takes a dataframe and returns a figure, and a command-line
subcommand resolves a cached run, builds the figure, and writes it under `artefacts/figures/`
with a JSON provenance sidecar.

The package covers the reproduction result, the phase-2 results, and the phase-4 class
trajectories across the strata.

## Implemented figures

- `figures reproduce` plots the reproduction of the named classes from an `analysis align`
  run: each recovered class signature against the value read from figure 1b of Litman et al.,
  one panel per named class, with the class proportions and the per-class profile correlation.
- `figures select` plots the model-selection criteria across the number of latent classes
  from an `analysis select` run: the information criteria, the cross-validated log-likelihood,
  and the smallest-class proportion together with the relative entropy, each with a reference
  line at the four classes chosen by Litman et al.
- `figures replicate` plots the cross-cohort replication from an `analysis replicate` run: the
  SPARK and SSC class signatures against the line of equality, and the per-category
  correlation, where the developmental category is the clear outlier.
- `figures stability` plots the stability of the reference fit from an `analysis stability`
  run: the profile-correlation and adjusted-Rand-index distributions, the per-category
  correlation, and the mean class-overlap matrix.
- `figures nmin` plots recovery against subsample size from an `analysis nmin` run: each
  refit's profile correlation, the per-size mean, the recovery benchmark, and the isotonic
  floor with its bootstrap interval, beside the smallest class proportion.
- `figures trajectory --axis <age_at_diagnosis|era>` plots each class's path through the strata
  of an `analysis trajectory` run, one panel per class, in the pooled four-class discriminant
  space: the pooled reference class as a ring, the stratum centroids coloured from the first
  stratum to the last, and an arrow for the net displacement, with a red ring where membership
  reorganised.
- `figures roughness` plots the trajectory roughness across both axes from the two
  `analysis trajectory` runs: the mean step between adjacent strata against the sampling-noise
  expectation, and each class's net young-to-old displacement against the ordering-shuffle null.

Each subcommand takes an optional `--run` (defaulting to the latest completed run of that
stage) and writes a PDF and a PNG under `artefacts/figures/<stage>/<run-hash>/` beside a JSON
sidecar that records the source run and the package version.

## Class trajectories across the strata

Each class's centroid, projected into the pooled four-class discriminant space, and how it
moves across the age-at-diagnosis and diagnostic-era strata. The projection is linear, so
positions and distances are honest, but it is an illustration: the movement claim rests on the
full-dimensional roughness and directional statistics below, not on the picture. The null here
is the pilot ordering-shuffle on the observed centroids; the confirmatory test is the
continuous-trend regression against the refit permutation null.

```{image} ../../_figures/trajectory_age_at_diagnosis.png
:alt: Each class's trajectory through the age-at-diagnosis strata
:width: 100%
```

```{image} ../../_figures/trajectory_era.png
:alt: Each class's trajectory through the diagnostic-era strata
:width: 100%
```

```{image} ../../_figures/roughness.png
:alt: Trajectory roughness and directional movement across both axes
:width: 100%
```

## Publishing to the documentation

The figures are written under `artefacts/`, which is gitignored, so they do not reach the
published documentation on their own. `figures publish` copies the rendered PNGs into
`docs/source/_figures/`, a committed directory the documentation pages embed, and writes a
provenance sidecar beside each recording the source stage, the source run hash, and the commit
it was built from. Run with no argument it publishes the whole set, taking each figure from the
latest completed run of its source stage; a figure that has not been rendered yet is skipped
with a note. Only the rendered PNGs cross into the committed tree: they are aggregate,
non-disclosive summaries, so committing them stays within the data governance that keeps the
rest of `artefacts/` out of the history.

## Reference

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Python API
:link: reference
:link-type: doc

The command-line interface, the house style and save helper, the run-resolution and loading
helpers, and the figure builders.
:::

::::

:::{toctree}
:hidden:
:caption: Reference

reference
:::
