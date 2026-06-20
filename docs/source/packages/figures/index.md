# figures

The figures package renders the analysis artefacts as Matplotlib figures. Each figure is
built by a function that takes a dataframe and returns a figure, and a command-line
subcommand resolves a cached run, builds the figure, and writes it under `artefacts/figures/`
with a JSON provenance sidecar.

The package covers the reproduction result and the phase-2 results, one figure each.

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

Each subcommand takes an optional `--run` (defaulting to the latest completed run of that
stage) and writes a PDF and a PNG under `artefacts/figures/<stage>/<run-hash>/` beside a JSON
sidecar that records the source run and the package version.

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
