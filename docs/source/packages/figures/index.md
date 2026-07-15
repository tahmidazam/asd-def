# figures

The figures package renders the analysis artefacts as Matplotlib figures. Each figure is built by a
function that takes a dataframe and returns a figure; a CLI subcommand resolves a cached `analysis`
run (the latest of its stage unless `--run` names one), builds the figure, and writes a PDF and a
PNG under `artefacts/figures/<stage>/<run-hash>/` beside a JSON sidecar that records the source run
and the package version.

## Figures by page

Each figure belongs to a hypothesis article or a foundational page and is drawn by one builder
module. The table maps a documentation page to its figures, the module that builds them, and the
command that renders them.

| Page | Figures | Module | Command |
| --- | --- | --- | --- |
| [$H_0^A$, $H_0^D$: invariance](../analysis/hypotheses/h0a-invariance.md) | `local_plane_*`, `local_panels_*`, `local_specificity`, `invariance_process_*` | {py:mod}`figures.trajectory_local`, {py:mod}`figures.invariance` | `local-trajectory`, `local-panels`, `local-specificity`, `invariance` |
| [$H_0^B$: prevalence](../analysis/hypotheses/h0b-prevalence.md) | `prevalence_*`, `prevalence_stacked_*` | {py:mod}`figures.prevalence` | `prevalence` |
| [$H_0^E$: direction](../analysis/hypotheses/h0e-direction.md) | `local_directional_*` | {py:mod}`figures.trajectory_local` | `local-directional` |
| [$H_0^F$: attribution by category](../analysis/hypotheses/h0f-attribution-categories.md) | `category_decomposition`, `dense_features` | {py:mod}`figures.category_decomposition`, {py:mod}`figures.dense_features` | `category-decomposition`, `dense-features` |
| [$H_0^G$: attribution by referent](../analysis/hypotheses/h0g-attribution-referent.md) | `referent_decomposition` | {py:mod}`figures.referent_decomposition` | `referent-decomposition` |
| [Screening orderings with the atlas](../analysis/guides/screening-orderings-with-the-atlas.md) | `displacement_atlas` | {py:mod}`figures.atlas` | `atlas` |
| [Reproducing the reference classes](../analysis/appendix/reproducing-the-reference-classes.md) | `reproduction` | {py:mod}`figures.reproduction` | `reproduce` |
| [Selecting the number of classes](../analysis/appendix/selecting-the-number-of-classes.md) | `selection_criteria` | {py:mod}`figures.selection` | `select` |
| [Replicating in the SSC](../analysis/appendix/replicating-in-the-ssc.md) | `replication` | {py:mod}`figures.replication` | `replicate` |
| [Stability under refitting](../analysis/appendix/stability-under-refitting.md) | `stability` | {py:mod}`figures.stability` | `stability` |
| [The minimum stratum size](../analysis/appendix/the-minimum-stratum-size.md) | `stratum_size` | {py:mod}`figures.nmin` | `nmin` |
| [The refit pilot](../analysis/archive/tracking-the-classes-across-strata.md) | `trajectory_*`, `roughness` | {py:mod}`figures.trajectory` | `trajectory` |

## Publishing

`artefacts/` is gitignored, so the figures do not reach the published site on their own. `figures
publish` copies the rendered PNGs into `docs/source/_figures/`, the committed directory the pages
embed, and writes a provenance sidecar beside each recording the source stage, run hash, and commit;
a figure not yet rendered is skipped. Only the PNGs cross into the committed tree: they are
aggregate, non-disclosive summaries, so committing them stays within the data governance that keeps
the rest of `artefacts/` out of the history.

:::{toctree}
:maxdepth: 1
:caption: Reference

reference
:::
