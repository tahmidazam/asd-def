# Splitting the era drift by referent

The {doc}`effect-size recast <../investigations/invariance-as-an-effect-size>` reports how far
each class profile drifts along diagnostic era, and the {doc}`attribution stage
<attributing-class-movement>` splits that drift across the seven author categories. This read asks
a different question of the same era drift: does it sit in the instruments that rate the child's
present state, or in the instruments that ask about the developmental history? The two answers
point at different mechanisms, so the split discriminates between them (the ATTR-REF hypothesis).

Each instrument carries a temporal referent. The RBS-R and the CBCL 6-18 rate current behaviour.
The SCQ is administered in its Lifetime form (its items ask whether a behaviour was ever present,
confirmed against the data dictionary), and the background-history items record developmental
milestones and childhood history, so both are retrospective. Drift concentrated in the
current-state instruments is the signature of a change in measurement timing: the same children
rated differently as the survey era moves. Drift concentrated in the retrospective instruments is
the signature of a change in the diagnosed population: the developmental histories of the children
who reach a diagnosis shift across era. The read is era only; the age variant is not built.

## Deriving the map

Each feature is assigned to its source instrument from the data dictionary, the same describe pass
the {doc}`feature typing <../reference>` uses, and each instrument carries a pre-registered
referent held as a constant in the code. Resolution fails loudly: a feature that no instrument
carries raises rather than being dropped, so a gap in the mapping cannot pass silently as an empty
grain. The developmental-milestone columns appear in both the child and the sibling
background-history tables under the same names; the proband's own child history is listed first
and wins, and the sibling table contributes nothing to the reference feature set.

On the 238-feature reference set the split is 193 current-state features (49 RBS-R, 144 CBCL 6-18)
against 45 retrospective features (34 SCQ Lifetime, 11 developmental milestones and history). The
current-state grain holds four times as many features, so a size-fair statistic is needed.

## The size-fair statistic

Writing $d_k(f)/\sigma$ for the separation-standardised per-feature displacement of class $k$ at
the endpoint focal point, the intensity of a referent grain is the per-feature root mean square

$$
\text{RMS}_k(\text{grain}) = \frac{\lVert d_k/\sigma \rVert_\text{grain}}{\sqrt{n_\text{grain}}},
$$

the quadratic mean over the grain's features. Dividing by the square root of the feature count
makes grains of different size comparable: under the null the drift is spread at equal per-feature
intensity across referents, so the two grains carry the same root mean square and the current
minus retrospective contrast is zero. A raw sum of squares would favour the larger grain by feature
count alone.

The test is the per-class contrast $\text{RMS}_k(\text{current}) - \text{RMS}_k(\text{retrospective})$.
It is signed: a positive contrast is current-dominant (the measurement-timing reading), a negative
contrast is retrospective-dominant (the diagnosed-population reading). Significance comes from the
family-clustered bootstrap the effect-size stage already ran: its stored per-feature displacement
replicates give the contrast a paired bootstrap distribution (both grains are re-read on the same
family resample), from which a two-sided add-one $p$-value follows, the same construction the
per-feature and directional tests use. Benjamini-Hochberg control is applied across the four
classes. No new bootstrap runs, because the raw draws are not persisted; the contrast is computed
in-stage from the live tube.

Alongside the test, the additive sum-of-squares share of each referent is reported as a descriptive
decomposition. The shares sum to one over the two disjoint referents, so they read as the fraction
of a class's squared drift that each referent accounts for. The per-instrument root mean square and
the per-referent count of features surviving the per-feature FDR are carried too, so the two-way
headline sits over a transparent per-instrument underlay.

## The outputs

The read is folded into the era `invariance-trajectory` run, so it needs no separate command. The
pre-registered instrument-to-referent map is digested into the run hash, so editing a referent
assignment invalidates the cache. Two tables are written. `referent_era.parquet` is the per-class
by-grain decomposition: for each referent and each instrument, the size-fair root mean square, the
additive share, and the FDR-surviving feature count. `referent_contrast_era.parquet` is the
per-class contrast test: the current minus retrospective contrast with its bootstrap interval, the
two-sided $p$-value, the Benjamini-Hochberg decision, and the mechanism reading. The run manifest
carries a `referent_contrast` block with the same headline.

The `figures` package draws the decomposition with one panel per class:

```
uv run analysis invariance-trajectory --axis era
uv run figures local-referent --axis era
```

Each panel sets the current-state root mean square beside the retrospective root mean square, with
the per-instrument values as points, and reads the contrast and its mechanism in the title. The
figure is a build-command render for now and is not published into the documentation.
