# $H_0^G$: is the era drift spread evenly across instruments regardless of referent?

:::{admonition} Definition
:class: note

$H_0^G$ (attribution by referent). Null: the era drift is distributed evenly across instruments
irrespective of their temporal {term}`referent`; each group carries the same per-feature intensity.
Alternative: the drift is differentially concentrated by referent, where concentration in
current-state instruments (RBS-R, CBCL 6-18) is the signature of a change in measurement timing and
concentration in retrospective instruments (the SCQ in its Lifetime form, the developmental
milestones and history) is the signature of a genuine change in the diagnosed population. Estimand:
the drift share by instrument referent, read per class as the current-minus-retrospective contrast
of size-fair displacement intensity. This read is on the {term}`diagnostic era` axis only, and it is
conditional on $H_0^A$ being rejected: it asks where an established drift sits, so it presupposes
that a drift exists.
:::

:::{admonition} Status
:class: tip

Rejected on the era axis. All four classes are retrospective-dominant: the per-class
current-minus-retrospective contrast runs from $-0.05$ to $-0.23$ in separation-standardised units,
every class rejecting at the bootstrap floor ($p = 0.002$). The era profile drift tracks
lifetime-referent report, the signature of a shifting diagnosed population, not measurement timing.
:::

## Method

The starting point is the frozen effect-size trajectory that $H_0^A$ rejects: for each class and each
feature, the separation-standardised displacement $d_k(f)/\sigma$ at the era endpoint, held exactly
as {doc}`the invariance read <../hypotheses/h0a-invariance>` produced it. The referent split is
computed in-stage from that trajectory, so it refits nothing and runs no new model.

Each feature is mapped to its source instrument from the data dictionary, and each instrument to a
pre-registered temporal referent. Two groups result: a current-state group (RBS-R and CBCL 6-18,
rating the child's present behaviour) and a retrospective group (the SCQ Lifetime form, whose items
ask whether a behaviour was ever present, and the developmental milestones and history). Because the
current-state group holds many more features than the retrospective group, a count-fair statistic is
needed. The intensity of a group is the per-feature root mean square of the standardised
displacement over its features, the quadratic mean rather than the raw sum, so a group of more
features does not win on count alone. The test is the signed per-class contrast
$\text{RMS}_k(\text{current}) - \text{RMS}_k(\text{retrospective})$: a positive contrast is
current-dominant (the measurement-timing reading), a negative contrast is retrospective-dominant
(the diagnosed-population reading).

Significance reuses the {term}`family-clustered bootstrap` the effect-size stage already ran. Its
stored per-feature displacement replicates give the contrast a paired bootstrap distribution, both
groups re-read on the same family resample, from which a two-sided add-one $p$-value follows.
Benjamini-Hochberg {term}`false discovery rate` control is applied across the four classes. Alongside
the test, the additive sum-of-squares share of each referent and the per-instrument root mean square
are reported as a descriptive underlay.

This read is folded into the era `analysis invariance-trajectory` run and needs no separate command;
the {doc}`referent-split guide <../guides/splitting-the-drift-by-referent>` sets out the mapping and
the statistic in full.

## Experimental design decisions

- Per-feature root mean square, not the raw sum of squares. On the 238-feature reference set the
  current-state group holds 193 features against 45 retrospective, so a raw sum of squares would
  favour the larger group by count. Dividing by the square root of the feature count reads each group
  at equal per-feature intensity, so under the null the contrast is zero.
- The SPARK SCQ is treated as retrospective. Its items are the Lifetime form, worded "ever ...",
  confirmed against the data dictionary, so it reports developmental history rather than current
  state and joins the retrospective group with the milestones.
- A paired bootstrap on the existing draws. Both groups are re-read on the same family resample, so
  the contrast inherits the family-clustered band the effect-size stage already produced; no new
  bootstrap runs, because the raw draws are read live from the running stage.
- The referent map is pre-registered and hashed into the run. Editing a referent assignment
  invalidates the cache, so the split cannot drift silently against the mapping it was declared with.

## Results

The four classes agree in sign and clear the threshold together. Every per-class contrast is
negative, from $-0.05$ to $-0.23$ in separation-standardised units, and each rejects at the bootstrap
floor ($p = 0.002$). The drift therefore sits in the retrospective instruments across the board, the
signature of a change in the diagnosed population rather than in when the child was rated.

Which retrospective instrument carries the drift splits by class. For Moderate challenges, Broadly
affected, and Social or behavioral the SCQ Lifetime carries it. For Mixed ASD with developmental
delay the developmental milestones and history dominate instead, with a per-instrument root mean
square of $0.58$ and a share of $0.43$, consistent with that class's developmental-history signal.

| Class | Contrast sign | Retrospective carrier |
| --- | --- | --- |
| Moderate challenges | Retrospective-dominant | SCQ Lifetime |
| Broadly affected | Retrospective-dominant | SCQ Lifetime |
| Social or behavioral | Retrospective-dominant | SCQ Lifetime |
| Mixed ASD with developmental delay | Retrospective-dominant | Developmental milestones and history |

The per-class contrasts span $-0.05$ to $-0.23$, all with $p = 0.002$.

## Handling the null

The decision rule is the signed contrast against its paired bootstrap distribution, Benjamini-Hochberg
controlled across the four classes at the {term}`false discovery rate`. A class rejects $H_0^G$ when
its current-minus-retrospective contrast is distinguishable from zero after the correction. All four
contrasts are negative and reject at the bootstrap floor ($p = 0.002$), surviving the false-discovery
step, so $H_0^G$ is rejected for every class and the rejection is in the retrospective direction
throughout.

## Discussion

The era drift in the class profiles is not spread evenly across instruments. It concentrates in the
retrospective group for all four classes, which reads as a change in the developmental histories of
the children who reach a diagnosis rather than a change in how a fixed population is rated as the
survey era moves. The split is limited to the endpoint focal point and to the referent map as
pre-registered; a feature that no instrument carries raises an error rather than being dropped silently, so a
gap in the mapping cannot pass as an empty group, but the reading is only as good as the instrument-level
referent assignments. The age variant is not built, so this is an era-only result.

This concerns the class profiles, the within-class centroid drift, and differs from the prevalence
read, where the era rise of the Social or behavioral class was the timing-driven signal. Here the
same era axis points the other way: the profile drift is retrospective, a diagnosed-population effect,
even though the prevalence signal on this axis was read as timing-driven.

## See also

- {doc}`Are the class profiles invariant, and is any drift small? <../hypotheses/h0a-invariance>`
  ($H_0^A$ and $H_0^D$), which establishes the era drift this read splits.
- {doc}`Splitting the era drift by referent <../guides/splitting-the-drift-by-referent>`, the guide
  to the mapping, the size-fair statistic, and the outputs.
- {doc}`The Python API <../reference>` for the `invariance-trajectory` stage.
