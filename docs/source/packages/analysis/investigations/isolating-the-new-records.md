# Isolating the records added since V9

:::{admonition} The question
:class: note

The reference fit on the `2026-03-23` release recovers the four classes but diverges from the
paper in two ways: the smallest class is about 15 per cent of the cohort rather than the
published 10, and the model-selection criteria over-extract further. That release is more than
twice the size of Litman's V9 cohort, so a natural question is whether the records added since V9
are the cause. This investigation cuts the cohort back to the probands present at the V9 freeze
and refits, with a size-matched random control to separate "fewer records" from "different
records".
:::

:::{admonition} The result
:class: tip

The new records are part of the story, not all of it. Cutting back to the V9 freeze leaves 5,324
probands, against Litman's 5,392, and shifts the class proportions: Moderate challenges rises from
29 to 34 per cent, onto the published value, while Social or behavioural falls from 39 to 31. The
size-matched control reproduces neither shift, so the movement tracks the specific V9 records, not
the smaller sample. But the cut does not reconcile the fit with the paper. The overall profile
correlation is unchanged at $r = 0.90$, the smallest class stays near 15 per cent rather than 10,
and the criteria still over-extract (their minimum falls from nine classes to eight, nowhere near
four). The records added since V9 move the proportions without landing them on the published
solution.
:::

:::{figure} /_figures/subset_comparison.png
:alt: Class proportions and model selection across the full release, the V9 subset, and a control
:width: 100%
:align: center

The cohort cuts compared, from `analysis` runs on the full `2026-03-23` release, the `--as-of`
V9 subset, and a size-matched random control. (A) The four named-class proportions by cut, against
the values Litman et al. (2025) report: the V9 subset (green) lifts Moderate challenges onto the
published value while the control (orange) stays near the full release, and the smallest class,
Broadly affected, sits above the published 10 per cent in every cut. (B) Each cut's Bayesian
information criterion across the number of classes, normalised within the cut; all three minimise
far to the right of the four classes retained, the subset at eight and the others at nine.
:::

## Reading the result

The V9 subset holds 5,324 probands, within 1.3 per cent of Litman's 5,392, so the reconstruction
recovers a cohort of the right size. The four classes reproduce on it with every anchor holding,
and the overall seven-category profile correlation is $r = 0.902$ (95 per cent bootstrap interval
$[0.887, 0.922]$), the same as on the full release. The cut does not change how well the profile
reproduces; it changes the class proportions.

| Class | V9 subset | Size-matched control | Full release | Litman 2025 |
| --- | --- | --- | --- | --- |
| Social or behavioural | 31 | 37 | 39 | 37 |
| Moderate challenges | 34 | 27 | 29 | 34 |
| Mixed ASD with developmental delay | 20 | 19 | 18 | 19 |
| Broadly affected | 15 | 17 | 15 | 10 |

Moderate challenges is the clearest movement: 29 per cent on the full release, 34 on the V9
subset, matching the published 34. The size-matched control, a random draw of the full release to
the same 5,324 probands, holds it at 27 per cent, near the full release, so the rise is not a
sample-size effect; it is the V9 records that carry it. Social or behavioural moves the other way,
from 39 down to 31, past the published 37 rather than onto it, and again the control does not
follow. So the records added since V9 do shift the proportions, and the shift is compositional,
but it does not simply pull them toward the paper.

The smallest class is the exception. Broadly affected sits near 15 per cent on the full release,
the subset, and the control alike, and reaches the published 10 in none of them. This divergence,
the larger of the two the reference fit shows, is not explained by the records added since V9: it
persists in the V9-era roster.

Model selection tells the same partial story. On the full release the information criteria
minimise at nine classes; on the V9 subset they minimise at eight, and the size-matched control
stays at nine, so the V9 records account for the one-class reduction. But eight is still far from
four, so the criteria over-extract on the V9-era cohort much as on the full one. The four-class
choice rests on reading the criteria rather than on their minimum here, as it did for the authors.

The two secondary checks add nothing that reopens the picture. Refitting the V9 subset many times
reproduces the profile at 0.89 to 0.90, slightly below the full release's 0.91 to 0.92 and with no
class ever collapsing, the small drop consistent with the smaller sample. Training the SSC
replication on the V9-era SPARK rather than the full release gives an overall $r = 0.875$, a shade
below the full-release 0.887, so the cross-cohort gap (carried by the developmental category, see
[replicating in the SSC](replicating-in-the-ssc)) does not close either; it sits on the SSC side,
not the SPARK sample.

Taken together, the records added since V9 are a contributor to the proportion differences and to
one class of the over-extraction, confirmed compositional by the control, but they do not account
for the divergences that matter: the inflated smallest class, the over-extraction past four
classes, and the SSC developmental gap. The residual points elsewhere, to the revised values the
retained probands carry (the cut keeps the V9 roster but with `2026-03-23` values), to the current
modelling stack, and to properties of the recovered classes that hold across the cohort cuts.

## How the cut is built

:::{dropdown} The records cutoff and the control

The cut keeps a proband only if it was present at the V9 freeze of 2022-12-12: registered by then
and with each cohort instrument completed by then. Three instruments carry a completion year and
are gated on it directly; the CBCL 6-18 has none, so its year is reconstructed from the
registration anchor. The size-matched control is a random draw of the full release to the subset's
size, so the two fits differ only in which probands they hold. The method and its limits, in
particular that the cut recovers the V9 roster carried with current values rather than the V9 data
itself, are in [subsetting the cohort to the V9 freeze](../guides/subsetting-to-the-v9-freeze).
:::

## Caveats

The control is a single random draw. Its proportions stay near the full release, which is what
makes the subset's shifts read as compositional, but a firmer claim would average several draws.
The cut also recovers the V9 roster with current values, not the V9 data, so a proband kept here
may carry a phenotype value revised since 2022; the reconstruction isolates which probands were
present, not which values were recorded then. Both limits point the same way: this investigation
attributes the proportion shifts to the V9 records as a population, and leaves the contribution of
the revised values, which would need the V9 file to measure, open.
