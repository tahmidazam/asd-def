# $H_0^H$: is the era drift an artefact of measurement timing?

:::{admonition} Definition
:class: note

Era axis only, and conditional on rejecting $H_0^A$. Null: the era drift is an artefact of
measurement timing; the class-parameter-by-year association is nullified after adjustment for the
{term}`measurement lag` (the gap between diagnosis and parent report) and age at evaluation, and does
not hold in a small-lag subsample. Alternative: the era drift persists net of the lag and age at
evaluation, and within the small-lag subsample, implicating a change in the diagnosed population
rather than in the timing of measurement. Estimand: the partial association of the class parameters
with year of diagnosis, conditional on the lag and age at evaluation.
:::

:::{admonition} Status
:class: tip

Partially written. The test is specified but not yet run as a standalone stage. One preview exists
from the neighbouring prevalence read ($H_0^B$): the era rise of the Social or behavioral class is
nullified once sex, the diagnosis-to-measurement lag, and age at evaluation are netted out, while the
era decline of the developmental class and the rise of Broadly affected survive that adjustment. So
the prevalence evidence points toward a mixed picture, with part of the era signal explained by
measurement timing and part not. The profile-level test below is still to come.
:::

## Method

The planned test reads the direct effect of year of diagnosis on each class parameter, holding the
{term}`measurement lag` and age at evaluation fixed. In measurement terms this is a
{term}`differential item functioning` model: does a feature shift with year of diagnosis once the
person's class and evaluation age are held constant? A parameter whose year association survives the
adjustment is drift in the diagnosed population; one that vanishes is a timing artefact. A small-lag
subsample re-test then checks that the surviving association is not a residual of the adjustment
model, by restricting to probands measured close to their diagnosis, where the lag cannot act.

This article will be completed once the direct-effect stage is implemented and run. Until then, the
prevalence adjustment in {doc}`../hypotheses/h0a-invariance` and the prevalence read are the closest
evidence on the record.

## See also

- {doc}`Are the class profiles invariant, and is any drift small? <../hypotheses/h0a-invariance>`
  ($H_0^A$), the drift this attribution conditions on.
- {doc}`Splitting the era drift by referent <../guides/splitting-the-drift-by-referent>`, the
  companion mechanism read ($H_0^G$), which separates a current-state from a lifetime-referent
  signature.
