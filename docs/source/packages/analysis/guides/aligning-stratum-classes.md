# Aligning stratum classes to the reference

Machinery for the stratified refit drift read. That read re-estimates the four-class
{term}`general finite mixture model` within each stratum, so it is the refit path the archived
{doc}`refit pilot <../archive/tracking-the-classes-across-strata>` used; the primary invariance read
is now the kernel effect size ({doc}`computing the invariance effect size
<computing-the-invariance-effect-size>`), which freezes the reference and needs no alignment.

A stratum fit numbers its classes in whatever order it converges on, so before any movement can be
measured the stratum's classes have to be matched to the reference's. Every proband in a stratum
carries two labels, the class its stratum fit gave it and the class the pooled reference gave it, and
the alignment compares those two labellings directly.

The alignment is a swappable method, chosen at the command line and recorded in the run hash. Two are
defined:

- Membership alignment (the default) pairs classes by who is in them, maximising the total Jaccard
  overlap across the four classes by Hungarian assignment. It reports each pair's Jaccard as a
  confidence and the adjusted Rand index for the partition as a whole.
- Centroid alignment pairs classes by profile, matching on the standardised distance between
  centroids. It is a cross-check, not the default, because it cannot tell a class that moved from one
  that reorganised its members.

The gap between the two is the point: a class has genuinely drifted only when its members are largely
unchanged and its profile has moved. A class whose matched Jaccard falls below one half is flagged as
reorganised and set aside from the drift count, which is also why a distance on the order of the whole
{term}`between-class separation` is read as a warning rather than a finding.

```
uv run analysis drift --axis age_at_diagnosis --alignment membership
```

`--alignment` takes `membership` or `centroid`. Because the alignment reads the stored fit summaries,
switching methods re-measures without re-fitting. Measuring how far an aligned class then moved is the
{doc}`sibling step <measuring-class-drift>`.
