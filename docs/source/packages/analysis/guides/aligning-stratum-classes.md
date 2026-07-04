# Aligning stratum classes to the reference

The stratified analysis re-estimates the four-class mixture model within each stratum and asks
how far each class has moved from the pooled reference. Before any movement can be measured, the
stratum's classes have to be matched to the reference's. The mixture model numbers its classes in
whatever order a fit converges on, so the stratum's class 0 is not the reference's class 0 and the
labels carry no meaning across fits. Alignment recovers the correspondence.

A stratum is a subset of the pooled cohort, not a separate sample, so every proband in it carries
two labels: the class the stratum fit gave it, and the class the pooled reference gave the same
proband. Both labellings sit on the same probands, which is what lets the alignment compare them
directly rather than infer the match from the class profiles alone.

The package makes the alignment a swappable method. Two are defined: membership alignment (the
default), which matches on who is in each class, and centroid alignment, which matches on the
class profiles. This guide describes the interface, the two methods, and the distinction they
support between a class that moved and one that reorganised. Measuring how far an aligned class
then moved is the {doc}`sibling step <measuring-class-drift>`.

## The alignment interface

An alignment method takes a stratum summary and the reference model and returns a class
alignment: a mapping from each fit class to a reference class, a per-class quality (the
confidence of each matched pair), and one overall number for the partition. The drift stage is
typed against the interface, not a concrete method, so the method is chosen at the command line
and the measurement code does not change.

The stored input is small by design. A fit is summarised once into its per-class feature means,
its per-class dispersions, and a contingency table of its labels against the reference labels;
these are the sufficient statistics any alignment or distance needs. Membership alignment reads
only the contingency table, centroid alignment reads only the centroids. Because the summaries
are stored, re-running with a different alignment re-measures in seconds rather than re-fitting,
the same decoupling the {doc}`distance methods <measuring-class-drift>` rely on.

## Membership alignment (the default)

Membership alignment pairs the classes by who is in them. For a candidate pairing of a fit class
$A$ with a reference class $B$, the overlap is the Jaccard index, the number of probands in both
divided by the number in either, $J = |A \cap B| / |A \cup B|$. The pairing that maximises the
total Jaccard across the four classes is found by Hungarian assignment (linear sum assignment on
$1 - J$), so each fit class is matched to exactly one reference class.

The Jaccard, rather than the raw count of shared probands, is what stops the unequal class sizes
distorting the match. The named classes run from about a seventh to two fifths of the cohort, so
a raw-overlap rule would pull every small class towards the largest one; dividing by the union
removes the size advantage.

Two numbers travel with the mapping. Each matched pair reports its Jaccard as the confidence of
that one correspondence. The partition as a whole reports the adjusted Rand index between the two
labellings, a chance-corrected agreement that is 0 for an arbitrary relabelling and 1 for
identical partitions. Both come from the stored contingency counts alone, so the per-proband
labels need not be kept.

## Centroid alignment (the cross-check)

Centroid alignment ignores membership and matches on the class profiles. Each feature is put on a
common scale by dividing by its pooled standard deviation, the standardised distance between every
fit centroid and every reference centroid is computed, and Hungarian assignment takes the pairing
of smallest total distance. The per-class quality is a closeness in $[0, 1]$, $1 / (1 + d)$ for a
matched distance $d$, and the overall number is the mean of those qualities.

This method sees only the centroids, so it cannot tell a class that moved from one that
reorganised: a stratum class that kept the reference centroid but drew an entirely different set
of probands looks like a clean match. That blind spot is why it is not the default. It is kept as
a cross-check, since a disagreement between the membership and centroid mappings marks a
correspondence that is not safe to measure drift on.

## Moved or reorganised

The two alignments answer different questions, and the gap between them is the point. Membership
asks whether the same probands are in the class; centroids ask whether the class profile is in
the same place. A class has genuinely drifted only when its members are largely unchanged and its
profile has moved. If the membership overlap is low, a large profile distance is not drift: the
stratum has divided the probands differently, and the class being compared is a different group
that happens to have a similar centre.

The drift read uses this directly. A class whose matched Jaccard falls below one half is flagged
as reorganised and set aside from the drift count, even when its measured distance is large and
clears the permutation null. The headline count is therefore the classes that both exceed the
null and keep their membership. The same logic explains why a measured distance on the order of
the full between-class separation is treated as a warning rather than a finding: a class rarely
moves as far as the gap between two distinct classes, so such a value usually marks a reorganised
pairing the membership alignment has already caught (see the separation baseline in
{doc}`measuring how far a class drifts <measuring-class-drift>`).

## Choosing the method

```
uv run analysis drift --axis age_at_diagnosis --alignment membership
```

`--alignment` takes `membership` (the default) or `centroid`. The choice is recorded in the run
manifest and enters the run hash, so a membership run and a centroid run cache separately and can
be compared. Because the alignment reads the stored fit summaries, switching methods re-measures
without re-fitting: the stratum fits and the permutation null that calibrates the drift are
computed once and reused.
