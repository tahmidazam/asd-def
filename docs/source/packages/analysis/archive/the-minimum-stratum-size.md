# The minimum viable stratum size

:::{admonition} The question
:class: note

How small can a stratum be before four-class recovery breaks down? The stratified analysis splits
the cohort by age at diagnosis and by diagnostic era, and a four-class model over more than two
hundred mixed features needs enough probands per class to estimate its densities. This
investigation fixes the floor empirically, which bounds how finely the cohort can be split.
:::

:::{admonition} The result
:class: tip

Recovery is reliable from about 1,000 probands. Refitting at descending sizes, the profile
correlation to the full-sample reference is poor below about 200 probands, marginal from there to
about 700, and first clears the $r \ge 0.90$ benchmark on its own at 1,000. A monotone (isotonic)
fit puts the floor at about 840 probands, with a 90 per cent bootstrap interval of 300 to 960. No
class collapses at any size, so the floor turns on profile fidelity rather than on a class
vanishing. The stratification bins are best kept comfortably above the floor, nearer 2,000, where
the smallest class has room.
:::

:::{figure} /_figures/stratum_size.png
:alt: Recovery against subsample size and the minimum viable stratum size
:width: 100%
:align: center

Recovery against subsample size, from an `analysis nmin` run (twelve sizes from 100 to 5,000,
fifteen refits each). (A) Each refit's profile correlation to the full-sample reference, with the
per-size mean, the $r \ge 0.90$ benchmark, and the isotonic floor at about 840 with its 90 per
cent bootstrap interval (300 to 960). Recovery is poor below about 200 probands, climbs through
the benchmark, and plateaus by 1,000. (B) The smallest class proportion stays near 0.14 at every
size, so no class collapses.
:::

## Reading the result

The sweep takes twelve sizes from 100 to 5,000 probands, fifteen refits each, against the
$r \ge 0.90$ benchmark the reproduction meets. Going below 500 finds where recovery breaks down:
on SPARK 2026-03-23 the per-size mean is about 0.72 at 100 and 150 probands, climbs through 0.84
to 0.89 between 200 and 700, first clears the benchmark at 1,000 (0.93), and stays near 0.91 to
0.93 from there to 5,000. The isotonic fit puts the floor, where recovery reaches the benchmark,
at about 840 probands, with a 90 per cent bootstrap interval of 300 to 960; the smallest size to
clear on its own is 1,000.

The measure is noisy at the small sizes, where a fit on a few hundred probands over more than two
hundred features is poorly determined and the per-fit scatter is wide below about 700. The robust
reading is that recovery is poor below about 200 probands, marginal from there to about 700, and
reliable from about 1,000. No fit collapsed a class at any size; the smallest class stays at about
0.13 to 0.15 of each subsample throughout, so the floor turns on profile fidelity rather than on a
class vanishing.

For the stratification bins this puts the floor near 1,000 probands, where recovery clears
reliably. There the smallest class holds only about 150 probands for a profile with several
hundred free parameters, which is thin, so the bins are best kept comfortably above that, nearer
2,000 where the smallest class has room.

## How the floor is estimated

:::{dropdown} The estimation procedure

The `nmin` stage fixes the floor empirically: it refits at descending sample sizes and records, at
each size, the profile correlation to the full-sample reference, with the smallest class
proportion and the average posterior certainty as degradation checks. The floor is the smallest
size at which a monotone (isotonic) fit of correlation against log-size reaches the reproduction
benchmark, with a bootstrap interval; pooling every fit makes it robust to the scatter the
recovery measure carries. That floor is the lower bound on the stratification bins.
:::

The sweep is seeded and resumable, like the other multi-seed stages; see
[the pipeline and its cache](../guides/pipeline-and-caching) for the seeding and checkpointing.
