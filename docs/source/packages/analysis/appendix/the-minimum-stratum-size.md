# The minimum viable stratum size

:::{admonition} The question
:class: note

How small can a stratum be before four-class recovery breaks down? Any stage that re-estimates the
four-class model within a bin of age at diagnosis or diagnostic era needs enough probands per class
to estimate its densities over more than two hundred mixed features. This investigation fixes that
floor empirically, which bounds how finely the cohort can be split.
:::

:::{admonition} The result
:class: tip

Recovery is reliable from about $1{,}000$ probands. Refitting at descending sizes, the profile
correlation to the full-sample reference first clears the $r \ge 0.90$ benchmark on its own at
$1{,}000$; an isotonic fit puts the floor at about $840$, with a $90\%$ bootstrap interval of $300$
to $960$. No class collapses at any size, so the floor turns on profile fidelity, not on a class
vanishing. The bins are best kept well above it, nearer $2{,}000$, where the smallest class has room.
:::

:::{figure} /_figures/stratum_size.png
:alt: Recovery against subsample size and the minimum viable stratum size
:width: 100%
:align: center

Recovery against subsample size, from an `analysis nmin` run ($12$ sizes from $100$ to $5{,}000$,
$15$ refits each). (A) Each refit's profile correlation to the full-sample reference, with the
per-size mean, the $r \ge 0.90$ benchmark, and the isotonic floor at about $840$ with its $90\%$
bootstrap interval ($300$ to $960$). Recovery is poor below about $200$ probands, climbs through the
benchmark, and plateaus by $1{,}000$. (B) The smallest class proportion stays near $0.14$ at every
size, so no class collapses.
:::

## Three regimes of recovery

The sweep refits at $12$ sizes from $100$ to $5{,}000$ probands, $15$ times each, and scores every
refit by its profile correlation to the full-sample reference. On SPARK 2026-03-23 recovery falls
into three regimes:

- **Poor below about $200$ probands.** The per-size mean is about $0.72$: a fit on a few hundred
  probands over more than two hundred features is too poorly determined to recover the profile, and
  the per-fit scatter is wide.
- **Marginal from $200$ to about $700$.** The mean climbs from $0.84$ to $0.89$, still short of the
  benchmark and still scattered.
- **Reliable from $1{,}000$.** Recovery clears the benchmark at $0.93$ and holds between $0.91$ and
  $0.93$ out to $5{,}000$.

The isotonic fit places the floor, where recovery reaches the benchmark, at about $840$ probands
($90\%$ bootstrap interval $300$ to $960$); the smallest size to clear on its own is $1{,}000$.

The floor turns on profile fidelity, not on a vanishing class. No fit collapsed a class at any size:
the smallest class holds $0.13$ to $0.15$ of each subsample throughout. Recovery degrades because the
densities are hard to estimate at small $n$, not because a class disappears.

## What it means for the bins

This floor gates the checks that re-estimate the model per stratum: the order test
[$H_0^C$](../hypotheses/h0c-order.md) and the archived
[refit pilot](../archive/tracking-the-classes-across-strata). The frozen-reference reads run a
continuous focal grid along the axes and are not bound by it.

For those refit checks the floor sits near $1{,}000$ probands, where recovery clears reliably. Even
there the smallest class holds only about $150$ probands for a profile with several hundred free
parameters, which is thin, so the bins are best kept nearer $2{,}000$, where the smallest class has
room.

## How the floor is estimated

:::{dropdown} The estimation procedure

The `nmin` stage refits at descending sample sizes and records, at each size, the profile
correlation to the full-sample reference, with the smallest class proportion and the average
posterior certainty as degradation checks. The floor is the smallest size at which a monotone
(isotonic) fit of correlation against log-size reaches the reproduction benchmark, with a bootstrap
interval; pooling every fit makes it robust to the scatter the recovery measure carries.
:::

The sweep is seeded and resumable, like the other multi-seed stages; see
[the pipeline and its cache](../guides/pipeline-and-caching) for the seeding and checkpointing.
