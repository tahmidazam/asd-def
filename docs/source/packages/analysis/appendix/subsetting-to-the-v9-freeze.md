# Subsetting the cohort to the V9 freeze

Litman et al. (2025) fit on a SPARK release this project does not hold. This guide explains how
the `cohort` stage cuts a later release back in time to the probands present at an earlier
freeze, so the model can be refit on an approximation of the data the authors used.

## The release they used

The paper names the release only as "SPARK Phenotype Dataset V9". The released preprocessing
pins the date, reading from a `SPARK_collection_v9_2022-12-12` directory and files such as
`scq_2022-12-12.csv`. So V9 was frozen on 2022-12-12, and that date is what the cut works from.

This project holds the later `2025-03-31` and `2026-03-23` releases. SPARK grows over time, so
`2026-03-23` holds every V9 proband still enrolled plus everyone recruited since. The reference
fit on it recovers the four classes but diverges from the paper in its class proportions, and a
natural question is whether the records added since V9 are the cause. Answering it needs the
cohort cut back to the freeze.

## What the cut recovers, and what it does not

The cut keeps a proband only if it was present at the freeze: enrolled by 2022-12-12 with each of
its four instruments completed by then. This recovers the V9 *roster*, the probands the authors
could have drawn from.

It does not recover the V9 *data*. SPARK revises phenotype between releases, back-filling
corrections and adding responses completed after a proband enrolled, so a kept proband carries
its `2026-03-23` values, not its V9 ones. The cut isolates which probands were present, not which
values were recorded then; matching the values would need the V9 file, which is not held. The
companion investigation reads the cut against a size-matched random subsample, so that "fewer
records" and "different records" can be told apart.

## The two gates

A proband enters only when both gates pass. Both turn on the calendar year, the resolution of the
date fields SPARK exposes.

The roster gate keeps a proband whose `core_descriptive_variables.registration_year` is at or
before 2022.

The completion gate keeps an instrument only if it was completed by the freeze year, read two
ways across the four cohort instruments:

- SCQ, RBS-R, and the background-history forms (child and sibling) carry a completion year,
  `eval_year`, so the gate is `eval_year <= 2022`, exact to the year.
- CBCL 6-18 carries no completion year, only the age at evaluation, so its year is reconstructed
  from the registration anchor,

  $$ \text{cbcl year} = \text{registration year} + (\text{age at evaluation} - \text{age at registration}), $$

  and the proband is kept when that year is below 2023, so a CBCL completed midway through 2022 is
  kept rather than rounded up.

A proband missing any gate field drops out, since the gate cannot then be confirmed. Because the
cohort is reduced to complete cases over all four instruments, a proband survives only when every
instrument cleared the freeze, the "present in V9" condition.

## Running it

The cut is the `--as-of` option on `cohort` and on every stage that builds one:

```
uv run analysis cohort -d spark -v 2026-03-23 --as-of 2022-12-12
```

It enters the run hash, so the subset is cached under its own hash and the full-release reference
artefacts are untouched. Passing `--as-of` to `fit`, `align`, `select`, `stability`, or
`replicate` fits on the subset throughout; for `replicate` the cut applies to the SPARK training
cohort only, since it is a SPARK timing field, and the SSC is projected onto in full.

The size-matched control is `--sample-n`, a random draw of the full release at a fixed
`--sample-seed`:

```
uv run analysis cohort -d spark -v 2026-03-23 --sample-n 5324 --sample-seed 0
```

Set `--sample-n` to the subset's size and the two fits differ only in which probands they hold,
not how many, so a difference between them is compositional rather than a matter of sample size.
The subset size and the results of fitting on it are reported in the investigation on the records
added since V9.
