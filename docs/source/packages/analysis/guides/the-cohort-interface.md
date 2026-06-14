# The cohort interface

The reproduction and the planned stability work both need a proband-by-feature matrix, and
the work is meant to run on two cohorts: SPARK and the SSC. Rather than write each analysis
twice, the package puts both cohorts behind one interface. A stage is written once against
that interface and runs on either cohort; each cohort has a backend that maps its raw tables
onto a shared feature schema. This guide describes the interface, the SPARK backend that the
reproduction uses, and the SSC backend and its current limits.

## One interface, two backends

A backend integrates its instruments into one harmonised, complete-case frame and returns a
`CohortMatrix`: the clustered features, the model covariates (sex and age at evaluation), and
the cohort and version the matrix came from. A backend also declares whether it can provide
the diagnosis-timing fields the stratified analysis will need; those fields exist in SPARK
but not in the SSC, so they are an optional capability rather than part of the core contract.

The backends read through the `dscat` catalogue: it resolves each instrument's source CSV
and the package reads only the columns a stage needs, so a whole instrument file is never
loaded into memory.

## The SPARK backend

The SPARK backend reproduces the integration in the authors' released preprocessing. It
reads the SCQ-Lifetime, the background-history child and sibling forms, the RBS-R, and the
CBCL 6-18; screens each on age and on its missingness counter; joins them on the proband id;
and keeps complete cases.

Two choices depart from the released code, both deliberate.

**The kept feature set is pinned to the authors' list.** The released preprocessing selects
features by dropping columns, so the kept set is implicit and shifts on a newer release. The
authors shared the final 238-feature list they fit on, and the backend pins to it: it reads
exactly those features plus the covariates and the screening columns. This is why the
package depends on the authors' list as an input rather than rederiving the selection.

**The screening follows the released code, not its prose description.** The cohort filters
are the ones the code applies: ages 4 to 18 on the SCQ, both background-history forms, and
the RBS-R; a missingness counter below one on the SCQ and the RBS-R; and no row filter on the
CBCL. The instrument validity-flag columns are dropped rather than used as filters. Because
the kept set is pinned, the released step that drops columns with more than ten per cent
missingness does not apply; the backend keeps the pinned features and reduces to complete
cases on them. The CBCL competence items, which arrive as strings, are recoded to the ordinal
integers the released preprocessing used.

On SPARK 2026-03-23 this yields 11,704 probands by 238 features, with sex and age at
evaluation as covariates. That cohort is larger than the authors' 5,392, because the release
is several years later and much larger; the reproduction is therefore judged on the class
profile and proportions, not on the cohort size (see
[Reproducing the reference classes](reproducing-the-reference-classes.md)).

## The SSC backend

The SSC backend harmonises the SSC proband instruments onto the same schema: the CBCL 6-18,
the RBS-R, the SCQ-Lifetime, the core descriptive table, and the background-history form. It
renames the SSC columns to the SPARK names, recodes sex and the SCQ yes/no answers, and
selects the shared features positively (a feature is provided when its SSC column exists
after renaming). The SSC instruments cover a subset of the schema, so the backend exposes
that subset.

Two limits are worth stating plainly. The authors built the SSC class structure from a
hand-cleaned background-history file that was not released, so the milestone mapping here is
the package's own: nine of the eleven SPARK milestone features map cleanly to the raw SSC
form, and two have no clean equivalent. The backend runs and produces a harmonised matrix,
but its fidelity to the authors' SSC pipeline and its coverage are matters for the planned
replication stage, not settled here. The SSC backend does not provide diagnosis timing, so
the age-at-diagnosis and diagnostic-era axes are SPARK-only.
