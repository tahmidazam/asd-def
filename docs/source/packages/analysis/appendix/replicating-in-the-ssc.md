# Replicating in the SSC

:::{admonition} The question
:class: note

Do the four classes reappear in a second, independent cohort? The reproduction and the stability
checks are all within SPARK. This investigation fits a fresh model on the SPARK features shared
with the SSC, projects it onto the SSC, and asks whether the seven-category class profiles agree
across the two cohorts, the same measure the authors used to declare replication.
:::

:::{admonition} The result
:class: tip

The classes replicate. The SPARK model projects onto 798 SSC probands across the 106 shared
features, and the seven-category profiles correlate at $r = 0.90$, beyond a permutation null
(calibrated $p = 0.006$). A proband bootstrap puts a 95 per cent interval of $[0.79, 0.94]$ on
that value, which includes the authors' published $r = 0.927$. All seven categories correlate at
$0.85$ or above and five of them at $0.90$ or above; the two lowest are developmental at $0.86$
and attention at $0.85$, and developmental departs most from the published value, the one category
the SSC builds entirely from parsed milestone ages. The interval stays wide because the SSC sample
is small and its two smallest classes hold only about four and five per cent of it.
:::

:::{figure} /_figures/replication.png
:alt: Cross-cohort replication of the class signatures between SPARK and the SSC
:width: 90%
:align: center

Cross-cohort replication, from `analysis replicate` runs projecting the SPARK model onto 798 SSC
probands. (A) Every class-by-category signature value, SSC against SPARK, around the line of
equality ($r = 0.90$, $[0.79, 0.94]$), with the authors' overall $r = 0.927$ noted alongside. (B)
The per-category correlation for two training conditions, the full `2026-03-23` release and the
cohort cut back to the records present at the authors' V9 freeze (see
{doc}`subsetting the cohort to the V9 freeze <../appendix/subsetting-to-the-v9-freeze>`), against the
values Litman et al. (2025) report (markers). The two conditions track each other (overall $r = 0.90$
for both), so the SPARK records the model trains on do not drive the replication; developmental, the
one category the SSC builds from parsed milestone ages rather than a standard instrument, reads
$0.86$ on the full release and $0.91$ on the subset, against the published $0.98$.
:::

## Reading the result

On the SSC 15.3 release the SPARK model projects onto 798 probands, across the 106 features the
two cohorts share, and the seven-category profiles correlate at $r = 0.90$. The permutation null
puts that value beyond chance: of the 172 label shuffles that yield a defined correlation, none
reaches the observed $r$, so the calibrated $p$-value is $(1 + 0) / (1 + 172) = 0.006$. A shuffle
that flattens a class profile gives an undefined correlation and drops from the null, so the
$p$-value rests on the shuffles that produced a usable profile; with more permutations the bound
would tighten, but it already places the observed correlation outside the null.

Resampling the 798 SSC probands, again with the projected labels held fixed, puts a 95 per cent
interval of $[0.79, 0.94]$ on the overall $r$, over 500 resamples. That interval is wider than the
reproduction's $[0.89, 0.92]$, because the SSC sample is small and its two smallest classes hold
only about four and five per cent of it, but it includes the authors' published $r = 0.927$. The
replication is therefore clearly above the permutation null and statistically consistent with the
published value, if less precise than the within-SPARK reproduction.

The correlation runs high across the seven categories, with one exception. Each value is shown
against the coefficient Litman et al. (2025) report for the same category:

| Category | This work | Litman 2025 |
| --- | --- | --- |
| restricted or repetitive | 0.97 | 0.97 |
| self-injury | 0.96 | 0.92 |
| anxiety or mood | 0.92 | 0.98 |
| disruptive behaviour | 0.90 | 0.94 |
| social or communication | 0.90 | 0.89 |
| developmental | 0.86 | 0.98 |
| attention | 0.85 | 0.92 |

Six of the seven sit within about $0.07$ of the published value, three of them at or above it.
Developmental shows the largest gap, $0.86$ against the published $0.98$. It is the one category
the SSC assembles entirely from the background-history milestone ages: of the eleven developmental
features, the SSC provides the nine milestone ages, while the second phrase milestone
(`combined_phrases_age_mos`) and the school item (`repeat_grade`) have no SSC source, so unlike
the others it carries none of the standard-instrument items (see
[parsing the SSC milestone ages](../guides/parsing-ssc-milestone-ages)). Two further properties
shape the value. Each per-category correlation is taken over the four classes alone, the coarsest
of the seven readings, so it moves sharply when one class shifts; and under projection two of the
four classes hold only about four and five per cent of the SSC, so a category whose signal rests on
those classes is estimated from very few probands. The gap is best read as the joint effect of the
milestone ages standing in for the authors' hand-cleaned file, the two absent features, and the
thin projected classes, rather than as any one of them. The overall $r$, taken over the full
four-class, seven-category profile, is the stabler summary.

The records the SPARK model trains on do not drive the gap. Retraining on the cohort cut back to the
records present at the authors' V9 freeze (the second condition in the figure; see
{doc}`subsetting the cohort to the V9 freeze <../appendix/subsetting-to-the-v9-freeze>`) gives an
overall correlation of $0.90$, matching the full release, with a developmental correlation of $0.91$
against $0.86$ on the full release. Developmental is no lower on the smaller, earlier cohort, so the
divergence from the published value sits on the SSC side of the projection, in the milestone ages and
the thin projected classes, not in the SPARK training records.

The class sizes under projection are more skewed than under the SPARK fit: the SSC places about
62 per cent of its probands in the largest class (against 39 per cent under the shared-feature
SPARK fit) and only about four and five per cent in the two smallest (against eight and 16 per
cent in SPARK). The replicate stage fits its own model on the shared features and carries the
class ids straight across both cohorts, so these are class sizes under one model, not a
re-alignment to the named reference.

## How the projection works

:::{dropdown} Fitting on the shared features and projecting

The `replicate` stage fits a fresh model on SPARK restricted to the features shared with the SSC,
then that fitted model predicts class labels on the SSC. Because both cohorts pass through the one
model, the class ids already correspond, so no cross-cohort label alignment is needed. The
replication measure is the correlation of the seven-category profiles between the two cohorts, the
same currency the authors used to declare replication.

Two points of care. StepMix validates a prediction input by its feature count, not its column
names, so the SSC measurement matrix is reindexed to the exact SPARK column order before
prediction. And the released code reports the correlation without a null; the package adds a
permutation null that shuffles the SSC class labels and recomputes the correlation, so the
observed value is read against chance.
:::

## Caveats

The replication carries two caveats. The SSC milestone ages are parsed from free text here, not
read from the authors' unreleased hand-cleaned file, and that parse is the whole of the
developmental category, the category that departs most from the published profile. And the
shared-feature complete-case reduction leaves 798 probands, below the full SSC release, so the
replication is read against the published value rather than offered as an exact reproduction.
