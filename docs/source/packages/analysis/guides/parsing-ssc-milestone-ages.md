# Parsing the SSC milestone ages

The developmental-milestone ages drive the "developmental" category that defines the *Mixed
ASD with developmental delay* class, so they have to cross from the SSC into the shared
schema cleanly. In SPARK each milestone is a numeric age in months (`walked_age_mos` and the
rest), coded from a dropdown. In the SSC the same milestones are stored as free text, and
the entries vary widely: `age_walked_alone` alone has more than four hundred distinct
spellings, from `13 months` and `1 yr` to a bare `12`. The SSC backend turns that free text
into months with `parse_age_months`, in `analysis.cohort.schema`.

The parsing is the package's own work. Litman et al. read these milestones from a
hand-cleaned background-history file that was never released, and it is the only SSC
instrument they pre-clean out of band (every other instrument is harmonised inline in their
`generate_ssc_data`). Without their cleaning recipe, the rules here are ours. They are a
documented deviation from the released pipeline, and their fidelity is assessed at the
replication stage rather than asserted.

## Why a plain numeric coercion is not enough

The SSC milestone columns are object-typed text, so `pandas.to_numeric` keeps only the
entries that are already bare numbers and turns everything else into a missing value. On the
SSC 15.3 proband release that recovers under a tenth of the recorded ages, which collapses
the complete-case sample before the model ever runs. Parsing the free text instead recovers
the large majority of the recorded values, so the cross-cohort sample is no longer thrown
away at this step.

## What the parser recognises

Each entry is lower-cased and lightly cleaned (surrounding punctuation, tildes, commas, and
plus signs removed), then matched against a set of forms:

- a number with a month unit is read as months: `13 months`, `13 mos`, `12 mon`, `13m`.
  Common abbreviations and typos for "months" are included. A bare number with no unit is
  read on the scale the SPARK distribution for that milestone makes more likely, so `12` for
  an early milestone is twelve months and `4` for a late one is four years (see [resolving the
  months-or-years scale](#resolving-the-months-or-years-scale)).
- a number with a year unit is multiplied by twelve, with an optional trailing months part:
  `1 year`, `2 yrs`, `1 yr 6 mo`.
- a number with a week unit is converted from weeks, using 30.4375 days per month (the mean
  Gregorian month): `6 weeks`, `6 wks`, `1 week`.
- a compound entry sums its distinct unit parts: `2 yrs 10 mos`, `1 year 2 weeks`, `3y;3m`.
- a half- or quarter-year fraction is read as a decimal first: `3 1/2 yrs`, `2 1/2 years`,
  and the half, quarter, and three-quarter fraction glyphs.
- a "years old" form is handled, whether written out or abbreviated: `4 years old`, `3 y.o.`,
  `4 y/o`, and `age 4`.
- `at birth` is zero.
- a range, or an "N or M" entry, is read as the midpoint of its endpoints: `12-14`,
  `18 months to 2 years`, `7 or 8 months`. When the left endpoint is a bare number it borrows
  the right endpoint's unit, so `1-2 years` reads as one to two years rather than one month to
  two years.
- a bound or inequality has the bound dropped and the stated age taken: `<3 mos`, `>42 mos`,
  `before 1 year`, `under 12 months`. HTML-escaped angle brackets (`&lt;`, `&gt;`) are decoded
  first.
- an explicit statement that the milestone was never reached (`never`, `not yet`, `hasn't
  walked`, `unable to`) is read as the SPARK `888 = not yet` code, not as a missing value, so a
  proband with a severe delay is kept rather than dropped at the complete-case step. The cue has
  to carry no digit, so a dated note such as `12 mos (lost at 15 mos)` is handled as a regression
  narrative below, not as a never-reached milestone.

## Resolving the months-or-years scale

A free-text age often omits its unit, and a bare number is then ambiguous: `4` is four months
for an early milestone such as smiling, but four years for a late one such as bowel training.
The parser settles the scale for each milestone against SPARK, where the same milestone is
recorded as a clean age in months. For a unit-less number $x$ it compares how likely $x$ months
and $12x$ months (the years reading) are under that milestone's SPARK distribution, estimated as
a kernel density in log-age, and takes the more likely; the years reading is considered only
when $12x$ stays within the milestone range.

The result follows each milestone's own distribution. A small number stays in months where
children reach the milestone in months (a child first walks at `13`, read as thirteen months),
and becomes years where they reach it in years (a child is bowel trained at `4`, read as
forty-eight months). A written unit is taken as written; only unit-less numbers are resolved
this way, including a bare number left after a bound such as `under 2`.

The reference distribution is the SPARK proband background-history milestones at the release the
reference fit uses. The resolver is built per milestone by
{py:func}`analysis.cohort.schema.build_milestone_disambiguator` and applied in the SSC backend.

## What it leaves missing, and why

Some entries carry no single age, so the parser returns a missing value and they drop at the
complete-case step, exactly as a missing milestone would. Leaving them missing is a choice,
not a gap:

- text with no number: `normal`, `on time`, `unsure`, `within normal limits`. These are not
  ages. (A `never` or `not yet` that states the milestone was not reached is the exception above,
  read as the SPARK not-yet code.)
- a calendar date entered in the age field: `03/2003`, `12/01`, `09/27/93`. Reading these as a
  number of months would be wrong, so they are excluded.
- a regression or loss narrative: `12 mos (lost at 15 mos)`, `18 months then declined till
  after 2`. These describe a trajectory with more than one age. A compound is summed only when
  its units are distinct and it carries no loss or sequencing cue, so a repeated unit or a word
  like "lost", "stopped", "regressed", or "then" sends the entry to missing rather than to an
  arbitrary single value.
- a milestone recorded in days, such as `1 day` or `2 days`. For these milestones a value in
  days is more likely a stray entry than a real age, so days are not converted.

## Choices worth stating

- The parsed ages stay continuous in months. SPARK codes its milestone dropdown in whole
  months above the first two years, but the features are modelled with a Gaussian density, so
  a value such as `6 weeks` is kept as roughly 1.4 months rather than rounded to the dropdown
  grid.
- A week is converted at 30.4375 days per month, the mean length of a Gregorian month, so
  weeks and months sit on one continuous scale.
- A parsed age above the SPARK `over 7 years` code is capped at 85 months, the top of the SPARK
  milestone dropdown, which also discards the occasional mis-parsed outlier.

## Result

On the SSC 15.3 proband release the parser reads about 97 per cent of the non-missing
milestone entries, against under a tenth for a plain numeric coercion. The values that remain
unread are dominated by the genuine non-ages above: free text with no number, calendar dates,
and regression narratives. The ceiling is the data itself, since a share of SSC milestone
responses are qualitative rather than point ages, and that is a fidelity limit on the SSC
replication regardless of how the text is parsed.

The SSC records one combining-language milestone, mapped to `combined_words_age_mos`. Two SPARK
background-history features have no SSC source: the second combining milestone
`combined_phrases_age_mos`, and the school item `repeat_grade`. That is a coverage gap in the
instrument, separate from parsing, so the SSC backend provides the nine milestones the SSC
collects.

The parsing entry point is {py:func}`analysis.cohort.schema.parse_age_months`. For where it
sits in the cohort layer, see [the cohort interface](the-cohort-interface).
