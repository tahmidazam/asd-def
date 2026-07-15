# Conditioning on demographics

The drift is the movement of each class centroid through the 238-feature phenotype space as the
cohort is ordered by diagnostic era or age at diagnosis. This guide covers the two ways a demographic
covariate is set against that movement, and it is careful about which question each one answers,
because the two are easy to conflate.

## Two questions, two reads

A demographic is a per-proband variable and the drift distance is a per-class quantity, so there is no
direct correlation between a demographic and the drift. The question has to be made precise, and it
splits in two.

- Correlated. Does the partition drift when the cohort is ordered by the demographic itself, rather
  than by time? This is the {doc}`displacement atlas <screening-orderings-with-the-atlas>` question,
  and the ordered demographics (income, education, the inferred parental ages, the family-history and
  complication counts) join the atlas as ordering axes. A related, one-number version is how much of
  the timing axis the demographic even spans, reported here as the axis $R^2$.
- Explains. Does the timing drift shrink once the demographic is removed from the phenotype? This is
  the stronger question, and it is what the conditioning stage measures.

## How conditioning works

The demographic is never clustered on, and it does not need to be: it is the variable that is
controlled for, not a feature the mixture was fitted on. Conditioning takes three steps.

1. Take the 238 clustered features, the space the drift lives in.
2. Regress each clustered feature on the demographic across probands and keep the residual, so the
   part of every feature that a linear reading of the demographic predicts is removed
   (`analysis.blocks.residualise`).
3. Recompute the class endpoint drift on the residualised features.

The shrinkage is the fraction of a class's drift removed by that residualising,
$1 - \lVert d_\text{conditioned} \rVert / \lVert d_\text{raw} \rVert$
(`analysis.blocks.conditioning_shrinkage`). A value near zero means the demographic leaves the drift
untouched; a value near one means it accounts for the movement; a small negative value means the drift
grew slightly, the noise around no effect. A one-hot demographic (family type, marital status, race)
enters as a low-dimensional block and is residualised the same way, so a nominal covariate is read
without being forced into an order.

## The ceiling

A covariate orthogonal to the timing axis cannot account for an axis-ordered drift, whatever feature
variance it explains, so its shrinkage is near zero by construction. The stage reports the axis $R^2$,
the coefficient of determination of the timing axis regressed on the covariate, beside each covariate's
shrinkage, so this ceiling is visible: a covariate with a near-zero axis $R^2$ has no room to explain a
timing drift, and its small shrinkage is expected rather than a surprise.

This is why the conditioning read is a descriptive partial association, not a causal claim. A covariate
that is itself downstream of the diagnosed phenotype (a family's living arrangement, say) would be
over-adjusted, and a covariate correlated with time will shrink the drift whether or not it drives the
phenotype. The stage measures how much of the movement a demographic linearly spans, and the prose
around it says no more.

## The covariate pool

The pool is `analysis.demographics.DEMOGRAPHICS`, grouped into four families: socioeconomic (household
income, area deprivation, parental education), family structure (family type, marital status, living
arrangement, the ASD family-history counts, multiple birth), parental (the inferred maternal and
paternal ages at birth, parental occupation), and individual (sex, race, the perinatal-complication
count). One covariate is derived: parental age at birth is the enrolled parent's registration age minus
the child's, joined through the biological-parent link.

Coverage is a modelling-cohort property, so it is computed at run time. A covariate joining fewer than a
thousand probands is dropped and named, and a covariate with no variance on the cohort is dropped too,
so a survey-version field that reaches only a few hundred probands or a flag that is constant on the
cohort is never read as if it were well covered.

## Running and reading it

```
uv run analysis demographic-conditioning --axis era
uv run analysis demographic-conditioning --axis age_at_diagnosis
uv run figures demographic-conditioning
```

Each run writes `demographic_conditioning_<axis>.parquet`, one row per covariate and reference class:
the shrinkage, the raw and conditioned magnitude, the covariate's axis $R^2$, and the joined sample
size. The figure lays the covariates out as rows against the four classes for each axis, with the axis
$R^2$ as a narrow panel to the left. Read the two panels together: a covariate can only appear in the
shrinkage panel if it first appears in the axis $R^2$ panel, so a near-empty $R^2$ column and a
near-empty shrinkage panel are the same finding seen twice. The investigation
{doc}`../appendix/do-demographics-explain-the-drift` reads the result on SPARK.
