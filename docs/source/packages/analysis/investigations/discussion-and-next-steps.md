# Discussion and next steps

The investigations so far reproduce the four data-driven classes of Litman et al. (2025) on
the SPARK release held here, name them as the authors did, and test how solid that reference
solution is before the novel work begins. This page reads the results together and sets out
what the rest of the analysis does. It is the interpretation, so it states what the evidence
supports and what it does not, and it describes the stratified analysis as planned work rather
than as a result already in hand.

The motivating question is whether the four classes are a stable property of the phenotype or
an artefact of fitting one mixture model to a pooled sample. Two facts about how an autism
diagnosis is made put that in doubt: age at diagnosis indexes partly distinct aetiology, and
the population carrying the diagnosis has shifted over calendar time. If the diagnosed
population differs systematically by age at diagnosis and by diagnostic era, a model fit on
the pool may be mixing populations. The test is to re-estimate the model within strata of
those two axes and ask whether the classes survive.

## What is established

The reference solution reproduces. On SPARK 2026-03-23 the model recovers four classes whose
proportions (about 39, 29, 18, and 15 per cent) match the published 37, 34, 19, and 10, whose
named-class anchors all hold, and whose seven-category profile correlates with the published
figure at $r = 0.90$, the same order as the authors' own cross-cohort replication ($r =
0.927$). The named fit this produces is the fixed target every later comparison is measured
against. The detail is in [reproducing the reference classes](reproducing-the-reference-classes).

How many classes the data support is a softer question than a single number. The information
criteria do not stop at four: at this sample size (11,704 probands) they keep falling and
minimise at nine, the over-extraction expected when a criterion's complexity penalty is light
relative to the likelihood. The cross-validated log-likelihood gains little past four, and the
higher-class solutions degenerate into tiny, poorly separated classes. Four classes is
retained as the authors chose it, by reading the criteria rather than by an automatic rule.

The four classes are stable as structures. Under both re-initialisation and resampling the
seven-category profile reproduces at about 0.91 to 0.92, and no fit ever collapsed a class.
Proband-level membership is softer: the adjusted Rand index sits around 0.63 to 0.65, because
probands near a class boundary move between fits. The class definitions are stable; who lands
in which class at the edges is less so. The stability and selection detail is in
[stability, selection, and replication](stability-selection-and-replication).

The classes reappear in a second cohort, with one qualification. Projected onto 771 SSC
probands, the seven-category profiles correlate at $r = 0.76$ ($p = 0.005$ against a
permutation null). The six categories built from the standard instruments (the CBCL, the
RBS-R, and the SCQ) replicate at 0.71 to 0.91; the developmental category is the exception at
0.49, and it is the one built from the SSC background-history milestone ages, which are parsed
from free text here rather than read from the authors' unreleased clean file. Within SPARK,
where the milestones are measured the same way as in the reference, the developmental category
is as stable as the rest (0.94), so its cross-cohort weakness is a property of the SSC parsing,
not of the class.

The stratified analysis needs a sample-size floor, and that floor is now fixed. Refitting at
descending sample sizes puts the minimum viable stratum size at about 1,000 probands as a
point estimate, with a 90 per cent bootstrap interval of 600 to 3,955. The recovery measure is
noisy enough that the floor is a range rather than a point, so the conservative reading is the
upper bound: the stratification bins are kept above about 2,000 probands, nearer 4,000 where
the smallest class is comfortable.

## How to read it

The pooled reference is solid enough at the profile level to anchor the stratified test. That
matters because the test compares stratum-specific fits against this reference: a fragile
baseline would make any drift uninterpretable, and the stability work shows the baseline is not
fragile. The class profiles reproduce across initialisations, resamples, and (for the
instrument-based categories) a second cohort.

Two qualifications are already visible, and both bear on the science rather than the tooling.

The first is the gap between profile stability and membership stability. Stable class profiles
with unstable boundary membership is the pattern a continuum with regions would produce: the
regions (the profiles) are reproducible, while a proband sitting between two regions is
assigned differently from fit to fit. The stability result does not decide between discrete
classes and a graded continuum; it shows the question is live, and it is one the sensitivity
work tests directly by comparing the four-class model against a lower-dimensional severity
gradient.

The second is the developmental category. It is the one place the cross-cohort picture weakens,
and the weakness is methodological. The authors pre-cleaned the SSC milestone ages out of band
with a file they did not release, so the milestone parsing here is the package's own. Closing
that gap needs either the authors' cleaning procedure (requested) or further validation of the
parser, and until then the developmental category's cross-cohort number is read with that
caveat. It does not weaken the within-SPARK result, where the same category is stable.

Robustness is necessary but not sufficient for validity. A partition can reproduce across
seeds, resamples, and cohorts and still reflect parent-reported, deficit-framed measurement
rather than a biological kind. The stability shown here is evidence the partition is
reproducible, not that the four classes are biologically grounded; the two claims are kept
separate, and the genetics arm and the construct-validity checks speak to the second.

## Next steps

The reference solution has passed its benchmark, so the order of the remaining work is fixed
to keep the novel analysis confirmatory rather than exploratory.

- Pre-registration. Before any stratified fit, freeze the bin definitions (above the
  stratum-size floor established here), the alignment method, the drift metrics, the
  construction of the permutation null, and the numeric thresholds that separate a stable read
  from a drift read. Freezing these in advance is what lets the stratified result be reported
  as confirmatory.
- The stratified analysis. Re-estimate the mixture model independently within strata of age at
  diagnosis and of diagnostic era, align each stratum's classes to the named reference, and
  measure how far each class profile moves. Read that movement against two baselines: a
  permutation null that re-fits within strata of the same sizes formed by shuffling the stratum
  labels, so the only structure removed is the association with the axis; and the distance
  between distinct reference classes, so a shift is judged on the scale of the partition itself.
  The era axis carries a specific threat, the lag between when the phenotype is measured and
  when the diagnosis was made, which is quantified and tested rather than assumed away.
- Sensitivity and triangulation. Test the standard threats to the original result: dependence
  on the feature set, confounding by intellectual disability and developmental delay, rater
  effects (re-deriving the classes from the SSC's clinician-administered instruments), and the
  categorical-versus-dimensional question the membership stability raises.
- Genetics. Once the SPARK and SSC genotype data are obtained, test whether the
  genotype-to-phenotype mapping drifts across the same strata (polygenic-score-by-class and
  rare-variant-burden-by-class associations), and whether the four-way structure survives once
  cognitive level is held constant. This arm is planned and gated on data access.

The machinery these steps need is built: the cohort interface over SPARK and the SSC, the
alignment by profile similarity, the drift metrics, the permutation null, and the stratum-size
floor that bounds the bins. What remains is to freeze the plan and run the stratified fits.
