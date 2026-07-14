"""Run configuration: constants, the reference pipeline's fixed choices, and input paths.

This module gathers the values that pin a run down: the cohort the reference fit uses,
the covariates and age window from Litman et al., the default mixture-model
hyperparameters, and the locations of the author-provided reference inputs (the final
feature list and the feature-to-category map) and the released typing pickles.

The reference inputs live under the gitignored ``.literature`` tree, because they come
from the authors' released code and our correspondence with them and are not ours to
redistribute. Each location can be overridden with an environment variable so a run can
point at a copy elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

# Cohort the reference fit is built on. The 2025-03-31 release is held too and is used as
# a release-axis stability check (plan section 7); 2026-03-23 is the reference.
REFERENCE_DATASET = "spark"
REFERENCE_VERSION = "2026-03-23"

# Covariates entered into the structural model, and the age-at-evaluation window, both as
# in the released ``GFMM.py``. Neither covariate is part of the clustered feature set.
COVARIATES: tuple[str, ...] = ("sex", "age_at_eval_years")
AGE_AT_EVAL_RANGE: tuple[int, int] = (4, 18)

# Intelligence-quotient threshold that splits the cross-cohort cognitive-impairment axis into
# intellectual-disability positive and negative strata (plan section 8). Set to 80 because
# SPARK's machine-learned cognitive-impairment flag (``ml_predicted_cog_impair``) is trained
# against measured full-scale IQ below 80, so dichotomising the SSC full-scale deviation IQ at
# the same value gives the two cohorts the same construct. This is a phase-5 sensitivity axis,
# not part of the frozen phase-4 pre-registration (section 12a).
ID_IQ_THRESHOLD = 80

# The freeze date of the SPARK Phenotype Dataset version Litman et al. fit on (V9). The paper
# text names only "V9"; the released preprocessing pins the date, hard-coding the directory
# ``SPARK_collection_v9_2022-12-12`` and per-instrument ``<table>_2022-12-12.csv``. Passing
# this as the cohort ``--as-of`` cutoff reconstructs the probands present at that freeze from a
# later release, to test whether divergences from the paper trace to records added since.
V9_CUTOFF = "2022-12-12"

# The instruments integrated into the cohort matrix, with the per-instrument validity flag
# and missingness counter the released preprocessing screens on (where the instrument has
# one). Order matters only for reproducible logging.
COHORT_INSTRUMENTS: tuple[str, ...] = (
    "scq",
    "background_history_child",
    "background_history_sibling",
    "rbsr",
    "cbcl_6_18",
)

# Default StepMix general finite mixture model settings. The reference fit overrides
# ``n_components`` only incidentally (it is already 4); ``n_init`` and the one-step
# covariate parametrisation match Litman et al. The remaining StepMix defaults (max_iter,
# tolerances) are recorded in each manifest rather than fixed here, so a StepMix upgrade
# that changes a default is visible.
DEFAULT_N_COMPONENTS = 4
DEFAULT_N_INIT = 200
DEFAULT_N_STEPS = 1

# Resamples for the proband bootstrap that puts a confidence interval on the headline
# reproduction and replication correlations (the labels are held fixed; only the cohort is
# resampled). Five hundred is enough for a stable 95 per cent percentile interval at a
# manageable cost, since each resample recomputes the per-class enrichment.
DEFAULT_N_BOOTSTRAP = 500
DEFAULT_BOOTSTRAP_SEED = 0

# Permutations for the stratified drift permutation null (plan section 12a). 1000 is the
# frozen confirmatory value, chosen for about three-figure p-value resolution and a stable
# 95th-percentile threshold across the strata-by-class tests. It is a knob, not a constant: a
# pilot or a debugging run passes a smaller value (down to 1) to exercise the instrumented
# pipeline cheaply, while the confirmatory result reported in the manuscript uses this default.
DEFAULT_N_PERMUTATIONS = 1000
DEFAULT_STRATIFY_SEED = 0

# Number-of-classes search (H0C hypothesis, plan section 7 / H0C). The
# confirmatory statistic is a warm-started parametric bootstrap likelihood-ratio test (BLRT)
# for $K$ against $K+1$ (splitting) and $K-1$ (merging), anchored at four classes and stepped
# outward to the caps. These defaults set the recipe folded into every run hash.
DEFAULT_ORDER_N_INIT = 5
DEFAULT_ORDER_K_ANCHOR = 4
DEFAULT_ORDER_K_CAP = 7
DEFAULT_ORDER_K_FLOOR = 2
# Warm-start recipe for the $K+1$ fit: split each of the $K$ classes in turn (so $K$ warm
# starts) and add this many random restarts, keeping the best. The observed and every
# bootstrap sample use this identical recipe, so the BLRT is not biased by under-fitting the
# alternative in the null.
DEFAULT_ORDER_N_RANDOM = 2
DEFAULT_ORDER_SPLIT_JITTER = 0.5
DEFAULT_ORDER_MAX_ITER = 500
# Staged bootstrap schedule: screen every step at ``B_SCREEN``, escalate to ``B_ESCALATE``
# only where the screen $p$ falls below ``ESCALATE_THRESHOLD``. The Phipson-Smyth add-one
# $p$-value makes the smallest attainable $p$ ``1 / (B + 1)``.
DEFAULT_ORDER_B_SCREEN = 199
DEFAULT_ORDER_B_ESCALATE = 999
DEFAULT_ORDER_ESCALATE_THRESHOLD = 0.10
# Rejection level for a single BLRT step, and the FDR level for the across-strata decision.
DEFAULT_ORDER_ALPHA = 0.05
DEFAULT_ORDER_SEED = 0

# Ordering-shuffle permutations for the trajectory stage's directional test (plan section 7).
# The statistic is a class's net young-to-old displacement; the null shuffles the stratum
# order. This is a cheap resampling of stored centroids, not a refit, so a high count is
# affordable; 20000 gives a stable 95th-percentile threshold and three-figure p-values.
DEFAULT_TRAJECTORY_SHUFFLES = 20000
DEFAULT_TRAJECTORY_SEED = 0

# Litman's per-fit class-ID to name mapping, recorded for reference only. StepMix assigns
# class IDs arbitrarily on every fit, so this mapping does not transfer to our fits; we
# recover the named classes by profile alignment (plan section 6a), not by this table.
LITMAN_CLASS_NAMES: dict[int, str] = {
    0: "Moderate challenges",
    1: "Broadly affected",
    2: "Social/behavioral",
    3: "Mixed ASD with DD",
}

# Where the released Litman code and the author correspondence sit, relative to the repo
# root. Both are gitignored.
_LITERATURE_SUBDIR = ".literature/litmanDecompositionPhenotypicHeterogeneity2025a"
_ATTACHMENTS_SUBDIR = f"{_LITERATURE_SUBDIR}/emails/feature-request/attachments"
_LITMAN_CODE_SUBDIR = f"{_LITERATURE_SUBDIR}/code/asd-pheno-classes"


def _resolve(env_var: str, default: Path) -> Path:
    """Return the environment-variable override for a path, or ``default``."""
    override = os.environ.get(env_var)
    return Path(override).expanduser().resolve() if override else default


def author_feature_list(root: Path) -> Path:
    """Return the path to the authors' final feature list (``mixture_model_columns.csv``).

    This 238-feature list is the authoritative set the general finite mixture model was
    fit on. It resolves the ambiguity the released preprocessing left open, because that
    code drops columns rather than naming the kept set (plan section 5).

    Parameters
    ----------
    root : Path
        The monorepo root.

    Returns
    -------
    Path
        Location of the feature-list CSV. Overridable with ``ANALYSIS_FEATURE_LIST``.
    """
    return _resolve(
        "ANALYSIS_FEATURE_LIST", root / _ATTACHMENTS_SUBDIR / "mixture_model_columns.csv"
    )


def author_category_map(root: Path) -> Path:
    """Return the path to the feature-to-category map (``feature_to_category_mapping.csv``).

    The map assigns each feature to one of the seven literature-defined categories and is
    used only to summarise results, never to fit (plan section 5).

    Parameters
    ----------
    root : Path
        The monorepo root.

    Returns
    -------
    Path
        Location of the category-map CSV. Overridable with ``ANALYSIS_CATEGORY_MAP``.
    """
    return _resolve(
        "ANALYSIS_CATEGORY_MAP", root / _ATTACHMENTS_SUBDIR / "feature_to_category_mapping.csv"
    )


def litman_typing_dir(root: Path) -> Path:
    """Return the directory holding the released typing pickles.

    The directory holds ``binary_columns.pkl``, ``categorical_columns.pkl``, and
    ``continuous_columns.pkl``: the feature-type assignments StepMix's densities depend
    on. We reconcile a dictionary-derived typing against these (plan section 6, step 2).

    Parameters
    ----------
    root : Path
        The monorepo root.

    Returns
    -------
    Path
        Location of the typing-pickle directory. Overridable with ``ANALYSIS_TYPING_DIR``.
    """
    return _resolve("ANALYSIS_TYPING_DIR", root / _LITMAN_CODE_SUBDIR / "PhenotypeClasses" / "data")
