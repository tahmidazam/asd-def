"""The shared feature schema and the cross-cohort harmonisation maps.

The feature set is the authors' final 238-feature list. The CBCL competence items arrive
as strings in SPARK and are recoded to the ordinal integers the released preprocessing
used. The SSC rename maps carry the second cohort's column names onto the SPARK names, so
both cohorts present the same schema to the rest of the pipeline (plan section 10).
"""

from __future__ import annotations

import csv
from pathlib import Path

# Recoding of the CBCL competence items from their string answers to ordinal integers,
# verbatim from the released ``process_integrate_phenotype_data.py``. Applied across the
# CBCL sub-frame; the numeric t-score and item columns are unaffected because their values
# are not among these string keys.
CBCL_REPLACEMENTS: dict[str, int] = {
    "above_average": 0,
    "average": 1,
    "below_average": 2,
    "failing": 3,
    # close_friends
    "none": 3,
    "1": 2,
    "2_3": 1,
    "4_more": 0,
    # contact friends outside school
    "less_1": 2,
    "1_2": 1,
    "3_more": 0,
    # gets-along / behaviour items
    "worse": 2,
    "better": 0,
    "has no brothers or sisters": 1,
}

# How the parent-reported sex string is encoded, as in the released SCQ preprocessing.
SEX_ENCODING: dict[str, int] = {"Male": 1, "Female": 0}

# SSC-to-SPARK column renames per instrument, verbatim from ``generate_ssc_data``. These
# define the harmonisation onto the shared schema for the second cohort (plan section 10).
SSC_CBCL_RENAME: dict[str, str] = {
    "add_adhd_t_score": "dsm5_attention_deficit_hyperactivity_t_score",
    "affective_problems_t_score": "dsm5_depressive_problems_t_score",
    "anxiety_problems_t_score": "dsm5_anxiety_problems_t_score",
    "conduct_problems_t_score": "dsm5_conduct_problems_t_score",
    "oppositional_defiant_t_score": "dsm5_oppositional_defiant_t_score",
    "rule_breaking_t_score": "rule_breaking_behavior_t_score",
    "withdrawn_t_score": "withdrawn_depressed_t_score",
    "somatic_prob_t_score": "dsm5_somatic_problems_t_score",
}

SSC_SCQ_RENAME: dict[str, str] = {
    "q08_hits_self_object": "q08_hits_self_against_object",
    "q09_hits_self_object": "q09_hits_self_with_object",
    "q28_communicatiion": "q28_communication",
    "summary_score": "final_score",
}

SSC_RBSR_RENAME: dict[str, str] = {"q39_insists_palce": "q39_insists_time"}

# SSC raw background-history milestone columns mapped onto the SPARK names. Nine of the 11
# SPARK background-history features map cleanly; ``combined_phrases_age_mos`` and
# ``repeat_grade`` have no clean SSC equivalent, so the SSC backend provides a subset.
# Litman et al. used a hand-cleaned background-history file that was not released, so this
# mapping is ours and its fidelity is confirmed in the SSC replication stage (phase 2).
SSC_BH_RENAME: dict[str, str] = {
    "age_smiled": "smiled_age_mos",
    "age_sat_wo_support": "sat_wo_support_age_mos",
    "age_crawled": "crawled_age_mos",
    "age_walked_alone": "walked_age_mos",
    "age_fed_self_w_spoon": "fed_self_spoon_age_mos",
    "age_used_words": "used_words_age_mos",
    "age_combined_words_short_sen": "combined_words_age_mos",
    "age_bladder_trained_day": "bladder_trained_age_mos",
    "age_bowel_trained": "bowel_trained_age_mos",
}

# How the SSC sex string and the SCQ yes/no answers are encoded onto the SPARK coding.
SSC_SEX_ENCODING: dict[str, int] = {"male": 1, "female": 0}
SSC_YES_NO: dict[str, int] = {"yes": 1, "no": 0}


def load_feature_list(path: Path) -> list[str]:
    """Read the authors' feature list from its one-column CSV.

    Parameters
    ----------
    path : Path
        Location of ``mixture_model_columns.csv``, which has a ``feature`` header.

    Returns
    -------
    list of str
        The feature names, in file order.
    """
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # discard the header
        return [row[0].strip() for row in reader if row and row[0].strip()]
