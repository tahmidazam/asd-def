"""Feature typing, reverse-coded items, and the category map.

StepMix fits a Gaussian density to each continuous feature, a Bernoulli to each binary
feature, and a multinomial to each categorical feature, so the typing has to be right. We
derive it three ways and reconcile them: from the data dictionary (via ``dscat``), from the
authors' released typing pickles, and from the observed cardinality in the cohort. The
dictionary rebuild is the primary signal and is required to agree with the pickles; where
they disagree, the run defers to the pickle typing (the reproduction target) and records
the disagreement (plan section 6, step 2).

This module also carries the 24 reverse-coded SCQ social items, which are flipped before
the seven-category summary, and loads the feature-to-category map used only for summaries.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dscat import queries

from analysis import config
from analysis.cohort import open_catalogue

# The 24 reverse-coded SCQ social items, verbatim from the released ``GFMM.py``. For these
# items a higher score is less impairment, so their enriched and depleted directions are
# swapped before the category summary (plan section 6, step 7).
REVERSE_CODED_SCQ: tuple[str, ...] = (
    "q02_conversation",
    "q09_expressions_appropriate",
    "q19_best_friend",
    "q20_talk_friendly",
    "q21_copy_you",
    "q22_point_things",
    "q23_gestures_wanted",
    "q24_nod_head",
    "q25_shake_head",
    "q26_look_directly",
    "q27_smile_back",
    "q28_things_interested",
    "q29_share",
    "q30_join_enjoyment",
    "q31_comfort",
    "q32_help_attention",
    "q33_range_expressions",
    "q34_copy_actions",
    "q35_make_believe",
    "q36_same_age",
    "q37_respond_positively",
    "q38_pay_attention",
    "q39_imaginative_games",
    "q40_cooperatively_games",
)

_PICKLE_FILES: dict[str, str] = {
    "binary": "binary_columns.pkl",
    "categorical": "categorical_columns.pkl",
    "continuous": "continuous_columns.pkl",
}


@dataclass
class Typing:
    """A partition of the feature set into the three StepMix density types.

    Attributes
    ----------
    continuous, binary, categorical : list of str
        The features modelled with a Gaussian, a Bernoulli, and a multinomial density
        respectively.
    """

    continuous: list[str]
    binary: list[str]
    categorical: list[str]

    def as_dict(self) -> dict[str, str]:
        """Return a mapping from each feature to its type."""
        out: dict[str, str] = {}
        for kind in ("continuous", "binary", "categorical"):
            for feature in getattr(self, kind):
                out[feature] = kind
        return out

    @property
    def counts(self) -> dict[str, int]:
        """Return the number of features of each type."""
        return {
            "continuous": len(self.continuous),
            "binary": len(self.binary),
            "categorical": len(self.categorical),
        }


def n_value_levels(value_coding: str | None) -> int:
    """Count the discrete coded levels in a dictionary value coding.

    Parameters
    ----------
    value_coding : str or None
        The value-coding text, where each coded level is a line containing ``=``.

    Returns
    -------
    int
        The number of coded levels.
    """
    if not value_coding:
        return 0
    return sum(1 for line in str(value_coding).splitlines() if "=" in line)


def infer_from_dictionary(field_type: str | None, value_coding: str | None) -> str:
    """Infer a feature type from its dictionary field type and value coding.

    Calculated scores and dropdown age codings are continuous. A radio item with exactly
    two coded levels is binary, otherwise categorical.

    Parameters
    ----------
    field_type : str or None
        The dictionary field type (for example ``"radio"`` or ``"calculated"``).
    value_coding : str or None
        The value-coding text.

    Returns
    -------
    str
        ``"continuous"``, ``"binary"``, or ``"categorical"``.
    """
    ft = (field_type or "").strip().lower()
    if ft in ("calculated", "dropdown"):
        return "continuous"
    if ft == "radio":
        return "binary" if n_value_levels(value_coding) == 2 else "categorical"
    return "categorical"


def load_pickle_typing(typing_dir: Path, features: list[str]) -> dict[str, str | None]:
    """Load the released typing for a set of features.

    Parameters
    ----------
    typing_dir : Path
        Directory holding the three typing pickles.
    features : list of str
        Features to type.

    Returns
    -------
    dict
        Each feature mapped to its pickle type (``str``), or ``None`` when it is absent
        from all pickles or appears in more than one (ambiguous).
    """
    sets: dict[str, set[str]] = {}
    for kind, filename in _PICKLE_FILES.items():
        with (typing_dir / filename).open("rb") as f:
            sets[kind] = set(pickle.load(f))
    typing: dict[str, str | None] = {}
    for feature in features:
        hits = [kind for kind, members in sets.items() if feature in members]
        typing[feature] = hits[0] if len(hits) == 1 else None
    return typing


def observed_cardinality(frame: pd.DataFrame, features: list[str]) -> dict[str, int]:
    """Return the number of distinct non-null values of each feature in the cohort.

    Parameters
    ----------
    frame : pandas.DataFrame
        The cohort frame.
    features : list of str
        Features to count.

    Returns
    -------
    dict of str to int
        Each feature mapped to its distinct-value count.
    """
    return {f: int(frame[f].nunique(dropna=True)) for f in features if f in frame.columns}


def dictionary_typing(
    root: Path, dataset: str, version: str, features: list[str]
) -> dict[str, str]:
    """Infer each feature's type from the data dictionary.

    Parameters
    ----------
    root : Path
        Repository root.
    dataset, version : str
        Dataset and version whose dictionary to read.
    features : list of str
        Features to type.

    Returns
    -------
    dict of str to str
        Each feature mapped to its dictionary-inferred type.
    """
    cat = open_catalogue(root)
    coding: dict[str, tuple[str, str]] = {}
    for instrument in config.COHORT_INSTRUMENTS:
        _, rows, _ = queries.describe(cat, instrument, dataset, version, limit=100000, offset=0)
        for row in rows:
            coding.setdefault(row["name"], (row["field_type"], row["value_coding"]))
    typing: dict[str, str] = {}
    for feature in features:
        field_type, value_coding = coding.get(feature, ("", ""))
        typing[feature] = infer_from_dictionary(field_type, value_coding)
    return typing


def reconcile(
    features: list[str],
    dict_typing: dict[str, str],
    pickle_typing: dict[str, str | None],
    observed: dict[str, int],
) -> tuple[Typing, pd.DataFrame]:
    """Reconcile the three typing signals into one typing and a report.

    The chosen type defers to the pickle typing where it exists (the reproduction target),
    otherwise to the dictionary. The report records all three signals, whether the
    dictionary and pickle agree, and whether the observed cardinality is consistent with
    the chosen type.

    Parameters
    ----------
    features : list of str
        Features to reconcile.
    dict_typing : dict of str to str
        Dictionary-inferred types.
    pickle_typing : dict
        Released pickle types, each ``str`` or ``None`` when absent or ambiguous.
    observed : dict of str to int
        Observed distinct-value counts.

    Returns
    -------
    tuple
        The reconciled typing and the per-feature reconciliation report.
    """
    records: list[dict[str, object]] = []
    chosen: dict[str, str] = {}
    for feature in features:
        d = dict_typing.get(feature)
        p = pickle_typing.get(feature)
        n_unique = observed.get(feature)
        resolved = p if p is not None else d
        if resolved is None:
            raise ValueError(f"no typing signal for feature {feature!r}")
        chosen[feature] = resolved
        observed_binary = None if n_unique is None else (n_unique == 2)
        observed_consistent = (
            None if observed_binary is None else (observed_binary == (resolved == "binary"))
        )
        records.append(
            {
                "feature": feature,
                "dictionary": d,
                "pickle": p,
                "chosen": resolved,
                "n_unique": n_unique,
                "dictionary_pickle_agree": (d == p) if p is not None else None,
                "observed_consistent": observed_consistent,
            }
        )
    typing = Typing(
        continuous=[f for f in features if chosen[f] == "continuous"],
        binary=[f for f in features if chosen[f] == "binary"],
        categorical=[f for f in features if chosen[f] == "categorical"],
    )
    return typing, pd.DataFrame.from_records(records)


def build_typing(
    root: Path,
    dataset: str,
    version: str,
    features: list[str],
    frame: pd.DataFrame | None = None,
) -> tuple[Typing, pd.DataFrame]:
    """Build the reconciled feature typing and its report.

    Parameters
    ----------
    root : Path
        Repository root.
    dataset, version : str
        Dataset and version whose dictionary to read.
    features : list of str
        The feature set to type.
    frame : pandas.DataFrame, optional
        The cohort frame, used for the observed-cardinality signal. When omitted, that
        signal is skipped.

    Returns
    -------
    tuple
        The reconciled typing and the reconciliation report.
    """
    dict_typing = dictionary_typing(root, dataset, version, features)
    pickle_typing = load_pickle_typing(config.litman_typing_dir(root), features)
    observed = observed_cardinality(frame, features) if frame is not None else {}
    return reconcile(features, dict_typing, pickle_typing, observed)


def load_category_map(path: Path) -> dict[str, str]:
    """Load the feature-to-category map.

    Parameters
    ----------
    path : Path
        Location of ``feature_to_category_mapping.csv``, with ``category`` and ``feature``
        columns.

    Returns
    -------
    dict of str to str
        Each feature mapped to its category.
    """
    df = pd.read_csv(path)
    return dict(zip(df["feature"], df["category"], strict=False))
