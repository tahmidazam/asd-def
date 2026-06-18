"""Aligning our recovered classes to Litman's four named classes.

Two of the three alignment routes the authors use are closed to us: we lack SPARK v9 and
their per-proband labels, and no reference model was released. We therefore align on the
named-class anchors: the substantive, data-driven characteristics the authors use to define
each class (the most-developmental class is Mixed ASD with DD; the highest-difficulty and
smallest class is Broadly affected; the largest, high-core, no developmental delay class is
Social/behavioral; the uniformly lowest is Moderate challenges). The assignment is
cross-validated for mutual consistency.

The published seven-category signatures (read from figure 1b) give a profile correlation
against each named class and an overall correlation, the analogue of the authors' own
replication measure (their SSC replication reported :math:`r = 0.927`). Those values are read
to the figure's resolution, not from a supplementary table (plan section 6a, step 1); the
values themselves are tabulated in the reproduction guide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.align import hungarian_align
from analysis.enrich import SEVEN_CATEGORIES

# Published class proportions, by name. The largest class is Social/behavioral and the
# smallest is Broadly affected (plan section 6a); the 19 and 34 per cent classes are read
# as Mixed ASD with DD and Moderate challenges respectively.
PUBLISHED_PROPORTIONS: dict[str, float] = {
    "Social/behavioral": 0.37,
    "Moderate challenges": 0.34,
    "Mixed ASD with DD": 0.19,
    "Broadly affected": 0.10,
}

# Published seven-category signatures (the "proportion and direction" per category), read
# from figure 1b of Litman et al. (2025) and reordered into ``SEVEN_CATEGORIES`` order (the
# figure lists restricted/repetitive before social/communication). These are estimated to
# the figure's resolution, not exact supplementary-table values, so the profile correlation
# is read against the figure (plan section 6a, step 1). Broadly affected is saturated near
# +1 across all categories in the figure.
_PUBLISHED_SIGNATURE: dict[str, list[float]] = {
    # anxiety/mood, attention, disruptive, self-injury, social/comm, restricted/rep, developmental
    "Broadly affected": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "Mixed ASD with DD": [-0.90, -0.45, -0.65, -0.10, 0.10, 0.05, 0.45],
    "Social/behavioral": [1.0, 1.0, 0.95, 0.50, 0.50, 0.45, -0.90],
    "Moderate challenges": [-0.90, -0.95, -1.0, -0.95, -1.0, -1.0, -1.0],
}

NAMED_CLASSES: tuple[str, ...] = tuple(_PUBLISHED_SIGNATURE)


@dataclass
class NamedAlignment:
    """The alignment of our classes to the named classes and its validation.

    Attributes
    ----------
    mapping : dict
        Each of our class ids mapped to a named class.
    correlations : dict
        Per-class Pearson correlation of our signature with the assigned named published
        signature. A class is ``None`` when its published profile is saturated (constant),
        which makes the correlation undefined.
    overall_correlation : float
        Pearson correlation over the full class-by-category matrix (our classes aligned to
        the named published classes), the analogue of the authors' replication measure.
    anchors : dict
        Each named anchor mapped to whether it held for the assigned class.
    anchors_hold : bool
        Whether every anchor held.
    cost : pandas.DataFrame
        The Hungarian cost matrix (our classes by named classes).
    """

    mapping: dict[object, str]
    correlations: dict[object, float | None]
    overall_correlation: float
    anchors: dict[str, bool]
    anchors_hold: bool
    cost: pd.DataFrame


def published_signature() -> pd.DataFrame:
    """Return the published seven-category signature (figure 1b), classes by category."""
    frame = pd.DataFrame.from_dict(
        _PUBLISHED_SIGNATURE, orient="index", columns=list(SEVEN_CATEGORIES)
    )
    frame.index.name = "named_class"
    return frame


def _assign_by_anchors(signature: pd.DataFrame, proportions: dict[int, float]) -> dict[int, str]:
    """Assign each class to a named class by its defining data-driven anchor.

    Broadly affected is the highest-difficulty class overall; Mixed ASD with developmental
    delay is the most developmental of the rest; Social/behavioral is the largest of the
    rest; Moderate challenges is the remaining class.

    Parameters
    ----------
    signature : pandas.DataFrame
        Our class-by-category signature, indexed by class id.
    proportions : dict of int to float
        Our class proportions by class id.

    Returns
    -------
    dict of int to str
        Each class id mapped to its named class.
    """
    overall = signature.mean(axis=1)
    remaining = list(signature.index)
    mapping: dict[int, str] = {}

    broadly = int(signature.index[overall.to_numpy().argmax()])
    mapping[broadly] = "Broadly affected"
    remaining.remove(broadly)

    dev = signature.loc[remaining, "developmental"]
    mixed = int(dev.index[dev.to_numpy().argmax()])
    mapping[mixed] = "Mixed ASD with DD"
    remaining.remove(mixed)

    social = max(remaining, key=lambda c: proportions[int(c)])
    mapping[int(social)] = "Social/behavioral"
    remaining.remove(social)

    mapping[int(remaining[0])] = "Moderate challenges"
    return mapping


def _validate_anchors(
    signature: pd.DataFrame, proportions: dict[int, float], mapping: dict[int, str]
) -> dict[str, bool]:
    """Cross-check that the anchor assignment is mutually consistent.

    Each check is an independent named-class characteristic, so all holding is evidence the
    four recovered classes line up with the published four (plan section 6a, step 5).

    Parameters
    ----------
    signature : pandas.DataFrame
        Our class-by-category signature, indexed by class id.
    proportions : dict of int to float
        Our class proportions by class id.
    mapping : dict of int to str
        The anchor assignment.

    Returns
    -------
    dict of str to bool
        Each check mapped to whether it held.
    """
    inverse = {name: cid for cid, name in mapping.items()}
    overall = signature.mean(axis=1)
    highest_overall = int(signature.index[overall.to_numpy().argmax()])
    lowest_overall = int(signature.index[overall.to_numpy().argmin()])
    smallest = min(proportions, key=lambda c: proportions[c])
    largest = max(proportions, key=lambda c: proportions[c])
    return {
        "broadly_affected_highest_overall": inverse["Broadly affected"] == highest_overall,
        "broadly_affected_smallest": inverse["Broadly affected"] == smallest,
        "social_behavioral_largest": inverse["Social/behavioral"] == largest,
        "moderate_challenges_lowest_overall": inverse["Moderate challenges"] == lowest_overall,
    }


def _safe_correlation(a: np.ndarray, b: np.ndarray, min_std: float = 0.05) -> float | None:
    """Return the Pearson correlation, or ``None`` when either profile is near-constant.

    A class whose seven-category profile is saturated (uniformly high or uniformly low) has
    almost no variance, so its correlation is dominated by rounding noise and is not
    meaningful; the named-class anchors confirm such classes instead.
    """
    if float(np.std(a)) < min_std or float(np.std(b)) < min_std:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        r = float(np.corrcoef(a, b)[0, 1])
    return None if np.isnan(r) else r


def align_to_named(signature: pd.DataFrame, proportions: dict[int, float]) -> NamedAlignment:
    """Align our recovered classes to the named classes and validate the mapping.

    Parameters
    ----------
    signature : pandas.DataFrame
        Our class-by-category signature, indexed by class id.
    proportions : dict of int to float
        Our class proportions by class id.

    Returns
    -------
    NamedAlignment
        The anchor mapping, the per-class and overall profile correlations against the
        published signature, the anchor consistency checks, and the cost matrix from the
        (secondary) Hungarian route.
    """
    mapping = _assign_by_anchors(signature, proportions)
    anchors = _validate_anchors(signature, proportions, mapping)

    target = published_signature()
    secondary = hungarian_align(signature, target, metric="correlation")

    correlations: dict[object, float | None] = {}
    ours_stacked: list[float] = []
    published_stacked: list[float] = []
    for cid, name in mapping.items():
        ours = signature.loc[cid].to_numpy(float)
        pub = target.loc[name].to_numpy(float)
        ours_stacked.extend(ours.tolist())
        published_stacked.extend(pub.tolist())
        correlations[int(cid)] = _safe_correlation(ours, pub)

    overall = _safe_correlation(np.array(ours_stacked), np.array(published_stacked)) or float("nan")

    return NamedAlignment(
        mapping={int(cid): name for cid, name in mapping.items()},
        correlations=correlations,
        overall_correlation=overall,
        anchors=anchors,
        anchors_hold=all(anchors.values()),
        cost=secondary.cost,
    )
