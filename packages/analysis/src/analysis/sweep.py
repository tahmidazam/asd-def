r"""The sweep: run one or more localisation schemes end to end and read them together.

The stratified analysis re-estimates the mixture model along an axis and measures how far the
reference classes move (plan section 7). :mod:`analysis.strata` and :mod:`analysis.localise`
give two ways to localise the re-estimation: hard bins (the frozen primary) and kernel windows
(the local-likelihood, LSEM, trajectory). This module runs any such
:class:`~analysis.localise.LocalisationScheme` through the same machinery, so a single run
fits the observed sweep, fits the matched permutation null, aligns each local fit to the
reference, measures its drift, and reads it against the null. Because every scheme reduces to
the method-independent :class:`~analysis.drift.StratumSummary`, the hard-bin and kernel arms
are read against one reference with one distance and reported side by side.

The heavy work (the fits) is separated from the cheap work (alignment and distance) exactly as
:mod:`analysis.drift` does: :func:`fit_local_summaries` returns the stored summaries, and
:func:`sweep_decision` measures over them, so a different alignment or distance re-measures with
no refit.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analysis import drift as drift_mod
from analysis.cohort import CohortMatrix
from analysis.drift import (
    AlignmentMethod,
    DistanceMethod,
    ReferenceModel,
    StratumSummary,
    compute_drift,
    summarise,
)
from analysis.features import Typing
from analysis.localise import (
    HardBins,
    KernelWindows,
    LocalFit,
    LocalisationScheme,
    fit_locale,
    permute_axis,
)
from analysis.strata import MaxEqualBins, QuantileBins


@dataclass
class LocaleSummary:
    """One local fit's position on the axis and its method-independent summary.

    Attributes
    ----------
    label : str
        The local fit's name (a stratum name, or ``focal=6.5``).
    position : float
        The local fit's location on the axis, the abscissa for the trajectory.
    summary : analysis.drift.StratumSummary
        The centroids, dispersions, and reference contingency the drift is measured from.
    """

    label: str
    position: float
    summary: StratumSummary


def summarise_locale(
    matrix: CohortMatrix,
    typing: Typing,
    locale: LocalFit,
    reference_labels: pd.Series,
    *,
    n_init: int,
    seed: int,
) -> StratumSummary:
    """Fit one local fit and return its method-independent summary.

    A hard bin (indicator weights that are one on its members) is summarised unweighted, so its
    summary is byte-identical to the current stratified fit's. A kernel window (fractional
    weights) is summarised with those weights, so its centroids are the local weighted means.
    """
    fit = fit_locale(matrix, typing, locale, n_init=n_init, random_state=seed)
    retained = locale.weights.reindex(fit.measurement_data.index).fillna(0.0)
    weights = None if bool((retained == 1.0).all()) else retained
    return summarise(fit.measurement_data, fit.labels, reference_labels, weights=weights)


def summarise_local_worker(
    features: pd.DataFrame,
    covariates: pd.DataFrame,
    typing: Typing,
    dataset: str,
    version: str,
    weights: pd.Series | None,
    reference_labels: pd.Series,
    n_init: int,
    seed: int,
) -> StratumSummary | None:
    """Fit one local fit from its already-subset rows and return its summary (picklable).

    A top-level function so it pickles for a process pool (mirrors
    :func:`analysis.drift.summarise_pseudo_stratum`). The caller drops the negligible-weight
    rows before submitting, so ``features`` holds only the retained probands and ``weights`` (or
    ``None`` for a hard bin, where every retained weight is one) is aligned to them. Returns
    ``None`` if the fit is degenerate (a singular covariate GLM), so the caller drops that local
    fit rather than letting one bad refit abort the sweep.
    """
    from analysis.model import fit_gfmm

    matrix = CohortMatrix(features, covariates, dataset, version)
    try:
        fit = fit_gfmm(
            matrix,
            typing,
            n_init=n_init,
            random_state=seed,
            sample_weight=None if weights is None else weights.to_numpy(dtype=float),
            progress_bar=0,
            verbose=0,
        )
    except drift_mod.DEGENERATE_FIT_ERRORS:
        return None
    return summarise(fit.measurement_data, fit.labels, reference_labels, weights=weights)


def fit_local_summaries(
    matrix: CohortMatrix,
    typing: Typing,
    axis_values: pd.Series,
    scheme: LocalisationScheme,
    reference_labels: pd.Series,
    *,
    n_init: int,
    seed: int,
) -> list[LocaleSummary]:
    """Fit the observed sweep: one summary per local fit, in axis order.

    Serial by design; the CLI spreads the fits across a process pool. Kept pure so the sweep
    logic is testable without the run machinery.
    """
    summaries: list[LocaleSummary] = []
    for offset, locale in enumerate(scheme.locales(axis_values)):
        summary = summarise_locale(
            matrix, typing, locale, reference_labels, n_init=n_init, seed=seed + offset
        )
        summaries.append(LocaleSummary(locale.label, locale.position, summary))
    return summaries


def fit_null_summaries(
    matrix: CohortMatrix,
    typing: Typing,
    axis_values: pd.Series,
    scheme: LocalisationScheme,
    reference_labels: pd.Series,
    *,
    n_init: int,
    n_permutations: int,
    seed: int,
) -> list[tuple[int, StratumSummary]]:
    """Fit the permutation null: for each permutation, shuffle the axis and re-run the scheme.

    Returns ``(locale_index, summary)`` pairs across all permutations, so the caller reads each
    observed local fit against the null draws at its own position. Serial; the CLI parallelises.
    """
    draws: list[tuple[int, StratumSummary]] = []
    for perm in range(n_permutations):
        permuted = permute_axis(axis_values, seed=perm)
        for offset, locale in enumerate(scheme.locales(permuted)):
            summary = summarise_locale(
                matrix,
                typing,
                locale,
                reference_labels,
                n_init=n_init,
                seed=seed + perm * 1000 + offset,
            )
            draws.append((offset, summary))
    return draws


def sweep_decision(
    scheme_name: str,
    observed: list[LocaleSummary],
    null: list[tuple[int, StratumSummary]],
    reference: ReferenceModel,
    aligner: AlignmentMethod,
    distancer: DistanceMethod,
) -> pd.DataFrame:
    """Align, measure, and read each observed local fit against the null, one row per class.

    Mirrors the ``drift`` stage's decision table so the hard-bin and kernel arms share a
    format: per local fit and reference class, the observed drift, its null 95th percentile and
    permutation p-value, the drift as a fraction of the between-class separation, the alignment
    confidence, and the Benjamini-Hochberg and reorganisation flags. The ``position`` and
    ``scheme`` columns let the arms be plotted as trajectories and compared.
    """
    from collections import defaultdict

    null_drift: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for locale_index, summary in null:
        result = compute_drift(summary, reference, aligner, distancer)
        for ref_class, value in result.distances.items():
            null_drift[locale_index][int(ref_class)].append(value)

    separation = drift_mod.class_separation(reference, distancer)
    rows: list[dict] = []
    for locale_index, locale in enumerate(observed):
        result = compute_drift(locale.summary, reference, aligner, distancer)
        for ref_class, obs in result.distances.items():
            read = drift_mod.read_against_null(obs, null_drift[locale_index][int(ref_class)])
            rows.append(
                {
                    **read,
                    "scheme": scheme_name,
                    "stratum": locale.label,
                    "position": locale.position,
                    "ref_class": int(ref_class),
                    "jaccard": float(result.alignment.quality.get(int(ref_class), float("nan"))),
                    "ari": float(result.alignment.overall),
                    "drift_vs_separation": obs / separation if separation else float("nan"),
                }
            )
    decision = pd.DataFrame(rows)
    decision["fdr_significant"] = drift_mod.benjamini_hochberg(
        decision["p_value"].to_numpy(), q=0.05
    )
    decision["reorganised"] = decision["jaccard"] < 0.5
    return decision


def parse_scheme(spec: str) -> LocalisationScheme:
    """Build a localisation scheme from a colon-separated command-line specification.

    Grammar (case-insensitive scheme name):

    - ``hardbins:max-equal:<floor>`` gives ``HardBins(MaxEqualBins(floor))``, the frozen primary.
    - ``hardbins:quantile:<q>`` gives ``HardBins(QuantileBins(q))``, the coarse sensitivity scheme.
    - ``kernel:<bandwidth>:<n_points>`` gives ``KernelWindows(bandwidth, grid=n_points)``, the
      local-likelihood trajectory.

    Parameters
    ----------
    spec : str
        The scheme specification.

    Returns
    -------
    LocalisationScheme
        The parsed scheme.

    Raises
    ------
    ValueError
        If the scheme name or its arguments are not recognised.
    """
    parts = spec.split(":")
    kind = parts[0].lower()
    if kind == "hardbins":
        if len(parts) != 3:
            raise ValueError(f"hardbins scheme needs 'hardbins:<policy>:<arg>', got {spec!r}")
        policy_name, arg = parts[1].lower(), parts[2]
        if policy_name == "max-equal":
            return HardBins(MaxEqualBins(min_bin_size=int(arg)))
        if policy_name == "quantile":
            return HardBins(QuantileBins(q=int(arg)))
        raise ValueError(f"unknown hardbins policy {policy_name!r}; use 'max-equal' or 'quantile'")
    if kind == "kernel":
        if len(parts) != 3:
            raise ValueError(f"kernel scheme needs 'kernel:<bandwidth>:<n_points>', got {spec!r}")
        return KernelWindows(bandwidth=float(parts[1]), grid=int(parts[2]))
    raise ValueError(f"unknown scheme {kind!r}; use 'hardbins' or 'kernel'")
