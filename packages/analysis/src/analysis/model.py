"""The StepMix general finite mixture model wrapper.

A thin layer over StepMix that builds the mixed-data descriptor from a reconciled typing
and fits the one-step covariate parametrisation Litman et al. use: a Gaussian, Bernoulli,
or multinomial measurement density per feature, with sex and age at evaluation as
structural covariates (plan section 6, step 3). The random restarts are delegated to
StepMix's own ``n_init``, as in the released code; StepMix shows the restart progress and
its iteration log is captured into the run log.

The released ``datadf.round()`` is applied here, immediately before fitting, so the cached
cohort matrix stays unrounded while the model sees the rounded values the authors fit on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from stepmix.stepmix import StepMix
from stepmix.utils import get_mixed_descriptor

from analysis import config
from analysis.cohort import CohortMatrix
from analysis.features import Typing


@dataclass
class FitResult:
    """A fitted GFMM with its labels and selection statistics.

    Attributes
    ----------
    model : StepMix
        The fitted estimator.
    labels : pandas.Series
        The hard class label per proband, indexed by proband id.
    measurement_data : pandas.DataFrame
        The descriptor-aligned measurement matrix the model was fit and predicted on.
    metrics : dict
        Selection statistics: average log-likelihood, AIC, BIC, sample-size-adjusted BIC,
        the number of probands, and the class proportions.
    """

    model: StepMix
    labels: pd.Series
    measurement_data: pd.DataFrame
    metrics: dict[str, object]


def prepare_inputs(
    matrix: CohortMatrix, typing: Typing, round_values: bool = True
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Build the StepMix measurement descriptor and the covariate matrix.

    Parameters
    ----------
    matrix : analysis.cohort.CohortMatrix
        The cohort feature and covariate matrices.
    typing : analysis.features.Typing
        The reconciled feature typing.
    round_values : bool, default True
        Round features and covariates to the nearest integer before fitting, as in the
        released ``GFMM.py``.

    Returns
    -------
    tuple
        The measurement data, the mixed-data descriptor, and the covariate matrix.
    """
    features = matrix.features.round() if round_values else matrix.features.copy()
    covariates = matrix.covariates.round() if round_values else matrix.covariates.copy()
    columns = set(features.columns)
    buckets = {
        "continuous": [c for c in typing.continuous if c in columns],
        "binary": [c for c in typing.binary if c in columns],
        "categorical": [c for c in typing.categorical if c in columns],
    }
    # Pass only the non-empty type buckets. StepMix builds one emission sub-model per bucket
    # it is given, and an empty bucket makes a degenerate sub-model that fails at fit time.
    # The full cohort has all three types, but a cohort reduced to the features shared with
    # another cohort (the replication stage) can drop a whole type, so this guard matters.
    present = {kind: cols for kind, cols in buckets.items() if cols}
    measurement_data, descriptor = get_mixed_descriptor(dataframe=features, **present)
    return measurement_data, descriptor, covariates


def fit_gfmm(
    matrix: CohortMatrix,
    typing: Typing,
    *,
    n_components: int = config.DEFAULT_N_COMPONENTS,
    n_init: int = config.DEFAULT_N_INIT,
    n_steps: int = config.DEFAULT_N_STEPS,
    random_state: int | None = None,
    sample_weight: np.ndarray | pd.Series | None = None,
    progress_bar: int = 1,
    verbose: int = 1,
) -> FitResult:
    """Fit the one-step covariate GFMM and predict a hard label per proband.

    Parameters
    ----------
    matrix : analysis.cohort.CohortMatrix
        The cohort feature and covariate matrices.
    typing : analysis.features.Typing
        The reconciled feature typing that sets each feature's density.
    n_components : int, optional
        Number of latent classes.
    n_init : int, optional
        Number of random restarts, delegated to StepMix.
    n_steps : int, optional
        StepMix estimation steps (one-step joint estimation by default).
    random_state : int or None, optional
        Seed for reproducible restarts.
    sample_weight : numpy.ndarray or pandas.Series or None, optional
        Per-proband weight on the fit, in the row order of the matrix. StepMix's expectation
        maximisation weights each proband's log-likelihood and its class responsibilities by
        this value, so a weight of zero excludes a proband and a weight of one includes it in
        full. This is what a :class:`~analysis.localise.LocalisationScheme` supplies: an
        indicator weight reproduces a hard-bin subset fit, a kernel weight gives a local
        (LSEM) fit. ``None`` leaves every proband at weight one, the pooled fit.
    progress_bar : int, optional
        StepMix progress-bar verbosity for the restart loop.
    verbose : int, optional
        StepMix log verbosity; its output is captured into the run log.

    Returns
    -------
    FitResult
        The fitted model, the predicted labels, the measurement data, and the selection
        statistics.
    """
    measurement_data, descriptor, covariates = prepare_inputs(matrix, typing)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    model = StepMix(
        n_components=n_components,
        measurement=descriptor,
        structural="covariate",
        n_steps=n_steps,
        n_init=n_init,
        random_state=random_state,
        progress_bar=progress_bar,
        verbose=verbose,
    )
    model.fit(measurement_data, covariates, sample_weight=weights)
    labels = pd.Series(model.predict(measurement_data), index=measurement_data.index, name="class")
    metrics = selection_metrics(model, measurement_data, covariates, labels)
    return FitResult(model=model, labels=labels, measurement_data=measurement_data, metrics=metrics)


def selection_metrics(
    model: StepMix, measurement_data: pd.DataFrame, covariates: pd.DataFrame, labels: pd.Series
) -> dict[str, object]:
    """Compute the information criteria and class proportions for a fit.

    Parameters
    ----------
    model : StepMix
        The fitted estimator.
    measurement_data : pandas.DataFrame
        The measurement matrix the model was fit on.
    covariates : pandas.DataFrame
        The covariate matrix.
    labels : pandas.Series
        The predicted hard labels.

    Returns
    -------
    dict
        Average log-likelihood, AIC, BIC, sample-size-adjusted BIC, the proband count, and
        the class proportions sorted by class id.
    """
    counts = labels.value_counts().sort_index()
    total = int(counts.sum())
    class_ids = [int(c) for c in counts.index.tolist()]
    class_counts = {c: int(n) for c, n in zip(class_ids, counts.tolist(), strict=True)}
    class_proportions = {c: round(n / total, 4) for c, n in class_counts.items()}
    return {
        "n_probands": int(len(labels)),
        "n_components": int(model.n_components),
        "avg_log_likelihood": float(model.score(measurement_data, covariates)),
        "aic": float(model.aic(measurement_data, covariates)),
        "bic": float(model.bic(measurement_data, covariates)),
        "sabic": float(model.sabic(measurement_data, covariates)),
        "class_counts": class_counts,
        "class_proportions": class_proportions,
        "smallest_class_proportion": min(class_proportions.values()),
    }


def class_centroids(measurement_data: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Return the per-class feature means (class-by-feature centroid matrix).

    Parameters
    ----------
    measurement_data : pandas.DataFrame
        The measurement matrix.
    labels : pandas.Series
        The predicted hard labels on the same index.

    Returns
    -------
    pandas.DataFrame
        Class-by-feature mean matrix, indexed by class id.
    """
    grouped = measurement_data.groupby(labels.to_numpy())
    centroids = grouped.mean()
    centroids.index = pd.Index(np.asarray(centroids.index, dtype=int), name="class")
    return centroids
