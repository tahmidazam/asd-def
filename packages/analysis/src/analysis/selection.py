"""Model selection: the number-of-components grid and its information criteria.

Reproduces the released ``GFMM_model_validation`` procedure (plan section 6, step 4). For
each candidate number of classes, the one-step covariate GFMM is fitted and scored on a
panel of criteria, repeated over several seeds and summarised as a mean and a standard
deviation. The criteria are the validation log-likelihood from 3-fold cross-validation,
the Akaike, Bayesian, sample-size-adjusted Bayesian, and consistent Akaike information
criteria, the approximate weight of evidence, the scaled relative entropy, the average
latent-class posterior probability (ALCPP), and the smallest class proportion.

Two faithfulness notes. StepMix 3.0.0 exposes ``aic``, ``bic``, ``sabic``, and ``caic`` as
methods whose formulas match the authors' hand-rolled helpers, so those are used directly;
the approximate weight of evidence is not a StepMix method and is computed here. The
"Lo-Mendell-Rubin likelihood-ratio test" in the released code is a naive chi-square on the
cross-validated log-likelihood differences with the degrees of freedom fixed at one, not
the analytically correct adjusted test; :func:`lmr_lrt_proxy` reproduces that approximation
and is documented as such. Litman et al. do not seed the per-iteration fits; we seed them
for reproducibility (plan section 11), which is the only deliberate divergence here.

The released decision for four components is asserted visually (a reference line at four on
every panel) rather than by an automatic rule, so this module reports the full criteria
table and leaves the choice to the methods write-up.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.stats import chi2
from sklearn.model_selection import GridSearchCV
from stepmix.stepmix import StepMix

from analysis import config, profiling
from analysis.checkpoint import SUFFIX, CheckpointLog
from analysis.cohort import CohortMatrix
from analysis.features import Typing
from analysis.model import prepare_inputs
from analysis.progress import task_bar

# Criteria computed per fit, in reporting order. ``val_log_likelihood`` is added from the
# cross-validation pass and ``lmr_lrt_p`` is derived from it, so neither is fitted here.
PER_FIT_CRITERIA: tuple[str, ...] = (
    "avg_log_likelihood",
    "aic",
    "bic",
    "sabic",
    "caic",
    "awe",
    "relative_entropy",
    "alcpp",
    "smallest_class_proportion",
)


@dataclass
class SelectionResult:
    """The model-selection grid and its summary.

    Attributes
    ----------
    per_iteration : pandas.DataFrame
        One row per (seed, number of components) with every criterion.
    summary : pandas.DataFrame
        Mean and standard deviation of each criterion per number of components, plus the
        mean Lo-Mendell-Rubin proxy :math:`p`-value.
    k_values : list of int
        The component counts gridded.
    """

    per_iteration: pd.DataFrame
    summary: pd.DataFrame
    k_values: list[int]


def awe(model: StepMix, measurement: pd.DataFrame, covariates: pd.DataFrame) -> float:
    r"""Return the approximate weight of evidence for a fitted model.

    The penalty sits between the consistent Akaike criterion and a heavier term:
    :math:`-2 \ell + k(\ln n + 1.5)`, where :math:`\ell` is the total log-likelihood,
    :math:`k` the number of free parameters, and :math:`n` the sample size. StepMix does
    not expose this criterion, so it is computed from ``score`` (the average log-likelihood)
    and ``n_parameters``, as in the authors' ``utils.awe``.

    Parameters
    ----------
    model : StepMix
        A fitted estimator.
    measurement : pandas.DataFrame
        The measurement matrix the model was scored on.
    covariates : pandas.DataFrame
        The structural covariate matrix.

    Returns
    -------
    float
        The approximate weight of evidence (lower is better).
    """
    n = measurement.shape[0]
    total_log_likelihood = model.score(measurement, covariates) * n
    return float(-2.0 * total_log_likelihood + model.n_parameters * (np.log(n) + 1.5))


def per_fit_criteria(
    model: StepMix, measurement: pd.DataFrame, covariates: pd.DataFrame
) -> dict[str, float]:
    """Compute the per-fit selection criteria for one fitted model.

    The information criteria and the average log-likelihood are scored with the covariate
    channel (as in the released validation code); the hard labels and posteriors used for
    the ALCPP and the smallest-class proportion come from the measurement posterior without
    covariates, matching how the reference labels are assigned (plan section 6, step 3).

    Parameters
    ----------
    model : StepMix
        A fitted estimator.
    measurement : pandas.DataFrame
        The measurement matrix.
    covariates : pandas.DataFrame
        The structural covariate matrix.

    Returns
    -------
    dict of str to float
        One value per entry in ``PER_FIT_CRITERIA``.
    """
    proba = model.predict_proba(measurement)
    labels = proba.argmax(axis=1)
    counts = np.bincount(labels, minlength=model.n_components)
    smallest_proportion = float(counts.min() / counts.sum())
    return {
        "avg_log_likelihood": float(model.score(measurement, covariates)),
        "aic": float(model.aic(measurement, covariates)),
        "bic": float(model.bic(measurement, covariates)),
        "sabic": float(model.sabic(measurement, covariates)),
        "caic": float(model.caic(measurement, covariates)),
        "awe": awe(model, measurement, covariates),
        "relative_entropy": float(model.relative_entropy(measurement, covariates)),
        "alcpp": float(np.mean(proba.max(axis=1))),
        "smallest_class_proportion": smallest_proportion,
    }


def _fit_model(
    measurement: pd.DataFrame,
    covariates: pd.DataFrame,
    descriptor: dict,
    *,
    n_components: int,
    n_init: int,
    seed: int,
) -> StepMix:
    """Fit one StepMix GFMM for the selection grid.

    Raises :class:`numpy.linalg.LinAlgError` when the structural covariate M-step fails to
    converge; :func:`run_selection` catches that and records the fit as missing.
    """
    model = StepMix(
        n_components=n_components,
        measurement=descriptor,
        structural="covariate",
        n_steps=config.DEFAULT_N_STEPS,
        n_init=n_init,
        random_state=seed,
        progress_bar=0,
        verbose=0,
    )
    model.fit(measurement, covariates)
    return model


def validation_log_likelihood(
    measurement: pd.DataFrame,
    covariates: pd.DataFrame,
    descriptor: dict,
    k_values: Sequence[int],
    *,
    seed: int,
    n_init: int,
    cv: int = 3,
) -> dict[int, float]:
    """Cross-validated mean log-likelihood per number of components.

    Uses scikit-learn's grid search over ``n_components`` with the StepMix average
    log-likelihood as the score, as in the released ``compute_LL`` (plan section 6, step 4).

    Parameters
    ----------
    measurement : pandas.DataFrame
        The measurement matrix.
    covariates : pandas.DataFrame
        The structural covariate matrix, passed as the supervised target so it enters the
        cross-validated score.
    descriptor : dict
        The StepMix mixed-data measurement descriptor.
    k_values : sequence of int
        The component counts to grid over.
    seed : int
        Random seed for the estimator.
    n_init : int
        Random restarts per fold fit.
    cv : int, default 3
        Number of cross-validation folds.

    Returns
    -------
    dict of int to float
        Each number of components mapped to its mean validation log-likelihood.
    """
    base = StepMix(
        measurement=descriptor,
        structural="covariate",
        n_steps=config.DEFAULT_N_STEPS,
        n_init=n_init,
        random_state=seed,
        progress_bar=0,
        verbose=0,
    )
    search = GridSearchCV(
        estimator=base, cv=cv, param_grid={"n_components": list(k_values)}, refit=False
    )
    search.fit(measurement, covariates)
    results = search.cv_results_
    return {
        int(k): float(score)
        for k, score in zip(results["param_n_components"], results["mean_test_score"], strict=True)
    }


def lmr_lrt_proxy(val_log_likelihood: dict[int, float]) -> dict[int, float]:
    r"""Naive likelihood-ratio-test :math:`p`-values between adjacent component counts.

    For consecutive counts :math:`k` and :math:`k+1` the statistic is
    :math:`-2(\ell_k - \ell_{k+1})` referred to a chi-square with one degree of freedom,
    keyed at :math:`k+1`. The log-likelihoods are the cross-validated per-sample means. This
    is the approximation the released code computes; it is not the analytically correct
    Lo-Mendell-Rubin adjusted test (the degrees of freedom are fixed at one rather than the
    difference in free parameters), and it is reported as a proxy only.

    Parameters
    ----------
    val_log_likelihood : dict of int to float
        Validation log-likelihood per number of components.

    Returns
    -------
    dict of int to float
        Each number of components (from the second upward) mapped to its proxy :math:`p`-value.
    """
    ks = sorted(val_log_likelihood)
    pvalues: dict[int, float] = {}
    for lower, upper in zip(ks, ks[1:], strict=False):
        statistic = -2.0 * (val_log_likelihood[lower] - val_log_likelihood[upper])
        pvalues[upper] = float(chi2.sf(statistic, df=1))
    return pvalues


def _run_iteration(
    measurement: pd.DataFrame,
    covariates: pd.DataFrame,
    descriptor: dict,
    k_list: list[int],
    *,
    iteration: int,
    seed: int,
    n_init: int,
    cv: int,
) -> list[dict[str, float]]:
    """Run one seeded grid iteration: the cross-validation pass, then a fit per K.

    A top-level function so it pickles for a process pool. Different iterations share no
    state, so this is the unit :func:`run_selection` parallelises and checkpoints.
    """
    val_ll = validation_log_likelihood(
        measurement, covariates, descriptor, k_list, seed=seed, n_init=n_init, cv=cv
    )
    lmr = lmr_lrt_proxy(val_ll)
    rows: list[dict[str, float]] = []
    for k in k_list:
        try:
            model = _fit_model(
                measurement, covariates, descriptor, n_components=k, n_init=n_init, seed=seed
            )
            criteria = per_fit_criteria(model, measurement, covariates)
        except LinAlgError:
            # The structural M-step did not converge (a near-singular covariate design, most
            # often at higher K); record the fit as missing and carry on, as the
            # cross-validation pass already does through its nan error score.
            criteria = dict.fromkeys(PER_FIT_CRITERIA, float("nan"))
        row: dict[str, float] = {"iteration": iteration, "n_components": k}
        row.update(criteria)
        row["val_log_likelihood"] = val_ll[k]
        row["lmr_lrt_p"] = lmr.get(k, float("nan"))
        rows.append(row)
    return rows


def run_selection(
    matrix: CohortMatrix,
    typing: Typing,
    *,
    k_values: Sequence[int],
    n_iterations: int,
    n_init: int,
    base_seed: int = 0,
    cv: int = 3,
    checkpoint_dir: Path | None = None,
    workers: int = 0,
) -> SelectionResult:
    """Run the component-count grid over several seeds and summarise the criteria.

    Iterations are independent (each is its own cross-validation pass plus a fit per K), so
    they run concurrently over a :class:`~concurrent.futures.ProcessPoolExecutor`, each pinned
    to a single BLAS thread (:func:`analysis.profiling.single_threaded_blas`) so concurrent
    workers do not oversubscribe the machine. Each seeded iteration is one resumable unit: when
    ``checkpoint_dir`` is given, an iteration's criterion rows are appended to a checkpoint as
    it completes, and a re-run over the same directory recomputes only the iterations missing
    from it (tracked by iteration index, not by checkpoint length, since concurrent iterations
    do not finish in submission order). Because iteration ``i`` always uses seed
    ``base_seed + i``, a resumed run reproduces what an uninterrupted run would have computed;
    the per-iteration rows are reassembled in iteration order regardless of completion order,
    so a resumed run's table matches an uninterrupted one exactly.

    A single-initialisation fit can fail to converge when the structural covariate M-step hits
    a near-singular design, most often at higher class counts. Such a fit has its criteria
    recorded as ``nan`` rather than raising, so one bad fit does not lose the whole grid; the
    per-component summary skips those entries. This matches the cross-validation pass, where
    scikit-learn already scores a failed fold as ``nan``.

    Parameters
    ----------
    matrix : analysis.cohort.CohortMatrix
        The cohort feature and covariate matrices.
    typing : analysis.features.Typing
        The reconciled feature typing.
    k_values : sequence of int
        The component counts to grid over.
    n_iterations : int
        Number of seeded repetitions (the released code uses 200).
    n_init : int
        Random restarts per fit (the released validation fits use one).
    base_seed : int, default 0
        Seeds are ``base_seed`` to ``base_seed + n_iterations - 1``.
    cv : int, default 3
        Cross-validation folds for the validation log-likelihood.
    checkpoint_dir : Path, optional
        Directory for the resumable checkpoint. When ``None`` the run is held in memory and
        nothing is written, so an interrupt loses the work. The directory must be specific to
        these parameters (the caller passes the stage's content-addressed run directory).
    workers : int, default 0
        Concurrent worker processes. ``0`` uses the logical core count minus one. ``1`` runs
        in-process instead of through a pool, which a test that monkeypatches a fitting
        dependency relies on (a spawned worker process would not see the patch).

    Returns
    -------
    SelectionResult
        The per-iteration criteria, the per-component summary, and the gridded counts.
    """
    measurement, descriptor, covariates = prepare_inputs(matrix, typing)
    k_list = [int(k) for k in k_values]

    log = CheckpointLog(checkpoint_dir / f"select{SUFFIX}") if checkpoint_dir else None
    done = log.load() if log else []
    by_iteration: dict[int, list[dict[str, float]]] = {
        rows[0]["iteration"]: rows for rows in done if rows
    }

    # One unit per (iteration, K) fit plus one unit per iteration for the CV pass.
    units_per_iteration = len(k_list) + 1
    total = n_iterations * units_per_iteration
    pending = [i for i in range(n_iterations) if i not in by_iteration]
    n_workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with task_bar(total, "select", initial=len(by_iteration) * units_per_iteration) as bar:

        def handle(iteration: int, iteration_rows: list[dict[str, float]]) -> None:
            if log:
                log.append(iteration_rows)
            by_iteration[iteration] = iteration_rows
            bar.set_postfix(iteration=iteration, bic=f"{iteration_rows[-1]['bic']:.0f}")
            bar.update(units_per_iteration)

        if n_workers <= 1:
            # In-process: no pool overhead for a single worker, and a monkeypatched
            # dependency (as the non-convergence-guard tests use) still takes effect, which a
            # spawned process would not see (it re-imports the module fresh).
            for iteration in pending:
                handle(
                    iteration,
                    _run_iteration(
                        measurement,
                        covariates,
                        descriptor,
                        k_list,
                        iteration=iteration,
                        seed=base_seed + iteration,
                        n_init=n_init,
                        cv=cv,
                    ),
                )
        else:
            with (
                profiling.single_threaded_blas(),
                ProcessPoolExecutor(max_workers=n_workers) as pool,
            ):
                futures = {
                    pool.submit(
                        _run_iteration,
                        measurement,
                        covariates,
                        descriptor,
                        k_list,
                        iteration=iteration,
                        seed=base_seed + iteration,
                        n_init=n_init,
                        cv=cv,
                    ): iteration
                    for iteration in pending
                }
                for future in as_completed(futures):
                    handle(futures[future], future.result())

    records = [row for iteration in range(n_iterations) for row in by_iteration[iteration]]
    per_iteration = pd.DataFrame.from_records(records)
    criteria = [*PER_FIT_CRITERIA, "val_log_likelihood", "lmr_lrt_p"]
    grouped = per_iteration.groupby("n_components")[criteria]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary = summary.reset_index()
    return SelectionResult(per_iteration=per_iteration, summary=summary, k_values=k_list)
