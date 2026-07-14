"""The cohort abstraction: one interface, a SPARK and an SSC backend behind it.

Every analysis is written once against :class:`Cohort` and the harmonised
:class:`CohortMatrix` it yields, so the reference fit, the stability checks, and the SSC
replication run on either cohort without change (plan section 10). Each backend maps its
raw tables onto the shared schema; the SPARK-only timing fields (age at diagnosis, era)
are exposed as an optional capability that the SSC backend need not provide.

The shared helpers here resolve a table's source CSV through the ``dscat`` catalogue and
read only the columns a stage needs, so a backend never loads a whole file into memory.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd
from dscat import queries
from dscat.index import Catalogue
from dscat.paths import index_path

from analysis import config
from analysis.paths import find_repo_root

if TYPE_CHECKING:
    from analysis.strata import BinningPolicy

INDEX_COLUMN = "subject_sp_id"


@dataclass
class CohortMatrix:
    """A harmonised proband-by-feature matrix with its covariates and provenance.

    Attributes
    ----------
    features : pandas.DataFrame
        Proband-by-feature matrix, indexed by proband id, holding only the requested
        clustered features (covariates excluded).
    covariates : pandas.DataFrame
        Proband-by-covariate matrix on the same index, holding the structural-model
        covariates.
    dataset : str
        Cohort the matrix was built from.
    version : str
        Dataset version the matrix was built from.
    """

    features: pd.DataFrame
    covariates: pd.DataFrame
    dataset: str
    version: str

    @property
    def feature_names(self) -> list[str]:
        """Return the feature column names."""
        return list(self.features.columns)

    @property
    def n_probands(self) -> int:
        """Return the number of probands (rows)."""
        return int(len(self.features))


@runtime_checkable
class Cohort(Protocol):
    """The contract every cohort backend satisfies.

    A backend integrates its instruments into one harmonised, complete-case
    proband-by-column frame (covariates and the shared feature schema), and declares
    whether it can provide the SPARK-only diagnosis-timing fields used for the
    stratification axes (plan sections 5 and 7).
    """

    dataset: str
    version: str

    def integrate(self) -> pd.DataFrame:
        """Return the harmonised, complete-case proband-by-column frame."""
        ...

    def supports_timing(self) -> bool:
        """Return whether the backend can provide diagnosis-timing fields."""
        ...

    def axis(
        self,
        name: str,
        index: pd.Index,
        covariates: pd.DataFrame,
        min_bin_size: int = 1000,
    ) -> tuple[pd.Series, BinningPolicy] | None:
        """Return a stratification variable and its default binning policy.

        The stratified analysis (plan section 7) re-estimates the mixture within strata of an
        axis. A backend resolves a named axis to the per-proband variable (on ``index``) and
        the policy that bins it, so a stage names an axis and never reads a cohort-specific
        column itself. This generalises :meth:`supports_timing`: the diagnosis-timing axes
        (``age_at_diagnosis``, ``era``) are SPARK-only, while the cognitive axes
        (``cognitive_impairment``, ``iq``) are shared, so the SSC backend provides the second
        pair and none of the first (plan sections 5 and 8).

        Parameters
        ----------
        name : str
            The axis to resolve.
        index : pandas.Index
            The modelling-cohort proband index the variable is built on.
        covariates : pandas.DataFrame
            The cohort covariates, on ``index``; the timing axes read age at evaluation and
            sex from here.
        min_bin_size : int, default 1000
            The floor passed to a size-based policy (:class:`~analysis.strata.MaxEqualBins`);
            ignored by the fixed dichotomy.

        Returns
        -------
        tuple or None
            The variable (a :class:`pandas.Series`) and its
            :class:`~analysis.strata.BinningPolicy`, or ``None`` when the backend does not
            provide ``name``.
        """
        ...

    def family_ids(self, index: pd.Index) -> pd.Series | None:
        """Return the per-proband family identifier, or ``None`` when the backend has none.

        The clustered bootstrap in :mod:`analysis.trajectory_local` resamples families rather
        than probands, so a backend that groups probands into families exposes the grouping key
        here. Like :meth:`axis`, this is an optional capability: a backend that cannot provide a
        family key returns ``None``.
        """
        ...


def open_catalogue(root: Path) -> Catalogue:
    """Open the ``dscat`` catalogue at the repository root."""
    return Catalogue.open(index_path(root))


def source_csv(
    cat: Catalogue, root: Path, dataset: str, version: str, table: str, role: str = ""
) -> Path:
    """Return the absolute path to a table's backing CSV.

    Parameters
    ----------
    cat : Catalogue
        Open catalogue.
    root : Path
        Repository root, used to resolve the catalogue's repo-relative path.
    dataset, version, table : str
        The table to locate.
    role : str, default ""
        Family role for cohorts that split a measure across role folders (SSC). SPARK
        tables use the empty role.

    Returns
    -------
    Path
        Absolute path to the CSV.

    Raises
    ------
    FileNotFoundError
        When no source row matches ``table`` and ``role``.
    """
    for row in queries.feature_sources(cat, dataset, version, table):
        if (row["role"] or "") == role:
            return root / row["file_path"]
    raise FileNotFoundError(f"no source CSV for {dataset}/{version} {table!r} role={role!r}")


def csv_columns(path: Path) -> list[str]:
    """Return a CSV's column names without reading its rows."""
    return list(pd.read_csv(path, nrows=0).columns)


def read_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read only the requested columns of a CSV, skipping any that are absent.

    Parameters
    ----------
    path : Path
        CSV to read.
    columns : sequence of str
        Columns to keep; columns not present in the file are ignored.

    Returns
    -------
    pandas.DataFrame
        The requested columns that exist in the file.
    """
    wanted = set(columns)
    return pd.read_csv(path, usecols=lambda c: c in wanted)


def build_matrix(
    integrated: pd.DataFrame,
    feature_names: Sequence[str],
    dataset: str,
    version: str,
    covariates: Sequence[str] = config.COVARIATES,
) -> CohortMatrix:
    """Split an integrated frame into a feature matrix and a covariate matrix.

    Parameters
    ----------
    integrated : pandas.DataFrame
        The backend's harmonised, complete-case frame.
    feature_names : sequence of str
        The clustered features to keep.
    dataset, version : str
        Provenance recorded on the matrix.
    covariates : sequence of str, optional
        Covariate columns to split out. Defaults to the structural-model covariates.

    Returns
    -------
    CohortMatrix
        The feature and covariate matrices on a shared index.

    Raises
    ------
    KeyError
        When a requested feature or covariate is absent from ``integrated``.
    """
    missing_features = [f for f in feature_names if f not in integrated.columns]
    if missing_features:
        raise KeyError(f"{len(missing_features)} requested features absent: {missing_features[:8]}")
    missing_covariates = [c for c in covariates if c not in integrated.columns]
    if missing_covariates:
        raise KeyError(f"covariates absent: {missing_covariates}")
    return CohortMatrix(
        features=integrated.loc[:, list(feature_names)].copy(),
        covariates=integrated.loc[:, list(covariates)].copy(),
        dataset=dataset,
        version=version,
    )


def get_cohort(
    dataset: str, version: str, root: Path | None = None, *, as_of: str | None = None
) -> Cohort:
    """Return the backend for a dataset.

    Parameters
    ----------
    dataset : str
        ``"spark"`` or ``"ssc"``.
    version : str
        Dataset version.
    root : Path, optional
        Repository root. Defaults to the discovered root.
    as_of : str, optional
        A records cutoff passed to the SPARK backend (for example ``"2022-12-12"``); it
        restricts the cohort to the probands present at that freeze. Ignored by cohorts that
        do not carry the timing fields the cutoff needs (the SSC).

    Returns
    -------
    Cohort
        The backend.

    Raises
    ------
    ValueError
        When ``dataset`` is not a known cohort.
    """
    root = root or find_repo_root()
    if dataset == "spark":
        from analysis.cohort.spark import SparkCohort

        return SparkCohort(root, version, as_of=as_of)
    if dataset == "ssc":
        from analysis.cohort.ssc import SscCohort

        return SscCohort(root, version)
    raise ValueError(f"unknown cohort {dataset!r}")
