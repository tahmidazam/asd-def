"""The SPARK cohort backend.

Reproduces the integration in the released ``process_integrate_phenotype_data.py``: the
SCQ, background-history (child and sibling), RBS-R, and CBCL 6-18 instruments are read,
screened on age and the per-instrument missingness counter, joined on the proband id, and
reduced to complete cases. Two things differ from the released code, both deliberate. The
kept columns are pinned to the authors' final feature list rather than rederived by
dropping columns (plan section 5), and only the needed columns of each CSV are read, so a
whole instrument file is never loaded (the ``dscat`` guardrail).

The released ``datadf.round()`` is a fit-time transform, applied in :mod:`analysis.model`
rather than here, so the cached cohort matrix matches the unrounded intermediate the
authors saved.

The backend takes an optional records cutoff (``as_of``) that restricts the cohort to the
probands present at an earlier SPARK freeze, so a later release can be cut back to (an
approximation of) the data Litman et al. fit on. See
:doc:`/packages/analysis/guides/subsetting-to-the-v9-freeze` for the method and its limits.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from analysis import config
from analysis.cohort import (
    INDEX_COLUMN,
    csv_columns,
    open_catalogue,
    read_columns,
    source_csv,
)
from analysis.cohort.schema import CBCL_REPLACEMENTS, SEX_ENCODING, load_feature_list

log = logging.getLogger("analysis.cohort")

_AGE = "age_at_eval_years"
_MISSING = "missing_values"
_EVAL_YEAR = "eval_year"
_REGISTRATION_TABLE = "core_descriptive_variables"
_REG_YEAR = "registration_year"
_REG_AGE = "age_at_registration_years"


class SparkCohort:
    """Build the harmonised SPARK proband-by-feature frame.

    Parameters
    ----------
    root : Path
        Repository root.
    version : str
        SPARK release version (for example ``"2026-03-23"``).
    as_of : str, optional
        A records cutoff. When set (for example ``"2022-12-12"``, Litman's V9 freeze), the
        cohort is restricted to probands registered by the cutoff year whose every cohort
        instrument was also completed by then. The cutoff is resolved to its calendar year.
        When ``None`` (the default), the full release is built and the behaviour matches the
        released preprocessing.
    """

    dataset = "spark"

    def __init__(self, root: Path, version: str, *, as_of: str | None = None) -> None:
        self.root = root
        self.version = version
        self.as_of = as_of
        self._cutoff_year = int(as_of[:4]) if as_of else None
        self._cat = open_catalogue(root)
        self._features = load_feature_list(config.author_feature_list(root))
        self._registration: pd.DataFrame | None = None

    def supports_timing(self) -> bool:
        """Return ``True``: SPARK carries the diagnosis-timing fields (plan section 5)."""
        return True

    def _path(self, table: str) -> Path:
        return source_csv(self._cat, self.root, self.dataset, self.version, table)

    def _instrument_features(self, available: list[str]) -> list[str]:
        """Return the author features present in an instrument, in feature-list order."""
        present = set(available)
        return [f for f in self._features if f in present]

    def _registration_frame(self) -> pd.DataFrame:
        """Return registration year and age at registration, indexed by proband.

        Read once and cached. Used by the records cutoff: the registration year is the v9
        roster gate, and the age at registration anchors CBCL's derived completion year
        (CBCL 6-18 carries no ``eval_year``).
        """
        if self._registration is None:
            path = self._path(_REGISTRATION_TABLE)
            df = read_columns(path, [INDEX_COLUMN, _REG_YEAR, _REG_AGE])
            self._registration = df.set_index(INDEX_COLUMN)
        return self._registration

    def _gate_eval_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows completed after the cutoff year and consume the ``eval_year`` column.

        Applies only when a cutoff is set; ``eval_year`` is an integer calendar year, so the
        gate is exact. A row with a missing ``eval_year`` cannot be confirmed as completed by
        the cutoff and is dropped.
        """
        if self._cutoff_year is None:
            return df
        gated = df[df[_EVAL_YEAR] <= self._cutoff_year]
        return gated.drop(columns=[_EVAL_YEAR])

    def _eval_year_cols(self) -> list[str]:
        """Return ``[eval_year]`` when a cutoff is active, else an empty list."""
        return [_EVAL_YEAR] if self._cutoff_year is not None else []

    def _read_scq(self) -> pd.DataFrame:
        path = self._path("scq")
        feats = self._instrument_features(csv_columns(path))
        df = read_columns(
            path, [INDEX_COLUMN, "sex", _AGE, _MISSING, *self._eval_year_cols(), *feats]
        )
        df = df[(df[_AGE] <= 18) & (df[_AGE] >= 4) & (df[_MISSING] < 1)]
        df = self._gate_eval_year(df)
        df = df.set_index(INDEX_COLUMN).drop(columns=[_MISSING])
        df["sex"] = df["sex"].replace(SEX_ENCODING).astype(int)
        log.info("scq: %d probands, %d features", len(df), len(feats))
        return df

    def _read_background_history(self) -> pd.DataFrame:
        frames = []
        for table in ("background_history_child", "background_history_sibling"):
            path = self._path(table)
            feats = self._instrument_features(csv_columns(path))
            df = read_columns(path, [INDEX_COLUMN, _AGE, *self._eval_year_cols(), *feats])
            df = df[(df[_AGE] <= 18) & (df[_AGE] >= 4)]
            df = self._gate_eval_year(df)
            frames.append(df.set_index(INDEX_COLUMN).drop(columns=[_AGE]))
        bh = pd.concat(frames, join="inner")
        bh = bh[~bh.index.duplicated(keep=False)]
        log.info("background history: %d rows (child + sibling), %d features", len(bh), bh.shape[1])
        return bh

    def _read_rbsr(self) -> pd.DataFrame:
        path = self._path("rbsr")
        feats = self._instrument_features(csv_columns(path))
        df = read_columns(path, [INDEX_COLUMN, _AGE, _MISSING, *self._eval_year_cols(), *feats])
        df = df[(df[_AGE] <= 18) & (df[_AGE] >= 4) & (df[_MISSING] < 1)]
        df = self._gate_eval_year(df)
        df = df.set_index(INDEX_COLUMN).drop(columns=[_AGE, _MISSING])
        log.info("rbsr: %d probands, %d features", len(df), len(feats))
        return df

    def _read_cbcl(self) -> pd.DataFrame:
        path = self._path("cbcl_6_18")
        feats = self._instrument_features(csv_columns(path))
        age_cols = [_AGE] if self._cutoff_year is not None else []
        df = read_columns(path, [INDEX_COLUMN, *age_cols, *feats]).set_index(INDEX_COLUMN)
        df = df.replace(CBCL_REPLACEMENTS)
        df = df[~df.index.duplicated(keep=False)]
        if self._cutoff_year is not None:
            keep = self._cbcl_within_cutoff(df[_AGE], self._registration_frame(), self._cutoff_year)
            df = df.loc[df.index.intersection(keep)].drop(columns=[_AGE])
        log.info("cbcl 6-18: %d probands, %d features", len(df), len(feats))
        return df

    @staticmethod
    def _cbcl_within_cutoff(
        age_at_eval: pd.Series, registration: pd.DataFrame, cutoff_year: int
    ) -> pd.Index:
        """Return the probands whose CBCL was completed by the cutoff year.

        CBCL 6-18 carries no ``eval_year``, so its completion year is reconstructed from the
        registration anchor: the registration year plus the age gained since registration.
        The result is a fractional calendar year; a proband is kept when that year's floor is
        at or before the cutoff (``floor(year) <= cutoff`` is ``year < cutoff + 1``). A proband
        missing a registration anchor drops out, since the year cannot be reconstructed.
        """
        derived_year = registration[_REG_YEAR] + (age_at_eval - registration[_REG_AGE])
        return derived_year[derived_year < cutoff_year + 1].index

    def _apply_roster_gate(self, complete: pd.DataFrame) -> pd.DataFrame:
        """Keep only probands registered by the cutoff year (the v9 roster gate)."""
        if self._cutoff_year is None:
            return complete
        reg = self._registration_frame()
        roster = reg.index[reg[_REG_YEAR] <= self._cutoff_year]
        keep = complete.index.intersection(roster)
        log.info(
            "v9 roster gate (registration_year <= %d): %d -> %d probands",
            self._cutoff_year,
            len(complete),
            len(keep),
        )
        return complete.loc[keep]

    def integrate(self) -> pd.DataFrame:
        """Integrate the instruments into the harmonised, complete-case cohort frame.

        Returns
        -------
        pandas.DataFrame
            Proband-by-column frame indexed by proband id, holding the covariates and the
            pinned feature set, coerced to numeric and reduced to complete cases. When a
            records cutoff is set, the frame is further restricted to the probands present at
            the cutoff.
        """
        scq = self._read_scq()
        bh = self._read_background_history()
        rbsr = self._read_rbsr()
        cbcl = self._read_cbcl()

        merged = pd.concat([scq, bh, rbsr, cbcl], axis=1, join="inner")
        merged = merged.loc[:, ~merged.columns.duplicated()]
        merged = merged.apply(pd.to_numeric, errors="coerce")

        before = len(merged)
        complete = merged.dropna(axis=0)
        complete = self._apply_roster_gate(complete)
        log.info(
            "integrated: %d probands before complete-case, %d after, %d columns",
            before,
            len(complete),
            complete.shape[1],
        )
        return complete
