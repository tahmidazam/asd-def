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


class SparkCohort:
    """Build the harmonised SPARK proband-by-feature frame.

    Parameters
    ----------
    root : Path
        Repository root.
    version : str
        SPARK release version (for example ``"2026-03-23"``).
    """

    dataset = "spark"

    def __init__(self, root: Path, version: str) -> None:
        self.root = root
        self.version = version
        self._cat = open_catalogue(root)
        self._features = load_feature_list(config.author_feature_list(root))

    def supports_timing(self) -> bool:
        """Return ``True``: SPARK carries the diagnosis-timing fields (plan section 5)."""
        return True

    def _path(self, table: str) -> Path:
        return source_csv(self._cat, self.root, self.dataset, self.version, table)

    def _instrument_features(self, available: list[str]) -> list[str]:
        """Return the author features present in an instrument, in feature-list order."""
        present = set(available)
        return [f for f in self._features if f in present]

    def _read_scq(self) -> pd.DataFrame:
        path = self._path("scq")
        feats = self._instrument_features(csv_columns(path))
        df = read_columns(path, [INDEX_COLUMN, "sex", _AGE, _MISSING, *feats])
        df = df[(df[_AGE] <= 18) & (df[_AGE] >= 4) & (df[_MISSING] < 1)]
        df = df.set_index(INDEX_COLUMN).drop(columns=[_MISSING])
        df["sex"] = df["sex"].replace(SEX_ENCODING).astype(int)
        log.info("scq: %d probands, %d features", len(df), len(feats))
        return df

    def _read_background_history(self) -> pd.DataFrame:
        frames = []
        for table in ("background_history_child", "background_history_sibling"):
            path = self._path(table)
            feats = self._instrument_features(csv_columns(path))
            df = read_columns(path, [INDEX_COLUMN, _AGE, *feats])
            df = df[(df[_AGE] <= 18) & (df[_AGE] >= 4)]
            frames.append(df.set_index(INDEX_COLUMN).drop(columns=[_AGE]))
        bh = pd.concat(frames, join="inner")
        bh = bh[~bh.index.duplicated(keep=False)]
        log.info("background history: %d rows (child + sibling), %d features", len(bh), bh.shape[1])
        return bh

    def _read_rbsr(self) -> pd.DataFrame:
        path = self._path("rbsr")
        feats = self._instrument_features(csv_columns(path))
        df = read_columns(path, [INDEX_COLUMN, _AGE, _MISSING, *feats])
        df = df[(df[_AGE] <= 18) & (df[_AGE] >= 4) & (df[_MISSING] < 1)]
        df = df.set_index(INDEX_COLUMN).drop(columns=[_AGE, _MISSING])
        log.info("rbsr: %d probands, %d features", len(df), len(feats))
        return df

    def _read_cbcl(self) -> pd.DataFrame:
        path = self._path("cbcl_6_18")
        feats = self._instrument_features(csv_columns(path))
        df = read_columns(path, [INDEX_COLUMN, *feats]).set_index(INDEX_COLUMN)
        df = df.replace(CBCL_REPLACEMENTS)
        df = df[~df.index.duplicated(keep=False)]
        log.info("cbcl 6-18: %d probands, %d features", len(df), len(feats))
        return df

    def integrate(self) -> pd.DataFrame:
        """Integrate the instruments into the harmonised, complete-case cohort frame.

        Returns
        -------
        pandas.DataFrame
            Proband-by-column frame indexed by proband id, holding the covariates and the
            pinned feature set, coerced to numeric and reduced to complete cases.
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
        log.info(
            "integrated: %d probands before complete-case, %d after, %d columns",
            before,
            len(complete),
            complete.shape[1],
        )
        return complete
