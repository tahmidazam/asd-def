"""The SSC cohort backend.

Harmonises the SSC proband instruments (CBCL 6-18, RBS-R, SCQ-Lifetime, core descriptive,
and background history) onto the shared SPARK feature schema, following the renames in the
released ``generate_ssc_data`` (plan section 10). Rather than reproduce the authors' column
drop-lists, which targeted their SSC version, features are selected positively: a SPARK
feature is provided when its SSC column (after renaming) exists. The SSC backend therefore
exposes the subset of the schema the SSC instruments cover.

Two caveats are recorded honestly. The authors read the background-history milestones from
a hand-cleaned file that was not released, so both the SSC-to-SPARK mapping (``SSC_BH_RENAME``)
and the parse of the raw free-text ages into months (``parse_age_months``) are ours, and
two SPARK milestone features have no SSC equivalent. The fidelity of this backend to the
authors' SSC pipeline, and the exact shared-feature contract, are confirmed in the SSC
replication stage (phase 2). The backend does not provide diagnosis-timing fields.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from analysis import config
from analysis.cohort import open_catalogue, source_csv
from analysis.cohort.schema import (
    MILESTONE_AGE_FEATURES,
    SSC_BH_RENAME,
    SSC_CBCL_RENAME,
    SSC_RBSR_RENAME,
    SSC_SCQ_RENAME,
    SSC_SEX_ENCODING,
    SSC_YES_NO,
    load_feature_list,
    parse_age_months,
)

log = logging.getLogger("analysis.cohort")

_INDEX = "individual"
_ROLE = "proband"


class SscCohort:
    """Build the harmonised SSC proband-by-feature frame.

    Parameters
    ----------
    root : Path
        Repository root.
    version : str
        SSC dataset version (for example ``"15.3"``).
    """

    dataset = "ssc"

    def __init__(self, root: Path, version: str) -> None:
        self.root = root
        self.version = version
        self._cat = open_catalogue(root)
        self._features = set(load_feature_list(config.author_feature_list(root)))

    def supports_timing(self) -> bool:
        """Return ``False``: SSC does not expose a clean diagnosis timestamp (plan section 5)."""
        return False

    def _read(self, table: str) -> pd.DataFrame:
        path = source_csv(self._cat, self.root, self.dataset, self.version, table, role=_ROLE)
        return pd.read_csv(path).set_index(_INDEX)

    def _core_descriptive(self) -> pd.DataFrame:
        df = self._read("ssc_core_descriptive")
        out = pd.DataFrame(index=df.index)
        if "sex" in df.columns:
            out["sex"] = df["sex"].replace(SSC_SEX_ENCODING)
        if "age_at_ados" in df.columns:
            out["age_at_eval_years"] = df["age_at_ados"] / 12
        return out

    def _scq(self) -> pd.DataFrame:
        items = self._read("scq_life_recode").rename(columns=SSC_SCQ_RENAME)
        items = items.replace(SSC_YES_NO)
        summary = self._read("scq_life").rename(columns=SSC_SCQ_RENAME)
        keep = [c for c in ("final_score",) if c in summary.columns]
        return items.join(summary[keep], how="outer")

    def _rbsr(self) -> pd.DataFrame:
        return self._read("rbs_r_raw").rename(columns=SSC_RBSR_RENAME)

    def _cbcl(self) -> pd.DataFrame:
        return self._read("cbcl_6_18").rename(columns=SSC_CBCL_RENAME)

    def _background_history(self) -> pd.DataFrame:
        df = self._read("ssc_background_hx").rename(columns=SSC_BH_RENAME)
        for col in df.columns.intersection(MILESTONE_AGE_FEATURES):
            df[col] = df[col].map(parse_age_months)
        return df

    def integrate(self) -> pd.DataFrame:
        """Integrate the SSC proband instruments into a harmonised, complete-case frame.

        Returns
        -------
        pandas.DataFrame
            Proband-by-column frame indexed by individual id, holding the covariates and
            the subset of the shared feature schema the SSC instruments provide, coerced to
            numeric and reduced to complete cases.
        """
        parts = [
            self._core_descriptive(),
            self._scq(),
            self._rbsr(),
            self._cbcl(),
            self._background_history(),
        ]
        merged = pd.concat(parts, axis=1, join="outer")
        merged = merged.loc[:, ~merged.columns.duplicated()]

        keep = [c for c in merged.columns if c in self._features or c in config.COVARIATES]
        merged = merged[keep].apply(pd.to_numeric, errors="coerce")

        shared_features = sorted(set(keep) - set(config.COVARIATES))
        before = len(merged)
        complete = merged.dropna(axis=0)
        log.info(
            "ssc integrated: %d shared features, %d probands before complete-case, %d after",
            len(shared_features),
            before,
            len(complete),
        )
        return complete
