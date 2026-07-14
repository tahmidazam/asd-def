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
two SPARK milestone features have no SSC equivalent. The SSC milestone ages are free text
without a consistent unit, so a bare number is ambiguous between months and years; the scale
is resolved per milestone against the SPARK reference distribution (``_milestone_priors`` and
``build_milestone_disambiguator``), so that a small number reads as months on an early
milestone and as years on a late one, as a human cleaner reads it. The fidelity of this
backend to the authors' SSC pipeline, and the exact shared-feature contract, are confirmed in
the SSC replication stage (phase 2). The backend does not provide diagnosis-timing fields.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from analysis.strata import BinningPolicy

from analysis import config
from analysis.cohort import open_catalogue, read_columns, source_csv
from analysis.cohort.schema import (
    MILESTONE_AGE_FEATURES,
    SSC_BH_RENAME,
    SSC_CBCL_RENAME,
    SSC_RBSR_RENAME,
    SSC_SCQ_RENAME,
    SSC_SEX_ENCODING,
    SSC_YES_NO,
    build_milestone_disambiguator,
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

    def family_ids(self, index: pd.Index) -> pd.Series | None:
        """Return ``None``: the family-clustered bootstrap is wired for the SPARK timing axes only.

        The local-trajectory recast (:mod:`analysis.trajectory_local`) runs on the SPARK-only age
        and era axes, so the SSC backend does not resolve a family key here. The one-family-per
        proband it would otherwise need is left unimplemented rather than guessed.
        """
        return None

    def axis(
        self,
        name: str,
        index: pd.Index,
        covariates: pd.DataFrame,
        min_bin_size: int = 1000,
    ) -> tuple[pd.Series, BinningPolicy] | None:
        """Resolve a stratification axis to its variable and binning policy.

        SSC provides the two shared cognitive axes and neither timing axis (it has no clean
        diagnosis timestamp, plan section 5). Both cognitive axes read the harmonised
        full-scale deviation IQ (``ssc_diagnosis.fs_deviation_score``): ``cognitive_impairment``
        dichotomises it at the intellectual-disability threshold (``config.ID_IQ_THRESHOLD``,
        the same construct as SPARK's flag), and ``iq`` bins it by equal frequency, using the
        continuous score SPARK lacks. ``age_at_diagnosis`` and ``era`` return ``None``.
        """
        from analysis import strata as strata_mod

        if name in ("cognitive_impairment", "iq"):
            values = self._read_axis_column("ssc_diagnosis", "fs_deviation_score", index)
            if name == "cognitive_impairment":
                return values, strata_mod.id_dichotomy(config.ID_IQ_THRESHOLD, low_is_impaired=True)
            return values, strata_mod.MaxEqualBins(min_bin_size=min_bin_size)
        return None

    def _read(self, table: str) -> pd.DataFrame:
        path = source_csv(self._cat, self.root, self.dataset, self.version, table, role=_ROLE)
        return pd.read_csv(path).set_index(_INDEX)

    def _read_axis_column(self, table: str, column: str, index: pd.Index) -> pd.Series:
        """Read one numeric column of a proband table, deduplicated and reindexed.

        Reads only the id and the requested column (the ``dscat`` guardrail), keeps the first
        row per proband, coerces to numeric, and aligns to the modelling-cohort index.
        """
        path = source_csv(self._cat, self.root, self.dataset, self.version, table, role=_ROLE)
        df = read_columns(path, [_INDEX, column]).set_index(_INDEX)
        df = df[~df.index.duplicated(keep="first")]
        return pd.to_numeric(df[column], errors="coerce").reindex(index)

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
        items = self._read("rbs_r_raw").rename(columns=SSC_RBSR_RENAME)
        # The RBS-R subscale totals (i_stereotyped_behavior_score ... vi_restricted) live in
        # the separate scored table under the same names the author feature list uses, so they
        # are joined alongside the raw items rather than dropped (the authors' SSC set carries
        # both the items and the subscale scores).
        scores = self._read("rbs_r")
        subscale = [c for c in scores.columns if c in self._features and c not in items.columns]
        return items.join(scores[subscale], how="outer")

    def _cbcl(self) -> pd.DataFrame:
        return self._read("cbcl_6_18").rename(columns=SSC_CBCL_RENAME)

    def _milestone_priors(self) -> dict[str, Callable[[float], float]]:
        """Build per-milestone scale resolvers from the SPARK reference distribution.

        The SSC milestone ages are free text without a consistent unit, so a bare number is
        ambiguous between months and years. SPARK records the same milestones as ages in
        months, so its distribution resolves the scale (:func:`build_milestone_disambiguator`).
        The prior is read from the reference SPARK release; if its source is unavailable the
        resolvers are omitted and a bare number is read as months.

        Returns
        -------
        dict of str to callable
            A scale resolver per milestone feature the prior could be built for.
        """
        try:
            path = source_csv(
                self._cat,
                self.root,
                config.REFERENCE_DATASET,
                config.REFERENCE_VERSION,
                "background_history_child",
            )
        except FileNotFoundError:
            log.warning(
                "SPARK milestone prior unavailable (%s/%s); SSC milestone scale not disambiguated",
                config.REFERENCE_DATASET,
                config.REFERENCE_VERSION,
            )
            return {}
        spark = read_columns(path, list(MILESTONE_AGE_FEATURES))
        priors: dict[str, Callable[[float], float]] = {}
        for col in MILESTONE_AGE_FEATURES:
            if col in spark.columns:
                values = pd.to_numeric(spark[col], errors="coerce").to_numpy(dtype=float)
                priors[col] = build_milestone_disambiguator(values)
        return priors

    def _background_history(self) -> pd.DataFrame:
        df = self._read("ssc_background_hx").rename(columns=SSC_BH_RENAME)
        priors = self._milestone_priors()
        for col in df.columns.intersection(MILESTONE_AGE_FEATURES):
            resolver = priors.get(col)
            df[col] = df[col].map(lambda v, d=resolver: parse_age_months(v, disambiguate=d))
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
