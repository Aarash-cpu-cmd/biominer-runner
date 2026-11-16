"""D23 bulk-Wald DE profile, vendored for the local runner.

Profile: ``spec/d23/ENGINE_PROFILE_v1.json`` (``bulk_wald_poscounts_raw_v1``).
Every value here is frozen compatibility policy, not a tuned choice. The
server keeps its own call site (``core.deseq._run_pydeseq2``); the parity
test fits the same synthetic matrix through both and compares the six core
columns, so the two implementations cannot drift unnoticed.

This module is deliberately free of server imports: the runner ships as a
standalone wheel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PROFILE_ID = "bulk_wald_poscounts_raw_v1"
PROFILE_VERSION = "1.0.0"
ENGINE = "pydeseq2"
ENGINE_VERSION = "0.5.4"
MIN_COUNTS = 10
MIN_SAMPLES = 2
ALPHA = 0.05
LFC_THRESHOLD = 1.0
N_CPUS_DEFAULT = 1
N_CPUS_MAX = 4
CORE_COLUMNS = ("baseMean", "log2FoldChange", "lfcSE", "stat",
                "pvalue", "padj")
FIT_DEFAULTS = {
    "fit_type": "parametric",
    "size_factors_fit_type": "poscounts",
    "refit_cooks": True,
    "low_memory": True,
    "min_replicates": 7,
    "min_mu": 0.5,
    "min_disp": 1e-8,
    "max_disp": 10.0,
    "beta_tol": 1e-8,
    "control_genes": None,
}
STATS_DEFAULTS = {
    "alpha": ALPHA,
    "cooks_filter": True,
    "independent_filter": True,
    "prior_LFC_var": None,
    "lfc_null": 0.0,
}


class NoTestableGenes(Exception):
    """Every gene failed the D23 prefilter; the engine must not run."""

    def __init__(self, message: str, details: dict | None = None):
        self.details = details or {}
        super().__init__(message)


def prefilter(counts: pd.DataFrame, *, min_counts: int = MIN_COUNTS,
              min_samples: int = MIN_SAMPLES) -> tuple[pd.DataFrame, dict]:
    """Apply ``count >= min_counts`` in ``>= min_samples`` samples.

    ``counts`` is samples x genes. Feature order is preserved.
    """
    keep = (counts >= min_counts).sum(axis=0) >= min_samples
    filtered = counts.loc[:, keep]
    stats = {
        "min_counts": int(min_counts),
        "min_samples": int(min_samples),
        "predicate": "sum(counts >= min_counts over samples) >= min_samples",
        "n_genes_before": int(counts.shape[1]),
        "n_genes_after": int(filtered.shape[1]),
        "n_genes_removed": int(counts.shape[1] - filtered.shape[1]),
        "feature_order_preserved": True,
    }
    return filtered, stats


def fit(counts: pd.DataFrame, conditions: pd.Series, numerator: str,
        denominator: str, *, n_cpus: int = N_CPUS_DEFAULT) -> dict:
    """Fit the frozen profile and return the raw result table plus details.

    ``counts`` is samples x genes (integer); ``conditions`` is indexed by the
    same sample ids. ``numerator`` is the tested level and ``denominator`` is
    the reference level.
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    if not 1 <= int(n_cpus) <= N_CPUS_MAX:
        raise ValueError(f"n_cpus must be within 1..{N_CPUS_MAX}")
    filtered, filter_stats = prefilter(counts)
    if filtered.shape[1] == 0:
        raise NoTestableGenes(
            f"no gene has count >= {MIN_COUNTS} in >= {MIN_SAMPLES} samples",
            {"gene_filter": filter_stats, "n_genes_input": int(counts.shape[1])})

    # pydeseq2 0.5.x builds the design through formulaic, which uses the first
    # categorical level as the baseline. Plain strings would pick an
    # alphabetical baseline and could invert the coefficient if the tested
    # level sorts before the reference. Put the reference first explicitly.
    meta = pd.DataFrame({
        "condition": pd.Categorical(
            conditions.astype(str),
            categories=[denominator, numerator],
        )
    }, index=conditions.index)

    inference = DefaultInference(n_cpus=int(n_cpus))
    dds = DeseqDataSet(
        counts=filtered,
        metadata=meta,
        design="~condition",
        refit_cooks=bool(FIT_DEFAULTS["refit_cooks"]),
        size_factors_fit_type=FIT_DEFAULTS["size_factors_fit_type"],
        low_memory=bool(FIT_DEFAULTS["low_memory"]),
        min_replicates=int(FIT_DEFAULTS["min_replicates"]),
        min_mu=float(FIT_DEFAULTS["min_mu"]),
        min_disp=float(FIT_DEFAULTS["min_disp"]),
        max_disp=float(FIT_DEFAULTS["max_disp"]),
        beta_tol=float(FIT_DEFAULTS["beta_tol"]),
        control_genes=FIT_DEFAULTS["control_genes"],
        inference=inference,
        quiet=True,
    )
    dds.deseq2()

    stats = DeseqStats(
        dds,
        contrast=["condition", numerator, denominator],
        alpha=float(STATS_DEFAULTS["alpha"]),
        cooks_filter=bool(STATS_DEFAULTS["cooks_filter"]),
        independent_filter=bool(STATS_DEFAULTS["independent_filter"]),
        prior_LFC_var=STATS_DEFAULTS["prior_LFC_var"],
        lfc_null=float(STATS_DEFAULTS["lfc_null"]),
        inference=inference,
        quiet=True,
    )
    stats.summary()
    table = stats.results_df.copy()[list(CORE_COLUMNS)]

    requested_fit = getattr(dds, "fit_type", None)
    effective_fit = dds.uns.get("disp_function_type") or requested_fit
    effective = {
        "fit_type_requested": requested_fit,
        "fit_type_effective": effective_fit,
        "dispersion_trend_fallback": bool(
            requested_fit == "parametric" and effective_fit != "parametric"),
        "size_factors_fit_type": getattr(dds, "size_factors_fit_type", None),
        "n_samples_used": int(filtered.shape[0]),
        "n_genes_used": int(filtered.shape[1]),
    }

    nonfinite = {}
    for column in CORE_COLUMNS:
        values = table[column].to_numpy(dtype="float64")
        nonfinite[column] = {
            "n_nan": int(np.isnan(values).sum()),
            "n_infinite": int(np.isinf(values).sum()),
        }
    return {
        "table": table,
        "filter": filter_stats,
        "effective": effective,
        "nonfinite": nonfinite,
    }


def classify(table: pd.DataFrame, *, alpha: float = ALPHA,
             lfc_threshold: float = LFC_THRESHOLD) -> dict:
    """Apply the D23 raw-value significance and effect rules."""
    padj = pd.to_numeric(table["padj"], errors="coerce").to_numpy(dtype="float64")
    lfc = pd.to_numeric(table["log2FoldChange"], errors="coerce").to_numpy(dtype="float64")
    significant = np.isfinite(padj) & (padj < alpha)
    effect = significant & np.isfinite(lfc) & (np.abs(lfc) >= lfc_threshold)
    return {
        "n_significant": int(significant.sum()),
        "n_up": int((significant & (lfc > 0)).sum()),
        "n_down": int((significant & (lfc < 0)).sum()),
        "n_effect_candidates": int(effect.sum()),
        "n_effect_up": int((effect & (lfc > 0)).sum()),
        "n_effect_down": int((effect & (lfc < 0)).sum()),
        "alpha": float(alpha),
        "lfc_threshold": float(lfc_threshold),
    }


def profile_record() -> dict:
    """The frozen profile as plain data, written into provenance."""
    return {
        "profile_id": PROFILE_ID,
        "profile_version": PROFILE_VERSION,
        "engine": ENGINE,
        "engine_version": ENGINE_VERSION,
        "gene_filter": {"min_counts": MIN_COUNTS, "min_samples": MIN_SAMPLES},
        "fit": dict(FIT_DEFAULTS),
        "stats": dict(STATS_DEFAULTS),
        "classification": {"padj_threshold": ALPHA,
                           "lfc_threshold": LFC_THRESHOLD,
                           "lfc_basis": "raw"},
        "shrinkage": False,
        "n_cpus": {"default": N_CPUS_DEFAULT, "max": N_CPUS_MAX},
    }
