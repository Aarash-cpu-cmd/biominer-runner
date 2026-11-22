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
PROFILE_VERSION = "1.1.0"  # fix round: canonical reduction order + stable MAP inference (F_C2)
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


def _make_inference(n_cpus: int):
    """Noise-robust deterministic MAP inference (benchmark finding F_C2 fix).

    pydeseq2 0.5.4's ``DefaultInference.alpha_mle`` delegates to
    ``utils.fit_alpha_mle`` (scipy L-BFGS-B on the log-alpha NB posterior).
    L-BFGS-B's line search is sensitive to ~1e-16 input noise: a benign
    relabel that changes the design-matrix reduction order can flip its
    convergence flag (7 genes on airway) and move the MAP estimate by ~1e-3,
    cascading into lfc/pvalue/padj (findings F_C2 / I5-I7 failures).

    This subclass keeps the EXACT same objective as ``utils.fit_alpha_mle.loss``
    (nb_nll + Cox-Reid ``0.5*slogdet(X'WX)`` + log-residual prior
    ``(log_a-log_alpha_hat)^2/(2*prior_disp_var)``) but minimizes it with a
    deterministic bounded Brent search (``minimize_scalar(method="bounded",
    xatol=1e-10)``) per gene. Brent stops on interval width, not on a
    line-search gradient test, so ~1e-16 noise cannot move the result beyond
    xatol and the convergence flag is stable. Genes are independent and
    assembled in order (joblib), preserving gene-order determinism. The
    ``grid_fit_alpha`` fallback is retained for the rare non-convergent case.
    """
    from joblib import Parallel, delayed, parallel_backend
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.utils import grid_fit_alpha, nb_nll
    from scipy.optimize import minimize_scalar

    def _map_one_gene(counts, design_matrix, mu, alpha_hat, min_disp, max_disp,
                      prior_disp_var, cr_reg, prior_reg):
        log_alpha_hat = np.log(alpha_hat)
        lo, hi = np.log(min_disp), np.log(max_disp)

        def loss(log_alpha):
            alpha = np.exp(log_alpha)
            reg = 0.0
            if cr_reg:
                W = mu / (1 + mu * alpha)
                reg += 0.5 * np.linalg.slogdet((design_matrix.T * W) @ design_matrix)[1]
            if prior_reg:
                reg += (log_alpha - log_alpha_hat) ** 2 / (2 * prior_disp_var)
            return float(nb_nll(counts, mu, alpha)) + reg

        res = minimize_scalar(loss, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-10, "maxiter": 500})
        if res.success:
            return float(np.exp(res.x)), True
        return float(np.exp(grid_fit_alpha(counts, design_matrix, mu, alpha_hat,
                                           min_disp, max_disp))), False

    class _StableInference(DefaultInference):
        def alpha_mle(self, counts, design_matrix, mu, alpha_hat, min_disp, max_disp,
                      prior_disp_var=None, cr_reg=True, prior_reg=False,
                      optimizer="L-BFGS-B"):
            with parallel_backend(self._backend, inner_max_num_threads=1):
                res = Parallel(n_jobs=self.n_cpus, verbose=self._joblib_verbosity,
                               batch_size=self._batch_size)(
                    delayed(_map_one_gene)(
                        counts[:, i], design_matrix, mu[:, i], alpha_hat[i],
                        min_disp, max_disp, prior_disp_var, cr_reg, prior_reg)
                    for i in range(counts.shape[1]))
            res = zip(*res, strict=False)
            dispersions_, converged_ = (np.array(m) for m in res)
            return dispersions_, converged_

    return _StableInference(n_cpus=int(n_cpus))


def fit_core(counts: pd.DataFrame, conditions: pd.Series, numerator: str,
             denominator: str, *, n_cpus: int = N_CPUS_DEFAULT):
    """Shared frozen fit: canonical reduction order + stable MAP inference.

    Returns ``(table, dds, stats, filter_stats, effective, nonfinite)``.
    ``de.fit`` wraps this with the runner's result contract; the benchmark's
    ``common.fit_capture`` calls the SAME core so the metamorphic/identity
    tests exercise the shipped engine, not a divergent copy.

    F_C2 fix — canonical reduction order: genes (columns) are sorted by ID and
    samples (rows) by name before fitting, so every order-dependent float
    reduction (the dispersion-trend gamma-GLM ``nanmean`` over genes, the NB
    score sums over samples) sees a canonical operand order. A pure relabel or
    row permutation therefore cannot inject ~1e-16 trend noise. The result
    table is restored to the original post-prefilter gene order, so feature
    order is preserved.
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    if not 1 <= int(n_cpus) <= N_CPUS_MAX:
        raise ValueError(f"n_cpus must be within 1..{N_CPUS_MAX}")
    filtered, filter_stats = prefilter(counts)
    if filtered.shape[1] == 0:
        raise NoTestableGenes(
            f"no gene has count >= {MIN_COUNTS} in >= {MIN_SAMPLES} samples",
            {"gene_filter": filter_stats, "n_genes_input": int(counts.shape[1])})

    # F_C2 fix: canonical reduction order (genes by ID, samples by name).
    orig_genes = filtered.columns
    gene_order = np.argsort(orig_genes.to_numpy(), kind="stable")
    sample_order = np.argsort(filtered.index.to_numpy(), kind="stable")
    canon = filtered.iloc[sample_order, gene_order]

    # F_C2 fix (I5 label-swap invariance): canonical contrast baseline. The two
    # condition levels are ordered alphabetically for the INTERNAL fit, so a
    # numerator/denominator swap reuses the IDENTICAL design parameterization
    # and fit (bit-identical mu / dispersions / beta). The requested contrast
    # direction is then applied as an EXACT sign flip on log2FoldChange and
    # stat below (lfcSE, baseMean, pvalue, padj are direction-invariant). This
    # removes the reparameterization noise that otherwise cascades
    # genewise -> trend -> MAP under a baseline flip (c2_c: MoM ~1e-13 ->
    # genewise max_rel 7.0 on a near-zero-dispersion gene -> MAP 3.4e-5).
    # formulaic uses the first categorical level as the baseline; we fix it
    # canonically rather than to the denominator so both directions share one
    # numerical fit. The output contrast is still exactly numerator-vs-denominator.
    levels = sorted([str(numerator), str(denominator)])
    canon_den, canon_num = levels[0], levels[1]
    cond = conditions.loc[canon.index]
    meta = pd.DataFrame({
        "condition": pd.Categorical(
            cond.astype(str),
            categories=[canon_den, canon_num],
        )
    }, index=cond.index)

    inference = _make_inference(n_cpus)
    dds = DeseqDataSet(
        counts=canon,
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
        contrast=["condition", canon_num, canon_den],
        alpha=float(STATS_DEFAULTS["alpha"]),
        cooks_filter=bool(STATS_DEFAULTS["cooks_filter"]),
        independent_filter=bool(STATS_DEFAULTS["independent_filter"]),
        prior_LFC_var=STATS_DEFAULTS["prior_LFC_var"],
        lfc_null=float(STATS_DEFAULTS["lfc_null"]),
        inference=inference,
        quiet=True,
    )
    stats.summary()
    # Restore the original post-prefilter gene order (canonical sort was
    # internal only); copy so the six columns stay independent of results_df.
    table = stats.results_df.copy()[list(CORE_COLUMNS)].reindex(orig_genes)
    # Apply the requested contrast direction as an exact sign flip when it is
    # reversed relative to the canonical baseline (I5 invariance by construction).
    if str(numerator) == canon_den:
        table["log2FoldChange"] = -table["log2FoldChange"]
        table["stat"] = -table["stat"]

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
    return table, dds, stats, filter_stats, effective, nonfinite


def fit(counts: pd.DataFrame, conditions: pd.Series, numerator: str,
        denominator: str, *, n_cpus: int = N_CPUS_DEFAULT) -> dict:
    """Fit the frozen profile and return the raw result table plus details.

    ``counts`` is samples x genes (integer); ``conditions`` is indexed by the
    same sample ids. ``numerator`` is the tested level and ``denominator`` is
    the reference level.
    """
    table, _dds, _stats, filter_stats, effective, nonfinite = fit_core(
        counts, conditions, numerator, denominator, n_cpus=n_cpus)
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
        "map_optimizer": "brent_bounded (xatol=1e-10, maxiter=500; grid_fit_alpha fallback)",
        "canonical_reduction_order": True,
        "fix_round": "F_C2: canonical gene/sample reduction order + noise-robust deterministic MAP",
        "n_cpus": {"default": N_CPUS_DEFAULT, "max": N_CPUS_MAX},
    }
