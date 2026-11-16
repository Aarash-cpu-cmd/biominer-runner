"""Fixed-version local runner for bulk two-group differential expression.

The runner is the only thing allowed to compute a differential table on the
client side. It refuses anything outside the frozen support matrix and writes
a provenance record that the server can check independently:

  * bulk RNA-seq counts: raw integer counts, or salmon/kallisto estimated
    counts summed per gene and rounded (declared in provenance)
  * two groups, unpaired, no batch adjustment
  * the D23 ``bulk_wald_poscounts_raw_v1`` engine profile, pydeseq2 0.5.4

Outputs, all under --out:
    de_table.tsv       the differential table (tab separated, NA for missing)
    gene_symbols.tsv   gene id -> display symbol used by pathway analysis
    figures/*.svg      volcano, PCA and heatmap rendered as deterministic SVG
    report.md          human-readable report quoting the digests above
    provenance.json    input digests, contrast, engine and dependency versions,
                       filter counts, timings, peak RSS and output digest
    run.log            human-readable log of what was read and produced
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import de, figures, quant, report
from .errors import Refusal

RUNNER_VERSION = "2.1.0"
PINNED_ENGINE = {"pydeseq2": de.ENGINE_VERSION}
TRACKED_PACKAGES = ("pydeseq2", "pandas", "numpy", "anndata", "scipy", "psutil")
MAX_GENES = 70_000
MAX_SAMPLES = 200
SIGNIFICANCE = "finite(padj) and padj < alpha"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "MISSING"


def dependencies() -> dict[str, str]:
    return {pkg: _version(pkg) for pkg in TRACKED_PACKAGES}


def check_engine() -> None:
    have = dependencies()
    for pkg, want in PINNED_ENGINE.items():
        if have[pkg] != want:
            raise Refusal(
                "ENGINE_MISMATCH",
                f"{pkg} is {have[pkg]}, the frozen runner requires {want}",
                {"required": PINNED_ENGINE, "found": have})


def _read_tsv(path: Path, code: str) -> pd.DataFrame:
    """TSV read whose parser failures surface as a structured Refusal."""
    try:
        return pd.read_csv(path, sep="\t", dtype=str)
    except Refusal:
        raise
    except Exception as exc:
        raise Refusal(code, f"cannot parse {path.name}: {exc}") from exc


def _read_sheet(path: Path) -> pd.DataFrame:
    sheet = _read_tsv(path, "SHEET_INVALID")
    sheet.columns = [c.strip() for c in sheet.columns]
    for required in ("sample_id", "condition"):
        if required not in sheet.columns:
            raise Refusal("SHEET_INVALID", f"sheet must contain {required}")
    sheet["sample_id"] = sheet["sample_id"].astype(str).str.strip()
    sheet["condition"] = sheet["condition"].astype(str).str.strip()
    if sheet["sample_id"].duplicated().any():
        dupes = sorted(sheet["sample_id"][sheet["sample_id"].duplicated()].unique())
        raise Refusal("SHEET_INVALID", "duplicate sample ids in the sheet",
                      {"duplicates": dupes[:20]})
    if sheet["sample_id"].eq("").any() or sheet["condition"].eq("").any():
        raise Refusal("SHEET_INVALID", "blank sample_id or condition")
    return sheet


def _read_counts(path: Path,
                 sample_ids: list[str]) -> tuple[pd.DataFrame, pd.Series | None]:
    """Read a gene x sample table; return (samples x genes, symbols or None).

    The first column is the gene key. An optional ``gene_symbol`` column (also
    ``gene_name`` or ``symbol``) is kept as display metadata for pathway
    analysis and excluded from the count matrix.
    """
    raw = _read_tsv(path, "COUNTS_INVALID")
    if raw.shape[1] < 2:
        raise Refusal("COUNTS_INVALID", "counts table has no sample columns")
    raw.columns = [str(c).strip() for c in raw.columns]
    gene_ids = raw.iloc[:, 0].astype(str).str.strip()
    raw = raw.iloc[:, 1:]
    symbol_col = next(
        (c for c in ("gene_symbol", "gene_name", "symbol") if c in raw.columns),
        None)
    symbols = None
    if symbol_col is not None:
        symbols = raw[symbol_col].astype(str).str.strip()
        raw = raw.drop(columns=[symbol_col])
    symbols = None if symbols is None else symbols.set_axis(gene_ids, axis=0)
    raw.index = pd.Index(gene_ids, name="gene")

    if raw.index.duplicated().any():
        dupes = sorted(raw.index[raw.index.duplicated()].unique())
        raise Refusal("COUNTS_INVALID", "duplicate gene identifiers",
                      {"duplicates": dupes[:20]})
    if (raw.index == "").any():
        raise Refusal("COUNTS_INVALID", "blank gene identifier")
    if len(raw.index) > MAX_GENES:
        raise Refusal("INPUT_TOO_LARGE", f"more than {MAX_GENES} genes")
    if len(raw.columns) > MAX_SAMPLES:
        raise Refusal("INPUT_TOO_LARGE", f"more than {MAX_SAMPLES} samples")
    missing = sorted(set(sample_ids) - set(raw.columns))
    extra = sorted(set(raw.columns) - set(sample_ids))
    if missing or extra:
        raise Refusal(
            "SHEET_MISMATCH",
            "counts columns and sheet samples differ",
            {"missing_in_counts": missing, "missing_in_sheet": extra})
    # A permutation of the same sample ids is not safe input: matching by
    # position is what a reader of the two files would do, so refuse rather
    # than silently reorder the sheet to fit the counts.
    sheet_order = [str(s) for s in sample_ids]
    counts_order = [str(c) for c in raw.columns]
    if counts_order != sheet_order:
        raise Refusal(
            "SHEET_ORDER_MISMATCH",
            "counts sample columns are not in sheet order",
            {"sheet_order": sheet_order[:40], "counts_order": counts_order[:40]})
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        rows, cols = numeric.isna().to_numpy().nonzero()
        culprit = f"{numeric.index[rows[0]]}/{numeric.columns[cols[0]]}"
        raise Refusal(
            "COUNTS_NOT_INTEGER",
            "counts contain values that are not numbers (TPM/FPKM or broken cells)",
            {"first_bad_cell": culprit})
    values = numeric.to_numpy(dtype=float)
    if (values < 0).any():
        raise Refusal("COUNTS_INVALID", "counts contain negative values")
    if not np.all(values == np.floor(values)):
        raise Refusal(
            "COUNTS_NOT_INTEGER",
            "counts must be raw integer counts; TPM/FPKM-style values are refused")
    return numeric.astype("int64").T, symbols  # samples x genes


def _check_design(sheet: pd.DataFrame, numerator: str, denominator: str,
                  *, paired: bool) -> dict:
    levels = sorted(sheet["condition"].unique())
    if len(levels) != 2:
        raise Refusal("DESIGN_UNSUPPORTED",
                      "exactly two condition levels are supported",
                      {"levels": levels})
    if numerator not in levels or denominator not in levels:
        raise Refusal("DESIGN_UNSUPPORTED",
                      "numerator and denominator must be the two conditions",
                      {"levels": levels})
    if numerator == denominator:
        raise Refusal("DESIGN_UNSUPPORTED", "numerator equals denominator")
    sizes = sheet["condition"].value_counts().to_dict()
    if sizes.get(numerator, 0) < 2 or sizes.get(denominator, 0) < 2:
        raise Refusal("DESIGN_UNSUPPORTED",
                      "each group needs at least two samples",
                      {"sizes": {k: int(v) for k, v in sizes.items()}})
    lowered = {c.casefold() for c in sheet.columns}
    if paired or "subject_id" in lowered or "pair_id" in lowered:
        raise Refusal("DESIGN_UNSUPPORTED",
                      "paired designs are not supported in V1")
    if "batch" in lowered:
        batches = sorted(set(sheet["batch"].astype(str)))
        if len(batches) > 1:
            raise Refusal("DESIGN_UNSUPPORTED",
                          "batch adjustment is not supported in V1",
                          {"batches": batches[:10]})
    return {"levels": levels, "sizes": {k: int(v) for k, v in sizes.items()}}


class _RssSampler:
    """Sample this process's RSS while the engine runs and keep the maximum.

    A reading taken after the run is not a peak. ``resource`` provides a true
    high-water mark on POSIX but does not exist on Windows, so this samples
    ``psutil.Process().memory_info().rss`` on a short interval instead.
    """

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.peak_mb: float | None = None
        self.error = ""
        self._process = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def __enter__(self) -> "_RssSampler":
        try:
            import psutil  # type: ignore
        except Exception as exc:  # noqa: BLE001 - optional dependency
            self.error = f"psutil unavailable: {exc}"
            return self
        try:
            self._process = psutil.Process(os.getpid())
        except Exception as exc:  # noqa: BLE001 - optional dependency
            self.error = f"psutil.Process failed: {exc}"
            return self
        self._thread = threading.Thread(target=self._run, name="bm-rss-sampler",
                                        daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        try:
            while True:
                self._sample()
                if self._stop.wait(self.interval):
                    break
        except Exception as exc:  # noqa: BLE001 - sampling must never fail the run
            self.error = f"rss sampler stopped: {type(exc).__name__}: {exc}"

    def _sample(self) -> None:
        value = self._process.memory_info().rss / (1 << 20)
        self.peak_mb = max(self.peak_mb or 0.0, float(value))


def _resource_peak_mb() -> float | None:
    try:
        import resource  # type: ignore
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 if sys.platform != "darwin" else (1 << 20)
        return round(value / divisor, 1)
    except Exception:  # noqa: BLE001 - unavailable on Windows
        return None


def _peak_rss_mb(sampled_peak_mb: float | None = None,
                 sampler_error: str = "") -> tuple[float | None, str]:
    resource_peak = _resource_peak_mb()
    if sampled_peak_mb is not None:
        peak = sampled_peak_mb if resource_peak is None else max(sampled_peak_mb,
                                                                 resource_peak)
        return round(float(peak), 1), "sampled via psutil"
    if resource_peak is not None:
        return resource_peak, ""
    try:
        import psutil  # type: ignore
        return (round(psutil.Process(os.getpid()).memory_info().rss / (1 << 20), 1),
                "instantaneous rss; psutil sampler unavailable")
    except Exception:  # noqa: BLE001 - nothing else to measure with
        note = "peak RSS is unavailable on this platform"
        if sampler_error:
            note = f"{note}; {sampler_error}"
        return None, note


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _resolve_n_cpus(params: dict | None) -> int:
    overrides = dict(params or {})
    unknown = sorted(set(overrides) - {"n_cpus"})
    if unknown:
        raise Refusal(
            "PARAMS_UNSUPPORTED",
            "the frozen engine profile accepts no parameter overrides except n_cpus",
            {"unsupported": unknown, "profile": de.PROFILE_ID})
    n_cpus = int(overrides.get("n_cpus", de.N_CPUS_DEFAULT))
    if not 1 <= n_cpus <= de.N_CPUS_MAX:
        raise Refusal("PARAMS_UNSUPPORTED",
                      f"n_cpus must be within 1..{de.N_CPUS_MAX}",
                      {"n_cpus": n_cpus})
    return n_cpus


def _write_gene_symbols(path: Path, genes: pd.Index,
                        symbols: pd.Series | None) -> str | None:
    if symbols is None:
        values = pd.Series(genes.astype(str), index=genes)
    else:
        values = symbols.reindex(genes).fillna(pd.Series(genes.astype(str), index=genes))
    frame = pd.DataFrame({"gene_id": genes.astype(str),
                          "gene_symbol": values.reindex(genes).astype(str)})
    frame.to_csv(path, sep="\t", index=False)
    return _file_sha(path)


def _write_counts(path: Path, counts: pd.DataFrame,
                  symbols: pd.Series | None) -> None:
    """Write a gene x sample table with an explicit gene_symbol column."""
    genes = counts.columns
    table = counts.T.copy()
    table.insert(0, "gene_symbol",
                 (symbols.reindex(genes).fillna(pd.Series(genes.astype(str),
                                                          index=genes))
                  if symbols is not None
                  else pd.Series(genes.astype(str), index=genes)).astype(str))
    table.index.name = "gene_id"
    table.to_csv(path, sep="\t", float_format="%.6f")


def run(counts_path: Path, sheet_path: Path, numerator: str,
        denominator: str, out_dir: Path, params: dict | None = None,
        *, input_transform: dict | None = None) -> dict:
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(message: str) -> None:
        log_lines.append(message)

    check_engine()
    deps = dependencies()
    n_cpus = _resolve_n_cpus(params)
    profile = de.profile_record()
    frozen_params = {
        "profile_id": profile["profile_id"],
        "profile_version": profile["profile_version"],
        "engine": profile["engine"],
        "engine_version": profile["engine_version"],
        **profile["fit"],
        **profile["stats"],
        "gene_filter": profile["gene_filter"],
        "classification": profile["classification"],
        "shrinkage": False,
        "n_cpus": n_cpus,
        "design": "~condition",
        "paired": False,
        "batch": False,
        "reference_level_first": True,
        "significance": SIGNIFICANCE,
    }

    sheet = _read_sheet(sheet_path)
    counts, symbols = _read_counts(counts_path, sheet["sample_id"].tolist())
    design = _check_design(sheet, numerator, denominator, paired=False)
    log(f"read {counts_path.name}: {counts.shape[1]} genes x {counts.shape[0]} samples")
    if input_transform:
        log(f"input transform: {input_transform.get('type', 'unknown')}")
    log(f"design: {numerator} vs {denominator}, groups {design['sizes']}")

    counts_sha = _file_sha(counts_path)
    sheet_sha = _file_sha(sheet_path)
    params_sha = _sha256_bytes(json.dumps(frozen_params, sort_keys=True,
                                         separators=(",", ":")).encode())

    conditions = sheet.set_index("sample_id").loc[counts.index, "condition"]

    sampler = _RssSampler()
    try:
        with sampler:
            fit = de.fit(counts, conditions, numerator, denominator, n_cpus=n_cpus)
    except de.NoTestableGenes as exc:
        raise Refusal(
            "NO_TESTABLE_GENES",
            str(exc),
            {"gene_filter": {"min_counts": de.MIN_COUNTS,
                             "min_samples": de.MIN_SAMPLES},
             "n_genes_input": int(counts.shape[1])})

    table = fit["table"]
    log(f"gene filter: {fit['filter']['n_genes_before']} -> "
        f"{fit['filter']['n_genes_after']} genes "
        f"(>= {de.MIN_COUNTS} counts in >= {de.MIN_SAMPLES} samples)")

    # D23: non-finite values serialize as null with a recorded reason.
    table = table.replace([np.inf, -np.inf], np.nan)
    table.index.name = "gene"
    table = table.reset_index()
    table = table.sort_values(["padj", "pvalue", "gene"],
                              na_position="last", kind="mergesort")
    table = table[["gene", "baseMean", "log2FoldChange", "lfcSE", "stat",
                   "pvalue", "padj"]]
    de_path = out_dir / "de_table.tsv"
    table.to_csv(de_path, sep="\t", index=False, na_rep="NA", float_format="%.10g")

    symbols_path = out_dir / "gene_symbols.tsv"
    symbols_sha = _write_gene_symbols(symbols_path, pd.Index(table["gene"]), symbols)

    counts_class = de.classify(table)
    log(f"wrote {de_path.name}: {len(table)} rows, "
        f"{counts_class['n_up']} up / {counts_class['n_down']} down at padj<{de.ALPHA}")

    log_counts, norm_report = figures.normalized_log(counts)
    fig_digests = figures.write_figures(
        out_dir, table, log_counts,
        sheet.set_index("sample_id").loc[counts.index, "condition"],
        norm_report, alpha=de.ALPHA, lfc_threshold=de.LFC_THRESHOLD,
        numerator=numerator, denominator=denominator)
    log("wrote figures: " + ", ".join(sorted(fig_digests)))
    output_info = {"de_table_sha256": _file_sha(de_path),
                   "gene_symbols_sha256": symbols_sha,
                   "n_rows": int(len(table)),
                   "n_significant": counts_class["n_significant"],
                   "n_up": counts_class["n_up"],
                   "n_down": counts_class["n_down"],
                   "n_effect_candidates": counts_class["n_effect_candidates"],
                   "n_effect_up": counts_class["n_effect_up"],
                   "n_effect_down": counts_class["n_effect_down"],
                   "alpha": de.ALPHA,
                   "lfc_threshold": de.LFC_THRESHOLD,
                   "figures": fig_digests}
    report_path = out_dir / "report.md"
    report_md = report.render_markdown(
        contrast={"numerator": numerator, "denominator": denominator},
        design=design,
        input_info={"counts_filename": counts_path.name,
                    "counts_sha256": counts_sha,
                    "sheet_filename": sheet_path.name,
                    "sheet_sha256": sheet_sha,
                    "n_genes": int(counts.shape[1]),
                    "n_samples": int(counts.shape[0]),
                    "sample_ids": [str(s) for s in counts.index],
                    "input_transform": input_transform
                    or {"type": "raw_integer_counts"}},
        output_info=output_info, params={**frozen_params,
                                         "params_sha256": params_sha},
        effective=fit["effective"], gene_filter=fit["filter"],
        classification=counts_class,
        deps={**deps, "python": platform.python_version()},
        table=table, figures=fig_digests,
        runner_version=RUNNER_VERSION, norm_report=norm_report)
    report_path.write_text(report_md, encoding="utf-8", newline="\n")
    output_info["report_sha256"] = _file_sha(report_path)
    log(f"wrote {report_path.name}")

    elapsed = round(time.time() - started, 2)
    peak, peak_note = _peak_rss_mb(sampler.peak_mb, sampler.error)
    provenance = {
        "schema": "bm_local_result_v1",
        "runner_version": RUNNER_VERSION,
        "execution_origin": "local",
        "validation_status": "schema_validated",
        "engine": "pydeseq2",
        "engine_version": deps.get("pydeseq2", "MISSING"),
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "dependencies": deps,
        "params": frozen_params,
        "engine_params_effective": fit["effective"],
        "gene_filter": fit["filter"],
        "nonfinite": fit["nonfinite"],
        "classification": counts_class,
        "params_sha256": params_sha,
        "contrast": {"numerator": numerator, "denominator": denominator,
                     "direction": f"{numerator}_vs_{denominator}"},
        "design": design,
        "input": {"counts_filename": counts_path.name,
                  "counts_sha256": counts_sha,
                  "sheet_filename": sheet_path.name,
                  "sheet_sha256": sheet_sha,
                  "n_genes": int(counts.shape[1]),
                  "n_samples": int(counts.shape[0]),
                  "sample_ids": [str(s) for s in counts.index],
                  "gene_symbol_source": "column" if symbols is not None else "gene_id",
                  "input_transform": input_transform or {"type": "raw_integer_counts"},
                  "gene_symbols_sha256": symbols_sha},
        "output": output_info,
        "metrics": {"exit_code": 0, "elapsed_s": elapsed,
                    "peak_rss_mb": peak, "peak_rss_note": peak_note,
                    "n_cpus": n_cpus},
        "created_at": _utcnow(),
    }
    provenance_path = out_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n",
                               encoding="utf-8")
    (out_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return provenance


def validate_inputs(quant_dir: Path | None, tx2gene_path: Path | None,
                    sheet_path: Path, numerator: str, denominator: str,
                    *, counts_path: Path | None = None) -> dict:
    """Read every input and report what a fit would see. Writes nothing.

    The same readers the run path uses parse the files here, so a pass
    means ``run`` will get past input handling and a refusal carries the
    same code the run would have raised.
    """
    sheet = _read_sheet(sheet_path)
    if quant_dir is not None:
        counts, symbols, transform = quant.build_counts(
            quant_dir, tx2gene_path, sheet["sample_id"].tolist())
        source = {"type": "quant_dir", "quant_dir": str(quant_dir),
                  "tx2gene": str(tx2gene_path),
                  "transform": transform}
    else:
        counts, symbols = _read_counts(
            counts_path, sheet["sample_id"].tolist())
        source = {"type": "counts", "counts": str(counts_path)}
    design = _check_design(sheet, numerator, denominator, paired=False)
    filtered, filter_stats = de.prefilter(counts)
    if filtered.shape[1] == 0:
        raise Refusal(
            "NO_TESTABLE_GENES",
            f"no gene has count >= {de.MIN_COUNTS} in "
            f">= {de.MIN_SAMPLES} samples",
            {"gene_filter": filter_stats,
             "n_genes_input": int(counts.shape[1])})
    return {"source": source,
            "sheet": {"sample_ids": [str(s) for s in sheet["sample_id"]],
                      "levels": design["levels"]},
            "counts": {"n_genes": int(counts.shape[1]),
                       "n_samples": int(counts.shape[0]),
                       "gene_symbol_source": "column"
                       if symbols is not None else "gene_id"},
            "design": {"numerator": numerator,
                       "denominator": denominator,
                       "sizes": design["sizes"]},
            "gene_filter": filter_stats,
            "n_genes_testable": int(filtered.shape[1])}


def run_quant(quant_dir: Path, tx2gene_path: Path, sheet_path: Path,
              numerator: str, denominator: str, out_dir: Path,
              params: dict | None = None) -> dict:
    """Convert salmon/kallisto output to integer gene counts and fit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = _read_sheet(sheet_path)
    counts, symbols, report = quant.build_counts(
        quant_dir, tx2gene_path, sheet["sample_id"].tolist())
    counts_path = out_dir / "counts.tsv"
    _write_counts(counts_path, counts, symbols)
    transform = {"type": "salmon_kallisto_est_counts", **report}
    return run(counts_path, sheet_path, numerator, denominator, out_dir, params,
               input_transform=transform)
