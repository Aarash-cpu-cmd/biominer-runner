"""Local GSEA (prerank) and ORA over a frozen GMT library.

Both analyses consume a registered DE table and a gene-symbol map, and write
their own provenance record (``bm_local_pathway_v1``) so the server can store
a pathway result without recomputing it. Human only in V1; mouse uses an MGI
ortholog table in V2, never case folding or name matching.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import Refusal
from .runner import RUNNER_VERSION, TRACKED_PACKAGES, _RssSampler, _file_sha, _version

SCHEMA = "bm_local_pathway_v1"
GSEA_ENGINE = "gseapy"
GSEA_ENGINE_VERSION = "1.3.0"
ORA_ENGINE = "hypergeometric"
PERMUTATIONS = 1000
SEED = 42
MIN_SIZE = 15
MAX_SIZE = 500
THREADS = 1
ALPHA = 0.05
LFC_THRESHOLD = 1.0
FDR_STRICT = 0.05
FDR_EXPLORATORY = 0.25
SUPPORTED_SPECIES = ("human",)
# Accept the canonical scientific name used across the rest of BioMiner
# (study_create / dataset_search / signature_search all use "Homo sapiens")
# as a synonym for the runner's colloquial "human". V1 stays human-only.
_SPECIES_SYNONYMS = {
    "human": "human",
    "homo sapiens": "human",
    "h. sapiens": "human",
    "hsapiens": "human",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _check_species(species: str) -> str:
    value = str(species or "").strip().lower()
    canonical = _SPECIES_SYNONYMS.get(value)
    if canonical is None:
        raise Refusal(
            "SPECIES_UNSUPPORTED",
            "V1 pathway analysis supports human only; mouse is a V2 feature",
            {"species": species, "supported": list(SUPPORTED_SPECIES)})
    return canonical


def _read_gmt(path: Path, species: str) -> tuple[dict[str, list[str]], dict]:
    if not path.is_file():
        raise Refusal("GENE_SET_INVALID", "gene set file does not exist",
                      {"path": str(path)})
    terms: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            term = parts[0].strip()
            genes = [g.strip().upper() for g in parts[2:] if g.strip()]
            if not term or not genes:
                continue
            terms.setdefault(term, []).extend(genes)
    # Deduplicate per term without changing the term order.
    terms = {term: sorted(set(genes)) for term, genes in terms.items()}
    if not terms:
        raise Refusal("GENE_SET_INVALID", "gene set file has no usable terms")
    report = {
        "n_terms": len(terms),
        "n_genes": len({g for genes in terms.values() for g in genes}),
        "sha256": _file_sha(path),
        "path": str(path),
        "species": species,
    }
    return terms, report


def _verify_manifest(manifest_path: Path | None, gene_set: dict) -> dict | None:
    if manifest_path is None:
        return None
    if not manifest_path.is_file():
        raise Refusal("GENE_SET_INVALID", "gene set manifest does not exist")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise Refusal("GENE_SET_INVALID", "gene set manifest is not JSON")
    expected = str(manifest.get("sha256") or "").strip().lower()
    if not expected:
        raise Refusal("GENE_SET_INVALID", "manifest has no sha256")
    if expected != gene_set["sha256"]:
        raise Refusal(
            "GENE_SET_DIGEST_MISMATCH",
            "gene set bytes do not match the frozen manifest",
            {"expected": expected, "found": gene_set["sha256"]})
    n_terms = manifest.get("n_terms")
    if n_terms is not None and int(n_terms) != gene_set["n_terms"]:
        raise Refusal(
            "GENE_SET_DIGEST_MISMATCH",
            "gene set term count does not match the frozen manifest",
            {"expected": n_terms, "found": gene_set["n_terms"]})
    return {"library": manifest.get("library"),
            "version": manifest.get("version"),
            "sha256": expected,
            "n_terms": gene_set["n_terms"]}


def _read_de_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    if "gene" not in table.columns:
        raise Refusal("DE_TABLE_INVALID", "DE table has no gene column")
    for required in ("stat", "padj", "log2FoldChange"):
        if required not in table.columns:
            raise Refusal("DE_TABLE_INVALID",
                          f"DE table has no {required} column",
                          {"columns": list(table.columns)})
    if len(table) == 0:
        raise Refusal("DE_TABLE_INVALID", "DE table has no rows")
    return table


def _read_symbols(path: Path | None) -> pd.Series | None:
    if path is None:
        return None
    if not path.is_file():
        raise Refusal("SYMBOLS_INVALID", "gene symbol file does not exist")
    # keep_default_na=False keeps literal tokens ("NA"/"None"/"NaN") as strings
    # rather than float NaN, which would break the downstream .str operations.
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "gene_id" not in frame.columns or "gene_symbol" not in frame.columns:
        raise Refusal("SYMBOLS_INVALID",
                      "gene symbol table needs gene_id and gene_symbol")
    series = frame.set_index("gene_id")["gene_symbol"].str.strip()
    missing = {"", "nan", "na", "none", "null", "n/a"}
    series = series[~series.str.lower().isin(missing)]
    series = series[series.index.astype(str).str.strip() != ""]
    # One symbol per gene_id (keep first); a duplicate index would otherwise
    # raise on reindex in _labels. Mirrors the ranking keep-first dedup policy.
    return series[~series.index.duplicated(keep="first")]


def _labels(genes: pd.Series, symbols: pd.Series | None, species: str) -> pd.Series:
    genes = genes.astype(str)
    if symbols is None or len(symbols) == 0:
        labels = genes.copy()
    else:
        symbols = symbols[~symbols.index.duplicated(keep="first")]
        labels = symbols.reindex(genes).fillna(genes)
    # Force string dtype: an all-missing mapping can leave a float dtype that
    # would otherwise raise on the .str accessor below.
    labels = labels.astype(str).str.strip()
    return labels if species != "human" else labels.str.upper()


def _build_rank(de_table: pd.DataFrame, symbols: pd.Series | None,
                species: str) -> tuple[pd.DataFrame, dict]:
    stat = pd.to_numeric(de_table["stat"], errors="coerce")
    labels = _labels(de_table["gene"], symbols, species)
    # Series.reindex returns an index keyed by the gene ids, which would
    # misalign against the positional DE table index; build arrays instead.
    frame = pd.DataFrame({"gene": np.asarray(de_table["gene"], dtype=str),
                          "label": np.asarray(labels, dtype=str),
                          "stat": stat.to_numpy(dtype="float64")})
    n_before = len(frame)
    frame = frame.dropna(subset=["stat"])
    frame = frame[frame["label"].ne("") & frame["label"].ne("NAN")]
    # F_D1 fix (FIX_ROUND_plan §5.2): choose the per-label representative by
    # max |stat| with a gene-id tiebreak -- both invariant under contrast
    # direction reversal (stat -> -stat). Sorting on signed stat made the kept
    # gene flip when the direction reversed, so the GSEA ranking was not
    # direction-covariant (finding F_D1). |stat| and gene-id are flip-invariant,
    # so each label keeps the SAME representative gene under flip; the ranking
    # becomes a pure reversal and ES/NES negate exactly. Final ranking is
    # ordered by signed stat (prerank convention).
    frame["_abs_stat"] = frame["stat"].abs()
    frame = frame.sort_values(["_abs_stat", "gene"], ascending=[False, True],
                              kind="mergesort")
    duplicated = frame["label"].duplicated().sum()
    first = frame[~frame["label"].duplicated(keep="first")].copy()
    rank = first.set_index("label")[["stat"]].sort_values(
        "stat", ascending=False, kind="mergesort")
    report = {
        "n_genes_in_de_table": int(n_before),
        "n_genes_after_dropna": int(len(frame)),
        "n_genes_after_uppercase_dedup": int(len(rank)),
        "n_duplicates_removed": int(duplicated),
        "dedup_rate": round(len(rank) / n_before, 4) if n_before else 0.0,
        "uppercase_applied": species == "human",
        "dedup_strategy": "keep_first (max |stat|, gene-id tiebreak; direction-independent)",
    }
    return rank, report


def _gmtset_mapping(rank: pd.DataFrame, terms: dict[str, list[str]]) -> float:
    ranked = set(rank.index)
    library = {g for genes in terms.values() for g in genes}
    return round(len(ranked & library) / len(ranked), 4) if ranked else 0.0


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def run_gsea(de_table_path: Path, gene_sets_path: Path, out_dir: Path, *,
             gene_symbols_path: Path | None = None,
             manifest_path: Path | None = None,
             species: str = "human",
             permutations: int = PERMUTATIONS, seed: int = SEED,
             min_size: int = MIN_SIZE, max_size: int = MAX_SIZE) -> dict:
    started = time.time()
    species = _check_species(species)
    out_dir.mkdir(parents=True, exist_ok=True)
    de_table = _read_de_table(de_table_path)
    symbols = _read_symbols(gene_symbols_path)
    terms, gene_set = _read_gmt(gene_sets_path, species)
    manifest = _verify_manifest(manifest_path, gene_set)

    rank, mapping = _build_rank(de_table, symbols, species)
    if rank.empty:
        raise Refusal("GSEA_EMPTY_RANKING",
                      "no gene has a finite ranking statistic; GSEA cannot run",
                      {"n_genes_in_de_table": int(len(de_table))})
    mapping["gene_set_mapping_rate"] = _gmtset_mapping(rank, terms)
    rank.to_csv(out_dir / "ranking.tsv", sep="\t", header=False)

    try:
        import gseapy as gp
    except ImportError:
        raise Refusal("ENGINE_MISMATCH", "gseapy is not installed")
    if _version("gseapy") != GSEA_ENGINE_VERSION:
        raise Refusal(
            "ENGINE_MISMATCH",
            f"gseapy is {_version('gseapy')}, the frozen runner requires "
            f"{GSEA_ENGINE_VERSION}",
            {"required": GSEA_ENGINE_VERSION, "found": _version("gseapy")})

    sampler = _RssSampler()
    with sampler:
        try:
            result = gp.prerank(
                rnk=rank,
                gene_sets=str(gene_sets_path),
                outdir=None,
                permutation_num=int(permutations),
                seed=int(seed),
                min_size=int(min_size),
                max_size=int(max_size),
                threads=THREADS,
                verbose=False,
            )
        except LookupError as exc:
            # gseapy raises LookupError when no gene set survives the size /
            # mapping filter; refuse cleanly instead of an INTERNAL_ERROR crash.
            raise Refusal(
                "GSEA_NO_TESTABLE_TERMS",
                "no gene set passed the size/mapping filter for this ranking",
                {"n_genes_in_ranking": int(len(rank)),
                 "detail": str(exc).splitlines()[0]}) from exc
    results = result.res2d.copy()

    results_path = out_dir / "gsea_results.tsv"
    results.to_csv(results_path, sep="\t", index=False, float_format="%.10g")
    _write_json(out_dir / "run_config.json", {
        "method": "gseapy.prerank",
        "rank_column": "stat",
        "gene_sets": str(gene_sets_path),
        "gene_set_sha256": gene_set["sha256"],
        "permutation_num": int(permutations),
        "seed": int(seed),
        "min_size": int(min_size),
        "max_size": int(max_size),
        "threads": THREADS,
        "outdir": None,
        "n_genes_in_ranking": int(len(rank)),
    })
    _write_json(out_dir / "mapping_report.json", mapping)
    _write_json(out_dir / "gene_set_provenance.json", {
        **gene_set,
        "manifest": manifest,
        "gseapy_version": GSEA_ENGINE_VERSION,
        "reproducible": True,
        "source_type": "local_gmt_file",
    })

    fdr_column = next((c for c in ("FDR q-val", "FDR", "fdr", "fdr q-val",
                                   "Adjusted P-value", "adj_p")
                       if c in results.columns), None)
    if fdr_column is not None:
        fdr = pd.to_numeric(results[fdr_column], errors="coerce")
        n_strict = int((fdr < FDR_STRICT).sum())
        n_exploratory = int((fdr < FDR_EXPLORATORY).sum())
    else:
        n_strict = n_exploratory = None
    elapsed = round(time.time() - started, 2)
    peak, peak_note = _peak(sampler)
    provenance = {
        "schema": SCHEMA,
        "result_kind": "gsea",
        "runner_version": RUNNER_VERSION,
        "execution_origin": "local",
        "validation_status": "schema_validated",
        "engine": GSEA_ENGINE,
        "engine_version": GSEA_ENGINE_VERSION,
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "dependencies": {pkg: _version(pkg) for pkg in TRACKED_PACKAGES},
        "parent_de_table_sha256": _file_sha(de_table_path),
        "gene_set": {**gene_set, "manifest": manifest},
        "params": {"rank_column": "stat", "permutation_num": int(permutations),
                   "seed": int(seed), "min_size": int(min_size),
                   "max_size": int(max_size), "threads": THREADS,
                   "species": species},
        "input": {"de_table_sha256": _file_sha(de_table_path),
                  "gene_symbols_sha256": (_file_sha(gene_symbols_path)
                                          if gene_symbols_path else None)},
        "output": {"pathway_table_sha256": _file_sha(results_path),
                   "n_terms": int(len(results)),
                   "n_significant_fdr_0_05": n_strict,
                   "n_hypothesis_fdr_0_25": n_exploratory,
                   "fdr_column": fdr_column},
        "metrics": {"exit_code": 0, "elapsed_s": elapsed,
                    "peak_rss_mb": peak, "peak_rss_note": peak_note},
        "created_at": _utcnow(),
    }
    _write_json(out_dir / "provenance.json", provenance)
    return provenance


def _peak(sampler: _RssSampler) -> tuple[float | None, str]:
    from .runner import _peak_rss_mb
    return _peak_rss_mb(sampler.peak_mb, sampler.error)


def _bh(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype="float64")
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, dtype="float64")
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


def run_ora(de_table_path: Path, gene_sets_path: Path, out_dir: Path, *,
            gene_symbols_path: Path | None = None,
            manifest_path: Path | None = None,
            species: str = "human") -> dict:
    started = time.time()
    species = _check_species(species)
    out_dir.mkdir(parents=True, exist_ok=True)
    de_table = _read_de_table(de_table_path)
    symbols = _read_symbols(gene_symbols_path)
    terms, gene_set = _read_gmt(gene_sets_path, species)
    manifest = _verify_manifest(manifest_path, gene_set)

    padj = pd.to_numeric(de_table["padj"], errors="coerce")
    lfc = pd.to_numeric(de_table["log2FoldChange"], errors="coerce")
    labels = _labels(de_table["gene"], symbols, species)
    finite = np.isfinite(padj.to_numpy(dtype="float64")) & \
        np.isfinite(lfc.to_numpy(dtype="float64"))
    effective = finite & (padj.to_numpy(dtype="float64") < ALPHA)
    up = effective & (lfc.to_numpy(dtype="float64") >= LFC_THRESHOLD)
    down = effective & (lfc.to_numpy(dtype="float64") <= -LFC_THRESHOLD)
    query_up = set(labels[up])
    query_down = set(labels[down])

    background = set(labels) & {g for genes in terms.values() for g in genes}
    rows: list[dict] = []
    provenance_rows = {}
    from scipy.stats import hypergeom
    for direction, query in (("up", query_up), ("down", query_down)):
        query_bg = query & background
        tested = []
        if not query_bg:
            provenance_rows[direction] = {"query_size": len(query),
                                          "query_in_background": 0,
                                          "terms_tested": 0}
            continue
        for term, genes in terms.items():
            term_bg = set(genes) & background
            if not term_bg:
                continue
            overlap = query_bg & term_bg
            k = len(overlap)
            n = len(term_bg)
            M = len(background)
            N = len(query_bg)
            pvalue = float(hypergeom.sf(k - 1, M, n, N)) if n and N else 1.0
            fold = ((k / N) / (n / M)) if N and n else 0.0
            rows.append({
                "direction": direction,
                "term": term,
                "overlap": k,
                "set_size": n,
                "background_size": M,
                "query_size": N,
                "fold_enrichment": round(fold, 6),
                "pvalue": pvalue,
                "genes": ",".join(sorted(overlap)),
            })
            tested.append(len(rows) - 1)
        if tested:
            q = _bh(np.array([rows[i]["pvalue"] for i in tested]))
            for index, qv in zip(tested, q):
                rows[index]["padj"] = float(qv)
        provenance_rows[direction] = {"query_size": len(query),
                                      "query_in_background": N,
                                      "terms_tested": len(tested)}

    if rows:
        frame = pd.DataFrame(rows)
        frame = frame.sort_values(["direction", "padj", "pvalue", "term"],
                                  kind="mergesort")
    else:
        frame = pd.DataFrame(columns=[
            "direction", "term", "overlap", "set_size", "background_size",
            "query_size", "fold_enrichment", "pvalue", "padj", "genes"])
    table_path = out_dir / "ora_results.tsv"
    frame.to_csv(table_path, sep="\t", index=False, float_format="%.10g")
    n_significant = int(
        (pd.to_numeric(frame["padj"], errors="coerce") < ALPHA).sum()
    ) if len(frame) else 0

    elapsed = round(time.time() - started, 2)
    provenance = {
        "schema": SCHEMA,
        "result_kind": "ora",
        "runner_version": RUNNER_VERSION,
        "execution_origin": "local",
        "validation_status": "schema_validated",
        "engine": ORA_ENGINE,
        "engine_version": _version("scipy"),
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "dependencies": {pkg: _version(pkg) for pkg in TRACKED_PACKAGES},
        "parent_de_table_sha256": _file_sha(de_table_path),
        "gene_set": {**gene_set, "manifest": manifest},
        "params": {"alpha": ALPHA, "lfc_threshold": LFC_THRESHOLD,
                   "lfc_basis": "raw", "correction": "benjamini_hochberg",
                   "queries": ["effect_up", "effect_down"],
                   "species": species},
        "counts": provenance_rows,
        "input": {"de_table_sha256": _file_sha(de_table_path),
                  "gene_symbols_sha256": (_file_sha(gene_symbols_path)
                                          if gene_symbols_path else None)},
        "output": {"pathway_table_sha256": _file_sha(table_path),
                   "n_terms": int(len(frame)),
                   "n_significant_padj_0_05": n_significant,
                   "background_size": len(background)},
        "metrics": {"exit_code": 0, "elapsed_s": elapsed},
        "created_at": _utcnow(),
    }
    _write_json(out_dir / "ora_provenance.json", provenance)
    return provenance
