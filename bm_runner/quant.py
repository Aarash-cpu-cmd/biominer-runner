"""Salmon/kallisto transcript quantification -> gene counts.

The D23 profile accepts raw integer gene counts. Quantification output is
fractional, so the runner sums transcript estimated counts per gene and rounds
half-up, recording the transform and the unrounded totals in provenance. This
is a declared V1 limitation, not a claim that rounding is statistically free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .errors import Refusal

QUANT_FILES = ("quant.sf", "abundance.tsv")
TRANSCRIPT_COLUMNS = ("Name", "target_id", "transcript_id")
COUNT_COLUMNS = ("NumReads", "est_counts", "counts")


def _read_tsv(path: Path, code: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t", dtype=str)
    except Refusal:
        raise
    except Exception as exc:
        raise Refusal(code, f"cannot parse {path.name}: {exc}") from exc


def read_tx2gene(path: Path) -> tuple[pd.DataFrame, dict, pd.Series | None]:
    """Read and validate a transcript-to-gene table.

    Expected header: ``transcript_id`` and ``gene_id``, optional
    ``gene_symbol``. Transcript ids must be unique.
    """
    table = _read_tsv(path, "TX2GENE_INVALID")
    table.columns = [str(c).strip() for c in table.columns]
    for required in ("transcript_id", "gene_id"):
        if required not in table.columns:
            raise Refusal(
                "TX2GENE_INVALID",
                f"tx2gene must contain a {required} column",
                {"columns": list(table.columns)[:10]})
    table["transcript_id"] = table["transcript_id"].astype(str).str.strip()
    table["gene_id"] = table["gene_id"].astype(str).str.strip()
    if table["transcript_id"].eq("").any() or table["gene_id"].eq("").any():
        raise Refusal("TX2GENE_INVALID", "blank transcript or gene id")
    if table["transcript_id"].duplicated().any():
        dupes = sorted(table["transcript_id"][table["transcript_id"].duplicated()].unique())
        raise Refusal("TX2GENE_INVALID", "duplicate transcript ids",
                      {"duplicates": dupes[:20]})
    symbols = None
    if "gene_symbol" in table.columns:
        table["gene_symbol"] = table["gene_symbol"].astype(str).str.strip()
        symbols = (table[table["gene_symbol"].ne("") & table["gene_symbol"].ne("nan")]
                   .drop_duplicates("gene_id")
                   .set_index("gene_id")["gene_symbol"])
    report = {"n_transcripts": int(len(table)),
              "n_genes": int(table["gene_id"].nunique()),
              "has_symbols": symbols is not None}
    return table[["transcript_id", "gene_id"]], report, symbols


def _find_quant_file(sample_dir: Path) -> Path | None:
    for name in QUANT_FILES:
        candidate = sample_dir / name
        if candidate.is_file():
            return candidate
    return None


def build_counts(quant_dir: Path, tx2gene_path: Path,
                 sample_ids: list[str]) -> tuple[pd.DataFrame, pd.Series | None, dict]:
    """Build integer gene counts (genes x samples) from a quant directory.

    Each sample is a subdirectory named by ``sample_id`` containing
    ``quant.sf`` or ``abundance.tsv``. Sample identity comes from the
    directory name, so any directory order is acceptable.
    """
    if not quant_dir.is_dir():
        raise Refusal("QUANT_INVALID", "quant directory does not exist")
    tx2gene, tx_report, symbols = read_tx2gene(tx2gene_path)

    found: dict[str, Path] = {}
    for entry in sorted(quant_dir.iterdir()):
        if not entry.is_dir():
            continue
        quant_file = _find_quant_file(entry)
        if quant_file is not None:
            found[entry.name] = quant_file
    missing = sorted(set(sample_ids) - set(found))
    extra = sorted(set(found) - set(sample_ids))
    if missing or extra:
        raise Refusal(
            "SHEET_MISMATCH",
            "quant samples and sheet samples differ",
            {"missing_quant": missing, "missing_in_sheet": extra})

    tx_lookup = tx2gene.set_index("transcript_id")["gene_id"]
    columns: dict[str, pd.Series] = {}
    per_sample = []
    for sample_id in sample_ids:
        path = found[sample_id]
        frame = _read_tsv(path, "QUANT_INVALID")
        frame.columns = [str(c).strip() for c in frame.columns]
        transcript_col = next((c for c in TRANSCRIPT_COLUMNS if c in frame.columns),
                              frame.columns[0])
        count_col = next((c for c in COUNT_COLUMNS if c in frame.columns), None)
        if count_col is None:
            raise Refusal(
                "QUANT_INVALID",
                f"{path.name} has no recognised count column",
                {"columns": list(frame.columns)[:10], "sample": sample_id})
        transcripts = frame[transcript_col].astype(str).str.strip()
        values = pd.to_numeric(frame[count_col], errors="coerce")
        if values.isna().any():
            raise Refusal("QUANT_INVALID", "quant file has non-numeric counts",
                          {"sample": sample_id})
        if (values < 0).any():
            raise Refusal("QUANT_INVALID", "quant file has negative counts",
                          {"sample": sample_id})
        genes = transcripts.map(tx_lookup)
        mapped = genes.notna()
        unmapped_reads = float(values[~mapped].sum())
        series = (pd.Series(values[mapped].to_numpy(dtype="float64"),
                            index=genes[mapped].astype(str))
                  .groupby(level=0, sort=False).sum())
        columns[sample_id] = series
        per_sample.append({
            "sample_id": sample_id,
            "file": path.name,
            "file_sha256": _file_sha(path),
            "n_transcripts": int(len(frame)),
            "n_unmapped_transcripts": int((~mapped).sum()),
            "unmapped_est_counts": round(unmapped_reads, 3),
            "total_est_counts": round(float(values.sum()), 3),
        })

    matrix = pd.DataFrame(columns).fillna(0.0)
    rounded = np.floor(matrix.to_numpy(dtype="float64") + 0.5)
    counts = pd.DataFrame(rounded.astype("int64"), index=matrix.index,
                          columns=matrix.columns)
    report = {
        "transform": "summed_est_counts_rounded_half_up",
        "tx2gene_sha256": _file_sha(tx2gene_path),
        "tx2gene": tx_report,
        "samples": per_sample,
        "n_genes": int(counts.shape[0]),
        "n_samples": int(counts.shape[1]),
        "unrounded_total": round(float(matrix.to_numpy().sum()), 3),
        "rounded_total": int(counts.to_numpy().sum()),
        "unrounded_sha256": _digest_floats(matrix),
    }
    return counts.T, symbols, report


def _file_sha(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_floats(frame: pd.DataFrame) -> str:
    """Stable digest of the unrounded matrix (labels and raw float bytes)."""
    import hashlib
    digest = hashlib.sha256()
    digest.update("\t".join(str(c) for c in frame.columns).encode())
    digest.update(b"\n")
    digest.update("\t".join(str(g) for g in frame.index).encode())
    digest.update(b"\n")
    digest.update(np.ascontiguousarray(frame.to_numpy(dtype="float64")).tobytes())
    return digest.hexdigest()
