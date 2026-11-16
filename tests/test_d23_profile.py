"""D23 alignment, input transforms, and parity with the legacy server fit."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RUNNER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNNER_DIR))

from bm_runner import de, quant  # noqa: E402
from bm_runner import runner as R  # noqa: E402

SAMPLES = [f"S{i}" for i in range(6)]
SHEET = pd.DataFrame({"sample_id": SAMPLES,
                      "condition": ["control"] * 3 + ["treated"] * 3})


def make_counts(effect: int = 24, down: int = 12, seed: int = 11,
                n_genes: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.negative_binomial(4, 0.3, size=(n_genes, 1)) + 6
    counts = np.repeat(base, len(SAMPLES), axis=1)
    counts = counts + rng.integers(0, 3, size=counts.shape)
    if effect:
        counts[:effect, 3:] = (counts[:effect, 3:] * 5) + 20
    if down:
        counts[effect:effect + down, 3:] = np.maximum(
            1, counts[effect:effect + down, 3:] // 4)
    return pd.DataFrame(counts.astype("int64"),
                        index=[f"G{i:04d}" for i in range(n_genes)],
                        columns=SAMPLES)


def write_inputs(tmp: Path, counts: pd.DataFrame,
                 sheet: pd.DataFrame = SHEET) -> tuple[Path, Path]:
    counts_path = tmp / "counts.tsv"
    sheet_path = tmp / "sheet.tsv"
    counts.to_csv(counts_path, sep="\t")
    sheet.to_csv(sheet_path, sep="\t", index=False)
    return counts_path, sheet_path


def test_prefilter_drops_low_count_genes(tmp_path):
    counts = make_counts(n_genes=120, effect=12, down=6)
    counts.iloc[:20, :] = 1  # three samples per group, count 1 everywhere
    counts_path, sheet_path = write_inputs(tmp_path, counts)
    prov = R.run(counts_path, sheet_path, "treated", "control", tmp_path / "out")

    gene_filter = prov["gene_filter"]
    expected = int(((counts >= 10).sum(axis=1) >= 2).sum())
    assert gene_filter["n_genes_before"] == 120
    assert gene_filter["n_genes_after"] == expected
    assert gene_filter["n_genes_removed"] == 120 - expected
    assert prov["engine_params_effective"]["n_genes_used"] == expected
    table = pd.read_csv(tmp_path / "out" / "de_table.tsv", sep="\t")
    assert len(table) == expected
    assert "G0000" not in set(table["gene"])


def test_no_testable_genes_is_refused(tmp_path):
    counts = pd.DataFrame(1, index=[f"G{i:04d}" for i in range(30)],
                          columns=SAMPLES)
    counts_path, sheet_path = write_inputs(tmp_path, counts)
    with pytest.raises(R.Refusal) as exc:
        R.run(counts_path, sheet_path, "treated", "control", tmp_path / "out")
    assert exc.value.code == "NO_TESTABLE_GENES"
    assert exc.value.details["n_genes_input"] == 30


def test_reference_level_is_first_even_when_alphabetically_later(tmp_path):
    """Numerator sorts first alphabetically; explicit reference wins."""
    counts = make_counts(n_genes=120, effect=12, down=0)
    sheet = SHEET.copy()
    sheet["condition"] = ["z_ctrl"] * 3 + ["a_treat"] * 3
    counts_path, sheet_path = write_inputs(tmp_path, counts, sheet)
    prov = R.run(counts_path, sheet_path, "a_treat", "z_ctrl", tmp_path / "out")

    table = pd.read_csv(tmp_path / "out" / "de_table.tsv", sep="\t").set_index("gene")
    assert table.loc["G0000", "log2FoldChange"] > 1.0
    assert prov["params"]["reference_level_first"] is True
    assert prov["classification"]["n_up"] > 0


def test_gene_symbol_column_is_kept(tmp_path):
    counts = make_counts(n_genes=80, effect=8, down=8)
    counts.insert(1, "gene_symbol", [f"SYM{i}" for i in range(len(counts))])
    counts_path, sheet_path = write_inputs(tmp_path, counts)
    prov = R.run(counts_path, sheet_path, "treated", "control", tmp_path / "out")

    assert prov["input"]["gene_symbol_source"] == "column"
    symbols = pd.read_csv(tmp_path / "out" / "gene_symbols.tsv", sep="\t")
    assert list(symbols.columns) == ["gene_id", "gene_symbol"]
    assert symbols.set_index("gene_id").loc["G0000", "gene_symbol"] == "SYM0"


def _write_quant(counts: pd.DataFrame, root: Path) -> tuple[Path, Path]:
    """Split each gene count across two transcripts and write salmon files."""
    rows = []
    for gene in counts.index:
        for suffix in ("t1", "t2"):
            rows.append({"transcript_id": f"{gene}_{suffix}", "gene_id": gene,
                         "gene_symbol": f"SYM-{gene}"})
    tx2gene = root / "tx2gene.tsv"
    pd.DataFrame(rows).to_csv(tx2gene, sep="\t", index=False)

    quant_dir = root / "quant"
    for sample in counts.columns:
        sample_dir = quant_dir / sample
        sample_dir.mkdir(parents=True)
        values = counts[sample]
        first = values // 2
        second = values - first
        frame = pd.DataFrame({
            "Name": [f"{g}_t1" for g in counts.index] + [f"{g}_t2" for g in counts.index],
            "Length": 1000,
            "EffectiveLength": 1000,
            "TPM": 1.0,
            "NumReads": list(first.astype(float)) + list(second.astype(float)),
        })
        frame.to_csv(sample_dir / "quant.sf", sep="\t", index=False)
    return quant_dir, tx2gene


def test_quant_input_roundtrips_to_the_same_fit(tmp_path):
    counts = make_counts(n_genes=120, effect=12, down=6)
    sheet_path = tmp_path / "sheet.tsv"
    SHEET.to_csv(sheet_path, sep="\t", index=False)
    quant_dir, tx2gene = _write_quant(counts, tmp_path)

    prov = R.run_quant(quant_dir, tx2gene, sheet_path, "treated", "control",
                       tmp_path / "quant-out")
    transform = prov["input"]["input_transform"]
    assert transform["type"] == "salmon_kallisto_est_counts"
    assert transform["transform"] == "summed_est_counts_rounded_half_up"
    assert transform["n_genes"] == 120
    assert prov["input"]["gene_symbol_source"] == "column"

    counts_path = tmp_path / "quant-out" / "counts.tsv"
    assert counts_path.is_file()
    first_sha = prov["output"]["de_table_sha256"]
    second = R.run(counts_path, sheet_path, "treated", "control",
                   tmp_path / "raw-out")
    assert second["output"]["de_table_sha256"] == first_sha


def test_quant_unmapped_transcripts_are_reported(tmp_path):
    counts = make_counts(n_genes=40, effect=4, down=4)
    sheet_path = tmp_path / "sheet.tsv"
    SHEET.to_csv(sheet_path, sep="\t", index=False)
    quant_dir, tx2gene = _write_quant(counts, tmp_path)
    with (quant_dir / "S0" / "quant.sf").open("a", encoding="utf-8") as handle:
        handle.write("TX_unknown\t1000\t1000\t0\t7.0\n")

    counts_frame, _symbols, report = quant.build_counts(
        quant_dir, tx2gene, SAMPLES)
    sample = next(s for s in report["samples"] if s["sample_id"] == "S0")
    assert sample["n_unmapped_transcripts"] >= 1
    assert sample["unmapped_est_counts"] >= 7.0


def test_max_gene_limit_covers_unfiltered_archs4():
    assert R.MAX_GENES >= 67186


def test_parity_with_legacy_server_fit():
    """Same matrix through the vendor profile and core.deseq._run_pydeseq2."""
    legacy = pytest.importorskip("biominer_server.core.deseq")
    os.environ["BIOMINER_DE_N_CPUS"] = "1"

    counts = make_counts(n_genes=400, effect=40, down=20)
    conditions = pd.Series(["control"] * 3 + ["treated"] * 3, index=SAMPLES)

    result = de.fit(counts.T, conditions, "treated", "control", n_cpus=1)
    ours = result["table"]

    legacy_result = legacy._run_pydeseq2(
        counts, conditions, "control", "treated",
        de.LFC_THRESHOLD, de.ALPHA, de.MIN_COUNTS, de.MIN_SAMPLES, 42)
    legacy_rows = {row["gene"]: row for row in legacy_result["de_table"]}

    assert set(ours.index) == set(legacy_rows)
    assert legacy_result["n_total"] == result["filter"]["n_genes_after"]

    legacy_frame = pd.DataFrame.from_dict(legacy_rows, orient="index")
    columns = {
        "baseMean": "baseMean",
        "log2FoldChange": "log2FoldChange_raw",
        "lfcSE": "lfcSE_raw",
        "stat": "stat",
        "pvalue": "pvalue",
        "padj": "padj",
    }
    for ours_col, legacy_col in columns.items():
        a = pd.to_numeric(ours[ours_col], errors="coerce")
        b = pd.to_numeric(legacy_frame[legacy_col], errors="coerce").reindex(a.index)
        both_nan = a.isna() & b.isna()
        same = np.isclose(a.fillna(0.0), b.fillna(0.0), rtol=0, atol=2e-6)
        assert bool((both_nan | same).all()), (
            f"{ours_col} differs; max |delta| = "
            f"{(a - b).abs().max(skipna=True)}")
