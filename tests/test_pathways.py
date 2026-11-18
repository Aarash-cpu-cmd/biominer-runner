"""GSEA and ORA local tests: determinism, provenance, and an independent ORA."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom

RUNNER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNNER_DIR))

from bm_runner import pathways as P  # noqa: E402

N_GENES = 300
UP = [f"G{i:04d}" for i in range(12)]
DOWN = [f"G{i:04d}" for i in range(12, 20)]


def _symbol(gene: str) -> str:
    return f"SYM-{gene}"


def _write_library(root: Path, *, use_symbols: bool = True) -> Path:
    rng = np.random.default_rng(3)
    terms = {
        "UP_PATH": UP + [f"G{i:04d}" for i in range(200, 208)],
        "DOWN_PATH": DOWN + [f"G{i:04d}" for i in range(208, 220)],
    }
    used = set(UP) | set(DOWN)
    for index in range(22):
        genes = [f"G{i:04d}" for i in rng.choice(
            [i for i in range(20, N_GENES) if f"G{i:04d}" not in used],
            30, replace=False)]
        terms[f"NOISE_{index:02d}"] = genes
    path = root / "library.gmt"
    lines = []
    for term, genes in terms.items():
        labels = [(g if not use_symbols else _symbol(g)).upper() for g in genes]
        lines.append("\t".join([term, "synthetic", *labels]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "library": "synthetic_v1", "version": "test", "n_terms": len(terms),
        "sha256": P._file_sha(path)}), encoding="utf-8")
    return path


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(5)
    rows = []
    for index in range(N_GENES):
        gene = f"G{index:04d}"
        if gene in UP:
            lfc, stat, padj = 2.0, 3.0 + rng.normal(0, 0.1), 0.001
        elif gene in DOWN:
            lfc, stat, padj = -2.0, -3.0 + rng.normal(0, 0.1), 0.001
        else:
            lfc, stat, padj = 0.0, rng.normal(0, 0.3), 0.5
        rows.append({"gene": gene, "baseMean": 100.0,
                     "log2FoldChange": lfc, "lfcSE": 0.5, "stat": stat,
                     "pvalue": 0.01 if abs(lfc) == 2.0 else 0.4,
                     "padj": padj})
    de_table = root / "de_table.tsv"
    pd.DataFrame(rows).to_csv(de_table, sep="\t", index=False)
    symbols = root / "gene_symbols.tsv"
    pd.DataFrame({"gene_id": [f"G{i:04d}" for i in range(N_GENES)],
                  "gene_symbol": [_symbol(f"G{i:04d}") for i in range(N_GENES)]
                  }).to_csv(symbols, sep="\t", index=False)
    manifest = root / "manifest.json"
    return de_table, symbols, manifest


def test_gsea_is_deterministic_and_records_provenance(tmp_path):
    library = _write_library(tmp_path)
    de_table, symbols, manifest = _write_inputs(tmp_path)
    first = P.run_gsea(de_table, library, tmp_path / "a",
                       gene_symbols_path=symbols, manifest_path=manifest,
                       permutations=100)
    second = P.run_gsea(de_table, library, tmp_path / "b",
                        gene_symbols_path=symbols, manifest_path=manifest,
                        permutations=100)

    assert first["schema"] == "bm_local_pathway_v1"
    assert first["result_kind"] == "gsea"
    assert first["engine"] == "gseapy"
    assert first["params"]["permutation_num"] == 100
    assert first["params"]["seed"] == 42
    assert first["output"]["pathway_table_sha256"] == \
        second["output"]["pathway_table_sha256"]
    assert (tmp_path / "a" / "ranking.tsv").is_file()
    config = json.loads((tmp_path / "a" / "run_config.json").read_text())
    assert config["rank_column"] == "stat"
    assert config["gene_set_sha256"] == P._file_sha(library)

    table = pd.read_csv(tmp_path / "a" / "gsea_results.tsv", sep="\t")
    assert "Term" in table.columns
    up = table.set_index("Term").loc["UP_PATH"]
    assert float(up["NES"]) > 0


def test_gsea_defaults_are_the_frozen_values():
    assert P.PERMUTATIONS == 1000
    assert P.SEED == 42
    assert P.MIN_SIZE == 15
    assert P.MAX_SIZE == 500
    assert P.GSEA_ENGINE_VERSION == "1.3.0"


def test_gene_set_manifest_mismatch_is_refused(tmp_path):
    library = _write_library(tmp_path)
    de_table, symbols, manifest = _write_inputs(tmp_path)
    manifest.write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(P.Refusal) as exc:
        P.run_ora(de_table, library, tmp_path / "out",
                  gene_symbols_path=symbols, manifest_path=manifest)
    assert exc.value.code == "GENE_SET_DIGEST_MISMATCH"


def test_ora_matches_independent_hypergeometric(tmp_path):
    library = _write_library(tmp_path)
    de_table, symbols, manifest = _write_inputs(tmp_path)
    prov = P.run_ora(de_table, library, tmp_path / "out",
                     gene_symbols_path=symbols, manifest_path=manifest)

    table = pd.read_csv(tmp_path / "out" / "ora_results.tsv", sep="\t")
    up = table[(table["direction"] == "up") & (table["term"] == "UP_PATH")].iloc[0]
    down = table[(table["direction"] == "down") & (table["term"] == "DOWN_PATH")].iloc[0]
    assert up["padj"] < 0.05
    assert down["padj"] < 0.05

    M = int(up["background_size"])
    n = int(up["set_size"])
    N = int(up["query_size"])
    k = int(up["overlap"])
    independent = float(hypergeom.sf(k - 1, M, n, N))
    assert up["pvalue"] == pytest.approx(independent, abs=1e-12)
    assert prov["params"]["correction"] == "benjamini_hochberg"
    assert prov["counts"]["up"]["terms_tested"] > 0
    assert (tmp_path / "out" / "ora_provenance.json").is_file()


def test_empty_query_is_a_legal_ora_result(tmp_path):
    library = _write_library(tmp_path)
    de_table, symbols, _manifest = _write_inputs(tmp_path)
    frame = pd.read_csv(de_table, sep="\t")
    frame["padj"] = 0.5
    frame["log2FoldChange"] = 0.0
    frame.to_csv(de_table, sep="\t", index=False)
    prov = P.run_ora(de_table, library, tmp_path / "out",
                     gene_symbols_path=symbols)
    table = pd.read_csv(tmp_path / "out" / "ora_results.tsv", sep="\t")
    assert len(table) == 0
    assert prov["counts"]["up"]["terms_tested"] == 0
    assert prov["output"]["n_terms"] == 0


def test_mouse_is_refused_in_v1(tmp_path):
    library = _write_library(tmp_path)
    de_table, symbols, _manifest = _write_inputs(tmp_path)
    with pytest.raises(P.Refusal) as exc:
        P.run_gsea(de_table, library, tmp_path / "out",
                   gene_symbols_path=symbols, species="mouse", permutations=50)
    assert exc.value.code == "SPECIES_UNSUPPORTED"


def test_cli_gsea_and_ora_dispatch(tmp_path):
    """The packaged entry point must dispatch gsea/ora without run-only flags."""
    from bm_runner.__main__ import main

    library = _write_library(tmp_path)
    de_table, symbols, _manifest = _write_inputs(tmp_path)
    common = ["--de-table", str(de_table), "--gene-sets", str(library),
              "--gene-symbols", str(symbols),
              "--gene-set-manifest", str(tmp_path / "manifest.json")]
    assert main(["gsea", *common, "--out", str(tmp_path / "cli_gsea")]) == 0
    assert main(["ora", *common, "--out", str(tmp_path / "cli_ora")]) == 0
    assert (tmp_path / "cli_gsea" / "gsea_results.tsv").is_file()
    assert (tmp_path / "cli_ora" / "ora_results.tsv").is_file()


def _write_gmt(root: Path, terms: dict[str, list[str]]) -> Path:
    path = root / "custom.gmt"
    path.write_text("\n".join(
        "\t".join([name, "synthetic", *genes]) for name, genes in terms.items()
    ) + "\n", encoding="utf-8")
    return path


def test_gsea_empty_ranking_is_refused(tmp_path):
    """All-NaN stat leaves an empty ranking; refuse cleanly, do not crash."""
    library = _write_library(tmp_path)
    de_table, _symbols, _manifest = _write_inputs(tmp_path)
    frame = pd.read_csv(de_table, sep="\t")
    frame["stat"] = np.nan
    frame.to_csv(de_table, sep="\t", index=False)
    with pytest.raises(P.Refusal) as exc:
        P.run_gsea(de_table, library, tmp_path / "out", permutations=50)
    assert exc.value.code == "GSEA_EMPTY_RANKING"


def test_gsea_zero_gmt_overlap_is_refused(tmp_path):
    """Ranking genes disjoint from every gene set -> clean refusal."""
    de_table, _symbols, _manifest = _write_inputs(tmp_path)
    library = _write_gmt(tmp_path, {
        "TERM_A": [f"X{i:04d}" for i in range(20)],
        "TERM_B": [f"X{i:04d}" for i in range(20, 40)],
    })
    with pytest.raises(P.Refusal) as exc:
        P.run_gsea(de_table, library, tmp_path / "out", permutations=50)
    assert exc.value.code == "GSEA_NO_TESTABLE_TERMS"


def test_gsea_all_terms_below_min_size_is_refused(tmp_path):
    """Every gene set smaller than min_size -> clean refusal, not a crash."""
    de_table, _symbols, _manifest = _write_inputs(tmp_path)
    library = _write_gmt(tmp_path, {
        "SMALL_A": [f"G{i:04d}" for i in range(5)],
        "SMALL_B": [f"G{i:04d}" for i in range(5, 10)],
    })
    with pytest.raises(P.Refusal) as exc:
        P.run_gsea(de_table, library, tmp_path / "out", permutations=50)
    assert exc.value.code == "GSEA_NO_TESTABLE_TERMS"


def test_ora_symbols_with_na_tokens_still_works(tmp_path):
    """Literal NA/None/NaN symbol tokens must not crash the .str pipeline."""
    library = _write_library(tmp_path)
    de_table, symbols, _manifest = _write_inputs(tmp_path)
    frame = pd.read_csv(symbols, sep="\t", dtype=str, keep_default_na=False)
    for gene, token in (("G0100", "NA"), ("G0101", "None"),
                        ("G0102", "NaN"), ("G0103", "  ")):
        frame.loc[frame["gene_id"] == gene, "gene_symbol"] = token
    frame.to_csv(symbols, sep="\t", index=False)
    prov = P.run_ora(de_table, library, tmp_path / "out",
                     gene_symbols_path=symbols)
    assert prov["result_kind"] == "ora"
    table = pd.read_csv(tmp_path / "out" / "ora_results.tsv", sep="\t")
    up = table[(table["direction"] == "up") & (table["term"] == "UP_PATH")].iloc[0]
    assert up["padj"] < 0.05


def test_ora_duplicate_gene_id_symbols_keep_first(tmp_path):
    """A duplicate gene_id in the symbol table must dedup, not raise on reindex."""
    library = _write_library(tmp_path)
    de_table, symbols, _manifest = _write_inputs(tmp_path)
    frame = pd.read_csv(symbols, sep="\t", dtype=str, keep_default_na=False)
    dup = pd.DataFrame([{"gene_id": "G0000", "gene_symbol": "DUP-SYM"}])
    pd.concat([frame, dup], ignore_index=True).to_csv(symbols, sep="\t", index=False)
    prov = P.run_ora(de_table, library, tmp_path / "out",
                     gene_symbols_path=symbols)
    assert prov["result_kind"] == "ora"
    table = pd.read_csv(tmp_path / "out" / "ora_results.tsv", sep="\t")
    up = table[(table["direction"] == "up") & (table["term"] == "UP_PATH")].iloc[0]
    # keep-first maps G0000 back to SYM-G0000, so UP_PATH stays enriched.
    assert up["padj"] < 0.05
    assert "SYM-G0000" in up["genes"]


def test_gsea_matches_the_legacy_server_module(tmp_path):
    """Same rank, GMT, seed and frozen parameters through both call sites."""
    ar = pytest.importorskip("biominer_server.core.analysis_runner")
    from biominer_server.core.models import ModuleStatus

    library = _write_library(tmp_path, use_symbols=False)
    de_table, _symbols, _manifest = _write_inputs(tmp_path)
    ours = P.run_gsea(de_table, library, tmp_path / "ours", permutations=1000)
    assert ours["params"]["permutation_num"] == 1000

    class Store:
        def __init__(self, root):
            self.base_dir = root
            self.root = root
            self.runs = root / "runs"

        def artifact_path(self, gse_id, run_id, rel):
            return self.runs / gse_id / run_id / rel

        def save_artifact(self, gse_id, run_id, rel, content):
            path = self.artifact_path(gse_id, run_id, rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            import hashlib
            return hashlib.sha256(content.encode("utf-8")).hexdigest()

    store = Store(tmp_path)
    de_csv = store.artifact_path("GSE_TEST", "run_test", "de/de_table.csv")
    de_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(de_table, sep="\t").to_csv(de_csv, index=False)

    from types import SimpleNamespace
    manifest = SimpleNamespace(
        gse_id="GSE_TEST", run_id="run_test",
        execution_plan={"gsea_gene_sets": [str(library)]},
        de_parameters=SimpleNamespace(seed=42),
    )
    result = ar._run_gsea_module(manifest, store)
    assert result.status == ModuleStatus.COMPLETED, result.error

    legacy = pd.read_csv(
        store.artifact_path("GSE_TEST", "run_test", "gsea/gsea_results.csv"))
    mine = pd.read_csv(tmp_path / "ours" / "gsea_results.tsv", sep="\t")
    assert set(mine["Term"]) == set(legacy["Term"])
    for column in ("ES", "NES", "NOM p-val", "FDR q-val"):
        a = pd.to_numeric(mine.set_index("Term")[column], errors="coerce")
        b = pd.to_numeric(legacy.set_index("Term")[column], errors="coerce")
        b = b.reindex(a.index)
        both_nan = a.isna() & b.isna()
        assert bool((both_nan | np.isclose(a.fillna(0), b.fillna(0),
                                           rtol=0, atol=1e-6)).all()), column
