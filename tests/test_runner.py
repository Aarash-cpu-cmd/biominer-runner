"""Local runner tests: real pydeseq2 on synthetic counts, plus refusals.

The reference environment is the repo venv (pydeseq2 0.5.4). One 2000 x 6 run
takes a few seconds, so the suite stays small on purpose; larger sizes belong
to benchmark C.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RUNNER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNNER_DIR))

from bm_runner import runner as R  # noqa: E402
from bm_runner.__main__ import main as cli_main  # noqa: E402

N_GENES = 2000
SAMPLES = [f"S{i}" for i in range(6)]
SHEET = pd.DataFrame({"sample_id": SAMPLES,
                      "condition": ["control"] * 3 + ["treated"] * 3})


def make_counts(effect: int = 180, down: int = 90, seed: int = 7,
                n_genes: int = N_GENES) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.negative_binomial(4, 0.3, size=(n_genes, 1)) + 6
    counts = np.repeat(base, len(SAMPLES), axis=1)
    counts = counts + rng.integers(0, 3, size=counts.shape)
    if effect:
        counts[:effect, 3:] = (counts[:effect, 3:] * 5) + 20
    if down:
        counts[effect:effect + down, 3:] = np.maximum(
            1, counts[effect:effect + down, 3:] // 4)
    return pd.DataFrame(counts.astype("int64"), index=[f"G{i:04d}" for i in range(n_genes)],
                        columns=SAMPLES)


def write_inputs(tmp: Path, counts: pd.DataFrame, sheet: pd.DataFrame = SHEET):
    counts_path = tmp / "counts.tsv"
    sheet_path = tmp / "sheet.tsv"
    counts.to_csv(counts_path, sep="\t")
    sheet.to_csv(sheet_path, sep="\t", index=False)
    return counts_path, sheet_path


def test_runner_produces_table_and_provenance(tmp_path):
    counts_path, sheet_path = write_inputs(tmp_path, make_counts())
    out = tmp_path / "out"
    prov = R.run(counts_path, sheet_path, "treated", "control", out)

    assert (out / "de_table.tsv").is_file()
    assert (out / "provenance.json").is_file()
    table = pd.read_csv(out / "de_table.tsv", sep="\t")
    assert list(table.columns) == ["gene", "baseMean", "log2FoldChange",
                                   "lfcSE", "stat", "pvalue", "padj"]
    after = prov["gene_filter"]["n_genes_after"]
    assert len(table) == after
    assert 0 < after <= N_GENES
    assert prov["gene_filter"]["min_counts"] == 10
    assert prov["gene_filter"]["min_samples"] == 2

    assert prov["schema"] == "bm_local_result_v1"
    assert prov["execution_origin"] == "local"
    assert prov["validation_status"] == "schema_validated"
    assert prov["engine"] == "pydeseq2" and prov["engine_version"] == "0.5.4"
    assert prov["contrast"] == {"numerator": "treated", "denominator": "control",
                                "direction": "treated_vs_control"}
    assert prov["input"]["n_genes"] == N_GENES
    assert prov["input"]["n_samples"] == 6
    assert len(prov["input"]["counts_sha256"]) == 64
    assert prov["output"]["de_table_sha256"] == R._file_sha(out / "de_table.tsv")
    assert prov["output"]["n_up"] > 50 and prov["output"]["n_down"] > 20
    assert prov["metrics"]["exit_code"] == 0
    assert prov["metrics"]["elapsed_s"] > 0
    effective = prov["engine_params_effective"]
    # pydeseq2 keeps dds.fit_type at the requested value and records a real
    # fallback only in uns["disp_function_type"], so both are reported.
    assert effective["fit_type_requested"] == "parametric"
    assert effective["fit_type_effective"] in ("parametric", "mean")
    assert effective["dispersion_trend_fallback"] == (
        effective["fit_type_effective"] != effective["fit_type_requested"])
    assert effective["n_samples_used"] == 6
    assert effective["n_genes_used"] == after
    assert effective["size_factors_fit_type"] == "poscounts"
    assert prov["params"]["size_factors_fit_type"] == "poscounts"
    assert prov["params"]["low_memory"] is True
    assert prov["params"]["n_cpus"] == 1
    assert prov["params"]["classification"]["lfc_threshold"] == 1.0
    assert prov["classification"]["n_effect_candidates"] <= after

    lfc = table.set_index("gene")["log2FoldChange"]
    assert lfc["G0000"] > 1.0          # engineered up in treated
    assert lfc["G0190"] < -1.0         # engineered down in treated
    assert abs(lfc["G1500"]) < 1.0    # untouched


def test_runner_is_deterministic(tmp_path):
    counts_path, sheet_path = write_inputs(tmp_path, make_counts(n_genes=400,
                                                                effect=40, down=20))
    first = R.run(counts_path, sheet_path, "treated", "control", tmp_path / "a")
    second = R.run(counts_path, sheet_path, "treated", "control", tmp_path / "b")
    assert first["output"]["de_table_sha256"] == second["output"]["de_table_sha256"]
    assert first["input"]["counts_sha256"] == second["input"]["counts_sha256"]


def _negative(df):
    out = df.copy()
    out.iloc[1, 0] = -5
    return out


@pytest.mark.parametrize("mutate,code", [
    (_negative, "COUNTS_INVALID"),
    (lambda df: df.astype(float) + 0.5, "COUNTS_NOT_INTEGER"),
])
def test_runner_refuses_bad_counts(tmp_path, mutate, code):
    counts = mutate(make_counts(n_genes=50, effect=5, down=5))
    counts_path, sheet_path = write_inputs(tmp_path, counts)
    with pytest.raises(R.Refusal) as exc:
        R.run(counts_path, sheet_path, "treated", "control", tmp_path / "out")
    assert exc.value.code == code


def test_runner_refuses_duplicate_genes_and_sheet_mismatch(tmp_path):
    counts = make_counts(n_genes=30, effect=0, down=0)
    counts.index = [f"G{i}" for i in range(len(counts) - 1)] + ["G0"]
    counts_path, sheet_path = write_inputs(tmp_path, counts)
    with pytest.raises(R.Refusal) as exc:
        R.run(counts_path, sheet_path, "treated", "control", tmp_path / "o1")
    assert exc.value.code == "COUNTS_INVALID"

    counts2 = make_counts(n_genes=30, effect=0, down=0)
    counts2 = counts2.drop(columns=["S5"])
    counts_path2, sheet_path2 = write_inputs(tmp_path, counts2)
    with pytest.raises(R.Refusal) as exc:
        R.run(counts_path2, sheet_path2, "treated", "control", tmp_path / "o2")
    assert exc.value.code == "SHEET_MISMATCH"
    assert exc.value.details["missing_in_counts"] == ["S5"]


def test_runner_refuses_shuffled_sample_order(tmp_path):
    """Same sample ids in a different order is refused, not silently reordered."""
    counts = make_counts(n_genes=30, effect=0, down=0)
    permuted = ["S4", "S0", "S5", "S2", "S1", "S3"]
    shuffled_counts = counts[permuted]
    (tmp_path / "shuffled").mkdir()
    counts_path, sheet_path = write_inputs(tmp_path / "shuffled",
                                           shuffled_counts, SHEET)
    with pytest.raises(R.Refusal) as exc:
        R.run(counts_path, sheet_path, "treated", "control", tmp_path / "o1")
    assert exc.value.code == "SHEET_ORDER_MISMATCH"
    assert exc.value.details["sheet_order"] == ["S0", "S1", "S2", "S3", "S4", "S5"]
    assert exc.value.details["counts_order"] == permuted

    # A permutation applied to both files keeps them in agreement, so it runs:
    # the check is order equality, not a fixed sample naming order.
    both_sheet = SHEET.iloc[[4, 0, 5, 2, 1, 3]].reset_index(drop=True)
    both_counts = counts[permuted]
    (tmp_path / "both").mkdir()
    counts_path2, sheet_path2 = write_inputs(tmp_path / "both", both_counts, both_sheet)
    prov = R.run(counts_path2, sheet_path2, "treated", "control", tmp_path / "o2")
    assert prov["input"]["sample_ids"] == permuted


def test_peak_rss_is_measured(tmp_path):
    """provenance must carry a real peak, not null, on Windows."""
    counts_path, sheet_path = write_inputs(
        tmp_path, make_counts(n_genes=60, effect=6, down=6))
    prov = R.run(counts_path, sheet_path, "treated", "control", tmp_path / "out")
    metrics = prov["metrics"]
    assert isinstance(metrics["peak_rss_mb"], (int, float))
    assert metrics["peak_rss_mb"] > 0
    if sys.platform == "win32":
        assert metrics["peak_rss_note"] == "sampled via psutil"
        assert metrics["peak_rss_mb"] >= 40.0


def test_runner_refuses_unsupported_designs(tmp_path):
    counts = make_counts(n_genes=30, effect=0, down=0)
    counts_path, _ = write_inputs(tmp_path, counts)

    three = SHEET.copy()
    three.loc[5, "condition"] = "third"
    sheet_path = tmp_path / "three.tsv"
    three.to_csv(sheet_path, sep="\t", index=False)
    with pytest.raises(R.Refusal) as exc:
        R.run(counts_path, sheet_path, "treated", "control", tmp_path / "o1")
    assert exc.value.code == "DESIGN_UNSUPPORTED"

    paired = SHEET.copy()
    paired["subject_id"] = ["a", "b", "c", "a", "b", "c"]
    sheet_path = tmp_path / "paired.tsv"
    paired.to_csv(sheet_path, sep="\t", index=False)
    with pytest.raises(R.Refusal) as exc:
        R.run(counts_path, sheet_path, "treated", "control", tmp_path / "o2")
    assert "paired" in exc.value.message

    batched = SHEET.copy()
    batched["batch"] = ["b1", "b2", "b1", "b2", "b1", "b2"]
    sheet_path = tmp_path / "batch.tsv"
    batched.to_csv(sheet_path, sep="\t", index=False)
    with pytest.raises(R.Refusal) as exc:
        R.run(counts_path, sheet_path, "treated", "control", tmp_path / "o3")
    assert "batch" in exc.value.message


def test_runner_refuses_thin_groups(tmp_path):
    thin = SHEET.copy()
    thin.loc[thin["condition"] == "treated", "condition"] = "control"
    thin.loc[5, "condition"] = "treated"
    counts = make_counts(n_genes=30, effect=0, down=0)
    counts_path, sheet_path = write_inputs(tmp_path, counts, thin)
    with pytest.raises(R.Refusal) as exc:
        R.run(counts_path, sheet_path, "treated", "control", tmp_path / "out")
    assert "at least two" in exc.value.message


def test_cli_exit_codes(tmp_path):
    counts_path, sheet_path = write_inputs(tmp_path, make_counts(n_genes=60,
                                                                effect=6, down=6))
    rc = cli_main(["run", "--counts", str(counts_path), "--sheet", str(sheet_path),
                   "--numerator", "treated", "--denominator", "control",
                   "--out", str(tmp_path / "ok")])
    assert rc == 0
    bad = make_counts(n_genes=40, effect=0, down=0).astype(float) + 0.25
    counts_bad, sheet_bad = write_inputs(tmp_path / "bad" if False else tmp_path, bad)
    rc = cli_main(["run", "--counts", str(counts_bad), "--sheet", str(sheet_bad),
                   "--numerator", "treated", "--denominator", "control",
                   "--out", str(tmp_path / "badout")])
    assert rc == 2

    out = subprocess.run([sys.executable, "-m", "bm_runner", "self-check"],
                         cwd=str(RUNNER_DIR), capture_output=True, text=True)
    assert out.returncode == 0
    doc = json.loads(out.stdout)
    assert doc["runner_version"] == R.RUNNER_VERSION
    assert doc["dependencies"]["pydeseq2"] == "0.5.4"


def test_run_emits_figures_and_report(tmp_path):
    counts_path, sheet_path = write_inputs(
        tmp_path, make_counts(n_genes=300, effect=30, down=15))
    out = tmp_path / "out"
    prov = R.run(counts_path, sheet_path, "treated", "control", out)
    for name in ("volcano.svg", "pca.svg", "heatmap.svg"):
        art = out / "figures" / name
        assert art.is_file()
        text = art.read_text(encoding="utf-8")
        assert text.startswith("<svg") and text.rstrip().endswith("</svg>")
    figures = prov["output"]["figures"]
    assert set(figures) == {"figures/volcano.svg", "figures/pca.svg",
                            "figures/heatmap.svg"}
    for name, digest in figures.items():
        assert digest == R._file_sha(out / name)
    report = out / "report.md"
    assert report.is_file()
    assert prov["output"]["report_sha256"] == R._file_sha(report)


def test_figures_and_report_are_byte_deterministic(tmp_path):
    counts_path, sheet_path = write_inputs(
        tmp_path, make_counts(n_genes=200, effect=20, down=10))
    first = R.run(counts_path, sheet_path, "treated", "control",
                  tmp_path / "a")
    second = R.run(counts_path, sheet_path, "treated", "control",
                   tmp_path / "b")
    assert first["output"]["figures"] == second["output"]["figures"]
    assert first["output"]["report_sha256"] == \
        second["output"]["report_sha256"]
    assert first["output"]["de_table_sha256"] == \
        second["output"]["de_table_sha256"]


def test_report_surfaces_dispersion_fallback(tmp_path):
    """The report must state a fallback when one happened and not otherwise."""
    counts_path, sheet_path = write_inputs(
        tmp_path, make_counts(n_genes=200, effect=20, down=10))
    out = tmp_path / "out"
    prov = R.run(counts_path, sheet_path, "treated", "control", out)
    text = (out / "report.md").read_text(encoding="utf-8")
    flagged = prov["engine_params_effective"]["dispersion_trend_fallback"]
    assert ("Dispersion trend fallback" in text) == flagged
    assert "execution_origin=local" in text
    assert "not a scientific qualification" in text


def test_validate_subcommand(tmp_path, capsys):
    counts_path, sheet_path = write_inputs(
        tmp_path, make_counts(n_genes=80, effect=8, down=8))
    capsys.readouterr()
    rc = cli_main(["validate", "--counts", str(counts_path),
                   "--sheet", str(sheet_path),
                   "--numerator", "treated", "--denominator", "control"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 0 and doc["ok"] is True
    assert doc["counts"]["n_genes"] == 80
    assert doc["design"]["sizes"] == {"control": 3, "treated": 3}
    assert doc["n_genes_testable"] >= 1
    # read-only: nothing besides the two input files may appear
    assert sorted(p.name for p in tmp_path.iterdir()) == \
        ["counts.tsv", "sheet.tsv"]

    rc = cli_main(["validate", "--counts", str(counts_path),
                   "--sheet", str(sheet_path),
                   "--numerator", "bogus", "--denominator", "control"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 2 and doc["ok"] is False
    assert doc["code"] == "DESIGN_UNSUPPORTED"


def test_validate_refuses_bad_counts(tmp_path, capsys):
    bad = make_counts(n_genes=40, effect=0, down=0).astype(float) + 0.25
    counts_path, sheet_path = write_inputs(tmp_path, bad)
    capsys.readouterr()
    rc = cli_main(["validate", "--counts", str(counts_path),
                   "--sheet", str(sheet_path),
                   "--numerator", "treated", "--denominator", "control"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 2 and doc["code"] == "COUNTS_NOT_INTEGER"


def test_validate_refusal_matrix(tmp_path, capsys):
    """Every malformed input must surface the run path's refusal code."""
    counts_path, sheet_path = write_inputs(
        tmp_path, make_counts(n_genes=40, effect=0, down=0))

    # missing sample columns: sheet asks for a column the matrix lacks
    bad_sheet = tmp_path / "sheet2.tsv"
    pd.DataFrame({"sample_id": ["S1", "S2", "S3", "S9"],
                  "condition": ["control", "control", "treated",
                                "treated"]}).to_csv(
        bad_sheet, sep="\t", index=False)
    rc = cli_main(["validate", "--counts", str(counts_path),
                   "--sheet", str(bad_sheet),
                   "--numerator", "treated", "--denominator", "control"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 2 and doc["ok"] is False

    # empty counts file
    empty = tmp_path / "empty.tsv"
    empty.write_text("")
    rc = cli_main(["validate", "--counts", str(empty),
                   "--sheet", str(sheet_path),
                   "--numerator", "treated", "--denominator", "control"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 2 and doc["ok"] is False

    # control character in a column name
    ctrl = tmp_path / "ctrl.tsv"
    cols = ["gene"] + [f"S{i}" for i in range(1, 7)]
    lines = ["\t".join(cols).replace("S4", "S\x004")]
    for g in range(40):
        lines.append("\t".join([f"G{g}"] + ["100"] * 6))
    ctrl.write_text("\n".join(lines))
    rc = cli_main(["validate", "--counts", str(ctrl),
                   "--sheet", str(sheet_path),
                   "--numerator", "treated", "--denominator", "control"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 2 and doc["ok"] is False
    # and it writes nothing
    assert sorted(p.name for p in tmp_path.iterdir()) == \
        ["counts.tsv", "ctrl.tsv", "empty.tsv", "sheet.tsv", "sheet2.tsv"]
