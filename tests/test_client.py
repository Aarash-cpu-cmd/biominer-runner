"""Unit tests for the BioMiner connector (no network, no engine).

These exercise the HTTP client and the workflow's data-shaping logic with
an httpx mock transport and tiny local fixtures. The real statistical
engine is covered by test_runner.py / test_pathways.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from biominer_client import BioMinerClient, BioMinerError, workflow


def _envelope(data=None, ok=True, error=None):
    return {"schema": "biominer_workspace_mcp_full_v1", "ok": ok,
            "data": data, "error": error, "limitations": [],
            "http_status": 200 if ok else 422}


def _client(handler):
    return BioMinerClient(base_url="https://example.test", token="t",
                          transport=httpx.MockTransport(handler))


# -- characteristics parsing ------------------------------------------
def test_parse_characteristics():
    chars = "treatment: Dexamethasone,tissue: airway smooth muscle,cell line: N61311"
    parsed = workflow._parse_characteristics(chars)
    assert parsed["treatment"] == "Dexamethasone"
    assert parsed["cell line"] == "N61311"
    assert workflow._parse_characteristics("") == {}


# -- call_tool ---------------------------------------------------------
def test_call_tool_returns_data():
    def handler(request):
        assert request.url.path.endswith("/workspace_info")
        return httpx.Response(200, json=_envelope({"tools": [1, 2, 3]}))

    with _client(handler) as c:
        assert c.call_tool("workspace_info", method="GET") == {"tools": [1, 2, 3]}


def test_call_tool_refusal_raises():
    def handler(request):
        return httpx.Response(422, json=_envelope(
            ok=False, error={"code": "INVALID_ARGUMENT", "message": "bad",
                             "details": {"fields": []}}))

    with _client(handler) as c:
        with pytest.raises(BioMinerError) as exc:
            c.call_tool("study_create", {})
        assert exc.value.code == "INVALID_ARGUMENT"


# -- download ----------------------------------------------------------
def test_download_ticket_verifies_sha256(tmp_path):
    payload = b"gene_id\tgene_symbol\tS1\nENSG1\tTSPAN6\t7\n"
    sha = hashlib.sha256(payload).hexdigest()

    def handler(request):
        assert request.url.path.endswith("/dl/abc")
        return httpx.Response(200, content=payload)

    with _client(handler) as c:
        dest = c.download_ticket(
            {"filename": "counts.tsv", "sha256": sha,
             "download_url": "https://example.test/dl/abc"}, tmp_path / "counts.tsv")
        assert dest.read_bytes() == payload


def test_download_ticket_digest_mismatch(tmp_path):
    def handler(request):
        return httpx.Response(200, content=b"corrupted")

    with _client(handler) as c:
        dest = tmp_path / "counts.tsv"
        with pytest.raises(BioMinerError) as exc:
            c.download_ticket({"filename": "counts.tsv", "sha256": "0" * 64,
                               "download_url": "https://example.test/dl/x"}, dest)
        assert exc.value.code == "DIGEST_MISMATCH"
        assert not dest.exists()


# -- upload ------------------------------------------------------------
def test_upload_bytes_returns_receipt():
    seen = {}

    def handler(request):
        if request.url.path.endswith("/upload_prepare"):
            body = json.loads(request.content)
            seen["prepare"] = body
            return httpx.Response(200, json=_envelope({"ticket": {
                "ticket_id": "upt_1", "upload_url": "https://example.test/ul/x"}}))
        if request.url.path.endswith("/ul/x"):
            seen["content_type"] = request.headers["content-type"]
            seen["bytes"] = request.content
            return httpx.Response(200, json={"data": {"receipt": {
                "receipt_id": "urc_1", "sha256": "s", "size": len(request.content)}}})
        raise AssertionError(f"unexpected {request.url}")

    with _client(handler) as c:
        receipt = c.upload_bytes("de_table.tsv", b"hello")
    assert receipt["receipt_id"] == "urc_1"
    assert seen["content_type"] == "application/octet-stream"
    assert seen["bytes"] == b"hello"
    assert seen["prepare"]["size_bytes"] == 5
    assert seen["prepare"]["sha256"] == hashlib.sha256(b"hello").hexdigest()


# -- build_contrast ----------------------------------------------------
def _make_export(tmp_path):
    sheet = pd.DataFrame({
        "sample_id": ["S1", "S2", "S3", "S4", "S5", "S6", "S7"],
        "series_id": ["GSE1"] * 7,
        "title": [f"sample {i}" for i in range(7)],
        "characteristics": [
            "treatment: Untreated,tissue: airway",
            "treatment: Untreated,tissue: airway",
            "treatment: Untreated,tissue: airway",
            "treatment: Dexamethasone,tissue: airway",
            "treatment: Dexamethasone,tissue: airway",
            "treatment: Dexamethasone,tissue: airway",
            "treatment: Albuterol,tissue: airway",
        ],
        "condition": [""] * 7,
    })
    sheet_path = tmp_path / "sample_sheet.tsv"
    sheet.to_csv(sheet_path, sep="\t", index=False)
    counts = pd.DataFrame({
        "gene_id": ["ENSG1", "ENSG2"],
        "gene_symbol": ["TSPAN6", "FKBP5"],
        **{f"S{i}": [10 * i, 20 * i] for i in range(1, 8)},
    })
    counts_path = tmp_path / "counts.tsv"
    counts.to_csv(counts_path, sep="\t", index=False)
    return {"counts_path": counts_path, "sheet_path": sheet_path}


def test_build_contrast_selects_and_orders(tmp_path):
    export = _make_export(tmp_path)
    out = tmp_path / "contrast"
    result = workflow.build_contrast(export, "Dexamethasone", "Untreated", out)
    assert result["group_sizes"] == {"Untreated": 3, "Dexamethasone": 3}
    assert result["n_samples"] == 6
    # denominator first, then numerator; Albuterol (S7) excluded
    assert result["sample_ids"] == ["S1", "S2", "S3", "S4", "S5", "S6"]

    sheet = pd.read_csv(result["sheet_path"], sep="\t", dtype=str)
    assert list(sheet["sample_id"]) == result["sample_ids"]
    assert list(sheet["condition"]) == ["Untreated"] * 3 + ["Dexamethasone"] * 3

    counts = pd.read_csv(result["counts_path"], sep="\t", dtype=str)
    # counts columns must match the sheet exactly and in order
    assert list(counts.columns) == ["gene_id", "gene_symbol"] + result["sample_ids"]
    assert "S7" not in counts.columns


def test_build_contrast_empty_group_raises(tmp_path):
    export = _make_export(tmp_path)
    with pytest.raises(ValueError, match="empty group"):
        workflow.build_contrast(export, "Dexamethasone", "DoesNotExist",
                                tmp_path / "c2")


def test_build_contrast_explicit_map(tmp_path):
    export = _make_export(tmp_path)
    cmap = {"S1": "treated", "S4": "control"}
    result = workflow.build_contrast(export, "treated", "control",
                                     tmp_path / "c3", condition_map=cmap)
    assert result["group_sizes"] == {"control": 1, "treated": 1}
    assert result["sample_ids"] == ["S4", "S1"]


# -- save_result -------------------------------------------------------
def test_save_result_uploads_and_registers(tmp_path):
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps({"schema": "bm_local_result_v1"}))
    de = tmp_path / "de_table.tsv"
    de.write_text("gene\tpadj\nENSG1\t0.01\n")
    calls = {}

    def handler(request):
        path = request.url.path
        if path.endswith("/upload_prepare"):
            body = json.loads(request.content)
            calls.setdefault("uploads", []).append(body["filename"])
            return httpx.Response(200, json=_envelope({"ticket": {
                "ticket_id": "upt", "upload_url": "https://example.test/ul/u"}}))
        if path.endswith("/ul/u"):
            return httpx.Response(200, json={"data": {"receipt": {
                "receipt_id": "urc_" + str(len(calls["uploads"]))}}})
        if path.endswith("/local_result_register"):
            calls["register"] = json.loads(request.content)
            return httpx.Response(200, json=_envelope({"result": {
                "result_id": "lres_1"}}))
        raise AssertionError(f"unexpected {path}")

    with _client(handler) as c:
        saved = workflow.save_result(
            c, "study_1", {"provenance_path": prov, "de_table_path": de},
            {"numerator": "Dexamethasone", "denominator": "Untreated"},
            idempotency_key="fixed")
    assert saved["register"]["result"]["result_id"] == "lres_1"
    assert calls["uploads"] == ["provenance.json", "de_table.tsv"]
    reg = calls["register"]
    assert reg["contrast"] == {"numerator": "Dexamethasone",
                               "denominator": "Untreated"}
    assert reg["study_id"] == "study_1"
    assert reg["idempotency_key"] == "fixed"
    assert reg["de_table_receipt_id"] and reg["provenance_receipt_id"]
