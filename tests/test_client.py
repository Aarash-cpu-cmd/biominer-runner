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


# -- auth: cold-start regressions -------------------------------------
def test_register_accepts_201_created():
    # The production /api/auth/register returns 201 Created (not 200) with a
    # bearer token. A client that demands exactly 200 breaks cold-start signup.
    def handler(request):
        assert request.url.path.endswith("/api/auth/register")
        return httpx.Response(201, json={"token": "tok_201", "username": "u",
                                         "expires_at": 123.0})

    with _client(handler) as c:
        assert c.register("u", "p") == "tok_201"
        assert c.token == "tok_201"


def test_login_accepts_200():
    def handler(request):
        return httpx.Response(200, json={"token": "tok_200"})

    with _client(handler) as c:
        assert c.login("u", "p") == "tok_200"


def test_auth_failure_redacts_token():
    # A misclassified body carrying a token must never echo the secret.
    def handler(request):
        return httpx.Response(403, json={"token": "SUPERSECRET", "detail": "no"})

    with _client(handler) as c:
        with pytest.raises(BioMinerError) as exc:
            c.register("u", "p")
        assert exc.value.code == "AUTH_FAILED"
        assert "SUPERSECRET" not in str(exc.value)
        assert "***" in exc.value.message


def test_redact_secrets_masks_credentials():
    from biominer_client.client import _redact_secrets
    out = _redact_secrets('{"token": "abc123", "password": "pw", "user": "bob"}')
    assert "abc123" not in out and "pw" not in out and "bob" in out
    # non-JSON falls back to a regex mask
    out2 = _redact_secrets('token="leakme" trailing')
    assert "leakme" not in out2


# -- idempotency: cold-start regression -------------------------------
def _save_handler(keys):
    def handler(request):
        path = request.url.path
        if path.endswith("/upload_prepare"):
            return httpx.Response(200, json=_envelope({"ticket": {
                "ticket_id": "upt", "upload_url": "https://example.test/ul/u"}}))
        if path.endswith("/ul/u"):
            return httpx.Response(200, json={"data": {"receipt": {
                "receipt_id": "urc"}}})
        if path.endswith("/local_result_register"):
            keys.append(json.loads(request.content)["idempotency_key"])
            return httpx.Response(200, json=_envelope({"result": {
                "result_id": "lres_1"}}))
        raise AssertionError(f"unexpected {path}")
    return handler


def test_save_result_idempotency_key_is_content_stable(tmp_path):
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps({"schema": "bm_local_result_v1"}))
    de = tmp_path / "de_table.tsv"
    de.write_text("gene\tpadj\nENSG1\t0.01\n")
    contrast = {"numerator": "Dexamethasone", "denominator": "Untreated"}
    keys = []
    with _client(_save_handler(keys)) as c:
        workflow.save_result(c, "study_1",
                             {"provenance_path": prov, "de_table_path": de},
                             contrast)
        workflow.save_result(c, "study_1",
                             {"provenance_path": prov, "de_table_path": de},
                             contrast)
        # identical content -> identical key (server replays, no duplicate)
        assert keys[0] == keys[1]
        de.write_text("gene\tpadj\nENSG1\t0.02\n")  # changed result bytes
        workflow.save_result(c, "study_1",
                             {"provenance_path": prov, "de_table_path": de},
                             contrast)
        assert keys[2] != keys[0]


def test_save_result_explicit_idempotency_key_honored(tmp_path):
    prov = tmp_path / "provenance.json"
    prov.write_text("{}")
    de = tmp_path / "de_table.tsv"
    de.write_text("gene\tpadj\nENSG1\t0.01\n")
    keys = []
    with _client(_save_handler(keys)) as c:
        workflow.save_result(c, "study_1",
                             {"provenance_path": prov, "de_table_path": de},
                             {"numerator": "A", "denominator": "B"},
                             idempotency_key="my-fixed-key")
    assert keys == ["my-fixed-key"]


def test_upload_receipt_cache_reuses_identical_content(tmp_path):
    # The server mints a new receipt_id per upload; a content-addressed cache
    # must reuse the receipt for identical bytes so re-registration replays.
    cache = tmp_path / "receipts.json"
    posts = []

    def handler(request):
        if request.url.path.endswith("/upload_prepare"):
            return httpx.Response(200, json=_envelope({"ticket": {
                "ticket_id": "upt", "upload_url": "https://example.test/ul/u"}}))
        if request.url.path.endswith("/ul/u"):
            posts.append(1)
            return httpx.Response(200, json={"data": {"receipt": {
                "receipt_id": "urc_fixed", "sha256": "s"}}})
        raise AssertionError(f"unexpected {request.url}")

    def mk():
        return BioMinerClient(base_url="https://example.test", token="t",
                              transport=httpx.MockTransport(handler),
                              receipt_cache=cache)

    with mk() as c1:
        r1 = c1.upload_bytes("de.tsv", b"hello")
    # a fresh session loads the persisted cache and skips the upload
    with mk() as c2:
        r2 = c2.upload_bytes("de.tsv", b"hello")
    assert r1["receipt_id"] == r2["receipt_id"] == "urc_fixed"
    assert len(posts) == 1
    # changed content triggers a new upload
    with mk() as c3:
        c3.upload_bytes("de.tsv", b"world")
    assert len(posts) == 2


# -- anonymous credential ----------------------------------------------
def test_mint_anonymous_sets_token_and_device_fp():
    import re
    seen = {}

    def handler(request):
        assert request.url.path == "/api/mcp/auth/anonymous"
        seen["fp"] = request.headers.get("X-BM-Device-Fp")
        return httpx.Response(200, json={"credential": "anon-abc123",
                                         "expires_in": 3600})

    with BioMinerClient(base_url="https://example.test", token=None,
                        transport=httpx.MockTransport(handler)) as c:
        assert c.token is None
        tok = c.mint_anonymous()
        assert tok == "anon-abc123"
        assert c.token == tok
        assert c._client.headers["Authorization"] == "Bearer anon-abc123"
    assert re.fullmatch(r"[A-Za-z0-9_-]{8,64}", seen["fp"])


def test_mint_anonymous_refusal_and_missing_credential():
    def deny(request):
        return httpx.Response(403, json={"error": "anonymous disabled"})

    with BioMinerClient(base_url="https://example.test", token=None,
                        transport=httpx.MockTransport(deny)) as c:
        with pytest.raises(BioMinerError) as exc:
            c.mint_anonymous()
        assert exc.value.code == "AUTH_FAILED"

    def empty(request):
        return httpx.Response(200, json={"ok": True})

    with BioMinerClient(base_url="https://example.test", token=None,
                        transport=httpx.MockTransport(empty)) as c:
        with pytest.raises(BioMinerError):
            c.mint_anonymous()


def test_ensure_token_reuses_existing_without_http():
    def handler(request):
        raise AssertionError("ensure_token must not call the network")

    with _client(handler) as c:
        assert c.ensure_token() == "t"


def test_default_device_fp_stable_and_server_compatible():
    import re
    from biominer_client.client import _default_device_fp
    fp = _default_device_fp()
    assert fp == _default_device_fp()
    assert re.fullmatch(r"[A-Za-z0-9_-]{8,64}", fp)


# -- condition discovery ------------------------------------------------
def test_summarize_conditions_counts_and_order(tmp_path):
    sheet = tmp_path / "sample_sheet.tsv"
    sheet.write_text(
        "sample_id\tseries_id\ttitle\tcharacteristics\tcondition\n"
        "S1\tGSE1\tt1\ttreatment: Dexamethasone,cell line: N1\t\n"
        "S2\tGSE1\tt2\ttreatment: Dexamethasone,cell line: N2\t\n"
        "S3\tGSE1\tt3\ttreatment: Untreated,cell line: N1\t\n"
        "S4\tGSE1\tt4\ttreatment: Untreated,cell line: N2\t\n",
        encoding="utf-8")
    summary = workflow.summarize_conditions({"sheet_path": sheet})
    assert summary["n_samples"] == 4
    fields = summary["condition_fields"]
    assert fields["treatment"] == {"Dexamethasone": 2, "Untreated": 2}
    assert fields["cell line"] == {"N1": 2, "N2": 2}
    assert "condition" not in fields  # empty column contributes nothing


def test_summarize_conditions_includes_populated_condition_column(tmp_path):
    sheet = tmp_path / "sample_sheet.tsv"
    sheet.write_text(
        "sample_id\tseries_id\ttitle\tcharacteristics\tcondition\n"
        "S1\tGSE1\tt1\t\tctrl\n"
        "S2\tGSE1\tt2\t\tcase\n"
        "S3\tGSE1\tt3\t\tcase\n",
        encoding="utf-8")
    fields = workflow.summarize_conditions(
        {"sheet_path": sheet})["condition_fields"]
    assert fields["condition"] == {"case": 2, "ctrl": 1}


# -- species follows the export manifest --------------------------------
def test_local_analysis_species_follows_manifest(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(workflow, "export_dataset", lambda c, g, o: {
        "export": {"organism": "Homo sapiens", "n_samples": 4},
        "counts_path": tmp_path / "c.tsv", "sheet_path": tmp_path / "s.tsv"})
    monkeypatch.setattr(workflow, "build_contrast",
                        lambda *a, **k: {"n_samples": 4, "numerator": "x",
                                         "denominator": "y"})
    monkeypatch.setattr(workflow, "run_de",
                        lambda *a, **k: {"de_table_path": tmp_path / "de.tsv"})
    monkeypatch.setattr(workflow, "fetch_gene_sets",
                        lambda *a, **k: {"go_bp": {}})

    def fake_enrich(de, gs, out, *, kind, library, species):
        captured[kind] = species
        return {"kind": kind}

    monkeypatch.setattr(workflow, "run_enrichment", fake_enrich)
    msgs = []
    workflow.local_analysis(None, "GSE1", "x", "y", tmp_path / "w",
                            register=False, progress=msgs.append)
    assert captured == {"gsea": "Homo sapiens", "ora": "Homo sapiens"}
    assert any("[3/6]" in m for m in msgs)

    captured.clear()
    workflow.local_analysis(None, "GSE1", "x", "y", tmp_path / "w2",
                            register=False, species="human")
    assert captured == {"gsea": "human", "ora": "human"}
