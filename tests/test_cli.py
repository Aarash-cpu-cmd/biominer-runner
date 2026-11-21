"""CLI regression tests (no network).

Cold-start bug: ``--token-file`` / ``--base-url`` were global-only arguments,
so the documented ``login --username x --token-file f`` form failed with
"unrecognized arguments". They must work both before and after the subcommand,
and an explicit global value must not be clobbered by the subparser default.
"""
from __future__ import annotations

import json

import pytest

from biominer_client import __main__ as cli


class _FakeClient:
    last = None

    def __init__(self, base_url=None, token=None, receipt_cache=None):
        self.base_url = base_url
        self.token = token
        self.receipt_cache = receipt_cache
        _FakeClient.last = self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def register(self, username, password):
        self.registered = (username, password)
        return "FAKE_TOKEN"

    def login(self, username, password):
        return "FAKE_TOKEN"

    def mint_anonymous(self):
        self.minted = True
        self.token = "ANON_TOKEN"
        return self.token


@pytest.fixture(autouse=True)
def _patch_client(monkeypatch):
    monkeypatch.setattr(cli, "BioMinerClient", _FakeClient)
    _FakeClient.last = None


def test_token_file_after_subcommand(tmp_path, capsys):
    tf = tmp_path / "tok.txt"
    rc = cli.main(["register", "--username", "u", "--password", "p",
                   "--token-file", str(tf)])
    assert rc == 0
    assert tf.read_text(encoding="utf-8").strip() == "FAKE_TOKEN"
    assert _FakeClient.last.registered == ("u", "p")


def test_token_file_before_subcommand_not_clobbered(tmp_path, capsys):
    tf = tmp_path / "tok.txt"
    rc = cli.main(["--token-file", str(tf), "register",
                   "--username", "u", "--password", "p"])
    assert rc == 0
    # the explicit global path must survive (not reset to .biominer_token)
    assert tf.read_text(encoding="utf-8").strip() == "FAKE_TOKEN"


def test_base_url_after_subcommand(tmp_path, capsys):
    tf = tmp_path / "tok.txt"
    rc = cli.main(["login", "--username", "u", "--password", "p",
                   "--base-url", "https://x.test", "--token-file", str(tf)])
    assert rc == 0
    assert _FakeClient.last.base_url == "https://x.test"


def test_default_token_file_when_omitted(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["register", "--username", "u", "--password", "p"])
    assert rc == 0
    assert (tmp_path / ".biominer_token").read_text(encoding="utf-8").strip() \
        == "FAKE_TOKEN"


# -- zero-account one-click path ----------------------------------------
def _fake_result(out):
    return {"de": {"provenance": {}},
            "contrast": {"n_samples": 4, "group_sizes": {"a": 2, "b": 2}},
            "enrichments": {}, "study_id": None, "saved": None,
            "workdir": out}


def test_keyless_run_mints_anonymous(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("BIOMINER_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_analysis(c, gse, num, den, out, **kw):
        captured.update(kw)
        captured["token"] = c.token
        return _fake_result(out)

    monkeypatch.setattr(cli.workflow, "local_analysis", fake_analysis)
    rc = cli.main(["run", "--gse", "GSE1", "--numerator", "a",
                   "--denominator", "b", "--out", str(tmp_path / "w"),
                   "--no-register"])
    assert rc == 0
    assert captured["token"] == "ANON_TOKEN"
    assert _FakeClient.last.minted
    # the ephemeral credential must NOT be persisted to the token file
    assert not (tmp_path / ".biominer_token").exists()


def test_conditions_command(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("BIOMINER_TOKEN", raising=False)
    monkeypatch.setattr(
        cli.workflow, "export_dataset",
        lambda c, g, o: {"export": {"organism": "Homo sapiens"},
                         "sheet_path": tmp_path / "s.tsv"})
    monkeypatch.setattr(
        cli.workflow, "summarize_conditions",
        lambda e: {"n_samples": 4,
                   "condition_fields": {"treatment": {"a": 2, "b": 2}}})
    rc = cli.main(["conditions", "--gse", "GSE1", "--out", str(tmp_path / "e")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["condition_fields"]["treatment"] == {"a": 2, "b": 2}
    assert out["organism"] == "Homo sapiens"
    assert "hint" in out


def test_run_species_and_condition_map_passthrough(tmp_path, monkeypatch):
    monkeypatch.delenv("BIOMINER_TOKEN", raising=False)
    captured = {}

    def fake_analysis(c, gse, num, den, out, **kw):
        captured.update(kw)
        return _fake_result(out)

    monkeypatch.setattr(cli.workflow, "local_analysis", fake_analysis)
    cm = tmp_path / "map.json"
    cm.write_text('{"S1": "a", "S2": "b"}', encoding="utf-8")
    rc = cli.main(["run", "--gse", "GSE1", "--numerator", "a",
                   "--denominator", "b", "--out", str(tmp_path / "w"),
                   "--species", "human", "--condition-map", str(cm),
                   "--no-register"])
    assert rc == 0
    assert captured["species"] == "human"
    assert captured["condition_map"] == {"S1": "a", "S2": "b"}
    assert callable(captured["progress"])


def test_run_empty_group_hint(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("BIOMINER_TOKEN", raising=False)

    def boom(*a, **k):
        raise ValueError("contrast 'a' vs 'b' has an empty group "
                         "(sizes: a=0, b=4); available labels: ['x']")

    monkeypatch.setattr(cli.workflow, "local_analysis", boom)
    rc = cli.main(["run", "--gse", "GSE1", "--numerator", "a",
                   "--denominator", "b", "--out", str(tmp_path / "w"),
                   "--no-register"])
    assert rc == 2
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INVALID_INPUT"
    assert "conditions --gse GSE1" in out["hint"]
