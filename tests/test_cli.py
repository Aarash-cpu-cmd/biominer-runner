"""CLI regression tests (no network).

Cold-start bug: ``--token-file`` / ``--base-url`` were global-only arguments,
so the documented ``login --username x --token-file f`` form failed with
"unrecognized arguments". They must work both before and after the subcommand,
and an explicit global value must not be clobbered by the subparser default.
"""
from __future__ import annotations

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
