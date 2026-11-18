"""Thin HTTP client for the BioMiner Workspace MCP (full-v1) REST surface.

This is the *connector* half of the local-runner workflow. It talks to a
BioMiner deployment over HTTPS, mints short-lived byte tickets, and moves
file bytes outside of JSON tool arguments. It computes no science: every
differential / enrichment result is produced locally by ``bm_runner`` and
only *registered* here.

No credentials are bundled. Supply a token via ``BIOMINER_TOKEN`` or call
:meth:`BioMinerClient.register` / :meth:`BioMinerClient.login`. The base URL
defaults to the public BioMiner endpoint and can be overridden with
``BIOMINER_BASE_URL``.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import httpx

try:  # certifi ships with httpx; fall back to the system store if absent.
    import certifi

    _DEFAULT_VERIFY: str | bool = certifi.where()
except Exception:  # pragma: no cover - certifi is a hard dependency in practice
    _DEFAULT_VERIFY = True

DEFAULT_BASE_URL = "https://biominer.hxzhi.cc.cd"
_API_PREFIX = "/api/workspace-mcp/full/v1"
_CHUNK = 1 << 20


class BioMinerError(RuntimeError):
    """A tool call returned ``ok=false`` (a structured business refusal)."""

    def __init__(self, code: str, message: str, details: Any = None,
                 http_status: int | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details
        self.http_status = http_status


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


class BioMinerClient:
    """A owner-scoped session against one BioMiner deployment.

    Parameters
    ----------
    base_url:
        Root of the deployment (no trailing slash). Defaults to
        ``BIOMINER_BASE_URL`` or the public endpoint.
    token:
        Bearer token. Defaults to ``BIOMINER_TOKEN``. May be ``None`` for
        the unauthenticated ``register`` / ``login`` calls.
    verify:
        TLS verification. Defaults to the certifi bundle. Never disable
        this against a real deployment.
    trust_env:
        Whether to honour proxy / netrc environment variables. Defaults to
        ``False`` (direct connection) so a local proxy cannot silently
        intercept the byte tickets.
    transport:
        Optional httpx transport override (used by tests to pin DNS or to
        inject a mock transport). Production callers leave this ``None``.
    """

    def __init__(self, base_url: str | None = None, token: str | None = None,
                 *, verify: str | bool = _DEFAULT_VERIFY,
                 trust_env: bool = False, timeout: float = 120.0,
                 transport: httpx.BaseTransport | None = None):
        self.base_url = (base_url or os.environ.get("BIOMINER_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.token = token or os.environ.get("BIOMINER_TOKEN")
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._client = httpx.Client(
            base_url=self.base_url, verify=verify, trust_env=trust_env,
            timeout=timeout, headers=headers, transport=transport,
            follow_redirects=True)

    # -- lifecycle -----------------------------------------------------
    def __enter__(self) -> "BioMinerClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _set_token(self, token: str) -> None:
        self.token = token
        self._client.headers["Authorization"] = f"Bearer {token}"

    # -- auth ----------------------------------------------------------
    def register(self, username: str, password: str) -> str:
        """Create a local account and return its bearer token."""
        r = self._client.post("/api/auth/register",
                              json={"username": username, "password": password})
        return self._consume_auth(r)

    def login(self, username: str, password: str) -> str:
        """Authenticate an existing account and return its bearer token."""
        r = self._client.post("/api/auth/login",
                              json={"username": username, "password": password})
        return self._consume_auth(r)

    def _consume_auth(self, r: httpx.Response) -> str:
        if r.status_code != 200:
            raise BioMinerError("AUTH_FAILED", r.text[:300],
                                http_status=r.status_code)
        token = (r.json() or {}).get("token")
        if not token:
            raise BioMinerError("AUTH_FAILED", "no token in auth response")
        self._set_token(token)
        return token

    # -- tool calls ----------------------------------------------------
    def call_tool(self, name: str, args: dict | None = None,
                  method: str = "POST") -> dict:
        """Invoke a workspace tool and return its ``data`` payload.

        Raises :class:`BioMinerError` on a structured refusal (``ok=false``)
        so callers never mistake a business refusal for success.
        """
        url = f"{_API_PREFIX}/{name}"
        if method.upper() == "GET":
            r = self._client.get(url, params=args or {})
        else:
            r = self._client.post(url, json=args or {})
        try:
            envelope = r.json()
        except Exception as exc:  # non-JSON (e.g. HTML error page)
            raise BioMinerError("BAD_RESPONSE",
                                f"HTTP {r.status_code}: {r.text[:200]}",
                                http_status=r.status_code) from exc
        if not envelope.get("ok", False):
            err = envelope.get("error") or {}
            raise BioMinerError(err.get("code", "UNKNOWN"),
                                err.get("message", "tool call failed"),
                                err.get("details"),
                                envelope.get("http_status", r.status_code))
        return envelope.get("data") or {}

    # -- byte transport ------------------------------------------------
    def download_ticket(self, ticket: dict, dest: Path) -> Path:
        """Download a ticket's bytes to ``dest`` and verify its SHA-256.

        Accepts the presigned ``download_url`` when present, otherwise the
        connector ``download_path`` (which carries the session credential).
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = ticket.get("download_url") or ticket["download_path"]
        h = hashlib.sha256()
        with self._client.stream("GET", url) as r:
            if r.status_code != 200:
                raise BioMinerError("DOWNLOAD_FAILED",
                                    f"HTTP {r.status_code} for {ticket.get('filename')}",
                                    http_status=r.status_code)
            with open(dest, "wb") as fh:
                for block in r.iter_bytes(_CHUNK):
                    h.update(block)
                    fh.write(block)
        expected = ticket.get("sha256")
        if expected and h.hexdigest() != expected:
            dest.unlink(missing_ok=True)
            raise BioMinerError("DIGEST_MISMATCH",
                                f"{dest.name}: sha256 {h.hexdigest()} != {expected}")
        return dest

    def upload_bytes(self, filename: str, data: bytes) -> dict:
        """Mint a ticket, POST the bytes, and return the server receipt.

        The receipt proves only that the server received exactly these
        bytes; it carries no scientific authority.
        """
        ticket = self.call_tool("upload_prepare", {
            "filename": filename, "size_bytes": len(data),
            "sha256": _sha256_bytes(data)})["ticket"]
        url = ticket.get("upload_url") or ticket["upload_path"]
        r = self._client.post(url, content=data,
                              headers={"Content-Type": "application/octet-stream"})
        if r.status_code != 200:
            raise BioMinerError("UPLOAD_FAILED",
                                f"HTTP {r.status_code}: {r.text[:200]}",
                                http_status=r.status_code)
        return r.json()["data"]["receipt"]

    def upload_file(self, path: Path) -> dict:
        """Upload a local file by name and return its receipt."""
        path = Path(path)
        return self.upload_bytes(path.name, path.read_bytes())

    # -- convenience ---------------------------------------------------
    def workspace_info(self) -> dict:
        return self.call_tool("workspace_info", method="GET")

    def gene_set_package(self) -> dict:
        """Return the frozen gene-set package manifest with download tickets."""
        return self.call_tool("gene_set_package_get", {}, method="GET")
