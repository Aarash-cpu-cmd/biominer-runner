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

import getpass
import hashlib
import json
import os
import socket
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


_SECRET_KEYS = ("token", "password", "api_key", "access_token", "secret")


def _redact_secrets(text: str) -> str:
    """Strip credential values from a response body before it reaches an
    error message or log. A misclassified success body can carry a bearer
    token; never echo it. Falls back to a regex mask for non-JSON text."""
    import json
    import re
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k in list(obj):
                if k.lower() in _SECRET_KEYS:
                    obj[k] = "***"
            return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass
    return re.sub(
        r'((?:token|password|api_key|access_token|secret)["\']?\s*[:=]\s*["\']?)'
        r'[^\s"\',}]+',
        r'\1***', text, flags=re.IGNORECASE)


def _default_device_fp() -> str:
    """Stable, non-secret device id for anonymous find-or-mint semantics.

    The server binds an anonymous session's owner to this fingerprint, so a
    stable per-machine value lets repeated CLI runs land in the same study
    namespace. Derived from hostname+username (never sent in clear as such —
    only a truncated sha256), matching the server's ``[A-Za-z0-9_-]{8,64}``.
    """
    seed = f"{socket.gethostname()}|{getpass.getuser()}"
    return "cli-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


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
    receipt_cache:
        Optional path to a persistent content-addressed receipt cache. The
        server mints a fresh ``receipt_id`` for every upload even when the
        bytes are identical, which would break idempotent re-registration
        (the register body would change). When a cache path is given, an
        upload whose ``(filename, sha256)`` was already uploaded reuses the
        stored receipt instead of re-uploading, so a retry replays cleanly.
    """

    def __init__(self, base_url: str | None = None, token: str | None = None,
                 *, verify: str | bool = _DEFAULT_VERIFY,
                 trust_env: bool = False, timeout: float = 120.0,
                 transport: httpx.BaseTransport | None = None,
                 receipt_cache: str | Path | None = None):
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
        self._receipt_cache_path = Path(receipt_cache) if receipt_cache else None
        self._receipts: dict[str, Any] = {}
        if self._receipt_cache_path and self._receipt_cache_path.is_file():
            try:
                self._receipts = json.loads(
                    self._receipt_cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._receipts = {}

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
        # Register returns 201 Created, login 200 OK; accept any 2xx.
        if not (200 <= r.status_code < 300):
            raise BioMinerError("AUTH_FAILED", _redact_secrets(r.text)[:300],
                                http_status=r.status_code)
        token = (r.json() or {}).get("token")
        if not token:
            raise BioMinerError("AUTH_FAILED", "no token in auth response")
        self._set_token(token)
        return token

    def mint_anonymous(self, device_fp: str | None = None) -> str:
        """Auto-issue an ephemeral anonymous credential — no account needed.

        Anonymous sessions have registered-user parity except cloud compute;
        the credential self-destructs on expiry or budget exhaustion. The
        device fingerprint keeps the study namespace stable across runs
        (find-or-mint server-side), so repeated CLI calls reuse one session.
        """
        r = self._client.post(
            "/api/mcp/auth/anonymous",
            headers={"X-BM-Device-Fp": device_fp or _default_device_fp()})
        if not (200 <= r.status_code < 300):
            raise BioMinerError("AUTH_FAILED", _redact_secrets(r.text)[:300],
                                http_status=r.status_code)
        token = (r.json() or {}).get("credential")
        if not isinstance(token, str) or not token:
            raise BioMinerError("AUTH_FAILED",
                                "anonymous auth returned no credential",
                                http_status=r.status_code)
        self._set_token(token)
        return token

    def ensure_token(self) -> str:
        """Return the current token, minting an anonymous one if keyless."""
        return self.token or self.mint_anonymous()

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
                                f"HTTP {r.status_code}: {_redact_secrets(r.text)[:200]}",
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

    def upload_bytes(self, filename: str, data: bytes, *,
                     semantic_sha: str | None = None) -> dict:
        """Mint a ticket, POST the bytes, and return the server receipt.

        The receipt proves only that the server received exactly these
        bytes; it carries no scientific authority. With a receipt cache
        configured, identical ``(filename, sha256)`` reuses the stored
        receipt so idempotent re-registration sees an unchanged body.

        ``semantic_sha`` adds a second cache index for content that is
        semantically identical but byte-different (e.g. provenance with a
        fresh ``created_at``/``metrics``): a re-run reuses the ORIGINAL
        receipt, keeping the register body — and thus the server-side
        idempotency binding — unchanged (bug #12).
        """
        sha = _sha256_bytes(data)
        cache_key = f"{filename}:{sha}"
        semantic_key = (f"{filename}:semantic:{semantic_sha}"
                        if semantic_sha else None)
        if self._receipt_cache_path is not None:
            if cache_key in self._receipts:
                return self._receipts[cache_key]
            if semantic_key and semantic_key in self._receipts:
                return self._receipts[semantic_key]
        ticket = self.call_tool("upload_prepare", {
            "filename": filename, "size_bytes": len(data),
            "sha256": sha})["ticket"]
        url = ticket.get("upload_url") or ticket["upload_path"]
        r = self._client.post(url, content=data,
                              headers={"Content-Type": "application/octet-stream"})
        if r.status_code != 200:
            raise BioMinerError("UPLOAD_FAILED",
                                f"HTTP {r.status_code}: {_redact_secrets(r.text)[:200]}",
                                http_status=r.status_code)
        receipt = r.json()["data"]["receipt"]
        if self._receipt_cache_path is not None:
            self._receipts[cache_key] = receipt
            if semantic_key:
                self._receipts[semantic_key] = receipt
            self._persist_receipts()
        return receipt

    def cached_receipts(self, filename: str) -> list[dict]:
        """Distinct cached receipts for ``filename``, newest first.

        Used to recover from a register CONFLICT: the idempotency key is
        already bound to an earlier request body, so the client retries
        with the receipts it originally uploaded (bug #12 migration).
        """
        seen: set[str] = set()
        out: list[dict] = []
        for key in reversed(list(self._receipts)):
            if not key.startswith(f"{filename}:"):
                continue
            receipt = self._receipts[key]
            rid = receipt.get("receipt_id")
            if rid and rid not in seen:
                seen.add(rid)
                out.append(receipt)
        return out

    def rebind_semantic_receipt(self, filename: str, semantic_sha: str,
                                receipt: dict) -> None:
        """Point the semantic cache index at the originally-bound receipt.

        After a CONFLICT recovery the semantic index may point at a newer
        upload that the server rejected; rebinding makes the next re-run
        replay directly without another CONFLICT round-trip (bug #12).
        """
        if self._receipt_cache_path is None:
            return
        self._receipts[f"{filename}:semantic:{semantic_sha}"] = receipt
        self._persist_receipts()

    def _persist_receipts(self) -> None:
        # Best-effort: a cache write failure must never fail an upload.
        try:
            self._receipt_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._receipt_cache_path.write_text(
                json.dumps(self._receipts), encoding="utf-8")
            try:
                os.chmod(self._receipt_cache_path, 0o600)
            except OSError:  # pragma: no cover - platform dependent
                pass
        except OSError:  # pragma: no cover - cache is optional
            pass

    def upload_file(self, path: Path, *,
                    semantic_sha: str | None = None) -> dict:
        """Upload a local file by name and return its receipt."""
        path = Path(path)
        return self.upload_bytes(path.name, path.read_bytes(),
                                 semantic_sha=semantic_sha)

    # -- convenience ---------------------------------------------------
    def workspace_info(self) -> dict:
        return self.call_tool("workspace_info", method="GET")

    def gene_set_package(self) -> dict:
        """Return the frozen gene-set package manifest with download tickets."""
        return self.call_tool("gene_set_package_get", {}, method="GET")
